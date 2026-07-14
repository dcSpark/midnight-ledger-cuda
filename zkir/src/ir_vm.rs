// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::ir::{IrMinorVersion, OldIrSource};

use super::ir::{Instruction as I, IrSource};
use anyhow::{anyhow, bail};
use base_crypto::fab::{Alignment, AlignmentAtom, AlignmentSegment};
use base_crypto::hash::persistent_hash;
use base_crypto::repr::BinaryHashRepr;
use group::{Group, ff::Field};
use midnight_circuits::instructions::{
    ArithInstructions, AssertionInstructions, AssignmentInstructions, BinaryInstructions,
    ControlFlowInstructions, ConversionInstructions, DecompositionInstructions, EccInstructions,
    EqualityInstructions, PublicInputInstructions, RangeCheckInstructions, ZeroInstructions,
};
use midnight_circuits::types::{
    AssignedBit, AssignedByte, AssignedNative, AssignedNativePoint, InnerValue,
};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};
use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};
use num_bigint::BigUint;
use serialize::{Deserializable, Serializable, VecExt, tagged_deserialize, tagged_serialize};
use std::cmp::Ordering;
use std::io::Read;
use transient_crypto::curve::EmbeddedGroupAffine;
use transient_crypto::curve::{FR_BITS, FR_BYTES_STORED, Fr};
use transient_crypto::curve::{embedded, outer};
use transient_crypto::fab::{AlignmentExt, ValueReprAlignedValue};
use transient_crypto::hash::{hash_to_curve, transient_commit, transient_hash};
use transient_crypto::proofs::{ProofPreimage, ProvingError};
use transient_crypto::repr::FieldRepr;

/// The raw data prior to proving. Note that this should *not* be considered part of the public
/// API, and is subject to change at any time. It may be used in combination with
/// [`IrSource::prove_unchecked`] to test malicious prover behavior.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct Preprocessed {
    pub memory: Vec<outer::Scalar>,
    pub pis: Vec<outer::Scalar>,
    pub pi_skips: Vec<Option<usize>>,
    pub binding_input: outer::Scalar,
    pub comm_comm: Option<(outer::Scalar, outer::Scalar)>,
    poseidon_chain_witness: Option<PoseidonChainWitness>,
}

#[derive(Clone, Debug)]
struct PoseidonChainWitness {
    instruction_start: usize,
    hash_count: usize,
    values: Vec<outer::Scalar>,
    #[cfg_attr(not(feature = "gpu-blitzar"), allow(dead_code))]
    direct_cpu_advice: bool,
}

fn lnot(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    a: &AssignedNative<outer::Scalar>,
) -> Result<AssignedNative<outer::Scalar>, Error> {
    let bit = std.is_zero(layouter, a)?;
    std.convert(layouter, &bit)
}

fn fab_decode_to_bytes(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    align: &Alignment,
    mut inputs: &[AssignedNative<outer::Scalar>],
) -> Result<Vec<AssignedByte<outer::Scalar>>, Error> {
    let mut res = Vec::with_bounded_capacity(align.bin_len());
    let _ = fab_decode_to_bytes_inner(std, layouter, align, &mut inputs, &mut res)?;
    let mut debug_values: Vec<u8> = Vec::new();
    for value in res.iter() {
        value.value().assert_if_known(|v| {
            debug_values.push(*v);
            true
        })
    }
    if !debug_values.is_empty() {
        trace!(bytes = ?debug_values, len = debug_values.len(), "bytes decoded in-circuit");
    }
    Ok(res)
}

fn fab_decode_to_bytes_inner(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    align: &Alignment,
    inputs: &mut &[AssignedNative<outer::Scalar>],
    res: &mut Vec<AssignedByte<outer::Scalar>>,
) -> Result<AssignedNative<outer::Scalar>, Error> {
    let mut acc = std.assign_fixed(layouter, 0.into())?;
    for segment in align.0.iter() {
        match segment {
            AlignmentSegment::Atom(atom) => {
                fab_decode_to_bytes_atom(std, layouter, atom, inputs, res)?;
                acc = std.add_constant(layouter, &acc, 1.into())?;
            }
            AlignmentSegment::Option(_) => {
                error!("in-circuit decoding of alignment options is not yet implemented!");
                return Err(Error::Synthesis(
                    "in-circuit decoding of alignment options is not yet implemented!".into(),
                ));
            }
        }
    }
    Ok(acc)
}

fn fab_decode_to_bytes_atom(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    align: &AlignmentAtom,
    inputs: &mut &[AssignedNative<outer::Scalar>],
    res: &mut Vec<AssignedByte<outer::Scalar>>,
) -> Result<(), Error> {
    match align {
        AlignmentAtom::Field => {
            if inputs.is_empty() {
                return Err(Error::Synthesis(
                    "Cannot decode field element from nothing".into(),
                ));
            }
            let value = &inputs[0];
            *inputs = &inputs[1..];
            res.extend(std.assigned_to_le_bytes(layouter, value, None)?);
            Ok(())
        }
        AlignmentAtom::Bytes { length } => {
            let stray = *length as usize % FR_BYTES_STORED;
            let chunks = *length as usize / FR_BYTES_STORED;
            let expected_size = chunks + (stray != 0) as usize;
            let mut bytes_from =
                |slice: &mut Vec<AssignedByte<outer::Scalar>>,
                 k,
                 f: AssignedNative<outer::Scalar>| {
                    let repr = std.assigned_to_le_bytes(layouter, &f, Some(k))?;
                    slice.extend(repr[..k].iter().cloned());
                    Ok::<_, Error>(())
                };
            if inputs.len() < expected_size {
                return Err(Error::Synthesis(
                    "Cannot decode byte value; not enough data provided".into(),
                ));
            }
            let mut res_vec = Vec::with_bounded_capacity(*length as usize - stray);
            if stray > 0 {
                bytes_from(&mut res_vec, stray, inputs[0].clone())?;
                *inputs = &inputs[1..];
            }
            for i in 0..chunks {
                bytes_from(res, FR_BYTES_STORED, inputs[chunks - 1 - i].clone())?;
            }
            *inputs = &inputs[chunks..];
            res.extend(res_vec);
            Ok(())
        }
        AlignmentAtom::Compress => {
            error!("Cannot decode compressed value from field elements");
            Err(Error::Synthesis(
                "Cannot decode compressed value from field elements".into(),
            ))
        }
    }
}

fn assemble_bytes(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    bytes: &[AssignedByte<outer::Scalar>],
) -> Result<AssignedNative<outer::Scalar>, Error> {
    const BITS: usize = 8;
    let mut powers = Vec::with_bounded_capacity(bytes.len());
    powers.push(std.convert(layouter, &bytes[0])?);
    for (i, byte) in bytes.iter().enumerate().skip(1) {
        let power = (0..i * BITS)
            .fold(Fr::from(1), |acc, _| acc * Fr::from(2))
            .0;
        let byte = std.convert(layouter, byte)?;
        powers.push(std.mul_by_constant(layouter, &byte, power)?);
    }
    let mut acc = powers[0].clone();
    for limb in powers[1..].iter() {
        acc = std.add(layouter, &acc, limb)?;
    }
    Ok(acc)
}

fn ecc_from_parts(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<outer::Scalar>,
    x: &AssignedNative<outer::Scalar>,
    y: &AssignedNative<outer::Scalar>,
) -> Result<AssignedNativePoint<embedded::AffineExtended>, Error> {
    let point = x
        .value()
        .zip(y.value())
        .map(|(x, y)| EmbeddedGroupAffine::new(Fr(*x), Fr(*y)));
    point.as_ref().error_if_known_and(|p| p.is_none())?;
    let point = point.map(|p| p.expect("After is_none check, point should exist").0);
    let point_var: AssignedNativePoint<embedded::AffineExtended> =
        std.jubjub().assign(layouter, point)?;

    std.assert_equal(layouter, x, &std.jubjub().x_coordinate(&point_var))?;
    std.assert_equal(layouter, y, &std.jubjub().y_coordinate(&point_var))?;
    Ok(point_var)
}

#[cfg(feature = "gpu-blitzar")]
fn poseidon_backend_mode() -> String {
    std::env::var("MIDNIGHT_GPU_POSEIDON_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_poseidon_backend_enabled() -> bool {
    midnight_curves::gpu_sppark::enabled()
        && matches!(
            poseidon_backend_mode().as_str(),
            "1" | "true"
                | "yes"
                | "on"
                | "gpu"
                | "cuda"
                | "sppark"
                | "auto"
                | "cpu"
                | "cpu-witness"
        )
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_poseidon_cuda_backend_enabled() -> bool {
    gpu_poseidon_backend_enabled()
        && matches!(
            poseidon_backend_mode().as_str(),
            "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark"
        )
}

#[cfg(feature = "gpu-blitzar")]
fn try_gpu_poseidon_chain_witness(
    instructions: &[I],
    start: usize,
    memory: &[Fr],
) -> Option<Result<PoseidonChainWitness, ProvingError>> {
    if !gpu_poseidon_backend_enabled() {
        return None;
    }
    let [initial, tail0, tail1, len] = poseidon_chain_span(instructions, start, memory.len())?;
    let min_len = std::env::var("MIDNIGHT_GPU_POSEIDON_MIN_CHAIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    if len < min_len {
        return None;
    }

    if matches!(
        poseidon_backend_mode().as_str(),
        "auto" | "cpu" | "cpu-witness"
    ) {
        let values = transient_crypto::proof_timer::timed(
            "proof_server.prover.zkir.preprocess.poseidon_chain_witness.cpu",
            || {
                midnight_circuits::hash::poseidon::hash_chain_circuit_witness(
                    memory[initial].0,
                    memory[tail0].0,
                    memory[tail1].0,
                    len,
                )
            },
        );
        return Some(Ok(PoseidonChainWitness {
            instruction_start: start,
            hash_count: len,
            values,
            direct_cpu_advice: false,
        }));
    }

    use midnight_circuits::hash::poseidon::{POSEIDON_HASH_WITNESS_LEN, constants::PoseidonField};
    let mds = <outer::Scalar as PoseidonField>::MDS
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let round_constants = <outer::Scalar as PoseidonField>::ROUND_CONSTANTS
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let result = transient_crypto::proof_timer::timed(
        "proof_server.prover.zkir.preprocess.poseidon_chain_witness.gpu",
        || {
            midnight_curves::gpu_sppark::try_bls12_381_poseidon_hash_chain_witness(
                memory[initial].0,
                memory[tail0].0,
                memory[tail1].0,
                outer::Scalar::from(3),
                &mds,
                &round_constants,
                len,
            )
            .expect("GPU enablement changed during Poseidon witness preprocessing")
            .and_then(|values| {
                if values.len() != len * POSEIDON_HASH_WITNESS_LEN {
                    return Err(format!(
                        "GPU Poseidon witness returned {} fields, expected {}",
                        values.len(),
                        len * POSEIDON_HASH_WITNESS_LEN
                    ));
                }
                Ok(PoseidonChainWitness {
                    instruction_start: start,
                    hash_count: len,
                    values,
                    direct_cpu_advice: false,
                })
            })
            .map_err(anyhow::Error::msg)
        },
    );
    Some(result)
}

fn poseidon_chain_span(instructions: &[I], start: usize, memory_len: usize) -> Option<[usize; 4]> {
    let I::TransientHash { inputs } = instructions.get(start)? else {
        return None;
    };
    let [initial, tail0, tail1] = inputs.as_slice() else {
        return None;
    };
    if (*initial as usize) >= memory_len
        || (*tail0 as usize) >= memory_len
        || (*tail1 as usize) >= memory_len
    {
        return None;
    }

    let mut len = 1usize;
    while let Some(I::TransientHash { inputs }) = instructions.get(start + len) {
        let [next, next_tail0, next_tail1] = inputs.as_slice() else {
            break;
        };
        if *next as usize != memory_len + len - 1 || next_tail0 != tail0 || next_tail1 != tail1 {
            break;
        }
        len += 1;
    }

    Some([*initial as usize, *tail0 as usize, *tail1 as usize, len])
}

#[cfg(feature = "gpu-blitzar")]
fn cpu_poseidon_direct_advice_enabled(len: usize) -> bool {
    let min_len = std::env::var("MIDNIGHT_GPU_POSEIDON_MIN_CHAIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    gpu_poseidon_backend_enabled()
        && matches!(
            poseidon_backend_mode().as_str(),
            "auto" | "cpu" | "cpu-witness"
        )
        && len >= min_len
}

#[cfg(feature = "gpu-blitzar")]
fn resident_poseidon_advice_enabled(domain_len: usize) -> bool {
    let min_len = std::env::var("MIDNIGHT_GPU_RESIDENT_FINALISE_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 15);
    domain_len >= min_len
}

#[cfg(feature = "gpu-blitzar")]
fn write_poseidon_permutation_advice(
    columns: &mut [Vec<outer::Scalar>],
    row_start: usize,
    inputs: [outer::Scalar; 3],
    witness: &[outer::Scalar],
    layout: &[(usize, usize)],
) -> Result<[outer::Scalar; 3], Error> {
    use midnight_circuits::hash::poseidon::{
        POSEIDON_PERMUTATION_WITNESS_LEN, constants::PoseidonField,
    };

    if witness.len() != POSEIDON_PERMUTATION_WITNESS_LEN || layout.len() != witness.len() {
        return Err(Error::Synthesis(
            "invalid precomputed Poseidon permutation witness".into(),
        ));
    }
    for (column, value) in inputs.into_iter().enumerate() {
        columns[column][row_start] = value;
        columns[column][row_start + 1] =
            value + <outer::Scalar as PoseidonField>::ROUND_CONSTANTS[0][column];
    }
    for (&value, &(column, row)) in witness.iter().zip(layout) {
        columns[column][row_start + row] = value;
    }

    let output = &witness[witness.len() - 3..];
    Ok([output[0], output[1], output[2]])
}

#[cfg(feature = "gpu-blitzar")]
fn terminal_poseidon_direct_advice(
    preprocessed: &Preprocessed,
    chain: [usize; 4],
    domain_len: usize,
    num_advice_columns: usize,
) -> Result<Vec<Vec<outer::Scalar>>, Error> {
    use midnight_circuits::hash::poseidon::{
        NB_POSEIDON_ADVICE_COLS, POSEIDON_HASH_WITNESS_LEN, POSEIDON_PERMUTATION_WITNESS_LEN,
        poseidon_permutation_witness_layout,
    };

    let [initial, tail0, tail1, hash_count] = chain;
    if num_advice_columns != NB_POSEIDON_ADVICE_COLS || preprocessed.memory.len() != 3 {
        return Err(Error::Synthesis(
            "terminal Poseidon direct-advice shape does not match the proving key".into(),
        ));
    }
    let last_used_row = 1usize
        .checked_add(41usize.saturating_mul(hash_count))
        .ok_or_else(|| Error::Synthesis("terminal Poseidon row count overflow".into()))?;
    if last_used_row >= domain_len {
        return Err(Error::Synthesis(
            "terminal Poseidon direct advice exceeds the proving domain".into(),
        ));
    }

    let chain_witness = preprocessed
        .poseidon_chain_witness
        .as_ref()
        .ok_or_else(|| Error::Synthesis("terminal Poseidon circuit witness is missing".into()))?;
    if chain_witness.instruction_start != 0 || chain_witness.hash_count != hash_count {
        return Err(Error::Synthesis(
            "terminal Poseidon circuit witness does not match the relation".into(),
        ));
    }

    let tail0_value = preprocessed.memory[tail0];
    let tail1_value = preprocessed.memory[tail1];
    let mut columns = if chain_witness.direct_cpu_advice {
        midnight_circuits::hash::poseidon::hash_chain_circuit_advice_columns(
            preprocessed.memory[initial],
            tail0_value,
            tail1_value,
            hash_count,
            domain_len,
            2,
        )
    } else {
        if chain_witness.values.len() != hash_count * POSEIDON_HASH_WITNESS_LEN {
            return Err(Error::Synthesis(
                "terminal Poseidon circuit witness has the wrong length".into(),
            ));
        }
        let mut columns = (0..num_advice_columns)
            .map(|_| vec![outer::Scalar::ZERO; domain_len])
            .collect::<Vec<_>>();
        let layout = poseidon_permutation_witness_layout();
        let mut previous = preprocessed.memory[initial];
        for hash_index in 0..hash_count {
            let row_start = 2 + 41 * hash_index;
            let witness_start = hash_index * POSEIDON_HASH_WITNESS_LEN;
            let hash_witness =
                &chain_witness.values[witness_start..witness_start + POSEIDON_HASH_WITNESS_LEN];
            let first_output = write_poseidon_permutation_advice(
                &mut columns,
                row_start,
                [previous, tail0_value, outer::Scalar::from(3)],
                &hash_witness[..POSEIDON_PERMUTATION_WITNESS_LEN],
                &layout,
            )?;

            let second_input0 = first_output[0] + tail1_value;
            columns[0][row_start + 20] = second_input0;
            columns[1][row_start + 20] = first_output[0];
            columns[2][row_start + 20] = tail1_value;
            let second_output = write_poseidon_permutation_advice(
                &mut columns,
                row_start + 21,
                [second_input0, first_output[1], first_output[2]],
                &hash_witness[POSEIDON_PERMUTATION_WITNESS_LEN..],
                &layout,
            )?;
            previous = second_output[0];
        }
        columns
    };
    for (column, &value) in preprocessed.memory.iter().enumerate() {
        columns[column][0] = value;
    }
    columns[0][1] = preprocessed.binding_input;

    Ok(columns)
}

#[cfg(feature = "gpu-blitzar")]
fn try_gpu_poseidon_chain(
    instructions: &[I],
    start: usize,
    memory: &[Fr],
) -> Option<Result<Vec<Fr>, ProvingError>> {
    if !gpu_poseidon_cuda_backend_enabled() {
        return None;
    }
    let [initial, tail0, tail1, len] = poseidon_chain_span(instructions, start, memory.len())?;
    let min_len = std::env::var("MIDNIGHT_GPU_POSEIDON_MIN_CHAIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    if len < min_len {
        return None;
    }

    use midnight_circuits::hash::poseidon::constants::PoseidonField;
    let mds = <outer::Scalar as PoseidonField>::MDS
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let round_constants = <outer::Scalar as PoseidonField>::ROUND_CONSTANTS
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let result = transient_crypto::proof_timer::timed(
        "proof_server.prover.zkir.preprocess.poseidon_chain.gpu",
        || {
            midnight_curves::gpu_sppark::try_bls12_381_poseidon_hash_chain(
                memory[initial].0,
                memory[tail0].0,
                memory[tail1].0,
                outer::Scalar::from(3),
                &mds,
                &round_constants,
                len,
            )
            .expect("GPU enablement changed during Poseidon preprocessing")
            .map(|values| values.into_iter().map(Fr).collect())
            .map_err(anyhow::Error::msg)
        },
    );
    Some(result)
}

impl IrSource {
    /// Performs a non-ZK run of a circuit, to ensure that constraints hold, and
    /// to produce a public input vector, and public input skip information.
    pub(crate) fn preprocess(
        &self,
        preimage: &ProofPreimage,
    ) -> Result<Preprocessed, ProvingError> {
        if preimage.inputs.len() != self.num_inputs as usize {
            bail!(
                "Expected {} inputs, received {}",
                self.num_inputs,
                preimage.inputs.len()
            );
        }
        let mut memory: Vec<_> = preimage.inputs.clone();
        let mut pis = vec![preimage.binding_input];
        if self.do_communications_commitment {
            pis.push(
                preimage
                    .communications_commitment
                    .ok_or(anyhow!("Expected communications commitment"))?
                    .0,
            );
        }
        let mut pi_skips = Vec::new();
        let mut public_transcript_inputs_idx: usize = 0;
        let mut public_transcript_outputs_idx: usize = 0;
        let mut private_transcript_outputs_idx: usize = 0;
        let mut outputs = Vec::new();
        let idx = |memory: &[Fr], i: u32| {
            let res = memory
                .get(i as usize)
                .copied()
                .ok_or(anyhow!("index out of bounds: {i}"));
            trace!(?res, "retrieved from {i}");
            res
        };
        let idx_bool = |memory: &[Fr], i: u32| {
            idx(memory, i).and_then(|val| {
                if val == 0.into() {
                    Ok(false)
                } else if val == 1.into() {
                    Ok(true)
                } else {
                    bail!("Expected boolean, found: {val:?}");
                }
            })
        };
        let idx_point = |memory: &[Fr], x: u32, y: u32| {
            let x = idx(memory, x)?;
            let y = idx(memory, y)?;
            EmbeddedGroupAffine::new(x, y)
                .ok_or(anyhow!("Elliptic curve point not on curve: ({x:?}, {y:?})"))
        };
        let idx_bits = |memory: &[Fr], i: u32, constrain: Option<u32>| {
            idx(memory, i).and_then(|val| {
                let mut bits = val
                    .0
                    .to_bytes_le()
                    .into_iter()
                    .flat_map(|byte| {
                        [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
                            .into_iter()
                            .map(move |mask| byte & mask != 0)
                    })
                    .collect::<Vec<_>>();
                if let Some(n) = constrain {
                    if n as usize >= FR_BITS {
                        bail!("Excessive bit bound");
                    }
                    if bits[n as usize..].iter().any(|b| *b) {
                        bail!("Bit bound failed: {val:?} is not {n}-bit");
                    }
                    bits.truncate(n as usize);
                }
                Ok(bits)
            })
        };
        let from_point =
            |p: EmbeddedGroupAffine| [p.x().unwrap_or(0.into()), p.y().unwrap_or(0.into())];
        fn from_bits<I: DoubleEndedIterator<Item = bool>>(bits: I) -> Fr {
            bits.rev()
                .fold(0.into(), |acc, bit| acc * 2.into() + bit.into())
        }
        #[cfg(feature = "gpu-blitzar")]
        let mut poseidon_chain_witness = None;
        #[cfg(not(feature = "gpu-blitzar"))]
        let poseidon_chain_witness = None;
        let mut instruction_index = 0usize;
        while instruction_index < self.instructions.len() {
            if let Some([_, _, _, len]) =
                poseidon_chain_span(&self.instructions, instruction_index, memory.len())
                && instruction_index + len == self.instructions.len()
            {
                #[cfg(feature = "gpu-blitzar")]
                if instruction_index == 0
                    && self.num_inputs == 3
                    && !self.do_communications_commitment
                    && cpu_poseidon_direct_advice_enabled(len)
                {
                    poseidon_chain_witness = Some(PoseidonChainWitness {
                        instruction_start: 0,
                        hash_count: len,
                        values: Vec::new(),
                        direct_cpu_advice: true,
                    });
                } else if let Some(result) =
                    try_gpu_poseidon_chain_witness(&self.instructions, instruction_index, &memory)
                {
                    poseidon_chain_witness = Some(result?);
                }
                // A terminal pure hash chain has no externally observable preprocess output.
                instruction_index += len;
                continue;
            }

            #[cfg(feature = "gpu-blitzar")]
            if let Some(result) =
                try_gpu_poseidon_chain(&self.instructions, instruction_index, &memory)
            {
                let values = result?;
                instruction_index += values.len();
                memory.extend(values);
                continue;
            }

            let ins = &self.instructions[instruction_index];
            trace!(?ins, "preprocess gate");
            let start_idx = memory.len();
            match ins {
                I::Add { a, b } => memory.push(idx(&memory, *a)? + idx(&memory, *b)?),
                I::Mul { a, b } => memory.push(idx(&memory, *a)? * idx(&memory, *b)?),
                I::Neg { a } => memory.push(-idx(&memory, *a)?),
                I::Not { a } => memory.push((!idx_bool(&memory, *a)?).into()),
                I::ConstrainEq { a, b } => {
                    if idx(&memory, *a)? != idx(&memory, *b)? {
                        bail!(
                            "Failed equality constraint: {:?} != {:?}",
                            idx(&memory, *a)?,
                            idx(&memory, *b)?
                        );
                    }
                }
                I::CondSelect { bit, a, b } => {
                    let (bit, a, b) = (
                        idx_bool(&memory, *bit)?,
                        idx(&memory, *a)?,
                        idx(&memory, *b)?,
                    );
                    memory.push(if bit { a } else { b })
                }
                I::Assert { cond } => {
                    if !idx_bool(&memory, *cond)? {
                        bail!("Failed direct assertion");
                    }
                }
                I::TestEq { a, b } => memory.push((idx(&memory, *a)? == idx(&memory, *b)?).into()),
                I::PublicInput { guard } => {
                    let val = match guard {
                        Some(guard) if !idx_bool(&memory, *guard)? => 0.into(),
                        _ => {
                            public_transcript_outputs_idx += 1;
                            preimage
                                .public_transcript_outputs
                                .get(public_transcript_outputs_idx - 1)
                                .copied()
                                .ok_or(anyhow!("Ran out of public transcript outputs"))?
                        }
                    };
                    memory.push(val);
                }
                I::DeclarePubInput { var } => {
                    pis.push(idx(&memory, *var)?);
                    public_transcript_inputs_idx += 1;
                }
                I::PrivateInput { guard } => match guard {
                    Some(guard) if !idx_bool(&memory, *guard)? => memory.push(0.into()),
                    _ => {
                        memory.push(
                            preimage
                                .private_transcript
                                .get(private_transcript_outputs_idx)
                                .copied()
                                .ok_or(anyhow!("Ran out of private transcript outputs"))?,
                        );
                        private_transcript_outputs_idx += 1;
                    }
                },
                I::Copy { var } => memory.push(idx(&memory, *var)?),
                I::ConstrainToBoolean { var } => drop(idx_bool(&memory, *var)?),
                I::ConstrainBits { var, bits } => drop(idx_bits(&memory, *var, Some(*bits))?),
                I::DivModPowerOfTwo { var, bits } => {
                    if *bits as usize > FR_BYTES_STORED * 8 {
                        bail!("Excessive bit count");
                    }
                    let var_bits = idx_bits(&memory, *var, None)?;
                    memory.push(from_bits(var_bits[*bits as usize..].iter().copied()));
                    memory.push(from_bits(var_bits[..*bits as usize].iter().copied()));
                }
                I::ReconstituteField {
                    divisor,
                    modulus,
                    bits,
                } => {
                    if *bits as usize > FR_BYTES_STORED * 8 {
                        bail!("Excessive bit count");
                    }
                    let fr_max = Fr::from(-1);
                    let max_bits = idx_bits(&[fr_max], 0, None)?;
                    let modulus_bits = idx_bits(&memory, *modulus, Some(*bits))?;
                    let divisor_bits = idx_bits(&memory, *divisor, Some(FR_BITS as u32 - *bits))?;
                    let cmp = modulus_bits
                        .iter()
                        .chain(divisor_bits.iter())
                        .rev()
                        .zip(max_bits[..FR_BITS].iter().rev())
                        .map(|(ab, max)| ab.cmp(max))
                        .fold(
                            Ordering::Equal,
                            |prefix, local| if prefix.is_eq() { local } else { prefix },
                        );
                    if cmp.is_gt() {
                        bail!("Reconstituted element overflows field");
                    }
                    let power = (0..*bits).fold(Fr::from(1), |acc, _| Fr::from(2) * acc);
                    memory.push(power * idx(&memory, *divisor)? + idx(&memory, *modulus)?);
                }
                I::LessThan { a, b, bits } => memory.push(
                    (from_bits(idx_bits(&memory, *a, Some(*bits))?.into_iter())
                        < from_bits(idx_bits(&memory, *b, Some(*bits))?.into_iter()))
                    .into(),
                ),
                I::TransientHash { inputs } => memory.push(transient_hash(
                    &inputs
                        .iter()
                        .map(|i| idx(&memory, *i))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                I::PersistentHash { alignment, inputs } => {
                    let inputs = inputs
                        .iter()
                        .map(|i| idx(&memory, *i))
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = alignment.parse_field_repr(&inputs).ok_or_else(|| {
                        error!("Inputs did not match alignment (inputs: {inputs:?}, alignment: {alignment:?})");
                        anyhow!("Inputs did not match alignment (inputs: {inputs:?}, alignment: {alignment:?})")
                    })?;
                    let mut repr = Vec::new();
                    ValueReprAlignedValue(value).binary_repr(&mut repr);
                    trace!(bytes = ?repr, "bytes decoded out-of-circuit");
                    let hash = persistent_hash(&repr);
                    memory.extend(hash.field_vec());
                }
                I::PiSkip { guard, count } => match guard {
                    Some(guard) if !idx_bool(&memory, *guard)? => {
                        pi_skips.push(Some(*count as usize));
                        public_transcript_inputs_idx -= *count as usize;
                    }
                    _ => {
                        pi_skips.push(None);
                        for i in 0..(*count as usize) {
                            let idx = public_transcript_inputs_idx - *count as usize + i;
                            let expected = preimage.public_transcript_inputs.get(idx).copied();
                            let computed = Some(pis[pis.len() - *count as usize + i]);
                            if expected != computed {
                                error!(
                                    ?idx,
                                    ?expected,
                                    ?computed,
                                    ?memory,
                                    ?pis,
                                    "Public transcript input mismatch"
                                );
                                bail!(
                                    "Public transcript input mismatch for input {idx}; expected: {expected:?}, computed: {computed:?}"
                                );
                            }
                        }
                    }
                },
                I::LoadImm { imm } => memory.push(*imm),
                I::Output { var } => outputs.push(idx(&memory, *var)?),
                I::EcAdd { a_x, a_y, b_x, b_y } => memory.extend(from_point(
                    idx_point(&memory, *a_x, *a_y)? + idx_point(&memory, *b_x, *b_y)?,
                )),
                I::HashToCurve { inputs } => {
                    let inputs = inputs
                        .iter()
                        .map(|var| idx(&memory, *var))
                        .collect::<Result<Vec<_>, _>>()?;
                    memory.extend(from_point(hash_to_curve(&inputs)))
                }
                I::EcMul { a_x, a_y, scalar } => memory.extend(from_point(
                    idx_point(&memory, *a_x, *a_y)? * idx(&memory, *scalar)?,
                )),
                I::EcMulGenerator { scalar } => memory.extend(from_point(
                    EmbeddedGroupAffine::generator() * idx(&memory, *scalar)?,
                )),
            }
            trace!(delta = ?memory[start_idx..], "Memory delta {}..{}", start_idx, memory.len());
            instruction_index += 1;
        }
        trace!(?outputs, "Finished instructions with output");
        if preimage.public_transcript_inputs.len() != public_transcript_inputs_idx
            || preimage.public_transcript_outputs.len() != public_transcript_outputs_idx
            || preimage.private_transcript.len() != private_transcript_outputs_idx
        {
            error!(
                public_transcript_inputs = ?preimage.public_transcript_inputs,
                public_transcript_outputs = ?preimage.public_transcript_outputs,
                private_transcript_outputs = ?preimage.private_transcript,
                ?public_transcript_inputs_idx,
                ?public_transcript_outputs_idx,
                ?private_transcript_outputs_idx,
                "Transcripts not fully consumed");
            bail!("Transcripts not fully consumed");
        }
        if self.do_communications_commitment {
            let comm_comm = preimage
                .communications_commitment
                .ok_or(anyhow!("Expected communications randomness"))?;
            let mut comm_comm_inputs: Vec<Fr> = Vec::new();
            comm_comm_inputs.extend(preimage.inputs.iter());
            comm_comm_inputs.extend(outputs.iter());
            if comm_comm.0 != transient_commit(&comm_comm_inputs[..], comm_comm.1) {
                error!(
                    ?comm_comm,
                    ?comm_comm_inputs,
                    "Communications commitment mismatch"
                );
                bail!("Communications commitment mismatch");
            }
        }
        Ok(Preprocessed {
            memory: memory.into_iter().map(|x| x.0).collect(),
            pis: pis.into_iter().map(|x| x.0).collect(),
            pi_skips,
            binding_input: preimage.binding_input.0,
            comm_comm: preimage
                .communications_commitment
                .map(|(comm, rand)| (comm.0, rand.0)),
            poseidon_chain_witness,
        })
    }
}

impl Relation for IrSource {
    type Instance = Vec<outer::Scalar>;

    type Witness = Preprocessed;

    type Error = Error;

    fn format_instance(
        instance: &Self::Instance,
    ) -> Result<Vec<outer::Scalar>, midnight_proofs::plonk::Error> {
        Ok(instance.clone())
    }

    #[cfg(feature = "gpu-blitzar")]
    fn direct_resident_advice_values(
        &self,
        witness: Value<&Self::Witness>,
        domain_len: usize,
        num_advice_columns: usize,
    ) -> Result<Value<Option<midnight_proofs::plonk::DirectResidentAdvice>>, Self::Error> {
        use midnight_circuits::hash::poseidon::{
            NB_POSEIDON_ADVICE_COLS, POSEIDON_HASH_WITNESS_LEN, constants::PoseidonField,
            poseidon_permutation_witness_layout,
        };

        let chain = if gpu_poseidon_backend_enabled()
            && resident_poseidon_advice_enabled(domain_len)
            && self.num_inputs == 3
            && !self.do_communications_commitment
            && num_advice_columns == NB_POSEIDON_ADVICE_COLS
        {
            poseidon_chain_span(&self.instructions, 0, self.num_inputs as usize)
                .filter(|[_, _, _, len]| *len == self.instructions.len())
        } else {
            None
        };
        let Some([initial, tail0, tail1, hash_count]) = chain else {
            return Ok(Value::known(None));
        };

        witness.map_with_result(|preprocessed| {
            transient_crypto::proof_timer::timed(
                "proof_server.prover.zkir.advice.direct_poseidon_resident",
                || {
                    let chain_witness =
                        preprocessed
                            .poseidon_chain_witness
                            .as_ref()
                            .ok_or_else(|| {
                                Error::Synthesis(
                                    "terminal Poseidon circuit witness is missing".into(),
                                )
                            })?;
                    if chain_witness.instruction_start != 0
                        || chain_witness.hash_count != hash_count
                    {
                        return Err(Error::Synthesis(
                            "terminal Poseidon circuit witness does not match the relation".into(),
                        ));
                    }
                    let compact = if chain_witness.direct_cpu_advice {
                        midnight_circuits::hash::poseidon::hash_chain_circuit_witness(
                            preprocessed.memory[initial],
                            preprocessed.memory[tail0],
                            preprocessed.memory[tail1],
                            hash_count,
                        )
                    } else {
                        chain_witness.values.clone()
                    };
                    if compact.len() != hash_count * POSEIDON_HASH_WITNESS_LEN {
                        return Err(Error::Synthesis(
                            "terminal Poseidon circuit witness has the wrong length".into(),
                        ));
                    }
                    let resident =
                        midnight_proofs::plonk::DirectResidentAdvice::from_poseidon_hash_chain(
                            &compact,
                            hash_count,
                            domain_len,
                            [
                                preprocessed.memory[0],
                                preprocessed.memory[1],
                                preprocessed.memory[2],
                            ],
                            preprocessed.binding_input,
                            preprocessed.memory[initial],
                            preprocessed.memory[tail0],
                            preprocessed.memory[tail1],
                            <outer::Scalar as PoseidonField>::ROUND_CONSTANTS[0],
                            &poseidon_permutation_witness_layout(),
                        )
                        .map_err(Error::Backend)?;
                    Ok(Some(resident))
                },
            )
        })
    }

    fn direct_advice_values(
        &self,
        _witness: Value<&Self::Witness>,
        _domain_len: usize,
        _num_advice_columns: usize,
    ) -> Result<Value<Option<Vec<Vec<outer::Scalar>>>>, Self::Error> {
        #[cfg(feature = "gpu-blitzar")]
        {
            let chain = if gpu_poseidon_backend_enabled()
                && self.num_inputs == 3
                && !self.do_communications_commitment
            {
                poseidon_chain_span(&self.instructions, 0, self.num_inputs as usize)
                    .filter(|[_, _, _, len]| *len == self.instructions.len())
            } else {
                None
            };
            if let Some(chain) = chain {
                return _witness.map_with_result(|preprocessed| {
                    transient_crypto::proof_timer::timed(
                        "proof_server.prover.zkir.advice.direct_poseidon",
                        || {
                            terminal_poseidon_direct_advice(
                                preprocessed,
                                chain,
                                _domain_len,
                                _num_advice_columns,
                            )
                            .map(Some)
                        },
                    )
                });
            }
        }
        Ok(Value::known(None))
    }

    fn circuit(
        &self,
        std: &ZkStdLib,
        layouter: &mut impl Layouter<outer::Scalar>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let input_values = witness
            .as_ref()
            .map(|preproc| preproc.memory[..self.num_inputs as usize].to_vec());
        let binding_input_value = witness.as_ref().map(|preproc| preproc.binding_input);
        let comm_comm_value = witness.as_ref().map(|preproc| preproc.comm_comm);

        let mut memory = std.assign_many(
            layouter,
            &input_values.transpose_vec(self.num_inputs as usize),
        )?;

        let inputs = memory.clone();
        let binding_input = std.assign(layouter, binding_input_value)?;

        let mut outputs = Vec::new();

        fn idx(
            memory: &[AssignedNative<outer::Scalar>],
            i: u32,
        ) -> Result<&AssignedNative<outer::Scalar>, Error> {
            memory
                .get(i as usize)
                .ok_or(Error::Synthesis(format!("missing index {i}")))
        }
        let seq_push = |cell: AssignedNative<outer::Scalar>,
                        mem: &mut Vec<AssignedNative<outer::Scalar>>,
                        seq: for<'a> fn(&'a Preprocessed) -> &'a [outer::Scalar]|
         -> Result<(), Error> {
            let idx = mem.len();

            witness.as_ref()
                .zip(cell.value())
                .error_if_known_and(|(preproc, v)| {
                    if idx < seq(preproc).len() && seq(preproc)[idx] != **v {
                        error!(prepare = ?seq(preproc), ?idx, ?v, "Misalignment between `prepare` and `synthesize` runs. This is a bug.");
                        eprintln!("Misalignment between `prepare` and `synthesize` runs. This is a bug.");
                        true
                    } else {
                        false
                    }
                })?;

            mem.push(cell);
            Ok(())
        };

        let mem_push =
            |cell: AssignedNative<outer::Scalar>,
             mem: &mut Vec<AssignedNative<outer::Scalar>>|
             -> Result<(), Error> { seq_push(cell, mem, |preproc| &preproc.memory) };

        let pi_push = |cell: AssignedNative<outer::Scalar>,
                       pis: &mut Vec<AssignedNative<outer::Scalar>>|
         -> Result<(), Error> { seq_push(cell, pis, |preproc| &preproc.pis) };

        let mut public_inputs = vec![];
        pi_push(binding_input, &mut public_inputs)?;

        if self.do_communications_commitment {
            let comm_comm_value = comm_comm_value.map(|c| {
                c.ok_or_else(|| {
                    error!("Communication commitment not present despite preproc. This is a bug.");
                    Error::Synthesis("Communication commitment not present despite preproc.".into())
                })
                .unwrap()
                .0
            });
            let comm_comm = std.assign(layouter, comm_comm_value)?;
            pi_push(comm_comm, &mut public_inputs)?;
        }
        #[cfg(feature = "gpu-blitzar")]
        let gpu_terminal_chain = if gpu_poseidon_backend_enabled() {
            poseidon_chain_span(&self.instructions, 0, memory.len())
                .filter(|[_, _, _, len]| *len == self.instructions.len())
        } else {
            None
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let gpu_terminal_chain: Option<[usize; 4]> = None;

        if let Some([_, _, _, hash_count]) = gpu_terminal_chain {
            use midnight_circuits::hash::poseidon::POSEIDON_HASH_WITNESS_LEN;

            let chain_witness = witness.as_ref().map_with_result(|preproc| {
                let chain = preproc.poseidon_chain_witness.as_ref().ok_or_else(|| {
                    Error::Synthesis("CUDA Poseidon chain witness is missing".into())
                })?;
                if chain.instruction_start != 0 || chain.hash_count != hash_count {
                    return Err(Error::Synthesis(
                        "CUDA Poseidon chain witness does not match the circuit".into(),
                    ));
                }
                Ok(chain.values.as_slice())
            })?;

            for (hash_index, ins) in self.instructions.iter().enumerate() {
                let I::TransientHash { inputs } = ins else {
                    return Err(Error::Synthesis(
                        "CUDA Poseidon chain contains a non-hash instruction".into(),
                    ));
                };
                let inputs = inputs
                    .iter()
                    .map(|input| idx(&memory, *input).cloned())
                    .collect::<Result<Vec<_>, _>>()?;
                let witness_start = hash_index * POSEIDON_HASH_WITNESS_LEN;
                let hash_witness = chain_witness.map(|values| {
                    &values[witness_start..witness_start + POSEIDON_HASH_WITNESS_LEN]
                });
                let output = std.poseidon_with_circuit_witness(layouter, &inputs, hash_witness)?;
                mem_push(output, &mut memory)?;
            }
        } else {
            for ins in self.instructions.iter() {
                match ins {
                    I::Assert { cond } => std.assert_non_zero(layouter, idx(&memory, *cond)?)?,
                    I::CondSelect { bit, a, b } if self.version == IrMinorVersion::V0 => {
                        let bit = std.is_zero(layouter, idx(&memory, *bit)?)?;
                        // Note that b comes first here, because the is_zero negates the bit.
                        // The negation is to ensure the bit bound. This may be
                        // excessive, but user input could violate it otherwise.
                        let result =
                            std.select(layouter, &bit, idx(&memory, *b)?, idx(&memory, *a)?)?;
                        mem_push(result, &mut memory)?;
                    }
                    I::CondSelect { bit, a, b } => {
                        let bit: AssignedBit<_> = std.convert(layouter, idx(&memory, *bit)?)?;
                        let result =
                            std.select(layouter, &bit, idx(&memory, *a)?, idx(&memory, *b)?)?;
                        mem_push(result, &mut memory)?;
                    }
                    I::ConstrainBits { var, bits } if self.version == IrMinorVersion::V0 => {
                        drop(std.assigned_to_le_bits(
                            layouter,
                            idx(&memory, *var)?,
                            Some(*bits as usize),
                            *bits as usize >= FR_BITS,
                        )?)
                    }
                    I::ConstrainBits { var, bits } => {
                        if *bits < FR_BITS as u32 {
                            let bound = BigUint::from(1u32) << *bits;
                            std.assert_lower_than_fixed(layouter, idx(&memory, *var)?, &bound)?;
                        }
                    }
                    I::ConstrainEq { a, b } => {
                        std.assert_equal(layouter, idx(&memory, *a)?, idx(&memory, *b)?)?
                    }
                    I::ConstrainToBoolean { var } => {
                        // Yes, this does insert a constraint.
                        let _: AssignedBit<_> = std.convert(layouter, idx(&memory, *var)?)?;
                    }
                    I::Copy { var } => mem_push(idx(&memory, *var)?.clone(), &mut memory)?,
                    I::DeclarePubInput { var } => {
                        pi_push(idx(&memory, *var)?.clone(), &mut public_inputs)?
                    }
                    I::PiSkip { .. } => {}
                    I::LoadImm { imm } => {
                        mem_push(std.assign_fixed(layouter, imm.0)?, &mut memory)?
                    }
                    I::Output { var } => outputs.push(idx(&memory, *var)?.clone()),
                    I::TransientHash { inputs } => mem_push(
                        std.poseidon(
                            layouter,
                            &inputs
                                .iter()
                                .map(|inp| idx(&memory, *inp).cloned())
                                .collect::<Result<Vec<_>, _>>()?,
                        )?,
                        &mut memory,
                    )?,
                    I::PersistentHash { alignment, inputs } => {
                        let inputs = inputs
                            .iter()
                            .map(|i| idx(&memory, *i).cloned())
                            .collect::<Result<Vec<_>, _>>()?;
                        let bytes = fab_decode_to_bytes(std, layouter, alignment, &inputs)?;
                        let res_bytes = std.sha2_256(layouter, &bytes)?;
                        mem_push(std.convert(layouter, &res_bytes[31])?, &mut memory)?;
                        mem_push(
                            assemble_bytes(std, layouter, &res_bytes[..31])?,
                            &mut memory,
                        )?;
                    }
                    I::TestEq { a, b } => {
                        let bit = std.is_equal(layouter, idx(&memory, *a)?, idx(&memory, *b)?)?;
                        mem_push(std.convert(layouter, &bit)?, &mut memory)?;
                    }
                    I::Add { a, b } => mem_push(
                        std.add(layouter, idx(&memory, *a)?, idx(&memory, *b)?)?,
                        &mut memory,
                    )?,
                    I::Mul { a, b } => mem_push(
                        std.mul(layouter, idx(&memory, *a)?, idx(&memory, *b)?, None)?,
                        &mut memory,
                    )?,
                    I::Neg { a } => mem_push(std.neg(layouter, idx(&memory, *a)?)?, &mut memory)?,
                    I::Not { a } => mem_push(lnot(std, layouter, idx(&memory, *a)?)?, &mut memory)?,
                    I::LessThan { a, b, bits } => {
                        // Adding mod 2 to meet library constraint that this is even
                        // Hidden req that this is >= 4
                        let bit = std.lower_than(
                            layouter,
                            idx(&memory, *a)?,
                            idx(&memory, *b)?,
                            u32::max(*bits + *bits % 2, 4),
                        )?;
                        mem_push(std.convert(layouter, &bit)?, &mut memory)?;
                    }
                    I::PublicInput { guard } | I::PrivateInput { guard } => {
                        let guard = guard.map(|guard| idx(&memory, guard)).transpose()?;
                        witness
                            .error_if_known_and(|preproc| memory.len() > preproc.memory.len())?;
                        let value = witness.as_ref().map(|preproc| preproc.memory[memory.len()]);
                        let value_cell = std.assign(layouter, value)?;
                        // If `guard` is Some, then we want to ensure that
                        // `value` is 0 if `guard` is 0
                        // That is: guard == 0 -> value == 0
                        // => value == 0 || guard
                        if let Some(guard) = guard {
                            let value_is_zero = std.is_zero(layouter, &value_cell)?;
                            let guard_bit = std.convert(layouter, guard)?;
                            let is_ok = std.or(layouter, &[value_is_zero, guard_bit])?;
                            let is_ok_field = std.convert(layouter, &is_ok)?;
                            std.assert_non_zero(layouter, &is_ok_field)?;
                        }
                        mem_push(value_cell, &mut memory)?;
                    }
                    I::DivModPowerOfTwo { var, bits } => {
                        let var = idx(&memory, *var)?;
                        let var_bits = std.assigned_to_le_bits(layouter, var, None, true)?;
                        let modulus =
                            std.assigned_from_le_bits(layouter, &var_bits[..*bits as usize])?;

                        let divisor =
                            std.assigned_from_le_bits(layouter, &var_bits[*bits as usize..])?;

                        mem_push(divisor, &mut memory)?;
                        mem_push(modulus, &mut memory)?;
                    }
                    I::ReconstituteField {
                        divisor,
                        modulus,
                        bits,
                    } if self.version == IrMinorVersion::V0 => {
                        let divisor_bits = std.assigned_to_le_bits(
                            layouter,
                            idx(&memory, *divisor)?,
                            Some(FR_BITS - *bits as usize),
                            *bits as usize >= FR_BITS,
                        )?;
                        let modulus_bits = std.assigned_to_le_bits(
                            layouter,
                            idx(&memory, *modulus)?,
                            Some(*bits as usize),
                            true,
                        )?;

                        let reconstituted = std.assigned_from_le_bits(
                            layouter,
                            &[modulus_bits, divisor_bits].concat(),
                        )?;
                        mem_push(reconstituted, &mut memory)?;
                    }
                    I::ReconstituteField {
                        divisor,
                        modulus,
                        bits,
                    } => {
                        let modulus = idx(&memory, *modulus)?.clone();
                        let divisor = idx(&memory, *divisor)?.clone();

                        std.assert_lower_than_fixed(
                            layouter,
                            &divisor,
                            &(BigUint::from(1u32) << (FR_BITS as u32 - *bits)),
                        )?;
                        std.assert_lower_than_fixed(
                            layouter,
                            &modulus,
                            &(BigUint::from(1u32) << *bits),
                        )?;

                        let reconstituted = std.linear_combination(
                            layouter,
                            &[
                                (outer::Scalar::from(1), modulus),
                                (outer::Scalar::from(2).pow([*bits as u64]), divisor),
                            ],
                            outer::Scalar::from(0),
                        )?;
                        mem_push(reconstituted, &mut memory)?;
                    }
                    I::EcAdd { a_x, a_y, b_x, b_y } => {
                        let a = ecc_from_parts(
                            std,
                            layouter,
                            idx(&memory, *a_x)?,
                            idx(&memory, *a_y)?,
                        )?;
                        let b = ecc_from_parts(
                            std,
                            layouter,
                            idx(&memory, *b_x)?,
                            idx(&memory, *b_y)?,
                        )?;
                        let c = EccInstructions::add(std.jubjub(), layouter, &a, &b)?;
                        mem_push(std.jubjub().x_coordinate(&c), &mut memory)?;
                        mem_push(std.jubjub().y_coordinate(&c), &mut memory)?;
                    }
                    I::EcMul { a_x, a_y, scalar } => {
                        let a = ecc_from_parts(
                            std,
                            layouter,
                            idx(&memory, *a_x)?,
                            idx(&memory, *a_y)?,
                        )?;
                        let scalar = std.jubjub().convert(layouter, idx(&memory, *scalar)?)?;
                        let b = std.jubjub().msm(layouter, &[scalar], &[a])?;
                        mem_push(std.jubjub().x_coordinate(&b), &mut memory)?;
                        mem_push(std.jubjub().y_coordinate(&b), &mut memory)?;
                    }
                    I::EcMulGenerator { scalar } => {
                        let g: AssignedNativePoint<embedded::AffineExtended> = std
                            .jubjub()
                            .assign_fixed(layouter, embedded::Affine::generator())?;
                        let scalar = std.jubjub().convert(layouter, idx(&memory, *scalar)?)?;
                        let b = std.jubjub().msm(layouter, &[scalar], &[g])?;
                        mem_push(std.jubjub().x_coordinate(&b), &mut memory)?;
                        mem_push(std.jubjub().y_coordinate(&b), &mut memory)?;
                    }
                    I::HashToCurve { inputs } => {
                        let inputs = inputs
                            .iter()
                            .map(|input| idx(&memory, *input).cloned())
                            .collect::<Result<Vec<_>, _>>()?;
                        let point = std.hash_to_curve(layouter, &inputs)?;
                        mem_push(std.jubjub().x_coordinate(&point), &mut memory)?;
                        mem_push(std.jubjub().y_coordinate(&point), &mut memory)?;
                    }
                }
            }
        }
        if self.do_communications_commitment {
            let comm_comm_rand_value = comm_comm_value.map(|c| {
                c.ok_or_else(|| {
                    error!("Communication commitment not present despite preproc. This is a bug.");
                    Error::Synthesis("Communication commitment not present despite preproc.".into())
                })
                .unwrap()
                .1
            });
            let comm_comm_rand = std.assign(layouter, comm_comm_rand_value)?;

            let mut preimage = vec![comm_comm_rand];
            preimage.extend(inputs.iter().cloned());
            preimage.extend(outputs.iter().cloned());
            let comm_comm = std.poseidon(layouter, &preimage)?;
            // Nb. The communications commitment is the second public input
            // by convention
            std.assert_equal(layouter, &comm_comm, &public_inputs[1])?;
        }

        public_inputs
            .iter()
            .try_for_each(|x| std.constrain_as_public_input(layouter, x))
    }

    fn used_chips(&self) -> ZkStdLibArch {
        let jubjub = self.instructions.iter().any(|op| {
            matches!(
                op,
                I::EcAdd { .. }
                    | I::EcMul { .. }
                    | I::EcMulGenerator { .. }
                    | I::HashToCurve { .. }
            )
        });
        let hash_to_curve = self
            .instructions
            .iter()
            .any(|op| matches!(op, I::HashToCurve { .. }));
        let poseidon = self.do_communications_commitment
            || self
                .instructions
                .iter()
                .any(|op| matches!(op, I::TransientHash { .. }));
        let sha2_256 = self
            .instructions
            .iter()
            .any(|op| matches!(op, I::PersistentHash { .. }));
        let nr_pow2range_cols = match self.version {
            IrMinorVersion::V0 => 1,
            IrMinorVersion::V1 | IrMinorVersion::V2 => 4,
        };
        ZkStdLibArch {
            jubjub: jubjub || hash_to_curve,
            poseidon: poseidon || hash_to_curve,
            sha2_256,
            sha2_512: false,
            sha3_256: false,
            keccak_256: false,
            blake2b: false,
            nr_pow2range_cols,
            secp256k1: false,
            p256: false,
            bls12_381: false,
            curve25519: false,
            base64: false,
            automaton: false,
        }
    }

    fn write_relation<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        if let Ok(old_ir) = OldIrSource::try_from(self) {
            Serializable::serialize(&old_ir, writer)
        } else {
            // 0xff is a magic byte indicating non-v0 versions. These are tagged, and in a vec
            // container. This byte is *not* representable in v0, as it is illegal for a SCALE
            // encoded u32.
            writer.write_all(&[0xff])?;
            let mut raw = Vec::new();
            tagged_serialize(&self, &mut raw)?;
            raw.serialize(writer)
        }
    }

    fn read_relation<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let first_byte: u8 = Deserializable::deserialize(reader, 0)?;
        if first_byte != 0xff {
            let mut cursor = &[first_byte][..];
            let mut reader = Read::chain(&mut cursor, reader);
            OldIrSource::deserialize(&mut reader, 0).map(Into::into)
        } else {
            let raw: Vec<u8> = Deserializable::deserialize(reader, 0)?;
            tagged_deserialize(&mut &raw[..])
        }
    }
}
