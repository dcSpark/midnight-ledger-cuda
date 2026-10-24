// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Proofs made with Metal selected, each verified on the CPU. A wrong batched
//! fixed-base MSM in the prover shows here as a proof that does not verify.

use midnight_circuits::{
    hash::poseidon::PoseidonChip,
    instructions::{AssignmentInstructions, PublicInputInstructions, hash::HashCPU},
};
use midnight_curves::{Bls12, gpu_metal::EnableGuard};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
    poly::kzg::params::ParamsKZG,
};
use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

type F = midnight_curves::Fq;

const K: u32 = 16;
/// Poseidon hashes in the chain; 3200 fills a k16 circuit.
const HASHES: usize = 3200;
const PROOFS: u64 = 40;

#[derive(Clone)]
struct PoseidonChain {
    hashes: usize,
}

impl Relation for PoseidonChain {
    type Instance = F;
    type Witness = F;
    type Error = Error;

    fn format_instance(instance: &F) -> Result<Vec<F>, Error> {
        Ok(vec![*instance])
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        _instance: Value<F>,
        witness: Value<F>,
    ) -> Result<(), Error> {
        let mut value = std_lib.assign(layouter, witness)?;
        for _ in 0..self.hashes {
            value = std_lib.poseidon(layouter, &[value])?;
        }
        std_lib.constrain_as_public_input(layouter, &value)
    }

    fn used_chips(&self) -> ZkStdLibArch {
        ZkStdLibArch { poseidon: true, nr_pow2range_cols: 4, ..ZkStdLibArch::default() }
    }

    fn write_relation<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&(self.hashes as u64).to_le_bytes())
    }

    fn read_relation<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(PoseidonChain { hashes: u64::from_le_bytes(bytes) as usize })
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn metal_proofs_verify_on_the_cpu() {
    let relation = PoseidonChain { hashes: HASHES };
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, ChaCha20Rng::seed_from_u64(1));
    let vk = midnight_zk_stdlib::setup_vk(&params, &relation);
    let pk = midnight_zk_stdlib::setup_pk(&relation, &vk);
    let witness = F::from(7u64);
    let mut instance = witness;
    for _ in 0..HASHES {
        instance = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[instance]);
    }

    let mut invalid = Vec::new();
    for proof_index in 0..PROOFS {
        let proof = {
            let _guard = EnableGuard::set(true);
            midnight_zk_stdlib::prove::<PoseidonChain, blake2b_simd::State>(
                &params,
                &pk,
                &relation,
                &instance,
                witness,
                ChaCha20Rng::seed_from_u64(1_000 + proof_index),
            )
            .expect("prove")
        };
        let verified = midnight_zk_stdlib::verify::<PoseidonChain, blake2b_simd::State>(
            &params.verifier_params(),
            &vk,
            &instance,
            None,
            &proof,
        );
        if verified.is_err() {
            invalid.push(proof_index);
        }
    }
    assert!(invalid.is_empty(), "Metal proofs that do not verify: {invalid:?} of {PROOFS}");
}
