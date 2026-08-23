// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::mem::{align_of, offset_of, size_of};
use std::sync::{Arc, Barrier};

use midnight_metal_prover::{
    Bls12_381Fp, Bls12_381Fr, Bls12_381G1Affine, Bls12_381G1Projective, MetalError, MetalSession,
    proof_byte_digest_cpu,
};

const FR_MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];
const FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

#[test]
fn metal_abi_probe_matches_the_rust_contract() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let probe = session
        .bls12_381_abi_probe()
        .expect("execute BLS12-381 ABI probe");

    assert_eq!(probe.abi_version, 1);
    assert_eq!(probe.fr_size, size_of::<Bls12_381Fr>() as u64);
    assert_eq!(probe.fr_alignment, align_of::<Bls12_381Fr>() as u64);
    assert_eq!(probe.fr_limbs_offset, offset_of!(Bls12_381Fr, limbs) as u64);
    assert_eq!(probe.fp_size, size_of::<Bls12_381Fp>() as u64);
    assert_eq!(probe.fp_alignment, align_of::<Bls12_381Fp>() as u64);
    assert_eq!(probe.fp_limbs_offset, offset_of!(Bls12_381Fp, limbs) as u64);
    assert_eq!(probe.g1_affine_size, size_of::<Bls12_381G1Affine>() as u64);
    assert_eq!(
        probe.g1_affine_alignment,
        align_of::<Bls12_381G1Affine>() as u64
    );
    assert_eq!(
        probe.g1_affine_x_offset,
        offset_of!(Bls12_381G1Affine, x) as u64
    );
    assert_eq!(
        probe.g1_affine_y_offset,
        offset_of!(Bls12_381G1Affine, y) as u64
    );
    assert_eq!(
        probe.g1_projective_size,
        size_of::<Bls12_381G1Projective>() as u64
    );
    assert_eq!(
        probe.g1_projective_alignment,
        align_of::<Bls12_381G1Projective>() as u64
    );
    assert_eq!(
        probe.g1_projective_x_offset,
        offset_of!(Bls12_381G1Projective, x) as u64
    );
    assert_eq!(
        probe.g1_projective_y_offset,
        offset_of!(Bls12_381G1Projective, y) as u64
    );
    assert_eq!(
        probe.g1_projective_z_offset,
        offset_of!(Bls12_381G1Projective, z) as u64
    );
    assert_eq!(
        probe.g1_homogeneous_size,
        size_of::<Bls12_381G1Projective>() as u64
    );
    assert_eq!(
        probe.g1_homogeneous_alignment,
        align_of::<Bls12_381G1Projective>() as u64
    );
    assert_eq!(
        probe.g1_homogeneous_x_offset,
        offset_of!(Bls12_381G1Projective, x) as u64
    );
    assert_eq!(
        probe.g1_homogeneous_y_offset,
        offset_of!(Bls12_381G1Projective, y) as u64
    );
    assert_eq!(
        probe.g1_homogeneous_z_offset,
        offset_of!(Bls12_381G1Projective, z) as u64
    );
    assert_eq!(probe.layout_fingerprint, 0x108c_b973_1d50_97c4);
    assert_ne!(probe.constants_fingerprint, 0);
}

#[test]
fn fr64_metal_operations_match_blst_on_boundaries_and_random_vectors() {
    let mut canonical_pairs = Vec::new();
    let boundaries = boundary_values();
    for lhs in boundaries.iter().copied() {
        for rhs in boundaries.iter().copied() {
            canonical_pairs.push((lhs, rhs));
        }
    }

    let mut state = 0x4d49_444e_4947_4854;
    for _ in 0..4_096 {
        canonical_pairs.push((random_canonical(&mut state), random_canonical(&mut state)));
    }

    let lhs = canonical_pairs
        .iter()
        .map(|(lhs, _)| montgomery_from_canonical(*lhs))
        .collect::<Vec<_>>();
    let rhs = canonical_pairs
        .iter()
        .map(|(_, rhs)| montgomery_from_canonical(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_fr64(&lhs, &rhs)
        .expect("execute batched Fr64 probe");

    for index in 0..lhs.len() {
        assert_eq!(
            actual.add[index],
            blst_add(lhs[index], rhs[index]),
            "add[{index}]"
        );
        assert_eq!(
            actual.sub[index],
            blst_sub(lhs[index], rhs[index]),
            "sub[{index}]"
        );
        assert_eq!(
            actual.mul[index],
            blst_mul(lhs[index], rhs[index]),
            "mul[{index}]"
        );
        assert_eq!(
            actual.square[index],
            blst_square(lhs[index]),
            "square[{index}]"
        );
    }
}

#[test]
fn fp64_metal_operations_match_blst_on_boundaries_and_random_vectors() {
    let mut canonical_pairs = Vec::new();
    let boundaries = fp_boundary_values();
    for lhs in boundaries.iter().copied() {
        for rhs in boundaries.iter().copied() {
            canonical_pairs.push((lhs, rhs));
        }
    }

    let mut state = 0x4650_3634_4d45_544c;
    for _ in 0..4_096 {
        canonical_pairs.push((
            random_fp_canonical(&mut state),
            random_fp_canonical(&mut state),
        ));
    }

    let lhs = canonical_pairs
        .iter()
        .map(|(lhs, _)| fp_montgomery_from_canonical(*lhs))
        .collect::<Vec<_>>();
    let rhs = canonical_pairs
        .iter()
        .map(|(_, rhs)| fp_montgomery_from_canonical(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_fp64(&lhs, &rhs)
        .expect("execute batched Fp64 probe");

    for index in 0..lhs.len() {
        assert_eq!(
            actual.add[index],
            blst_fp_add(lhs[index], rhs[index]),
            "add[{index}]"
        );
        assert_eq!(
            actual.sub[index],
            blst_fp_sub(lhs[index], rhs[index]),
            "sub[{index}]"
        );
        assert_eq!(
            actual.mul[index],
            blst_fp_mul(lhs[index], rhs[index]),
            "mul[{index}]"
        );
        assert_eq!(
            actual.square[index],
            blst_fp_square(lhs[index]),
            "square[{index}]"
        );
    }
}

#[test]
fn fp32_metal_operations_match_blst_on_boundaries_and_random_vectors() {
    let mut canonical_pairs = Vec::new();
    let boundaries = fp_boundary_values();
    for lhs in boundaries.iter().copied() {
        for rhs in boundaries.iter().copied() {
            canonical_pairs.push((lhs, rhs));
        }
    }

    let mut state = 0x4650_3332_4d45_544c;
    for _ in 0..4_096 {
        canonical_pairs.push((
            random_fp_canonical(&mut state),
            random_fp_canonical(&mut state),
        ));
    }

    let lhs = canonical_pairs
        .iter()
        .map(|(lhs, _)| fp_montgomery_from_canonical(*lhs))
        .collect::<Vec<_>>();
    let rhs = canonical_pairs
        .iter()
        .map(|(_, rhs)| fp_montgomery_from_canonical(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_fp32(&lhs, &rhs)
        .expect("execute batched Fp32 diagnostic probe");

    for index in 0..lhs.len() {
        assert_eq!(
            actual.add[index],
            blst_fp_add(lhs[index], rhs[index]),
            "fp32 add[{index}]"
        );
        assert_eq!(
            actual.sub[index],
            blst_fp_sub(lhs[index], rhs[index]),
            "fp32 sub[{index}]"
        );
        assert_eq!(
            actual.mul[index],
            blst_fp_mul(lhs[index], rhs[index]),
            "fp32 mul[{index}]"
        );
        assert_eq!(
            actual.square[index],
            blst_fp_square(lhs[index]),
            "fp32 square[{index}]"
        );
    }
}

#[test]
fn fp32_wide_metal_operations_match_blst_on_boundaries_and_random_vectors() {
    let mut canonical_pairs = Vec::new();
    let boundaries = fp_boundary_values();
    for lhs in boundaries.iter().copied() {
        for rhs in boundaries.iter().copied() {
            canonical_pairs.push((lhs, rhs));
        }
    }

    let mut state = 0x4650_3357_4944_4521;
    for _ in 0..4_096 {
        canonical_pairs.push((
            random_fp_canonical(&mut state),
            random_fp_canonical(&mut state),
        ));
    }

    let lhs = canonical_pairs
        .iter()
        .map(|(lhs, _)| fp_montgomery_from_canonical(*lhs))
        .collect::<Vec<_>>();
    let rhs = canonical_pairs
        .iter()
        .map(|(_, rhs)| fp_montgomery_from_canonical(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_fp32_wide(&lhs, &rhs)
        .expect("execute batched wide-accumulator Fp32 diagnostic probe");

    for index in 0..lhs.len() {
        assert_eq!(
            actual.add[index],
            blst_fp_add(lhs[index], rhs[index]),
            "fp32-wide add[{index}]"
        );
        assert_eq!(
            actual.sub[index],
            blst_fp_sub(lhs[index], rhs[index]),
            "fp32-wide sub[{index}]"
        );
        assert_eq!(
            actual.mul[index],
            blst_fp_mul(lhs[index], rhs[index]),
            "fp32-wide mul[{index}]"
        );
        assert_eq!(
            actual.square[index],
            blst_fp_square(lhs[index]),
            "fp32-wide square[{index}]"
        );
    }
}

#[test]
fn field64_probes_handle_empty_and_reject_mismatched_batches() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();

    let empty = session
        .probe_bls12_381_fr64(&[], &[])
        .expect("empty batch is valid");
    assert!(empty.add.is_empty());
    assert!(empty.sub.is_empty());
    assert!(empty.mul.is_empty());
    assert!(empty.square.is_empty());
    assert_eq!(session.activity().command_buffers, 0);

    let error = session
        .probe_bls12_381_fr64(&[Bls12_381Fr::default()], &[])
        .expect_err("mismatched batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));

    let empty = session
        .probe_bls12_381_fp64(&[], &[])
        .expect("empty batch is valid");
    assert!(empty.add.is_empty());
    assert!(empty.sub.is_empty());
    assert!(empty.mul.is_empty());
    assert!(empty.square.is_empty());
    assert_eq!(session.activity().command_buffers, 0);

    let error = session
        .probe_bls12_381_fp64(&[Bls12_381Fp::default()], &[])
        .expect_err("mismatched batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));

    let empty = session
        .probe_bls12_381_fp32(&[], &[])
        .expect("empty Fp32 batch is valid");
    assert!(empty.add.is_empty());
    assert!(empty.sub.is_empty());
    assert!(empty.mul.is_empty());
    assert!(empty.square.is_empty());
    assert_eq!(session.activity().command_buffers, 0);

    let error = session
        .probe_bls12_381_fp32(&[Bls12_381Fp::default()], &[])
        .expect_err("mismatched Fp32 batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));

    let empty = session
        .probe_bls12_381_fp32_wide(&[], &[])
        .expect("empty wide-accumulator Fp32 batch is valid");
    assert!(empty.add.is_empty());
    assert!(empty.sub.is_empty());
    assert!(empty.mul.is_empty());
    assert!(empty.square.is_empty());
    assert_eq!(session.activity().command_buffers, 0);

    let error = session
        .probe_bls12_381_fp32_wide(&[Bls12_381Fp::default()], &[])
        .expect_err("mismatched wide-accumulator Fp32 batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));
}

#[test]
fn clear_words_probe_replaces_nonzero_sentinels() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let sentinels = [u64::MAX, 1, 0xdead_beef_cafe_babe, 17, 99];
    let cleared = session
        .clear_words_probe(&sentinels)
        .expect("execute clear-words probe");

    assert_eq!(cleared, vec![0; sentinels.len()]);
}

#[test]
fn shared_session_serializes_concurrent_encode_execute_and_readback() {
    const WORKERS: usize = 6;
    let session = Arc::new(MetalSession::new().expect("initialize Metal runtime"));
    let barrier = Arc::new(Barrier::new(WORKERS));
    let mut workers = Vec::new();

    for worker in 0..WORKERS {
        let session = Arc::clone(&session);
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let bytes = (0..(257 + worker))
                .map(|index| (index * 29 + worker * 17) as u8)
                .collect::<Vec<_>>();
            let canonical_lhs = [worker as u64 + 2, 0, 0, 0];
            let canonical_rhs = [worker as u64 * 3 + 7, 0, 0, 0];
            let lhs = montgomery_from_canonical(canonical_lhs);
            let rhs = montgomery_from_canonical(canonical_rhs);

            barrier.wait();
            let digest = session
                .proof_byte_digest(&bytes)
                .expect("concurrent digest execution");
            assert_eq!(digest, proof_byte_digest_cpu(&bytes));

            let field = session
                .probe_bls12_381_fr64(&[lhs], &[rhs])
                .expect("concurrent field execution");
            assert_eq!(field.mul[0], blst_mul(lhs, rhs));
        }));
    }

    for worker in workers {
        worker.join().expect("worker did not panic");
    }
    assert_eq!(session.activity().execution_failures, 0);
}

fn boundary_values() -> Vec<[u64; 4]> {
    let mut modulus_minus_one = FR_MODULUS;
    modulus_minus_one[0] -= 1;
    let mut modulus_minus_two = FR_MODULUS;
    modulus_minus_two[0] -= 2;

    vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        modulus_minus_one,
        modulus_minus_two,
        [u64::MAX, 0, 0, 0],
        [0, 1, 0, 0],
        [u64::MAX, u64::MAX, 0, 0],
        [0, 0, 1, 0],
        [u64::MAX, u64::MAX, u64::MAX, 0],
        [0, 0, 0, 1],
        [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555, 0, 0],
    ]
}

fn fp_boundary_values() -> Vec<[u64; 6]> {
    let mut modulus_minus_one = FP_MODULUS;
    modulus_minus_one[0] -= 1;
    let mut modulus_minus_two = FP_MODULUS;
    modulus_minus_two[0] -= 2;

    vec![
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
        modulus_minus_one,
        modulus_minus_two,
        [u64::MAX, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [u64::MAX, u64::MAX, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [u64::MAX, u64::MAX, u64::MAX, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, 1],
        [0, 0, 0, 0, 0, 1],
        [
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0,
            0,
        ],
    ]
}

fn random_canonical(state: &mut u64) -> [u64; 4] {
    loop {
        let candidate = [
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state) & 0x7fff_ffff_ffff_ffff,
        ];
        if less_than(&candidate, &FR_MODULUS) {
            return candidate;
        }
    }
}

fn random_fp_canonical(state: &mut u64) -> [u64; 6] {
    loop {
        let candidate = [
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state) & 0x1fff_ffff_ffff_ffff,
        ];
        if less_than_fp(&candidate, &FP_MODULUS) {
            return candidate;
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn less_than(lhs: &[u64; 4], rhs: &[u64; 4]) -> bool {
    for index in (0..4).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] < rhs[index];
        }
    }
    false
}

fn less_than_fp(lhs: &[u64; 6], rhs: &[u64; 6]) -> bool {
    for index in (0..6).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] < rhs[index];
        }
    }
    false
}

fn montgomery_from_canonical(canonical: [u64; 4]) -> Bls12_381Fr {
    assert!(less_than(&canonical, &FR_MODULUS));
    let mut raw = blst::blst_fr::default();
    // SAFETY: `raw` is valid output and `canonical` supplies exactly four
    // little-endian canonical limbs for the duration of the BLST call.
    unsafe {
        blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr());
    }
    Bls12_381Fr { limbs: raw.l }
}

fn fp_montgomery_from_canonical(canonical: [u64; 6]) -> Bls12_381Fp {
    assert!(less_than_fp(&canonical, &FP_MODULUS));
    let mut raw = blst::blst_fp::default();
    // SAFETY: `raw` is valid output and `canonical` supplies exactly six
    // little-endian canonical limbs for the duration of the BLST call.
    unsafe {
        blst::blst_fp_from_uint64(&mut raw, canonical.as_ptr());
    }
    Bls12_381Fp { limbs: raw.l }
}

fn blst_add(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // scalar-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fr_add(&mut output, &lhs, &rhs);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // scalar-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fr_sub(&mut output, &lhs, &rhs);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // scalar-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fr_mul(&mut output, &lhs, &rhs);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_square(value: Bls12_381Fr) -> Bls12_381Fr {
    let value = blst::blst_fr { l: value.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: the input and output are initialized, non-overlapping BLST
    // scalar-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fr_sqr(&mut output, &value);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_fp_add(lhs: Bls12_381Fp, rhs: Bls12_381Fp) -> Bls12_381Fp {
    let lhs = blst::blst_fp { l: lhs.limbs };
    let rhs = blst::blst_fp { l: rhs.limbs };
    let mut output = blst::blst_fp::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // base-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fp_add(&mut output, &lhs, &rhs);
    }
    Bls12_381Fp { limbs: output.l }
}

fn blst_fp_sub(lhs: Bls12_381Fp, rhs: Bls12_381Fp) -> Bls12_381Fp {
    let lhs = blst::blst_fp { l: lhs.limbs };
    let rhs = blst::blst_fp { l: rhs.limbs };
    let mut output = blst::blst_fp::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // base-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fp_sub(&mut output, &lhs, &rhs);
    }
    Bls12_381Fp { limbs: output.l }
}

fn blst_fp_mul(lhs: Bls12_381Fp, rhs: Bls12_381Fp) -> Bls12_381Fp {
    let lhs = blst::blst_fp { l: lhs.limbs };
    let rhs = blst::blst_fp { l: rhs.limbs };
    let mut output = blst::blst_fp::default();
    // SAFETY: all inputs and the output are initialized, non-overlapping BLST
    // base-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fp_mul(&mut output, &lhs, &rhs);
    }
    Bls12_381Fp { limbs: output.l }
}

fn blst_fp_square(value: Bls12_381Fp) -> Bls12_381Fp {
    let value = blst::blst_fp { l: value.limbs };
    let mut output = blst::blst_fp::default();
    // SAFETY: the input and output are initialized, non-overlapping BLST
    // base-field objects that live for the duration of the call.
    unsafe {
        blst::blst_fp_sqr(&mut output, &value);
    }
    Bls12_381Fp { limbs: output.l }
}
