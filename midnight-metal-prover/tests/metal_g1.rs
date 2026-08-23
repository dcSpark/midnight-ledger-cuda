// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::mem::size_of;

use midnight_metal_prover::{Bls12_381G1Affine, Bls12_381G1Projective, MetalError, MetalSession};

const RANDOM_CASES: usize = 512;

#[test]
fn g1_metal_operations_match_blst_on_boundaries_and_random_points() {
    let cases = differential_cases(RANDOM_CASES);
    let projective = cases
        .iter()
        .map(|(_, projective, _)| raw_projective(*projective))
        .collect::<Vec<_>>();
    let affine = cases
        .iter()
        .map(|(_, _, affine)| raw_affine(*affine))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_g1(&projective, &affine)
        .expect("execute batched G1 probe");

    for (index, (label, projective, affine)) in cases.iter().enumerate() {
        assert_eq!(
            actual.normalized[index],
            normalize(*projective),
            "normalize[{index}] {label}"
        );
        assert_eq!(
            actual.doubled[index],
            double(*projective),
            "double[{index}] {label}"
        );
        assert_eq!(
            actual.mixed_sum[index],
            mixed_add(*projective, *affine),
            "mixed_add[{index}] {label}"
        );
    }

    assert_eq!(actual.normalized[0], Bls12_381G1Affine::default());
    assert_eq!(actual.doubled[0], Bls12_381G1Affine::default());
    assert_eq!(actual.mixed_sum[0], Bls12_381G1Affine::default());
    assert_eq!(actual.mixed_sum[4], Bls12_381G1Affine::default());
}

#[test]
fn complete_projective_add_matches_blst_on_boundaries_and_random_points() {
    let cases = complete_add_cases(RANDOM_CASES);
    let lhs = cases
        .iter()
        .map(|(_, lhs, _)| raw_projective(*lhs))
        .collect::<Vec<_>>();
    let rhs = cases
        .iter()
        .map(|(_, _, rhs)| raw_projective(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let actual = session
        .probe_bls12_381_g1_complete_add(&lhs, &rhs)
        .expect("execute complete projective-add probe");

    for (index, (label, lhs, rhs)) in cases.iter().enumerate() {
        assert_eq!(
            actual[index],
            projective_add(*lhs, *rhs),
            "complete_add[{index}] {label}"
        );
    }

    assert_eq!(actual[0], Bls12_381G1Affine::default());
    assert_eq!(actual[4], Bls12_381G1Affine::default());
}

#[test]
fn homogeneous_complete_add_chains_match_repeated_blst_addition() {
    let additions_per_chain = 7;
    let cases = complete_add_cases(64);
    let lhs = cases
        .iter()
        .map(|(_, lhs, _)| raw_projective(*lhs))
        .collect::<Vec<_>>();
    let rhs = cases
        .iter()
        .map(|(_, _, rhs)| raw_projective(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();
    let actual = session
        .probe_bls12_381_g1_complete_add_chain(&lhs, &rhs, additions_per_chain, 2)
        .expect("execute homogeneous complete-add chains");

    for (index, (label, lhs, rhs)) in cases.iter().enumerate() {
        assert_eq!(
            normalize(raw_to_projective(actual.projective[index])),
            repeated_projective_add(*lhs, *rhs, additions_per_chain),
            "complete_add_chain[{index}] {label}"
        );
    }
    assert_eq!(actual.timed_commands, 1);
    assert_eq!(actual.timed_dispatches, 2);
    assert_ne!(actual.timed_duration, std::time::Duration::ZERO);
    let activity = session.activity();
    assert_eq!(activity.command_buffers, 3);
    assert_eq!(activity.dispatches, 5);
    assert_eq!(activity.host_waits, 3);
}

#[test]
fn g1_probe_handles_empty_and_rejects_mismatched_batches() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();

    let empty = session
        .probe_bls12_381_g1(&[], &[])
        .expect("empty G1 batch is valid");
    assert!(empty.normalized.is_empty());
    assert!(empty.doubled.is_empty());
    assert!(empty.mixed_sum.is_empty());
    assert_eq!(session.activity().command_buffers, 0);

    let error = session
        .probe_bls12_381_g1(&[Bls12_381G1Projective::default()], &[])
        .expect_err("mismatched G1 batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));

    let complete_empty = session
        .probe_bls12_381_g1_complete_add(&[], &[])
        .expect("empty complete-add batch is valid");
    assert!(complete_empty.is_empty());

    let error = session
        .probe_bls12_381_g1_complete_add(&[Bls12_381G1Projective::default()], &[])
        .expect_err("mismatched complete-add batches must be rejected");
    assert!(matches!(
        error,
        MetalError::LengthMismatch { left: 1, right: 0 }
    ));
}

#[test]
fn homogeneous_identity_fill_uses_zero_montgomery_one_zero() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();

    assert!(
        session
            .probe_bls12_381_g1_homogeneous_identity_fill(0)
            .expect("empty identity fill is valid")
    );
    assert_eq!(session.activity().command_buffers, 0);

    let count = 17;
    assert!(
        session
            .probe_bls12_381_g1_homogeneous_identity_fill(count)
            .expect("execute homogeneous identity fill")
    );
    let activity = session.activity();
    assert_eq!(activity.command_buffers, 1);
    assert_eq!(activity.dispatches, 1);
    assert_eq!(activity.host_waits, 1);
    assert_eq!(activity.host_write_bytes, size_of::<u32>() as u64);
    assert_eq!(
        activity.host_read_bytes,
        (count * size_of::<Bls12_381G1Projective>()) as u64
    );
}

#[test]
#[ignore = "release-mode end-to-end Metal G1 throughput diagnostic"]
fn report_g1_probe_throughput() {
    if cfg!(debug_assertions) {
        panic!(
            "run this diagnostic with `cargo test -p midnight-metal-prover --release \
             report_g1_probe_throughput -- --ignored --nocapture`"
        );
    }

    let elements = environment_usize("MIDNIGHT_METAL_G1_BENCH_ELEMENTS", 2_048);
    let rounds = environment_usize("MIDNIGHT_METAL_G1_BENCH_ROUNDS", 4);
    assert!(elements != 0, "G1 benchmark elements must be nonzero");
    assert!(rounds != 0, "G1 benchmark rounds must be nonzero");

    let cases = differential_cases(elements);
    let projective = cases
        .iter()
        .take(elements)
        .map(|(_, projective, _)| raw_projective(*projective))
        .collect::<Vec<_>>();
    let affine = cases
        .iter()
        .take(elements)
        .map(|(_, _, affine)| raw_affine(*affine))
        .collect::<Vec<_>>();
    assert_eq!(projective.len(), elements);

    let session = MetalSession::new().expect("initialize Metal runtime");
    session
        .probe_bls12_381_g1(&projective, &affine)
        .expect("warm up G1 pipelines");
    let start = std::time::Instant::now();
    let mut last = None;
    for _ in 0..rounds {
        last = Some(
            session
                .probe_bls12_381_g1(&projective, &affine)
                .expect("execute G1 throughput probe"),
        );
    }
    let elapsed = start.elapsed();
    assert_eq!(
        last.expect("at least one benchmark round").mixed_sum.len(),
        elements
    );

    let point_triplets = elements as f64 * rounds as f64;
    let device = session.device_info().expect("query Metal device");
    println!(
        "metal_g1_probe_throughput device={:?} elements={elements} rounds={rounds} \
         point_triplets_per_second={:.0} affine_outputs_per_second={:.0} elapsed_ms={:.3}",
        device.name,
        point_triplets / elapsed.as_secs_f64(),
        3.0 * point_triplets / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1_000.0
    );
}

#[test]
#[ignore = "release-mode homogeneous complete-add throughput diagnostic"]
fn report_g1_complete_add_chain_throughput() {
    if cfg!(debug_assertions) {
        panic!(
            "run this diagnostic with `cargo test -p midnight-metal-prover --release \
             report_g1_complete_add_chain_throughput -- --ignored --nocapture`"
        );
    }

    let elements = environment_usize("MIDNIGHT_METAL_G1_CHAIN_ELEMENTS", 8_192);
    let additions = environment_usize("MIDNIGHT_METAL_G1_CHAIN_ADDITIONS", 64);
    let timed_dispatches = environment_usize("MIDNIGHT_METAL_G1_CHAIN_DISPATCHES", 4);
    assert!(elements != 0, "G1 chain elements must be nonzero");
    let additions = u32::try_from(additions).expect("G1 chain additions fit u32");
    let timed_dispatches =
        u32::try_from(timed_dispatches).expect("G1 chain dispatch count fits u32");
    assert!(additions != 0, "G1 chain additions must be nonzero");
    assert!(
        timed_dispatches != 0,
        "G1 chain dispatch count must be nonzero"
    );

    let cases = complete_add_cases(elements);
    let lhs = cases
        .iter()
        .take(elements)
        .map(|(_, lhs, _)| raw_projective(*lhs))
        .collect::<Vec<_>>();
    let rhs = cases
        .iter()
        .take(elements)
        .map(|(_, _, rhs)| raw_projective(*rhs))
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    session.reset_activity();
    let result = session
        .probe_bls12_381_g1_complete_add_chain(&lhs, &rhs, additions, timed_dispatches)
        .expect("execute homogeneous complete-add throughput chains");
    assert_eq!(result.projective.len(), elements);

    let group_adds = elements as f64 * f64::from(additions) * f64::from(timed_dispatches);
    let group_adds_per_second = group_adds / result.timed_duration.as_secs_f64();
    let effective_fp_muls_per_second = 12.0 * group_adds_per_second;
    let activity = session.activity();
    let device = session.device_info().expect("query Metal device");
    println!(
        "metal_g1_complete_add_chain device={:?} elements={elements} additions_per_chain={} \
         timed_commands={} timed_dispatches={} timed_waits=1 total_commands={} \
         total_dispatches={} total_waits={} group_adds_per_second={:.0} \
         effective_fp_muls_per_second={:.0} elapsed_ms={:.3}",
        device.name,
        additions,
        result.timed_commands,
        result.timed_dispatches,
        activity.command_buffers,
        activity.dispatches,
        activity.host_waits,
        group_adds_per_second,
        effective_fp_muls_per_second,
        result.timed_duration.as_secs_f64() * 1_000.0
    );
}

fn differential_cases(random_cases: usize) -> Vec<(String, blst::blst_p1, blst::blst_p1_affine)> {
    let identity = blst::blst_p1::default();
    let affine_identity = blst::blst_p1_affine::default();
    let generator = generator();
    let affine_generator = to_affine(generator);
    let negative_generator = negate_affine(affine_generator);
    let doubled_generator = scalar_multiple(2);
    let non_subgroup_affine = authoritative_non_subgroup_affine();
    let non_subgroup = projective_from_affine(non_subgroup_affine);

    let mut noncanonical_identity = generator;
    noncanonical_identity.z = blst::blst_fp::default();
    let mut cases = vec![
        ("infinity + infinity".to_owned(), identity, affine_identity),
        (
            "infinity + generator".to_owned(),
            identity,
            affine_generator,
        ),
        (
            "generator + infinity".to_owned(),
            generator,
            affine_identity,
        ),
        (
            "generator + generator".to_owned(),
            generator,
            affine_generator,
        ),
        (
            "generator + inverse".to_owned(),
            generator,
            negative_generator,
        ),
        (
            "double-generator + generator".to_owned(),
            doubled_generator,
            affine_generator,
        ),
        (
            "noncanonical projective infinity".to_owned(),
            noncanonical_identity,
            affine_generator,
        ),
        (
            "non-subgroup projective + generator".to_owned(),
            non_subgroup,
            affine_generator,
        ),
        (
            "generator + non-subgroup affine".to_owned(),
            generator,
            non_subgroup_affine,
        ),
    ];

    let mut state = 0x4731_4d45_5441_4c21;
    for index in 0..random_cases {
        let lhs_scalar = splitmix64(&mut state) | 1;
        let rhs_scalar = splitmix64(&mut state) | 1;
        let scale = splitmix64(&mut state) | 1;
        let projective = rescale_projective(scalar_multiple(lhs_scalar), scale);
        let mut affine = to_affine(scalar_multiple(rhs_scalar));
        let case_kind = index % 32;
        if case_kind == 0 {
            affine = to_affine(projective);
        } else if case_kind == 1 {
            affine = negate_affine(to_affine(projective));
        } else if case_kind == 2 {
            affine = affine_identity;
        }
        cases.push((format!("random scalar case {index}"), projective, affine));
    }
    cases
}

fn complete_add_cases(random_cases: usize) -> Vec<(String, blst::blst_p1, blst::blst_p1)> {
    let identity = blst::blst_p1::default();
    let generator = generator();
    let negative_generator = negate_projective(generator);
    let doubled_generator = scalar_multiple(2);
    let non_subgroup = projective_from_affine(authoritative_non_subgroup_affine());

    let mut noncanonical_identity = generator;
    noncanonical_identity.z = blst::blst_fp::default();
    let rescaled_generator = rescale_projective(generator, 7);
    let rescaled_negative = rescale_projective(negative_generator, 11);

    let mut cases = vec![
        ("infinity + infinity".to_owned(), identity, identity),
        ("infinity + generator".to_owned(), identity, generator),
        ("generator + infinity".to_owned(), generator, identity),
        ("generator + generator".to_owned(), generator, generator),
        (
            "generator + inverse".to_owned(),
            generator,
            negative_generator,
        ),
        (
            "double-generator + generator".to_owned(),
            doubled_generator,
            generator,
        ),
        (
            "noncanonical lhs infinity".to_owned(),
            noncanonical_identity,
            generator,
        ),
        (
            "noncanonical rhs infinity".to_owned(),
            generator,
            noncanonical_identity,
        ),
        (
            "equal point with distinct Jacobian scales".to_owned(),
            generator,
            rescaled_generator,
        ),
        (
            "inverse points with distinct Jacobian scales".to_owned(),
            rescaled_generator,
            rescaled_negative,
        ),
        (
            "non-subgroup + generator".to_owned(),
            non_subgroup,
            generator,
        ),
        (
            "generator + non-subgroup".to_owned(),
            generator,
            non_subgroup,
        ),
        (
            "non-subgroup doubling through complete add".to_owned(),
            non_subgroup,
            non_subgroup,
        ),
    ];

    let mut state = 0x5243_4241_4444_3037;
    for index in 0..random_cases {
        let lhs_scalar = splitmix64(&mut state) | 1;
        let rhs_scalar = splitmix64(&mut state) | 1;
        let lhs_scale = splitmix64(&mut state) | 1;
        let rhs_scale = splitmix64(&mut state) | 1;
        let lhs = rescale_projective(scalar_multiple(lhs_scalar), lhs_scale);
        let mut rhs = rescale_projective(scalar_multiple(rhs_scalar), rhs_scale);
        let case_kind = index % 32;
        if case_kind == 0 {
            rhs = rescale_projective(lhs, rhs_scale);
        } else if case_kind == 1 {
            rhs = rescale_projective(negate_projective(lhs), rhs_scale);
        } else if case_kind == 2 {
            rhs = identity;
        }
        cases.push((format!("random projective case {index}"), lhs, rhs));
    }
    cases
}

fn generator() -> blst::blst_p1 {
    // SAFETY: BLST returns a process-lifetime pointer to its immutable G1 generator.
    unsafe { *blst::blst_p1_generator() }
}

fn scalar_multiple(scalar: u64) -> blst::blst_p1 {
    if scalar == 0 {
        return blst::blst_p1::default();
    }
    let scalar = scalar.to_le_bytes();
    let mut output = blst::blst_p1::default();
    // SAFETY: the generator pointer and eight-byte little-endian scalar remain
    // valid for the call, and `output` is a valid non-overlapping destination.
    unsafe {
        blst::blst_p1_mult(
            &mut output,
            blst::blst_p1_generator(),
            scalar.as_ptr(),
            u64::BITS as usize,
        );
    }
    output
}

fn rescale_projective(mut point: blst::blst_p1, scale: u64) -> blst::blst_p1 {
    let scale = fp_from_u64(scale);
    let mut scale_squared = blst::blst_fp::default();
    let mut scale_cubed = blst::blst_fp::default();
    // SAFETY: every BLST field operand is initialized and each destination is
    // distinct from its immutable inputs for the duration of the calls.
    unsafe {
        blst::blst_fp_sqr(&mut scale_squared, &scale);
        blst::blst_fp_mul(&mut scale_cubed, &scale_squared, &scale);
        let input = point.x;
        blst::blst_fp_mul(&mut point.x, &input, &scale_squared);
        let input = point.y;
        blst::blst_fp_mul(&mut point.y, &input, &scale_cubed);
        let input = point.z;
        blst::blst_fp_mul(&mut point.z, &input, &scale);
    }
    point
}

fn fp_from_u64(value: u64) -> blst::blst_fp {
    let canonical = [value, 0, 0, 0, 0, 0];
    let mut output = blst::blst_fp::default();
    // SAFETY: `canonical` contains the six little-endian input limbs expected
    // by BLST and `output` remains valid for the duration of the call.
    unsafe {
        blst::blst_fp_from_uint64(&mut output, canonical.as_ptr());
    }
    output
}

fn authoritative_non_subgroup_affine() -> blst::blst_p1_affine {
    // This is the exact search domain and acceptance predicate used by
    // vendored midnight-curves 0.3.1 `g1_non_torsion_point`: scan x from one,
    // solve y^2=x^3+4, then require BLST's on-curve predicate and reject its
    // prime-subgroup predicate. No map-to-curve or unchecked test generator is
    // invented here.
    let four = fp_from_u64(4);
    for x_value in 1..=u64::MAX {
        let x = fp_from_u64(x_value);
        let mut x_squared = blst::blst_fp::default();
        let mut x_cubed = blst::blst_fp::default();
        let mut rhs = blst::blst_fp::default();
        let mut y = blst::blst_fp::default();
        // SAFETY: all field operands and distinct destinations are initialized
        // and remain valid for each BLST call.
        let has_square_root = unsafe {
            blst::blst_fp_sqr(&mut x_squared, &x);
            blst::blst_fp_mul(&mut x_cubed, &x_squared, &x);
            blst::blst_fp_add(&mut rhs, &x_cubed, &four);
            blst::blst_fp_sqrt(&mut y, &rhs)
        };
        if has_square_root {
            let point = blst::blst_p1_affine { x, y };
            // SAFETY: `point` contains initialized Fp coordinates.
            let (on_curve, in_prime_subgroup) = unsafe {
                (
                    blst::blst_p1_affine_on_curve(&point),
                    blst::blst_p1_affine_in_g1(&point),
                )
            };
            if on_curve && !in_prime_subgroup {
                return point;
            }
        }
    }
    unreachable!("the authoritative G1 non-subgroup search must find a point")
}

fn projective_from_affine(affine: blst::blst_p1_affine) -> blst::blst_p1 {
    let mut projective = blst::blst_p1::default();
    // SAFETY: both point objects are initialized and non-overlapping.
    unsafe {
        blst::blst_p1_from_affine(&mut projective, &affine);
    }
    projective
}

fn negate_affine(affine: blst::blst_p1_affine) -> blst::blst_p1_affine {
    let mut projective = blst::blst_p1::default();
    // SAFETY: all BLST point pointers refer to initialized, valid storage.
    unsafe {
        blst::blst_p1_from_affine(&mut projective, &affine);
        blst::blst_p1_cneg(&mut projective, true);
    }
    to_affine(projective)
}

fn negate_projective(mut projective: blst::blst_p1) -> blst::blst_p1 {
    // SAFETY: `projective` is initialized and BLST permits in-place negation.
    unsafe {
        blst::blst_p1_cneg(&mut projective, true);
    }
    projective
}

fn normalize(point: blst::blst_p1) -> Bls12_381G1Affine {
    raw_affine(to_affine(point))
}

fn double(point: blst::blst_p1) -> Bls12_381G1Affine {
    let mut output = blst::blst_p1::default();
    // SAFETY: input and output are initialized non-overlapping BLST points.
    unsafe {
        blst::blst_p1_double(&mut output, &point);
    }
    normalize(output)
}

fn mixed_add(projective: blst::blst_p1, affine: blst::blst_p1_affine) -> Bls12_381G1Affine {
    let mut output = blst::blst_p1::default();
    // SAFETY: both inputs and the non-overlapping output are initialized BLST
    // point objects that remain valid for the duration of the call.
    unsafe {
        blst::blst_p1_add_or_double_affine(&mut output, &projective, &affine);
    }
    normalize(output)
}

fn projective_add(lhs: blst::blst_p1, rhs: blst::blst_p1) -> Bls12_381G1Affine {
    let mut output = blst::blst_p1::default();
    // SAFETY: both inputs and the non-overlapping output are initialized BLST
    // point objects that remain valid for the duration of the call.
    unsafe {
        blst::blst_p1_add_or_double(&mut output, &lhs, &rhs);
    }
    normalize(output)
}

fn repeated_projective_add(
    mut accumulator: blst::blst_p1,
    addend: blst::blst_p1,
    additions: u32,
) -> Bls12_381G1Affine {
    for _ in 0..additions {
        let mut output = blst::blst_p1::default();
        // SAFETY: both inputs and the non-overlapping output are initialized
        // BLST point objects that remain valid for the duration of the call.
        unsafe {
            blst::blst_p1_add_or_double(&mut output, &accumulator, &addend);
        }
        accumulator = output;
    }
    normalize(accumulator)
}

fn to_affine(point: blst::blst_p1) -> blst::blst_p1_affine {
    let mut output = blst::blst_p1_affine::default();
    // SAFETY: the input and output are initialized BLST point objects.
    unsafe {
        blst::blst_p1_to_affine(&mut output, &point);
    }
    output
}

fn raw_projective(point: blst::blst_p1) -> Bls12_381G1Projective {
    Bls12_381G1Projective {
        x: point.x.l,
        y: point.y.l,
        z: point.z.l,
    }
}

fn raw_to_projective(point: Bls12_381G1Projective) -> blst::blst_p1 {
    let mut output = blst::blst_p1::default();
    output.x.l = point.x;
    output.y.l = point.y;
    output.z.l = point.z;
    output
}

fn raw_affine(point: blst::blst_p1_affine) -> Bls12_381G1Affine {
    Bls12_381G1Affine {
        x: point.x.l,
        y: point.y.l,
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn environment_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|error| panic!("invalid {name}={value:?}: {error}"))
        })
        .unwrap_or(default)
}
