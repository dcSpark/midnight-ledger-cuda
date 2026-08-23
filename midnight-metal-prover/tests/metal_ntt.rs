// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Barrier};

use midnight_metal_prover::{Bls12_381Fr, MetalError, MetalSession, NttDirection, NttDomainSpec};

const ONE: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x0000_0001_ffff_fffe,
        0x5884_b7fa_0003_4802,
        0x998c_4fef_ecbc_4ff5,
        0x1824_b159_acc5_056f,
    ],
};
const TWO_INV: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x0000_0000_ffff_ffff,
        0xac42_5bfd_0001_a401,
        0xccc6_27f7_f65e_27fa,
        0x0c12_58ac_d662_82b7,
    ],
};
const TWO_ADIC_ROOT: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0xb9b5_8d8c_5f0e_466a,
        0x5b1b_4c80_1819_d7ec,
        0x0af5_3ae3_52a3_1e64,
        0x5bf3_adda_19e9_b27b,
    ],
};
const TWO_ADIC_ROOT_INV: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x4256_481a_dcf3_219a,
        0x45f3_7b7f_96b6_cad3,
        0xf9c3_f1d7_5f7a_3b27,
        0x2d2f_c049_658a_fd43,
    ],
};
const ZETA: Bls12_381Fr = Bls12_381Fr {
    limbs: [
        0x92d9_090b_0930_11d2,
        0xfc9c_bd71_9d6a_a073,
        0xc1f1_4ef0_cd65_a1a6,
        0x017f_6d35_e72f_cdeb,
    ],
};

#[test]
fn small_natural_order_batches_match_blst_cpu_ntts() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    for log_size in 0..=12 {
        let domain = domain(log_size);
        let domain_len = 1usize << log_size;
        let batches = match log_size {
            0..=5 => 3,
            6..=8 => 2,
            _ => 1,
        };
        let values = deterministic_values(domain_len * batches, u64::from(log_size) + 0x51);

        let mut expected_forward = values.clone();
        cpu_ntt_batches(&mut expected_forward, domain_len, domain.root, None);
        let actual_forward = session
            .probe_bls12_381_ntt(&values, domain, NttDirection::Forward)
            .unwrap_or_else(|error| panic!("forward k={log_size}: {error}"));
        assert_fr_slices(
            &format!("small forward k={log_size}"),
            &expected_forward,
            &actual_forward,
        );

        let mut expected_inverse = values.clone();
        cpu_ntt_batches(
            &mut expected_inverse,
            domain_len,
            domain.inverse_root,
            Some(domain.inverse_size),
        );
        let actual_inverse = session
            .probe_bls12_381_ntt(&values, domain, NttDirection::Inverse)
            .unwrap_or_else(|error| panic!("inverse k={log_size}: {error}"));
        assert_fr_slices(
            &format!("small inverse k={log_size}"),
            &expected_inverse,
            &actual_inverse,
        );
    }
}

#[test]
fn product_domains_k13_through_k17_match_cpu_and_roundtrip() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    for log_size in 13..=17 {
        let domain = domain(log_size);
        let domain_len = 1usize << log_size;
        let batches = if matches!(log_size, 13 | 17) { 2 } else { 1 };
        let values = deterministic_values(domain_len * batches, 0x1300 + u64::from(log_size));
        let mut expected = values.clone();
        cpu_ntt_batches(&mut expected, domain_len, domain.root, None);

        let actual = session
            .probe_bls12_381_ntt(&values, domain, NttDirection::Forward)
            .unwrap_or_else(|error| panic!("product forward k={log_size}: {error}"));
        assert_fr_slices(&format!("product forward k={log_size}"), &expected, &actual);

        let restored = session
            .probe_bls12_381_ntt(&actual, domain, NttDirection::Inverse)
            .unwrap_or_else(|error| panic!("product inverse k={log_size}: {error}"));
        assert_fr_slices(
            &format!("product roundtrip k={log_size}"),
            &values,
            &restored,
        );
    }
}

#[test]
fn resident_coset_extension_and_interpolation_match_cpu() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    for (coefficient_log, extended_log, batches) in [(4, 7, 3), (10, 13, 2)] {
        let coefficient_len = 1usize << coefficient_log;
        let extended_len = 1usize << extended_log;
        let extended_domain = domain(extended_log);
        let coefficients = deterministic_values(
            coefficient_len * batches,
            0xc05e_7000 + u64::from(extended_log),
        );
        let expected_evaluations =
            cpu_coeff_to_extended(&coefficients, coefficient_len, extended_domain, batches);

        let actual_evaluations = session
            .probe_bls12_381_coeff_to_extended(&coefficients, coefficient_len, extended_domain)
            .unwrap_or_else(|error| {
                panic!("coeff_to_extended k={coefficient_log}->{extended_log}: {error}")
            });
        assert_fr_slices(
            &format!("coeff_to_extended k={coefficient_log}->{extended_log}"),
            &expected_evaluations,
            &actual_evaluations,
        );

        let expected_coefficients =
            cpu_extended_to_coeff(&expected_evaluations, extended_domain, batches);
        let actual_coefficients = session
            .probe_bls12_381_extended_to_coeff(&expected_evaluations, extended_domain)
            .unwrap_or_else(|error| panic!("extended_to_coeff k={extended_log}: {error}"));
        assert_fr_slices(
            &format!("extended_to_coeff k={extended_log}"),
            &expected_coefficients,
            &actual_coefficients,
        );

        for batch in 0..batches {
            assert_fr_slices(
                "coset roundtrip source prefix",
                &coefficients[batch * coefficient_len..(batch + 1) * coefficient_len],
                &actual_coefficients[batch * extended_len..batch * extended_len + coefficient_len],
            );
            assert!(
                actual_coefficients
                    [batch * extended_len + coefficient_len..(batch + 1) * extended_len]
                    .iter()
                    .all(|value| *value == Bls12_381Fr::default()),
                "zero extension was not restored for batch {batch}"
            );
        }
    }
}

#[test]
fn resident_roundtrip_has_one_work_command_plus_mandatory_clear() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let domain = domain(17);
    let domain_len = 1usize << domain.log_size;
    let values = deterministic_values(domain_len * 2, 0xac71_7171);

    // Populate the immutable domain cache outside the measured activity window.
    session
        .probe_bls12_381_ntt(&values[..domain_len], domain, NttDirection::Forward)
        .expect("warm cached NTT plan");
    session.reset_activity();
    let restored = session
        .probe_bls12_381_ntt_roundtrip(&values, domain)
        .expect("resident NTT roundtrip");
    assert_fr_slices("resident activity roundtrip", &values, &restored);

    let activity = session.activity();
    let value_bytes = size_of_val(values.as_slice()) as u64;
    assert_eq!(activity.command_buffers, 2);
    assert_eq!(activity.clear_commands, 1);
    assert_eq!(activity.command_buffers - activity.clear_commands, 1);
    assert_eq!(activity.dispatches, 6); // Four NTT passes plus two full clears.
    assert_eq!(activity.clear_dispatches, 2);
    assert_eq!(activity.dispatches - activity.clear_dispatches, 4);
    assert_eq!(activity.host_waits, 2);
    assert_eq!(activity.clear_waits, 1);
    assert_eq!(activity.host_waits - activity.clear_waits, 1);
    assert_eq!(activity.allocations, 2);
    assert_eq!(activity.allocated_bytes, 2 * value_bytes);
    assert_eq!(activity.pool_misses, 2);
    assert_eq!(activity.pool_hits, 0);
    assert_eq!(activity.pool_returns, 2);
    assert_eq!(activity.cleared_bytes, 2 * value_bytes);
    assert_eq!(activity.host_write_bytes, value_bytes);
    assert_eq!(activity.host_read_bytes, value_bytes);
    assert_eq!(activity.materializations, 1);
    assert_eq!(activity.whole_polynomial_materializations, 1);
    assert_eq!(activity.blit_bytes, 0);
    assert_eq!(activity.execution_failures, 0);
}

#[test]
fn concurrent_callers_share_one_validated_domain_plan() {
    const WORKERS: usize = 4;
    let session = Arc::new(MetalSession::new().expect("initialize Metal runtime"));
    let barrier = Arc::new(Barrier::new(WORKERS));
    let domain = domain(9);
    session.reset_activity();

    let mut workers = Vec::new();
    for worker in 0..WORKERS {
        let session = Arc::clone(&session);
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let values = deterministic_values(1 << domain.log_size, 0x7170_0000 + worker as u64);
            barrier.wait();
            let restored = session
                .probe_bls12_381_ntt_roundtrip(&values, domain)
                .expect("concurrent resident NTT roundtrip");
            assert_fr_slices("concurrent resident NTT", &values, &restored);
        }));
    }
    for worker in workers {
        worker.join().expect("NTT worker did not panic");
    }

    let activity = session.activity();
    assert_eq!(activity.allocations, 6); // Four immutable plans plus two secret slots.
    assert_eq!(activity.pool_misses, 2);
    assert_eq!(activity.pool_hits, 2 * (WORKERS as u64 - 1));
    assert_eq!(activity.pool_returns, 2 * WORKERS as u64);
    assert_eq!(activity.command_buffers, 2 * WORKERS as u64);
    assert_eq!(activity.clear_commands, WORKERS as u64);
    assert_eq!(
        activity.command_buffers - activity.clear_commands,
        WORKERS as u64
    );
    assert_eq!(activity.dispatches, 4 * WORKERS as u64);
    assert_eq!(activity.clear_dispatches, 2 * WORKERS as u64);
    assert_eq!(
        activity.dispatches - activity.clear_dispatches,
        2 * WORKERS as u64
    ); // One radix-512 pass in each direction.
    assert_eq!(activity.host_waits, 2 * WORKERS as u64);
    assert_eq!(activity.clear_waits, WORKERS as u64);
    assert_eq!(activity.host_waits - activity.clear_waits, WORKERS as u64);
    assert_eq!(activity.execution_failures, 0);
}

#[test]
fn invalid_domains_and_batch_shapes_fail_before_submission() {
    let session = MetalSession::new().expect("initialize Metal runtime");
    let valid = domain(4);
    let mut invalid = valid;
    invalid.root = ONE;
    session.reset_activity();
    let error = session
        .probe_bls12_381_ntt(&[], invalid, NttDirection::Forward)
        .expect_err("nonprimitive root must fail even for an empty input");
    assert!(matches!(error, MetalError::NttDomain(_)));

    let error = session
        .probe_bls12_381_ntt(&[ONE; 15], valid, NttDirection::Forward)
        .expect_err("partial NTT batch must fail");
    assert!(matches!(
        error,
        MetalError::NttBatch {
            elements: 15,
            batch_len: 16
        }
    ));

    let error = session
        .probe_bls12_381_coeff_to_extended(&[ONE; 6], 3, valid)
        .expect_err("non-power-of-two coefficient batches must fail");
    assert!(matches!(error, MetalError::NttCoefficientBatch { .. }));
    assert_eq!(session.activity().command_buffers, 0);
}

fn domain(log_size: u32) -> NttDomainSpec {
    assert!(log_size <= 32);
    let mut root = TWO_ADIC_ROOT;
    let mut inverse_root = TWO_ADIC_ROOT_INV;
    for _ in log_size..32 {
        root = blst_square(root);
        inverse_root = blst_square(inverse_root);
    }
    let mut inverse_size = ONE;
    for _ in 0..log_size {
        inverse_size = blst_mul(inverse_size, TWO_INV);
    }
    NttDomainSpec {
        log_size,
        root,
        inverse_root,
        inverse_size,
        coset_generator: ZETA,
        inverse_coset_generator: blst_square(ZETA),
    }
}

fn deterministic_values(len: usize, seed: u64) -> Vec<Bls12_381Fr> {
    let mut state = seed;
    (0..len)
        .map(|index| {
            let random = splitmix64(&mut state);
            fr_from_u64(random ^ (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93))
        })
        .collect()
}

fn cpu_ntt_batches(
    values: &mut [Bls12_381Fr],
    batch_len: usize,
    root: Bls12_381Fr,
    final_scale: Option<Bls12_381Fr>,
) {
    assert!(batch_len.is_power_of_two());
    assert!(values.len().is_multiple_of(batch_len));
    for batch in values.chunks_exact_mut(batch_len) {
        cpu_ntt(batch, root, final_scale);
    }
}

fn cpu_ntt(values: &mut [Bls12_381Fr], root: Bls12_381Fr, scale: Option<Bls12_381Fr>) {
    let n = values.len();
    assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    for index in 0..n {
        let reversed = if bits == 0 {
            index
        } else {
            index.reverse_bits() >> (usize::BITS - bits)
        };
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut width = 2;
    while width <= n {
        let half = width / 2;
        let width_root = blst_pow(root, (n / width) as u64);
        for chunk in values.chunks_exact_mut(width) {
            let mut power = ONE;
            for index in 0..half {
                let even = chunk[index];
                let odd = blst_mul(chunk[index + half], power);
                chunk[index] = blst_add(even, odd);
                chunk[index + half] = blst_sub(even, odd);
                power = blst_mul(power, width_root);
            }
        }
        width *= 2;
    }
    if let Some(scale) = scale {
        for value in values {
            *value = blst_mul(*value, scale);
        }
    }
}

fn cpu_coeff_to_extended(
    coefficients: &[Bls12_381Fr],
    coefficient_len: usize,
    domain: NttDomainSpec,
    batches: usize,
) -> Vec<Bls12_381Fr> {
    let extended_len = 1usize << domain.log_size;
    let mut output = Vec::with_capacity(extended_len * batches);
    for batch in 0..batches {
        let mut extended = vec![Bls12_381Fr::default(); extended_len];
        extended[..coefficient_len]
            .copy_from_slice(&coefficients[batch * coefficient_len..(batch + 1) * coefficient_len]);
        distribute_coset(&mut extended[..coefficient_len], domain, true);
        cpu_ntt(&mut extended, domain.root, None);
        output.extend(extended);
    }
    output
}

fn cpu_extended_to_coeff(
    evaluations: &[Bls12_381Fr],
    domain: NttDomainSpec,
    batches: usize,
) -> Vec<Bls12_381Fr> {
    let extended_len = 1usize << domain.log_size;
    let mut output = evaluations.to_vec();
    assert_eq!(output.len(), extended_len * batches);
    for batch in output.chunks_exact_mut(extended_len) {
        cpu_ntt(batch, domain.inverse_root, Some(domain.inverse_size));
        distribute_coset(batch, domain, false);
    }
    output
}

fn distribute_coset(values: &mut [Bls12_381Fr], domain: NttDomainSpec, ingress: bool) {
    let (first, second) = if ingress {
        (domain.coset_generator, domain.inverse_coset_generator)
    } else {
        (domain.inverse_coset_generator, domain.coset_generator)
    };
    for (index, value) in values.iter_mut().enumerate() {
        match index % 3 {
            1 => *value = blst_mul(*value, first),
            2 => *value = blst_mul(*value, second),
            _ => {}
        }
    }
}

fn fr_from_u64(value: u64) -> Bls12_381Fr {
    let canonical = [value, 0, 0, 0];
    let mut raw = blst::blst_fr::default();
    // SAFETY: `raw` is valid output and `canonical` supplies the four expected
    // little-endian limbs for the duration of the BLST call.
    unsafe {
        blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr());
    }
    Bls12_381Fr { limbs: raw.l }
}

fn blst_add(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    blst_binary(lhs, rhs, blst::blst_fr_add)
}

fn blst_sub(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    blst_binary(lhs, rhs, blst::blst_fr_sub)
}

fn blst_mul(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    blst_binary(lhs, rhs, blst::blst_fr_mul)
}

fn blst_binary(
    lhs: Bls12_381Fr,
    rhs: Bls12_381Fr,
    operation: unsafe extern "C" fn(*mut blst::blst_fr, *const blst::blst_fr, *const blst::blst_fr),
) -> Bls12_381Fr {
    let lhs = blst::blst_fr { l: lhs.limbs };
    let rhs = blst::blst_fr { l: rhs.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: the operation receives initialized, non-overlapping BLST field
    // objects that live for the complete call.
    unsafe {
        operation(&mut output, &lhs, &rhs);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_square(value: Bls12_381Fr) -> Bls12_381Fr {
    let value = blst::blst_fr { l: value.limbs };
    let mut output = blst::blst_fr::default();
    // SAFETY: the initialized input and output are non-overlapping and live for
    // the duration of the BLST call.
    unsafe {
        blst::blst_fr_sqr(&mut output, &value);
    }
    Bls12_381Fr { limbs: output.l }
}

fn blst_pow(mut base: Bls12_381Fr, mut exponent: u64) -> Bls12_381Fr {
    let mut result = ONE;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = blst_mul(result, base);
        }
        base = blst_square(base);
        exponent >>= 1;
    }
    result
}

fn assert_fr_slices(label: &str, expected: &[Bls12_381Fr], actual: &[Bls12_381Fr]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    if let Some(index) = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
    {
        panic!(
            "{label}: mismatch at {index}: expected {:?}, actual {:?}",
            expected[index], actual[index]
        );
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
