// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

//! Best-case host-round-trip NTT boundary for the profiled proof workload.
//!
//! Calls with the same domain size are packed into one Metal batch. This is
//! more favorable than the current leaf call graph, but unlike the resident
//! kernel-rate probe it includes allocation, upload, submission, completion,
//! and materialization. It decides whether an NTT-only bridge merits wiring;
//! it is not a full-proof speedup claim.

use std::time::{Duration, Instant};

use super::{MetalSession, NttDirection, bls12_381_ntt_domain};
use crate::Bls12_381Fr;

const PROFILED_SHAPES: [(u32, usize); 5] = [(13, 17), (14, 50), (15, 68), (16, 52), (17, 52)];

#[test]
#[ignore = "release-mode best-case NTT leaf boundary"]
fn report_profiled_ntt_leaf_boundary() {
    if cfg!(debug_assertions) {
        panic!(
            "run with `cargo test --release -p midnight-metal-prover --lib \
             metal::ntt_leaf_ceiling::report_profiled_ntt_leaf_boundary \
             -- --ignored --exact --nocapture --test-threads=1`"
        );
    }

    let samples = environment_usize("MIDNIGHT_METAL_NTT_LEAF_SAMPLES", 5);
    assert!(samples != 0, "NTT leaf sample count must be nonzero");

    let inputs = PROFILED_SHAPES
        .iter()
        .map(|&(log_size, batches)| {
            let domain_len = 1usize << log_size;
            let mut values = vec![Bls12_381Fr::default(); domain_len * batches];
            for (index, value) in values.iter_mut().enumerate() {
                value.limbs = if index.is_multiple_of(2) {
                    super::generated::FR_R
                } else {
                    super::generated::FR_R2
                };
            }
            (log_size, batches, values)
        })
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    run_roundtrip_population(&session, &inputs).expect("warm profiled NTT population");

    let mut forward_samples = Vec::with_capacity(samples);
    let mut inverse_samples = Vec::with_capacity(samples);
    let mut total_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        session.reset_activity();
        let sample =
            run_roundtrip_population(&session, &inputs).expect("measure profiled NTT population");
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 10);
        assert_eq!(activity.dispatches, 22);
        assert_eq!(activity.host_waits, 10);
        assert_eq!(activity.host_write_bytes, 858_259_456);
        assert_eq!(activity.host_read_bytes, 858_259_456);
        assert_eq!(activity.blit_bytes, 0);
        assert_eq!(activity.execution_failures, 0);

        forward_samples.push(sample.forward);
        inverse_samples.push(sample.inverse);
        total_samples.push(sample.total);
        println!(
            "metal_profiled_ntt_leaf_sample sample={sample_index} forward_ms={:.3} \
             inverse_ms={:.3} roundtrip_ms={:.3} allocations={} allocated_bytes={}",
            sample.forward.as_secs_f64() * 1_000.0,
            sample.inverse.as_secs_f64() * 1_000.0,
            sample.total.as_secs_f64() * 1_000.0,
            activity.allocations,
            activity.allocated_bytes,
        );
    }

    let forward_median = median(&forward_samples);
    let inverse_median = median(&inverse_samples);
    let total_median = median(&total_samples);
    let total_p95 = p95(&total_samples);
    println!(
        "metal_profiled_ntt_leaf_boundary samples={samples} transformed_elements=13410304 \
         forward_median_ms={:.3} inverse_median_ms={:.3} roundtrip_median_ms={:.3} \
         roundtrip_p95_ms={:.3} one_population_average_ms={:.3}",
        forward_median.as_secs_f64() * 1_000.0,
        inverse_median.as_secs_f64() * 1_000.0,
        total_median.as_secs_f64() * 1_000.0,
        total_p95.as_secs_f64() * 1_000.0,
        total_median.as_secs_f64() * 500.0,
    );
}

#[derive(Clone, Copy)]
struct PopulationSample {
    forward: Duration,
    inverse: Duration,
    total: Duration,
}

fn run_roundtrip_population(
    session: &MetalSession,
    inputs: &[(u32, usize, Vec<Bls12_381Fr>)],
) -> Result<PopulationSample, crate::MetalError> {
    let total_start = Instant::now();
    let mut forward = Duration::ZERO;
    let mut inverse = Duration::ZERO;
    for (log_size, batches, input) in inputs {
        debug_assert_eq!(input.len(), (1usize << *log_size) * batches);
        let domain = bls12_381_ntt_domain(*log_size).expect("construct canonical NTT domain");
        let start = Instant::now();
        let transformed = session.probe_bls12_381_ntt(input, domain, NttDirection::Forward)?;
        forward += start.elapsed();

        let start = Instant::now();
        let roundtrip = session.probe_bls12_381_ntt(&transformed, domain, NttDirection::Inverse)?;
        inverse += start.elapsed();
        assert_eq!(&roundtrip, input, "Metal NTT roundtrip must be exact");
    }
    Ok(PopulationSample {
        forward,
        inverse,
        total: total_start.elapsed(),
    })
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2
    } else {
        sorted[midpoint]
    }
}

fn p95(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() * 95).div_ceil(100) - 1]
}

fn environment_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("environment value must be usize"))
        .unwrap_or(default)
}
