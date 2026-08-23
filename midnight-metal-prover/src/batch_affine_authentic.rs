// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Authentic-shape batch-affine decision benchmark.
//!
//! The manifest is a content-minimized but still sensitive diagnostic because
//! run occupancy is witness-derived. It stays outside the repository with mode
//! 0600. This test reconstructs only synthetic SRS indices, signs, and points.

use std::{path::PathBuf, time::Duration};

use super::{
    MetalSession,
    batch_affine_benchmark::{BatchAffineBenchmarkChunk, synthetic_packed_records},
    batch_affine_manifest::load_authentic_replay_manifest,
};
use crate::Bls12_381G1Affine;

const DEFAULT_MANIFEST: &str = "/tmp/metal-msm-replay-seed24302.json";
const SRS_LEN: usize = 32_768;
const WARMUPS: usize = 2;
const SAMPLES: usize = 20;
const AUTHENTIC_TOTAL_PAIRS: u64 = 46_039_066;
const IMMEDIATE_STOP_RATE: f64 = 126_300_000.0;
const REVIEW_RATE: f64 = 153_500_000.0;
const IMMEDIATE_STOP_MS: f64 = 364.6;
const PROMOTION_MS: f64 = 300.0;

#[test]
#[ignore = "release-mode sensitive authentic-shape Metal decision benchmark"]
fn authentic_four_chunk_batch_affine_decision_gate() {
    if std::hint::black_box(cfg!(debug_assertions)) {
        panic!("the authentic Metal decision gate must run with --release");
    }
    let manifest_path = std::env::var_os("MIDNIGHT_METAL_MSM_REPLAY_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST));
    let workload =
        load_authentic_replay_manifest(&manifest_path).expect("validate authentic replay manifest");
    assert_eq!(workload.chunks.len(), 4);

    let srs = synthetic_srs(SRS_LEN);
    let packed_records = workload
        .chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            synthetic_packed_records(
                &chunk.run_lengths,
                SRS_LEN,
                0x2430_2000_0000_0000u64 ^ index as u64,
            )
            .unwrap_or_else(|error| panic!("expand synthetic records for {}: {error}", chunk.label))
        })
        .collect::<Vec<_>>();
    let descriptions = workload
        .chunks
        .iter()
        .zip(&packed_records)
        .map(|(chunk, records)| BatchAffineBenchmarkChunk {
            run_lengths: &chunk.run_lengths,
            packed_records: records,
        })
        .collect::<Vec<_>>();

    let session = MetalSession::new().expect("initialize Metal runtime");
    let device = session
        .device_info()
        .expect("read Metal device information");
    session.reset_activity();
    let preparation_started = std::time::Instant::now();
    let prepared = session
        .prepare_batch_affine_benchmark(&srs, &descriptions)
        .expect("prepare authentic batch-affine workload");
    let preparation = preparation_started.elapsed();
    let counters = prepared.counters();
    let preparation_activity = session.activity();
    assert_eq!(counters.chunk_count, 4);
    assert_eq!(counters.leaf_count, workload.total_records);
    assert_eq!(counters.root_count, workload.total_runs);
    assert_eq!(counters.pair_reduction_count, workload.total_pairs);
    assert_eq!(preparation_activity.command_buffers, 0);
    assert_eq!(preparation_activity.dispatches, 0);
    assert_eq!(preparation_activity.host_waits, 0);
    assert_eq!(
        preparation_activity.allocations,
        counters.metal_allocation_count
    );
    assert_eq!(
        preparation_activity.allocated_bytes,
        counters.allocated_bytes
    );
    assert_eq!(
        preparation_activity.host_write_bytes,
        counters.host_upload_bytes
    );
    assert_eq!(preparation_activity.host_read_bytes, 0);
    assert_eq!(preparation_activity.blit_bytes, 0);
    assert_eq!(preparation_activity.execution_failures, 0);

    for _ in 0..WARMUPS {
        let sample = prepared
            .execute()
            .expect("execute warm batch-affine sample");
        assert_eq!(sample.status, Default::default());
        assert_eq!(sample.counters, counters);
    }

    session.reset_activity();
    let mut total_samples = Vec::with_capacity(SAMPLES);
    let mut encoding_samples = Vec::with_capacity(SAMPLES);
    let mut wait_samples = Vec::with_capacity(SAMPLES);
    let mut status_samples = Vec::with_capacity(SAMPLES);
    let mut gpu_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let sample = prepared
            .execute()
            .expect("execute timed batch-affine sample");
        assert_eq!(sample.status, Default::default());
        assert_eq!(sample.counters, counters);
        assert!(
            sample.total >= sample.encoding + sample.submit_and_wait + sample.status_read,
            "total timing must include encoding, completion, and status read"
        );
        total_samples.push(sample.total);
        encoding_samples.push(sample.encoding);
        wait_samples.push(sample.submit_and_wait);
        status_samples.push(sample.status_read);
        if let Some(gpu) = sample.gpu {
            gpu_samples.push(gpu);
        }
    }

    let activity = session.activity();
    assert_eq!(activity.command_buffers, SAMPLES as u64);
    assert_eq!(
        activity.dispatches,
        counters.dispatch_count * SAMPLES as u64
    );
    assert_eq!(activity.host_waits, SAMPLES as u64);
    assert_eq!(activity.allocations, 0);
    assert_eq!(activity.allocated_bytes, 0);
    assert_eq!(activity.host_write_bytes, 0);
    assert_eq!(
        activity.host_read_bytes,
        counters.host_status_read_bytes * SAMPLES as u64
    );
    assert_eq!(activity.blit_bytes, 0);
    assert_eq!(activity.execution_failures, 0);

    let total_median = median(&total_samples);
    let total_p95 = p95(&total_samples);
    let rate = workload.total_pairs as f64 / total_median.as_secs_f64();
    let projected_ms = AUTHENTIC_TOTAL_PAIRS as f64 / rate * 1_000.0;
    let decision = if rate < IMMEDIATE_STOP_RATE || projected_ms > IMMEDIATE_STOP_MS {
        "reject"
    } else if rate >= REVIEW_RATE && projected_ms <= PROMOTION_MS {
        "promote"
    } else {
        "review"
    };
    let gpu_median = (gpu_samples.len() == SAMPLES).then(|| median(&gpu_samples));

    eprintln!(
        "metal_batch_affine_decision device={:?} decision={} chunks={} sample_pairs={} \
         authentic_pairs={} warmups={} samples={} preparation_ms={:.3} median_ms={:.3} \
         p95_ms={:.3} pairs_per_second={:.0} projected_authentic_ms={:.3} \
         encoding_median_ms={:.3} wait_median_ms={:.3} status_median_ms={:.6} \
         gpu_median_ms={} dispatches_per_sample={} carries={} allocations={} \
         levels={} logical_device_bytes={} allocated_bytes={} immutable_bytes={} scratch_bytes={} root_bytes={} \
         host_upload_bytes={} host_read_bytes_per_sample={} current_allocated_bytes={} \
         recommended_working_set_bytes={} raw_total_ms={:?}",
        device.name,
        decision,
        counters.chunk_count,
        workload.total_pairs,
        AUTHENTIC_TOTAL_PAIRS,
        WARMUPS,
        SAMPLES,
        milliseconds(preparation),
        milliseconds(total_median),
        milliseconds(total_p95),
        rate,
        projected_ms,
        milliseconds(median(&encoding_samples)),
        milliseconds(median(&wait_samples)),
        milliseconds(median(&status_samples)),
        gpu_median
            .map(|duration| format!("{:.3}", milliseconds(duration)))
            .unwrap_or_else(|| "unavailable".to_owned()),
        counters.dispatch_count,
        counters.carry_copy_count,
        counters.metal_allocation_count,
        counters.level_count,
        counters.logical_device_bytes,
        counters.allocated_bytes,
        counters.immutable_chunk_bytes,
        counters.shared_scratch_bytes,
        counters.root_output_bytes,
        counters.host_upload_bytes,
        counters.host_status_read_bytes,
        activity.current_allocated_bytes,
        device.recommended_working_set_bytes,
        total_samples
            .iter()
            .copied()
            .map(milliseconds)
            .collect::<Vec<_>>(),
    );
}

pub(super) fn synthetic_srs(len: usize) -> Vec<Bls12_381G1Affine> {
    assert!(len != 0);
    let generator = unsafe { *blst::blst_p1_generator() };
    let mut projective = Vec::with_capacity(len);
    let mut current = generator;
    for _ in 0..len {
        projective.push(current);
        let mut next = blst::blst_p1::default();
        // SAFETY: the current point and process-lifetime generator are valid,
        // initialized G1 values, and `next` is a distinct destination.
        unsafe {
            blst::blst_p1_add_or_double(&mut next, &current, &generator);
        }
        current = next;
    }
    let pointers = projective
        .iter()
        .map(std::ptr::from_ref)
        .collect::<Vec<_>>();
    let mut affine = vec![blst::blst_p1_affine::default(); len];
    // SAFETY: every pointer addresses an initialized projective point retained
    // by `projective`, and `affine` provides `len` disjoint output slots.
    unsafe {
        blst::blst_p1s_to_affine(affine.as_mut_ptr(), pointers.as_ptr(), len);
    }
    affine
        .into_iter()
        .map(|point| Bls12_381G1Affine {
            x: point.x.l,
            y: point.y.l,
        })
        .collect()
}

pub(super) fn median(samples: &[Duration]) -> Duration {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Duration::from_secs_f64(
            (sorted[midpoint - 1].as_secs_f64() + sorted[midpoint].as_secs_f64()) / 2.0,
        )
    } else {
        sorted[midpoint]
    }
}

pub(super) fn p95(samples: &[Duration]) -> Duration {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (95 * sorted.len()).div_ceil(100);
    sorted[rank - 1]
}

pub(super) fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_statistics_use_midpoint_median_and_nearest_rank_p95() {
        let samples = (1..=20).map(Duration::from_millis).collect::<Vec<_>>();
        assert_eq!(median(&samples), Duration::from_micros(10_500));
        assert_eq!(p95(&samples), Duration::from_millis(19));
    }

    #[test]
    fn synthetic_srs_is_distinct_and_in_the_prime_subgroup() {
        let srs = synthetic_srs(33);
        assert_eq!(srs.len(), 33);
        assert!(srs.windows(2).all(|pair| pair[0] != pair[1]));
        for raw in srs {
            let point = blst::blst_p1_affine {
                x: blst::blst_fp { l: raw.x },
                y: blst::blst_fp { l: raw.y },
            };
            // SAFETY: `point` contains initialized raw Fp coordinates.
            assert!(unsafe { blst::blst_p1_affine_in_g1(&point) });
        }
    }
}
