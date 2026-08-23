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

//! Sensitive authentic-shape decision gate for the final segmented falsifier.

use std::{path::PathBuf, time::Duration};

use super::{
    MetalSession,
    batch_affine_authentic::{median, milliseconds, p95, synthetic_srs},
    batch_affine_benchmark::{BatchAffineBenchmarkChunk, synthetic_packed_records},
    batch_affine_manifest::load_authentic_replay_manifest,
    flat_segmented_benchmark::{
        FlatSegmentedBenchmarkCounters, FlatSegmentedBenchmarkSample,
        PreparedFlatSegmentedBenchmark,
    },
    flat_segmented_plan::FlatSegmentedWidth,
};

const DEFAULT_MANIFEST: &str = "/tmp/metal-msm-replay-seed24302.json";
const SRS_LEN: usize = 32_768;
const WARMUPS: usize = 2;
const SAMPLES: usize = 20;
const AUTHENTIC_TOTAL_PAIRS: u64 = 46_039_066;
const PROMOTION_RATE: f64 = 153_500_000.0;
const PROMOTION_MS: f64 = 300.0;

#[test]
#[ignore = "release-mode sensitive authentic-shape flat-segmented decision benchmark"]
fn authentic_four_chunk_flat_segmented_decision_gate() {
    if std::hint::black_box(cfg!(debug_assertions)) {
        panic!("the authentic flat-segmented decision gate must run with --release");
    }
    assert_eq!(
        std::env::var("MIDNIGHT_METAL_SENSITIVE_DIAGNOSTIC").as_deref(),
        Ok("1"),
        "set MIDNIGHT_METAL_SENSITIVE_DIAGNOSTIC=1 to acknowledge local sensitive diagnostics"
    );

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
    let preparation_128_started = std::time::Instant::now();
    let prepared_128 = session
        .prepare_flat_segmented_benchmark(&srs, &descriptions, FlatSegmentedWidth::G128)
        .expect("prepare G128 flat-segmented workload");
    let preparation_128 = preparation_128_started.elapsed();
    let counters_128 = prepared_128.counters();
    assert_preparation_activity(&session, counters_128, &workload);

    session.reset_activity();
    let preparation_256_started = std::time::Instant::now();
    let prepared_256 = session
        .prepare_flat_segmented_benchmark(&srs, &descriptions, FlatSegmentedWidth::G256)
        .expect("prepare G256 flat-segmented workload");
    let preparation_256 = preparation_256_started.elapsed();
    let counters_256 = prepared_256.counters();
    assert_preparation_activity(&session, counters_256, &workload);

    for _ in 0..WARMUPS {
        assert_zero_sample(
            prepared_128.execute().expect("execute warm G128 sample"),
            counters_128,
        );
        assert_zero_sample(
            prepared_256.execute().expect("execute warm G256 sample"),
            counters_256,
        );
    }

    session.reset_activity();
    let mut samples_128 = Vec::with_capacity(SAMPLES);
    let mut samples_256 = Vec::with_capacity(SAMPLES);
    for sample_index in 0..SAMPLES {
        if sample_index.is_multiple_of(2) {
            samples_128.push(measure(&prepared_128, counters_128));
            samples_256.push(measure(&prepared_256, counters_256));
        } else {
            samples_256.push(measure(&prepared_256, counters_256));
            samples_128.push(measure(&prepared_128, counters_128));
        }
    }

    let activity = session.activity();
    assert_eq!(activity.command_buffers, 2 * SAMPLES as u64);
    assert_eq!(
        activity.dispatches,
        (counters_128.dispatch_count + counters_256.dispatch_count) * SAMPLES as u64
    );
    assert_eq!(activity.host_waits, 2 * SAMPLES as u64);
    assert_eq!(activity.allocations, 0);
    assert_eq!(activity.allocated_bytes, 0);
    assert_eq!(activity.host_write_bytes, 0);
    assert_eq!(
        activity.host_read_bytes,
        (counters_128.host_status_read_bytes + counters_256.host_status_read_bytes)
            * SAMPLES as u64
    );
    assert_eq!(activity.blit_bytes, 0);
    assert_eq!(activity.execution_failures, 0);

    report(
        &device.name,
        preparation_128,
        counters_128,
        &samples_128,
        activity.current_allocated_bytes,
        device.recommended_working_set_bytes,
    );
    report(
        &device.name,
        preparation_256,
        counters_256,
        &samples_256,
        activity.current_allocated_bytes,
        device.recommended_working_set_bytes,
    );
}

fn assert_preparation_activity(
    session: &MetalSession,
    counters: FlatSegmentedBenchmarkCounters,
    workload: &super::batch_affine_manifest::AuthenticReplayWorkload,
) {
    assert_eq!(counters.chunk_count, 4);
    assert_eq!(counters.leaf_count, workload.total_records);
    assert_eq!(counters.root_count, workload.total_runs);
    assert_eq!(counters.pair_reduction_count, workload.total_pairs);
    let activity = session.activity();
    assert_eq!(activity.command_buffers, 0);
    assert_eq!(activity.dispatches, 0);
    assert_eq!(activity.host_waits, 0);
    assert_eq!(activity.allocations, counters.metal_allocation_count);
    assert_eq!(activity.allocated_bytes, counters.allocated_bytes);
    assert_eq!(activity.host_write_bytes, counters.host_upload_bytes);
    assert_eq!(activity.host_read_bytes, 0);
    assert_eq!(activity.blit_bytes, 0);
    assert_eq!(activity.execution_failures, 0);
}

fn measure(
    prepared: &PreparedFlatSegmentedBenchmark<'_>,
    counters: FlatSegmentedBenchmarkCounters,
) -> FlatSegmentedBenchmarkSample {
    let sample = prepared
        .execute()
        .expect("execute timed flat-segmented sample");
    assert_zero_sample(sample, counters);
    sample
}

fn assert_zero_sample(
    sample: FlatSegmentedBenchmarkSample,
    counters: FlatSegmentedBenchmarkCounters,
) {
    assert_eq!(sample.status, Default::default());
    assert_eq!(sample.counters, counters);
    assert!(
        sample.total >= sample.encoding + sample.submit_and_wait + sample.status_read,
        "total timing must contain encoding, completion, and status read"
    );
}

fn report(
    device_name: &str,
    preparation: Duration,
    counters: FlatSegmentedBenchmarkCounters,
    samples: &[FlatSegmentedBenchmarkSample],
    current_allocated_bytes: u64,
    recommended_working_set_bytes: u64,
) {
    let totals = samples
        .iter()
        .map(|sample| sample.total)
        .collect::<Vec<_>>();
    let encodings = samples
        .iter()
        .map(|sample| sample.encoding)
        .collect::<Vec<_>>();
    let waits = samples
        .iter()
        .map(|sample| sample.submit_and_wait)
        .collect::<Vec<_>>();
    let statuses = samples
        .iter()
        .map(|sample| sample.status_read)
        .collect::<Vec<_>>();
    let gpu = samples
        .iter()
        .filter_map(|sample| sample.gpu)
        .collect::<Vec<_>>();
    let total_median = median(&totals);
    let rate = counters.pair_reduction_count as f64 / total_median.as_secs_f64();
    let scheduled_rate = counters.scheduled_lane_round_count as f64 / total_median.as_secs_f64();
    let projected_ms = AUTHENTIC_TOTAL_PAIRS as f64 / rate * 1_000.0;
    let decision = if rate >= PROMOTION_RATE && projected_ms <= PROMOTION_MS {
        "promote"
    } else {
        "reject"
    };
    let gpu_median = (gpu.len() == samples.len()).then(|| median(&gpu));

    eprintln!(
        "SENSITIVE_DIAGNOSTIC metal_flat_segmented_decision device={:?} decision={} width={} \
         chunks={} useful_pairs={} authentic_pairs={} warmups={} samples={} preparation_ms={:.3} \
         median_ms={:.3} p95_ms={:.3} useful_pairs_per_second={:.0} \
         scheduled_lane_rounds_per_second={:.0} projected_authentic_ms={:.3} \
         promotion_sample_ms={:.3} encoding_median_ms={:.3} wait_median_ms={:.3} \
         status_median_ms={:.6} gpu_median_ms={} passes={} groups={} real_tiles={} padded_tiles={} \
         inversions={} scheduled_lane_rounds={} dispatches_per_sample={} allocations={} \
         execution_width={} pipeline_max_threads={} dynamic_threadgroup_bytes={} \
         static_threadgroup_bytes={} logical_device_bytes={} allocated_bytes={} \
         immutable_bytes={} scratch_bytes={} root_bytes={} host_upload_bytes={} \
         host_read_bytes_per_sample={} current_allocated_bytes={} recommended_working_set_bytes={} \
         raw_total_ms={:?}",
        device_name,
        decision,
        counters.group_width,
        counters.chunk_count,
        counters.pair_reduction_count,
        AUTHENTIC_TOTAL_PAIRS,
        WARMUPS,
        samples.len(),
        milliseconds(preparation),
        milliseconds(total_median),
        milliseconds(p95(&totals)),
        rate,
        scheduled_rate,
        projected_ms,
        counters.pair_reduction_count as f64 / PROMOTION_RATE * 1_000.0,
        milliseconds(median(&encodings)),
        milliseconds(median(&waits)),
        milliseconds(median(&statuses)),
        gpu_median
            .map(|duration| format!("{:.3}", milliseconds(duration)))
            .unwrap_or_else(|| "unavailable".to_owned()),
        counters.pass_count,
        counters.group_count,
        counters.real_tile_count,
        counters.padded_tile_count,
        counters.batch_inversion_count,
        counters.scheduled_lane_round_count,
        counters.dispatch_count,
        counters.metal_allocation_count,
        counters.pipeline_execution_width,
        counters.pipeline_max_threads,
        counters.dynamic_threadgroup_bytes,
        counters.static_threadgroup_bytes,
        counters.logical_device_bytes,
        counters.allocated_bytes,
        counters.immutable_chunk_bytes,
        counters.shared_scratch_bytes,
        counters.root_output_bytes,
        counters.host_upload_bytes,
        counters.host_status_read_bytes,
        current_allocated_bytes,
        recommended_working_set_bytes,
        totals.iter().copied().map(milliseconds).collect::<Vec<_>>(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_sample_time_is_derived_from_the_useful_workload() {
        let useful_pairs = 6_794_123f64;
        let required_ms = useful_pairs / PROMOTION_RATE * 1_000.0;
        assert!((required_ms - 44.261_387_6).abs() < 0.000_001);
    }
}
