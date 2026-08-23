// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

//! Optimistic arithmetic ceiling for the last global batch-affine layout.
//!
//! The kernel deliberately removes product-tree storage, inversion, routing,
//! barriers, integration, and erasure. Each register-local pair iteration still
//! executes the exact `7 M/S + 9 A/S` field-operation population implied by a
//! globally batched affine reduction. Missing the promotion rate here rejects
//! every layout with the same arithmetic; passing is necessary, not sufficient.

use std::{
    ffi::c_void,
    mem::size_of,
    ptr::NonNull,
    time::{Duration, Instant},
};

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

use super::batch_affine_benchmark::completed_gpu_duration;
use super::{BatchAffineCeilingParameters, MetalSession, PendingCommand, TrackedBuffer};
use crate::{Bls12_381Fp, Bls12_381G1Affine, MetalError};

const PROMOTION_RATE: f64 = 153_500_000.0;
const SCAN_RESEARCH_RATE: f64 = 200_000_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatchAffineCeilingSample {
    total: Duration,
    gpu: Option<Duration>,
    pair_iterations: u64,
    first_point: Bls12_381G1Affine,
    first_scan: Bls12_381Fp,
}

struct PreparedBatchAffineCeiling<'session> {
    session: &'session MetalSession,
    lhs: TrackedBuffer,
    rhs: TrackedBuffer,
    points_a: TrackedBuffer,
    points_b: TrackedBuffer,
    scan: TrackedBuffer,
    element_count: u32,
}

impl MetalSession {
    fn prepare_batch_affine_ceiling<'session>(
        &'session self,
        lhs: &[Bls12_381G1Affine],
        rhs: &[Bls12_381G1Affine],
    ) -> Result<PreparedBatchAffineCeiling<'session>, MetalError> {
        if lhs.is_empty() {
            return Err(MetalError::Dispatch(
                "batch-affine ceiling requires at least one pair",
            ));
        }
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        let element_count =
            u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let point_bytes = lhs
            .len()
            .checked_mul(size_of::<Bls12_381G1Affine>())
            .ok_or(MetalError::LengthOverflow(lhs.len()))?;
        let scan_bytes = lhs
            .len()
            .checked_mul(size_of::<Bls12_381Fp>())
            .ok_or(MetalError::LengthOverflow(lhs.len()))?;

        let lhs = self.buffer_from_slice(lhs, "midnight.batch_affine.ceiling.lhs")?;
        let rhs = self.buffer_from_slice(rhs, "midnight.batch_affine.ceiling.rhs")?;
        let points_a = self.buffer(point_bytes, "midnight.batch_affine.ceiling.points_a")?;
        let points_b = self.buffer(point_bytes, "midnight.batch_affine.ceiling.points_b")?;
        let scan = self.buffer(scan_bytes, "midnight.batch_affine.ceiling.scan")?;
        Ok(PreparedBatchAffineCeiling {
            session: self,
            lhs,
            rhs,
            points_a,
            points_b,
            scan,
            element_count,
        })
    }
}

impl PreparedBatchAffineCeiling<'_> {
    fn legal_threadgroup_widths(&self) -> Vec<usize> {
        let pipeline = &self.session.batch_affine_ceiling.execute;
        let execution_width = pipeline.threadExecutionWidth();
        let maximum_width = pipeline.maxTotalThreadsPerThreadgroup();
        (execution_width..=maximum_width)
            .step_by(execution_width)
            .collect()
    }

    fn execute(
        &self,
        iterations: u32,
        timed_dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<BatchAffineCeilingSample, MetalError> {
        if iterations == 0 || timed_dispatches == 0 {
            return Err(MetalError::Dispatch(
                "batch-affine ceiling iterations and dispatches must be nonzero",
            ));
        }
        let pipeline = &self.session.batch_affine_ceiling.execute;
        let execution_width = pipeline.threadExecutionWidth();
        if threadgroup_width == 0
            || threadgroup_width > pipeline.maxTotalThreadsPerThreadgroup()
            || !threadgroup_width.is_multiple_of(execution_width)
        {
            return Err(MetalError::Dispatch(
                "batch-affine ceiling threadgroup width is unsupported",
            ));
        }
        let pair_iterations = u64::from(self.element_count)
            .checked_mul(u64::from(iterations))
            .and_then(|value| value.checked_mul(u64::from(timed_dispatches)))
            .ok_or(MetalError::Dispatch(
                "batch-affine ceiling operation count overflow",
            ))?;
        let parameters = BatchAffineCeilingParameters {
            element_count: self.element_count,
            iterations,
            reserved_0: 0,
            reserved_1: 0,
        };

        let _execution = self.session.execution_guard();
        let total_start = Instant::now();
        let command = self
            .session
            .command_buffer("midnight.batch_affine.arithmetic_ceiling")?;
        if !command.retainedReferences() {
            return Err(MetalError::Dispatch(
                "batch-affine ceiling requires retained command references",
            ));
        }
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(pipeline);

        for dispatch_index in 0..timed_dispatches {
            let source = if dispatch_index == 0 {
                &self.lhs
            } else if dispatch_index.is_multiple_of(2) {
                &self.points_b
            } else {
                &self.points_a
            };
            let destination = if dispatch_index.is_multiple_of(2) {
                &self.points_a
            } else {
                &self.points_b
            };
            // SAFETY: preparation allocated every buffer for exactly
            // `element_count` values; all buffers remain retained through the
            // serial encoder and completion wait.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.rhs.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.scan.raw), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&parameters).cast::<c_void>(),
                    size_of::<BatchAffineCeilingParameters>(),
                    4,
                );
            }
            let groups = (self.element_count as usize).div_ceil(threadgroup_width);
            self.session.dispatch_threadgroups(
                &encoder,
                pipeline,
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_width,
                    height: 1,
                    depth: 1,
                },
            )?;
        }
        encoder.endEncoding();

        let timing_command = command.clone();
        self.session.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                self.lhs.raw.clone(),
                self.rhs.raw.clone(),
                self.points_a.raw.clone(),
                self.points_b.raw.clone(),
                self.scan.raw.clone(),
            ],
        })?;
        let total = total_start.elapsed();
        let gpu = completed_gpu_duration(&timing_command);
        let final_points = if timed_dispatches.is_multiple_of(2) {
            &self.points_b
        } else {
            &self.points_a
        };
        let first_point = self.session.read_values(final_points, 1)?[0];
        let first_scan = self.session.read_values(&self.scan, 1)?[0];
        Ok(BatchAffineCeilingSample {
            total,
            gpu,
            pair_iterations,
            first_point,
            first_scan,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::batch_affine_authentic::{median, p95, synthetic_srs};

    const DEFAULT_ELEMENTS: usize = 1 << 18;
    const DEFAULT_ITERATIONS: u32 = 16;
    const DEFAULT_DISPATCHES: u32 = 4;

    #[test]
    fn invalid_inputs_are_rejected_before_metal_allocations() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        assert!(session.prepare_batch_affine_ceiling(&[], &[]).is_err());
        assert!(
            session
                .prepare_batch_affine_ceiling(&[Bls12_381G1Affine::default()], &[])
                .is_err()
        );
        assert_eq!(session.activity().allocations, 0);
    }

    #[test]
    fn ceiling_probe_is_deterministic_and_has_an_exact_activity_boundary() {
        let points = synthetic_srs(64);
        let session = MetalSession::new().expect("initialize Metal runtime");
        let prepared = session
            .prepare_batch_affine_ceiling(&points[..32], &points[32..])
            .expect("prepare batch-affine ceiling");
        let width = prepared.legal_threadgroup_widths()[0];
        let first = prepared
            .execute(2, 2, width)
            .expect("execute ceiling probe");

        prepared.session.reset_activity();
        let second = prepared.execute(2, 2, width).expect("repeat ceiling probe");
        assert_eq!(first.first_point, second.first_point);
        assert_eq!(first.first_scan, second.first_scan);
        assert_ne!(second.first_scan, Bls12_381Fp::default());
        assert_eq!(second.pair_iterations, 128);

        let activity = prepared.session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.dispatches, 2);
        assert_eq!(activity.host_waits, 1);
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.allocated_bytes, 0);
        assert_eq!(activity.host_write_bytes, 0);
        assert_eq!(activity.host_read_bytes, 144);
        assert_eq!(activity.blit_bytes, 0);
        assert_eq!(activity.execution_failures, 0);
    }

    #[test]
    #[ignore = "release-mode optimistic batch-affine arithmetic ceiling"]
    fn report_batch_affine_arithmetic_ceiling() {
        if cfg!(debug_assertions) {
            panic!(
                "run with `cargo test --release -p midnight-metal-prover --lib \
                 metal::batch_affine_ceiling::tests::report_batch_affine_arithmetic_ceiling \
                 -- --ignored --exact --nocapture --test-threads=1`"
            );
        }
        let elements =
            environment_usize("MIDNIGHT_METAL_AFFINE_CEILING_ELEMENTS", DEFAULT_ELEMENTS);
        let iterations = environment_u32(
            "MIDNIGHT_METAL_AFFINE_CEILING_ITERATIONS",
            DEFAULT_ITERATIONS,
        );
        let timed_dispatches = environment_u32(
            "MIDNIGHT_METAL_AFFINE_CEILING_DISPATCHES",
            DEFAULT_DISPATCHES,
        );
        let samples = environment_usize("MIDNIGHT_METAL_AFFINE_CEILING_SAMPLES", 1);
        assert!(elements != 0 && iterations != 0 && timed_dispatches != 0 && samples != 0);

        let points = synthetic_srs(
            elements
                .checked_mul(2)
                .expect("ceiling point population fits usize"),
        );
        let session = MetalSession::new().expect("initialize Metal runtime");
        let prepared = session
            .prepare_batch_affine_ceiling(&points[..elements], &points[elements..])
            .expect("prepare batch-affine ceiling");
        let device = prepared
            .session
            .device_info()
            .expect("query Metal device information");
        let legal_widths = prepared.legal_threadgroup_widths();
        let widths = if let Some(width) =
            environment_optional_usize("MIDNIGHT_METAL_AFFINE_CEILING_THREADGROUP_WIDTH")
        {
            assert!(
                legal_widths.contains(&width),
                "requested ceiling threadgroup width is unsupported"
            );
            vec![width]
        } else {
            legal_widths
        };
        let mut best = (0.0f64, 0usize, 0.0f64);

        for width in widths {
            for _ in 0..2 {
                prepared
                    .execute(2, 1, width)
                    .expect("warm batch-affine ceiling pipeline");
            }
            let mut total_samples = Vec::with_capacity(samples);
            let mut gpu_samples = Vec::with_capacity(samples);
            let mut pair_iterations = 0u64;
            for sample_index in 0..samples {
                prepared.session.reset_activity();
                let sample = prepared
                    .execute(iterations, timed_dispatches, width)
                    .expect("measure batch-affine arithmetic ceiling");
                pair_iterations = sample.pair_iterations;
                total_samples.push(sample.total);
                if let Some(gpu) = sample.gpu {
                    gpu_samples.push(gpu);
                }
                let activity = prepared.session.activity();
                assert_eq!(activity.command_buffers, 1);
                assert_eq!(activity.dispatches, u64::from(timed_dispatches));
                assert_eq!(activity.host_waits, 1);
                assert_eq!(activity.allocations, 0);
                assert_eq!(activity.host_write_bytes, 0);
                assert_eq!(activity.host_read_bytes, 144);
                assert_eq!(activity.blit_bytes, 0);
                assert_eq!(activity.execution_failures, 0);
                println!(
                    "metal_batch_affine_arithmetic_ceiling_sample threadgroup_width={width} \
                     sample={sample_index} gpu_ms={} wall_ms={:.3}",
                    sample
                        .gpu
                        .map(|duration| format!("{:.3}", duration.as_secs_f64() * 1_000.0))
                        .unwrap_or_else(|| "unavailable".to_owned()),
                    sample.total.as_secs_f64() * 1_000.0,
                );
            }
            let gpu_or_total = if gpu_samples.len() == total_samples.len() {
                &gpu_samples
            } else {
                &total_samples
            };
            let gpu_median = median(gpu_or_total);
            let gpu_p95 = p95(gpu_or_total);
            let wall_median = median(&total_samples);
            let wall_p95 = p95(&total_samples);
            let gpu_rate = pair_iterations as f64 / gpu_median.as_secs_f64();
            let wall_rate = pair_iterations as f64 / wall_median.as_secs_f64();
            if gpu_rate > best.0 {
                best = (gpu_rate, width, wall_rate);
            }
            println!(
                "metal_batch_affine_arithmetic_ceiling device={:?} elements={elements} \
                 iterations={iterations} timed_dispatches={timed_dispatches} samples={samples} \
                 threadgroup_width={width} pair_iterations={pair_iterations} \
                 gpu_pairs_per_second={gpu_rate:.0} wall_pairs_per_second={wall_rate:.0} \
                 gpu_median_ms={:.3} gpu_p95_ms={:.3} wall_median_ms={:.3} wall_p95_ms={:.3}",
                device.name,
                gpu_median.as_secs_f64() * 1_000.0,
                gpu_p95.as_secs_f64() * 1_000.0,
                wall_median.as_secs_f64() * 1_000.0,
                wall_p95.as_secs_f64() * 1_000.0,
            );
        }

        let decision = if best.0 < PROMOTION_RATE {
            "REJECT_ALL_AFFINE_SCANS"
        } else if best.0 < SCAN_RESEARCH_RATE {
            "REJECT_INSUFFICIENT_HEADROOM"
        } else {
            "HIERARCHICAL_SCAN_FALSIFIER_ELIGIBLE"
        };
        println!(
            "metal_batch_affine_arithmetic_ceiling_gate promotion_pairs_per_second={PROMOTION_RATE:.0} \
             scan_research_pairs_per_second={SCAN_RESEARCH_RATE:.0} best_gpu_pairs_per_second={:.0} \
             best_wall_pairs_per_second={:.0} best_threadgroup_width={} ratio_to_promotion={:.4} \
             decision={decision}",
            best.0,
            best.2,
            best.1,
            best.0 / PROMOTION_RATE,
        );
    }

    fn environment_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|value| value.parse().expect("environment value must be usize"))
            .unwrap_or(default)
    }

    fn environment_u32(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .map(|value| value.parse().expect("environment value must be u32"))
            .unwrap_or(default)
    }

    fn environment_optional_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .map(|value| value.parse().expect("environment value must be usize"))
    }

    const _: () = {
        assert!(size_of::<Bls12_381G1Affine>() + size_of::<Bls12_381Fp>() == 144);
    };
}
