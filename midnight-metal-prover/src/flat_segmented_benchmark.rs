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

//! Prepared benchmark for the threadgroup-local segmented affine reducer.
//!
//! This remains diagnostic-only. It validates every run, record, descriptor,
//! byte extent, and device limit before allocating Metal buffers. Timed samples
//! reuse all storage, submit one command, wait once, retain roots on device, and
//! read only the shared 16-byte status.

use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr::NonNull,
    time::{Duration, Instant},
};

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLSize,
};

use super::{
    BatchAffinePoint32, BatchAffineStatus, FlatSegmentedParameters, MetalPod, MetalSession,
    PendingCommand, TrackedBuffer, pod,
};
use crate::{Bls12_381G1Affine, MetalError};

use super::batch_affine_benchmark::{
    BatchAffineBenchmarkChunk, checked_buffer, completed_gpu_duration, read_status,
};
use super::flat_segmented_plan::{
    FlatSegmentedPlan, FlatSegmentedTile, FlatSegmentedWidth, compile_flat_segmented_plan,
    validate_flat_segmented_plan,
};

const PACKED_RECORD_INDEX_MASK: u32 = (1 << 31) - 1;
const INPUT_RECORDS: u32 = 0;
const INPUT_POINTS: u32 = 1;

impl pod::Sealed for FlatSegmentedTile {}
// SAFETY: the repr(C) descriptor contains exactly four u32 words, has no
// padding, and every bit pattern is valid on both Rust and Metal.
unsafe impl MetalPod for FlatSegmentedTile {}

/// Static accounting for one prepared width and workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FlatSegmentedBenchmarkCounters {
    pub(super) chunk_count: u64,
    pub(super) leaf_count: u64,
    pub(super) root_count: u64,
    pub(super) pair_reduction_count: u64,
    pub(super) pass_count: u64,
    pub(super) real_tile_count: u64,
    pub(super) padded_tile_count: u64,
    pub(super) group_count: u64,
    pub(super) scheduled_lane_round_count: u64,
    pub(super) batch_inversion_count: u64,
    pub(super) dispatch_count: u64,
    pub(super) command_buffer_count: u64,
    pub(super) host_wait_count: u64,
    pub(super) metal_allocation_count: u64,
    pub(super) group_width: u64,
    pub(super) pipeline_execution_width: u64,
    pub(super) pipeline_max_threads: u64,
    pub(super) dynamic_threadgroup_bytes: u64,
    pub(super) static_threadgroup_bytes: u64,
    pub(super) srs_bytes: u64,
    pub(super) immutable_chunk_bytes: u64,
    pub(super) shared_scratch_bytes: u64,
    pub(super) root_output_bytes: u64,
    pub(super) allocated_bytes: u64,
    pub(super) host_upload_bytes: u64,
    pub(super) host_status_read_bytes: u64,
    /// Nominal buffer traffic from the shader ABI, excluding cache behavior,
    /// register spills, and threadgroup-memory traffic.
    pub(super) logical_device_bytes: u64,
    pub(super) max_intermediate_points: u64,
}

/// Timing and status from one complete multi-chunk sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FlatSegmentedBenchmarkSample {
    pub(super) encoding: Duration,
    pub(super) submit_and_wait: Duration,
    pub(super) status_read: Duration,
    pub(super) total: Duration,
    pub(super) gpu: Option<Duration>,
    pub(super) status: BatchAffineStatus,
    pub(super) counters: FlatSegmentedBenchmarkCounters,
}

/// Resident buffers and immutable scheduling for repeated samples.
pub(super) struct PreparedFlatSegmentedBenchmark<'session> {
    session: &'session MetalSession,
    srs: TrackedBuffer,
    chunks: Box<[PreparedChunk]>,
    points_a: TrackedBuffer,
    points_b: TrackedBuffer,
    status: TrackedBuffer,
    width: FlatSegmentedWidth,
    dynamic_threadgroup_bytes: usize,
    counters: FlatSegmentedBenchmarkCounters,
}

struct PreparedChunk {
    records: TrackedBuffer,
    tiles: TrackedBuffer,
    group_round_counts: TrackedBuffer,
    roots: TrackedBuffer,
    passes: Box<[PreparedPass]>,
    root_count: u32,
}

#[derive(Clone, Copy)]
struct PreparedPass {
    tile_byte_offset: usize,
    group_byte_offset: usize,
    parameters: FlatSegmentedParameters,
}

struct HostWorkload {
    chunks: Box<[HostChunk]>,
    max_intermediate_points: u32,
}

struct HostChunk {
    records: Box<[u32]>,
    plan: FlatSegmentedPlan,
    tiles: Box<[FlatSegmentedTile]>,
    group_round_counts: Box<[u32]>,
    passes: Box<[PreparedPass]>,
}

#[derive(Clone, Copy)]
struct PipelineGeometry {
    execution_width: usize,
    max_threads: usize,
    dynamic_threadgroup_bytes: usize,
    static_threadgroup_bytes: usize,
}

impl MetalSession {
    /// Prepares one width of the bounded flat-segmented decision benchmark.
    pub(super) fn prepare_flat_segmented_benchmark<'session>(
        &'session self,
        srs: &[Bls12_381G1Affine],
        chunks: &[BatchAffineBenchmarkChunk<'_>],
        width: FlatSegmentedWidth,
    ) -> Result<PreparedFlatSegmentedBenchmark<'session>, MetalError> {
        let host = HostWorkload::build(srs.len(), chunks, width)?;
        let geometry = self.flat_segmented_geometry(width)?;
        let counters = benchmark_counters(srs.len(), &host, width, geometry)?;

        let mut prepared_chunks = Vec::new();
        prepared_chunks
            .try_reserve_exact(host.chunks.len())
            .map_err(|_| MetalError::Dispatch("flat-segmented prepared chunk allocation failed"))?;

        let _execution = self.execution_guard();
        let srs_buffer = self.buffer_from_slice(srs, "midnight.flat_segmented.srs")?;
        for chunk in host.chunks {
            let records =
                self.buffer_from_slice(&chunk.records, "midnight.flat_segmented.records")?;
            let tiles = self.buffer_from_slice(&chunk.tiles, "midnight.flat_segmented.tiles")?;
            let group_round_counts = self.buffer_from_slice(
                &chunk.group_round_counts,
                "midnight.flat_segmented.group_round_counts",
            )?;
            let roots = checked_buffer::<Bls12_381G1Affine>(
                self,
                chunk.plan.root_count,
                "midnight.flat_segmented.roots",
            )?;
            prepared_chunks.push(PreparedChunk {
                records,
                tiles,
                group_round_counts,
                roots,
                passes: chunk.passes,
                root_count: chunk.plan.root_count,
            });
        }

        let points_a = checked_buffer::<BatchAffinePoint32>(
            self,
            host.max_intermediate_points.max(1),
            "midnight.flat_segmented.points_a",
        )?;
        let points_b = checked_buffer::<BatchAffinePoint32>(
            self,
            host.max_intermediate_points.max(1),
            "midnight.flat_segmented.points_b",
        )?;
        let status =
            checked_buffer::<BatchAffineStatus>(self, 1, "midnight.flat_segmented.status")?;

        Ok(PreparedFlatSegmentedBenchmark {
            session: self,
            srs: srs_buffer,
            chunks: prepared_chunks.into_boxed_slice(),
            points_a,
            points_b,
            status,
            width,
            dynamic_threadgroup_bytes: geometry.dynamic_threadgroup_bytes,
            counters,
        })
    }

    fn flat_segmented_geometry(
        &self,
        width: FlatSegmentedWidth,
    ) -> Result<PipelineGeometry, MetalError> {
        let pipeline = &self.flat_segmented.reduce;
        let group_width = width.lanes() as usize;
        let execution_width = pipeline.threadExecutionWidth();
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        if group_width > max_threads || !group_width.is_multiple_of(execution_width) {
            return Err(MetalError::Dispatch(
                "flat-segmented width violates the pipeline limits",
            ));
        }
        let tree_elements = group_width
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(MetalError::Dispatch(
                "flat-segmented threadgroup tree size overflow",
            ))?;
        let dynamic_threadgroup_bytes = tree_elements
            .checked_mul(size_of::<super::BatchAffineFp32>())
            .ok_or(MetalError::Dispatch(
                "flat-segmented threadgroup byte size overflow",
            ))?;
        let static_threadgroup_bytes = pipeline.staticThreadgroupMemoryLength();
        let total = dynamic_threadgroup_bytes
            .checked_add(static_threadgroup_bytes)
            .ok_or(MetalError::Dispatch(
                "flat-segmented total threadgroup byte size overflow",
            ))?;
        if total > self.device.maxThreadgroupMemoryLength() {
            return Err(MetalError::Dispatch(
                "flat-segmented threadgroup storage exceeds the device limit",
            ));
        }
        Ok(PipelineGeometry {
            execution_width,
            max_threads,
            dynamic_threadgroup_bytes,
            static_threadgroup_bytes,
        })
    }
}

impl PreparedFlatSegmentedBenchmark<'_> {
    pub(super) fn counters(&self) -> FlatSegmentedBenchmarkCounters {
        self.counters
    }

    /// Executes all chunks in one encoder, command buffer, and completion wait.
    pub(super) fn execute(&self) -> Result<FlatSegmentedBenchmarkSample, MetalError> {
        let _execution = self.session.execution_guard();
        let total_start = Instant::now();
        let command = self
            .session
            .command_buffer("midnight.flat_segmented.benchmark.sample")?;
        if !command.retainedReferences() {
            return Err(MetalError::Dispatch(
                "flat-segmented benchmark requires retained references",
            ));
        }
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.session.clear_words);
        // SAFETY: status is two aligned u64 words retained through completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 0);
        }
        self.session
            .dispatch(&encoder, &self.session.clear_words, 2);

        for chunk in &self.chunks {
            if let Err(error) = self.encode_chunk(&encoder, chunk) {
                encoder.endEncoding();
                return Err(error);
            }
        }
        encoder.endEncoding();
        let encoding = total_start.elapsed();

        let timing_command = command.clone();
        let wait_start = Instant::now();
        self.session.submit_and_wait(PendingCommand {
            command,
            _resources: Vec::new(),
        })?;
        let submit_and_wait = wait_start.elapsed();
        let status_read_start = Instant::now();
        let status = read_status(self.session, &self.status)?;
        let status_read = status_read_start.elapsed();
        let total = total_start.elapsed();
        let gpu = completed_gpu_duration(&timing_command);

        Ok(FlatSegmentedBenchmarkSample {
            encoding,
            submit_and_wait,
            status_read,
            total,
            gpu,
            status,
            counters: self.counters,
        })
    }

    fn encode_chunk(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        chunk: &PreparedChunk,
    ) -> Result<(), MetalError> {
        for (pass_index, pass) in chunk.passes.iter().enumerate() {
            let (input_points, output_points) = if pass_index.is_multiple_of(2) {
                (&self.points_b, &self.points_a)
            } else {
                (&self.points_a, &self.points_b)
            };
            encoder.setComputePipelineState(&self.session.flat_segmented.reduce);
            // SAFETY: the canonical planner validates every descriptor range and
            // output slot. Bound buffers and byte offsets remain retained and in
            // bounds for this synchronous command.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.srs.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&chunk.records.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&chunk.tiles.raw), pass.tile_byte_offset, 2);
                encoder.setBuffer_offset_atIndex(
                    Some(&chunk.group_round_counts.raw),
                    pass.group_byte_offset,
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(&input_points.raw), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&output_points.raw), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 6);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&pass.parameters).cast::<c_void>(),
                    size_of::<FlatSegmentedParameters>(),
                    7,
                );
                encoder.setThreadgroupMemoryLength_atIndex(self.dynamic_threadgroup_bytes, 0);
            }
            self.session.dispatch_threadgroups(
                encoder,
                &self.session.flat_segmented.reduce,
                MTLSize {
                    width: pass.parameters.group_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.width.lanes() as usize,
                    height: 1,
                    depth: 1,
                },
            )?;
        }

        let root_source = if chunk.passes.len().is_multiple_of(2) {
            &self.points_b
        } else {
            &self.points_a
        };
        encoder.setComputePipelineState(&self.session.batch_affine.export_roots);
        // SAFETY: the terminal planner pass writes one dense point per original
        // run and the public root allocation has that exact extent.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&root_source.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&chunk.roots.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&chunk.root_count).cast::<c_void>(),
                size_of::<u32>(),
                3,
            );
        }
        self.session.dispatch(
            encoder,
            &self.session.batch_affine.export_roots,
            chunk.root_count as usize,
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn read_roots(
        &self,
        chunk_index: usize,
    ) -> Result<Vec<Bls12_381G1Affine>, MetalError> {
        let chunk = self.chunks.get(chunk_index).ok_or(MetalError::Dispatch(
            "flat-segmented chunk index is out of range",
        ))?;
        let _execution = self.session.execution_guard();
        self.session
            .read_values(&chunk.roots, chunk.root_count as usize)
    }
}

impl HostWorkload {
    fn build(
        srs_len: usize,
        chunks: &[BatchAffineBenchmarkChunk<'_>],
        width: FlatSegmentedWidth,
    ) -> Result<Self, MetalError> {
        validate_inputs(srs_len, chunks)?;
        let srs_count = checked_u32(srs_len, "SRS length")?;
        let mut host_chunks = Vec::new();
        host_chunks
            .try_reserve_exact(chunks.len())
            .map_err(|_| MetalError::Dispatch("flat-segmented host chunk allocation failed"))?;
        let mut max_intermediate_points = 0u32;

        for chunk in chunks {
            let plan = compile_flat_segmented_plan(chunk.run_lengths, width)
                .map_err(|_| MetalError::Dispatch("flat-segmented plan compilation failed"))?;
            validate_flat_segmented_plan(&plan)
                .map_err(|_| MetalError::Dispatch("flat-segmented plan validation failed"))?;
            validate_records(srs_len, chunk.packed_records, &plan)?;
            if plan.passes.is_empty() {
                return Err(MetalError::Dispatch(
                    "flat-segmented plan must materialize at least one pass",
                ));
            }

            let mut records = Vec::new();
            records
                .try_reserve_exact(chunk.packed_records.len())
                .map_err(|_| MetalError::Dispatch("flat-segmented record copy failed"))?;
            records.extend_from_slice(chunk.packed_records);

            let total_tiles = plan.passes.iter().try_fold(0usize, |total, pass| {
                total
                    .checked_add(pass.tiles.len())
                    .ok_or(MetalError::Dispatch("flat-segmented tile count overflow"))
            })?;
            let total_groups = plan.passes.iter().try_fold(0usize, |total, pass| {
                total
                    .checked_add(pass.group_round_counts.len())
                    .ok_or(MetalError::Dispatch("flat-segmented group count overflow"))
            })?;
            let mut tiles = Vec::new();
            tiles
                .try_reserve_exact(total_tiles)
                .map_err(|_| MetalError::Dispatch("flat-segmented tile copy failed"))?;
            let mut round_counts = Vec::new();
            round_counts
                .try_reserve_exact(total_groups)
                .map_err(|_| MetalError::Dispatch("flat-segmented group copy failed"))?;

            let mut passes = Vec::new();
            passes
                .try_reserve_exact(plan.passes.len())
                .map_err(|_| MetalError::Dispatch("flat-segmented pass copy failed"))?;
            for (pass_index, pass) in plan.passes.iter().enumerate() {
                let tile_offset = checked_u32(tiles.len(), "tile offset")?;
                let group_offset = checked_u32(round_counts.len(), "group offset")?;
                tiles.extend_from_slice(&pass.tiles);
                round_counts.extend_from_slice(&pass.group_round_counts);
                passes.push(PreparedPass {
                    tile_byte_offset: byte_offset::<FlatSegmentedTile>(tile_offset)?,
                    group_byte_offset: byte_offset::<u32>(group_offset)?,
                    parameters: FlatSegmentedParameters {
                        tile_count: checked_u32(pass.tiles.len(), "pass tile count")?,
                        input_count: pass.input_count,
                        output_count: pass.output_count,
                        srs_count,
                        group_count: checked_u32(
                            pass.group_round_counts.len(),
                            "pass group count",
                        )?,
                        input_kind: if pass_index == 0 {
                            INPUT_RECORDS
                        } else {
                            INPUT_POINTS
                        },
                        group_width: width.lanes(),
                        reserved: 0,
                    },
                });
            }
            max_intermediate_points = max_intermediate_points.max(plan.max_intermediate_points);
            host_chunks.push(HostChunk {
                records: records.into_boxed_slice(),
                plan,
                tiles: tiles.into_boxed_slice(),
                group_round_counts: round_counts.into_boxed_slice(),
                passes: passes.into_boxed_slice(),
            });
        }
        Ok(Self {
            chunks: host_chunks.into_boxed_slice(),
            max_intermediate_points,
        })
    }
}

fn validate_inputs(
    srs_len: usize,
    chunks: &[BatchAffineBenchmarkChunk<'_>],
) -> Result<(), MetalError> {
    if srs_len == 0 || srs_len > PACKED_RECORD_INDEX_MASK as usize + 1 {
        return Err(MetalError::Dispatch(
            "flat-segmented SRS length exceeds the packed record ABI",
        ));
    }
    if chunks.is_empty() {
        return Err(MetalError::Dispatch(
            "flat-segmented benchmark requires at least one chunk",
        ));
    }
    if chunks
        .iter()
        .any(|chunk| chunk.run_lengths.is_empty() || chunk.run_lengths.contains(&0))
    {
        return Err(MetalError::Dispatch(
            "flat-segmented benchmark requires nonempty runs",
        ));
    }
    Ok(())
}

fn validate_records(
    srs_len: usize,
    records: &[u32],
    plan: &FlatSegmentedPlan,
) -> Result<(), MetalError> {
    if records.len() != plan.input_count as usize {
        return Err(MetalError::LengthMismatch {
            left: records.len(),
            right: plan.input_count as usize,
        });
    }
    if records
        .iter()
        .any(|record| (record & PACKED_RECORD_INDEX_MASK) as usize >= srs_len)
    {
        return Err(MetalError::Dispatch(
            "flat-segmented record index exceeds the SRS",
        ));
    }
    Ok(())
}

fn benchmark_counters(
    srs_len: usize,
    host: &HostWorkload,
    width: FlatSegmentedWidth,
    geometry: PipelineGeometry,
) -> Result<FlatSegmentedBenchmarkCounters, MetalError> {
    let srs_bytes = byte_count::<Bls12_381G1Affine>(srs_len, "flat SRS bytes")?;
    let mut leaf_count = 0u64;
    let mut root_count = 0u64;
    let mut pair_reduction_count = 0u64;
    let mut pass_count = 0u64;
    let mut real_tile_count = 0u64;
    let mut padded_tile_count = 0u64;
    let mut group_count = 0u64;
    let mut scheduled_lane_round_count = 0u64;
    let mut batch_inversion_count = 0u64;
    let mut immutable_chunk_bytes = 0u64;
    let mut root_output_bytes = 0u64;
    let mut logical_device_bytes = size_of::<BatchAffineStatus>() as u64;

    for chunk in &host.chunks {
        let counters = chunk.plan.counters;
        leaf_count = checked_sum(leaf_count, counters.input_count, "flat leaf count")?;
        root_count = checked_sum(root_count, counters.root_count, "flat root count")?;
        pair_reduction_count = checked_sum(
            pair_reduction_count,
            counters.useful_reduction_count,
            "flat pair count",
        )?;
        pass_count = checked_sum(pass_count, counters.pass_count, "flat pass count")?;
        real_tile_count = checked_sum(
            real_tile_count,
            counters.real_tile_count,
            "flat real tile count",
        )?;
        padded_tile_count = checked_sum(
            padded_tile_count,
            counters.padded_tile_count,
            "flat padded tile count",
        )?;
        group_count = checked_sum(group_count, counters.group_count, "flat group count")?;
        scheduled_lane_round_count = checked_sum(
            scheduled_lane_round_count,
            counters.scheduled_lane_rounds,
            "flat scheduled rounds",
        )?;
        batch_inversion_count = checked_sum(
            batch_inversion_count,
            counters.inversion_count,
            "flat inversion count",
        )?;

        let record_bytes = byte_count::<u32>(chunk.records.len(), "flat record bytes")?;
        let tile_bytes = byte_count::<FlatSegmentedTile>(chunk.tiles.len(), "flat tile bytes")?;
        let group_bytes = byte_count::<u32>(chunk.group_round_counts.len(), "flat group bytes")?;
        immutable_chunk_bytes = [record_bytes, tile_bytes, group_bytes]
            .into_iter()
            .try_fold(immutable_chunk_bytes, |total, bytes| {
                checked_sum(total, bytes, "flat immutable bytes")
            })?;
        root_output_bytes = checked_sum(
            root_output_bytes,
            byte_count::<Bls12_381G1Affine>(chunk.plan.root_count as usize, "flat root bytes")?,
            "flat root bytes",
        )?;

        for (pass_index, pass) in chunk.plan.passes.iter().enumerate() {
            logical_device_bytes = checked_sum(
                logical_device_bytes,
                scaled(
                    checked_u32(pass.tiles.len(), "logical tile count")?,
                    20,
                    "flat schedule traffic",
                )?,
                "flat logical bytes",
            )?;
            let input_bytes = if pass_index == 0 { 100 } else { 96 };
            logical_device_bytes = checked_sum(
                logical_device_bytes,
                scaled(pass.input_count, input_bytes, "flat input traffic")?,
                "flat logical bytes",
            )?;
            logical_device_bytes = checked_sum(
                logical_device_bytes,
                scaled(pass.output_count, 96, "flat output traffic")?,
                "flat logical bytes",
            )?;
        }
        logical_device_bytes = checked_sum(
            logical_device_bytes,
            scaled(chunk.plan.root_count, 192, "flat export traffic")?,
            "flat logical bytes",
        )?;
    }

    let point_bytes = byte_count::<BatchAffinePoint32>(
        host.max_intermediate_points.max(1) as usize,
        "flat point bytes",
    )?
    .checked_mul(2)
    .ok_or(MetalError::Dispatch("flat point byte overflow"))?;
    let status_bytes = size_of::<BatchAffineStatus>() as u64;
    let shared_scratch_bytes = checked_sum(point_bytes, status_bytes, "flat scratch bytes")?;
    let allocated_bytes = [
        srs_bytes,
        immutable_chunk_bytes,
        shared_scratch_bytes,
        root_output_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| {
        checked_sum(total, bytes, "flat allocated bytes")
    })?;
    let host_upload_bytes = checked_sum(srs_bytes, immutable_chunk_bytes, "flat upload bytes")?;
    let chunk_count = host.chunks.len() as u64;
    let metal_allocation_count = chunk_count
        .checked_mul(4)
        .and_then(|value| value.checked_add(4))
        .ok_or(MetalError::Dispatch("flat allocation count overflow"))?;

    Ok(FlatSegmentedBenchmarkCounters {
        chunk_count,
        leaf_count,
        root_count,
        pair_reduction_count,
        pass_count,
        real_tile_count,
        padded_tile_count,
        group_count,
        scheduled_lane_round_count,
        batch_inversion_count,
        dispatch_count: 1 + pass_count + chunk_count,
        command_buffer_count: 1,
        host_wait_count: 1,
        metal_allocation_count,
        group_width: u64::from(width.lanes()),
        pipeline_execution_width: geometry.execution_width as u64,
        pipeline_max_threads: geometry.max_threads as u64,
        dynamic_threadgroup_bytes: geometry.dynamic_threadgroup_bytes as u64,
        static_threadgroup_bytes: geometry.static_threadgroup_bytes as u64,
        srs_bytes,
        immutable_chunk_bytes,
        shared_scratch_bytes,
        root_output_bytes,
        allocated_bytes,
        host_upload_bytes,
        host_status_read_bytes: status_bytes,
        logical_device_bytes,
        max_intermediate_points: u64::from(host.max_intermediate_points),
    })
}

fn byte_offset<T>(elements: u32) -> Result<usize, MetalError> {
    (elements as usize)
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::LengthOverflow(elements as usize))
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| match field {
        "SRS length" => MetalError::LengthOverflow(value),
        _ => MetalError::Dispatch("flat-segmented value exceeds the u32 ABI"),
    })
}

fn byte_count<T>(elements: usize, field: &'static str) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::LengthOverflow(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::Dispatch(field))
}

fn checked_sum(left: u64, right: u64, field: &'static str) -> Result<u64, MetalError> {
    left.checked_add(right).ok_or(MetalError::Dispatch(field))
}

fn scaled(elements: u32, bytes: u64, field: &'static str) -> Result<u64, MetalError> {
    u64::from(elements)
        .checked_mul(bytes)
        .ok_or(MetalError::Dispatch(field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::batch_affine_probe::compile_balanced_run_plan;

    const NEGATE: u32 = 1 << 31;

    #[test]
    fn flat_segmented_reducer_matches_the_global_tree_oracle_across_widths() {
        let srs = [
            Bls12_381G1Affine::default(),
            raw_affine(scalar_multiple(1)),
            raw_affine(scalar_multiple(2)),
            raw_affine(scalar_multiple(3)),
        ];
        let run_lengths = [1usize, 2, 3, 31, 32, 33, 65, 257, 513];
        let pattern = [0, 1, NEGATE | 1, 2, 2, 0, 3, NEGATE | 2, 1, 3];
        let records = pattern
            .into_iter()
            .cycle()
            .take(run_lengths.iter().sum())
            .collect::<Vec<_>>();
        let balanced = compile_balanced_run_plan(&run_lengths).expect("compile oracle plan");
        let session = MetalSession::new().expect("initialize Metal runtime");
        let oracle = session
            .replay_batch_affine(&srs, &records, &balanced)
            .expect("execute global-tree oracle");
        assert_eq!(oracle.status, BatchAffineStatus::default());

        let chunk = [BatchAffineBenchmarkChunk {
            run_lengths: &run_lengths,
            packed_records: &records,
        }];
        for width in [FlatSegmentedWidth::G128, FlatSegmentedWidth::G256] {
            let prepared = session
                .prepare_flat_segmented_benchmark(&srs, &chunk, width)
                .unwrap_or_else(|error| panic!("prepare {width:?}: {error}"));
            let sample = prepared
                .execute()
                .unwrap_or_else(|error| panic!("execute {width:?}: {error}"));
            assert_eq!(sample.status, BatchAffineStatus::default());
            assert_eq!(
                sample.counters.pair_reduction_count,
                balanced.total_pair_count
            );
            assert_eq!(
                prepared.read_roots(0).expect("read flat roots"),
                oracle.roots,
                "width {width:?}"
            );
        }
    }

    #[test]
    fn ragged_16797_and_cross_tile_cancellation_match_the_oracle() {
        let srs = [
            raw_affine(scalar_multiple(1)),
            raw_affine(scalar_multiple(32)),
        ];
        let run_lengths = [33usize, 16_797];
        let mut records = Vec::with_capacity(run_lengths.iter().sum());
        records.extend(std::iter::repeat_n(0u32, 32));
        records.push(NEGATE | 1);
        records
            .extend((0usize..16_797).map(|index| if index.is_multiple_of(2) { 0 } else { NEGATE }));

        let balanced = compile_balanced_run_plan(&run_lengths).expect("compile ragged oracle plan");
        let session = MetalSession::new().expect("initialize Metal runtime");
        let oracle = session
            .replay_batch_affine(&srs, &records, &balanced)
            .expect("execute ragged global-tree oracle");
        assert_eq!(oracle.status, BatchAffineStatus::default());
        assert_eq!(oracle.roots[0], Bls12_381G1Affine::default());

        let chunks = [BatchAffineBenchmarkChunk {
            run_lengths: &run_lengths,
            packed_records: &records,
        }];
        let prepared = session
            .prepare_flat_segmented_benchmark(&srs, &chunks, FlatSegmentedWidth::G256)
            .expect("prepare ragged flat workload");
        let sample = prepared.execute().expect("execute ragged flat workload");
        assert_eq!(sample.status, BatchAffineStatus::default());
        assert_eq!(
            prepared.read_roots(0).expect("read ragged roots"),
            oracle.roots
        );
    }

    #[test]
    fn later_invalid_chunk_is_rejected_before_any_metal_allocation() {
        let srs = [raw_affine(scalar_multiple(1))];
        let chunks = [
            BatchAffineBenchmarkChunk {
                run_lengths: &[1],
                packed_records: &[0],
            },
            BatchAffineBenchmarkChunk {
                run_lengths: &[1],
                packed_records: &[1],
            },
        ];
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let error =
            match session.prepare_flat_segmented_benchmark(&srs, &chunks, FlatSegmentedWidth::G256)
            {
                Ok(_) => panic!("invalid later chunk must fail"),
                Err(error) => error,
            };
        assert!(matches!(error, MetalError::Dispatch(_)));
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
        assert_eq!(activity.host_waits, 0);
        assert_eq!(activity.allocations, 0);
    }

    #[test]
    fn prepared_counter_identity_matches_activity() {
        let srs = [
            raw_affine(scalar_multiple(1)),
            raw_affine(scalar_multiple(2)),
        ];
        let run_lengths = [1usize, 33, 65];
        let records = [0u32, 1]
            .into_iter()
            .cycle()
            .take(run_lengths.iter().sum())
            .collect::<Vec<_>>();
        let chunks = [BatchAffineBenchmarkChunk {
            run_lengths: &run_lengths,
            packed_records: &records,
        }];
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let prepared = session
            .prepare_flat_segmented_benchmark(&srs, &chunks, FlatSegmentedWidth::G128)
            .expect("prepare counter workload");
        let counters = prepared.counters();
        let preparation = session.activity();
        assert_eq!(preparation.allocations, counters.metal_allocation_count);
        assert_eq!(preparation.allocated_bytes, counters.allocated_bytes);
        assert_eq!(preparation.host_write_bytes, counters.host_upload_bytes);
        assert_eq!(preparation.command_buffers, 0);

        session.reset_activity();
        let sample = prepared.execute().expect("execute counter workload");
        assert_eq!(sample.status, BatchAffineStatus::default());
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.dispatches, counters.dispatch_count);
        assert_eq!(activity.host_waits, 1);
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.host_write_bytes, 0);
        assert_eq!(activity.host_read_bytes, size_of_val(&sample.status) as u64);
    }

    fn scalar_multiple(scalar: u64) -> blst::blst_p1 {
        let scalar = scalar.to_le_bytes();
        let mut output = blst::blst_p1::default();
        // SAFETY: the generator and scalar bytes are valid for the complete call.
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

    fn raw_affine(point: blst::blst_p1) -> Bls12_381G1Affine {
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: input and output are initialized and disjoint.
        unsafe {
            blst::blst_p1_to_affine(&mut affine, &point);
        }
        Bls12_381G1Affine {
            x: affine.x.l,
            y: affine.y.l,
        }
    }
}
