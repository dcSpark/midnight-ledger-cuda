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

//! Prepared, reusable multi-chunk batch-affine performance diagnostic.
//!
//! This test-only harness deliberately remains separate from proof routing. It
//! validates and compiles the entire workload before asking Metal for a buffer,
//! uploads immutable inputs once, and reuses one maximum-sized scratch arena.
//! Every sample encodes all chunks into one encoder and one command buffer.

use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr::NonNull,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::batch_affine_probe::{
    BatchAffineRunPlan, compile_balanced_run_plan, validate_balanced_run_plan,
};
use super::{
    BatchAffineFp32, BatchAffineGatherParameters, BatchAffineLevelParameters, BatchAffinePoint32,
    BatchAffineProductParameters, BatchAffineStatus, MetalSession, PendingCommand, TrackedBuffer,
};
use crate::{Bls12_381G1Affine, MetalError};

const PACKED_RECORD_NEGATE: u32 = 1 << 31;
const PACKED_RECORD_INDEX_MASK: u32 = PACKED_RECORD_NEGATE - 1;

/// Borrowed host description of one independently prepared replay chunk.
#[derive(Clone, Copy, Debug)]
pub(super) struct BatchAffineBenchmarkChunk<'a> {
    pub(super) run_lengths: &'a [usize],
    pub(super) packed_records: &'a [u32],
}

/// Static work and memory accounting for every execution of a prepared replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchAffineBenchmarkCounters {
    pub(super) chunk_count: u64,
    pub(super) leaf_count: u64,
    pub(super) root_count: u64,
    pub(super) pair_reduction_count: u64,
    pub(super) carry_copy_count: u64,
    pub(super) level_count: u64,
    pub(super) dispatch_count: u64,
    pub(super) command_buffer_count: u64,
    pub(super) host_wait_count: u64,
    pub(super) metal_allocation_count: u64,
    pub(super) srs_bytes: u64,
    pub(super) immutable_chunk_bytes: u64,
    pub(super) shared_scratch_bytes: u64,
    pub(super) root_output_bytes: u64,
    pub(super) allocated_bytes: u64,
    pub(super) host_upload_bytes: u64,
    pub(super) host_status_read_bytes: u64,
    /// Nominal per-sample device-buffer traffic implied by the kernel ABI.
    /// This is not a claim about cache misses or physical DRAM traffic.
    pub(super) logical_device_bytes: u64,
    pub(super) max_chunk_leaves: u64,
    pub(super) max_level_pairs: u64,
    pub(super) max_product_tree_slots: u64,
}

/// Timings and the sole host-visible device result from one replay sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchAffineBenchmarkSample {
    /// Host time spent creating and encoding the command buffer.
    pub(super) encoding: Duration,
    /// Host time from commit through the completion wait.
    pub(super) submit_and_wait: Duration,
    /// Host time to read the sole 16-byte status after completion.
    pub(super) status_read: Duration,
    /// Host time from command creation through the completed status read.
    pub(super) total: Duration,
    /// Device execution interval reported by the completed command buffer.
    pub(super) gpu: Option<Duration>,
    /// The 16-byte diagnostic status read after timing stopped.
    pub(super) status: BatchAffineStatus,
    /// Static accounting for this prepared workload.
    pub(super) counters: BatchAffineBenchmarkCounters,
}

/// Metal buffers and topology retained across repeated benchmark samples.
pub(super) struct PreparedBatchAffineBenchmark<'session> {
    session: &'session MetalSession,
    _srs: TrackedBuffer,
    chunks: Box<[PreparedChunk]>,
    points_a: TrackedBuffer,
    points_b: TrackedBuffer,
    numerators: TrackedBuffer,
    cases: TrackedBuffer,
    product_tree: TrackedBuffer,
    status: TrackedBuffer,
    counters: BatchAffineBenchmarkCounters,
}

struct PreparedChunk {
    plan: BatchAffineRunPlan,
    product_trees: Box<[ProductTreePlan]>,
    records: TrackedBuffer,
    destination_map: TrackedBuffer,
    roots: TrackedBuffer,
    gather: BatchAffineGatherParameters,
    root_count: u32,
}

struct HostWorkload {
    chunks: Box<[HostChunk]>,
    max_chunk_leaves: u32,
    max_level_pairs: u32,
    max_product_tree_slots: u32,
    counters: BatchAffineBenchmarkCounters,
}

struct HostChunk {
    plan: BatchAffineRunPlan,
    product_trees: Box<[ProductTreePlan]>,
    ordered_records: Box<[u32]>,
    root_count: u32,
}

struct ProductTreePlan {
    reductions: Box<[BatchAffineProductParameters]>,
    root: BatchAffineProductParameters,
    slots: u32,
}

impl MetalSession {
    /// Validates, allocates, and uploads a reusable batch-affine benchmark.
    ///
    /// `HostWorkload::build` performs every fallible topology, index, extent,
    /// and byte calculation before this method takes the execution lock or
    /// invokes a Metal allocation primitive.
    pub(super) fn prepare_batch_affine_benchmark<'session>(
        &'session self,
        srs: &[Bls12_381G1Affine],
        chunks: &[BatchAffineBenchmarkChunk<'_>],
    ) -> Result<PreparedBatchAffineBenchmark<'session>, MetalError> {
        let host = HostWorkload::build(srs.len(), chunks)?;
        let mut prepared_chunks = Vec::new();
        prepared_chunks
            .try_reserve_exact(host.chunks.len())
            .map_err(|_| MetalError::Dispatch("batch-affine prepared chunk allocation failed"))?;

        let _execution = self.execution_guard();
        let srs_buffer = self.buffer_from_slice(srs, "midnight.batch_affine.benchmark.srs")?;
        for chunk in host.chunks {
            let records = self.buffer_from_slice(
                &chunk.ordered_records,
                "midnight.batch_affine.benchmark.records",
            )?;
            let dummy_map = [0u32];
            let map_values = if chunk.plan.destination_map.is_empty() {
                &dummy_map[..]
            } else {
                &chunk.plan.destination_map
            };
            let destination_map = self.buffer_from_slice(
                map_values,
                "midnight.batch_affine.benchmark.destination_map",
            )?;
            let roots = checked_buffer::<Bls12_381G1Affine>(
                self,
                chunk.root_count,
                "midnight.batch_affine.benchmark.roots",
            )?;
            let leaf_count = chunk.plan.leaf_count;
            prepared_chunks.push(PreparedChunk {
                plan: chunk.plan,
                product_trees: chunk.product_trees,
                records,
                destination_map,
                roots,
                gather: BatchAffineGatherParameters {
                    record_offset: 0,
                    record_count: leaf_count,
                    srs_count: checked_u32(srs.len(), "SRS length")?,
                    reserved: 0,
                },
                root_count: chunk.root_count,
            });
        }

        let points_a = checked_buffer::<BatchAffinePoint32>(
            self,
            host.max_chunk_leaves,
            "midnight.batch_affine.benchmark.points_a",
        )?;
        let points_b = checked_buffer::<BatchAffinePoint32>(
            self,
            host.max_chunk_leaves,
            "midnight.batch_affine.benchmark.points_b",
        )?;
        let numerators = checked_buffer::<BatchAffineFp32>(
            self,
            host.max_level_pairs.max(1),
            "midnight.batch_affine.benchmark.numerators",
        )?;
        let cases = checked_buffer::<u32>(
            self,
            host.max_level_pairs.max(1),
            "midnight.batch_affine.benchmark.cases",
        )?;
        let product_tree = checked_buffer::<BatchAffineFp32>(
            self,
            host.max_product_tree_slots.max(1),
            "midnight.batch_affine.benchmark.product_tree",
        )?;
        let status =
            checked_buffer::<BatchAffineStatus>(self, 1, "midnight.batch_affine.benchmark.status")?;

        Ok(PreparedBatchAffineBenchmark {
            session: self,
            _srs: srs_buffer,
            chunks: prepared_chunks.into_boxed_slice(),
            points_a,
            points_b,
            numerators,
            cases,
            product_tree,
            status,
            counters: host.counters,
        })
    }
}

impl PreparedBatchAffineBenchmark<'_> {
    /// Returns the immutable accounting computed before Metal preparation.
    pub(super) fn counters(&self) -> BatchAffineBenchmarkCounters {
        self.counters
    }

    /// Executes every prepared chunk sequentially in one encoder and command.
    ///
    /// No Metal buffer or Rust collection allocation occurs on the success
    /// path. The command buffer and encoder are the per-sample driver objects.
    /// Root outputs remain resident; only the shared 16-byte status is read,
    /// after the completion wait. That read has its own interval and is also
    /// included in the end-to-end `total` required by the promotion gate.
    pub(super) fn execute(&self) -> Result<BatchAffineBenchmarkSample, MetalError> {
        let _execution = self.session.execution_guard();
        let total_start = Instant::now();
        let command = self
            .session
            .command_buffer("midnight.batch_affine.benchmark.sample")?;
        if !command.retainedReferences() {
            return Err(MetalError::Dispatch(
                "batch-affine benchmark requires a retained-reference command buffer",
            ));
        }
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.session.clear_words);
        // SAFETY: the status buffer is two aligned u64 words retained by this
        // prepared object for the entire synchronous command lifetime.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 0);
        }
        self.session
            .dispatch(&encoder, &self.session.clear_words, 2);

        for chunk in &self.chunks {
            self.encode_chunk(&encoder, chunk);
        }
        encoder.endEncoding();
        let encoding = total_start.elapsed();

        let timing_command = command.clone();
        let wait_start = Instant::now();
        // `commandBuffer`, unlike `commandBufferWithUnretainedReferences`,
        // retains encoded resources. The prepared object is also borrowed until
        // this synchronous wait returns, so an empty retention vector requires
        // no per-sample Rust heap allocation without weakening resource life.
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

        Ok(BatchAffineBenchmarkSample {
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
    ) {
        encoder.setComputePipelineState(&self.session.batch_affine.gather_signed_leaves);
        // SAFETY: preparation validated every packed SRS index and allocated
        // exact record and point extents. All buffers outlive this encoder.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self._srs.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&chunk.records.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.points_a.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 3);
        }
        set_parameter(encoder, &chunk.gather, 4);
        self.session.dispatch(
            encoder,
            &self.session.batch_affine.gather_signed_leaves,
            chunk.plan.leaf_count as usize,
        );

        let mut source_slot = 0usize;
        for (level, tree) in chunk.plan.levels.iter().zip(&chunk.product_trees) {
            let parameters = BatchAffineLevelParameters {
                destination_map_offset: level.destination_map_offset,
                pair_count: level.pair_count,
                carry_count: level.carry_count,
                destination_len: level.destination_count,
            };
            let (source, destination) = if source_slot == 0 {
                (&self.points_a, &self.points_b)
            } else {
                (&self.points_b, &self.points_a)
            };

            encoder.setComputePipelineState(&self.session.batch_affine.prepare_pairs);
            // SAFETY: host topology validation covers every point and scratch
            // access for this level in the shared maximum-sized arena.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.numerators.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.cases.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.product_tree.raw), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 4);
            }
            set_parameter(encoder, &parameters, 5);
            self.session.dispatch(
                encoder,
                &self.session.batch_affine.prepare_pairs,
                level.pair_count as usize,
            );

            for reduction in tree.reductions.iter() {
                encoder.setComputePipelineState(&self.session.batch_affine.reduce_products);
                // SAFETY: the precomputed compact tree offsets are in bounds of
                // the shared tree allocation and execute in encoder order.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&self.product_tree.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 1);
                }
                set_parameter(encoder, reduction, 2);
                self.session.dispatch(
                    encoder,
                    &self.session.batch_affine.reduce_products,
                    reduction.output_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.session.batch_affine.invert_product_root);
            // SAFETY: `tree.root` selects the initialized one-element root of
            // the compact product tree for this level.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.product_tree.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 1);
            }
            set_parameter(encoder, &tree.root, 2);
            self.session
                .dispatch(encoder, &self.session.batch_affine.invert_product_root, 1);

            for reduction in tree.reductions.iter().rev() {
                let expansion = BatchAffineProductParameters {
                    input_offset: reduction.output_offset,
                    input_count: reduction.output_count,
                    output_offset: reduction.input_offset,
                    output_count: reduction.input_count,
                };
                encoder.setComputePipelineState(&self.session.batch_affine.expand_product_inverses);
                // SAFETY: reverse traversal expands each initialized parent
                // inverse into the validated child range of the same tree.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&self.product_tree.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 1);
                }
                set_parameter(encoder, &expansion, 2);
                self.session.dispatch(
                    encoder,
                    &self.session.batch_affine.expand_product_inverses,
                    expansion.input_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.session.batch_affine.finish_pairs);
            // SAFETY: inverse expansion leaves one base inverse per pair and
            // the validated map assigns every output a unique destination.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.numerators.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.cases.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.product_tree.raw), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&chunk.destination_map.raw), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 6);
            }
            set_parameter(encoder, &parameters, 7);
            self.session.dispatch(
                encoder,
                &self.session.batch_affine.finish_pairs,
                level.pair_count as usize,
            );

            if level.carry_count != 0 {
                encoder.setComputePipelineState(&self.session.batch_affine.copy_carries);
                // SAFETY: topology validation covers the carry suffix, map
                // suffix, and disjoint destination range.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&chunk.destination_map.raw), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 3);
                }
                set_parameter(encoder, &parameters, 4);
                self.session.dispatch(
                    encoder,
                    &self.session.batch_affine.copy_carries,
                    level.carry_count as usize,
                );
            }

            source_slot ^= 1;
        }

        let root_source = if source_slot == 0 {
            &self.points_a
        } else {
            &self.points_b
        };
        encoder.setComputePipelineState(&self.session.batch_affine.export_roots);
        // SAFETY: nonempty-run validation makes root positions linear and the
        // per-chunk root allocation contains exactly `root_count` affine slots.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&root_source.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&chunk.roots.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 2);
        }
        set_parameter(encoder, &chunk.root_count, 3);
        self.session.dispatch(
            encoder,
            &self.session.batch_affine.export_roots,
            chunk.root_count as usize,
        );
    }
}

impl HostWorkload {
    fn build(srs_len: usize, chunks: &[BatchAffineBenchmarkChunk<'_>]) -> Result<Self, MetalError> {
        validate_srs_and_chunks(srs_len, chunks)?;
        let mut host_chunks = Vec::new();
        host_chunks
            .try_reserve_exact(chunks.len())
            .map_err(|_| MetalError::Dispatch("batch-affine host chunk allocation failed"))?;

        for chunk in chunks {
            let plan = compile_balanced_run_plan(chunk.run_lengths)
                .map_err(|_| MetalError::Dispatch("batch-affine topology compilation failed"))?;
            validate_chunk(srs_len, chunk.packed_records, &plan)?;
            let ordered_records = reorder_records(chunk.packed_records, &plan)?;
            let product_trees = product_tree_plans(&plan)?;
            let root_count = checked_u32(plan.run_lengths.len(), "root count")?;
            host_chunks.push(HostChunk {
                plan,
                product_trees,
                ordered_records: ordered_records.into_boxed_slice(),
                root_count,
            });
        }

        let (max_chunk_leaves, max_level_pairs, max_product_tree_slots) =
            maximum_extents(&host_chunks);
        let counters = benchmark_counters(
            srs_len,
            &host_chunks,
            max_chunk_leaves,
            max_level_pairs,
            max_product_tree_slots,
        )?;
        Ok(Self {
            chunks: host_chunks.into_boxed_slice(),
            max_chunk_leaves,
            max_level_pairs,
            max_product_tree_slots,
            counters,
        })
    }
}

fn validate_srs_and_chunks(
    srs_len: usize,
    chunks: &[BatchAffineBenchmarkChunk<'_>],
) -> Result<(), MetalError> {
    if srs_len == 0 || srs_len > PACKED_RECORD_INDEX_MASK as usize + 1 {
        return Err(MetalError::Dispatch(
            "batch-affine benchmark SRS length exceeds the packed record ABI",
        ));
    }
    if chunks.is_empty() {
        return Err(MetalError::Dispatch(
            "batch-affine benchmark requires at least one chunk",
        ));
    }
    if chunks
        .iter()
        .any(|chunk| chunk.run_lengths.is_empty() || chunk.run_lengths.contains(&0))
    {
        return Err(MetalError::Dispatch(
            "batch-affine benchmark requires nonempty runs and chunks",
        ));
    }
    Ok(())
}

fn validate_chunk(
    srs_len: usize,
    packed_records: &[u32],
    plan: &BatchAffineRunPlan,
) -> Result<(), MetalError> {
    validate_balanced_run_plan(plan)
        .map_err(|_| MetalError::Dispatch("batch-affine topology validation failed"))?;
    if packed_records.len() != plan.leaf_count as usize {
        return Err(MetalError::LengthMismatch {
            left: packed_records.len(),
            right: plan.leaf_count as usize,
        });
    }
    if packed_records
        .iter()
        .any(|record| (record & PACKED_RECORD_INDEX_MASK) as usize >= srs_len)
    {
        return Err(MetalError::Dispatch(
            "batch-affine benchmark record index exceeds the SRS",
        ));
    }
    if !is_linear(&plan.root_source_positions, plan.run_lengths.len())
        || !is_linear(&plan.root_run_indices, plan.run_lengths.len())
    {
        return Err(MetalError::Dispatch(
            "batch-affine benchmark requires linear nonempty roots",
        ));
    }
    Ok(())
}

fn is_linear(values: &[u32], expected_len: usize) -> bool {
    values.len() == expected_len
        && values
            .iter()
            .enumerate()
            .all(|(index, &value)| value as usize == index)
}

fn reorder_records(
    packed_records: &[u32],
    plan: &BatchAffineRunPlan,
) -> Result<Vec<u32>, MetalError> {
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(plan.initial_source_order.len())
        .map_err(|_| MetalError::Dispatch("batch-affine record-order allocation failed"))?;
    for &source_index in plan.initial_source_order.iter() {
        let record = packed_records
            .get(source_index as usize)
            .ok_or(MetalError::Dispatch(
                "batch-affine source order escaped the record input",
            ))?;
        ordered.push(*record);
    }
    Ok(ordered)
}

fn product_tree_plans(plan: &BatchAffineRunPlan) -> Result<Box<[ProductTreePlan]>, MetalError> {
    let mut trees = Vec::new();
    trees
        .try_reserve_exact(plan.levels.len())
        .map_err(|_| MetalError::Dispatch("batch-affine product-plan allocation failed"))?;
    for level in plan.levels.iter() {
        trees.push(product_tree_plan(level.pair_count)?);
    }
    Ok(trees.into_boxed_slice())
}

fn product_tree_plan(pair_count: u32) -> Result<ProductTreePlan, MetalError> {
    if pair_count == 0 {
        return Err(MetalError::Dispatch(
            "batch-affine reduction level has no pairs",
        ));
    }
    let mut reductions = Vec::new();
    reductions
        .try_reserve_exact(u32::BITS as usize)
        .map_err(|_| MetalError::Dispatch("batch-affine product-level allocation failed"))?;
    let mut input_offset = 0u32;
    let mut input_count = pair_count;
    let mut next_offset = pair_count;
    while input_count > 1 {
        let output_count = input_count.div_ceil(2);
        let output_offset = next_offset;
        next_offset = next_offset
            .checked_add(output_count)
            .ok_or(MetalError::Dispatch(
                "batch-affine product-tree offset overflow",
            ))?;
        reductions.push(BatchAffineProductParameters {
            input_offset,
            input_count,
            output_offset,
            output_count,
        });
        input_offset = output_offset;
        input_count = output_count;
    }
    Ok(ProductTreePlan {
        reductions: reductions.into_boxed_slice(),
        root: BatchAffineProductParameters {
            input_offset,
            input_count: 1,
            output_offset: input_offset,
            output_count: 1,
        },
        slots: next_offset,
    })
}

fn maximum_extents(chunks: &[HostChunk]) -> (u32, u32, u32) {
    let mut max_chunk_leaves = 0u32;
    let mut max_level_pairs = 0u32;
    let mut max_product_tree_slots = 0u32;
    for chunk in chunks {
        max_chunk_leaves = max_chunk_leaves.max(chunk.plan.leaf_count);
        for (level, tree) in chunk.plan.levels.iter().zip(&chunk.product_trees) {
            max_level_pairs = max_level_pairs.max(level.pair_count);
            max_product_tree_slots = max_product_tree_slots.max(tree.slots);
        }
    }
    (max_chunk_leaves, max_level_pairs, max_product_tree_slots)
}

fn benchmark_counters(
    srs_len: usize,
    chunks: &[HostChunk],
    max_chunk_leaves: u32,
    max_level_pairs: u32,
    max_product_tree_slots: u32,
) -> Result<BatchAffineBenchmarkCounters, MetalError> {
    let srs_bytes = byte_count::<Bls12_381G1Affine>(srs_len, "SRS bytes")?;
    let mut leaf_count = 0u64;
    let mut root_count = 0u64;
    let mut pair_reduction_count = 0u64;
    let mut carry_copy_count = 0u64;
    let mut level_count = 0u64;
    let mut dispatch_count = 1u64; // One GPU status clear for the whole sample.
    let mut logical_device_bytes = size_of::<BatchAffineStatus>() as u64;
    let mut immutable_chunk_bytes = 0u64;
    let mut root_output_bytes = 0u64;

    for chunk in chunks {
        leaf_count = checked_sum(leaf_count, u64::from(chunk.plan.leaf_count), "leaf count")?;
        root_count = checked_sum(root_count, u64::from(chunk.root_count), "root count")?;
        pair_reduction_count = checked_sum(
            pair_reduction_count,
            chunk.plan.total_pair_count,
            "pair count",
        )?;
        immutable_chunk_bytes = checked_sum(
            immutable_chunk_bytes,
            byte_count::<u32>(chunk.ordered_records.len(), "record bytes")?,
            "immutable bytes",
        )?;
        immutable_chunk_bytes = checked_sum(
            immutable_chunk_bytes,
            byte_count::<u32>(
                chunk.plan.destination_map.len().max(1),
                "destination-map bytes",
            )?,
            "immutable bytes",
        )?;
        root_output_bytes = checked_sum(
            root_output_bytes,
            byte_count::<Bls12_381G1Affine>(chunk.root_count as usize, "root bytes")?,
            "root-output bytes",
        )?;
        logical_device_bytes = checked_sum(
            logical_device_bytes,
            scaled_bytes(chunk.plan.leaf_count, 196, "gather traffic")?,
            "logical device bytes",
        )?;
        logical_device_bytes = checked_sum(
            logical_device_bytes,
            scaled_bytes(chunk.root_count, 192, "root-export traffic")?,
            "logical device bytes",
        )?;

        dispatch_count = checked_sum(dispatch_count, 2, "dispatch count")?; // gather + export
        for (level, tree) in chunk.plan.levels.iter().zip(&chunk.product_trees) {
            level_count = checked_sum(level_count, 1, "level count")?;
            carry_copy_count = checked_sum(
                carry_copy_count,
                u64::from(level.carry_count),
                "carry count",
            )?;
            logical_device_bytes = checked_sum(
                logical_device_bytes,
                scaled_bytes(level.pair_count, 684, "pair traffic")?,
                "logical device bytes",
            )?;
            logical_device_bytes = checked_sum(
                logical_device_bytes,
                scaled_bytes(level.carry_count, 196, "carry traffic")?,
                "logical device bytes",
            )?;
            // One 48-byte root read and write for the fixed-exponent inverse.
            logical_device_bytes = checked_sum(logical_device_bytes, 96, "logical device bytes")?;
            for reduction in tree.reductions.iter() {
                let reduction_elements = checked_sum(
                    u64::from(reduction.input_count),
                    u64::from(reduction.output_count),
                    "product-reduction traffic",
                )?;
                logical_device_bytes = checked_sum(
                    logical_device_bytes,
                    reduction_elements
                        .checked_mul(size_of::<BatchAffineFp32>() as u64)
                        .ok_or(MetalError::Dispatch("product-reduction byte overflow"))?,
                    "logical device bytes",
                )?;
                let expansion_elements = u64::from(reduction.output_count)
                    .checked_add(u64::from(reduction.input_count).saturating_mul(2))
                    .ok_or(MetalError::Dispatch("product-expansion traffic overflow"))?;
                logical_device_bytes = checked_sum(
                    logical_device_bytes,
                    expansion_elements
                        .checked_mul(size_of::<BatchAffineFp32>() as u64)
                        .ok_or(MetalError::Dispatch("product-expansion byte overflow"))?,
                    "logical device bytes",
                )?;
            }
            let tree_passes = u64::try_from(tree.reductions.len())
                .map_err(|_| MetalError::LengthOverflow(tree.reductions.len()))?;
            let level_dispatches = 3u64
                .checked_add(tree_passes.saturating_mul(2))
                .and_then(|value| value.checked_add(u64::from(level.carry_count != 0)))
                .ok_or(MetalError::Dispatch("batch-affine dispatch-count overflow"))?;
            dispatch_count = checked_sum(dispatch_count, level_dispatches, "dispatch count")?;
        }
    }

    let points_bytes =
        byte_count::<BatchAffinePoint32>(max_chunk_leaves as usize, "point arena bytes")?
            .checked_mul(2)
            .ok_or(MetalError::Dispatch("batch-affine point-byte overflow"))?;
    let numerator_bytes =
        byte_count::<BatchAffineFp32>(max_level_pairs.max(1) as usize, "numerator bytes")?;
    let case_bytes = byte_count::<u32>(max_level_pairs.max(1) as usize, "case bytes")?;
    let tree_bytes = byte_count::<BatchAffineFp32>(
        max_product_tree_slots.max(1) as usize,
        "product-tree bytes",
    )?;
    let status_bytes = byte_count::<BatchAffineStatus>(1, "status bytes")?;
    let shared_scratch_bytes = [
        points_bytes,
        numerator_bytes,
        case_bytes,
        tree_bytes,
        status_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| {
        checked_sum(total, bytes, "shared scratch bytes")
    })?;
    let allocated_bytes = [
        srs_bytes,
        immutable_chunk_bytes,
        shared_scratch_bytes,
        root_output_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| {
        checked_sum(total, bytes, "allocated bytes")
    })?;
    let host_upload_bytes = checked_sum(srs_bytes, immutable_chunk_bytes, "host upload bytes")?;
    let chunk_count =
        u64::try_from(chunks.len()).map_err(|_| MetalError::LengthOverflow(chunks.len()))?;
    let metal_allocation_count = chunk_count
        .checked_mul(3)
        .and_then(|value| value.checked_add(7))
        .ok_or(MetalError::Dispatch(
            "batch-affine allocation-count overflow",
        ))?;

    Ok(BatchAffineBenchmarkCounters {
        chunk_count,
        leaf_count,
        root_count,
        pair_reduction_count,
        carry_copy_count,
        level_count,
        dispatch_count,
        command_buffer_count: 1,
        host_wait_count: 1,
        metal_allocation_count,
        srs_bytes,
        immutable_chunk_bytes,
        shared_scratch_bytes,
        root_output_bytes,
        allocated_bytes,
        host_upload_bytes,
        host_status_read_bytes: status_bytes,
        logical_device_bytes,
        max_chunk_leaves: u64::from(max_chunk_leaves),
        max_level_pairs: u64::from(max_level_pairs),
        max_product_tree_slots: u64::from(max_product_tree_slots),
    })
}

/// Expands run lengths into deterministic, irregular signed packed SRS records.
///
/// This helper contains no raw diagnostic parser and emits no witness-derived
/// metadata. It is suitable for reconstructing topology-only benchmark inputs.
pub(super) fn synthetic_packed_records(
    run_lengths: &[usize],
    srs_len: usize,
    seed: u64,
) -> Result<Vec<u32>, MetalError> {
    if srs_len == 0 || srs_len > PACKED_RECORD_INDEX_MASK as usize + 1 {
        return Err(MetalError::Dispatch(
            "synthetic packed-record SRS length exceeds the ABI",
        ));
    }
    let record_count = run_lengths.iter().try_fold(0usize, |total, &length| {
        total
            .checked_add(length)
            .ok_or(MetalError::LengthOverflow(length))
    })?;
    checked_u32(record_count, "synthetic record count")?;
    let mut records = Vec::new();
    records
        .try_reserve_exact(record_count)
        .map_err(|_| MetalError::Dispatch("synthetic packed-record allocation failed"))?;
    let mut state = seed;
    for _ in 0..record_count {
        let random = splitmix64(&mut state);
        let index = (random % srs_len as u64) as u32;
        let sign = ((random >> 32) as u32) & PACKED_RECORD_NEGATE;
        records.push(index | sign);
    }
    Ok(records)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(super) fn checked_buffer<T>(
    session: &MetalSession,
    elements: u32,
    label: &str,
) -> Result<TrackedBuffer, MetalError> {
    let elements = elements as usize;
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::LengthOverflow(elements))?;
    session.buffer(bytes, label)
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| match field {
        "SRS length" | "root count" | "synthetic record count" => MetalError::LengthOverflow(value),
        _ => MetalError::Dispatch("batch-affine value exceeds the u32 ABI"),
    })
}

fn byte_count<T>(elements: usize, field: &'static str) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::LengthOverflow(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::Dispatch(field))
}

fn checked_sum(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, MetalError> {
    lhs.checked_add(rhs).ok_or(MetalError::Dispatch(field))
}

fn scaled_bytes(elements: u32, bytes: u64, field: &'static str) -> Result<u64, MetalError> {
    u64::from(elements)
        .checked_mul(bytes)
        .ok_or(MetalError::Dispatch(field))
}

trait BatchAffineParameter: Copy {}

impl BatchAffineParameter for u32 {}
impl BatchAffineParameter for BatchAffineGatherParameters {}
impl BatchAffineParameter for BatchAffineLevelParameters {}
impl BatchAffineParameter for BatchAffineProductParameters {}

fn set_parameter<T: BatchAffineParameter>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    parameter: &T,
    index: usize,
) {
    // SAFETY: every accepted parameter is a fully initialized repr(C) scalar
    // or ABI-probed four-u32 block copied by Metal during command encoding.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(parameter).cast::<c_void>(),
            size_of::<T>(),
            index,
        );
    }
}

pub(super) fn completed_gpu_duration(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Option<Duration> {
    // objc2-metal gates these two generated accessors behind its optional
    // CoreFoundation feature, but both selectors return the ABI-equivalent f64
    // CFTimeInterval and are part of MTLCommandBuffer on supported macOS. This
    // call occurs only after successful completion, as required by Metal.
    let start: f64 = unsafe { objc2::msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { objc2::msg_send![command, GPUEndTime] };
    let seconds = end - start;
    if start <= 0.0 || !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let nanoseconds = seconds * 1_000_000_000.0;
    if nanoseconds > u64::MAX as f64 {
        return None;
    }
    Some(Duration::from_nanos(nanoseconds.round() as u64))
}

pub(super) fn read_status(
    session: &MetalSession,
    buffer: &TrackedBuffer,
) -> Result<BatchAffineStatus, MetalError> {
    let requested = size_of::<BatchAffineStatus>();
    if requested > buffer.bytes {
        return Err(MetalError::BufferBounds {
            requested,
            available: buffer.bytes,
        });
    }
    // SAFETY: the completed GPU clear initializes all status bytes, every
    // subsequent status write uses u32 atomics, and the shared mapping is
    // aligned and contains one complete repr(C) status value.
    let status = unsafe {
        buffer
            .raw
            .contents()
            .as_ptr()
            .cast::<BatchAffineStatus>()
            .read()
    };
    session
        .activity
        .host_read_bytes
        .fetch_add(size_of_val(&status) as u64, Ordering::Relaxed);
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_workload_reorders_chunks_and_accounts_for_one_shared_arena() {
        let first_records = [10u32, 11, 12, 20, 21];
        let second_records = [7u32];
        let chunks = [
            BatchAffineBenchmarkChunk {
                run_lengths: &[3, 2],
                packed_records: &first_records,
            },
            BatchAffineBenchmarkChunk {
                run_lengths: &[1],
                packed_records: &second_records,
            },
        ];
        let workload = HostWorkload::build(32, &chunks).expect("compile host workload");

        assert_eq!(&*workload.chunks[0].ordered_records, &[10, 11, 20, 21, 12]);
        assert_eq!(&*workload.chunks[1].ordered_records, &[7]);
        assert_eq!(workload.max_chunk_leaves, 5);
        assert_eq!(workload.max_level_pairs, 2);
        assert_eq!(workload.max_product_tree_slots, 3);
        assert_eq!(workload.counters.chunk_count, 2);
        assert_eq!(workload.counters.leaf_count, 6);
        assert_eq!(workload.counters.root_count, 3);
        assert_eq!(workload.counters.pair_reduction_count, 3);
        assert_eq!(workload.counters.carry_copy_count, 2);
        assert_eq!(workload.counters.level_count, 2);
        assert_eq!(workload.counters.dispatch_count, 15);
        assert_eq!(workload.counters.command_buffer_count, 1);
        assert_eq!(workload.counters.host_wait_count, 1);
        assert_eq!(workload.counters.metal_allocation_count, 13);
        assert_eq!(
            workload.counters.allocated_bytes,
            workload.counters.srs_bytes
                + workload.counters.immutable_chunk_bytes
                + workload.counters.shared_scratch_bytes
                + workload.counters.root_output_bytes
        );
    }

    #[test]
    fn host_validation_rejects_every_chunk_before_device_preparation() {
        let valid_records = [0u32, 1];
        let invalid_records = [0u32, 8];
        let chunks = [
            BatchAffineBenchmarkChunk {
                run_lengths: &[2],
                packed_records: &valid_records,
            },
            BatchAffineBenchmarkChunk {
                run_lengths: &[2],
                packed_records: &invalid_records,
            },
        ];

        let error = match HostWorkload::build(8, &chunks) {
            Ok(_) => panic!("later invalid chunk must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, MetalError::Dispatch(_)));
    }

    #[test]
    fn synthetic_records_are_reproducible_signed_and_in_range() {
        let lengths = [3usize, 5, 89];
        let first =
            synthetic_packed_records(&lengths, 17, 0x0002_4302).expect("expand synthetic records");
        let second =
            synthetic_packed_records(&lengths, 17, 0x0002_4302).expect("repeat synthetic records");
        let different = synthetic_packed_records(&lengths, 17, 0x0002_4303)
            .expect("expand different synthetic records");

        assert_eq!(first, second);
        assert_ne!(first, different);
        assert_eq!(first.len(), lengths.into_iter().sum::<usize>());
        assert!(
            first
                .iter()
                .all(|record| (record & PACKED_RECORD_INDEX_MASK) < 17)
        );
        assert!(
            first
                .iter()
                .any(|record| record & PACKED_RECORD_NEGATE == 0)
        );
        assert!(
            first
                .iter()
                .any(|record| record & PACKED_RECORD_NEGATE != 0)
        );
    }

    #[test]
    fn empty_runs_chunks_and_srs_are_rejected() {
        assert!(HostWorkload::build(1, &[]).is_err());
        assert!(
            HostWorkload::build(
                1,
                &[BatchAffineBenchmarkChunk {
                    run_lengths: &[1, 0],
                    packed_records: &[0],
                }],
            )
            .is_err()
        );
        assert!(
            HostWorkload::build(
                0,
                &[BatchAffineBenchmarkChunk {
                    run_lengths: &[1],
                    packed_records: &[0],
                }],
            )
            .is_err()
        );
    }
}
