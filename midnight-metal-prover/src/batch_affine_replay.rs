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

use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{
    BatchAffineGatherParameters, BatchAffineLevelParameters, BatchAffinePoint32,
    BatchAffineProductParameters, BatchAffineStatus, MetalPod, MetalSession, PendingCommand,
    TrackedBuffer, pod,
};
use crate::{Bls12_381G1Affine, MetalError};

use super::batch_affine_probe::{BatchAffineRunPlan, validate_balanced_run_plan};

const PACKED_RECORD_NEGATE: u32 = 1 << 31;
const PACKED_RECORD_INDEX_MASK: u32 = PACKED_RECORD_NEGATE - 1;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct BatchAffineReplayResult {
    pub(super) roots: Vec<Bls12_381G1Affine>,
    pub(super) status: BatchAffineStatus,
}

struct ProductTreePlan {
    reductions: Box<[BatchAffineProductParameters]>,
    root: BatchAffineProductParameters,
    slots: u32,
}

impl pod::Sealed for BatchAffineStatus {}
// SAFETY: the repr(C) status contains exactly four `u32` words, has no padding,
// and every bit pattern is valid on both the host and Metal sides.
unsafe impl MetalPod for BatchAffineStatus {}

impl MetalSession {
    pub(super) fn replay_batch_affine(
        &self,
        srs: &[Bls12_381G1Affine],
        packed_records: &[u32],
        plan: &BatchAffineRunPlan,
    ) -> Result<BatchAffineReplayResult, MetalError> {
        validate_replay_inputs(srs, packed_records, plan)?;
        let ordered_records = reorder_records(packed_records, plan)?;
        let product_trees = product_tree_plans(plan)?;

        let max_pairs = plan
            .levels
            .iter()
            .map(|level| level.pair_count)
            .max()
            .unwrap_or(0);
        let max_tree_slots = product_trees
            .iter()
            .map(|tree| tree.slots)
            .max()
            .unwrap_or(0);
        let root_count = u32::try_from(plan.run_lengths.len())
            .map_err(|_| MetalError::LengthOverflow(plan.run_lengths.len()))?;
        let srs_count =
            u32::try_from(srs.len()).map_err(|_| MetalError::LengthOverflow(srs.len()))?;

        let _execution = self.execution_guard();
        let srs_buffer = self.buffer_from_slice(srs, "midnight.batch_affine.srs")?;
        let record_buffer =
            self.buffer_from_slice(&ordered_records, "midnight.batch_affine.preordered_records")?;
        let dummy_map = [0u32];
        let destination_map_values = if plan.destination_map.is_empty() {
            &dummy_map[..]
        } else {
            &plan.destination_map
        };
        let destination_map = self.buffer_from_slice(
            destination_map_values,
            "midnight.batch_affine.destination_map",
        )?;
        let points_a = checked_buffer::<BatchAffinePoint32>(
            self,
            plan.leaf_count,
            "midnight.batch_affine.points_a",
        )?;
        let points_b = checked_buffer::<BatchAffinePoint32>(
            self,
            plan.leaf_count,
            "midnight.batch_affine.points_b",
        )?;
        let numerators = checked_buffer::<super::BatchAffineFp32>(
            self,
            max_pairs.max(1),
            "midnight.batch_affine.numerators",
        )?;
        let cases = checked_buffer::<u32>(self, max_pairs.max(1), "midnight.batch_affine.cases")?;
        let product_tree = checked_buffer::<super::BatchAffineFp32>(
            self,
            max_tree_slots.max(1),
            "midnight.batch_affine.product_tree",
        )?;
        let status = checked_buffer::<BatchAffineStatus>(self, 1, "midnight.batch_affine.status")?;
        let public_roots = checked_buffer::<Bls12_381G1Affine>(
            self,
            root_count,
            "midnight.batch_affine.public_roots",
        )?;

        let gather = BatchAffineGatherParameters {
            record_offset: 0,
            record_count: plan.leaf_count,
            srs_count,
            reserved: 0,
        };
        let command = self.command_buffer("midnight.batch_affine.correctness_replay")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.clear_words);
        // SAFETY: the retained status allocation contains two aligned `u64`
        // words, and this dispatch completes before later encoder commands use it.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.clear_words, 2);

        encoder.setComputePipelineState(&self.batch_affine.gather_signed_leaves);
        // SAFETY: the retained buffers have the exact public-affine, packed-u32,
        // internal-point, and status layouts required by the gather kernel.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&srs_buffer.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&record_buffer.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&points_a.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 3);
        }
        set_parameter(&encoder, &gather, 4);
        self.dispatch(
            &encoder,
            &self.batch_affine.gather_signed_leaves,
            plan.leaf_count as usize,
        );

        let mut source_slot = 0usize;
        for ((level, tree), level_index) in plan.levels.iter().zip(&product_trees).zip(0usize..) {
            let parameters = BatchAffineLevelParameters {
                destination_map_offset: level.destination_map_offset,
                pair_count: level.pair_count,
                carry_count: level.carry_count,
                destination_len: level.destination_count,
            };
            let (source, destination) = if source_slot == 0 {
                (&points_a, &points_b)
            } else {
                (&points_b, &points_a)
            };

            encoder.setComputePipelineState(&self.batch_affine.prepare_pairs);
            // SAFETY: topology validation proves that the pair-major source and
            // scratch extents cover this level, and every resource is retained.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&numerators.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&cases.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&product_tree.raw), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 4);
            }
            set_parameter(&encoder, &parameters, 5);
            self.dispatch(
                &encoder,
                &self.batch_affine.prepare_pairs,
                level.pair_count as usize,
            );

            for reduction in tree.reductions.iter() {
                encoder.setComputePipelineState(&self.batch_affine.reduce_products);
                // SAFETY: the checked concatenated-tree offsets cover every
                // input and output element used by this reduction pass.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&product_tree.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 1);
                }
                set_parameter(&encoder, reduction, 2);
                self.dispatch(
                    &encoder,
                    &self.batch_affine.reduce_products,
                    reduction.output_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.batch_affine.invert_product_root);
            // SAFETY: `tree.root` identifies the single initialized product root
            // in the retained in-place product allocation.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&product_tree.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 1);
            }
            set_parameter(&encoder, &tree.root, 2);
            self.dispatch(&encoder, &self.batch_affine.invert_product_root, 1);

            for reduction in tree.reductions.iter().rev() {
                let expansion = BatchAffineProductParameters {
                    input_offset: reduction.output_offset,
                    input_count: reduction.output_count,
                    output_offset: reduction.input_offset,
                    output_count: reduction.input_count,
                };
                encoder.setComputePipelineState(&self.batch_affine.expand_product_inverses);
                // SAFETY: reversing a validated child-to-parent reduction gives
                // an in-bounds parent-to-child inverse expansion in the same tree.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&product_tree.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 1);
                }
                set_parameter(&encoder, &expansion, 2);
                self.dispatch(
                    &encoder,
                    &self.batch_affine.expand_product_inverses,
                    expansion.input_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.batch_affine.finish_pairs);
            // SAFETY: the inverse expansion leaves one inverse per pair at the
            // tree base; the validated map gives each pair a unique destination.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&numerators.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&cases.raw), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&product_tree.raw), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&destination_map.raw), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 6);
            }
            set_parameter(&encoder, &parameters, 7);
            self.dispatch(
                &encoder,
                &self.batch_affine.finish_pairs,
                level.pair_count as usize,
            );

            if level.carry_count != 0 {
                encoder.setComputePipelineState(&self.batch_affine.copy_carries);
                // SAFETY: topology validation proves the carry suffix and its
                // disjoint destination-map suffix are both in bounds.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&destination_map.raw), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 3);
                }
                set_parameter(&encoder, &parameters, 4);
                self.dispatch(
                    &encoder,
                    &self.batch_affine.copy_carries,
                    level.carry_count as usize,
                );
            }

            source_slot = (level_index + 1) & 1;
        }

        let root_source = if source_slot == 0 {
            &points_a
        } else {
            &points_b
        };
        encoder.setComputePipelineState(&self.batch_affine.export_roots);
        // SAFETY: replay validation requires linear root positions and indices,
        // and both retained buffers contain exactly `root_count` point slots.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&root_source.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&public_roots.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 2);
        }
        set_parameter(&encoder, &root_count, 3);
        self.dispatch(
            &encoder,
            &self.batch_affine.export_roots,
            root_count as usize,
        );
        encoder.endEncoding();

        let mut resources = Vec::new();
        resources.try_reserve_exact(10).map_err(|_| {
            MetalError::Dispatch("batch-affine resource retention allocation failed")
        })?;
        resources.extend([
            srs_buffer.raw.clone(),
            record_buffer.raw.clone(),
            destination_map.raw.clone(),
            points_a.raw.clone(),
            points_b.raw.clone(),
            numerators.raw.clone(),
            cases.raw.clone(),
            product_tree.raw.clone(),
            status.raw.clone(),
            public_roots.raw.clone(),
        ]);
        self.submit_and_wait(PendingCommand {
            command,
            _resources: resources,
        })?;

        let roots = self.read_values(&public_roots, root_count as usize)?;
        let status = self
            .read_values::<BatchAffineStatus>(&status, 1)?
            .into_iter()
            .next()
            .ok_or(MetalError::BufferBounds {
                requested: size_of::<BatchAffineStatus>(),
                available: status.bytes,
            })?;
        Ok(BatchAffineReplayResult { roots, status })
    }
}

fn validate_replay_inputs(
    srs: &[Bls12_381G1Affine],
    packed_records: &[u32],
    plan: &BatchAffineRunPlan,
) -> Result<(), MetalError> {
    validate_balanced_run_plan(plan)
        .map_err(|_| MetalError::Dispatch("batch-affine topology validation failed"))?;
    if plan.run_lengths.is_empty() || plan.run_lengths.contains(&0) {
        return Err(MetalError::Dispatch(
            "batch-affine replay requires nonempty runs",
        ));
    }
    if srs.is_empty() || srs.len() > (PACKED_RECORD_INDEX_MASK as usize + 1) {
        return Err(MetalError::Dispatch(
            "batch-affine SRS length exceeds the packed record index ABI",
        ));
    }
    if packed_records.len() != plan.leaf_count as usize {
        return Err(MetalError::LengthMismatch {
            left: packed_records.len(),
            right: plan.leaf_count as usize,
        });
    }
    if packed_records
        .iter()
        .any(|record| (record & PACKED_RECORD_INDEX_MASK) as usize >= srs.len())
    {
        return Err(MetalError::Dispatch(
            "batch-affine record index exceeds the SRS",
        ));
    }
    let root_count = plan.run_lengths.len();
    if !is_linear(&plan.root_source_positions, root_count)
        || !is_linear(&plan.root_run_indices, root_count)
    {
        return Err(MetalError::Dispatch(
            "batch-affine export requires linear root positions and run indices",
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

fn product_tree_plans(plan: &BatchAffineRunPlan) -> Result<Vec<ProductTreePlan>, MetalError> {
    let mut trees = Vec::new();
    trees
        .try_reserve_exact(plan.levels.len())
        .map_err(|_| MetalError::Dispatch("batch-affine product-plan allocation failed"))?;
    for level in plan.levels.iter() {
        trees.push(product_tree_plan(level.pair_count)?);
    }
    Ok(trees)
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

fn checked_buffer<T>(
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
    // SAFETY: every call supplies a fully initialized repr(C) scalar or one of
    // the ABI-probed four-u32 parameter blocks. Metal copies these bytes during
    // command encoding, before the borrowed stack value can be dropped.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(parameter).cast::<c_void>(),
            size_of::<T>(),
            index,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::batch_affine_probe::compile_balanced_run_plan;

    #[test]
    fn one_command_batch_affine_replay_matches_blst_exceptional_cases() {
        let identity = Bls12_381G1Affine::default();
        let generator = scalar_multiple(1);
        let double_generator = scalar_multiple(2);
        let triple_generator = scalar_multiple(3);
        let srs = [
            identity,
            raw_affine(generator),
            raw_affine(double_generator),
            raw_affine(triple_generator),
        ];
        let run_lengths = [2usize, 2, 2, 2, 2, 2, 1, 3, 3];
        let packed_records = [
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            2,
            1,
            1,
            1,
            PACKED_RECORD_NEGATE | 1,
            PACKED_RECORD_NEGATE | 2,
            1,
            2,
            3,
            1,
            PACKED_RECORD_NEGATE | 1,
            2,
        ];
        let plan = compile_balanced_run_plan(&run_lengths).expect("compile replay topology");
        let expected = blst_run_sums(&srs, &packed_records, &run_lengths);

        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let actual = session
            .replay_batch_affine(&srs, &packed_records, &plan)
            .expect("execute one-command batch-affine replay");

        assert_eq!(actual.status, BatchAffineStatus::default());
        assert_eq!(actual.roots, expected);
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.host_waits, 1);
        assert_eq!(activity.execution_failures, 0);
        assert_eq!(activity.blit_bytes, 0);
        assert_eq!(
            activity.host_read_bytes,
            (expected.len() * size_of::<Bls12_381G1Affine>() + size_of::<BatchAffineStatus>())
                as u64
        );
    }

    #[test]
    fn all_singleton_replay_matches_blst_without_a_product_tree() {
        let srs = [
            Bls12_381G1Affine::default(),
            raw_affine(scalar_multiple(1)),
            raw_affine(scalar_multiple(2)),
            raw_affine(scalar_multiple(3)),
        ];
        let run_lengths = [1usize; 6];
        let packed_records = [
            0,
            1,
            PACKED_RECORD_NEGATE | 1,
            2,
            PACKED_RECORD_NEGATE | 3,
            3,
        ];
        let plan = compile_balanced_run_plan(&run_lengths).expect("compile singleton topology");
        assert!(plan.levels.is_empty());
        assert_eq!(plan.total_pair_count, 0);
        let expected = blst_run_sums(&srs, &packed_records, &run_lengths);

        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let actual = session
            .replay_batch_affine(&srs, &packed_records, &plan)
            .expect("execute all-singleton batch-affine replay");

        assert_eq!(actual.status, BatchAffineStatus::default());
        assert_eq!(actual.roots, expected);
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.host_waits, 1);
    }

    #[test]
    fn product_tree_inversion_boundaries_match_blst() {
        const PAIR_COUNTS: [usize; 16] = [
            1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        ];
        let srs = [
            raw_affine(scalar_multiple(5)),
            raw_affine(scalar_multiple(9)),
        ];
        let session = MetalSession::new().expect("initialize Metal runtime");

        for pair_count in PAIR_COUNTS {
            let run_lengths = [pair_count * 2];
            let packed_records = [0u32, 1]
                .into_iter()
                .cycle()
                .take(run_lengths[0])
                .collect::<Vec<_>>();
            let plan = compile_balanced_run_plan(&run_lengths)
                .unwrap_or_else(|error| panic!("compile {pair_count}-pair topology: {error}"));
            assert_eq!(plan.levels[0].pair_count as usize, pair_count);
            let expected = blst_run_sums(&srs, &packed_records, &run_lengths);

            session.reset_activity();
            let actual = session
                .replay_batch_affine(&srs, &packed_records, &plan)
                .unwrap_or_else(|error| panic!("replay {pair_count}-pair topology: {error}"));

            assert_eq!(actual.status, BatchAffineStatus::default());
            assert_eq!(actual.roots, expected, "pair count {pair_count}");
            let activity = session.activity();
            assert_eq!(activity.command_buffers, 1, "pair count {pair_count}");
            assert_eq!(activity.host_waits, 1, "pair count {pair_count}");
        }
    }

    #[test]
    fn deterministic_ragged_signed_subgroup_runs_match_blst() {
        const SRS_LEN: usize = 67;
        let mut state = 0x4254_4348_4146_4649;
        let srs = (0..SRS_LEN)
            .map(|_| raw_affine(scalar_multiple(splitmix64(&mut state) | 1)))
            .collect::<Vec<_>>();
        let run_lengths = [1usize, 3, 5, 7, 11, 17, 33, 65, 129, 257, 513, 16_797];
        let leaf_count = run_lengths.iter().sum::<usize>();
        let packed_records = (0..leaf_count)
            .map(|_| {
                let index = (splitmix64(&mut state) % SRS_LEN as u64) as u32;
                let sign = if splitmix64(&mut state) & 1 == 0 {
                    0
                } else {
                    PACKED_RECORD_NEGATE
                };
                sign | index
            })
            .collect::<Vec<_>>();
        let plan = compile_balanced_run_plan(&run_lengths).expect("compile ragged topology");
        assert_eq!(plan.run_lengths.last(), Some(&16_797));
        let expected = blst_run_sums(&srs, &packed_records, &run_lengths);

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .replay_batch_affine(&srs, &packed_records, &plan)
            .expect("execute ragged batch-affine replay");

        assert_eq!(actual.status, BatchAffineStatus::default());
        assert_eq!(actual.roots, expected);
    }

    #[test]
    fn authoritative_on_curve_non_subgroup_runs_match_blst() {
        let non_subgroup = authoritative_non_subgroup_affine();
        // SAFETY: `non_subgroup` contains initialized Fp coordinates produced by
        // BLST's square-root routine and accepted by its on-curve predicate.
        unsafe {
            assert!(blst::blst_p1_affine_on_curve(&non_subgroup));
            assert!(!blst::blst_p1_affine_in_g1(&non_subgroup));
        }
        let srs = [
            Bls12_381G1Affine::default(),
            raw_affine(scalar_multiple(1)),
            raw_blst_affine(non_subgroup),
            raw_affine(scalar_multiple(2)),
        ];
        let run_lengths = [2usize, 2, 2, 2, 3, 5];
        let packed_records = [
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            PACKED_RECORD_NEGATE | 2,
            0,
            2,
            0,
            2,
            PACKED_RECORD_NEGATE | 1,
            3,
            PACKED_RECORD_NEGATE | 2,
            1,
        ];
        let plan = compile_balanced_run_plan(&run_lengths)
            .expect("compile authoritative non-subgroup topology");
        let expected = blst_run_sums(&srs, &packed_records, &run_lengths);

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .replay_batch_affine(&srs, &packed_records, &plan)
            .expect("execute authoritative non-subgroup replay");

        assert_eq!(actual.status, BatchAffineStatus::default());
        assert_eq!(actual.roots, expected);
    }

    #[test]
    fn out_of_range_packed_record_is_rejected_before_commands_or_waits() {
        let srs = [raw_affine(scalar_multiple(1))];
        let packed_records = [PACKED_RECORD_NEGATE | 1];
        let plan = compile_balanced_run_plan(&[1]).expect("compile invalid-record topology");
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();

        let error = session
            .replay_batch_affine(&srs, &packed_records, &plan)
            .expect_err("out-of-range packed record must fail on the host");

        assert!(matches!(
            error,
            MetalError::Dispatch("batch-affine record index exceeds the SRS")
        ));
        let activity = session.activity();
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
        assert_eq!(activity.host_waits, 0);
        assert_eq!(activity.allocations, 0);
    }

    fn blst_run_sums(
        srs: &[Bls12_381G1Affine],
        packed_records: &[u32],
        run_lengths: &[usize],
    ) -> Vec<Bls12_381G1Affine> {
        let mut roots = Vec::with_capacity(run_lengths.len());
        let mut offset = 0usize;
        for &run_length in run_lengths {
            let mut accumulator = blst::blst_p1::default();
            for &record in &packed_records[offset..offset + run_length] {
                let source_index = (record & PACKED_RECORD_INDEX_MASK) as usize;
                let affine = blst::blst_p1_affine {
                    x: blst::blst_fp {
                        l: srs[source_index].x,
                    },
                    y: blst::blst_fp {
                        l: srs[source_index].y,
                    },
                };
                let mut point = blst::blst_p1::default();
                // SAFETY: the raw affine coordinates and distinct projective
                // destination are initialized and remain valid for this call.
                unsafe {
                    blst::blst_p1_from_affine(&mut point, &affine);
                    blst::blst_p1_cneg(&mut point, record & PACKED_RECORD_NEGATE != 0);
                }
                let mut sum = blst::blst_p1::default();
                // SAFETY: both initialized operands and the distinct result
                // storage remain valid for the complete BLST addition call.
                unsafe {
                    blst::blst_p1_add_or_double(&mut sum, &accumulator, &point);
                }
                accumulator = sum;
            }
            roots.push(raw_affine(accumulator));
            offset += run_length;
        }
        roots
    }

    fn scalar_multiple(scalar: u64) -> blst::blst_p1 {
        let scalar = scalar.to_le_bytes();
        let mut output = blst::blst_p1::default();
        // SAFETY: BLST's process-lifetime generator and the initialized scalar
        // bytes remain valid, and `output` is a distinct destination.
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
        // SAFETY: the initialized projective input and distinct affine output
        // remain valid for BLST's canonical conversion.
        unsafe {
            blst::blst_p1_to_affine(&mut affine, &point);
        }
        Bls12_381G1Affine {
            x: affine.x.l,
            y: affine.y.l,
        }
    }

    fn raw_blst_affine(point: blst::blst_p1_affine) -> Bls12_381G1Affine {
        Bls12_381G1Affine {
            x: point.x.l,
            y: point.y.l,
        }
    }

    fn fp_from_u64(value: u64) -> blst::blst_fp {
        let canonical = [value, 0, 0, 0, 0, 0];
        let mut output = blst::blst_fp::default();
        // SAFETY: `canonical` has the six little-endian limbs BLST expects and
        // `output` is valid, aligned, and disjoint from the immutable input.
        unsafe {
            blst::blst_fp_from_uint64(&mut output, canonical.as_ptr());
        }
        output
    }

    fn authoritative_non_subgroup_affine() -> blst::blst_p1_affine {
        // This mirrors vendored midnight-curves 0.3.1
        // `g1_non_torsion_point`: scan x from one, solve y^2=x^3+4, require
        // BLST's on-curve predicate, and reject its prime-subgroup predicate.
        let four = fp_from_u64(4);
        for x_value in 1..=u64::MAX {
            let x = fp_from_u64(x_value);
            let mut x_squared = blst::blst_fp::default();
            let mut x_cubed = blst::blst_fp::default();
            let mut rhs = blst::blst_fp::default();
            let mut y = blst::blst_fp::default();
            // SAFETY: every Fp input is initialized and each output is distinct
            // from its immutable inputs for the complete BLST operation.
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

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
