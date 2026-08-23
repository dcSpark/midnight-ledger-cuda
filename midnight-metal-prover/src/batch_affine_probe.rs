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

use std::fmt;

/// Host topology for one balanced reduction level.
///
/// The source point slot contains all pair operands first, two adjacent values
/// per pair, followed by the singleton carries. Pair and carry kernels emit in
/// that same order. Their shared map slice permutes those outputs into the next
/// level's identical layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchAffineLevelPlan {
    pub(super) input_count: u32,
    pub(super) pair_count: u32,
    pub(super) carry_count: u32,
    pub(super) destination_count: u32,
    pub(super) destination_map_offset: u32,
}

/// Complete compact topology for reducing independent nonempty affine runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BatchAffineRunPlan {
    /// Original run lengths, including empty runs.
    pub(super) run_lengths: Box<[u32]>,
    /// Number of leaves across every run.
    pub(super) leaf_count: u32,
    /// Maps level-zero source positions to the caller's flat leaf order.
    pub(super) initial_source_order: Box<[u32]>,
    /// Reduction levels, excluding the terminal all-root state.
    pub(super) levels: Box<[BatchAffineLevelPlan]>,
    /// Concatenated output-to-next-source permutations for every level.
    pub(super) destination_map: Box<[u32]>,
    /// Final point-slot positions to export, in [`Self::root_run_indices`] order.
    pub(super) root_source_positions: Box<[u32]>,
    /// Original run index for each exported root; empty runs are omitted.
    pub(super) root_run_indices: Box<[u32]>,
    /// Exact number of affine pair reductions across every level.
    pub(super) total_pair_count: u64,
}

/// Rejected topology input or corrupted compiled plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BatchAffinePlanError {
    ArithmeticOverflow(&'static str),
    ValueTooLarge {
        field: &'static str,
        value: usize,
    },
    Allocation(&'static str),
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    OutOfRange {
        field: &'static str,
        index: usize,
        value: u32,
        limit: u32,
    },
    Collision {
        field: &'static str,
        index: usize,
        value: u32,
    },
    Hole {
        field: &'static str,
        value: u32,
    },
    TopologyMismatch {
        field: &'static str,
        index: usize,
        expected: u32,
        actual: u32,
    },
}

impl fmt::Display for BatchAffinePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow(field) => write!(formatter, "{field} arithmetic overflow"),
            Self::ValueTooLarge { field, value } => {
                write!(formatter, "{field} value {value} does not fit the GPU ABI")
            }
            Self::Allocation(field) => write!(formatter, "could not allocate {field}"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} length mismatch: expected {expected}, found {actual}"
            ),
            Self::OutOfRange {
                field,
                index,
                value,
                limit,
            } => write!(formatter, "{field}[{index}]={value} is outside 0..{limit}"),
            Self::Collision {
                field,
                index,
                value,
            } => write!(formatter, "{field}[{index}] repeats destination {value}"),
            Self::Hole { field, value } => {
                write!(formatter, "{field} does not contain destination {value}")
            }
            Self::TopologyMismatch {
                field,
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "{field}[{index}] expected {expected}, found {actual}"
            ),
        }
    }
}

impl std::error::Error for BatchAffinePlanError {}

/// Compiles run lengths into pair-major source layouts and exact permutations.
pub(super) fn compile_balanced_run_plan(
    run_lengths: &[usize],
) -> Result<BatchAffineRunPlan, BatchAffinePlanError> {
    checked_u32(run_lengths.len(), "run count")?;
    let mut lengths = Vec::new();
    lengths
        .try_reserve_exact(run_lengths.len())
        .map_err(|_| BatchAffinePlanError::Allocation("run lengths"))?;

    let mut leaf_count = 0u32;
    for &length in run_lengths {
        let length = checked_u32(length, "run length")?;
        leaf_count = leaf_count
            .checked_add(length)
            .ok_or(BatchAffinePlanError::ArithmeticOverflow("leaf count"))?;
        lengths.push(length);
    }

    let mut root_run_indices = Vec::new();
    root_run_indices
        .try_reserve_exact(lengths.len())
        .map_err(|_| BatchAffinePlanError::Allocation("root run indices"))?;
    for (index, &length) in lengths.iter().enumerate() {
        if length != 0 {
            root_run_indices.push(checked_u32(index, "root run index")?);
        }
    }

    let initial_source_order = initial_source_order(&lengths, leaf_count)?;
    let mut counts = lengths.clone();
    let mut levels = Vec::new();
    let mut destination_map = Vec::new();
    let mut total_pair_count = 0u64;

    while counts.iter().any(|&count| count > 1) {
        let destination_map_offset = checked_u32(destination_map.len(), "destination map offset")?;
        let (level, level_map, next_counts) = derive_level(&counts, destination_map_offset)?;
        destination_map
            .try_reserve(level_map.len())
            .map_err(|_| BatchAffinePlanError::Allocation("destination maps"))?;
        destination_map.extend_from_slice(&level_map);
        checked_u32(destination_map.len(), "destination map length")?;

        levels
            .try_reserve(1)
            .map_err(|_| BatchAffinePlanError::Allocation("level plans"))?;
        levels.push(level);
        total_pair_count = total_pair_count
            .checked_add(u64::from(level.pair_count))
            .ok_or(BatchAffinePlanError::ArithmeticOverflow("total pair count"))?;
        counts = next_counts;
    }

    let root_count = checked_u32(root_run_indices.len(), "root count")?;
    let root_source_positions = (0..root_count).collect::<Vec<_>>();
    let expected_pairs = u64::from(leaf_count)
        .checked_sub(u64::from(root_count))
        .ok_or(BatchAffinePlanError::ArithmeticOverflow(
            "leaf/root pair identity",
        ))?;
    if total_pair_count != expected_pairs {
        return Err(BatchAffinePlanError::ArithmeticOverflow(
            "balanced pair-count identity",
        ));
    }

    let plan = BatchAffineRunPlan {
        run_lengths: lengths.into_boxed_slice(),
        leaf_count,
        initial_source_order: initial_source_order.into_boxed_slice(),
        levels: levels.into_boxed_slice(),
        destination_map: destination_map.into_boxed_slice(),
        root_source_positions: root_source_positions.into_boxed_slice(),
        root_run_indices: root_run_indices.into_boxed_slice(),
        total_pair_count,
    };
    validate_balanced_run_plan(&plan)?;
    Ok(plan)
}

/// Validates both structural permutations and the run-derived topology.
pub(super) fn validate_balanced_run_plan(
    plan: &BatchAffineRunPlan,
) -> Result<(), BatchAffinePlanError> {
    let mut leaf_count = 0u32;
    for &length in &plan.run_lengths {
        leaf_count = checked_add(leaf_count, length, "validated leaf count")?;
    }
    if leaf_count != plan.leaf_count {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "leaf count",
            expected: leaf_count as usize,
            actual: plan.leaf_count as usize,
        });
    }
    validate_permutation(
        "initial source order",
        &plan.initial_source_order,
        leaf_count,
    )?;
    let expected_initial_order = initial_source_order(&plan.run_lengths, leaf_count)?;
    compare_values(
        "initial source order",
        &plan.initial_source_order,
        &expected_initial_order,
    )?;

    let mut expected_roots = Vec::new();
    for (index, &length) in plan.run_lengths.iter().enumerate() {
        if length != 0 {
            expected_roots.push(checked_u32(index, "root run index")?);
        }
    }
    compare_values("root run indices", &plan.root_run_indices, &expected_roots)?;
    let expected_root_count = checked_u32(expected_roots.len(), "root count")?;
    let expected_root_positions = (0..expected_root_count).collect::<Vec<_>>();
    compare_values(
        "root source positions",
        &plan.root_source_positions,
        &expected_root_positions,
    )?;

    let mut map_end = 0usize;
    let mut counts = plan.run_lengths.to_vec();
    let mut total_pairs = 0u64;
    let mut level_index = 0usize;
    while counts.iter().any(|&count| count > 1) {
        let level = plan
            .levels
            .get(level_index)
            .ok_or(BatchAffinePlanError::LengthMismatch {
                field: "level plans",
                expected: level_index + 1,
                actual: plan.levels.len(),
            })?;
        let map_offset = checked_u32(map_end, "validated destination map offset")?;
        let (expected_level, expected_map, next_counts) = derive_level(&counts, map_offset)?;
        compare_level(level_index, *level, expected_level)?;

        let requested_end = map_end.checked_add(expected_map.len()).ok_or(
            BatchAffinePlanError::ArithmeticOverflow("validated destination map end"),
        )?;
        if map_end > plan.destination_map.len() {
            return Err(BatchAffinePlanError::Hole {
                field: "level destination map",
                value: 0,
            });
        }
        let available_end = requested_end.min(plan.destination_map.len());
        let actual_map = &plan.destination_map[map_end..available_end];
        validate_permutation(
            "level destination map",
            actual_map,
            expected_level.destination_count,
        )?;
        compare_values("level destination map", actual_map, &expected_map)?;
        map_end = requested_end;
        total_pairs = total_pairs
            .checked_add(u64::from(expected_level.pair_count))
            .ok_or(BatchAffinePlanError::ArithmeticOverflow(
                "validated total pair count",
            ))?;
        counts = next_counts;
        level_index += 1;
    }
    if plan.levels.len() != level_index {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "level plans",
            expected: level_index,
            actual: plan.levels.len(),
        });
    }

    if map_end != plan.destination_map.len() {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "destination map",
            expected: map_end,
            actual: plan.destination_map.len(),
        });
    }
    let terminal_count = counts.iter().try_fold(0u32, |total, &count| {
        checked_add(total, count, "terminal root count")
    })?;
    if terminal_count != expected_root_count {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "terminal roots",
            expected: expected_roots.len(),
            actual: terminal_count as usize,
        });
    }
    if total_pairs != plan.total_pair_count {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "total pair count",
            expected: total_pairs as usize,
            actual: plan.total_pair_count as usize,
        });
    }
    let expected_pairs = u64::from(leaf_count)
        .checked_sub(expected_roots.len() as u64)
        .ok_or(BatchAffinePlanError::ArithmeticOverflow(
            "validated leaf/root pair identity",
        ))?;
    if plan.total_pair_count != expected_pairs {
        return Err(BatchAffinePlanError::LengthMismatch {
            field: "pair-count identity",
            expected: expected_pairs as usize,
            actual: plan.total_pair_count as usize,
        });
    }
    Ok(())
}

fn derive_level(
    counts: &[u32],
    destination_map_offset: u32,
) -> Result<(BatchAffineLevelPlan, Vec<u32>, Vec<u32>), BatchAffinePlanError> {
    let mut input_count = 0u32;
    let mut pair_count = 0u32;
    let mut carry_count = 0u32;
    let mut next_pair_inputs = 0u32;
    let mut next_carries = 0u32;
    let mut next_counts = Vec::new();
    next_counts
        .try_reserve_exact(counts.len())
        .map_err(|_| BatchAffinePlanError::Allocation("next run counts"))?;

    for &count in counts {
        input_count = checked_add(input_count, count, "level input count")?;
        pair_count = checked_add(pair_count, count / 2, "level pair count")?;
        carry_count = checked_add(carry_count, count & 1, "level carry count")?;
        let next_count = checked_add(count / 2, count & 1, "next run count")?;
        let paired_next_nodes = checked_mul(next_count / 2, 2, "next paired-node count")?;
        next_pair_inputs =
            checked_add(next_pair_inputs, paired_next_nodes, "next pair-input count")?;
        next_carries = checked_add(next_carries, next_count & 1, "next carry count")?;
        next_counts.push(next_count);
    }

    let destination_count = checked_add(pair_count, carry_count, "destination count")?;
    let expected_destination_count =
        checked_add(next_pair_inputs, next_carries, "next destination count")?;
    if destination_count != expected_destination_count {
        return Err(BatchAffinePlanError::ArithmeticOverflow(
            "balanced destination identity",
        ));
    }
    let expected_input_count = checked_add(
        checked_mul(pair_count, 2, "paired source count")?,
        carry_count,
        "source layout count",
    )?;
    if input_count != expected_input_count {
        return Err(BatchAffinePlanError::ArithmeticOverflow(
            "balanced source identity",
        ));
    }

    let mut level_map = Vec::new();
    level_map
        .try_reserve_exact(destination_count as usize)
        .map_err(|_| BatchAffinePlanError::Allocation("level destination map"))?;
    level_map.resize(destination_count as usize, u32::MAX);

    let mut emitted_pair = 0u32;
    let mut emitted_carry = pair_count;
    let mut target_pair = 0u32;
    let mut target_carry = next_pair_inputs;
    for (&count, &next_count) in counts.iter().zip(&next_counts) {
        let current_pairs = count / 2;
        let current_has_carry = count & 1;
        let paired_next_nodes = checked_mul(next_count / 2, 2, "paired next nodes")?;

        for logical_index in 0..next_count {
            let emitted = if logical_index < current_pairs {
                checked_add(emitted_pair, logical_index, "pair emission index")?
            } else {
                debug_assert_eq!(current_has_carry, 1);
                emitted_carry
            };
            let target = if logical_index < paired_next_nodes {
                checked_add(target_pair, logical_index, "paired destination index")?
            } else {
                target_carry
            };
            level_map[emitted as usize] = target;
        }

        emitted_pair = checked_add(emitted_pair, current_pairs, "pair emission cursor")?;
        if current_has_carry != 0 {
            emitted_carry = checked_add(emitted_carry, 1, "carry emission cursor")?;
        }
        target_pair = checked_add(target_pair, paired_next_nodes, "pair target cursor")?;
        if next_count & 1 != 0 {
            target_carry = checked_add(target_carry, 1, "carry target cursor")?;
        }
    }

    validate_permutation("level destination map", &level_map, destination_count)?;
    Ok((
        BatchAffineLevelPlan {
            input_count,
            pair_count,
            carry_count,
            destination_count,
            destination_map_offset,
        },
        level_map,
        next_counts,
    ))
}

fn initial_source_order(
    run_lengths: &[u32],
    leaf_count: u32,
) -> Result<Vec<u32>, BatchAffinePlanError> {
    let mut order = Vec::new();
    order
        .try_reserve_exact(leaf_count as usize)
        .map_err(|_| BatchAffinePlanError::Allocation("initial source order"))?;

    let mut offset = 0u32;
    for &length in run_lengths {
        let paired = (length / 2) * 2;
        for local_index in 0..paired {
            order.push(checked_add(offset, local_index, "paired leaf index")?);
        }
        offset = checked_add(offset, length, "leaf run offset")?;
    }
    offset = 0;
    for &length in run_lengths {
        if length & 1 != 0 {
            order.push(checked_add(offset, length - 1, "carry leaf index")?);
        }
        offset = checked_add(offset, length, "leaf carry offset")?;
    }
    validate_permutation("initial source order", &order, leaf_count)?;
    Ok(order)
}

fn validate_permutation(
    field: &'static str,
    values: &[u32],
    limit: u32,
) -> Result<(), BatchAffinePlanError> {
    let mut seen = Vec::new();
    seen.try_reserve_exact(limit as usize)
        .map_err(|_| BatchAffinePlanError::Allocation("permutation validation"))?;
    seen.resize(limit as usize, false);
    for (index, &value) in values.iter().enumerate() {
        if value >= limit {
            return Err(BatchAffinePlanError::OutOfRange {
                field,
                index,
                value,
                limit,
            });
        }
        if seen[value as usize] {
            return Err(BatchAffinePlanError::Collision {
                field,
                index,
                value,
            });
        }
        seen[value as usize] = true;
    }
    if let Some(value) = seen.iter().position(|present| !present) {
        return Err(BatchAffinePlanError::Hole {
            field,
            value: value as u32,
        });
    }
    if values.len() != limit as usize {
        return Err(BatchAffinePlanError::LengthMismatch {
            field,
            expected: limit as usize,
            actual: values.len(),
        });
    }
    Ok(())
}

fn compare_values(
    field: &'static str,
    actual: &[u32],
    expected: &[u32],
) -> Result<(), BatchAffinePlanError> {
    if actual.len() != expected.len() {
        return Err(BatchAffinePlanError::LengthMismatch {
            field,
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    if let Some((index, (&actual, &expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(BatchAffinePlanError::TopologyMismatch {
            field,
            index,
            expected,
            actual,
        });
    }
    Ok(())
}

fn compare_level(
    index: usize,
    actual: BatchAffineLevelPlan,
    expected: BatchAffineLevelPlan,
) -> Result<(), BatchAffinePlanError> {
    for (field, actual, expected) in [
        (
            "level input count",
            actual.input_count,
            expected.input_count,
        ),
        ("level pair count", actual.pair_count, expected.pair_count),
        (
            "level carry count",
            actual.carry_count,
            expected.carry_count,
        ),
        (
            "level destination count",
            actual.destination_count,
            expected.destination_count,
        ),
        (
            "destination map offset",
            actual.destination_map_offset,
            expected.destination_map_offset,
        ),
    ] {
        if actual != expected {
            return Err(BatchAffinePlanError::TopologyMismatch {
                field,
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, BatchAffinePlanError> {
    u32::try_from(value).map_err(|_| BatchAffinePlanError::ValueTooLarge { field, value })
}

fn checked_add(lhs: u32, rhs: u32, field: &'static str) -> Result<u32, BatchAffinePlanError> {
    lhs.checked_add(rhs)
        .ok_or(BatchAffinePlanError::ArithmeticOverflow(field))
}

fn checked_mul(lhs: u32, rhs: u32, field: &'static str) -> Result<u32, BatchAffinePlanError> {
    lhs.checked_mul(rhs)
        .ok_or(BatchAffinePlanError::ArithmeticOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_singleton_runs_need_no_reduction() {
        let empty = compile_balanced_run_plan(&[]).expect("compile empty plan");
        assert_eq!(empty.leaf_count, 0);
        assert!(empty.initial_source_order.is_empty());
        assert!(empty.levels.is_empty());
        assert!(empty.root_source_positions.is_empty());
        assert_eq!(empty.total_pair_count, 0);

        let singleton = compile_balanced_run_plan(&[0, 1, 0, 1]).expect("compile singleton plan");
        assert_eq!(&*singleton.initial_source_order, &[0, 1]);
        assert!(singleton.levels.is_empty());
        assert_eq!(&*singleton.root_source_positions, &[0, 1]);
        assert_eq!(&*singleton.root_run_indices, &[1, 3]);
        assert_eq!(singleton.total_pair_count, 0);
    }

    #[test]
    fn odd_ragged_runs_preserve_pair_major_layout_and_root_order() {
        let plan = compile_balanced_run_plan(&[3, 0, 4, 1, 5]).expect("compile ragged plan");
        assert_eq!(plan.leaf_count, 13);
        assert_eq!(
            &*plan.initial_source_order,
            &[0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 2, 7, 12]
        );
        assert_eq!(&*plan.root_run_indices, &[0, 2, 3, 4]);
        assert_eq!(&*plan.root_source_positions, &[0, 1, 2, 3]);
        assert_eq!(plan.total_pair_count, 9);
        assert_eq!(
            &*plan.destination_map,
            &[0, 2, 3, 4, 5, 1, 6, 7, 2, 3, 0, 4, 1, 3, 0, 1, 2]
        );
        assert_eq!(
            plan.levels
                .iter()
                .map(|level| (
                    level.input_count,
                    level.pair_count,
                    level.carry_count,
                    level.destination_count,
                ))
                .collect::<Vec<_>>(),
            vec![(13, 5, 3, 8), (8, 3, 2, 5), (5, 1, 3, 4)]
        );
        validate_balanced_run_plan(&plan).expect("validate ragged plan");
    }

    #[test]
    fn concentrated_run_has_logarithmic_levels_and_exact_pair_identity() {
        let plan = compile_balanced_run_plan(&[0, 16_797, 0]).expect("compile concentrated plan");
        assert_eq!(plan.levels.len(), 15);
        assert_eq!(plan.total_pair_count, 16_796);
        assert_eq!(&*plan.root_run_indices, &[1]);
        assert_eq!(&*plan.root_source_positions, &[0]);
        assert_eq!(
            plan.levels.first().map(|level| level.pair_count),
            Some(8_398)
        );
        assert_eq!(
            plan.levels.last().map(|level| level.destination_count),
            Some(1)
        );
    }

    #[test]
    fn validator_rejects_destination_collision() {
        let mut plan = compile_balanced_run_plan(&[4]).expect("compile plan");
        plan.destination_map[1] = plan.destination_map[0];
        assert!(matches!(
            validate_balanced_run_plan(&plan),
            Err(BatchAffinePlanError::Collision {
                field: "level destination map",
                ..
            })
        ));
    }

    #[test]
    fn validator_rejects_destination_hole() {
        let mut plan = compile_balanced_run_plan(&[5]).expect("compile plan");
        plan.destination_map = plan.destination_map[..plan.destination_map.len() - 1]
            .to_vec()
            .into_boxed_slice();
        assert!(matches!(
            validate_balanced_run_plan(&plan),
            Err(BatchAffinePlanError::Hole {
                field: "level destination map",
                ..
            })
        ));
    }

    #[test]
    fn validator_rejects_out_of_range_destination() {
        let mut plan = compile_balanced_run_plan(&[4]).expect("compile plan");
        plan.destination_map[0] = plan.levels[0].destination_count;
        assert!(matches!(
            validate_balanced_run_plan(&plan),
            Err(BatchAffinePlanError::OutOfRange {
                field: "level destination map",
                ..
            })
        ));
    }

    #[test]
    fn validator_rejects_valid_but_run_mixing_permutation() {
        let mut plan = compile_balanced_run_plan(&[3, 4, 5]).expect("compile plan");
        plan.destination_map.swap(0, 1);
        assert!(matches!(
            validate_balanced_run_plan(&plan),
            Err(BatchAffinePlanError::TopologyMismatch {
                field: "level destination map",
                ..
            })
        ));
    }
}
