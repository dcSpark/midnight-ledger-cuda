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

//! Pure host planner for the diagnostic flat segmented affine reducer.
//!
//! Every nonempty run is split into ordered tiles of at most 32 points. One GPU
//! lane streams one tile into an affine accumulator while all lanes in its
//! threadgroup batch-invert the adjusted denominators for a round. Tile outputs
//! stay associated with their original run and are tiled again until every run
//! has one root.
//!
//! There is always at least one pass. In particular, singleton-only inputs need
//! a gather pass so the flat kernel materializes their roots in the destination
//! buffer.
//!
//! Scheduling order is independent of semantic order. Descriptors retain exact
//! input/output offsets, are sorted by `(input_len, output_offset)` to reduce
//! ragged lane work, and are padded with neutral zero-length descriptors to the
//! selected threadgroup width. The only supported widths are the two bounded
//! benchmark candidates, 128 and 256 lanes.

use std::{fmt, mem::size_of};

pub(super) const FLAT_TILE_CAPACITY: u32 = 32;
const FLAT_FP32_BYTES: u64 = 48;

/// The only threadgroup widths admitted by the bounded flat-reducer benchmark.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum FlatSegmentedWidth {
    G128 = 128,
    G256 = 256,
}

impl FlatSegmentedWidth {
    pub(super) const fn lanes(self) -> u32 {
        self as u32
    }

    pub(super) fn from_lanes(lanes: u32) -> Result<Self, FlatSegmentedPlanError> {
        Self::try_from(lanes)
    }
}

impl TryFrom<u32> for FlatSegmentedWidth {
    type Error = FlatSegmentedPlanError;

    fn try_from(lanes: u32) -> Result<Self, Self::Error> {
        match lanes {
            128 => Ok(Self::G128),
            256 => Ok(Self::G256),
            _ => Err(FlatSegmentedPlanError::UnsupportedGroupWidth(lanes)),
        }
    }
}

/// One semantic tile or one neutral padding lane.
///
/// A real descriptor has `input_len` in `1..=32`. A padding descriptor is the
/// all-zero value and must never issue an input load or output write.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FlatSegmentedTile {
    pub(super) input_offset: u32,
    pub(super) input_len: u32,
    pub(super) output_offset: u32,
    pub(super) reserved: u32,
}

/// One point-buffer ping-pong pass.
///
/// `tiles` contains the sorted real descriptors first, followed by neutral
/// descriptors until its length is a multiple of `plan.width.lanes()`.
/// `group_round_counts[group]` belongs to the corresponding fixed-width chunk
/// of `tiles` and equals `max(input_len) - 1` for that chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FlatSegmentedPass {
    pub(super) input_count: u32,
    pub(super) output_count: u32,
    pub(super) tiles: Box<[FlatSegmentedTile]>,
    pub(super) group_round_counts: Box<[u32]>,
    pub(super) useful_reductions: u64,
    pub(super) scheduled_lane_rounds: u64,
    pub(super) inversion_count: u64,
}

/// Exact aggregate accounting for a compiled plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FlatSegmentedCounters {
    pub(super) input_count: u64,
    pub(super) root_count: u64,
    pub(super) useful_reduction_count: u64,
    pub(super) pass_count: u64,
    pub(super) real_tile_count: u64,
    pub(super) padded_tile_count: u64,
    pub(super) padding_tile_count: u64,
    pub(super) group_count: u64,
    pub(super) scheduled_lane_rounds: u64,
    pub(super) inversion_count: u64,
    pub(super) descriptor_bytes: u64,
    pub(super) group_round_count_bytes: u64,
    pub(super) threadgroup_product_tree_bytes: u64,
    pub(super) max_intermediate_points: u64,
    pub(super) max_pass_input_count: u64,
    pub(super) max_pass_output_count: u64,
    pub(super) max_pass_group_count: u64,
    pub(super) max_round_count: u64,
}

/// Canonical descriptor schedule for one independently reduced chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FlatSegmentedPlan {
    pub(super) width: FlatSegmentedWidth,
    pub(super) run_lengths: Box<[u32]>,
    pub(super) passes: Box<[FlatSegmentedPass]>,
    pub(super) input_count: u32,
    pub(super) root_count: u32,
    pub(super) total_useful_reductions: u64,
    pub(super) total_scheduled_lane_rounds: u64,
    pub(super) total_inversions: u64,
    /// Largest destination point buffer required by any pass.
    pub(super) max_intermediate_points: u32,
    pub(super) counters: FlatSegmentedCounters,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum FlatSegmentedPlanError {
    EmptyRuns,
    ZeroRun { index: usize },
    UnsupportedGroupWidth(u32),
    ValueTooLarge { field: &'static str, value: usize },
    ArithmeticOverflow(&'static str),
    Allocation(&'static str),
    NonCanonical { field: &'static str, index: usize },
}

impl fmt::Display for FlatSegmentedPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRuns => write!(formatter, "flat segmented plans require at least one run"),
            Self::ZeroRun { index } => {
                write!(formatter, "flat segmented run {index} is empty")
            }
            Self::UnsupportedGroupWidth(width) => write!(
                formatter,
                "flat segmented group width {width} is unsupported; expected 128 or 256"
            ),
            Self::ValueTooLarge { field, value } => {
                write!(
                    formatter,
                    "flat segmented {field} value {value} exceeds the u32 ABI"
                )
            }
            Self::ArithmeticOverflow(field) => {
                write!(formatter, "flat segmented {field} arithmetic overflow")
            }
            Self::Allocation(field) => {
                write!(formatter, "could not allocate flat segmented {field}")
            }
            Self::NonCanonical { field, index } => write!(
                formatter,
                "flat segmented {field} differs from the canonical plan at index {index}"
            ),
        }
    }
}

impl std::error::Error for FlatSegmentedPlanError {}

/// Compiles nonempty run lengths into canonical sorted and padded pass arrays.
pub(super) fn compile_flat_segmented_plan(
    run_lengths: &[usize],
    width: FlatSegmentedWidth,
) -> Result<FlatSegmentedPlan, FlatSegmentedPlanError> {
    build_flat_segmented_plan(run_lengths, width)
}

/// Rebuilds the canonical plan and rejects any mutation or stale counter.
///
/// Exact reconstruction also prevents an in-bounds descriptor permutation from
/// combining different parent runs or silently changing the ragged schedule.
pub(super) fn validate_flat_segmented_plan(
    plan: &FlatSegmentedPlan,
) -> Result<(), FlatSegmentedPlanError> {
    let mut run_lengths = Vec::new();
    run_lengths
        .try_reserve_exact(plan.run_lengths.len())
        .map_err(|_| FlatSegmentedPlanError::Allocation("validation run lengths"))?;
    run_lengths.extend(plan.run_lengths.iter().map(|&length| length as usize));
    let expected = build_flat_segmented_plan(&run_lengths, plan.width)?;
    compare_plan(plan, &expected)
}

fn build_flat_segmented_plan(
    run_lengths: &[usize],
    width: FlatSegmentedWidth,
) -> Result<FlatSegmentedPlan, FlatSegmentedPlanError> {
    if run_lengths.is_empty() {
        return Err(FlatSegmentedPlanError::EmptyRuns);
    }

    let root_count = checked_u32(run_lengths.len(), "root count")?;
    let mut canonical_lengths = Vec::new();
    canonical_lengths
        .try_reserve_exact(run_lengths.len())
        .map_err(|_| FlatSegmentedPlanError::Allocation("run lengths"))?;
    let mut input_count = 0u32;
    for (index, &length) in run_lengths.iter().enumerate() {
        if length == 0 {
            return Err(FlatSegmentedPlanError::ZeroRun { index });
        }
        let length = checked_u32(length, "run length")?;
        input_count = input_count
            .checked_add(length)
            .ok_or(FlatSegmentedPlanError::ArithmeticOverflow("input count"))?;
        canonical_lengths.push(length);
    }

    let lanes = width.lanes();
    let lanes_usize = lanes as usize;
    let mut current_lengths = canonical_lengths.clone();
    let mut current_count = input_count;
    let mut passes = Vec::<FlatSegmentedPass>::new();
    let mut total_useful_reductions = 0u64;
    let mut total_real_tiles = 0u64;
    let mut total_padded_tiles = 0u64;
    let mut total_groups = 0u64;
    let mut total_scheduled_lane_rounds = 0u64;
    let mut total_inversions = 0u64;
    let mut max_intermediate_points = 0u32;
    let mut max_pass_input_count = 0u64;
    let mut max_pass_output_count = 0u64;
    let mut max_pass_group_count = 0u64;
    let mut max_round_count = 0u64;

    // Deliberately a do-while shape: singleton-only inputs still require a
    // gather pass to materialize roots into the flat kernel's output buffer.
    loop {
        let mut next_lengths = Vec::new();
        next_lengths
            .try_reserve_exact(current_lengths.len())
            .map_err(|_| FlatSegmentedPlanError::Allocation("next run lengths"))?;
        let mut output_count = 0u32;
        for &length in &current_lengths {
            let next_length = length.div_ceil(FLAT_TILE_CAPACITY);
            debug_assert_ne!(next_length, 0);
            output_count = output_count.checked_add(next_length).ok_or(
                FlatSegmentedPlanError::ArithmeticOverflow("pass output count"),
            )?;
            next_lengths.push(next_length);
        }

        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(output_count as usize)
            .map_err(|_| FlatSegmentedPlanError::Allocation("semantic tiles"))?;
        let mut input_base = 0u32;
        let mut output_base = 0u32;
        for (&length, &tile_count) in current_lengths.iter().zip(&next_lengths) {
            for tile_index in 0..tile_count {
                let relative_input = tile_index.checked_mul(FLAT_TILE_CAPACITY).ok_or(
                    FlatSegmentedPlanError::ArithmeticOverflow("tile input offset"),
                )?;
                let input_offset = input_base.checked_add(relative_input).ok_or(
                    FlatSegmentedPlanError::ArithmeticOverflow("tile input offset"),
                )?;
                let remaining = length.checked_sub(relative_input).ok_or(
                    FlatSegmentedPlanError::ArithmeticOverflow("tile input length"),
                )?;
                let output_offset = output_base.checked_add(tile_index).ok_or(
                    FlatSegmentedPlanError::ArithmeticOverflow("tile output offset"),
                )?;
                tiles.push(FlatSegmentedTile {
                    input_offset,
                    input_len: remaining.min(FLAT_TILE_CAPACITY),
                    output_offset,
                    reserved: 0,
                });
            }
            input_base = input_base.checked_add(length).ok_or(
                FlatSegmentedPlanError::ArithmeticOverflow("pass input cursor"),
            )?;
            output_base = output_base.checked_add(tile_count).ok_or(
                FlatSegmentedPlanError::ArithmeticOverflow("pass output cursor"),
            )?;
        }
        if input_base != current_count || output_base != output_count {
            return Err(FlatSegmentedPlanError::ArithmeticOverflow(
                "pass cursor identity",
            ));
        }

        tiles.sort_unstable_by_key(|tile| (tile.input_len, tile.output_offset));
        let real_tile_count = tiles.len();
        if real_tile_count != output_count as usize {
            return Err(FlatSegmentedPlanError::ArithmeticOverflow(
                "real tile/output identity",
            ));
        }
        let group_count = real_tile_count.div_ceil(lanes_usize);
        let padded_tile_count = group_count.checked_mul(lanes_usize).ok_or(
            FlatSegmentedPlanError::ArithmeticOverflow("padded tile count"),
        )?;
        checked_u32(padded_tile_count, "pass padded tile count")?;
        let padding_tile_count = padded_tile_count.checked_sub(real_tile_count).ok_or(
            FlatSegmentedPlanError::ArithmeticOverflow("padding tile count"),
        )?;
        tiles
            .try_reserve_exact(padding_tile_count)
            .map_err(|_| FlatSegmentedPlanError::Allocation("padding tiles"))?;
        tiles.resize(padded_tile_count, FlatSegmentedTile::default());

        let mut group_round_counts = Vec::new();
        group_round_counts
            .try_reserve_exact(group_count)
            .map_err(|_| FlatSegmentedPlanError::Allocation("group round counts"))?;
        let mut inversion_count = 0u64;
        for group_tiles in tiles.chunks_exact(lanes_usize) {
            let max_input_len = group_tiles
                .iter()
                .map(|tile| tile.input_len)
                .max()
                .unwrap_or(0);
            let round_count =
                max_input_len
                    .checked_sub(1)
                    .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(
                        "empty scheduled group",
                    ))?;
            group_round_counts.push(round_count);
            inversion_count = checked_add_u64(
                inversion_count,
                u64::from(round_count),
                "pass inversion count",
            )?;
            max_round_count = max_round_count.max(u64::from(round_count));
        }
        if group_round_counts.len() != group_count {
            return Err(FlatSegmentedPlanError::ArithmeticOverflow(
                "group count identity",
            ));
        }

        let useful_reductions = u64::from(current_count)
            .checked_sub(u64::from(output_count))
            .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(
                "pass useful reductions",
            ))?;
        let scheduled_lane_rounds = inversion_count.checked_mul(u64::from(lanes)).ok_or(
            FlatSegmentedPlanError::ArithmeticOverflow("pass scheduled lane rounds"),
        )?;

        passes
            .try_reserve(1)
            .map_err(|_| FlatSegmentedPlanError::Allocation("passes"))?;
        passes.push(FlatSegmentedPass {
            input_count: current_count,
            output_count,
            tiles: tiles.into_boxed_slice(),
            group_round_counts: group_round_counts.into_boxed_slice(),
            useful_reductions,
            scheduled_lane_rounds,
            inversion_count,
        });

        total_useful_reductions = checked_add_u64(
            total_useful_reductions,
            useful_reductions,
            "total useful reductions",
        )?;
        total_real_tiles =
            checked_add_u64(total_real_tiles, real_tile_count as u64, "total real tiles")?;
        total_padded_tiles = checked_add_u64(
            total_padded_tiles,
            padded_tile_count as u64,
            "total padded tiles",
        )?;
        total_groups = checked_add_u64(total_groups, group_count as u64, "total groups")?;
        total_scheduled_lane_rounds = checked_add_u64(
            total_scheduled_lane_rounds,
            scheduled_lane_rounds,
            "total scheduled lane rounds",
        )?;
        total_inversions = checked_add_u64(total_inversions, inversion_count, "total inversions")?;
        max_intermediate_points = max_intermediate_points.max(output_count);
        max_pass_input_count = max_pass_input_count.max(u64::from(current_count));
        max_pass_output_count = max_pass_output_count.max(u64::from(output_count));
        max_pass_group_count = max_pass_group_count.max(group_count as u64);

        current_count = output_count;
        current_lengths = next_lengths;
        if current_lengths.iter().all(|&length| length == 1) {
            break;
        }
    }

    if current_count != root_count {
        return Err(FlatSegmentedPlanError::ArithmeticOverflow(
            "terminal root identity",
        ));
    }
    let expected_useful_reductions = u64::from(input_count)
        .checked_sub(u64::from(root_count))
        .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(
            "input/root reduction identity",
        ))?;
    if total_useful_reductions != expected_useful_reductions {
        return Err(FlatSegmentedPlanError::ArithmeticOverflow(
            "telescoping reduction identity",
        ));
    }
    let expected_lane_rounds = total_inversions.checked_mul(u64::from(lanes)).ok_or(
        FlatSegmentedPlanError::ArithmeticOverflow("lane-round/inversion identity"),
    )?;
    if total_scheduled_lane_rounds != expected_lane_rounds {
        return Err(FlatSegmentedPlanError::ArithmeticOverflow(
            "lane-round/inversion identity",
        ));
    }

    let padding_tile_count = total_padded_tiles.checked_sub(total_real_tiles).ok_or(
        FlatSegmentedPlanError::ArithmeticOverflow("total padding tiles"),
    )?;
    let descriptor_bytes = checked_mul_u64(
        total_padded_tiles,
        size_of::<FlatSegmentedTile>() as u64,
        "descriptor bytes",
    )?;
    let group_round_count_bytes = checked_mul_u64(
        total_groups,
        size_of::<u32>() as u64,
        "group round-count bytes",
    )?;
    let product_tree_elements = u64::from(lanes)
        .checked_mul(2)
        .and_then(|elements| elements.checked_sub(1))
        .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(
            "threadgroup product tree elements",
        ))?;
    let threadgroup_product_tree_bytes = checked_mul_u64(
        product_tree_elements,
        FLAT_FP32_BYTES,
        "threadgroup product tree bytes",
    )?;
    let counters = FlatSegmentedCounters {
        input_count: u64::from(input_count),
        root_count: u64::from(root_count),
        useful_reduction_count: total_useful_reductions,
        pass_count: passes.len() as u64,
        real_tile_count: total_real_tiles,
        padded_tile_count: total_padded_tiles,
        padding_tile_count,
        group_count: total_groups,
        scheduled_lane_rounds: total_scheduled_lane_rounds,
        inversion_count: total_inversions,
        descriptor_bytes,
        group_round_count_bytes,
        threadgroup_product_tree_bytes,
        max_intermediate_points: u64::from(max_intermediate_points),
        max_pass_input_count,
        max_pass_output_count,
        max_pass_group_count,
        max_round_count,
    };

    Ok(FlatSegmentedPlan {
        width,
        run_lengths: canonical_lengths.into_boxed_slice(),
        passes: passes.into_boxed_slice(),
        input_count,
        root_count,
        total_useful_reductions,
        total_scheduled_lane_rounds,
        total_inversions,
        max_intermediate_points,
        counters,
    })
}

fn compare_plan(
    actual: &FlatSegmentedPlan,
    expected: &FlatSegmentedPlan,
) -> Result<(), FlatSegmentedPlanError> {
    if actual.width != expected.width {
        return Err(noncanonical("width", 0));
    }
    compare_slice("run lengths", &actual.run_lengths, &expected.run_lengths)?;
    compare_slice("passes", &actual.passes, &expected.passes)?;
    if actual.input_count != expected.input_count {
        return Err(noncanonical("input count", 0));
    }
    if actual.root_count != expected.root_count {
        return Err(noncanonical("root count", 0));
    }
    if actual.total_useful_reductions != expected.total_useful_reductions {
        return Err(noncanonical("total useful reductions", 0));
    }
    if actual.total_scheduled_lane_rounds != expected.total_scheduled_lane_rounds {
        return Err(noncanonical("total scheduled lane rounds", 0));
    }
    if actual.total_inversions != expected.total_inversions {
        return Err(noncanonical("total inversions", 0));
    }
    if actual.max_intermediate_points != expected.max_intermediate_points {
        return Err(noncanonical("max intermediate points", 0));
    }
    if actual.counters != expected.counters {
        return Err(noncanonical("counters", 0));
    }
    Ok(())
}

fn compare_slice<T: PartialEq>(
    field: &'static str,
    actual: &[T],
    expected: &[T],
) -> Result<(), FlatSegmentedPlanError> {
    let common = actual.len().min(expected.len());
    if let Some(index) = (0..common).find(|&index| actual[index] != expected[index]) {
        return Err(noncanonical(field, index));
    }
    if actual.len() != expected.len() {
        return Err(noncanonical(field, common));
    }
    Ok(())
}

const fn noncanonical(field: &'static str, index: usize) -> FlatSegmentedPlanError {
    FlatSegmentedPlanError::NonCanonical { field, index }
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, FlatSegmentedPlanError> {
    u32::try_from(value).map_err(|_| FlatSegmentedPlanError::ValueTooLarge { field, value })
}

fn checked_add_u64(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, FlatSegmentedPlanError> {
    left.checked_add(right)
        .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(field))
}

fn checked_mul_u64(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, FlatSegmentedPlanError> {
    left.checked_mul(right)
        .ok_or(FlatSegmentedPlanError::ArithmeticOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn descriptor_abi_is_exactly_four_u32_words() {
        assert_eq!(size_of::<FlatSegmentedTile>(), 16);
        assert_eq!(align_of::<FlatSegmentedTile>(), 4);
        assert_eq!(offset_of!(FlatSegmentedTile, input_offset), 0);
        assert_eq!(offset_of!(FlatSegmentedTile, input_len), 4);
        assert_eq!(offset_of!(FlatSegmentedTile, output_offset), 8);
        assert_eq!(offset_of!(FlatSegmentedTile, reserved), 12);
    }

    #[test]
    fn group_width_is_restricted_to_the_two_bounded_candidates() {
        assert_eq!(
            FlatSegmentedWidth::from_lanes(128),
            Ok(FlatSegmentedWidth::G128)
        );
        assert_eq!(
            FlatSegmentedWidth::from_lanes(256),
            Ok(FlatSegmentedWidth::G256)
        );
        for invalid in [0, 1, 32, 64, 127, 129, 255, 257, 512] {
            assert!(matches!(
                FlatSegmentedWidth::from_lanes(invalid),
                Err(FlatSegmentedPlanError::UnsupportedGroupWidth(value)) if value == invalid
            ));
        }
    }

    #[test]
    fn empty_and_zero_runs_are_rejected() {
        assert!(matches!(
            compile_flat_segmented_plan(&[], FlatSegmentedWidth::G128),
            Err(FlatSegmentedPlanError::EmptyRuns)
        ));
        assert!(matches!(
            compile_flat_segmented_plan(&[1, 0, 2], FlatSegmentedWidth::G256),
            Err(FlatSegmentedPlanError::ZeroRun { index: 1 })
        ));
    }

    #[test]
    fn all_singletons_get_one_root_materialization_pass() {
        for width in [FlatSegmentedWidth::G128, FlatSegmentedWidth::G256] {
            let plan = compile_flat_segmented_plan(&[1; 257], width).expect("compile singletons");
            assert_eq!(plan.input_count, 257);
            assert_eq!(plan.root_count, 257);
            assert_eq!(plan.passes.len(), 1);
            let pass = &plan.passes[0];
            assert_eq!((pass.input_count, pass.output_count), (257, 257));
            assert_eq!(pass.useful_reductions, 0);
            assert_eq!(pass.inversion_count, 0);
            assert_eq!(pass.scheduled_lane_rounds, 0);
            assert_eq!(
                pass.tiles.len(),
                257usize.div_ceil(width.lanes() as usize) * width.lanes() as usize
            );
            assert!(real_tiles(pass).iter().all(|tile| tile.input_len == 1));
            assert_eq!(plan.max_intermediate_points, 257);
            assert_eq!(
                plan.counters.threadgroup_product_tree_bytes,
                tree_bytes(width)
            );
            assert_pass_coverage(&plan);
            validate_flat_segmented_plan(&plan).expect("validate singletons");
        }
    }

    #[test]
    fn thirty_three_and_singleton_telescope_across_two_passes() {
        let plan = compile_flat_segmented_plan(&[33, 1], FlatSegmentedWidth::G256)
            .expect("compile two-pass plan");
        assert_eq!(plan.passes.len(), 2);
        assert_eq!(plan.total_useful_reductions, 32);

        let first = &plan.passes[0];
        assert_eq!((first.input_count, first.output_count), (34, 3));
        assert_eq!(first.useful_reductions, 31);
        assert_eq!(first.tiles.len(), 256);
        assert_eq!(first.group_round_counts.as_ref(), &[31]);
        assert_eq!(first.inversion_count, 31);
        assert_eq!(first.scheduled_lane_rounds, 31 * 256);
        assert_eq!(
            real_tiles(first),
            &[
                FlatSegmentedTile {
                    input_offset: 32,
                    input_len: 1,
                    output_offset: 1,
                    reserved: 0,
                },
                FlatSegmentedTile {
                    input_offset: 33,
                    input_len: 1,
                    output_offset: 2,
                    reserved: 0,
                },
                FlatSegmentedTile {
                    input_offset: 0,
                    input_len: 32,
                    output_offset: 0,
                    reserved: 0,
                },
            ]
        );

        let second = &plan.passes[1];
        assert_eq!((second.input_count, second.output_count), (3, 2));
        assert_eq!(second.useful_reductions, 1);
        assert_eq!(second.group_round_counts.as_ref(), &[1]);
        assert_eq!(second.inversion_count, 1);
        validate_flat_segmented_plan(&plan).expect("validate two-pass plan");
    }

    #[test]
    fn tile_capacity_boundaries_preserve_exact_reduction_identity() {
        let lengths = [31usize, 32, 33, 63, 64, 65];
        let plan = compile_flat_segmented_plan(&lengths, FlatSegmentedWidth::G128)
            .expect("compile tile boundaries");
        assert_eq!(plan.passes.len(), 2);
        assert_eq!(plan.passes[0].output_count, 11);
        assert_eq!(plan.passes[0].useful_reductions, 277);
        assert_eq!(plan.passes[1].output_count, lengths.len() as u32);
        assert_eq!(plan.passes[1].useful_reductions, 5);
        assert_eq!(
            plan.total_useful_reductions,
            lengths.iter().sum::<usize>() as u64 - lengths.len() as u64
        );
        assert_pass_coverage(&plan);
    }

    #[test]
    fn group_boundaries_pad_exactly_for_both_widths() {
        for width in [FlatSegmentedWidth::G128, FlatSegmentedWidth::G256] {
            let lanes = width.lanes() as usize;
            for real_count in [
                lanes - 1,
                lanes,
                lanes + 1,
                2 * lanes - 1,
                2 * lanes,
                2 * lanes + 1,
            ] {
                let lengths = vec![32usize; real_count];
                let plan = compile_flat_segmented_plan(&lengths, width)
                    .unwrap_or_else(|error| panic!("compile {width:?}/{real_count}: {error}"));
                assert_eq!(plan.passes.len(), 1);
                let pass = &plan.passes[0];
                let expected_groups = real_count.div_ceil(lanes);
                assert_eq!(real_tiles(pass).len(), real_count);
                assert_eq!(pass.group_round_counts.len(), expected_groups);
                assert_eq!(pass.tiles.len(), expected_groups * lanes);
                assert_eq!(
                    plan.counters.padding_tile_count,
                    (expected_groups * lanes - real_count) as u64
                );
                assert!(pass.group_round_counts.iter().all(|&rounds| rounds == 31));
                assert_pass_coverage(&plan);
            }
        }
    }

    #[test]
    fn scheduling_is_sorted_then_neutrally_padded() {
        let plan = compile_flat_segmented_plan(&[1, 32, 2, 31, 33], FlatSegmentedWidth::G128)
            .expect("compile sorted schedule");
        for pass in &plan.passes {
            let real = real_tiles(pass);
            assert!(real.windows(2).all(|pair| {
                (pair[0].input_len, pair[0].output_offset)
                    <= (pair[1].input_len, pair[1].output_offset)
            }));
            assert!(
                padding_tiles(pass)
                    .iter()
                    .all(|tile| *tile == FlatSegmentedTile::default())
            );
        }
        assert_pass_coverage(&plan);
    }

    #[test]
    fn large_ragged_boundaries_terminate_and_preserve_every_reduction() {
        let lengths = [
            1usize, 2, 31, 32, 33, 63, 64, 65, 1_023, 1_024, 1_025, 16_797,
        ];
        for width in [FlatSegmentedWidth::G128, FlatSegmentedWidth::G256] {
            let plan = compile_flat_segmented_plan(&lengths, width)
                .unwrap_or_else(|error| panic!("compile {width:?}: {error}"));
            assert_eq!(plan.passes.len(), 3);
            assert_eq!(plan.passes.last().map(|pass| pass.output_count), Some(12));
            assert_eq!(plan.counters.max_round_count, 31);
            assert_eq!(
                plan.total_useful_reductions,
                lengths.iter().sum::<usize>() as u64 - lengths.len() as u64
            );
            assert_eq!(
                plan.total_scheduled_lane_rounds,
                plan.total_inversions * u64::from(width.lanes())
            );
            assert_pass_coverage(&plan);
            validate_flat_segmented_plan(&plan).expect("validate ragged plan");
        }
    }

    #[test]
    fn widths_change_only_padding_and_group_work_not_semantics() {
        let lengths = [3usize, 5, 17, 31, 32, 33, 257, 1_025, 16_797];
        let plan128 =
            compile_flat_segmented_plan(&lengths, FlatSegmentedWidth::G128).expect("compile G128");
        let plan256 =
            compile_flat_segmented_plan(&lengths, FlatSegmentedWidth::G256).expect("compile G256");
        assert_eq!(plan128.run_lengths, plan256.run_lengths);
        assert_eq!(plan128.input_count, plan256.input_count);
        assert_eq!(plan128.root_count, plan256.root_count);
        assert_eq!(
            plan128.total_useful_reductions,
            plan256.total_useful_reductions
        );
        assert_eq!(
            plan128.counters.real_tile_count,
            plan256.counters.real_tile_count
        );
        assert_eq!(
            plan128
                .passes
                .iter()
                .map(|pass| (
                    pass.input_count,
                    pass.output_count,
                    pass.useful_reductions,
                    real_tiles(pass).to_vec(),
                ))
                .collect::<Vec<_>>(),
            plan256
                .passes
                .iter()
                .map(|pass| (
                    pass.input_count,
                    pass.output_count,
                    pass.useful_reductions,
                    real_tiles(pass).to_vec(),
                ))
                .collect::<Vec<_>>()
        );
        assert!(plan128.counters.group_count >= plan256.counters.group_count);
        assert_eq!(plan128.counters.threadgroup_product_tree_bytes, 12_240);
        assert_eq!(plan256.counters.threadgroup_product_tree_bytes, 24_528);
    }

    #[test]
    fn counters_match_owned_buffers_and_top_level_totals() {
        let plan =
            compile_flat_segmented_plan(&[1, 2, 32, 33, 1_025, 16_797], FlatSegmentedWidth::G128)
                .expect("compile accounting plan");
        let padded_tiles = plan
            .passes
            .iter()
            .map(|pass| pass.tiles.len() as u64)
            .sum::<u64>();
        let groups = plan
            .passes
            .iter()
            .map(|pass| pass.group_round_counts.len() as u64)
            .sum::<u64>();
        assert_eq!(plan.counters.descriptor_bytes, padded_tiles * 16);
        assert_eq!(plan.counters.group_round_count_bytes, groups * 4);
        assert_eq!(
            plan.counters.useful_reduction_count,
            plan.total_useful_reductions
        );
        assert_eq!(
            plan.counters.scheduled_lane_rounds,
            plan.total_scheduled_lane_rounds
        );
        assert_eq!(plan.counters.inversion_count, plan.total_inversions);
        assert_eq!(
            plan.counters.max_intermediate_points,
            u64::from(plan.max_intermediate_points)
        );
    }

    #[test]
    fn validation_rejects_descriptor_round_pass_total_and_counter_mutations() {
        let original =
            compile_flat_segmented_plan(&[2, 31, 32, 33, 1_025], FlatSegmentedWidth::G256)
                .expect("compile mutation target");

        let mut descriptor = original.clone();
        descriptor.passes[0].tiles[0].reserved = 1;
        assert!(matches!(
            validate_flat_segmented_plan(&descriptor),
            Err(FlatSegmentedPlanError::NonCanonical {
                field: "passes",
                ..
            })
        ));

        let mut rounds = original.clone();
        rounds.passes[0].group_round_counts[0] ^= 1;
        assert!(matches!(
            validate_flat_segmented_plan(&rounds),
            Err(FlatSegmentedPlanError::NonCanonical {
                field: "passes",
                ..
            })
        ));

        let mut pass = original.clone();
        pass.passes[0].useful_reductions += 1;
        assert!(matches!(
            validate_flat_segmented_plan(&pass),
            Err(FlatSegmentedPlanError::NonCanonical {
                field: "passes",
                ..
            })
        ));

        let mut total = original.clone();
        total.total_inversions += 1;
        assert!(matches!(
            validate_flat_segmented_plan(&total),
            Err(FlatSegmentedPlanError::NonCanonical {
                field: "total inversions",
                ..
            })
        ));

        let mut counters = original.clone();
        counters.counters.scheduled_lane_rounds += 1;
        assert!(matches!(
            validate_flat_segmented_plan(&counters),
            Err(FlatSegmentedPlanError::NonCanonical {
                field: "counters",
                ..
            })
        ));
    }

    #[test]
    fn input_count_overflow_is_rejected_before_large_allocations() {
        let error = compile_flat_segmented_plan(&[u32::MAX as usize, 1], FlatSegmentedWidth::G128)
            .expect_err("input sum must overflow the ABI");
        assert!(matches!(
            error,
            FlatSegmentedPlanError::ArithmeticOverflow("input count")
        ));
    }

    #[test]
    fn compilation_is_deterministic() {
        let lengths = [1usize, 2, 7, 31, 32, 33, 255, 256, 257, 16_797];
        for width in [FlatSegmentedWidth::G128, FlatSegmentedWidth::G256] {
            let first = compile_flat_segmented_plan(&lengths, width).expect("first compile");
            let second = compile_flat_segmented_plan(&lengths, width).expect("second compile");
            assert_eq!(first, second);
        }
    }

    fn tree_bytes(width: FlatSegmentedWidth) -> u64 {
        (2 * u64::from(width.lanes()) - 1) * FLAT_FP32_BYTES
    }

    fn real_tiles(pass: &FlatSegmentedPass) -> &[FlatSegmentedTile] {
        &pass.tiles[..pass.output_count as usize]
    }

    fn padding_tiles(pass: &FlatSegmentedPass) -> &[FlatSegmentedTile] {
        &pass.tiles[pass.output_count as usize..]
    }

    fn assert_pass_coverage(plan: &FlatSegmentedPlan) {
        let mut lengths = plan.run_lengths.to_vec();
        let mut useful_total = 0u64;
        let mut lane_round_total = 0u64;
        let mut inversion_total = 0u64;
        let mut max_intermediate = 0u32;

        for pass in &plan.passes {
            assert_eq!(pass.input_count, lengths.iter().sum::<u32>());
            let mut expected = Vec::new();
            let mut next_lengths = Vec::with_capacity(lengths.len());
            let mut input_base = 0u32;
            let mut output_base = 0u32;
            for &length in &lengths {
                let tile_count = length.div_ceil(FLAT_TILE_CAPACITY);
                next_lengths.push(tile_count);
                for tile_index in 0..tile_count {
                    let relative = tile_index * FLAT_TILE_CAPACITY;
                    expected.push(FlatSegmentedTile {
                        input_offset: input_base + relative,
                        input_len: (length - relative).min(FLAT_TILE_CAPACITY),
                        output_offset: output_base + tile_index,
                        reserved: 0,
                    });
                }
                input_base += length;
                output_base += tile_count;
            }
            expected.sort_unstable_by_key(|tile| (tile.input_len, tile.output_offset));
            assert_eq!(real_tiles(pass), expected.as_slice());
            assert_eq!(pass.output_count, output_base);
            assert!(
                padding_tiles(pass)
                    .iter()
                    .all(|tile| *tile == FlatSegmentedTile::default())
            );
            let lanes = plan.width.lanes() as usize;
            assert_eq!(pass.tiles.len() % lanes, 0);
            let expected_round_counts = pass
                .tiles
                .chunks_exact(lanes)
                .map(|group| group.iter().map(|tile| tile.input_len).max().unwrap() - 1)
                .collect::<Vec<_>>();
            assert_eq!(pass.group_round_counts.as_ref(), expected_round_counts);
            assert_eq!(
                pass.useful_reductions,
                u64::from(pass.input_count - pass.output_count)
            );
            assert_eq!(
                pass.inversion_count,
                pass.group_round_counts
                    .iter()
                    .map(|&rounds| u64::from(rounds))
                    .sum::<u64>()
            );
            assert_eq!(
                pass.scheduled_lane_rounds,
                pass.inversion_count * u64::from(plan.width.lanes())
            );

            useful_total += pass.useful_reductions;
            lane_round_total += pass.scheduled_lane_rounds;
            inversion_total += pass.inversion_count;
            max_intermediate = max_intermediate.max(pass.output_count);
            lengths = next_lengths;
        }

        assert!(!plan.passes.is_empty());
        assert!(lengths.iter().all(|&length| length == 1));
        assert_eq!(useful_total, plan.total_useful_reductions);
        assert_eq!(lane_round_total, plan.total_scheduled_lane_rounds);
        assert_eq!(inversion_total, plan.total_inversions);
        assert_eq!(max_intermediate, plan.max_intermediate_points);
        assert_eq!(
            plan.total_useful_reductions,
            u64::from(plan.input_count - plan.root_count)
        );
    }
}
