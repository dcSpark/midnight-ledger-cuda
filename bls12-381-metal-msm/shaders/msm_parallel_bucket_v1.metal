// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Parallel bucket-reduction candidate, private shader ABI version 1.
// The host splits every non-empty bucket run into contiguous tiles of at most
// 128 routed records. One 32-thread threadgroup lets each lane consume up to
// four leaves before a balanced lane tree; a second 32-thread threadgroup
// reduces the contiguous tile partials for each run.

#ifndef BLS12_381_METAL_MSM_PARALLEL_BUCKET_V1
#define BLS12_381_METAL_MSM_PARALLEL_BUCKET_V1

#include "msm_pipeline.metal"

constant uint BLS12_381_METAL_MSM_PARALLEL_BUCKET_ABI_V1 = 1U;
constant uint BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1 = 32U;
constant uint BLS12_381_METAL_MSM_PARALLEL_BUCKET_TILE_RECORDS_V1 = 128U;

// Descriptors are ordered by run. `run_index` makes the second pass able to
// reject an overlap, gap, or tile assigned to the wrong original bucket run.
struct Bls12381MetalMsmBucketTileV1 {
    uint record_offset;
    uint record_count;
    uint run_index;
    uint reserved;
};

// Run descriptors must form an exact, ordered partition of [0, tile_count).
// The output run-root index is the descriptor's dispatch index.
struct Bls12381MetalMsmTileRunV1 {
    uint tile_offset;
    uint tile_count;
    uint reserved_0;
    uint reserved_1;
};

// Shared by both passes. The explicit version and launch constants prevent a
// host compiled for another schedule from accidentally dispatching this ABI.
struct Bls12381MetalMsmParallelBucketParametersV1 {
    uint abi_version;
    uint base_count;
    uint record_count;
    uint tile_count;
    uint run_count;
    uint threads_per_threadgroup;
    uint max_records_per_tile;
    uint reserved;
};

static_assert(sizeof(Bls12381MetalMsmBucketTileV1) == 16, "tile v1 ABI");
static_assert(alignof(Bls12381MetalMsmBucketTileV1) == 4, "tile v1 alignment");
static_assert(sizeof(Bls12381MetalMsmTileRunV1) == 16, "tile run v1 ABI");
static_assert(alignof(Bls12381MetalMsmTileRunV1) == 4, "tile run v1 alignment");
static_assert(
    sizeof(Bls12381MetalMsmParallelBucketParametersV1) == 32,
    "parallel bucket parameters v1 ABI");
static_assert(
    alignof(Bls12381MetalMsmParallelBucketParametersV1) == 4,
    "parallel bucket parameters v1 alignment");
static_assert(
    __builtin_offsetof(Bls12381MetalMsmBucketTileV1, record_offset) == 0 &&
        __builtin_offsetof(Bls12381MetalMsmBucketTileV1, record_count) == 4 &&
        __builtin_offsetof(Bls12381MetalMsmBucketTileV1, run_index) == 8 &&
        __builtin_offsetof(Bls12381MetalMsmBucketTileV1, reserved) == 12,
    "tile v1 field offsets");
static_assert(
    __builtin_offsetof(Bls12381MetalMsmTileRunV1, tile_offset) == 0 &&
        __builtin_offsetof(Bls12381MetalMsmTileRunV1, tile_count) == 4 &&
        __builtin_offsetof(Bls12381MetalMsmTileRunV1, reserved_0) == 8 &&
        __builtin_offsetof(Bls12381MetalMsmTileRunV1, reserved_1) == 12,
    "tile run v1 field offsets");
static_assert(
    __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, abi_version) == 0 &&
        __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, base_count) == 4 &&
        __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, record_count) == 8 &&
        __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, tile_count) == 12 &&
        __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, run_count) == 16 &&
        __builtin_offsetof(
            Bls12381MetalMsmParallelBucketParametersV1,
            threads_per_threadgroup) == 20 &&
        __builtin_offsetof(
            Bls12381MetalMsmParallelBucketParametersV1,
            max_records_per_tile) == 24 &&
        __builtin_offsetof(Bls12381MetalMsmParallelBucketParametersV1, reserved) == 28,
    "parallel bucket parameters v1 field offsets");

inline __attribute__((always_inline))
bool bls12_381_msm_parallel_launch_valid_v1(
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint3 threads,
    uint3 groups,
    uint expected_groups)
{
    return parameters.abi_version ==
            BLS12_381_METAL_MSM_PARALLEL_BUCKET_ABI_V1 &&
        parameters.threads_per_threadgroup ==
            BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1 &&
        parameters.max_records_per_tile ==
            BLS12_381_METAL_MSM_PARALLEL_BUCKET_TILE_RECORDS_V1 &&
        parameters.reserved == 0U &&
        threads.x == BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1 &&
        threads.y == 1U && threads.z == 1U &&
        groups.x == expected_groups && groups.y == 1U && groups.z == 1U;
}

inline __attribute__((always_inline))
uint bls12_381_msm_parallel_tree_span_v1(uint count)
{
    // Callers validate 1 <= count <= 128 before reaching this helper.
    uint value = count - 1U;
    value |= value >> 1U;
    value |= value >> 2U;
    value |= value >> 4U;
    value |= value >> 8U;
    value |= value >> 16U;
    return value + 1U;
}

inline __attribute__((always_inline))
bool bls12_381_msm_bucket_tile_payload_valid_v1(
    Bls12381MetalMsmBucketTileV1 tile,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters)
{
    return tile.reserved == 0U && tile.run_index < parameters.run_count &&
        tile.record_count != 0U &&
        tile.record_count <= BLS12_381_METAL_MSM_PARALLEL_BUCKET_TILE_RECORDS_V1 &&
        bls12_381_msm_range_valid(
            tile.record_offset, tile.record_count, parameters.record_count);
}

inline __attribute__((always_inline))
bool bls12_381_msm_bucket_tile_partition_valid_v1(
    device const Bls12381MetalMsmBucketTileV1* tiles,
    uint tile_index,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters)
{
    const Bls12381MetalMsmBucketTileV1 tile = tiles[tile_index];
    bool valid = bls12_381_msm_bucket_tile_payload_valid_v1(tile, parameters);
    if (!valid) {
        return false;
    }

    if (tile_index == 0U) {
        valid = tile.record_offset == 0U && tile.run_index == 0U;
    } else {
        const Bls12381MetalMsmBucketTileV1 previous = tiles[tile_index - 1U];
        const bool previous_valid =
            bls12_381_msm_bucket_tile_payload_valid_v1(previous, parameters);
        const bool run_contiguous = previous_valid &&
            (tile.run_index == previous.run_index ||
             tile.run_index == previous.run_index + 1U);
        valid = previous_valid && run_contiguous &&
            tile.record_offset == previous.record_offset + previous.record_count;
    }
    if (tile_index + 1U == parameters.tile_count) {
        valid = valid &&
            tile.record_offset + tile.record_count == parameters.record_count &&
            tile.run_index + 1U == parameters.run_count;
    }
    return valid;
}

inline __attribute__((always_inline))
bool bls12_381_msm_bucket_tile_for_run_valid_v1(
    device const Bls12381MetalMsmBucketTileV1* tiles,
    uint tile_index,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint expected_run)
{
    return tiles[tile_index].run_index == expected_run &&
        bls12_381_msm_bucket_tile_partition_valid_v1(
            tiles, tile_index, parameters);
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_homogeneous_from_signed_affine_v1(
    Bls12381MetalMsmAffine point,
    uint flags)
{
    const ulong identity = bls12_381_msm_g1_affine_identity_mask(point);
    point = bls12_381_msm_g1_signed_affine(point, flags);

    Bls12381MetalMsmHomogeneous result;
    result.x = bls12_381_msm_fp_select(
        point.x, bls12_381_msm_fp_zero(), identity);
    result.y = bls12_381_msm_fp_select(
        point.y, bls12_381_msm_fp_one(), identity);
    result.z = bls12_381_msm_fp_select(
        bls12_381_msm_fp_one(), bls12_381_msm_fp_zero(), identity);
    return result;
}

// Buffers:
//   0 prepared affine bases [base_count]
//   1 grouped signed records [record_count]
//   2 tile descriptors [tile_count]
//   3 homogeneous tile partials [tile_count]
//   4 shared status (one Bls12381MetalMsmStatus)
//   5 one Bls12381MetalMsmParallelBucketParametersV1
//
// Dispatch exactly tile_count threadgroups of (32, 1, 1). Static threadgroup
// storage is 32 * 144 + 32 * 4 = 4,736 bytes. Inactive lanes contribute
// the canonical identity, so every tree level has the same barriers for short
// and non-power-of-two tiles.
kernel void bls12_381_metal_msm_bucket_tiles_parallel_v1(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmHomogeneous* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    if (group.y != 0U || group.z != 0U) {
        return;
    }
    const uint tile_index = group.x;
    if (tile_index >= parameters.tile_count) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    threadgroup Bls12381MetalMsmHomogeneous partials[32];
    threadgroup uint valid[32];

    const bool launch_valid =
        bls12_381_msm_parallel_launch_valid_v1(
            parameters, threads, groups, parameters.tile_count);
    Bls12381MetalMsmBucketTileV1 tile;
    if (launch_valid) {
        tile = tiles[tile_index];
    } else {
        tile.record_offset = 0U;
        tile.record_count = 0U;
        tile.run_index = 0U;
        tile.reserved = 0U;
    }
    const bool descriptor_valid = launch_valid &&
        bls12_381_msm_bucket_tile_partition_valid_v1(
            tiles, tile_index, parameters);

    // This branch is uniform across the entire threadgroup, so it cannot make
    // a later threadgroup barrier divergent.
    if (!descriptor_valid) {
        if (lane == 0U) {
            tile_roots[tile_index] = bls12_381_msm_g1_identity();
            bls12_381_msm_status_descriptor_error(
                status,
                launch_valid
                    ? BLS12_381_METAL_MSM_STATUS_INVALID_RUN
                    : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    Bls12381MetalMsmHomogeneous accumulator = bls12_381_msm_g1_identity();
    uint lane_valid = 1U;
    bool lane_has_point = false;
    for (uint local = lane; local < tile.record_count;
         local += BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1) {
        const Bls12381MetalMsmSignedRecord record =
            records[tile.record_offset + local];
        if ((record.flags & ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
            lane_valid = 0U;
            bls12_381_msm_status_record_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
        } else if (record.point_index >= parameters.base_count) {
            lane_valid = 0U;
            bls12_381_msm_status_record_error(
                status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
        } else {
            const Bls12381MetalMsmAffine point =
                bls12_381_msm_g1_signed_affine(
                    bases[record.point_index], record.flags);
            if (!lane_has_point) {
                accumulator = bls12_381_msm_homogeneous_from_signed_affine_v1(
                    point, 0U);
                lane_has_point = true;
            } else {
                accumulator = bls12_381_msm_g1_mixed_add(accumulator, point);
            }
        }
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(
        tile.record_count, BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1);
    const uint tree_span = bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = tree_span >> 1U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            partials[lane] = bls12_381_msm_g1_add(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0U) {
        // A malformed record never produces a usable partial even if a caller
        // neglects to inspect the status buffer.
        tile_roots[tile_index] = valid[0] != 0U
            ? partials[0]
            : bls12_381_msm_g1_identity();
    }
}

// Buffers:
//   0 homogeneous tile partials [tile_count]
//   1 tile descriptors [tile_count]
//   2 ordered run descriptors [run_count]
//   3 homogeneous run roots [run_count]
//   4 shared status (one Bls12381MetalMsmStatus)
//   5 the same Bls12381MetalMsmParallelBucketParametersV1
//
// Dispatch exactly run_count threadgroups of (32, 1, 1). Each lane reduces a
// strided subset before the same balanced 32-way tree. Threadgroup storage is
// again exactly 4,736 bytes, independent of the number of tiles in a run.
kernel void bls12_381_metal_msm_bucket_tile_partials_parallel_v1(
    device const Bls12381MetalMsmHomogeneous* tile_roots [[buffer(0)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(1)]],
    device const Bls12381MetalMsmTileRunV1* runs [[buffer(2)]],
    device Bls12381MetalMsmHomogeneous* run_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    if (group.y != 0U || group.z != 0U) {
        return;
    }
    const uint run_index = group.x;
    if (run_index >= parameters.run_count) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    threadgroup Bls12381MetalMsmHomogeneous partials[32];
    threadgroup uint valid[32];

    const bool launch_valid =
        bls12_381_msm_parallel_launch_valid_v1(
            parameters, threads, groups, parameters.run_count);
    Bls12381MetalMsmTileRunV1 run;
    if (launch_valid) {
        run = runs[run_index];
    } else {
        run.tile_offset = 0U;
        run.tile_count = 0U;
        run.reserved_0 = 0U;
        run.reserved_1 = 0U;
    }

    bool descriptor_valid = launch_valid &&
        (run.reserved_0 | run.reserved_1) == 0U &&
        run.tile_count != 0U &&
        bls12_381_msm_range_valid(
            run.tile_offset, run.tile_count, parameters.tile_count);
    if (descriptor_valid) {
        if (run_index == 0U) {
            descriptor_valid = run.tile_offset == 0U;
        } else {
            const Bls12381MetalMsmTileRunV1 previous = runs[run_index - 1U];
            const bool previous_valid =
                (previous.reserved_0 | previous.reserved_1) == 0U &&
                previous.tile_count != 0U &&
                bls12_381_msm_range_valid(
                    previous.tile_offset,
                    previous.tile_count,
                    parameters.tile_count);
            descriptor_valid = previous_valid &&
                run.tile_offset == previous.tile_offset + previous.tile_count;
        }
        if (run_index + 1U == parameters.run_count) {
            descriptor_valid = descriptor_valid &&
                run.tile_offset + run.tile_count == parameters.tile_count;
        }
    }

    // Uniform early exit: no thread reaches a barrier alone.
    if (!descriptor_valid) {
        if (lane == 0U) {
            run_roots[run_index] = bls12_381_msm_g1_identity();
            bls12_381_msm_status_descriptor_error(
                status,
                launch_valid
                    ? BLS12_381_METAL_MSM_STATUS_INVALID_RUN
                    : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    // Most dense Pippenger buckets fit in one tile. Copying that
    // partial avoids 127 identity additions and all threadgroup barriers.
    if (run.tile_count == 1U) {
        if (lane == 0U) {
            const uint tile_index = run.tile_offset;
            if (bls12_381_msm_bucket_tile_for_run_valid_v1(
                    tiles, tile_index, parameters, run_index)) {
                run_roots[run_index] = tile_roots[tile_index];
            } else {
                run_roots[run_index] = bls12_381_msm_g1_identity();
                bls12_381_msm_status_descriptor_error(
                    status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
            }
        }
        return;
    }

    Bls12381MetalMsmHomogeneous accumulator = bls12_381_msm_g1_identity();
    uint lane_valid = 1U;
    bool lane_has_partial = false;
    for (uint local = lane; local < run.tile_count;
         local += BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1) {
        const uint tile_index = run.tile_offset + local;
        const bool tile_valid = bls12_381_msm_bucket_tile_for_run_valid_v1(
            tiles, tile_index, parameters, run_index);
        if (!tile_valid) {
            lane_valid = 0U;
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        } else if (!lane_has_partial) {
            accumulator = tile_roots[tile_index];
            lane_has_partial = true;
        } else {
            accumulator = bls12_381_msm_g1_add(
                accumulator, tile_roots[tile_index]);
        }
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(
        run.tile_count, BLS12_381_METAL_MSM_PARALLEL_BUCKET_THREADS_V1);
    const uint tree_span = bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = tree_span >> 1U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            partials[lane] = bls12_381_msm_g1_add(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0U) {
        run_roots[run_index] = valid[0] != 0U
            ? partials[0]
            : bls12_381_msm_g1_identity();
    }
}

#endif
