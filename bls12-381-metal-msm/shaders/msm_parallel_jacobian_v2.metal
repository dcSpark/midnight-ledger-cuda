// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Branch-completed Jacobian bucket reduction, private shader ABI version 2.
// The trusted-local observation profile permits the exceptional-case branches.
// Ordinary mixed additions use 7M+4S instead of the constant-schedule RCB
// mixed formula; identity, doubling, and inverse cases remain exact.

#ifndef BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_V2
#define BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_V2

#include "msm_parallel_bucket_v1.metal"

constant uint BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_ABI_V2 = 2U;
constant uint BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_THREADS_V2 = 32U;
constant uint BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_TILE_RECORDS_V2 = 128U;
constant uint BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_THREADS_V2 = 64U;
constant uint BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_TILE_RECORDS_V2 = 256U;

inline __attribute__((always_inline))
bool bls12_381_msm_jacobian_fp_is_zero_v2(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_zero_mask(value) != 0UL;
}

inline __attribute__((always_inline))
bool bls12_381_msm_jacobian_fp_is_exact_zero_v2(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_exact_zero_mask(value) != 0UL;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_identity_v2()
{
    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_zero();
    result.y = bls12_381_msm_fp_one();
    result.z = bls12_381_msm_fp_zero();
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_canonicalize_v2(
    Bls12381MetalMsmJacobian point)
{
    point.x = bls12_381_msm_fp_reduce_once(point.x);
    point.y = bls12_381_msm_fp_reduce_once(point.y);
    point.z = bls12_381_msm_fp_reduce_once(point.z);
    return point;
}

inline __attribute__((always_inline))
bool bls12_381_msm_jacobian_is_identity_v2(Bls12381MetalMsmJacobian point)
{
    return bls12_381_msm_jacobian_fp_is_exact_zero_v2(point.z);
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_from_signed_affine_v2(
    Bls12381MetalMsmAffine point,
    uint flags)
{
    if (bls12_381_msm_g1_affine_identity_mask(point) != 0UL) {
        return bls12_381_msm_jacobian_identity_v2();
    }
    point = bls12_381_msm_g1_signed_affine(point, flags);
    Bls12381MetalMsmJacobian result;
    result.x = point.x;
    result.y = point.y;
    result.z = bls12_381_msm_fp_one();
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_double_v2(
    Bls12381MetalMsmJacobian point)
{
    if (bls12_381_msm_jacobian_is_identity_v2(point) ||
        bls12_381_msm_jacobian_fp_is_exact_zero_v2(point.y)) {
        return bls12_381_msm_jacobian_identity_v2();
    }

    // dbl-2009-l for a=0.
    const Bls12381MetalMsmFp a = bls12_381_msm_fp_square(point.x);
    const Bls12381MetalMsmFp b = bls12_381_msm_fp_square(point.y);
    const Bls12381MetalMsmFp c = bls12_381_msm_fp_square(b);
    Bls12381MetalMsmFp d = bls12_381_msm_fp_square(
        bls12_381_msm_fp_add(point.x, b));
    d = bls12_381_msm_fp_double(
        bls12_381_msm_fp_sub(bls12_381_msm_fp_sub(d, a), c));
    const Bls12381MetalMsmFp e = bls12_381_msm_fp_triple(a);
    const Bls12381MetalMsmFp f = bls12_381_msm_fp_square(e);

    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_sub(
        f, bls12_381_msm_fp_double(d));
    result.y = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(
            e, bls12_381_msm_fp_sub(d, result.x)),
        bls12_381_msm_fp_times_eight(c));
    result.z = bls12_381_msm_fp_double(
        bls12_381_msm_fp_mul(point.y, point.z));
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_mixed_add_v2(
    Bls12381MetalMsmJacobian lhs,
    Bls12381MetalMsmAffine rhs)
{
    if (bls12_381_msm_g1_affine_identity_mask(rhs) != 0UL) {
        return lhs;
    }
    if (bls12_381_msm_jacobian_is_identity_v2(lhs)) {
        return bls12_381_msm_jacobian_from_signed_affine_v2(rhs, 0U);
    }

    // madd-2007-bl. H/r detect the only exceptional cases before the
    // incomplete ordinary formula is evaluated.
    const Bls12381MetalMsmFp z1z1 =
        bls12_381_msm_fp_square(lhs.z);
    const Bls12381MetalMsmFp u2 =
        bls12_381_msm_fp_mul(rhs.x, z1z1);
    const Bls12381MetalMsmFp s2 = bls12_381_msm_fp_mul(
        bls12_381_msm_fp_mul(rhs.y, lhs.z), z1z1);
    const Bls12381MetalMsmFp h =
        bls12_381_msm_fp_sub(u2, lhs.x);
    const Bls12381MetalMsmFp r0 =
        bls12_381_msm_fp_sub(s2, lhs.y);
    if (bls12_381_msm_jacobian_fp_is_zero_v2(h)) {
        return bls12_381_msm_jacobian_fp_is_zero_v2(r0)
            ? bls12_381_msm_jacobian_double_v2(lhs)
            : bls12_381_msm_jacobian_identity_v2();
    }

    const Bls12381MetalMsmFp hh = bls12_381_msm_fp_square(h);
    const Bls12381MetalMsmFp i =
        bls12_381_msm_fp_double(bls12_381_msm_fp_double(hh));
    const Bls12381MetalMsmFp j = bls12_381_msm_fp_mul(h, i);
    const Bls12381MetalMsmFp r = bls12_381_msm_fp_double(r0);
    const Bls12381MetalMsmFp v = bls12_381_msm_fp_mul(lhs.x, i);

    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(
            bls12_381_msm_fp_square(r), j),
        bls12_381_msm_fp_double(v));
    result.y = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(
            r, bls12_381_msm_fp_sub(v, result.x)),
        bls12_381_msm_fp_double(
            bls12_381_msm_fp_mul(lhs.y, j)));
    result.z = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(
            bls12_381_msm_fp_square(
                bls12_381_msm_fp_add(lhs.z, h)),
            z1z1),
        hh);
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_jacobian_add_v2(
    Bls12381MetalMsmJacobian lhs,
    Bls12381MetalMsmJacobian rhs)
{
    if (bls12_381_msm_jacobian_is_identity_v2(lhs)) {
        return rhs;
    }
    if (bls12_381_msm_jacobian_is_identity_v2(rhs)) {
        return lhs;
    }

    // add-2007-bl. Projective equality/inversion is detected through the
    // cross-multiplied affine coordinates.
    const Bls12381MetalMsmFp z1z1 =
        bls12_381_msm_fp_square(lhs.z);
    const Bls12381MetalMsmFp z2z2 =
        bls12_381_msm_fp_square(rhs.z);
    const Bls12381MetalMsmFp u1 =
        bls12_381_msm_fp_mul(lhs.x, z2z2);
    const Bls12381MetalMsmFp u2 =
        bls12_381_msm_fp_mul(rhs.x, z1z1);
    const Bls12381MetalMsmFp s1 = bls12_381_msm_fp_mul(
        bls12_381_msm_fp_mul(lhs.y, rhs.z), z2z2);
    const Bls12381MetalMsmFp s2 = bls12_381_msm_fp_mul(
        bls12_381_msm_fp_mul(rhs.y, lhs.z), z1z1);
    const Bls12381MetalMsmFp h =
        bls12_381_msm_fp_sub(u2, u1);
    const Bls12381MetalMsmFp r0 =
        bls12_381_msm_fp_sub(s2, s1);
    if (bls12_381_msm_jacobian_fp_is_zero_v2(h)) {
        return bls12_381_msm_jacobian_fp_is_zero_v2(r0)
            ? bls12_381_msm_jacobian_double_v2(lhs)
            : bls12_381_msm_jacobian_identity_v2();
    }

    const Bls12381MetalMsmFp hh = bls12_381_msm_fp_square(h);
    const Bls12381MetalMsmFp i =
        bls12_381_msm_fp_double(bls12_381_msm_fp_double(hh));
    const Bls12381MetalMsmFp j = bls12_381_msm_fp_mul(h, i);
    const Bls12381MetalMsmFp r = bls12_381_msm_fp_double(r0);
    const Bls12381MetalMsmFp v = bls12_381_msm_fp_mul(u1, i);

    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(
            bls12_381_msm_fp_square(r), j),
        bls12_381_msm_fp_double(v));
    result.y = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(
            r, bls12_381_msm_fp_sub(v, result.x)),
        bls12_381_msm_fp_double(
            bls12_381_msm_fp_mul(s1, j)));
    result.z = bls12_381_msm_fp_mul(
        bls12_381_msm_fp_sub(
            bls12_381_msm_fp_sub(
                bls12_381_msm_fp_square(
                    bls12_381_msm_fp_add(lhs.z, rhs.z)),
                z1z1),
            z2z2),
        h);
    return result;
}

inline __attribute__((always_inline))
bool bls12_381_msm_bucket_tile_payload_valid_v2(
    Bls12381MetalMsmBucketTileV1 tile,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint max_records_per_tile)
{
    return tile.reserved == 0U && tile.run_index < parameters.run_count &&
        tile.record_count != 0U &&
        tile.record_count <= max_records_per_tile &&
        bls12_381_msm_range_valid(
            tile.record_offset, tile.record_count, parameters.record_count);
}

inline __attribute__((always_inline))
bool bls12_381_msm_bucket_tile_partition_valid_v2(
    device const Bls12381MetalMsmBucketTileV1* tiles,
    uint tile_index,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint max_records_per_tile)
{
    const Bls12381MetalMsmBucketTileV1 tile = tiles[tile_index];
    bool valid = bls12_381_msm_bucket_tile_payload_valid_v2(
        tile, parameters, max_records_per_tile);
    if (!valid) {
        return false;
    }
    if (tile_index == 0U) {
        valid = tile.record_offset == 0U && tile.run_index == 0U;
    } else {
        const Bls12381MetalMsmBucketTileV1 previous = tiles[tile_index - 1U];
        const bool previous_valid =
            bls12_381_msm_bucket_tile_payload_valid_v2(
                previous, parameters, max_records_per_tile);
        valid = previous_valid &&
            (tile.run_index == previous.run_index ||
             tile.run_index == previous.run_index + 1U) &&
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
bool bls12_381_msm_bucket_tile_for_run_valid_v2(
    device const Bls12381MetalMsmBucketTileV1* tiles,
    uint tile_index,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint expected_run,
    uint max_records_per_tile)
{
    return tiles[tile_index].run_index == expected_run &&
        bls12_381_msm_bucket_tile_partition_valid_v2(
            tiles, tile_index, parameters, max_records_per_tile);
}

inline __attribute__((always_inline))
bool bls12_381_msm_parallel_launch_valid_v2(
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    uint3 threads,
    uint3 groups,
    uint expected_groups,
    uint expected_threads,
    uint expected_records_per_tile)
{
    return parameters.abi_version ==
            BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_ABI_V2 &&
        parameters.threads_per_threadgroup == expected_threads &&
        parameters.max_records_per_tile == expected_records_per_tile &&
        parameters.reserved == 0U &&
        threads.x == expected_threads &&
        threads.y == 1U && threads.z == 1U &&
        groups.x == expected_groups && groups.y == 1U && groups.z == 1U;
}

inline __attribute__((always_inline))
void bls12_381_msm_bucket_tiles_jacobian_impl_v2(
    device const Bls12381MetalMsmAffine* bases,
    device const Bls12381MetalMsmSignedRecord* records,
    device const Bls12381MetalMsmBucketTileV1* tiles,
    device Bls12381MetalMsmJacobian* tile_roots,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    threadgroup Bls12381MetalMsmJacobian* partials,
    threadgroup uint* valid,
    uint lane,
    uint3 threads,
    uint3 groups,
    uint3 group,
    uint expected_threads,
    uint expected_records_per_tile)
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

    const bool launch_valid = bls12_381_msm_parallel_launch_valid_v2(
        parameters,
        threads,
        groups,
        parameters.tile_count,
        expected_threads,
        expected_records_per_tile);
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
        bls12_381_msm_bucket_tile_partition_valid_v2(
            tiles, tile_index, parameters, expected_records_per_tile);
    if (!descriptor_valid) {
        if (lane == 0U) {
            tile_roots[tile_index] =
                bls12_381_msm_jacobian_identity_v2();
            bls12_381_msm_status_descriptor_error(
                status,
                launch_valid
                    ? BLS12_381_METAL_MSM_STATUS_INVALID_RUN
                    : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    Bls12381MetalMsmJacobian accumulator =
        bls12_381_msm_jacobian_identity_v2();
    uint lane_valid = 1U;
    for (uint local = lane; local < tile.record_count;
         local += expected_threads) {
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
            accumulator =
                bls12_381_msm_jacobian_mixed_add_v2(accumulator, point);
        }
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(
        tile.record_count, expected_threads);
    const uint tree_span =
        bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = tree_span >> 1U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            partials[lane] = bls12_381_msm_jacobian_add_v2(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        tile_roots[tile_index] = valid[0] != 0U
            ? partials[0]
            : bls12_381_msm_jacobian_identity_v2();
    }
}

kernel void bls12_381_metal_msm_bucket_tiles_jacobian_small_v2(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmJacobian* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[32];
    threadgroup uint valid[32];
    bls12_381_msm_bucket_tiles_jacobian_impl_v2(
        bases,
        records,
        tiles,
        tile_roots,
        status,
        parameters,
        partials,
        valid,
        lane,
        threads,
        groups,
        group,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_THREADS_V2,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_TILE_RECORDS_V2);
}

kernel void bls12_381_metal_msm_bucket_tiles_jacobian_large_v2(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmJacobian* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[64];
    threadgroup uint valid[64];
    bls12_381_msm_bucket_tiles_jacobian_impl_v2(
        bases,
        records,
        tiles,
        tile_roots,
        status,
        parameters,
        partials,
        valid,
        lane,
        threads,
        groups,
        group,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_THREADS_V2,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_TILE_RECORDS_V2);
}

inline __attribute__((always_inline))
void bls12_381_msm_bucket_partials_jacobian_impl_v2(
    device const Bls12381MetalMsmJacobian* tile_roots,
    device const Bls12381MetalMsmBucketTileV1* tiles,
    device const Bls12381MetalMsmTileRunV1* runs,
    device Bls12381MetalMsmJacobian* run_roots,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters,
    threadgroup Bls12381MetalMsmJacobian* partials,
    threadgroup uint* valid,
    uint lane,
    uint3 threads,
    uint3 groups,
    uint3 group,
    uint expected_threads,
    uint expected_records_per_tile)
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

    const bool launch_valid = bls12_381_msm_parallel_launch_valid_v2(
        parameters,
        threads,
        groups,
        parameters.run_count,
        expected_threads,
        expected_records_per_tile);
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
            descriptor_valid =
                (previous.reserved_0 | previous.reserved_1) == 0U &&
                previous.tile_count != 0U &&
                bls12_381_msm_range_valid(
                    previous.tile_offset,
                    previous.tile_count,
                    parameters.tile_count) &&
                run.tile_offset ==
                    previous.tile_offset + previous.tile_count;
        }
        if (run_index + 1U == parameters.run_count) {
            descriptor_valid = descriptor_valid &&
                run.tile_offset + run.tile_count == parameters.tile_count;
        }
    }
    if (!descriptor_valid) {
        if (lane == 0U) {
            run_roots[run_index] =
                bls12_381_msm_jacobian_identity_v2();
            bls12_381_msm_status_descriptor_error(
                status,
                launch_valid
                    ? BLS12_381_METAL_MSM_STATUS_INVALID_RUN
                    : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (run.tile_count == 1U) {
        if (lane == 0U) {
            const uint tile_index = run.tile_offset;
            if (bls12_381_msm_bucket_tile_for_run_valid_v2(
                    tiles,
                    tile_index,
                    parameters,
                    run_index,
                    expected_records_per_tile)) {
                run_roots[run_index] = tile_roots[tile_index];
            } else {
                run_roots[run_index] =
                    bls12_381_msm_jacobian_identity_v2();
                bls12_381_msm_status_descriptor_error(
                    status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
            }
        }
        return;
    }

    Bls12381MetalMsmJacobian accumulator =
        bls12_381_msm_jacobian_identity_v2();
    uint lane_valid = 1U;
    for (uint local = lane; local < run.tile_count;
         local += expected_threads) {
        const uint tile_index = run.tile_offset + local;
        if (!bls12_381_msm_bucket_tile_for_run_valid_v2(
                tiles,
                tile_index,
                parameters,
                run_index,
                expected_records_per_tile)) {
            lane_valid = 0U;
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        } else {
            accumulator = bls12_381_msm_jacobian_add_v2(
                accumulator, tile_roots[tile_index]);
        }
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(
        run.tile_count, expected_threads);
    const uint tree_span =
        bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = tree_span >> 1U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            partials[lane] = bls12_381_msm_jacobian_add_v2(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        run_roots[run_index] = valid[0] != 0U
            ? partials[0]
            : bls12_381_msm_jacobian_identity_v2();
    }
}

kernel void bls12_381_metal_msm_bucket_partials_jacobian_small_v2(
    device const Bls12381MetalMsmJacobian* tile_roots [[buffer(0)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(1)]],
    device const Bls12381MetalMsmTileRunV1* runs [[buffer(2)]],
    device Bls12381MetalMsmJacobian* run_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[32];
    threadgroup uint valid[32];
    bls12_381_msm_bucket_partials_jacobian_impl_v2(
        tile_roots,
        tiles,
        runs,
        run_roots,
        status,
        parameters,
        partials,
        valid,
        lane,
        threads,
        groups,
        group,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_THREADS_V2,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_SMALL_TILE_RECORDS_V2);
}

kernel void bls12_381_metal_msm_bucket_partials_jacobian_large_v2(
    device const Bls12381MetalMsmJacobian* tile_roots [[buffer(0)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(1)]],
    device const Bls12381MetalMsmTileRunV1* runs [[buffer(2)]],
    device Bls12381MetalMsmJacobian* run_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmParallelBucketParametersV1& parameters
        [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[64];
    threadgroup uint valid[64];
    bls12_381_msm_bucket_partials_jacobian_impl_v2(
        tile_roots,
        tiles,
        runs,
        run_roots,
        status,
        parameters,
        partials,
        valid,
        lane,
        threads,
        groups,
        group,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_THREADS_V2,
        BLS12_381_METAL_MSM_PARALLEL_JACOBIAN_LARGE_TILE_RECORDS_V2);
}

kernel void bls12_381_metal_msm_integrate_windows_jacobian_v2(
    device const Bls12381MetalMsmJacobian* run_roots [[buffer(0)]],
    device const uint* dense_run_map [[buffer(1)]],
    device const Bls12381MetalMsmWindowDescriptor* windows [[buffer(2)]],
    device Bls12381MetalMsmJacobian* window_points [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmIntegrationParameters& parameters [[buffer(5)]],
    uint window_index [[thread_position_in_grid]])
{
    if (window_index >= parameters.window_count) {
        return;
    }
    const Bls12381MetalMsmWindowDescriptor descriptor =
        windows[window_index];
    const bool span_valid = descriptor.bucket_count != 0U &&
        bls12_381_msm_range_valid(
            descriptor.dense_map_offset,
            descriptor.bucket_count,
            parameters.dense_map_count);
    const bool output_valid =
        descriptor.window_point_index < parameters.window_point_count;
    if (output_valid) {
        window_points[descriptor.window_point_index] =
            bls12_381_msm_jacobian_identity_v2();
    }
    if (!span_valid || !output_valid || descriptor.reserved != 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_WINDOW);
        return;
    }

    Bls12381MetalMsmJacobian running =
        bls12_381_msm_jacobian_identity_v2();
    Bls12381MetalMsmJacobian weighted =
        bls12_381_msm_jacobian_identity_v2();
    for (uint remaining = descriptor.bucket_count;
         remaining != 0U;
         --remaining) {
        const uint map_index =
            descriptor.dense_map_offset + remaining - 1U;
        const uint run_index = dense_run_map[map_index];
        if (run_index != BLS12_381_METAL_MSM_DENSE_MAP_IDENTITY) {
            if (run_index >= parameters.run_count) {
                bls12_381_msm_status_descriptor_error(
                    status, BLS12_381_METAL_MSM_STATUS_INVALID_DENSE_MAP);
            } else {
                running = bls12_381_msm_jacobian_add_v2(
                    running, run_roots[run_index]);
            }
        }
        weighted =
            bls12_381_msm_jacobian_add_v2(weighted, running);
    }
    window_points[descriptor.window_point_index] = weighted;
}

kernel void bls12_381_metal_msm_collect_jacobian_v2(
    device const Bls12381MetalMsmJacobian* window_points [[buffer(0)]],
    device const Bls12381MetalMsmCollectDescriptor* descriptors [[buffer(1)]],
    device Bls12381MetalMsmJacobian* outputs [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmCollectParameters& parameters [[buffer(4)]],
    uint descriptor_index [[thread_position_in_grid]])
{
    if (descriptor_index >= parameters.descriptor_count) {
        return;
    }
    const Bls12381MetalMsmCollectDescriptor descriptor =
        descriptors[descriptor_index];
    const bool span_valid = descriptor.window_count != 0U &&
        bls12_381_msm_range_valid(
            descriptor.window_point_offset,
            descriptor.window_count,
            parameters.window_point_count);
    const bool bits_valid = descriptor.window_bits != 0U &&
        descriptor.window_bits <= 16U;
    const bool output_valid =
        descriptor.output_index < parameters.output_count;
    if (output_valid) {
        outputs[descriptor.output_index] =
            bls12_381_msm_jacobian_identity_v2();
    }
    if (!span_valid || !bits_valid || !output_valid ||
        parameters.reserved != 0U) {
        bls12_381_msm_status_descriptor_error(
            status,
            parameters.reserved == 0U
                ? BLS12_381_METAL_MSM_STATUS_INVALID_COLLECT
                : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        return;
    }

    Bls12381MetalMsmJacobian accumulator =
        bls12_381_msm_jacobian_identity_v2();
    for (uint remaining = descriptor.window_count;
         remaining != 0U;
         --remaining) {
        if (remaining != descriptor.window_count) {
            for (uint bit = 0U; bit < descriptor.window_bits; ++bit) {
                accumulator =
                    bls12_381_msm_jacobian_double_v2(accumulator);
            }
        }
        accumulator = bls12_381_msm_jacobian_add_v2(
            accumulator,
            window_points[
                descriptor.window_point_offset + remaining - 1U]);
    }
    outputs[descriptor.output_index] =
        bls12_381_msm_jacobian_canonicalize_v2(accumulator);
}

#endif
