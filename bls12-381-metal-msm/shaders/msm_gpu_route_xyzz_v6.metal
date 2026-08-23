// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Compact long-bucket tiling over the validated v3 router and v5 XYZZ path.
// Runs of at most 4096 records stay on v5. Longer runs reserve only the tiles
// they need, reduce those tiles through an indirect dispatch, and combine the
// contiguous partial span back into the ordinary dense bucket buffer.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V6
#define BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V6

#include "msm_gpu_route_xyzz_v5.metal"

constant uint BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS = 4096U;
constant uint BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS = 4096U;

kernel void bls12_381_metal_msm_gpu_build_long_tiles_xyzz_v6(
    device const atomic_uint* histogram [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device uint* tile_offsets [[buffer(2)]],
    device uint* tile_counts [[buffer(3)]],
    device Bls12381MetalMsmBucketTileV1* tiles [[buffer(4)]],
    device atomic_uint* indirect_groups [[buffer(5)]],
    device Bls12381MetalMsmStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(7)]],
    constant uint& tile_capacity [[buffer(8)]],
    uint dense [[thread_position_in_grid]])
{
    if (dense >= parameters.dense_bucket_count) {
        return;
    }
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters) ||
        tile_capacity == 0U) {
        if (dense == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint count = atomic_load_explicit(
        &histogram[dense], memory_order_relaxed);
    if (count <= BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS) {
        return;
    }
    const uint offset = offsets[dense];
    if (!bls12_381_msm_range_valid(
            offset, count, parameters.max_record_count)) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }

    const uint count_tiles =
        (count + BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS - 1U) /
        BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS;
    const uint first_tile = atomic_fetch_add_explicit(
        &indirect_groups[0], count_tiles, memory_order_relaxed);
    if (first_tile > tile_capacity ||
        count_tiles > tile_capacity - first_tile) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }

    tile_offsets[dense] = first_tile;
    tile_counts[dense] = count_tiles;
    for (uint tile = 0U; tile < count_tiles; ++tile) {
        const uint local_offset =
            tile * BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS;
        Bls12381MetalMsmBucketTileV1 descriptor;
        descriptor.record_offset = offset + local_offset;
        descriptor.record_count = min(
            BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS,
            count - local_offset);
        descriptor.run_index = dense;
        descriptor.reserved = 0U;
        tiles[first_tile + tile] = descriptor;
    }
}

kernel void bls12_381_metal_msm_gpu_finalize_long_tiles_xyzz_v6(
    device atomic_uint* indirect_groups [[buffer(0)]],
    device Bls12381MetalMsmStatus* status [[buffer(1)]],
    constant uint& tile_capacity [[buffer(2)]],
    uint position [[thread_position_in_grid]])
{
    if (position != 0U) {
        return;
    }
    const uint count = atomic_load_explicit(
        &indirect_groups[0], memory_order_relaxed);
    if (count > tile_capacity) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        atomic_store_explicit(&indirect_groups[0], 0U, memory_order_relaxed);
    }
    atomic_store_explicit(&indirect_groups[1], 1U, memory_order_relaxed);
    atomic_store_explicit(&indirect_groups[2], 1U, memory_order_relaxed);
}

inline __attribute__((always_inline))
void bls12_381_msm_gpu_reduce_short_xyzz_impl_v6(
    device const Bls12381MetalMsmAffine* bases,
    device const Bls12381MetalMsmSignedRecord* records,
    device const atomic_uint* histogram,
    device const uint* offsets,
    device Bls12381MetalMsmXyzzV4* bucket_roots,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters,
    threadgroup Bls12381MetalMsmXyzzV4* partials,
    threadgroup uint* valid,
    uint lane,
    uint3 threads,
    uint3 groups,
    uint3 group,
    uint expected_threads,
    uint long_bucket_records)
{
    const uint buckets_per_group = 2U;
    const bool launch_valid =
        group.y == 0U && group.z == 0U &&
        bls12_381_msm_gpu_route_parameters_valid_v3(parameters) &&
        parameters.threads_per_group == expected_threads &&
        parameters.buckets_per_group == buckets_per_group &&
        threads.x == expected_threads && threads.y == 1U && threads.z == 1U &&
        groups.x == (parameters.dense_bucket_count + 1U) / buckets_per_group &&
        groups.y == 1U && groups.z == 1U;
    if (!launch_valid) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint dense_base = group.x * buckets_per_group;
    const uint first_count = dense_base < parameters.dense_bucket_count
        ? atomic_load_explicit(&histogram[dense_base], memory_order_relaxed)
        : 0U;
    const uint second_count = dense_base + 1U < parameters.dense_bucket_count
        ? atomic_load_explicit(&histogram[dense_base + 1U], memory_order_relaxed)
        : 0U;
    const bool has_long_bucket =
        first_count > long_bucket_records ||
        second_count > long_bucket_records;
    if (!has_long_bucket) {
        bls12_381_msm_gpu_reduce_adaptive_xyzz_impl_v5(
            bases, records, histogram, offsets, bucket_roots, status,
            parameters, partials, valid, lane, threads, groups, group,
            expected_threads);
        return;
    }

    for (uint bucket_in_group = 0U;
         bucket_in_group < buckets_per_group;
         ++bucket_in_group) {
        const uint dense = dense_base + bucket_in_group;
        const bool dense_valid = dense < parameters.dense_bucket_count;
        const uint count = dense_valid
            ? atomic_load_explicit(&histogram[dense], memory_order_relaxed)
            : 0U;
        if (count > long_bucket_records) {
            if (lane == 0U) {
                bucket_roots[dense] = bls12_381_msm_xyzz_identity_v4();
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        const uint offset = dense_valid ? offsets[dense] : 0U;
        const bool range_valid = dense_valid && bls12_381_msm_range_valid(
            offset, count, parameters.max_record_count);
        Bls12381MetalMsmXyzzV4 accumulator =
            bls12_381_msm_xyzz_identity_v4();
        uint lane_valid = static_cast<uint>(!dense_valid || range_valid);
        if (range_valid && count != 0U) {
            for (uint local = lane; local < count; local += expected_threads) {
                const Bls12381MetalMsmSignedRecord record =
                    records[offset + local];
                if ((record.flags &
                     ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
                    lane_valid = 0U;
                    bls12_381_msm_status_record_error(
                        status,
                        BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
                } else if (record.point_index >= parameters.base_count) {
                    lane_valid = 0U;
                    bls12_381_msm_status_record_error(
                        status,
                        BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
                } else {
                    const Bls12381MetalMsmAffine point =
                        bls12_381_msm_g1_signed_affine(
                            bases[record.point_index], record.flags);
                    accumulator = bls12_381_msm_xyzz_mixed_add_v4(
                        accumulator, point);
                }
            }
        } else if (dense_valid && count != 0U && lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        }
        partials[lane] = accumulator;
        valid[lane] = lane_valid;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint active_lanes = min(count, expected_threads);
        const uint tree_span = count == 0U
            ? 1U
            : bls12_381_msm_parallel_tree_span_v1(active_lanes);
        for (uint stride = expected_threads >> 1U;
             stride != 0U;
             stride >>= 1U) {
            if (lane < stride && lane + stride < tree_span) {
                partials[lane] = bls12_381_msm_xyzz_add_v4(
                    partials[lane], partials[lane + stride]);
                valid[lane] &= valid[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (dense_valid && lane == 0U) {
            bucket_roots[dense] = count != 0U && valid[0] != 0U
                ? partials[0]
                : bls12_381_msm_xyzz_identity_v4();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_small_v6(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmXyzzV4 partials[32];
    threadgroup uint valid[32];
    bls12_381_msm_gpu_reduce_short_xyzz_impl_v6(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_large_v6(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmXyzzV4 partials[64];
    threadgroup uint valid[64];
    bls12_381_msm_gpu_reduce_short_xyzz_impl_v6(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

inline __attribute__((always_inline))
void bls12_381_msm_gpu_reduce_long_tile_xyzz_impl_v6(
    device const Bls12381MetalMsmAffine* bases,
    device const Bls12381MetalMsmSignedRecord* records,
    device const Bls12381MetalMsmBucketTileV1* tiles,
    device Bls12381MetalMsmXyzzV4* tile_roots,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters,
    constant uint& tile_capacity,
    threadgroup Bls12381MetalMsmXyzzV4* partials,
    threadgroup uint* valid,
    uint lane,
    uint3 threads,
    uint3 groups,
    uint3 group,
    uint expected_threads)
{
    const uint tile_index = group.x;
    const bool launch_valid =
        group.y == 0U && group.z == 0U && groups.y == 1U && groups.z == 1U &&
        threads.x == expected_threads && threads.y == 1U && threads.z == 1U &&
        parameters.threads_per_group == expected_threads &&
        (parameters.buckets_per_group == 2U ||
         parameters.buckets_per_group == 4U ||
         parameters.buckets_per_group == 8U ||
         parameters.buckets_per_group == 16U) &&
        tile_index < tile_capacity &&
        bls12_381_msm_gpu_route_parameters_valid_v3(parameters);
    if (!launch_valid) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const Bls12381MetalMsmBucketTileV1 tile = tiles[tile_index];
    const bool descriptor_valid =
        tile.reserved == 0U &&
        tile.run_index < parameters.dense_bucket_count &&
        tile.record_count != 0U &&
        tile.record_count <= BLS12_381_METAL_MSM_XYZZ_V6_TILE_RECORDS &&
        bls12_381_msm_range_valid(
            tile.record_offset,
            tile.record_count,
            parameters.max_record_count);
    Bls12381MetalMsmXyzzV4 accumulator =
        bls12_381_msm_xyzz_identity_v4();
    uint lane_valid = static_cast<uint>(descriptor_valid);
    if (descriptor_valid) {
        for (uint local = lane;
             local < tile.record_count;
             local += expected_threads) {
            const Bls12381MetalMsmSignedRecord record =
                records[tile.record_offset + local];
            if ((record.flags &
                 ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status,
                    BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
            } else if (record.point_index >= parameters.base_count) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
            } else {
                const Bls12381MetalMsmAffine point =
                    bls12_381_msm_g1_signed_affine(
                        bases[record.point_index], record.flags);
                accumulator = bls12_381_msm_xyzz_mixed_add_v4(
                    accumulator, point);
            }
        }
    } else if (lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(tile.record_count, expected_threads);
    const uint tree_span = descriptor_valid
        ? bls12_381_msm_parallel_tree_span_v1(active_lanes)
        : 1U;
    for (uint stride = expected_threads >> 1U;
         stride != 0U;
         stride >>= 1U) {
        if (lane < stride && lane + stride < tree_span) {
            partials[lane] = bls12_381_msm_xyzz_add_v4(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        tile_roots[tile_index] = valid[0] != 0U
            ? partials[0]
            : bls12_381_msm_xyzz_identity_v4();
    }
}

kernel void bls12_381_metal_msm_gpu_reduce_long_tiles_xyzz_small_v6(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmXyzzV4* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmXyzzV4 partials[32];
    threadgroup uint valid[32];
    bls12_381_msm_gpu_reduce_long_tile_xyzz_impl_v6(
        bases, records, tiles, tile_roots, status, parameters, tile_capacity,
        partials, valid, lane, threads, groups, group, 32U);
}

kernel void bls12_381_metal_msm_gpu_reduce_long_tiles_xyzz_large_v6(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmXyzzV4* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmXyzzV4 partials[64];
    threadgroup uint valid[64];
    bls12_381_msm_gpu_reduce_long_tile_xyzz_impl_v6(
        bases, records, tiles, tile_roots, status, parameters, tile_capacity,
        partials, valid, lane, threads, groups, group, 64U);
}

kernel void bls12_381_metal_msm_gpu_combine_long_tiles_xyzz_v6(
    device const uint* tile_offsets [[buffer(0)]],
    device const uint* tile_counts [[buffer(1)]],
    device const Bls12381MetalMsmXyzzV4* tile_roots [[buffer(2)]],
    device Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint dense [[thread_position_in_grid]])
{
    if (dense >= parameters.dense_bucket_count) {
        return;
    }
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        if (dense == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const uint count = tile_counts[dense];
    if (count == 0U) {
        return;
    }
    const uint offset = tile_offsets[dense];
    if (offset > tile_capacity || count > tile_capacity - offset) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }
    Bls12381MetalMsmXyzzV4 root = bls12_381_msm_xyzz_identity_v4();
    for (uint tile = 0U; tile < count; ++tile) {
        root = bls12_381_msm_xyzz_add_v4(
            root, tile_roots[offset + tile]);
    }
    bucket_roots[dense] = root;
}

#endif
