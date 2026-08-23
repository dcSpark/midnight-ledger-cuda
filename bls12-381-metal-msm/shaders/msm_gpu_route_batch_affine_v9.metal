// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// GPU-routed batch-affine tile reduction. Each 128-record tile batches the
// first 64 point additions behind one global field inversion, then reduces the
// resulting affine pairs with the validated XYZZ tree. This is the standalone
// library's Metal-native counterpart to the inversion-amortized arithmetic
// used by high-throughput Pippenger implementations.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_BATCH_AFFINE_V9
#define BLS12_381_METAL_MSM_GPU_ROUTE_BATCH_AFFINE_V9

#include "msm_gpu_route_xyzz_v8.metal"
#include "msm_batch_affine.metal"

constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS = 64U;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS = 128U;

kernel void bls12_381_metal_msm_gpu_build_batch_affine_tiles_v9(
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
    if (count == 0U) {
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
        (count + BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS - 1U) /
        BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS;
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
            tile * BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS;
        Bls12381MetalMsmBucketTileV1 descriptor;
        descriptor.record_offset = offset + local_offset;
        descriptor.record_count = min(
            BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS,
            count - local_offset);
        descriptor.run_index = dense;
        descriptor.reserved = 0U;
        tiles[first_tile + tile] = descriptor;
    }
}

kernel void bls12_381_metal_msm_gpu_initialize_batch_affine_products_v9(
    device Bls12381MetalMsmFp* product_tree [[buffer(0)]],
    constant uint& tile_capacity [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < tile_capacity) {
        product_tree[index] = bls12_381_msm_batch_affine_fp32_one();
    }
}

inline __attribute__((always_inline))
Bls12381MetalMsmAffine bls12_381_msm_gpu_load_signed_affine_v9(
    device const Bls12381MetalMsmAffine* bases,
    device const Bls12381MetalMsmSignedRecord* records,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters,
    uint record_offset,
    bool active)
{
    if (!active) {
        return bls12_381_msm_batch_affine_point_identity();
    }
    const Bls12381MetalMsmSignedRecord record = records[record_offset];
    if ((record.flags & ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
        return bls12_381_msm_batch_affine_point_identity();
    }
    if (record.point_index >= parameters.base_count) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
        return bls12_381_msm_batch_affine_point_identity();
    }
    return bls12_381_msm_g1_signed_affine(
        bases[record.point_index], record.flags);
}

inline __attribute__((always_inline))
bool bls12_381_msm_gpu_batch_affine_tile_valid_v9(
    Bls12381MetalMsmBucketTileV1 tile,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters)
{
    return tile.reserved == 0U &&
        tile.run_index < parameters.dense_bucket_count &&
        tile.record_count != 0U &&
        tile.record_count <= BLS12_381_METAL_MSM_BATCH_AFFINE_V9_TILE_RECORDS &&
        bls12_381_msm_range_valid(
            tile.record_offset,
            tile.record_count,
            parameters.max_record_count);
}

kernel void bls12_381_metal_msm_gpu_prepare_batch_affine_tiles_v9(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmFp* product_tree [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint tile_index = group.x;
    const bool launch_valid =
        group.y == 0U && group.z == 0U &&
        threads.x == BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS &&
        threads.y == 1U && threads.z == 1U &&
        parameters.threads_per_group ==
            BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS &&
        parameters.buckets_per_group == 2U &&
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
        bls12_381_msm_gpu_batch_affine_tile_valid_v9(tile, parameters);
    if (!descriptor_valid && lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
    }
    const uint pair_offset = 2U * lane;
    const bool lhs_active = descriptor_valid && pair_offset < tile.record_count;
    const bool rhs_active = descriptor_valid && pair_offset + 1U < tile.record_count;
    const Bls12381MetalMsmAffine lhs =
        bls12_381_msm_gpu_load_signed_affine_v9(
            bases, records, status, parameters,
            tile.record_offset + pair_offset, lhs_active);
    const Bls12381MetalMsmAffine rhs =
        bls12_381_msm_gpu_load_signed_affine_v9(
            bases, records, status, parameters,
            tile.record_offset + pair_offset + 1U, rhs_active);

    threadgroup Bls12381MetalMsmFp tree[127];
    tree[63U + lane] = bls12_381_msm_batch_affine_adjusted_denominator(lhs, rhs);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 32U; node_count != 0U; node_count >>= 1U) {
        if (lane < node_count) {
            const uint node = node_count - 1U + lane;
            tree[node] = bls12_381_msm_fp_mul(
                tree[2U * node + 1U], tree[2U * node + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        product_tree[tile_index] = tree[0];
    }
}

kernel void bls12_381_metal_msm_gpu_finish_batch_affine_tiles_v9(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device const Bls12381MetalMsmFp* inverse_products [[buffer(3)]],
    device Bls12381MetalMsmXyzzV4* tile_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(6)]],
    constant uint& tile_capacity [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint tile_index = group.x;
    const bool launch_valid =
        group.y == 0U && group.z == 0U &&
        threads.x == BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS &&
        threads.y == 1U && threads.z == 1U &&
        parameters.threads_per_group ==
            BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS &&
        parameters.buckets_per_group == 2U &&
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
        bls12_381_msm_gpu_batch_affine_tile_valid_v9(tile, parameters);
    if (!descriptor_valid && lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
    }
    const uint pair_offset = 2U * lane;
    const bool lhs_active = descriptor_valid && pair_offset < tile.record_count;
    const bool rhs_active = descriptor_valid && pair_offset + 1U < tile.record_count;
    // The product tree and XYZZ partials have disjoint lifetimes. Aliasing them
    // keeps this kernel below the two-resident-threadgroup memory threshold.
    threadgroup Bls12381MetalMsmXyzzV4 partials[64];
    threadgroup Bls12381MetalMsmFp* tree =
        reinterpret_cast<threadgroup Bls12381MetalMsmFp*>(partials);
    {
        const Bls12381MetalMsmAffine lhs =
            bls12_381_msm_gpu_load_signed_affine_v9(
                bases, records, status, parameters,
                tile.record_offset + pair_offset, lhs_active);
        const Bls12381MetalMsmAffine rhs =
            bls12_381_msm_gpu_load_signed_affine_v9(
                bases, records, status, parameters,
                tile.record_offset + pair_offset + 1U, rhs_active);
        tree[63U + lane] =
            bls12_381_msm_batch_affine_adjusted_denominator(lhs, rhs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 32U; node_count != 0U; node_count >>= 1U) {
        if (lane < node_count) {
            const uint node = node_count - 1U + lane;
            tree[node] = bls12_381_msm_fp_mul(
                tree[2U * node + 1U], tree[2U * node + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        tree[0] = inverse_products[tile_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 1U;
         node_count < BLS12_381_METAL_MSM_BATCH_AFFINE_V9_THREADS;
         node_count <<= 1U) {
        if (lane < node_count) {
            const uint node = node_count - 1U + lane;
            const uint left = 2U * node + 1U;
            const uint right = left + 1U;
            const Bls12381MetalMsmFp parent_inverse = tree[node];
            const Bls12381MetalMsmFp left_product = tree[left];
            const Bls12381MetalMsmFp right_product = tree[right];
            tree[left] = bls12_381_msm_fp_mul(parent_inverse, right_product);
            tree[right] = bls12_381_msm_fp_mul(parent_inverse, left_product);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const Bls12381MetalMsmFp inverse_denominator = tree[63U + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const Bls12381MetalMsmAffine lhs =
        bls12_381_msm_gpu_load_signed_affine_v9(
            bases, records, status, parameters,
            tile.record_offset + pair_offset, lhs_active);
    const Bls12381MetalMsmAffine rhs =
        bls12_381_msm_gpu_load_signed_affine_v9(
            bases, records, status, parameters,
            tile.record_offset + pair_offset + 1U, rhs_active);
    const Bls12381MetalMsmBatchAffinePreparedPair prepared =
        bls12_381_msm_batch_affine_prepare_point_pair(lhs, rhs);
    const Bls12381MetalMsmAffine pair_result =
        bls12_381_msm_batch_affine_finish_point_pair(
            lhs,
            rhs,
            prepared.numerator,
            inverse_denominator,
            prepared.formula_valid,
            prepared.lhs_infinity,
            prepared.rhs_infinity);
    partials[lane] = bls12_381_msm_xyzz_from_affine_v4(pair_result);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pair_count = descriptor_valid
        ? (tile.record_count + 1U) >> 1U
        : 0U;
    const uint tree_span = pair_count == 0U
        ? 1U
        : bls12_381_msm_parallel_tree_span_v1(pair_count);
    for (uint stride = 32U; stride != 0U; stride >>= 1U) {
        if (lane < stride && lane + stride < tree_span) {
            partials[lane] = bls12_381_msm_xyzz_add_v4(
                partials[lane], partials[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        tile_roots[tile_index] = descriptor_valid
            ? partials[0]
            : bls12_381_msm_xyzz_identity_v4();
    }
}

#endif
