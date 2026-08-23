// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Fine-grained tiling for the structured proof polynomials seen by Midnight.
// V6 reserves tiles only above 4096 records, leaving medium-heavy buckets with
// long serial lane chains. V8 tiles above 2048 records while reusing the
// validated v6 indirect-dispatch, tile-reduction, and combine stages.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V8
#define BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V8

#include "msm_gpu_route_xyzz_v7.metal"

constant uint BLS12_381_METAL_MSM_XYZZ_V8_TILE_RECORDS = 2048U;
constant uint BLS12_381_METAL_MSM_XYZZ_V8_LONG_BUCKET_RECORDS = 2048U;

kernel void bls12_381_metal_msm_gpu_build_fine_tiles_xyzz_v8(
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
    if (count <= BLS12_381_METAL_MSM_XYZZ_V8_LONG_BUCKET_RECORDS) {
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
        (count + BLS12_381_METAL_MSM_XYZZ_V8_TILE_RECORDS - 1U) /
        BLS12_381_METAL_MSM_XYZZ_V8_TILE_RECORDS;
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
            tile * BLS12_381_METAL_MSM_XYZZ_V8_TILE_RECORDS;
        Bls12381MetalMsmBucketTileV1 descriptor;
        descriptor.record_offset = offset + local_offset;
        descriptor.record_count = min(
            BLS12_381_METAL_MSM_XYZZ_V8_TILE_RECORDS,
            count - local_offset);
        descriptor.run_index = dense;
        descriptor.reserved = 0U;
        tiles[first_tile + tile] = descriptor;
    }
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_small_v8(
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
        BLS12_381_METAL_MSM_XYZZ_V8_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_large_v8(
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
        BLS12_381_METAL_MSM_XYZZ_V8_LONG_BUCKET_RECORDS);
}

#endif
