// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Packs ordinary buckets more densely than v6 while retaining its indirect
// long-bucket tiling. Long buckets are deliberately left as identity here and
// overwritten by the v6 tile-combine pass before window integration.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V7
#define BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V7

#include "msm_gpu_route_xyzz_v6.metal"

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack4_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U, 4U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack4_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U, 4U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack8_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U, 8U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack8_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U, 8U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack16_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U, 16U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

kernel void bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack16_v7(
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
    bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U, 16U,
        BLS12_381_METAL_MSM_XYZZ_V6_LONG_BUCKET_RECORDS);
}

#endif
