// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// XYZZ v5 keeps the two-bucket launch geometry from v4, but detects highly
// imbalanced adjacent bucket runs after histogramming.  Those pairs are
// reduced one bucket at a time with the full threadgroup so a structured long
// run cannot leave half of the SIMD lanes idle.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V5
#define BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V5

#include "msm_gpu_route_xyzz_v4.metal"

inline __attribute__((always_inline))
void bls12_381_msm_gpu_reduce_adaptive_xyzz_impl_v5(
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
    uint expected_threads)
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
    const uint lanes_per_bucket = expected_threads >> 1U;
    const uint larger_count = max(first_count, second_count);
    const uint static_iterations = larger_count / lanes_per_bucket +
        static_cast<uint>(larger_count % lanes_per_bucket != 0U);
    const uint sequential_iterations =
        first_count / expected_threads +
        static_cast<uint>(first_count % expected_threads != 0U) +
        second_count / expected_threads +
        static_cast<uint>(second_count % expected_threads != 0U);
    // Sequential full-group reduction executes a second tree.  Require enough
    // saved mixed-add iterations to cover its barriers and full-point adds.
    const uint minimum_saved_iterations = expected_threads == 32U ? 8U : 10U;
    const bool use_full_group =
        static_iterations > sequential_iterations + minimum_saved_iterations;

    if (!use_full_group) {
        bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
            bases, records, histogram, offsets, bucket_roots, status,
            parameters, partials, valid, lane, threads, groups, group,
            expected_threads, buckets_per_group, 0xffffffffU);
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
        const uint offset = dense_valid ? offsets[dense] : 0U;
        const bool range_valid = dense_valid && bls12_381_msm_range_valid(
            offset, count, parameters.max_record_count);
        Bls12381MetalMsmXyzzV4 accumulator =
            bls12_381_msm_xyzz_identity_v4();
        uint lane_valid = static_cast<uint>(!dense_valid || range_valid);
        if (range_valid && count != 0U) {
            for (uint local = lane; local < count; local += expected_threads) {
                const Bls12381MetalMsmSignedRecord record = records[offset + local];
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

kernel void bls12_381_metal_msm_gpu_reduce_adaptive_xyzz_small_v5(
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
    bls12_381_msm_gpu_reduce_adaptive_xyzz_impl_v5(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U);
}

kernel void bls12_381_metal_msm_gpu_reduce_adaptive_xyzz_large_v5(
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
    bls12_381_msm_gpu_reduce_adaptive_xyzz_impl_v5(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U);
}

#endif
