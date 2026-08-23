// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// XYZZ reduction and collection over the independently validated v3 Metal
// routing ABI.  Keeping separate entry points leaves the Jacobian strategy
// available as an executable benchmark and rollback comparator.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V4
#define BLS12_381_METAL_MSM_GPU_ROUTE_XYZZ_V4

#include "msm_gpu_route_dense_v3.metal"
#include "msm_xyzz_v4.metal"

constant uint BLS12_381_METAL_MSM_XYZZ_ABI_V4 = 4U;

kernel void bls12_381_metal_msm_xyzz_abi_v4(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0U) {
        return;
    }
    output[0] = BLS12_381_METAL_MSM_XYZZ_ABI_V4;
    output[1] = sizeof(Bls12381MetalMsmXyzzV4);
    output[2] = alignof(Bls12381MetalMsmXyzzV4);
}

inline __attribute__((always_inline))
void bls12_381_msm_gpu_reduce_dense_xyzz_impl_v4(
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
    uint expected_buckets_per_group,
    uint maximum_records_per_bucket)
{
    if (group.y != 0U || group.z != 0U) {
        return;
    }
    const uint lanes_per_bucket = expected_threads / expected_buckets_per_group;
    const bool launch_valid =
        bls12_381_msm_gpu_route_parameters_valid_v3(parameters) &&
        parameters.threads_per_group == expected_threads &&
        parameters.buckets_per_group == expected_buckets_per_group &&
        threads.x == expected_threads && threads.y == 1U && threads.z == 1U &&
        groups.x == (parameters.dense_bucket_count +
            expected_buckets_per_group - 1U) / expected_buckets_per_group &&
        groups.y == 1U && groups.z == 1U;
    if (!launch_valid) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint bucket_in_group = lane / lanes_per_bucket;
    const uint local_lane = lane - bucket_in_group * lanes_per_bucket;
    const uint dense = group.x * expected_buckets_per_group + bucket_in_group;
    const bool dense_valid = dense < parameters.dense_bucket_count;
    const uint count = dense_valid
        ? atomic_load_explicit(&histogram[dense], memory_order_relaxed)
        : 0U;
    const uint offset = dense_valid ? offsets[dense] : 0U;
    const bool range_valid = dense_valid && bls12_381_msm_range_valid(
        offset, count, parameters.max_record_count);
    const bool reduce_bucket = range_valid && count <= maximum_records_per_bucket;
    Bls12381MetalMsmXyzzV4 accumulator = bls12_381_msm_xyzz_identity_v4();
    uint lane_valid = static_cast<uint>(!dense_valid || range_valid);
    if (reduce_bucket && count != 0U) {
        for (uint local = local_lane; local < count; local += lanes_per_bucket) {
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
    } else if (dense_valid && !range_valid && count != 0U && local_lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
    }
    partials[lane] = accumulator;
    valid[lane] = lane_valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_lanes = min(count, lanes_per_bucket);
    const uint tree_span = count == 0U
        ? 1U
        : bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = lanes_per_bucket >> 1U; stride != 0U; stride >>= 1U) {
        if (local_lane < stride && local_lane + stride < tree_span) {
            partials[lane] = bls12_381_msm_xyzz_add_v4(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dense_valid && local_lane == 0U) {
        bucket_roots[dense] = count != 0U && reduce_bucket && valid[lane] != 0U
            ? partials[lane]
            : bls12_381_msm_xyzz_identity_v4();
    }
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_v4(
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
        partials, valid, lane, threads, groups, group, 32U, 2U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_v4(
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
        partials, valid, lane, threads, groups, group, 64U, 2U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_pack4_v4(
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
        partials, valid, lane, threads, groups, group, 32U, 4U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_pack4_v4(
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
        partials, valid, lane, threads, groups, group, 64U, 4U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_pack8_v4(
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
        partials, valid, lane, threads, groups, group, 32U, 8U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_pack8_v4(
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
        partials, valid, lane, threads, groups, group, 64U, 8U, 0xffffffffU);
}

kernel void bls12_381_metal_msm_gpu_integrate_dense_xyzz_v4(
    device const Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(0)]],
    device Bls12381MetalMsmXyzzV4* window_points [[buffer(1)]],
    device Bls12381MetalMsmStatus* status [[buffer(2)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(3)]],
    uint window_point_index [[thread_position_in_grid]])
{
    const uint window_point_count =
        parameters.output_count * parameters.windows_per_scalar;
    if (window_point_index >= window_point_count) {
        return;
    }
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        if (window_point_index == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const uint dense_base =
        window_point_index * parameters.buckets_per_window;
    Bls12381MetalMsmXyzzV4 running = bls12_381_msm_xyzz_identity_v4();
    Bls12381MetalMsmXyzzV4 weighted = bls12_381_msm_xyzz_identity_v4();
    for (uint remaining = parameters.buckets_per_window;
         remaining != 0U;
         --remaining) {
        const Bls12381MetalMsmXyzzV4 bucket =
            bucket_roots[dense_base + remaining - 1U];
        running = bls12_381_msm_xyzz_add_v4(running, bucket);
        weighted = bls12_381_msm_xyzz_add_v4(weighted, running);
    }
    window_points[window_point_index] = weighted;
}

kernel void bls12_381_metal_msm_gpu_collect_dense_xyzz_v4(
    device const Bls12381MetalMsmXyzzV4* window_points [[buffer(0)]],
    device Bls12381MetalMsmJacobian* outputs [[buffer(1)]],
    device Bls12381MetalMsmStatus* status [[buffer(2)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(3)]],
    uint output_index [[thread_position_in_grid]])
{
    if (output_index >= parameters.output_count) {
        return;
    }
    outputs[output_index] = bls12_381_msm_jacobian_identity_v2();
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        if (output_index == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const uint window_base = output_index * parameters.windows_per_scalar;
    Bls12381MetalMsmXyzzV4 accumulator = bls12_381_msm_xyzz_identity_v4();
    for (uint remaining = parameters.windows_per_scalar;
         remaining != 0U;
         --remaining) {
        if (remaining != parameters.windows_per_scalar) {
            for (uint bit = 0U; bit < parameters.window_bits; ++bit) {
                accumulator = bls12_381_msm_xyzz_double_v4(accumulator);
            }
        }
        accumulator = bls12_381_msm_xyzz_add_v4(
            accumulator, window_points[window_base + remaining - 1U]);
    }
    outputs[output_index] = bls12_381_msm_xyzz_to_jacobian_v4(accumulator);
}

#endif
