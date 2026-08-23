// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Metal-side Booth routing and one-group-per-bucket Jacobian reduction.

#ifndef BLS12_381_METAL_MSM_GPU_ROUTE_DENSE_V3
#define BLS12_381_METAL_MSM_GPU_ROUTE_DENSE_V3

#include "msm_parallel_jacobian_v2.metal"

struct Bls12381MetalMsmScalarV3 {
    ulong limbs[4];
};

struct Bls12381MetalMsmGpuRouteJobV3 {
    uint base_offset;
    uint scalar_offset;
    uint term_count;
    uint output_index;
    uint record_segment_offset;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

struct Bls12381MetalMsmGpuRouteParametersV3 {
    uint abi_version;
    uint base_count;
    uint scalar_count;
    uint job_count;
    uint output_count;
    uint window_bits;
    uint windows_per_scalar;
    uint buckets_per_window;
    uint dense_bucket_count;
    uint max_record_count;
    uint max_terms_per_job;
    uint threads_per_group;
    uint buckets_per_group;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

static_assert(sizeof(Bls12381MetalMsmScalarV3) == 32, "GPU scalar v3 ABI");
static_assert(sizeof(Bls12381MetalMsmGpuRouteJobV3) == 32, "GPU route job v3 ABI");
static_assert(
    sizeof(Bls12381MetalMsmGpuRouteParametersV3) == 64,
    "GPU route parameters v3 ABI");

constant uint BLS12_381_METAL_MSM_GPU_ROUTE_ABI_V4 = 4U;

inline __attribute__((always_inline))
bool bls12_381_msm_gpu_route_parameters_valid_v3(
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters)
{
    const bool bits_valid = parameters.window_bits >= 6U &&
        parameters.window_bits <= 10U;
    const uint expected_windows = bits_valid
        ? 255U / parameters.window_bits + 1U
        : 0U;
    const uint expected_buckets = bits_valid
        ? 1U << (parameters.window_bits - 1U)
        : 0U;
    return parameters.abi_version == BLS12_381_METAL_MSM_GPU_ROUTE_ABI_V4 &&
        parameters.job_count == parameters.output_count &&
        parameters.output_count != 0U &&
        parameters.windows_per_scalar == expected_windows &&
        parameters.buckets_per_window == expected_buckets &&
        parameters.dense_bucket_count ==
            parameters.output_count * expected_windows * expected_buckets &&
        (parameters.threads_per_group == 32U ||
         parameters.threads_per_group == 64U) &&
        (parameters.buckets_per_group == 2U ||
         parameters.buckets_per_group == 4U ||
         parameters.buckets_per_group == 8U ||
         parameters.buckets_per_group == 16U) &&
        (parameters.reserved_0 | parameters.reserved_1 |
         parameters.reserved_2) == 0U;
}

inline __attribute__((always_inline))
bool bls12_381_msm_gpu_route_job_valid_v3(
    Bls12381MetalMsmGpuRouteJobV3 job,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters)
{
    const bool segment_offset_valid =
        job.record_segment_offset <= parameters.max_record_count;
    const uint remaining_records = segment_offset_valid
        ? parameters.max_record_count - job.record_segment_offset
        : 0U;
    return (job.reserved_0 | job.reserved_1 | job.reserved_2) == 0U &&
        job.output_index < parameters.output_count &&
        bls12_381_msm_range_valid(
            job.base_offset, job.term_count, parameters.base_count) &&
        bls12_381_msm_range_valid(
            job.scalar_offset, job.term_count, parameters.scalar_count) &&
        segment_offset_valid && parameters.windows_per_scalar != 0U &&
        job.term_count <= remaining_records / parameters.windows_per_scalar;
}

inline __attribute__((always_inline))
uint bls12_381_msm_scalar_byte_v3(
    Bls12381MetalMsmScalarV3 scalar,
    uint byte_index)
{
    if (byte_index >= 32U) {
        return 0U;
    }
    const uint limb_index = byte_index >> 3U;
    const uint shift = (byte_index & 7U) << 3U;
    return static_cast<uint>(scalar.limbs[limb_index] >> shift) & 0xffU;
}

struct Bls12381MetalMsmBoothDigitV3 {
    uint magnitude;
    uint flags;
};

inline __attribute__((always_inline))
Bls12381MetalMsmBoothDigitV3 bls12_381_msm_booth_digit_v3(
    Bls12381MetalMsmScalarV3 scalar,
    uint window_index,
    uint window_bits)
{
    const uint bit_offset = window_index * window_bits;
    const uint skip_bits = bit_offset == 0U ? 0U : bit_offset - 1U;
    const uint skip_bytes = skip_bits >> 3U;
    uint value = 0U;
#pragma unroll
    for (uint byte = 0U; byte < 4U; ++byte) {
        value |= bls12_381_msm_scalar_byte_v3(
            scalar, skip_bytes + byte) << (byte << 3U);
    }
    if (window_index == 0U) {
        value <<= 1U;
    }
    value >>= skip_bits - (skip_bytes << 3U);
    value &= (1U << (window_bits + 1U)) - 1U;

    const bool positive = (value & (1U << window_bits)) == 0U;
    const uint rounded = (value + 1U) >> 1U;
    Bls12381MetalMsmBoothDigitV3 result;
    result.magnitude = positive
        ? rounded
        : (~(rounded - 1U) & ((1U << window_bits) - 1U));
    result.flags = positive || result.magnitude == 0U
        ? 0U
        : BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE;
    return result;
}

inline __attribute__((always_inline))
uint bls12_381_msm_dense_bucket_v3(
    uint output_index,
    uint window_index,
    uint magnitude,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters)
{
    return ((output_index * parameters.windows_per_scalar + window_index) *
        parameters.buckets_per_window) + magnitude - 1U;
}

kernel void bls12_381_metal_msm_gpu_route_abi_v4(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0U) {
        return;
    }
    output[0] = BLS12_381_METAL_MSM_GPU_ROUTE_ABI_V4;
    output[1] = sizeof(Bls12381MetalMsmScalarV3);
    output[2] = alignof(Bls12381MetalMsmScalarV3);
    output[3] = sizeof(Bls12381MetalMsmGpuRouteJobV3);
    output[4] = alignof(Bls12381MetalMsmGpuRouteJobV3);
    output[5] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteJobV3, record_segment_offset);
    output[6] = sizeof(Bls12381MetalMsmGpuRouteParametersV3);
    output[7] = alignof(Bls12381MetalMsmGpuRouteParametersV3);
    output[8] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, base_count);
    output[9] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, output_count);
    output[10] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, window_bits);
    output[11] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, dense_bucket_count);
    output[12] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, max_record_count);
    output[13] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, max_terms_per_job);
    output[14] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, threads_per_group);
    output[15] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteParametersV3, buckets_per_group);
}

kernel void bls12_381_metal_msm_gpu_histogram_v4(
    device const Bls12381MetalMsmScalarV3* scalars [[buffer(0)]],
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(1)]],
    device atomic_uint* histogram [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(4)]],
    uint3 position [[thread_position_in_grid]])
{
    if (position.y >= parameters.job_count ||
        position.z >= parameters.windows_per_scalar) {
        return;
    }
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        if (position.x == 0U && position.y == 0U && position.z == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[position.y];
    if (!bls12_381_msm_gpu_route_job_valid_v3(job, parameters)) {
        if (position.x == 0U && position.z == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        }
        return;
    }
    if (position.x >= job.term_count) {
        return;
    }
    const Bls12381MetalMsmBoothDigitV3 digit = bls12_381_msm_booth_digit_v3(
        scalars[job.scalar_offset + position.x],
        position.z,
        parameters.window_bits);
    if (digit.magnitude == 0U) {
        return;
    }
    if (digit.magnitude > parameters.buckets_per_window) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
        return;
    }
    const uint dense = bls12_381_msm_dense_bucket_v3(
        job.output_index, position.z, digit.magnitude, parameters);
    atomic_fetch_add_explicit(&histogram[dense], 1U, memory_order_relaxed);
}

kernel void bls12_381_metal_msm_gpu_scan_buckets_v4(
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(0)]],
    device const atomic_uint* histogram [[buffer(1)]],
    device uint* offsets [[buffer(2)]],
    device atomic_uint* cursors [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(5)]],
    uint window_job_index [[thread_position_in_grid]])
{
    const uint window_job_count =
        parameters.job_count * parameters.windows_per_scalar;
    if (window_job_index >= window_job_count) {
        return;
    }
    if (!bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        if (window_job_index == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const uint job_index =
        window_job_index / parameters.windows_per_scalar;
    const uint window_index =
        window_job_index - job_index * parameters.windows_per_scalar;
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[job_index];
    if (!bls12_381_msm_gpu_route_job_valid_v3(job, parameters)) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        return;
    }
    const uint dense_base =
        (job.output_index * parameters.windows_per_scalar + window_index) *
        parameters.buckets_per_window;
    const uint segment_base = job.record_segment_offset +
        window_index * job.term_count;
    uint cursor = segment_base;
    for (uint bucket = 0U; bucket < parameters.buckets_per_window; ++bucket) {
        const uint dense = dense_base + bucket;
        const uint count = atomic_load_explicit(
            &histogram[dense], memory_order_relaxed);
        offsets[dense] = cursor;
        atomic_store_explicit(&cursors[dense], 0U, memory_order_relaxed);
        cursor += count;
    }
    if (cursor < segment_base || cursor - segment_base > job.term_count ||
        cursor > parameters.max_record_count) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
    }
}

kernel void bls12_381_metal_msm_gpu_scatter_v4(
    device const Bls12381MetalMsmScalarV3* scalars [[buffer(0)]],
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device atomic_uint* cursors [[buffer(4)]],
    device Bls12381MetalMsmSignedRecord* records [[buffer(5)]],
    device Bls12381MetalMsmStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(7)]],
    device const uint* point_indices [[buffer(8)]],
    uint3 position [[thread_position_in_grid]])
{
    if (position.y >= parameters.job_count ||
        position.z >= parameters.windows_per_scalar ||
        !bls12_381_msm_gpu_route_parameters_valid_v3(parameters)) {
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[position.y];
    if (!bls12_381_msm_gpu_route_job_valid_v3(job, parameters) ||
        position.x >= job.term_count) {
        return;
    }
    const Bls12381MetalMsmBoothDigitV3 digit = bls12_381_msm_booth_digit_v3(
        scalars[job.scalar_offset + position.x],
        position.z,
        parameters.window_bits);
    if (digit.magnitude == 0U) {
        return;
    }
    const uint dense = bls12_381_msm_dense_bucket_v3(
        job.output_index, position.z, digit.magnitude, parameters);
    const uint local = atomic_fetch_add_explicit(
        &cursors[dense], 1U, memory_order_relaxed);
    const uint count = atomic_load_explicit(
        &histogram[dense], memory_order_relaxed);
    const uint offset = offsets[dense];
    if (local >= count || offset > parameters.max_record_count ||
        local >= parameters.max_record_count - offset) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }
    const uint point_index = point_indices[job.scalar_offset + position.x];
    if (point_index >= parameters.base_count) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
        return;
    }
    Bls12381MetalMsmSignedRecord record;
    record.point_index = point_index;
    record.flags = digit.flags;
    records[offset + local] = record;
}

inline __attribute__((always_inline))
void bls12_381_msm_gpu_reduce_dense_impl_v3(
    device const Bls12381MetalMsmAffine* bases,
    device const Bls12381MetalMsmSignedRecord* records,
    device const atomic_uint* histogram,
    device const uint* offsets,
    device Bls12381MetalMsmJacobian* bucket_roots,
    device Bls12381MetalMsmStatus* status,
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters,
    threadgroup Bls12381MetalMsmJacobian* partials,
    threadgroup uint* valid,
    uint lane,
    uint3 threads,
    uint3 groups,
    uint3 group,
    uint expected_threads)
{
    if (group.y != 0U || group.z != 0U) {
        return;
    }
    const uint expected_buckets_per_group = 2U;
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
    Bls12381MetalMsmJacobian accumulator =
        bls12_381_msm_jacobian_identity_v2();
    uint lane_valid = static_cast<uint>(!dense_valid || range_valid);
    if (range_valid && count != 0U) {
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
                accumulator = bls12_381_msm_jacobian_mixed_add_v2(
                    accumulator, point);
            }
        }
    } else if (dense_valid && count != 0U && local_lane == 0U) {
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
    // All subgroups must cross the same barriers even when their buckets
    // contain different numbers of records.  The tree-span predicate keeps
    // inactive lanes out of the arithmetic without making the loop divergent.
    for (uint stride = lanes_per_bucket >> 1U; stride != 0U; stride >>= 1U) {
        if (local_lane < stride && local_lane + stride < tree_span) {
            partials[lane] = bls12_381_msm_jacobian_add_v2(
                partials[lane], partials[lane + stride]);
            valid[lane] &= valid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dense_valid && local_lane == 0U) {
        bucket_roots[dense] = count != 0U && valid[lane] != 0U
            ? partials[lane]
            : bls12_381_msm_jacobian_identity_v2();
    }
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_small_v3(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device Bls12381MetalMsmJacobian* bucket_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[32];
    threadgroup uint valid[32];
    bls12_381_msm_gpu_reduce_dense_impl_v3(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 32U);
}

kernel void bls12_381_metal_msm_gpu_reduce_dense_large_v3(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device Bls12381MetalMsmJacobian* bucket_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmGpuRouteParametersV3& parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmJacobian partials[64];
    threadgroup uint valid[64];
    bls12_381_msm_gpu_reduce_dense_impl_v3(
        bases, records, histogram, offsets, bucket_roots, status, parameters,
        partials, valid, lane, threads, groups, group, 64U);
}

kernel void bls12_381_metal_msm_gpu_integrate_dense_v3(
    device const Bls12381MetalMsmJacobian* bucket_roots [[buffer(0)]],
    device Bls12381MetalMsmJacobian* window_points [[buffer(1)]],
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
    Bls12381MetalMsmJacobian running =
        bls12_381_msm_jacobian_identity_v2();
    Bls12381MetalMsmJacobian weighted =
        bls12_381_msm_jacobian_identity_v2();
    for (uint remaining = parameters.buckets_per_window;
         remaining != 0U;
         --remaining) {
        const Bls12381MetalMsmJacobian bucket =
            bucket_roots[dense_base + remaining - 1U];
        running = bls12_381_msm_jacobian_add_v2(running, bucket);
        weighted = bls12_381_msm_jacobian_add_v2(weighted, running);
    }
    window_points[window_point_index] = weighted;
}

kernel void bls12_381_metal_msm_gpu_collect_dense_v3(
    device const Bls12381MetalMsmJacobian* window_points [[buffer(0)]],
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
    Bls12381MetalMsmJacobian accumulator =
        bls12_381_msm_jacobian_identity_v2();
    for (uint remaining = parameters.windows_per_scalar;
         remaining != 0U;
         --remaining) {
        if (remaining != parameters.windows_per_scalar) {
            for (uint bit = 0U; bit < parameters.window_bits; ++bit) {
                accumulator = bls12_381_msm_jacobian_double_v2(accumulator);
            }
        }
        accumulator = bls12_381_msm_jacobian_add_v2(
            accumulator, window_points[window_base + remaining - 1U]);
    }
    outputs[output_index] =
        bls12_381_msm_jacobian_canonicalize_v2(accumulator);
}

#endif
