// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Fixed-base MSM over window-shifted affine bases. Booth windows route into a
// single magnitude-bucket set per output because layer j already contains
// 2^(j*w) P. Weighted integration is split into independent blocks and a short
// second level; no window collection pass is required.

#ifndef BLS12_381_METAL_MSM_FIXED_BASE_V10
#define BLS12_381_METAL_MSM_FIXED_BASE_V10

#include "msm_gpu_route_xyzz_v4.metal"

struct Bls12381MetalMsmFixedRouteParametersV10 {
    uint abi_version;
    uint base_count;
    uint shifted_base_count;
    uint scalar_count;
    uint job_count;
    uint output_count;
    uint window_bits;
    uint windows_per_scalar;
    uint buckets_per_output;
    uint dense_bucket_count;
    uint max_record_count;
    uint max_terms_per_job;
    uint threads_per_group;
    uint buckets_per_group;
    uint integration_block_buckets;
    uint integration_blocks_per_output;
};

struct Bls12381MetalMsmResidentScalarCopyV1 {
    uint source_count;
    uint destination_offset;
    uint destination_capacity;
    uint reserved;
};

static_assert(
    sizeof(Bls12381MetalMsmFixedRouteParametersV10) == 64,
    "fixed route parameters v10 ABI");
static_assert(
    sizeof(Bls12381MetalMsmResidentScalarCopyV1) == 16,
    "resident scalar copy v1 ABI");

constant uint BLS12_381_METAL_MSM_FIXED_ROUTE_ABI_V10 = 10U;
constant uint BLS12_381_METAL_MSM_FIXED_THREADS_V10 = 64U;
constant uint BLS12_381_METAL_MSM_FIXED_LONG_THREADS_V10 = 32U;
constant uint BLS12_381_METAL_MSM_FIXED_BUCKETS_PER_GROUP_V10 = 8U;
constant uint BLS12_381_METAL_MSM_FIXED_BATCHED_BUCKETS_PER_GROUP_V10 = 8U;
constant uint BLS12_381_METAL_MSM_FIXED_INTEGRATION_BLOCK_V10 = 64U;
constant uint BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10 = 2048U;
constant uint BLS12_381_METAL_MSM_FIXED_LONG_BUCKET_RECORDS_V10 = 512U;
constant ulong BLS12_381_METAL_MSM_RESIDENT_SCALAR_ABI_V1 = 1UL;

inline __attribute__((always_inline))
bool bls12_381_msm_resident_scalar_reduced_v1(
    Bls12381MetalMsmScalarV3 value)
{
    for (uint index = 4U; index != 0U; --index) {
        const uint limb = index - 1U;
        if (value.limbs[limb] != BLS12_381_METAL_MSM_FR_MODULUS[limb]) {
            return value.limbs[limb] < BLS12_381_METAL_MSM_FR_MODULUS[limb];
        }
    }
    return false;
}

inline __attribute__((always_inline))
ulong bls12_381_msm_resident_add_carry_v1(
    ulong lhs,
    ulong rhs,
    ulong addend,
    thread ulong& carry)
{
    ulong low = lhs * rhs;
    ulong high = mulhi(lhs, rhs);
    ulong next = low + addend;
    high += static_cast<ulong>(next < low);
    low = next;
    next = low + carry;
    high += static_cast<ulong>(next < low);
    carry = high;
    return next;
}

inline __attribute__((always_inline))
ulong bls12_381_msm_resident_add_word_v1(
    ulong value,
    thread ulong& carry)
{
    const ulong result = value + carry;
    carry = static_cast<ulong>(result < value);
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmScalarV3 bls12_381_msm_resident_reduce_once_v1(
    Bls12381MetalMsmScalarV3 value)
{
    Bls12381MetalMsmScalarV3 subtracted;
    ulong borrow = 0UL;
#pragma unroll
    for (uint index = 0U; index < 4U; ++index) {
        const ulong first = value.limbs[index] -
            BLS12_381_METAL_MSM_FR_MODULUS[index];
        const ulong first_borrow = static_cast<ulong>(
            value.limbs[index] < BLS12_381_METAL_MSM_FR_MODULUS[index]);
        subtracted.limbs[index] = first - borrow;
        const ulong second_borrow = static_cast<ulong>(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    const ulong use_subtracted = 0UL - (borrow ^ 1UL);
#pragma unroll
    for (uint index = 0U; index < 4U; ++index) {
        value.limbs[index] =
            (value.limbs[index] & ~use_subtracted) |
            (subtracted.limbs[index] & use_subtracted);
    }
    return value;
}

inline __attribute__((always_inline))
Bls12381MetalMsmScalarV3 bls12_381_msm_resident_to_canonical_v1(
    Bls12381MetalMsmScalarV3 montgomery)
{
    // Multiplication by the canonical integer one removes one Montgomery R.
    const Bls12381MetalMsmScalarV3 one = {{1UL, 0UL, 0UL, 0UL}};
    ulong temporary[6] = {0UL, 0UL, 0UL, 0UL, 0UL, 0UL};
#pragma unroll
    for (uint rhs_index = 0U; rhs_index < 4U; ++rhs_index) {
        ulong carry = 0UL;
#pragma unroll
        for (uint lhs_index = 0U; lhs_index < 4U; ++lhs_index) {
            temporary[lhs_index] = bls12_381_msm_resident_add_carry_v1(
                montgomery.limbs[lhs_index],
                one.limbs[rhs_index],
                temporary[lhs_index],
                carry);
        }
        temporary[4] = bls12_381_msm_resident_add_word_v1(
            temporary[4], carry);
        temporary[5] = carry;

        carry = 0UL;
        const ulong reduction = temporary[0] * BLS12_381_METAL_MSM_FR_INV;
        (void)bls12_381_msm_resident_add_carry_v1(
            reduction,
            BLS12_381_METAL_MSM_FR_MODULUS[0],
            temporary[0],
            carry);
#pragma unroll
        for (uint index = 1U; index < 4U; ++index) {
            temporary[index - 1U] = bls12_381_msm_resident_add_carry_v1(
                reduction,
                BLS12_381_METAL_MSM_FR_MODULUS[index],
                temporary[index],
                carry);
        }
        temporary[3] = bls12_381_msm_resident_add_word_v1(
            temporary[4], carry);
        temporary[4] = temporary[5] + carry;
    }
    Bls12381MetalMsmScalarV3 result;
#pragma unroll
    for (uint index = 0U; index < 4U; ++index) {
        result.limbs[index] = temporary[index];
    }
    return bls12_381_msm_resident_reduce_once_v1(result);
}

inline __attribute__((always_inline))
uint bls12_381_msm_fixed_buckets_per_group_v10(
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters)
{
    return parameters.window_bits >= 13U && parameters.output_count >= 4U
        ? BLS12_381_METAL_MSM_FIXED_BATCHED_BUCKETS_PER_GROUP_V10
        : BLS12_381_METAL_MSM_FIXED_BUCKETS_PER_GROUP_V10;
}

inline __attribute__((always_inline))
uint bls12_381_msm_fixed_integration_blocks_per_group_v10(
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters)
{
    if (parameters.window_bits >= 13U && parameters.output_count >= 8U) {
        return 16U;
    }
    return parameters.window_bits >= 13U && parameters.output_count >= 4U
        ? 8U
        : 4U;
}

inline __attribute__((always_inline))
bool bls12_381_msm_fixed_parameters_valid_v10(
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters)
{
    const bool bits_valid = parameters.window_bits >= 11U &&
        parameters.window_bits <= 16U;
    const uint expected_windows = bits_valid
        ? 255U / parameters.window_bits + 1U
        : 0U;
    const uint expected_buckets = bits_valid
        ? 1U << (parameters.window_bits - 1U)
        : 0U;
    const bool shifted_product_valid = parameters.base_count != 0U &&
        expected_windows != 0U &&
        parameters.base_count <= 0xffffffffU / expected_windows;
    const bool dense_product_valid = parameters.output_count != 0U &&
        expected_buckets != 0U &&
        parameters.output_count <= 0xffffffffU / expected_buckets;
    return parameters.abi_version == BLS12_381_METAL_MSM_FIXED_ROUTE_ABI_V10 &&
        parameters.job_count == parameters.output_count &&
        parameters.windows_per_scalar == expected_windows &&
        parameters.buckets_per_output == expected_buckets &&
        shifted_product_valid &&
        parameters.shifted_base_count == parameters.base_count * expected_windows &&
        dense_product_valid &&
        parameters.dense_bucket_count == parameters.output_count * expected_buckets &&
        parameters.threads_per_group == BLS12_381_METAL_MSM_FIXED_THREADS_V10 &&
        parameters.buckets_per_group ==
            bls12_381_msm_fixed_buckets_per_group_v10(parameters) &&
        parameters.integration_block_buckets ==
            BLS12_381_METAL_MSM_FIXED_INTEGRATION_BLOCK_V10 &&
        expected_buckets % parameters.integration_block_buckets == 0U &&
        parameters.integration_blocks_per_output ==
            expected_buckets / parameters.integration_block_buckets;
}

inline __attribute__((always_inline))
bool bls12_381_msm_fixed_job_valid_v10(
    Bls12381MetalMsmGpuRouteJobV3 job,
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters)
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
uint bls12_381_msm_fixed_dense_bucket_v10(
    uint output_index,
    uint magnitude,
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters)
{
    return output_index * parameters.buckets_per_output + magnitude - 1U;
}

kernel void bls12_381_metal_msm_fixed_route_abi_v10(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0U) {
        return;
    }
    output[0] = BLS12_381_METAL_MSM_FIXED_ROUTE_ABI_V10;
    output[1] = sizeof(Bls12381MetalMsmScalarV3);
    output[2] = alignof(Bls12381MetalMsmScalarV3);
    output[3] = sizeof(Bls12381MetalMsmGpuRouteJobV3);
    output[4] = alignof(Bls12381MetalMsmGpuRouteJobV3);
    output[5] = __builtin_offsetof(
        Bls12381MetalMsmGpuRouteJobV3, record_segment_offset);
    output[6] = sizeof(Bls12381MetalMsmFixedRouteParametersV10);
    output[7] = alignof(Bls12381MetalMsmFixedRouteParametersV10);
    output[8] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, base_count);
    output[9] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, shifted_base_count);
    output[10] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, output_count);
    output[11] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, window_bits);
    output[12] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, dense_bucket_count);
    output[13] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, max_record_count);
    output[14] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, integration_block_buckets);
    output[15] = __builtin_offsetof(
        Bls12381MetalMsmFixedRouteParametersV10, integration_blocks_per_output);
    output[16] = BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10;
    output[17] = BLS12_381_METAL_MSM_FIXED_LONG_BUCKET_RECORDS_V10;
}

kernel void bls12_381_metal_msm_resident_scalar_abi_v1(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0U) {
        return;
    }
    output[0] = BLS12_381_METAL_MSM_RESIDENT_SCALAR_ABI_V1;
    output[1] = sizeof(Bls12381MetalMsmScalarV3);
    output[2] = alignof(Bls12381MetalMsmScalarV3);
    output[3] = sizeof(Bls12381MetalMsmResidentScalarCopyV1);
    output[4] = alignof(Bls12381MetalMsmResidentScalarCopyV1);
    output[5] = __builtin_offsetof(
        Bls12381MetalMsmResidentScalarCopyV1, destination_offset);
    output[6] = __builtin_offsetof(
        Bls12381MetalMsmResidentScalarCopyV1, destination_capacity);
    output[7] = __builtin_offsetof(
        Bls12381MetalMsmResidentScalarCopyV1, reserved);
#pragma unroll
    for (uint limb = 0U; limb < 4U; ++limb) {
        output[8U + limb] = BLS12_381_METAL_MSM_FR_MODULUS[limb];
    }
    output[12] = BLS12_381_METAL_MSM_FR_INV;
}

kernel void bls12_381_metal_msm_resident_montgomery_to_canonical_v1(
    device const Bls12381MetalMsmScalarV3* source [[buffer(0)]],
    device Bls12381MetalMsmScalarV3* destination [[buffer(1)]],
    device Bls12381MetalMsmStatus* status [[buffer(2)]],
    constant Bls12381MetalMsmResidentScalarCopyV1& copy [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    const bool extent_valid = copy.reserved == 0U &&
        bls12_381_msm_range_valid(
            copy.destination_offset,
            copy.source_count,
            copy.destination_capacity);
    if (!extent_valid) {
        if (index == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (index >= copy.source_count) {
        return;
    }
    const Bls12381MetalMsmScalarV3 value = source[index];
    if (!bls12_381_msm_resident_scalar_reduced_v1(value)) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        Bls12381MetalMsmScalarV3 zero = {{0UL, 0UL, 0UL, 0UL}};
        destination[copy.destination_offset + index] = zero;
        return;
    }
    destination[copy.destination_offset + index] =
        bls12_381_msm_resident_to_canonical_v1(value);
}

kernel void bls12_381_metal_msm_fixed_histogram_v10(
    device const Bls12381MetalMsmScalarV3* scalars [[buffer(0)]],
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(1)]],
    device atomic_uint* histogram [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(4)]],
    uint3 position [[thread_position_in_grid]])
{
    if (position.y >= parameters.job_count ||
        position.z >= parameters.windows_per_scalar) {
        return;
    }
    if (!bls12_381_msm_fixed_parameters_valid_v10(parameters)) {
        if (position.x == 0U && position.y == 0U && position.z == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[position.y];
    if (!bls12_381_msm_fixed_job_valid_v10(job, parameters)) {
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
    if (digit.magnitude > parameters.buckets_per_output) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
        return;
    }
    const uint dense = bls12_381_msm_fixed_dense_bucket_v10(
        job.output_index, digit.magnitude, parameters);
    atomic_fetch_add_explicit(&histogram[dense], 1U, memory_order_relaxed);
}

kernel void bls12_381_metal_msm_fixed_scan_buckets_v10(
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(0)]],
    device const atomic_uint* histogram [[buffer(1)]],
    device uint* offsets [[buffer(2)]],
    device atomic_uint* cursors [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(5)]],
    uint job_index [[thread_position_in_grid]])
{
    if (job_index >= parameters.job_count) {
        return;
    }
    if (!bls12_381_msm_fixed_parameters_valid_v10(parameters)) {
        if (job_index == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[job_index];
    if (!bls12_381_msm_fixed_job_valid_v10(job, parameters)) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        return;
    }
    const uint dense_base = job.output_index * parameters.buckets_per_output;
    const uint segment_capacity = job.term_count * parameters.windows_per_scalar;
    uint cursor = job.record_segment_offset;
    for (uint bucket = 0U; bucket < parameters.buckets_per_output; ++bucket) {
        const uint dense = dense_base + bucket;
        const uint count = atomic_load_explicit(
            &histogram[dense], memory_order_relaxed);
        offsets[dense] = cursor;
        atomic_store_explicit(&cursors[dense], 0U, memory_order_relaxed);
        cursor += count;
    }
    if (cursor < job.record_segment_offset ||
        cursor - job.record_segment_offset > segment_capacity ||
        cursor > parameters.max_record_count) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
    }
}

kernel void bls12_381_metal_msm_fixed_build_long_tiles_v10(
    device const atomic_uint* histogram [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device uint* tile_offsets [[buffer(2)]],
    device uint* tile_counts [[buffer(3)]],
    device Bls12381MetalMsmBucketTileV1* tiles [[buffer(4)]],
    device atomic_uint* indirect_groups [[buffer(5)]],
    device Bls12381MetalMsmStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(7)]],
    constant uint& tile_capacity [[buffer(8)]],
    uint dense [[thread_position_in_grid]])
{
    if (dense >= parameters.dense_bucket_count) {
        return;
    }
    if (!bls12_381_msm_fixed_parameters_valid_v10(parameters) ||
        tile_capacity == 0U) {
        if (dense == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint count = atomic_load_explicit(
        &histogram[dense], memory_order_relaxed);
    if (count <= BLS12_381_METAL_MSM_FIXED_LONG_BUCKET_RECORDS_V10) {
        return;
    }
    const uint offset = offsets[dense];
    if (!bls12_381_msm_range_valid(
            offset, count, parameters.max_record_count)) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }

    const uint count_tiles = 1U +
        (count - 1U) / BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10;
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
            tile * BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10;
        Bls12381MetalMsmBucketTileV1 descriptor;
        descriptor.record_offset = offset + local_offset;
        descriptor.record_count = min(
            BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10,
            count - local_offset);
        descriptor.run_index = dense;
        descriptor.reserved = 0U;
        tiles[first_tile + tile] = descriptor;
    }
}

kernel void bls12_381_metal_msm_fixed_finalize_long_tiles_v10(
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

kernel void bls12_381_metal_msm_fixed_scatter_v10(
    device const Bls12381MetalMsmScalarV3* scalars [[buffer(0)]],
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device atomic_uint* cursors [[buffer(4)]],
    device Bls12381MetalMsmSignedRecord* records [[buffer(5)]],
    device Bls12381MetalMsmStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(7)]],
    device const uint* point_indices [[buffer(8)]],
    uint3 position [[thread_position_in_grid]])
{
    if (position.y >= parameters.job_count ||
        position.z >= parameters.windows_per_scalar ||
        !bls12_381_msm_fixed_parameters_valid_v10(parameters)) {
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[position.y];
    if (!bls12_381_msm_fixed_job_valid_v10(job, parameters) ||
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
    const uint dense = bls12_381_msm_fixed_dense_bucket_v10(
        job.output_index, digit.magnitude, parameters);
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
    if (point_index >= parameters.base_count ||
        position.z > (parameters.shifted_base_count - point_index - 1U) /
            parameters.base_count) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
        return;
    }
    Bls12381MetalMsmSignedRecord record;
    record.point_index = position.z * parameters.base_count + point_index;
    record.flags = digit.flags;
    records[offset + local] = record;
}

kernel void bls12_381_metal_msm_fixed_scatter_resident_v10(
    device const Bls12381MetalMsmScalarV3* scalars [[buffer(0)]],
    device const Bls12381MetalMsmGpuRouteJobV3* jobs [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device atomic_uint* cursors [[buffer(4)]],
    device Bls12381MetalMsmSignedRecord* records [[buffer(5)]],
    device Bls12381MetalMsmStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(7)]],
    uint3 position [[thread_position_in_grid]])
{
    if (position.y >= parameters.job_count ||
        position.z >= parameters.windows_per_scalar ||
        !bls12_381_msm_fixed_parameters_valid_v10(parameters)) {
        return;
    }
    const Bls12381MetalMsmGpuRouteJobV3 job = jobs[position.y];
    if (!bls12_381_msm_fixed_job_valid_v10(job, parameters) ||
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
    const uint dense = bls12_381_msm_fixed_dense_bucket_v10(
        job.output_index, digit.magnitude, parameters);
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
    const uint point_index = job.base_offset + position.x;
    if (point_index < job.base_offset || point_index >= parameters.base_count ||
        position.z > (parameters.shifted_base_count - point_index - 1U) /
            parameters.base_count) {
        bls12_381_msm_status_record_error(
            status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
        return;
    }
    Bls12381MetalMsmSignedRecord record;
    record.point_index = position.z * parameters.base_count + point_index;
    record.flags = digit.flags;
    records[offset + local] = record;
}

kernel void bls12_381_metal_msm_fixed_reduce_xyzz_v10(
    device const Bls12381MetalMsmAffine* shifted_bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const atomic_uint* histogram [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    device Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(4)]],
    device Bls12381MetalMsmStatus* status [[buffer(5)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint lanes_per_bucket = BLS12_381_METAL_MSM_FIXED_THREADS_V10 /
        parameters.buckets_per_group;
    const bool launch_valid = group.y == 0U && group.z == 0U &&
        bls12_381_msm_fixed_parameters_valid_v10(parameters) &&
        threads.x == BLS12_381_METAL_MSM_FIXED_THREADS_V10 &&
        threads.y == 1U && threads.z == 1U &&
        groups.x == (parameters.dense_bucket_count +
            parameters.buckets_per_group - 1U) /
            parameters.buckets_per_group &&
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
    const uint dense = group.x * parameters.buckets_per_group + bucket_in_group;
    const bool dense_valid = dense < parameters.dense_bucket_count;
    const uint count = dense_valid
        ? atomic_load_explicit(&histogram[dense], memory_order_relaxed)
        : 0U;
    const uint offset = dense_valid ? offsets[dense] : 0U;
    const bool range_valid = dense_valid && bls12_381_msm_range_valid(
        offset, count, parameters.max_record_count);
    const bool tiled = count >
        BLS12_381_METAL_MSM_FIXED_LONG_BUCKET_RECORDS_V10;
    Bls12381MetalMsmXyzzV4 accumulator = bls12_381_msm_xyzz_identity_v4();
    uint accumulator_state =
        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4;
    uint lane_valid = static_cast<uint>(!dense_valid || range_valid);
    if (range_valid && count != 0U && !tiled) {
        for (uint local = local_lane; local < count; local += lanes_per_bucket) {
            const Bls12381MetalMsmSignedRecord record = records[offset + local];
            if ((record.flags & ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
            } else if (record.point_index >= parameters.shifted_base_count) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
            } else {
                const Bls12381MetalMsmAffine point =
                    bls12_381_msm_g1_signed_affine(
                        shifted_bases[record.point_index], record.flags);
                accumulator = bls12_381_msm_xyzz_mixed_add_tracked_v4(
                    accumulator, point, accumulator_state);
            }
        }
    } else if (dense_valid && count != 0U && !range_valid && local_lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
    }
    const uint active_lanes = tiled ? 0U : min(count, lanes_per_bucket);
    const uint tree_span = count == 0U || tiled
        ? 1U
        : bls12_381_msm_parallel_tree_span_v1(active_lanes);
    for (uint stride = lanes_per_bucket >> 1U; stride != 0U; stride >>= 1U) {
        const Bls12381MetalMsmXyzzV4 right =
            bls12_381_msm_xyzz_simd_shuffle_down_v4(
                accumulator, static_cast<ushort>(stride));
        const uint right_valid = simd_shuffle_down(
            lane_valid, static_cast<ushort>(stride));
        // Every lane adds; only the lanes in the tree keep the sum. With the add
        // inside the lane-dependent branch, between SIMD shuffles, a SIMD
        // group's bucket roots came back wrong on rare runs (about one batched
        // output in 1,000 at width 16), like the integration kernel below.
        const Bls12381MetalMsmXyzzV4 combined =
            bls12_381_msm_xyzz_add_v4(accumulator, right);
        if (local_lane < stride && local_lane + stride < tree_span) {
            accumulator = combined;
            lane_valid &= right_valid;
        }
    }
    if (dense_valid && local_lane == 0U) {
        bucket_roots[dense] = count != 0U && !tiled && lane_valid != 0U
            ? accumulator
            : bls12_381_msm_xyzz_identity_v4();
    }
}

kernel void bls12_381_metal_msm_fixed_reduce_long_tiles_xyzz_v10(
    device const Bls12381MetalMsmAffine* shifted_bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketTileV1* tiles [[buffer(2)]],
    device Bls12381MetalMsmXyzzV4* tile_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint tile_index = group.x;
    const bool launch_valid =
        group.y == 0U && group.z == 0U && groups.y == 1U && groups.z == 1U &&
        threads.x == BLS12_381_METAL_MSM_FIXED_LONG_THREADS_V10 &&
        threads.y == 1U && threads.z == 1U &&
        parameters.threads_per_group ==
            BLS12_381_METAL_MSM_FIXED_THREADS_V10 &&
        tile_index < tile_capacity &&
        bls12_381_msm_fixed_parameters_valid_v10(parameters);
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
        tile.record_count <= BLS12_381_METAL_MSM_FIXED_TILE_RECORDS_V10 &&
        bls12_381_msm_range_valid(
            tile.record_offset,
            tile.record_count,
            parameters.max_record_count);
    Bls12381MetalMsmXyzzV4 accumulator =
        bls12_381_msm_xyzz_identity_v4();
    uint accumulator_state =
        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4;
    uint lane_valid = static_cast<uint>(descriptor_valid);
    if (descriptor_valid) {
        for (uint local = lane;
             local < tile.record_count;
             local += BLS12_381_METAL_MSM_FIXED_LONG_THREADS_V10) {
            const Bls12381MetalMsmSignedRecord record =
                records[tile.record_offset + local];
            if ((record.flags &
                 ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status,
                    BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
            } else if (record.point_index >= parameters.shifted_base_count) {
                lane_valid = 0U;
                bls12_381_msm_status_record_error(
                    status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
            } else {
                const Bls12381MetalMsmAffine point =
                    bls12_381_msm_g1_signed_affine(
                        shifted_bases[record.point_index], record.flags);
                accumulator = bls12_381_msm_xyzz_mixed_add_tracked_v4(
                    accumulator, point, accumulator_state);
            }
        }
    } else if (lane == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
    }
    const uint active_lanes = min(
        tile.record_count, BLS12_381_METAL_MSM_FIXED_LONG_THREADS_V10);
    const uint tree_span = descriptor_valid
        ? bls12_381_msm_parallel_tree_span_v1(active_lanes)
        : 1U;
    for (uint stride = 16U;
         stride != 0U;
         stride >>= 1U) {
        const Bls12381MetalMsmXyzzV4 right =
            bls12_381_msm_xyzz_simd_shuffle_down_v4(
                accumulator, static_cast<ushort>(stride));
        const uint right_valid = simd_shuffle_down(
            lane_valid, static_cast<ushort>(stride));
        if (lane < stride && lane + stride < tree_span) {
            accumulator = bls12_381_msm_xyzz_add_v4(accumulator, right);
            lane_valid &= right_valid;
        }
    }
    if (lane == 0U) {
        tile_roots[tile_index] = lane_valid != 0U
            ? accumulator
            : bls12_381_msm_xyzz_identity_v4();
    }
}

kernel void bls12_381_metal_msm_fixed_combine_long_tiles_xyzz_v10(
    device const uint* tile_offsets [[buffer(0)]],
    device const uint* tile_counts [[buffer(1)]],
    device const Bls12381MetalMsmXyzzV4* tile_roots [[buffer(2)]],
    device Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(5)]],
    constant uint& tile_capacity [[buffer(6)]],
    uint dense [[thread_position_in_grid]])
{
    if (dense >= parameters.dense_bucket_count) {
        return;
    }
    if (!bls12_381_msm_fixed_parameters_valid_v10(parameters)) {
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
    Bls12381MetalMsmXyzzV4 root = tile_roots[offset];
    for (uint tile = 1U; tile < count; ++tile) {
        root = bls12_381_msm_xyzz_add_v4(
            root, tile_roots[offset + tile]);
    }
    bucket_roots[dense] = root;
}

kernel void bls12_381_metal_msm_fixed_integrate_blocks_v10(
    device const Bls12381MetalMsmXyzzV4* bucket_roots [[buffer(0)]],
    device Bls12381MetalMsmXyzzV4* block_sums [[buffer(1)]],
    device Bls12381MetalMsmXyzzV4* block_weighted [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint blocks_per_group =
        bls12_381_msm_fixed_integration_blocks_per_group_v10(parameters);
    const uint lanes_per_block =
        BLS12_381_METAL_MSM_FIXED_THREADS_V10 / blocks_per_group;
    const uint buckets_per_lane =
        BLS12_381_METAL_MSM_FIXED_INTEGRATION_BLOCK_V10 / lanes_per_block;
    const uint block_count = parameters.output_count *
        parameters.integration_blocks_per_output;
    const bool launch_valid = group.y == 0U && group.z == 0U &&
        groups.y == 1U && groups.z == 1U &&
        threads.x == BLS12_381_METAL_MSM_FIXED_THREADS_V10 &&
        threads.y == 1U && threads.z == 1U &&
        block_count % blocks_per_group == 0U &&
        groups.x == block_count / blocks_per_group &&
        bls12_381_msm_fixed_parameters_valid_v10(parameters);
    if (!launch_valid) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint block_in_group = lane / lanes_per_block;
    const uint local_lane = lane - block_in_group * lanes_per_block;
    const uint block_index = group.x * blocks_per_group + block_in_group;
    const uint output_index = block_index /
        parameters.integration_blocks_per_output;
    const uint local_block = block_index - output_index *
        parameters.integration_blocks_per_output;
    const uint bucket_base = output_index * parameters.buckets_per_output +
        local_block * parameters.integration_block_buckets;
    uint remaining = parameters.integration_block_buckets -
        local_lane * buckets_per_lane;
    Bls12381MetalMsmXyzzV4 running =
        bucket_roots[bucket_base + remaining - 1U];
    Bls12381MetalMsmXyzzV4 weighted = running;
    for (uint local = 1U; local < buckets_per_lane; ++local) {
        --remaining;
        running = bls12_381_msm_xyzz_add_v4(
            running, bucket_roots[bucket_base + remaining - 1U]);
        weighted = bls12_381_msm_xyzz_add_v4(weighted, running);
    }

    for (uint stride = 1U; stride < lanes_per_block; stride <<= 1U) {
        const Bls12381MetalMsmXyzzV4 right_sum =
            bls12_381_msm_xyzz_simd_shuffle_down_v4(
                running, static_cast<ushort>(stride));
        const Bls12381MetalMsmXyzzV4 right_weighted =
            bls12_381_msm_xyzz_simd_shuffle_down_v4(
                weighted, static_cast<ushort>(stride));
        // Every lane combines; only the lanes that own the pair keep it. With
        // the doubling loop inside the lane-dependent branch, between this
        // stride's shuffles and the next, some batched outputs came back wrong
        // on some runs (widths 12 to 16, M5 Pro, macOS 27.0.1).
        Bls12381MetalMsmXyzzV4 scaled = running;
        for (uint scale = buckets_per_lane * stride;
             scale > 1U;
             scale >>= 1U) {
            scaled = bls12_381_msm_xyzz_double_v4(scaled);
        }
        const Bls12381MetalMsmXyzzV4 combined_weighted =
            bls12_381_msm_xyzz_add_v4(
                bls12_381_msm_xyzz_add_v4(weighted, scaled),
                right_weighted);
        const Bls12381MetalMsmXyzzV4 combined_running =
            bls12_381_msm_xyzz_add_v4(running, right_sum);
        if ((local_lane & ((stride << 1U) - 1U)) == 0U) {
            weighted = combined_weighted;
            running = combined_running;
        }
    }
    if (local_lane == 0U) {
        block_sums[block_index] = running;
        block_weighted[block_index] = weighted;
    }
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_fixed_mul_small_v10(
    Bls12381MetalMsmXyzzV4 point,
    uint scalar)
{
    if (scalar == 0U) {
        return bls12_381_msm_xyzz_identity_v4();
    }
    if (scalar == 1U) {
        return point;
    }
    Bls12381MetalMsmXyzzV4 result = bls12_381_msm_xyzz_identity_v4();
    Bls12381MetalMsmXyzzV4 addend = point;
    while (scalar != 0U) {
        if ((scalar & 1U) != 0U) {
            result = bls12_381_msm_xyzz_add_v4(result, addend);
        }
        scalar >>= 1U;
        if (scalar != 0U) {
            addend = bls12_381_msm_xyzz_double_v4(addend);
        }
    }
    return result;
}

kernel void bls12_381_metal_msm_fixed_finish_integration_v10(
    device const Bls12381MetalMsmXyzzV4* block_sums [[buffer(0)]],
    device const Bls12381MetalMsmXyzzV4* block_weighted [[buffer(1)]],
    device Bls12381MetalMsmJacobian* outputs [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmFixedRouteParametersV10& parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup Bls12381MetalMsmXyzzV4 partial_weighted[64];
    threadgroup Bls12381MetalMsmXyzzV4 partial_correction[64];
    const bool launch_valid = group.y == 0U && group.z == 0U &&
        groups.x == parameters.output_count && groups.y == 1U && groups.z == 1U &&
        threads.x == BLS12_381_METAL_MSM_FIXED_THREADS_V10 &&
        threads.y == 1U && threads.z == 1U &&
        bls12_381_msm_fixed_parameters_valid_v10(parameters);
    if (!launch_valid) {
        if (lane == 0U) {
            bls12_381_msm_status_descriptor_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    const uint output_index = group.x;
    const uint block_base = output_index *
        parameters.integration_blocks_per_output;
    Bls12381MetalMsmXyzzV4 weighted = bls12_381_msm_xyzz_identity_v4();
    Bls12381MetalMsmXyzzV4 correction = bls12_381_msm_xyzz_identity_v4();
    bool initialized = false;
    for (uint block = lane;
         block < parameters.integration_blocks_per_output;
         block += BLS12_381_METAL_MSM_FIXED_THREADS_V10) {
        const Bls12381MetalMsmXyzzV4 block_value =
            block_weighted[block_base + block];
        const Bls12381MetalMsmXyzzV4 block_correction =
            bls12_381_msm_fixed_mul_small_v10(
                block_sums[block_base + block], block);
        if (!initialized) {
            weighted = block_value;
            correction = block_correction;
            initialized = true;
        } else {
            weighted = bls12_381_msm_xyzz_add_v4(weighted, block_value);
            correction = bls12_381_msm_xyzz_add_v4(
                correction, block_correction);
        }
    }
    partial_weighted[lane] = weighted;
    partial_correction[lane] = correction;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = BLS12_381_METAL_MSM_FIXED_THREADS_V10 >> 1U;
         stride != 0U;
         stride >>= 1U) {
        if (lane < stride) {
            partial_weighted[lane] = bls12_381_msm_xyzz_add_v4(
                partial_weighted[lane], partial_weighted[lane + stride]);
            partial_correction[lane] = bls12_381_msm_xyzz_add_v4(
                partial_correction[lane], partial_correction[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        Bls12381MetalMsmXyzzV4 scaled_correction = partial_correction[0];
        for (uint scale = parameters.integration_block_buckets;
             scale > 1U;
             scale >>= 1U) {
            scaled_correction =
                bls12_381_msm_xyzz_double_v4(scaled_correction);
        }
        const Bls12381MetalMsmXyzzV4 result = bls12_381_msm_xyzz_add_v4(
            partial_weighted[0], scaled_correction);
        outputs[output_index] = bls12_381_msm_xyzz_to_jacobian_v4(result);
    }
}

#endif
