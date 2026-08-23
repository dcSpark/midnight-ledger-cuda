// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_BATCH_AFFINE_METAL
#define MIDNIGHT_BLS12_381_BATCH_AFFINE_METAL

#include "fp32.metal"

// Diagnostic-only affine representation. (0, 0) is infinity, matching the
// public BLST-compatible affine encoding. All finite coordinates remain in the
// existing R=2^384 Montgomery representation.
struct MidnightBatchAffinePoint32 {
    Bls12_381Fp32 x;
    Bls12_381Fp32 y;
};

// Register-only result shared by the global-tree oracle and experimental
// reducers. Mask fields are either all-zero or all-one words.
struct MidnightBatchAffinePreparedPair {
    Bls12_381Fp32 numerator;
    Bls12_381Fp32 denominator;
    uint formula_valid;
    uint lhs_infinity;
    uint rhs_infinity;
};

struct MidnightBatchAffineGatherParameters {
    uint record_offset;
    uint record_count;
    uint srs_count;
    uint reserved;
};

struct MidnightBatchAffineLevelParameters {
    uint destination_map_offset;
    uint pair_count;
    uint carry_count;
    uint destination_len;
};

struct MidnightBatchAffineProductParameters {
    uint input_offset;
    uint input_count;
    uint output_offset;
    uint output_count;
};

struct MidnightBatchAffineStatus {
    atomic_uint flags;
    atomic_uint record_errors;
    atomic_uint destination_errors;
    atomic_uint reserved;
};

struct MidnightBatchAffineCeilingParameters {
    uint element_count;
    uint iterations;
    uint reserved_0;
    uint reserved_1;
};

constant uint MIDNIGHT_BATCH_AFFINE_CASE_FORMULA_VALID = 1U << 0;
constant uint MIDNIGHT_BATCH_AFFINE_CASE_LHS_INFINITY = 1U << 1;
constant uint MIDNIGHT_BATCH_AFFINE_CASE_RHS_INFINITY = 1U << 2;

constant uint MIDNIGHT_BATCH_AFFINE_STATUS_ADJUSTED_DENOMINATOR_ZERO = 1U << 0;
constant uint MIDNIGHT_BATCH_AFFINE_STATUS_PRODUCT_ZERO = 1U << 1;
constant uint MIDNIGHT_BATCH_AFFINE_STATUS_INVERSE_ZERO = 1U << 2;
constant uint MIDNIGHT_BATCH_AFFINE_STATUS_INVALID_PARAMETERS = 1U << 3;
constant uint MIDNIGHT_BATCH_AFFINE_STATUS_RECORD_OUT_OF_RANGE = 1U << 4;
constant uint MIDNIGHT_BATCH_AFFINE_STATUS_DESTINATION_OUT_OF_RANGE = 1U << 5;
// Bit 6 is reserved. Coordinate shape alone cannot distinguish every finite
// curve point from a malformed infinity encoding; immutable key import owns
// that validation.

inline Bls12_381Fp32 batch_affine_fp32_zero()
{
    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = 0;
    }
    return result;
}

inline Bls12_381Fp32 batch_affine_fp32_one()
{
    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = MIDNIGHT_BLS12_381_FP32_R[index];
    }
    return result;
}

inline uint batch_affine_fp32_zero_mask(Bls12_381Fp32 value)
{
    uint aggregate = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        aggregate |= value.limbs[index];
    }
    return 0U - static_cast<uint>(aggregate == 0);
}

inline uint batch_affine_fp32_equal_mask(
    Bls12_381Fp32 lhs,
    Bls12_381Fp32 rhs)
{
    uint aggregate = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        aggregate |= lhs.limbs[index] ^ rhs.limbs[index];
    }
    return 0U - static_cast<uint>(aggregate == 0);
}

inline Bls12_381Fp32 batch_affine_fp32_select(
    Bls12_381Fp32 when_clear,
    Bls12_381Fp32 when_set,
    uint mask)
{
    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] =
            (when_clear.limbs[index] & ~mask) |
            (when_set.limbs[index] & mask);
    }
    return result;
}

inline MidnightBatchAffinePoint32 batch_affine_point_identity()
{
    MidnightBatchAffinePoint32 result;
    result.x = batch_affine_fp32_zero();
    result.y = batch_affine_fp32_zero();
    return result;
}

inline uint batch_affine_point_infinity_mask(
    MidnightBatchAffinePoint32 point)
{
    return batch_affine_fp32_zero_mask(point.x) &
        batch_affine_fp32_zero_mask(point.y);
}

inline MidnightBatchAffinePoint32 batch_affine_point_select(
    MidnightBatchAffinePoint32 when_clear,
    MidnightBatchAffinePoint32 when_set,
    uint mask)
{
    MidnightBatchAffinePoint32 result;
    result.x = batch_affine_fp32_select(when_clear.x, when_set.x, mask);
    result.y = batch_affine_fp32_select(when_clear.y, when_set.y, mask);
    return result;
}

inline MidnightBatchAffinePreparedPair batch_affine_prepare_point_pair(
    MidnightBatchAffinePoint32 lhs,
    MidnightBatchAffinePoint32 rhs)
{
    const uint lhs_infinity = batch_affine_point_infinity_mask(lhs);
    const uint rhs_infinity = batch_affine_point_infinity_mask(rhs);
    const uint finite = ~(lhs_infinity | rhs_infinity);
    const uint same_x = batch_affine_fp32_equal_mask(lhs.x, rhs.x);
    const uint same_y = batch_affine_fp32_equal_mask(lhs.y, rhs.y);
    const uint doubling = finite & same_x & same_y;
    const uint normal = finite & ~same_x;

    const Bls12_381Fp32 normal_denominator = fp32_sub(rhs.x, lhs.x);
    const Bls12_381Fp32 normal_numerator = fp32_sub(rhs.y, lhs.y);
    const Bls12_381Fp32 double_denominator = fp32_add(lhs.y, lhs.y);
    const Bls12_381Fp32 x_squared =
        fp32_montgomery_mul_wide(lhs.x, lhs.x);
    const Bls12_381Fp32 double_numerator =
        fp32_add(fp32_add(x_squared, x_squared), x_squared);

    MidnightBatchAffinePreparedPair result;
    result.denominator = batch_affine_fp32_select(
        normal_denominator, double_denominator, doubling);
    result.numerator = batch_affine_fp32_select(
        normal_numerator, double_numerator, doubling);
    const uint doubling_valid =
        doubling & ~batch_affine_fp32_zero_mask(double_denominator);
    result.formula_valid = normal | doubling_valid;
    result.denominator = batch_affine_fp32_select(
        result.denominator,
        batch_affine_fp32_one(),
        ~result.formula_valid);
    result.lhs_infinity = lhs_infinity;
    result.rhs_infinity = rhs_infinity;
    return result;
}

inline MidnightBatchAffinePoint32 batch_affine_finish_point_pair(
    MidnightBatchAffinePoint32 lhs,
    MidnightBatchAffinePoint32 rhs,
    Bls12_381Fp32 numerator,
    Bls12_381Fp32 inverse_denominator,
    uint formula_mask,
    uint lhs_infinity_mask,
    uint rhs_infinity_mask)
{
    const Bls12_381Fp32 lambda = fp32_montgomery_mul_wide(
        numerator, inverse_denominator);
    const Bls12_381Fp32 lambda_squared =
        fp32_montgomery_mul_wide(lambda, lambda);

    MidnightBatchAffinePoint32 formula_result;
    formula_result.x = fp32_sub(fp32_sub(lambda_squared, lhs.x), rhs.x);
    formula_result.y = fp32_sub(
        fp32_montgomery_mul_wide(
            lambda, fp32_sub(lhs.x, formula_result.x)),
        lhs.y);

    MidnightBatchAffinePoint32 result = batch_affine_point_select(
        batch_affine_point_identity(), formula_result, formula_mask);
    result = batch_affine_point_select(result, rhs, lhs_infinity_mask);
    return batch_affine_point_select(result, lhs, rhs_infinity_mask);
}

inline void batch_affine_status_or(
    device MidnightBatchAffineStatus* status,
    uint flags)
{
    atomic_fetch_or_explicit(&status->flags, flags, memory_order_relaxed);
}

inline void batch_affine_status_destination_error(
    device MidnightBatchAffineStatus* status,
    uint invalid)
{
    const uint mask = 0U - invalid;
    batch_affine_status_or(
        status,
        MIDNIGHT_BATCH_AFFINE_STATUS_DESTINATION_OUT_OF_RANGE & mask);
    atomic_fetch_add_explicit(
        &status->destination_errors, invalid, memory_order_relaxed);
}

inline Bls12_381Fp32 batch_affine_fp32_inverse_fixed(
    Bls12_381Fp32 value)
{
    // Fixed public exponent p-2. The simple schedule is retained for the
    // correctness probe so chain selection can be benchmarked independently.
    Bls12_381Fp32 result = batch_affine_fp32_one();
    for (int bit = 383; bit >= 0; --bit) {
        result = fp32_montgomery_mul_wide(result, result);
        const uint word_index = static_cast<uint>(bit) >> 6;
        const uint bit_index = static_cast<uint>(bit) & 63U;
        if (((MIDNIGHT_BLS12_381_FP_MODULUS_MINUS_TWO[word_index] >>
              bit_index) &
             1UL) != 0) {
            result = fp32_montgomery_mul_wide(result, value);
        }
    }
    return result;
}

kernel void midnight_bls12_381_batch_affine_abi_probe(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }

    output[0] = 1;
    output[1] = sizeof(Bls12_381Fp32);
    output[2] = alignof(Bls12_381Fp32);
    output[3] = __builtin_offsetof(Bls12_381Fp32, limbs);
    output[4] = sizeof(MidnightBatchAffinePoint32);
    output[5] = alignof(MidnightBatchAffinePoint32);
    output[6] = __builtin_offsetof(MidnightBatchAffinePoint32, x);
    output[7] = __builtin_offsetof(MidnightBatchAffinePoint32, y);
    output[8] = sizeof(MidnightBatchAffineGatherParameters);
    output[9] = alignof(MidnightBatchAffineGatherParameters);
    output[10] = __builtin_offsetof(
        MidnightBatchAffineGatherParameters, record_offset);
    output[11] = __builtin_offsetof(
        MidnightBatchAffineGatherParameters, record_count);
    output[12] = __builtin_offsetof(
        MidnightBatchAffineGatherParameters, srs_count);
    output[13] = __builtin_offsetof(
        MidnightBatchAffineGatherParameters, reserved);
    output[14] = sizeof(MidnightBatchAffineLevelParameters);
    output[15] = alignof(MidnightBatchAffineLevelParameters);
    output[16] = __builtin_offsetof(
        MidnightBatchAffineLevelParameters, destination_map_offset);
    output[17] = __builtin_offsetof(
        MidnightBatchAffineLevelParameters, pair_count);
    output[18] = __builtin_offsetof(
        MidnightBatchAffineLevelParameters, carry_count);
    output[19] = __builtin_offsetof(
        MidnightBatchAffineLevelParameters, destination_len);
    output[20] = sizeof(MidnightBatchAffineProductParameters);
    output[21] = alignof(MidnightBatchAffineProductParameters);
    output[22] = __builtin_offsetof(
        MidnightBatchAffineProductParameters, input_offset);
    output[23] = __builtin_offsetof(
        MidnightBatchAffineProductParameters, input_count);
    output[24] = __builtin_offsetof(
        MidnightBatchAffineProductParameters, output_offset);
    output[25] = __builtin_offsetof(
        MidnightBatchAffineProductParameters, output_count);
    output[26] = sizeof(MidnightBatchAffineStatus);
    output[27] = alignof(MidnightBatchAffineStatus);
    output[28] = __builtin_offsetof(MidnightBatchAffineStatus, flags);
    output[29] = __builtin_offsetof(
        MidnightBatchAffineStatus, record_errors);
    output[30] = __builtin_offsetof(
        MidnightBatchAffineStatus, destination_errors);
    output[31] = __builtin_offsetof(MidnightBatchAffineStatus, reserved);
}

kernel void midnight_bls12_381_batch_affine_ceiling_abi_probe(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }

    output[0] = 1;
    output[1] = sizeof(MidnightBatchAffineCeilingParameters);
    output[2] = alignof(MidnightBatchAffineCeilingParameters);
    output[3] = __builtin_offsetof(
        MidnightBatchAffineCeilingParameters, element_count);
    output[4] = __builtin_offsetof(
        MidnightBatchAffineCeilingParameters, iterations);
    output[5] = __builtin_offsetof(
        MidnightBatchAffineCeilingParameters, reserved_0);
    output[6] = __builtin_offsetof(
        MidnightBatchAffineCeilingParameters, reserved_1);
}

kernel void midnight_bls12_381_gather_signed_leaves(
    device const Bls12_381G1Affine* srs [[buffer(0)]],
    device const uint* packed_records [[buffer(1)]],
    device MidnightBatchAffinePoint32* destination [[buffer(2)]],
    device MidnightBatchAffineStatus* status [[buffer(3)]],
    constant MidnightBatchAffineGatherParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.record_count) {
        return;
    }
    const uint packed = packed_records[parameters.record_offset + index];
    const uint source_index = packed & 0x7fffffffU;
    // The diagnostic host validates a nonempty SRS and every packed index
    // before encoding. Keeping that invariant at the ownership boundary avoids
    // secret-derived bounds control flow in this gather kernel.
    const uint negate_mask = 0U - (packed >> 31);

    MidnightBatchAffinePoint32 point;
    point.x = fp32_from_public(srs[source_index].x);
    point.y = fp32_from_public(srs[source_index].y);
    const Bls12_381Fp32 negated_y =
        fp32_sub(batch_affine_fp32_zero(), point.y);
    point.y = batch_affine_fp32_select(point.y, negated_y, negate_mask);
    destination[index] = point;
}

kernel void midnight_bls12_381_prepare_pairs(
    device const MidnightBatchAffinePoint32* source [[buffer(0)]],
    device Bls12_381Fp32* numerators [[buffer(1)]],
    device uint* cases [[buffer(2)]],
    device Bls12_381Fp32* denominator_leaves [[buffer(3)]],
    device MidnightBatchAffineStatus* status [[buffer(4)]],
    constant MidnightBatchAffineLevelParameters& parameters [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.pair_count) {
        return;
    }

    const MidnightBatchAffinePoint32 lhs = source[2 * index];
    const MidnightBatchAffinePoint32 rhs = source[2 * index + 1];
    const MidnightBatchAffinePreparedPair prepared =
        batch_affine_prepare_point_pair(lhs, rhs);

    numerators[index] = prepared.numerator;
    denominator_leaves[index] = prepared.denominator;
    cases[index] =
        ((prepared.formula_valid & 1U) *
         MIDNIGHT_BATCH_AFFINE_CASE_FORMULA_VALID) |
        ((prepared.lhs_infinity & 1U) *
         MIDNIGHT_BATCH_AFFINE_CASE_LHS_INFINITY) |
        ((prepared.rhs_infinity & 1U) *
         MIDNIGHT_BATCH_AFFINE_CASE_RHS_INFINITY);
}

kernel void midnight_bls12_381_reduce_products(
    device Bls12_381Fp32* tree [[buffer(0)]],
    device MidnightBatchAffineStatus* status [[buffer(1)]],
    constant MidnightBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const uint expected_output_count = (parameters.input_count + 1U) >> 1;
    if (parameters.input_count == 0 ||
        parameters.output_count != expected_output_count) {
        if (index == 0) {
            batch_affine_status_or(
                status, MIDNIGHT_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (index >= parameters.output_count) {
        return;
    }

    const uint left_index = parameters.input_offset + 2U * index;
    const uint right_relative = 2U * index + 1U;
    const Bls12_381Fp32 left = tree[left_index];
    Bls12_381Fp32 right = batch_affine_fp32_one();
    if (right_relative < parameters.input_count) {
        right = tree[parameters.input_offset + right_relative];
    }
    const Bls12_381Fp32 product =
        fp32_montgomery_mul_wide(left, right);
    tree[parameters.output_offset + index] = product;
}

kernel void midnight_bls12_381_invert_product_root(
    device Bls12_381Fp32* tree [[buffer(0)]],
    device MidnightBatchAffineStatus* status [[buffer(1)]],
    constant MidnightBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    if (parameters.input_count != 1) {
        batch_affine_status_or(
            status, MIDNIGHT_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        return;
    }

    const Bls12_381Fp32 root = tree[parameters.input_offset];
    const Bls12_381Fp32 inverse = batch_affine_fp32_inverse_fixed(root);
    const uint zero =
        (batch_affine_fp32_zero_mask(root) |
         batch_affine_fp32_zero_mask(inverse)) &
        1U;
    batch_affine_status_or(
        status,
        MIDNIGHT_BATCH_AFFINE_STATUS_INVERSE_ZERO & (0U - zero));
    tree[parameters.input_offset] = inverse;
}

kernel void midnight_bls12_381_expand_product_inverses(
    device Bls12_381Fp32* tree [[buffer(0)]],
    device MidnightBatchAffineStatus* status [[buffer(1)]],
    constant MidnightBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const uint expected_input_count = (parameters.output_count + 1U) >> 1;
    if (parameters.output_count == 0 ||
        parameters.input_count != expected_input_count) {
        if (index == 0) {
            batch_affine_status_or(
                status, MIDNIGHT_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (index >= parameters.input_count) {
        return;
    }

    const uint left_relative = 2U * index;
    const uint right_relative = left_relative + 1U;
    const Bls12_381Fp32 parent_inverse =
        tree[parameters.input_offset + index];
    const Bls12_381Fp32 left_product =
        tree[parameters.output_offset + left_relative];
    Bls12_381Fp32 right_product = batch_affine_fp32_one();
    if (right_relative < parameters.output_count) {
        right_product = tree[parameters.output_offset + right_relative];
    }

    tree[parameters.output_offset + left_relative] =
        fp32_montgomery_mul_wide(parent_inverse, right_product);
    if (right_relative < parameters.output_count) {
        tree[parameters.output_offset + right_relative] =
            fp32_montgomery_mul_wide(parent_inverse, left_product);
    }
}

kernel void midnight_bls12_381_finish_pairs(
    device const MidnightBatchAffinePoint32* source [[buffer(0)]],
    device const Bls12_381Fp32* numerators [[buffer(1)]],
    device const uint* cases [[buffer(2)]],
    device const Bls12_381Fp32* inverse_denominators [[buffer(3)]],
    device const uint* destination_map [[buffer(4)]],
    device MidnightBatchAffinePoint32* destination [[buffer(5)]],
    device MidnightBatchAffineStatus* status [[buffer(6)]],
    constant MidnightBatchAffineLevelParameters& parameters [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.pair_count) {
        return;
    }
    if (parameters.destination_len == 0) {
        batch_affine_status_destination_error(status, 1);
        return;
    }

    const MidnightBatchAffinePoint32 lhs = source[2U * index];
    const MidnightBatchAffinePoint32 rhs = source[2U * index + 1U];
    const uint pair_case = cases[index];
    const uint formula_mask =
        0U - ((pair_case & MIDNIGHT_BATCH_AFFINE_CASE_FORMULA_VALID) != 0);
    const uint lhs_infinity_mask =
        0U - ((pair_case & MIDNIGHT_BATCH_AFFINE_CASE_LHS_INFINITY) != 0);
    const uint rhs_infinity_mask =
        0U - ((pair_case & MIDNIGHT_BATCH_AFFINE_CASE_RHS_INFINITY) != 0);
    const MidnightBatchAffinePoint32 result = batch_affine_finish_point_pair(
        lhs,
        rhs,
        numerators[index],
        inverse_denominators[index],
        formula_mask,
        lhs_infinity_mask,
        rhs_infinity_mask);

    const uint mapped_index =
        destination_map[parameters.destination_map_offset + index];
    if (mapped_index >= parameters.destination_len) {
        batch_affine_status_destination_error(status, 1);
        return;
    }
    destination[mapped_index] = result;
}

kernel void midnight_bls12_381_copy_carries(
    device const MidnightBatchAffinePoint32* source [[buffer(0)]],
    device const uint* destination_map [[buffer(1)]],
    device MidnightBatchAffinePoint32* destination [[buffer(2)]],
    device MidnightBatchAffineStatus* status [[buffer(3)]],
    constant MidnightBatchAffineLevelParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.carry_count) {
        return;
    }
    if (parameters.destination_len == 0) {
        batch_affine_status_destination_error(status, 1);
        return;
    }

    const uint source_index = 2U * parameters.pair_count + index;
    const uint map_index =
        parameters.destination_map_offset + parameters.pair_count + index;
    const uint mapped_index = destination_map[map_index];
    if (mapped_index >= parameters.destination_len) {
        batch_affine_status_destination_error(status, 1);
        return;
    }
    destination[mapped_index] = source[source_index];
}

kernel void midnight_bls12_381_export_roots(
    device const MidnightBatchAffinePoint32* source [[buffer(0)]],
    device Bls12_381G1Affine* public_roots [[buffer(1)]],
    device MidnightBatchAffineStatus* status [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) {
        return;
    }

    const MidnightBatchAffinePoint32 point = source[index];
    public_roots[index].x = fp32_to_public(point.x);
    public_roots[index].y = fp32_to_public(point.y);
}

// Optimistic arithmetic ceiling only: inputs and the final result touch device
// memory once per dispatch, while all pair iterations remain register-local.
// prepare contributes 1M/S + 5A/S, the surrogate scan contributes 3M/S, and
// finish contributes 3M/S + 4A/S: exactly 7M/S + 9A/S per pair iteration.
kernel void midnight_bls12_381_batch_affine_arithmetic_ceiling(
    device const Bls12_381G1Affine* lhs_input [[buffer(0)]],
    device const Bls12_381G1Affine* rhs_input [[buffer(1)]],
    device Bls12_381G1Affine* point_output [[buffer(2)]],
    device Bls12_381Fp* scan_output [[buffer(3)]],
    constant MidnightBatchAffineCeilingParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.element_count || parameters.iterations == 0U) {
        return;
    }

    MidnightBatchAffinePoint32 lhs;
    lhs.x = fp32_from_public(lhs_input[index].x);
    lhs.y = fp32_from_public(lhs_input[index].y);
    MidnightBatchAffinePoint32 rhs;
    rhs.x = fp32_from_public(rhs_input[index].x);
    rhs.y = fp32_from_public(rhs_input[index].y);
    Bls12_381Fp32 scan = lhs.x;

    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        const MidnightBatchAffinePreparedPair prepared =
            batch_affine_prepare_point_pair(lhs, rhs);
        scan = fp32_montgomery_mul_wide(scan, prepared.denominator);
        scan = fp32_montgomery_mul_wide(scan, prepared.numerator);
        scan = fp32_montgomery_mul_wide(scan, prepared.denominator);
        lhs = batch_affine_finish_point_pair(
            lhs,
            rhs,
            prepared.numerator,
            scan,
            prepared.formula_valid,
            prepared.lhs_infinity,
            prepared.rhs_infinity);
    }

    point_output[index].x = fp32_to_public(lhs.x);
    point_output[index].y = fp32_to_public(lhs.y);
    // A direct observable makes all three surrogate scan products live even
    // when an exceptional point case masks the formula result.
    scan_output[index] = fp32_to_public(scan);
}

#endif
