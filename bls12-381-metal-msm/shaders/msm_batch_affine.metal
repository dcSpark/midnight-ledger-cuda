// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#ifndef BLS12_381_METAL_MSM_BATCH_AFFINE
#define BLS12_381_METAL_MSM_BATCH_AFFINE

#include "msm_g1.metal"

// Global batch-affine reducer extracted from the validated prover diagnostic.
// It reduces independent signed runs with one batch inversion per tree level.
// Mask fields are either all-zero or all-one words.
struct Bls12381MetalMsmBatchAffinePreparedPair {
    Bls12381MetalMsmFp numerator;
    Bls12381MetalMsmFp denominator;
    uint formula_valid;
    uint lhs_infinity;
    uint rhs_infinity;
};

struct Bls12381MetalMsmBatchAffineGatherParameters {
    uint record_offset;
    uint record_count;
    uint srs_count;
    uint reserved;
};

struct Bls12381MetalMsmBatchAffineLevelParameters {
    uint destination_map_offset;
    uint pair_count;
    uint carry_count;
    uint destination_len;
};

struct Bls12381MetalMsmBatchAffineProductParameters {
    uint input_offset;
    uint input_count;
    uint output_offset;
    uint output_count;
};

struct Bls12381MetalMsmBatchAffineStatus {
    atomic_uint flags;
    atomic_uint record_errors;
    atomic_uint destination_errors;
    atomic_uint reserved;
};

constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_FORMULA_VALID = 1U << 0;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_LHS_INFINITY = 1U << 1;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_RHS_INFINITY = 1U << 2;
#ifdef BLS12_381_METAL_MSM_DIAGNOSTICS
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1 = 32U;
#endif

constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_ADJUSTED_DENOMINATOR_ZERO = 1U << 0;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_PRODUCT_ZERO = 1U << 1;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVERSE_ZERO = 1U << 2;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS = 1U << 3;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_RECORD_OUT_OF_RANGE = 1U << 4;
constant uint BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_DESTINATION_OUT_OF_RANGE = 1U << 5;
// Bit 6 is reserved. Coordinate shape alone cannot distinguish every finite
// curve point from a malformed infinity encoding; immutable key import owns
// that validation.

inline Bls12381MetalMsmFp bls12_381_msm_batch_affine_fp32_zero()
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = 0;
    }
    return result;
}

inline Bls12381MetalMsmFp bls12_381_msm_batch_affine_fp32_one()
{
    return bls12_381_msm_fp_one();
}

inline uint bls12_381_msm_batch_affine_fp32_zero_mask(Bls12381MetalMsmFp value)
{
    uint aggregate = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        aggregate |= value.limbs[index];
    }
    return 0U - static_cast<uint>(aggregate == 0);
}

inline uint bls12_381_msm_batch_affine_fp32_equal_mask(
    Bls12381MetalMsmFp lhs,
    Bls12381MetalMsmFp rhs)
{
    uint aggregate = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        aggregate |= lhs.limbs[index] ^ rhs.limbs[index];
    }
    return 0U - static_cast<uint>(aggregate == 0);
}

inline Bls12381MetalMsmFp bls12_381_msm_batch_affine_fp32_select(
    Bls12381MetalMsmFp when_clear,
    Bls12381MetalMsmFp when_set,
    uint mask)
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] =
            (when_clear.limbs[index] & ~mask) |
            (when_set.limbs[index] & mask);
    }
    return result;
}

inline Bls12381MetalMsmAffine bls12_381_msm_batch_affine_point_identity()
{
    Bls12381MetalMsmAffine result;
    result.x = bls12_381_msm_batch_affine_fp32_zero();
    result.y = bls12_381_msm_batch_affine_fp32_zero();
    return result;
}

inline uint bls12_381_msm_batch_affine_point_infinity_mask(
    Bls12381MetalMsmAffine point)
{
    return bls12_381_msm_batch_affine_fp32_zero_mask(point.x) &
        bls12_381_msm_batch_affine_fp32_zero_mask(point.y);
}

inline Bls12381MetalMsmAffine bls12_381_msm_batch_affine_point_select(
    Bls12381MetalMsmAffine when_clear,
    Bls12381MetalMsmAffine when_set,
    uint mask)
{
    Bls12381MetalMsmAffine result;
    result.x = bls12_381_msm_batch_affine_fp32_select(when_clear.x, when_set.x, mask);
    result.y = bls12_381_msm_batch_affine_fp32_select(when_clear.y, when_set.y, mask);
    return result;
}

inline Bls12381MetalMsmBatchAffinePreparedPair bls12_381_msm_batch_affine_prepare_point_pair(
    Bls12381MetalMsmAffine lhs,
    Bls12381MetalMsmAffine rhs)
{
    const uint lhs_infinity = bls12_381_msm_batch_affine_point_infinity_mask(lhs);
    const uint rhs_infinity = bls12_381_msm_batch_affine_point_infinity_mask(rhs);
    const uint finite = ~(lhs_infinity | rhs_infinity);
    const uint same_x = bls12_381_msm_batch_affine_fp32_equal_mask(lhs.x, rhs.x);
    const uint same_y = bls12_381_msm_batch_affine_fp32_equal_mask(lhs.y, rhs.y);
    const uint doubling = finite & same_x & same_y;
    const uint normal = finite & ~same_x;

    const Bls12381MetalMsmFp normal_denominator = bls12_381_msm_fp_sub(rhs.x, lhs.x);
    const Bls12381MetalMsmFp normal_numerator = bls12_381_msm_fp_sub(rhs.y, lhs.y);
    const Bls12381MetalMsmFp double_denominator = bls12_381_msm_fp_add(lhs.y, lhs.y);
    const Bls12381MetalMsmFp x_squared =
        bls12_381_msm_fp_mul(lhs.x, lhs.x);
    const Bls12381MetalMsmFp double_numerator =
        bls12_381_msm_fp_add(bls12_381_msm_fp_add(x_squared, x_squared), x_squared);

    Bls12381MetalMsmBatchAffinePreparedPair result;
    result.denominator = bls12_381_msm_batch_affine_fp32_select(
        normal_denominator, double_denominator, doubling);
    result.numerator = bls12_381_msm_batch_affine_fp32_select(
        normal_numerator, double_numerator, doubling);
    const uint doubling_valid =
        doubling & ~bls12_381_msm_batch_affine_fp32_zero_mask(double_denominator);
    result.formula_valid = normal | doubling_valid;
    result.denominator = bls12_381_msm_batch_affine_fp32_select(
        result.denominator,
        bls12_381_msm_batch_affine_fp32_one(),
        ~result.formula_valid);
    result.lhs_infinity = lhs_infinity;
    result.rhs_infinity = rhs_infinity;
    return result;
}

inline Bls12381MetalMsmFp bls12_381_msm_batch_affine_adjusted_denominator(
    Bls12381MetalMsmAffine lhs,
    Bls12381MetalMsmAffine rhs)
{
    const uint lhs_infinity = bls12_381_msm_batch_affine_point_infinity_mask(lhs);
    const uint rhs_infinity = bls12_381_msm_batch_affine_point_infinity_mask(rhs);
    const uint finite = ~(lhs_infinity | rhs_infinity);
    const uint same_x = bls12_381_msm_batch_affine_fp32_equal_mask(lhs.x, rhs.x);
    const uint same_y = bls12_381_msm_batch_affine_fp32_equal_mask(lhs.y, rhs.y);
    const uint doubling = finite & same_x & same_y;
    const uint normal = finite & ~same_x;
    const Bls12381MetalMsmFp normal_denominator =
        bls12_381_msm_fp_sub(rhs.x, lhs.x);
    const Bls12381MetalMsmFp double_denominator =
        bls12_381_msm_fp_add(lhs.y, lhs.y);
    Bls12381MetalMsmFp denominator = bls12_381_msm_batch_affine_fp32_select(
        normal_denominator, double_denominator, doubling);
    const uint doubling_valid =
        doubling & ~bls12_381_msm_batch_affine_fp32_zero_mask(double_denominator);
    return bls12_381_msm_batch_affine_fp32_select(
        denominator,
        bls12_381_msm_batch_affine_fp32_one(),
        ~(normal | doubling_valid));
}

inline Bls12381MetalMsmAffine bls12_381_msm_batch_affine_finish_point_pair(
    Bls12381MetalMsmAffine lhs,
    Bls12381MetalMsmAffine rhs,
    Bls12381MetalMsmFp numerator,
    Bls12381MetalMsmFp inverse_denominator,
    uint formula_mask,
    uint lhs_infinity_mask,
    uint rhs_infinity_mask)
{
    const Bls12381MetalMsmFp lambda = bls12_381_msm_fp_mul(
        numerator, inverse_denominator);
    const Bls12381MetalMsmFp lambda_squared =
        bls12_381_msm_fp_mul(lambda, lambda);

    Bls12381MetalMsmAffine formula_result;
    formula_result.x = bls12_381_msm_fp_sub(bls12_381_msm_fp_sub(lambda_squared, lhs.x), rhs.x);
    formula_result.y = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(
            lambda, bls12_381_msm_fp_sub(lhs.x, formula_result.x)),
        lhs.y);

    Bls12381MetalMsmAffine result = bls12_381_msm_batch_affine_point_select(
        bls12_381_msm_batch_affine_point_identity(), formula_result, formula_mask);
    result = bls12_381_msm_batch_affine_point_select(result, rhs, lhs_infinity_mask);
    return bls12_381_msm_batch_affine_point_select(result, lhs, rhs_infinity_mask);
}

inline void bls12_381_msm_batch_affine_status_or(
    device Bls12381MetalMsmBatchAffineStatus* status,
    uint flags)
{
    atomic_fetch_or_explicit(&status->flags, flags, memory_order_relaxed);
}

inline void bls12_381_msm_batch_affine_status_destination_error(
    device Bls12381MetalMsmBatchAffineStatus* status,
    uint invalid)
{
    const uint mask = 0U - invalid;
    bls12_381_msm_batch_affine_status_or(
        status,
        BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_DESTINATION_OUT_OF_RANGE & mask);
    atomic_fetch_add_explicit(
        &status->destination_errors, invalid, memory_order_relaxed);
}

inline void bls12_381_msm_batch_affine_status_record_error(
    device Bls12381MetalMsmBatchAffineStatus* status,
    uint flags)
{
    bls12_381_msm_batch_affine_status_or(status, flags);
    atomic_fetch_add_explicit(
        &status->record_errors, 1U, memory_order_relaxed);
}

inline Bls12381MetalMsmFp bls12_381_msm_batch_affine_fp32_inverse_fixed(
    Bls12381MetalMsmFp value)
{
    // Fixed public exponent p-2. The simple schedule is retained for the
    // correctness probe so chain selection can be benchmarked independently.
    Bls12381MetalMsmFp result = bls12_381_msm_batch_affine_fp32_one();
    for (int bit = 383; bit >= 0; --bit) {
        result = bls12_381_msm_fp_mul(result, result);
        const uint word_index = static_cast<uint>(bit) >> 6;
        const uint bit_index = static_cast<uint>(bit) & 63U;
        if (((BLS12_381_METAL_MSM_FP_MODULUS_MINUS_TWO[word_index] >>
              bit_index) &
             1UL) != 0) {
            result = bls12_381_msm_fp_mul(result, value);
        }
    }
    return result;
}

kernel void bls12_381_metal_msm_batch_affine_gather_signed_leaves(
    device const Bls12381MetalMsmAffine* srs [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device Bls12381MetalMsmAffine* destination [[buffer(2)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmBatchAffineGatherParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.record_count) {
        return;
    }
    if (parameters.reserved != 0U) {
        bls12_381_msm_batch_affine_status_record_error(
            status,
            BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        destination[index] = bls12_381_msm_batch_affine_point_identity();
        return;
    }
    const Bls12381MetalMsmSignedRecord record =
        records[parameters.record_offset + index];
    if ((record.flags & ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U ||
        record.point_index >= parameters.srs_count) {
        bls12_381_msm_batch_affine_status_record_error(
            status,
            BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_RECORD_OUT_OF_RANGE);
        destination[index] = bls12_381_msm_batch_affine_point_identity();
        return;
    }
    const uint source_index = record.point_index;
    const uint negate_mask = 0U - (record.flags & 1U);

    Bls12381MetalMsmAffine point;
    point.x = srs[source_index].x;
    point.y = srs[source_index].y;
    const Bls12381MetalMsmFp negated_y =
        bls12_381_msm_fp_sub(bls12_381_msm_batch_affine_fp32_zero(), point.y);
    point.y = bls12_381_msm_batch_affine_fp32_select(point.y, negated_y, negate_mask);
    destination[index] = point;
}

#ifdef BLS12_381_METAL_MSM_DIAGNOSTICS
kernel void bls12_381_metal_msm_batch_affine_group_products_v1(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device Bls12381MetalMsmFp* group_products [[buffer(1)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(2)]],
    constant Bls12381MetalMsmBatchAffineLevelParameters& parameters [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    if (threads.x != BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1 ||
        threads.y != 1U || threads.z != 1U) {
        if (lane == 0U) {
            bls12_381_msm_batch_affine_status_or(
                status,
                BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint pair_index =
        group.x * BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1 + lane;
    Bls12381MetalMsmFp denominator = bls12_381_msm_batch_affine_fp32_one();
    if (pair_index < parameters.pair_count) {
        denominator = bls12_381_msm_batch_affine_adjusted_denominator(
            source[2U * pair_index], source[2U * pair_index + 1U]);
    }

    threadgroup Bls12381MetalMsmFp tree[63];
    tree[31U + lane] = denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 16U; node_count != 0U; node_count >>= 1U) {
        if (lane < node_count) {
            const uint node = node_count - 1U + lane;
            tree[node] = bls12_381_msm_fp_mul(
                tree[2U * node + 1U], tree[2U * node + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        group_products[group.x] = tree[0];
    }
}

kernel void bls12_381_metal_msm_batch_affine_finish_groups_v1(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device const Bls12381MetalMsmFp* inverse_group_products [[buffer(1)]],
    device const uint* destination_map [[buffer(2)]],
    device Bls12381MetalMsmAffine* destination [[buffer(3)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmBatchAffineLevelParameters& parameters [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    if (threads.x != BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1 ||
        threads.y != 1U || threads.z != 1U) {
        if (lane == 0U) {
            bls12_381_msm_batch_affine_status_or(
                status,
                BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const uint pair_index =
        group.x * BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1 + lane;
    const bool active = pair_index < parameters.pair_count;
    Bls12381MetalMsmAffine lhs = bls12_381_msm_batch_affine_point_identity();
    Bls12381MetalMsmAffine rhs = bls12_381_msm_batch_affine_point_identity();
    Bls12381MetalMsmBatchAffinePreparedPair prepared;
    prepared.numerator = bls12_381_msm_batch_affine_fp32_zero();
    prepared.denominator = bls12_381_msm_batch_affine_fp32_one();
    prepared.formula_valid = 0U;
    prepared.lhs_infinity = 0xffffffffU;
    prepared.rhs_infinity = 0xffffffffU;
    if (active) {
        lhs = source[2U * pair_index];
        rhs = source[2U * pair_index + 1U];
        prepared = bls12_381_msm_batch_affine_prepare_point_pair(lhs, rhs);
    }

    threadgroup Bls12381MetalMsmFp tree[63];
    tree[31U + lane] = prepared.denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 16U; node_count != 0U; node_count >>= 1U) {
        if (lane < node_count) {
            const uint node = node_count - 1U + lane;
            tree[node] = bls12_381_msm_fp_mul(
                tree[2U * node + 1U], tree[2U * node + 2U]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        tree[0] = inverse_group_products[group.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node_count = 1U;
         node_count < BLS12_381_METAL_MSM_BATCH_AFFINE_GROUP_THREADS_V1;
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

    if (!active) {
        return;
    }
    if (parameters.destination_len == 0U) {
        bls12_381_msm_batch_affine_status_destination_error(status, 1U);
        return;
    }
    const Bls12381MetalMsmAffine result =
        bls12_381_msm_batch_affine_finish_point_pair(
            lhs,
            rhs,
            prepared.numerator,
            tree[31U + lane],
            prepared.formula_valid,
            prepared.lhs_infinity,
            prepared.rhs_infinity);
    const uint mapped_index =
        destination_map[parameters.destination_map_offset + pair_index];
    if (mapped_index >= parameters.destination_len) {
        bls12_381_msm_batch_affine_status_destination_error(status, 1U);
        return;
    }
    destination[mapped_index] = result;
}
#endif

kernel void bls12_381_metal_msm_batch_affine_prepare_pairs(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device Bls12381MetalMsmFp* numerators [[buffer(1)]],
    device uint* cases [[buffer(2)]],
    device Bls12381MetalMsmFp* denominator_leaves [[buffer(3)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmBatchAffineLevelParameters& parameters [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.pair_count) {
        return;
    }

    const Bls12381MetalMsmAffine lhs = source[2 * index];
    const Bls12381MetalMsmAffine rhs = source[2 * index + 1];
    const Bls12381MetalMsmBatchAffinePreparedPair prepared =
        bls12_381_msm_batch_affine_prepare_point_pair(lhs, rhs);

    numerators[index] = prepared.numerator;
    denominator_leaves[index] = prepared.denominator;
    cases[index] =
        ((prepared.formula_valid & 1U) *
         BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_FORMULA_VALID) |
        ((prepared.lhs_infinity & 1U) *
         BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_LHS_INFINITY) |
        ((prepared.rhs_infinity & 1U) *
         BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_RHS_INFINITY);
}

kernel void bls12_381_metal_msm_batch_affine_reduce_products(
    device Bls12381MetalMsmFp* tree [[buffer(0)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(1)]],
    constant Bls12381MetalMsmBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const uint expected_output_count = (parameters.input_count + 1U) >> 1;
    if (parameters.input_count == 0 ||
        parameters.output_count != expected_output_count) {
        if (index == 0) {
            bls12_381_msm_batch_affine_status_or(
                status, BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (index >= parameters.output_count) {
        return;
    }

    const uint left_index = parameters.input_offset + 2U * index;
    const uint right_relative = 2U * index + 1U;
    const Bls12381MetalMsmFp left = tree[left_index];
    Bls12381MetalMsmFp right = bls12_381_msm_batch_affine_fp32_one();
    if (right_relative < parameters.input_count) {
        right = tree[parameters.input_offset + right_relative];
    }
    const Bls12381MetalMsmFp product =
        bls12_381_msm_fp_mul(left, right);
    tree[parameters.output_offset + index] = product;
}

kernel void bls12_381_metal_msm_batch_affine_invert_product_root(
    device Bls12381MetalMsmFp* tree [[buffer(0)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(1)]],
    constant Bls12381MetalMsmBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    if (parameters.input_count != 1) {
        bls12_381_msm_batch_affine_status_or(
            status, BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        return;
    }

    const Bls12381MetalMsmFp root = tree[parameters.input_offset];
    const Bls12381MetalMsmFp inverse = bls12_381_msm_batch_affine_fp32_inverse_fixed(root);
    const uint zero =
        (bls12_381_msm_batch_affine_fp32_zero_mask(root) |
         bls12_381_msm_batch_affine_fp32_zero_mask(inverse)) &
        1U;
    bls12_381_msm_batch_affine_status_or(
        status,
        BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVERSE_ZERO & (0U - zero));
    tree[parameters.input_offset] = inverse;
}

kernel void bls12_381_metal_msm_batch_affine_expand_product_inverses(
    device Bls12381MetalMsmFp* tree [[buffer(0)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(1)]],
    constant Bls12381MetalMsmBatchAffineProductParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const uint expected_input_count = (parameters.output_count + 1U) >> 1;
    if (parameters.output_count == 0 ||
        parameters.input_count != expected_input_count) {
        if (index == 0) {
            bls12_381_msm_batch_affine_status_or(
                status, BLS12_381_METAL_MSM_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }
    if (index >= parameters.input_count) {
        return;
    }

    const uint left_relative = 2U * index;
    const uint right_relative = left_relative + 1U;
    const Bls12381MetalMsmFp parent_inverse =
        tree[parameters.input_offset + index];
    const Bls12381MetalMsmFp left_product =
        tree[parameters.output_offset + left_relative];
    Bls12381MetalMsmFp right_product = bls12_381_msm_batch_affine_fp32_one();
    if (right_relative < parameters.output_count) {
        right_product = tree[parameters.output_offset + right_relative];
    }

    tree[parameters.output_offset + left_relative] =
        bls12_381_msm_fp_mul(parent_inverse, right_product);
    if (right_relative < parameters.output_count) {
        tree[parameters.output_offset + right_relative] =
            bls12_381_msm_fp_mul(parent_inverse, left_product);
    }
}

kernel void bls12_381_metal_msm_batch_affine_finish_pairs(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device const Bls12381MetalMsmFp* numerators [[buffer(1)]],
    device const uint* cases [[buffer(2)]],
    device const Bls12381MetalMsmFp* inverse_denominators [[buffer(3)]],
    device const uint* destination_map [[buffer(4)]],
    device Bls12381MetalMsmAffine* destination [[buffer(5)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(6)]],
    constant Bls12381MetalMsmBatchAffineLevelParameters& parameters [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.pair_count) {
        return;
    }
    if (parameters.destination_len == 0) {
        bls12_381_msm_batch_affine_status_destination_error(status, 1);
        return;
    }

    const Bls12381MetalMsmAffine lhs = source[2U * index];
    const Bls12381MetalMsmAffine rhs = source[2U * index + 1U];
    const uint pair_case = cases[index];
    const uint formula_mask =
        0U - ((pair_case & BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_FORMULA_VALID) != 0);
    const uint lhs_infinity_mask =
        0U - ((pair_case & BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_LHS_INFINITY) != 0);
    const uint rhs_infinity_mask =
        0U - ((pair_case & BLS12_381_METAL_MSM_BATCH_AFFINE_CASE_RHS_INFINITY) != 0);
    const Bls12381MetalMsmAffine result = bls12_381_msm_batch_affine_finish_point_pair(
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
        bls12_381_msm_batch_affine_status_destination_error(status, 1);
        return;
    }
    destination[mapped_index] = result;
}

kernel void bls12_381_metal_msm_batch_affine_copy_carries(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device const uint* destination_map [[buffer(1)]],
    device Bls12381MetalMsmAffine* destination [[buffer(2)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmBatchAffineLevelParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.carry_count) {
        return;
    }
    if (parameters.destination_len == 0) {
        bls12_381_msm_batch_affine_status_destination_error(status, 1);
        return;
    }

    const uint source_index = 2U * parameters.pair_count + index;
    const uint map_index =
        parameters.destination_map_offset + parameters.pair_count + index;
    const uint mapped_index = destination_map[map_index];
    if (mapped_index >= parameters.destination_len) {
        bls12_381_msm_batch_affine_status_destination_error(status, 1);
        return;
    }
    destination[mapped_index] = source[source_index];
}

kernel void bls12_381_metal_msm_batch_affine_export_roots(
    device const Bls12381MetalMsmAffine* source [[buffer(0)]],
    device Bls12381MetalMsmHomogeneous* roots [[buffer(1)]],
    device Bls12381MetalMsmBatchAffineStatus* status [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) {
        return;
    }

    const Bls12381MetalMsmAffine point = source[index];
    const ulong identity = static_cast<ulong>(
        bls12_381_msm_batch_affine_point_infinity_mask(point));
    Bls12381MetalMsmHomogeneous root;
    root.x = bls12_381_msm_fp_select(
        point.x, bls12_381_msm_fp_zero(), identity);
    root.y = bls12_381_msm_fp_select(
        point.y, bls12_381_msm_fp_one(), identity);
    root.z = bls12_381_msm_fp_select(
        bls12_381_msm_fp_one(), bls12_381_msm_fp_zero(), identity);
    roots[index] = root;
}


#endif
