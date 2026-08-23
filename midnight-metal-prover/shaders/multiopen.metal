// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#ifndef MIDNIGHT_BLS12_381_MULTIOPEN_METAL
#define MIDNIGHT_BLS12_381_MULTIOPEN_METAL

#include <metal_stdlib>

using namespace metal;

struct MidnightMultiopenLincombParameters {
    uint poly_len;
    uint poly_count;
    uint q_count;
    uint reserved;
};

struct MidnightMultiopenDivisionParameters {
    uint poly_len;
    uint poly_count;
    uint block_count;
    uint block_size;
    uint round;
    uint use_direct_point;
    uint reserved_0;
    uint reserved_1;
    Bls12_381Fr direct_point;
};

struct MidnightMultiopenEvalParameters {
    uint poly_len;
    uint poly_count;
    uint block_count;
    uint block_size;
    Bls12_381Fr point;
};

struct MidnightMultiopenCombineParameters {
    uint poly_len;
    uint poly_count;
    uint reserved_0;
    uint reserved_1;
};

static_assert(sizeof(MidnightMultiopenLincombParameters) == 16);
static_assert(sizeof(MidnightMultiopenDivisionParameters) == 64);
static_assert(
    __builtin_offsetof(MidnightMultiopenDivisionParameters, direct_point) == 32);
static_assert(sizeof(MidnightMultiopenEvalParameters) == 48);
static_assert(__builtin_offsetof(MidnightMultiopenEvalParameters, point) == 16);
static_assert(sizeof(MidnightMultiopenCombineParameters) == 16);

kernel void midnight_bls12_381_multiopen_lincomb(
    device const Bls12_381Fr* polynomials [[buffer(0)]],
    device const uint* q_offsets [[buffer(1)]],
    device const uint* q_poly_indices [[buffer(2)]],
    device const Bls12_381Fr* q_coefficients [[buffer(3)]],
    device Bls12_381Fr* q_polynomials [[buffer(4)]],
    constant MidnightMultiopenLincombParameters& parameters [[buffer(5)]],
    uint flat_index [[thread_position_in_grid]])
{
    const ulong total
        = static_cast<ulong>(parameters.poly_len) * parameters.q_count;
    if (flat_index >= total || parameters.poly_len == 0U) {
        return;
    }
    const uint q_index = flat_index / parameters.poly_len;
    const uint coefficient_index = flat_index % parameters.poly_len;
    Bls12_381Fr value = midnight_graph_zero();
    for (uint entry = q_offsets[q_index]; entry < q_offsets[q_index + 1U]; ++entry) {
        const uint poly_index = q_poly_indices[entry];
        value = fr_add(
            value,
            fr_montgomery_mul(
                polynomials[
                    static_cast<ulong>(poly_index) * parameters.poly_len
                    + coefficient_index],
                q_coefficients[entry]));
    }
    q_polynomials[flat_index] = value;
}

inline bool midnight_multiopen_division_point(
    uint poly_index,
    device const uint* point_offsets,
    device const Bls12_381Fr* points,
    constant MidnightMultiopenDivisionParameters& parameters,
    thread Bls12_381Fr& point)
{
    if (parameters.use_direct_point != 0U) {
        point = parameters.direct_point;
        return true;
    }
    const uint start = point_offsets[poly_index];
    const uint end = point_offsets[poly_index + 1U];
    if (parameters.round >= end - start) {
        point = midnight_graph_zero();
        return false;
    }
    point = points[start + parameters.round];
    return true;
}

struct MidnightMultiopenTransform {
    Bls12_381Fr multiplier;
    Bls12_381Fr addition;
};

inline MidnightMultiopenTransform midnight_multiopen_compose(
    MidnightMultiopenTransform left,
    MidnightMultiopenTransform right)
{
    MidnightMultiopenTransform result;
    result.multiplier = fr_montgomery_mul(right.multiplier, left.multiplier);
    result.addition = fr_add(
        fr_montgomery_mul(right.multiplier, left.addition),
        right.addition);
    return result;
}

kernel void midnight_bls12_381_multiopen_division_summarize(
    device const Bls12_381Fr* source [[buffer(0)]],
    device Bls12_381Fr* destination [[buffer(1)]],
    device const uint* point_offsets [[buffer(2)]],
    device const Bls12_381Fr* points [[buffer(3)]],
    device Bls12_381Fr* transforms [[buffer(4)]],
    device Bls12_381Fr* local_multipliers [[buffer(5)]],
    constant MidnightMultiopenDivisionParameters& parameters [[buffer(6)]],
    threadgroup Bls12_381Fr* scratch [[threadgroup(0)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint block_index = group_position.x;
    const uint poly_index = group_position.y;
    if (block_index >= parameters.block_count
        || poly_index >= parameters.poly_count
        || lane >= parameters.block_size
        || parameters.poly_len == 0U) {
        return;
    }
    const ulong flat_block
        = static_cast<ulong>(poly_index) * parameters.block_count + block_index;
    const ulong poly_offset
        = static_cast<ulong>(poly_index) * parameters.poly_len;
    const uint linear_index = block_index * parameters.block_size + lane;
    Bls12_381Fr point;
    const bool active = midnight_multiopen_division_point(
        poly_index, point_offsets, points, parameters, point);
    if (!active) {
        if (linear_index < parameters.poly_len) {
            destination[poly_offset + linear_index]
                = source[poly_offset + linear_index];
        }
        if (lane == 0U) {
            const ulong transform_offset = 2UL * flat_block;
            transforms[transform_offset] = midnight_graph_one();
            transforms[transform_offset + 1UL] = midnight_graph_zero();
        }
        return;
    }

    const uint output_len = parameters.poly_len - parameters.round - 1U;
    const uint start = min(block_index * parameters.block_size, output_len);
    const uint end = min(start + parameters.block_size, output_len);
    const uint count = end - start;
    const bool valid = lane < count;
    const uint source_index = valid ? end - lane : 0U;
    threadgroup Bls12_381Fr* shared_multipliers = scratch;
    threadgroup Bls12_381Fr* shared_additions = scratch + parameters.block_size;
    shared_multipliers[lane] = valid ? point : midnight_graph_one();
    shared_additions[lane]
        = valid ? source[poly_offset + source_index] : midnight_graph_zero();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Upsweep computes one affine summary for the complete block. The combine
    // order follows high-to-low synthetic division and is intentionally
    // noncommutative.
    for (uint stride = 1U; stride < parameters.block_size; stride <<= 1U) {
        const uint worker_count = parameters.block_size / (2U * stride);
        if (lane < worker_count) {
            const uint left_index = (2U * lane + 1U) * stride - 1U;
            const uint right_index = left_index + stride;
            const MidnightMultiopenTransform left = {
                shared_multipliers[left_index], shared_additions[left_index]};
            const MidnightMultiopenTransform right = {
                shared_multipliers[right_index], shared_additions[right_index]};
            const MidnightMultiopenTransform combined
                = midnight_multiopen_compose(left, right);
            shared_multipliers[right_index] = combined.multiplier;
            shared_additions[right_index] = combined.addition;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0U) {
        const uint root = parameters.block_size - 1U;
        const ulong transform_offset = 2UL * flat_block;
        transforms[transform_offset] = shared_multipliers[root];
        transforms[transform_offset + 1UL] = shared_additions[root];
        shared_multipliers[root] = midnight_graph_one();
        shared_additions[root] = midnight_graph_zero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Downsweep leaves the exclusive high-to-low transform at every lane.
    for (uint stride = parameters.block_size >> 1U; stride != 0U; stride >>= 1U) {
        const uint worker_count = parameters.block_size / (2U * stride);
        if (lane < worker_count) {
            const uint left_index = (2U * lane + 1U) * stride - 1U;
            const uint right_index = left_index + stride;
            const MidnightMultiopenTransform left_total = {
                shared_multipliers[left_index], shared_additions[left_index]};
            const MidnightMultiopenTransform prefix = {
                shared_multipliers[right_index], shared_additions[right_index]};
            const MidnightMultiopenTransform right_prefix
                = midnight_multiopen_compose(prefix, left_total);
            shared_multipliers[left_index] = prefix.multiplier;
            shared_additions[left_index] = prefix.addition;
            shared_multipliers[right_index] = right_prefix.multiplier;
            shared_additions[right_index] = right_prefix.addition;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        const MidnightMultiopenTransform prefix = {
            shared_multipliers[lane], shared_additions[lane]};
        const MidnightMultiopenTransform coefficient = {
            point, source[poly_offset + source_index]};
        const MidnightMultiopenTransform inclusive
            = midnight_multiopen_compose(prefix, coefficient);
        destination[poly_offset + source_index - 1U] = inclusive.addition;
        if (block_index == 0U) {
            local_multipliers[
                static_cast<ulong>(poly_index) * parameters.block_size + lane]
                = inclusive.multiplier;
        }
    }
    if (linear_index >= output_len && linear_index < parameters.poly_len) {
        destination[poly_offset + linear_index] = midnight_graph_zero();
    }
}

kernel void midnight_bls12_381_multiopen_division_scan_blocks(
    device const Bls12_381Fr* transforms [[buffer(0)]],
    device Bls12_381Fr* block_inputs [[buffer(1)]],
    constant MidnightMultiopenDivisionParameters& parameters [[buffer(2)]],
    uint poly_index [[thread_position_in_grid]])
{
    if (poly_index >= parameters.poly_count) {
        return;
    }
    Bls12_381Fr state = midnight_graph_zero();
    const ulong first_block
        = static_cast<ulong>(poly_index) * parameters.block_count;
    for (uint block = parameters.block_count; block != 0U; --block) {
        const ulong flat_block = first_block + block - 1U;
        block_inputs[flat_block] = state;
        const ulong transform_offset = 2UL * flat_block;
        state = fr_add(
            fr_montgomery_mul(transforms[transform_offset], state),
            transforms[transform_offset + 1UL]);
    }
}

kernel void midnight_bls12_381_multiopen_division_fill(
    device Bls12_381Fr* destination [[buffer(0)]],
    device const uint* point_offsets [[buffer(1)]],
    device const Bls12_381Fr* points [[buffer(2)]],
    device const Bls12_381Fr* block_inputs [[buffer(3)]],
    device const Bls12_381Fr* local_multipliers [[buffer(4)]],
    constant MidnightMultiopenDivisionParameters& parameters [[buffer(5)]],
    uint flat_index [[thread_position_in_grid]])
{
    const ulong total
        = static_cast<ulong>(parameters.poly_count) * parameters.poly_len;
    if (flat_index >= total || parameters.poly_len == 0U) {
        return;
    }
    const uint poly_index = flat_index / parameters.poly_len;
    const uint coefficient_index = flat_index % parameters.poly_len;
    const ulong poly_offset
        = static_cast<ulong>(poly_index) * parameters.poly_len;
    Bls12_381Fr point;
    const bool active = midnight_multiopen_division_point(
        poly_index, point_offsets, points, parameters, point);
    if (!active) {
        return;
    }

    const uint output_len = parameters.poly_len - parameters.round - 1U;
    if (coefficient_index >= output_len) {
        return;
    }
    const uint block_index = coefficient_index / parameters.block_size;
    const uint end = min(
        (block_index + 1U) * parameters.block_size,
        output_len);
    const uint lane_from_high = end - coefficient_index - 1U;
    const ulong flat_block
        = static_cast<ulong>(poly_index) * parameters.block_count + block_index;
    destination[poly_offset + coefficient_index] = fr_add(
        destination[poly_offset + coefficient_index],
        fr_montgomery_mul(
            local_multipliers[
                static_cast<ulong>(poly_index) * parameters.block_size
                + lane_from_high],
            block_inputs[flat_block]));
}

kernel void midnight_bls12_381_multiopen_combine(
    device const Bls12_381Fr* polynomials [[buffer(0)]],
    device const Bls12_381Fr* weights [[buffer(1)]],
    device Bls12_381Fr* output [[buffer(2)]],
    constant MidnightMultiopenCombineParameters& parameters [[buffer(3)]],
    uint coefficient_index [[thread_position_in_grid]])
{
    if (coefficient_index >= parameters.poly_len) {
        return;
    }
    Bls12_381Fr value = midnight_graph_zero();
    for (uint poly_index = 0; poly_index < parameters.poly_count; ++poly_index) {
        value = fr_add(
            value,
            fr_montgomery_mul(
                polynomials[
                    static_cast<ulong>(poly_index) * parameters.poly_len
                    + coefficient_index],
                weights[poly_index]));
    }
    output[coefficient_index] = value;
}

kernel void midnight_bls12_381_multiopen_combine_final(
    device const Bls12_381Fr* q_polynomials [[buffer(0)]],
    device const Bls12_381Fr* f_polynomial [[buffer(1)]],
    device const Bls12_381Fr* weights [[buffer(2)]],
    device Bls12_381Fr* output [[buffer(3)]],
    constant MidnightMultiopenCombineParameters& parameters [[buffer(4)]],
    uint coefficient_index [[thread_position_in_grid]])
{
    if (coefficient_index >= parameters.poly_len) {
        return;
    }
    Bls12_381Fr value = fr_montgomery_mul(
        f_polynomial[coefficient_index], weights[parameters.poly_count]);
    for (uint poly_index = 0; poly_index < parameters.poly_count; ++poly_index) {
        value = fr_add(
            value,
            fr_montgomery_mul(
                q_polynomials[
                    static_cast<ulong>(poly_index) * parameters.poly_len
                    + coefficient_index],
                weights[poly_index]));
    }
    output[coefficient_index] = value;
}

kernel void midnight_bls12_381_multiopen_eval_summarize(
    device const Bls12_381Fr* polynomials [[buffer(0)]],
    device Bls12_381Fr* transforms [[buffer(1)]],
    constant MidnightMultiopenEvalParameters& parameters [[buffer(2)]],
    threadgroup Bls12_381Fr* scratch [[threadgroup(0)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint block_index = group_position.x;
    const uint poly_index = group_position.y;
    if (block_index >= parameters.block_count
        || poly_index >= parameters.poly_count
        || lane >= parameters.block_size) {
        return;
    }
    const ulong flat_block
        = static_cast<ulong>(poly_index) * parameters.block_count + block_index;
    const uint start = min(block_index * parameters.block_size, parameters.poly_len);
    const uint end = min(start + parameters.block_size, parameters.poly_len);
    const uint count = end - start;
    const ulong poly_offset
        = static_cast<ulong>(poly_index) * parameters.poly_len;
    const bool valid = lane < count;
    threadgroup Bls12_381Fr* shared_multipliers = scratch;
    threadgroup Bls12_381Fr* shared_additions = scratch + parameters.block_size;
    shared_multipliers[lane] = valid ? parameters.point : midnight_graph_one();
    shared_additions[lane] = valid
        ? polynomials[poly_offset + end - lane - 1U]
        : midnight_graph_zero();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1U; stride < parameters.block_size; stride <<= 1U) {
        const uint worker_count = parameters.block_size / (2U * stride);
        if (lane < worker_count) {
            const uint left_index = (2U * lane + 1U) * stride - 1U;
            const uint right_index = left_index + stride;
            const MidnightMultiopenTransform left = {
                shared_multipliers[left_index], shared_additions[left_index]};
            const MidnightMultiopenTransform right = {
                shared_multipliers[right_index], shared_additions[right_index]};
            const MidnightMultiopenTransform combined
                = midnight_multiopen_compose(left, right);
            shared_multipliers[right_index] = combined.multiplier;
            shared_additions[right_index] = combined.addition;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0U) {
        const uint root = parameters.block_size - 1U;
        const ulong transform_offset = 2UL * flat_block;
        transforms[transform_offset] = shared_multipliers[root];
        transforms[transform_offset + 1UL] = shared_additions[root];
    }
}

kernel void midnight_bls12_381_multiopen_eval_finish(
    device const Bls12_381Fr* transforms [[buffer(0)]],
    device Bls12_381Fr* evaluations [[buffer(1)]],
    constant MidnightMultiopenEvalParameters& parameters [[buffer(2)]],
    uint poly_index [[thread_position_in_grid]])
{
    if (poly_index >= parameters.poly_count) {
        return;
    }
    Bls12_381Fr value = midnight_graph_zero();
    const ulong first_block
        = static_cast<ulong>(poly_index) * parameters.block_count;
    for (uint block = parameters.block_count; block != 0U; --block) {
        const ulong transform_offset = 2UL * (first_block + block - 1U);
        value = fr_add(
            fr_montgomery_mul(transforms[transform_offset], value),
            transforms[transform_offset + 1UL]);
    }
    evaluations[poly_index] = value;
}

kernel void midnight_bls12_381_multiopen_subtract_evaluation(
    device Bls12_381Fr* polynomial [[buffer(0)]],
    device const Bls12_381Fr* evaluation [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0U) {
        polynomial[0] = fr_sub(polynomial[0], evaluation[0]);
    }
}

#endif
