// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_FLAT_SEGMENTED_METAL
#define MIDNIGHT_BLS12_381_FLAT_SEGMENTED_METAL

#include "batch_affine.metal"

struct MidnightFlatSegmentedTile {
    uint input_offset;
    uint input_len;
    uint output_offset;
    uint reserved;
};

struct MidnightFlatSegmentedParameters {
    uint tile_count;
    uint input_count;
    uint output_count;
    uint srs_count;
    uint group_count;
    uint input_kind;
    uint group_width;
    uint reserved;
};

constant ulong MIDNIGHT_FLAT_SEGMENTED_ABI_VERSION = 1UL;
constant uint MIDNIGHT_FLAT_SEGMENTED_INPUT_RECORDS = 0U;
constant uint MIDNIGHT_FLAT_SEGMENTED_INPUT_POINTS = 1U;

inline MidnightBatchAffinePoint32 flat_segmented_load_input(
    device const Bls12_381G1Affine* srs,
    device const uint* packed_records,
    device const MidnightBatchAffinePoint32* input_points,
    uint input_kind,
    uint input_index)
{
    if (input_kind == MIDNIGHT_FLAT_SEGMENTED_INPUT_RECORDS) {
        const uint packed = packed_records[input_index];
        const uint source_index = packed & 0x7fffffffU;
        const uint negate_mask = 0U - (packed >> 31);
        MidnightBatchAffinePoint32 point;
        point.x = fp32_from_public(srs[source_index].x);
        point.y = fp32_from_public(srs[source_index].y);
        const Bls12_381Fp32 negated_y =
            fp32_sub(batch_affine_fp32_zero(), point.y);
        point.y = batch_affine_fp32_select(
            point.y, negated_y, negate_mask);
        return point;
    }
    return input_points[input_index];
}

kernel void midnight_bls12_381_flat_segmented_abi_probe(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }
    output[0] = MIDNIGHT_FLAT_SEGMENTED_ABI_VERSION;
    output[1] = sizeof(MidnightFlatSegmentedTile);
    output[2] = alignof(MidnightFlatSegmentedTile);
    output[3] = __builtin_offsetof(
        MidnightFlatSegmentedTile, input_offset);
    output[4] = __builtin_offsetof(MidnightFlatSegmentedTile, input_len);
    output[5] = __builtin_offsetof(
        MidnightFlatSegmentedTile, output_offset);
    output[6] = __builtin_offsetof(MidnightFlatSegmentedTile, reserved);
    output[7] = sizeof(MidnightFlatSegmentedParameters);
    output[8] = alignof(MidnightFlatSegmentedParameters);
    output[9] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, tile_count);
    output[10] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, input_count);
    output[11] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, output_count);
    output[12] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, srs_count);
    output[13] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, group_count);
    output[14] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, input_kind);
    output[15] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, group_width);
    output[16] = __builtin_offsetof(
        MidnightFlatSegmentedParameters, reserved);
}

// Each real lane reduces one same-run tile of at most 32 points. Descriptors
// are sorted only for scheduling; their explicit offsets preserve semantic run
// ownership and dense output order. Padded lanes use input_len=0 and contribute
// Montgomery one at every group-uniform round.
kernel void midnight_bls12_381_flat_segmented_reduce(
    device const Bls12_381G1Affine* srs [[buffer(0)]],
    device const uint* packed_records [[buffer(1)]],
    device const MidnightFlatSegmentedTile* tiles [[buffer(2)]],
    device const uint* group_round_counts [[buffer(3)]],
    device const MidnightBatchAffinePoint32* input_points [[buffer(4)]],
    device MidnightBatchAffinePoint32* output_points [[buffer(5)]],
    device MidnightBatchAffineStatus* status [[buffer(6)]],
    constant MidnightFlatSegmentedParameters& parameters [[buffer(7)]],
    threadgroup Bls12_381Fp32* product_tree [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    const uint group_index = group_position.x;
    const uint width = parameters.group_width;
    const bool supported_width = width == 128U || width == 256U;
    const bool valid_parameters = supported_width &&
        parameters.tile_count % width == 0U &&
        parameters.tile_count / width == parameters.group_count &&
        group_index < parameters.group_count &&
        (parameters.input_kind == MIDNIGHT_FLAT_SEGMENTED_INPUT_RECORDS ||
         parameters.input_kind == MIDNIGHT_FLAT_SEGMENTED_INPUT_POINTS);
    if (!valid_parameters) {
        if (lane == 0) {
            batch_affine_status_or(
                status, MIDNIGHT_BATCH_AFFINE_STATUS_INVALID_PARAMETERS);
        }
        return;
    }

    const MidnightFlatSegmentedTile tile =
        tiles[group_index * width + lane];
    const bool active = tile.input_len != 0;
    MidnightBatchAffinePoint32 accumulator = batch_affine_point_identity();
    if (active) {
        accumulator = flat_segmented_load_input(
            srs,
            packed_records,
            input_points,
            parameters.input_kind,
            tile.input_offset);
    }

    const uint round_count = group_round_counts[group_index];
    const uint leaf_index = width - 1U + lane;
    for (uint round = 0; round < round_count; ++round) {
        MidnightBatchAffinePoint32 rhs = batch_affine_point_identity();
        if (active && round + 1U < tile.input_len) {
            rhs = flat_segmented_load_input(
                srs,
                packed_records,
                input_points,
                parameters.input_kind,
                tile.input_offset + round + 1U);
        }
        const MidnightBatchAffinePreparedPair prepared =
            batch_affine_prepare_point_pair(accumulator, rhs);
        product_tree[leaf_index] = prepared.denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint node_count = width >> 1;
             node_count != 0;
             node_count >>= 1) {
            if (lane < node_count) {
                const uint parent = node_count - 1U + lane;
                product_tree[parent] = fp32_montgomery_mul_wide(
                    product_tree[2U * parent + 1U],
                    product_tree[2U * parent + 2U]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            const Bls12_381Fp32 product = product_tree[0];
            const Bls12_381Fp32 inverse =
                batch_affine_fp32_inverse_fixed(product);
            product_tree[0] = inverse;
            const uint product_zero =
                batch_affine_fp32_zero_mask(product) & 1U;
            const uint inverse_zero =
                batch_affine_fp32_zero_mask(inverse) & 1U;
            if ((product_zero | inverse_zero) != 0) {
                const uint flags =
                    product_zero *
                        MIDNIGHT_BATCH_AFFINE_STATUS_PRODUCT_ZERO |
                    inverse_zero *
                        MIDNIGHT_BATCH_AFFINE_STATUS_INVERSE_ZERO;
                batch_affine_status_or(status, flags);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint node_count = 1U;
             node_count < width;
             node_count <<= 1) {
            if (lane < node_count) {
                const uint parent = node_count - 1U + lane;
                const uint left = 2U * parent + 1U;
                const uint right = left + 1U;
                const Bls12_381Fp32 parent_inverse = product_tree[parent];
                const Bls12_381Fp32 left_product = product_tree[left];
                const Bls12_381Fp32 right_product = product_tree[right];
                product_tree[left] = fp32_montgomery_mul_wide(
                    parent_inverse, right_product);
                product_tree[right] = fp32_montgomery_mul_wide(
                    parent_inverse, left_product);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        accumulator = batch_affine_finish_point_pair(
            accumulator,
            rhs,
            prepared.numerator,
            product_tree[leaf_index],
            prepared.formula_valid,
            prepared.lhs_infinity,
            prepared.rhs_infinity);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        output_points[tile.output_offset] = accumulator;
    }
}

#endif
