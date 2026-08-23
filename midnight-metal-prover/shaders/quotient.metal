// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef MIDNIGHT_BLS12_381_QUOTIENT_METAL
#define MIDNIGHT_BLS12_381_QUOTIENT_METAL

#include <metal_stdlib>

using namespace metal;

struct MidnightQuotientColumnRef {
    uint kind;
    uint index;
};

static_assert(sizeof(MidnightQuotientColumnRef) == 8);
static_assert(__builtin_offsetof(MidnightQuotientColumnRef, kind) == 0);
static_assert(__builtin_offsetof(MidnightQuotientColumnRef, index) == 4);

struct MidnightQuotientCopyParameters {
    uint len;
    uint source_stride;
    uint destination_stride;
    uint reserved;
};

struct MidnightQuotientPermutationParameters {
    uint len;
    uint stride;
    int rotation_scale;
    int last_rotation;
    uint fixed_columns;
    uint advice_offset;
    uint advice_columns;
    uint instance_offset;
    uint instance_columns;
    uint permutation_offset;
    uint permutation_sets;
    uint permutation_column_count;
    uint chunk_len;
    uint reserved;
    Bls12_381Fr beta;
    Bls12_381Fr gamma;
    Bls12_381Fr y;
    Bls12_381Fr root;
    Bls12_381Fr coset_generator;
    Bls12_381Fr delta;
};

struct MidnightQuotientMultiplyParameters {
    uint len;
    uint period;
};

struct MidnightQuotientSplitParameters {
    uint piece_len;
    uint piece_count;
    uint source_len;
    uint reserved;
};

struct MidnightQuotientCombineParameters {
    uint piece_len;
    uint piece_count;
    uint reserved_0;
    uint reserved_1;
    Bls12_381Fr splitting_factor;
};

static_assert(sizeof(MidnightQuotientCopyParameters) == 16);
static_assert(__builtin_offsetof(MidnightQuotientCopyParameters, len) == 0);
static_assert(
    __builtin_offsetof(MidnightQuotientCopyParameters, source_stride) == 4);
static_assert(
    __builtin_offsetof(MidnightQuotientCopyParameters, destination_stride) == 8);
static_assert(__builtin_offsetof(MidnightQuotientCopyParameters, reserved) == 12);
static_assert(sizeof(MidnightQuotientPermutationParameters) == 248);
static_assert(__builtin_offsetof(MidnightQuotientPermutationParameters, len) == 0);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, reserved) == 52);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, beta) == 56);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, gamma) == 88);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, y) == 120);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, root) == 152);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, coset_generator) == 184);
static_assert(
    __builtin_offsetof(MidnightQuotientPermutationParameters, delta) == 216);
static_assert(sizeof(MidnightQuotientMultiplyParameters) == 8);
static_assert(__builtin_offsetof(MidnightQuotientMultiplyParameters, len) == 0);
static_assert(__builtin_offsetof(MidnightQuotientMultiplyParameters, period) == 4);
static_assert(sizeof(MidnightQuotientSplitParameters) == 16);
static_assert(__builtin_offsetof(MidnightQuotientSplitParameters, piece_len) == 0);
static_assert(__builtin_offsetof(MidnightQuotientSplitParameters, piece_count) == 4);
static_assert(__builtin_offsetof(MidnightQuotientSplitParameters, source_len) == 8);
static_assert(__builtin_offsetof(MidnightQuotientSplitParameters, reserved) == 12);
static_assert(sizeof(MidnightQuotientCombineParameters) == 48);
static_assert(__builtin_offsetof(MidnightQuotientCombineParameters, piece_len) == 0);
static_assert(__builtin_offsetof(MidnightQuotientCombineParameters, piece_count) == 4);
static_assert(
    __builtin_offsetof(MidnightQuotientCombineParameters, splitting_factor) == 16);

inline Bls12_381Fr midnight_quotient_column_value(
    MidnightQuotientColumnRef reference,
    device const Bls12_381Fr* fixed_values,
    device const Bls12_381Fr* dynamic_values,
    constant MidnightQuotientPermutationParameters& parameters,
    uint row)
{
    if (reference.kind == 0U) {
        return fixed_values[
            static_cast<ulong>(reference.index) * parameters.stride + row];
    }
    if (reference.kind == 1U) {
        return dynamic_values[
            static_cast<ulong>(parameters.advice_offset + reference.index)
                * parameters.stride + row];
    }
    if (reference.kind == 2U) {
        return dynamic_values[
            static_cast<ulong>(parameters.instance_offset + reference.index)
                * parameters.stride + row];
    }
    return midnight_graph_zero();
}

kernel void midnight_bls12_381_quotient_copy(
    device const Bls12_381Fr* source [[buffer(0)]],
    device Bls12_381Fr* destination [[buffer(1)]],
    constant MidnightQuotientCopyParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < parameters.len) {
        destination[index * parameters.destination_stride]
            = source[index * parameters.source_stride];
    }
}

kernel void midnight_bls12_381_quotient_permutation_accumulate(
    device Bls12_381Fr* numerator [[buffer(0)]],
    device const Bls12_381Fr* fixed_values [[buffer(1)]],
    device const Bls12_381Fr* dynamic_values [[buffer(2)]],
    device const Bls12_381Fr* selectors [[buffer(3)]],
    device const MidnightQuotientColumnRef* columns [[buffer(4)]],
    device const Bls12_381Fr* permutation_cosets [[buffer(5)]],
    device const Bls12_381Fr* extended_coset_points [[buffer(6)]],
    constant MidnightQuotientPermutationParameters& parameters [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= parameters.len || parameters.permutation_sets == 0
        || parameters.chunk_len == 0) {
        return;
    }

    const uint next_row = midnight_graph_rotated_row(
        row, 1, parameters.rotation_scale, parameters.len);
    const uint final_row = midnight_graph_rotated_row(
        row, parameters.last_rotation, parameters.rotation_scale, parameters.len);
    const Bls12_381Fr one = midnight_graph_one();
    const Bls12_381Fr l0 = selectors[row];
    const Bls12_381Fr l_last = selectors[
        static_cast<ulong>(parameters.stride) + row];
    const Bls12_381Fr l_active = selectors[
        2UL * static_cast<ulong>(parameters.stride) + row];
    const ulong product_stride = parameters.stride;
    const ulong first_product_offset
        = static_cast<ulong>(parameters.permutation_offset) * product_stride;
    const ulong last_product_offset = static_cast<ulong>(
        parameters.permutation_offset + parameters.permutation_sets - 1U)
        * product_stride;

    Bls12_381Fr value = numerator[row];
    const Bls12_381Fr first_product = dynamic_values[first_product_offset + row];
    const Bls12_381Fr last_product = dynamic_values[last_product_offset + row];
    value = midnight_graph_accumulate(
        value, parameters.y, fr_montgomery_mul(fr_sub(one, first_product), l0));
    value = midnight_graph_accumulate(
        value,
        parameters.y,
        fr_montgomery_mul(
            fr_sub(fr_montgomery_mul(last_product, last_product), last_product),
            l_last));

    for (uint set_index = 1; set_index < parameters.permutation_sets; ++set_index) {
        const ulong current_offset = static_cast<ulong>(
            parameters.permutation_offset + set_index) * product_stride;
        const ulong previous_offset = current_offset - product_stride;
        const Bls12_381Fr term = fr_montgomery_mul(
            fr_sub(
                dynamic_values[current_offset + row],
                dynamic_values[previous_offset + final_row]),
            l0);
        value = midnight_graph_accumulate(value, parameters.y, term);
    }

    const Bls12_381Fr point = extended_coset_points[row];
    Bls12_381Fr current_delta = fr_montgomery_mul(parameters.beta, point);
    for (uint set_index = 0; set_index < parameters.permutation_sets; ++set_index) {
        const ulong product_offset = static_cast<ulong>(
            parameters.permutation_offset + set_index) * product_stride;
        Bls12_381Fr left = dynamic_values[product_offset + next_row];
        Bls12_381Fr right = dynamic_values[product_offset + row];
        const uint column_start = set_index * parameters.chunk_len;
        const uint column_end = min(
            column_start + parameters.chunk_len,
            parameters.permutation_column_count);
        for (uint column_index = column_start;
             column_index < column_end;
             ++column_index) {
            const Bls12_381Fr column = midnight_quotient_column_value(
                columns[column_index], fixed_values, dynamic_values, parameters, row);
            const Bls12_381Fr sigma = permutation_cosets[
                static_cast<ulong>(column_index) * parameters.stride + row];
            left = fr_montgomery_mul(
                left,
                fr_add(fr_add(column, fr_montgomery_mul(parameters.beta, sigma)),
                       parameters.gamma));
            right = fr_montgomery_mul(
                right,
                fr_add(fr_add(column, current_delta), parameters.gamma));
            current_delta = fr_montgomery_mul(current_delta, parameters.delta);
        }
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(fr_sub(left, right), l_active));
    }
    numerator[row] = value;
}

kernel void midnight_bls12_381_quotient_multiply_repeating(
    device Bls12_381Fr* values [[buffer(0)]],
    device const Bls12_381Fr* factors [[buffer(1)]],
    constant MidnightQuotientMultiplyParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < parameters.len) {
        values[index] = fr_montgomery_mul(values[index], factors[index % parameters.period]);
    }
}

kernel void midnight_bls12_381_quotient_split_blind(
    device const Bls12_381Fr* quotient [[buffer(0)]],
    device const Bls12_381Fr* blinds [[buffer(1)]],
    device Bls12_381Fr* pieces [[buffer(2)]],
    constant MidnightQuotientSplitParameters& parameters [[buffer(3)]],
    uint flat_index [[thread_position_in_grid]])
{
    const uint total = parameters.piece_len * parameters.piece_count;
    if (flat_index >= total) {
        return;
    }
    const uint piece = flat_index / parameters.piece_len;
    const uint index = flat_index % parameters.piece_len;
    if (index + 1U == parameters.piece_len) {
        pieces[flat_index] = piece + 1U < parameters.piece_count
            ? blinds[piece]
            : midnight_graph_zero();
        return;
    }
    const uint source_index = piece * (parameters.piece_len - 1U) + index;
    Bls12_381Fr value = source_index < parameters.source_len
        ? quotient[source_index]
        : midnight_graph_zero();
    if (index == 0U && piece != 0U) {
        value = fr_sub(value, blinds[piece - 1U]);
    }
    pieces[flat_index] = value;
}

kernel void midnight_bls12_381_quotient_combine_pieces(
    device const Bls12_381Fr* pieces [[buffer(0)]],
    device Bls12_381Fr* combined [[buffer(1)]],
    constant MidnightQuotientCombineParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.piece_len || parameters.piece_count == 0U) {
        return;
    }
    Bls12_381Fr value = pieces[
        static_cast<ulong>(parameters.piece_count - 1U) * parameters.piece_len + index];
    for (uint piece = parameters.piece_count - 1U; piece != 0U; --piece) {
        value = fr_add(
            fr_montgomery_mul(value, parameters.splitting_factor),
            pieces[static_cast<ulong>(piece - 1U) * parameters.piece_len + index]);
    }
    combined[index] = value;
}

#endif
