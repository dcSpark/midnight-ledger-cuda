// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef MIDNIGHT_BLS12_381_EXPRESSION_METAL
#define MIDNIGHT_BLS12_381_EXPRESSION_METAL

#include "field.metal"

struct MidnightGraphValueRef {
    uint kind;
    uint a;
    uint b;
};

struct MidnightGraphOperation {
    uint opcode;
    uint target;
    uint parts_offset;
    uint parts_len;
    MidnightGraphValueRef lhs;
    MidnightGraphValueRef rhs;
};

// This block is deliberately 256 bytes. Its fixed-width layout is shared with
// the checked host compiler and versioned independently from proving keys.
struct MidnightGraphEvalParameters {
    uint len;
    uint stride;
    int rotation_scale;
    uint post_kind;

    uint constants_offset;
    uint constants_count;
    uint rotations_offset;
    uint rotations_count;
    uint operations_offset;
    uint operations_count;
    uint parts_offset;
    uint parts_count;

    uint fixed_columns;
    uint advice_offset;
    uint advice_columns;
    uint instance_offset;
    uint instance_columns;
    uint challenge_count;
    uint previous_mode;
    uint lookup_product_column;
    uint lookup_input_column;
    uint lookup_table_column;
    uint trash_column;
    uint trash_selector_fixed_column;

    Bls12_381Fr beta;
    Bls12_381Fr gamma;
    Bls12_381Fr theta;
    Bls12_381Fr trash_challenge;
    Bls12_381Fr y;
};

constant uint MIDNIGHT_GRAPH_VALUE_CONSTANT = 0;
constant uint MIDNIGHT_GRAPH_VALUE_INTERMEDIATE = 1;
constant uint MIDNIGHT_GRAPH_VALUE_FIXED = 2;
constant uint MIDNIGHT_GRAPH_VALUE_ADVICE = 3;
constant uint MIDNIGHT_GRAPH_VALUE_INSTANCE = 4;
constant uint MIDNIGHT_GRAPH_VALUE_CHALLENGE = 5;
constant uint MIDNIGHT_GRAPH_VALUE_BETA = 6;
constant uint MIDNIGHT_GRAPH_VALUE_GAMMA = 7;
constant uint MIDNIGHT_GRAPH_VALUE_THETA = 8;
constant uint MIDNIGHT_GRAPH_VALUE_TRASH_CHALLENGE = 9;
constant uint MIDNIGHT_GRAPH_VALUE_Y = 10;
constant uint MIDNIGHT_GRAPH_VALUE_PREVIOUS = 11;

constant uint MIDNIGHT_GRAPH_OP_ADD = 0;
constant uint MIDNIGHT_GRAPH_OP_SUB = 1;
constant uint MIDNIGHT_GRAPH_OP_MUL = 2;
constant uint MIDNIGHT_GRAPH_OP_SQUARE = 3;
constant uint MIDNIGHT_GRAPH_OP_DOUBLE = 4;
constant uint MIDNIGHT_GRAPH_OP_NEGATE = 5;
constant uint MIDNIGHT_GRAPH_OP_HORNER = 6;
constant uint MIDNIGHT_GRAPH_OP_STORE = 7;

constant uint MIDNIGHT_GRAPH_POST_STORE = 0;
constant uint MIDNIGHT_GRAPH_POST_LOOKUP = 1;
constant uint MIDNIGHT_GRAPH_POST_TRASH = 2;
constant uint MIDNIGHT_GRAPH_SCRATCH_8 = 8;
constant uint MIDNIGHT_GRAPH_SCRATCH_16 = 16;
constant uint MIDNIGHT_GRAPH_SCRATCH_32 = 32;
constant uint MIDNIGHT_GRAPH_SCRATCH_64 = 64;
constant uint MIDNIGHT_GRAPH_SCRATCH_128 = 128;
constant uint MIDNIGHT_GRAPH_SCRATCH_256 = 256;
constant uint MIDNIGHT_GRAPH_ABI_VERSION = 1;

inline ulong midnight_graph_abi_mix(ulong fingerprint, ulong value)
{
    return (fingerprint ^ value) * 0x00000100000001b3UL;
}

inline Bls12_381Fr midnight_graph_zero()
{
    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = 0;
    }
    return result;
}

inline Bls12_381Fr midnight_graph_one()
{
    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = MIDNIGHT_BLS12_381_FR_R[index];
    }
    return result;
}

inline uint midnight_graph_rotated_row(
    uint row,
    int rotation,
    int rotation_scale,
    uint len)
{
    const long modulus = static_cast<long>(len);
    const long position = static_cast<long>(row)
        + static_cast<long>(rotation) * static_cast<long>(rotation_scale);
    const long remainder = position % modulus;
    return static_cast<uint>(remainder < 0 ? remainder + modulus : remainder);
}

inline Bls12_381Fr midnight_graph_dynamic_value(
    device const Bls12_381Fr* values,
    ulong column,
    uint row,
    uint stride)
{
    return values[column * static_cast<ulong>(stride) + row];
}

inline Bls12_381Fr midnight_graph_load_value(
    MidnightGraphValueRef source,
    device const Bls12_381Fr* constants,
    device const int* rotations,
    device const Bls12_381Fr* fixed_values,
    device const Bls12_381Fr* dynamic_values,
    device const Bls12_381Fr* challenges,
    thread Bls12_381Fr* intermediates,
    Bls12_381Fr previous,
    uint row,
    constant MidnightGraphEvalParameters& parameters)
{
    if (source.kind == MIDNIGHT_GRAPH_VALUE_CONSTANT) {
        return constants[
            static_cast<ulong>(parameters.constants_offset) + source.a];
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_INTERMEDIATE) {
        return intermediates[source.a];
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_FIXED) {
        const int rotation = rotations[
            static_cast<ulong>(parameters.rotations_offset) + source.b];
        const uint rotated_row = midnight_graph_rotated_row(
            row, rotation, parameters.rotation_scale, parameters.len);
        return midnight_graph_dynamic_value(
            fixed_values, source.a, rotated_row, parameters.stride);
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_ADVICE) {
        const int rotation = rotations[
            static_cast<ulong>(parameters.rotations_offset) + source.b];
        const uint rotated_row = midnight_graph_rotated_row(
            row, rotation, parameters.rotation_scale, parameters.len);
        return midnight_graph_dynamic_value(
            dynamic_values,
            static_cast<ulong>(parameters.advice_offset) + source.a,
            rotated_row,
            parameters.stride);
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_INSTANCE) {
        const int rotation = rotations[
            static_cast<ulong>(parameters.rotations_offset) + source.b];
        const uint rotated_row = midnight_graph_rotated_row(
            row, rotation, parameters.rotation_scale, parameters.len);
        return midnight_graph_dynamic_value(
            dynamic_values,
            static_cast<ulong>(parameters.instance_offset) + source.a,
            rotated_row,
            parameters.stride);
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_CHALLENGE) {
        return challenges[source.a];
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_BETA) {
        return parameters.beta;
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_GAMMA) {
        return parameters.gamma;
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_THETA) {
        return parameters.theta;
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_TRASH_CHALLENGE) {
        return parameters.trash_challenge;
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_Y) {
        return parameters.y;
    }
    if (source.kind == MIDNIGHT_GRAPH_VALUE_PREVIOUS) {
        return previous;
    }
    return midnight_graph_zero();
}

template <uint ScratchCapacity>
inline Bls12_381Fr midnight_graph_evaluate_operations(
    device const Bls12_381Fr* constants,
    device const int* rotations,
    device const MidnightGraphOperation* operations,
    device const MidnightGraphValueRef* horner_parts,
    device const Bls12_381Fr* fixed_values,
    device const Bls12_381Fr* dynamic_values,
    device const Bls12_381Fr* challenges,
    Bls12_381Fr previous,
    uint row,
    constant MidnightGraphEvalParameters& parameters)
{
    Bls12_381Fr intermediates[ScratchCapacity];
    Bls12_381Fr result = midnight_graph_zero();

    for (uint index = 0; index < parameters.operations_count; ++index) {
        const MidnightGraphOperation operation =
            operations[static_cast<ulong>(parameters.operations_offset) + index];
        const Bls12_381Fr lhs = midnight_graph_load_value(
            operation.lhs,
            constants,
            rotations,
            fixed_values,
            dynamic_values,
            challenges,
            intermediates,
            previous,
            row,
            parameters);

        if (operation.opcode == MIDNIGHT_GRAPH_OP_ADD
            || operation.opcode == MIDNIGHT_GRAPH_OP_SUB
            || operation.opcode == MIDNIGHT_GRAPH_OP_MUL) {
            const Bls12_381Fr rhs = midnight_graph_load_value(
                operation.rhs,
                constants,
                rotations,
                fixed_values,
                dynamic_values,
                challenges,
                intermediates,
                previous,
                row,
                parameters);
            if (operation.opcode == MIDNIGHT_GRAPH_OP_ADD) {
                result = fr_add(lhs, rhs);
            } else if (operation.opcode == MIDNIGHT_GRAPH_OP_SUB) {
                result = fr_sub(lhs, rhs);
            } else {
                result = fr_montgomery_mul(lhs, rhs);
            }
        } else if (operation.opcode == MIDNIGHT_GRAPH_OP_SQUARE) {
            result = fr_montgomery_mul(lhs, lhs);
        } else if (operation.opcode == MIDNIGHT_GRAPH_OP_DOUBLE) {
            result = fr_add(lhs, lhs);
        } else if (operation.opcode == MIDNIGHT_GRAPH_OP_NEGATE) {
            result = fr_sub(midnight_graph_zero(), lhs);
        } else if (operation.opcode == MIDNIGHT_GRAPH_OP_HORNER) {
            const Bls12_381Fr factor = midnight_graph_load_value(
                operation.rhs,
                constants,
                rotations,
                fixed_values,
                dynamic_values,
                challenges,
                intermediates,
                previous,
                row,
                parameters);
            result = lhs;
            for (uint part_index = 0; part_index < operation.parts_len; ++part_index) {
                const MidnightGraphValueRef part = horner_parts[
                    static_cast<ulong>(parameters.parts_offset)
                    + operation.parts_offset + part_index];
                const Bls12_381Fr value = midnight_graph_load_value(
                    part,
                    constants,
                    rotations,
                    fixed_values,
                    dynamic_values,
                    challenges,
                    intermediates,
                    previous,
                    row,
                    parameters);
                result = fr_add(fr_montgomery_mul(result, factor), value);
            }
        } else {
            result = lhs;
        }
        intermediates[operation.target] = result;
    }
    return result;
}

inline Bls12_381Fr midnight_graph_accumulate(
    Bls12_381Fr accumulator,
    Bls12_381Fr y,
    Bls12_381Fr term)
{
    return fr_add(fr_montgomery_mul(accumulator, y), term);
}

template <uint ScratchCapacity>
inline void midnight_graph_evaluate_row(
    device const Bls12_381Fr* constants,
    device const int* rotations,
    device const MidnightGraphOperation* operations,
    device const MidnightGraphValueRef* horner_parts,
    device const Bls12_381Fr* fixed_values,
    device const Bls12_381Fr* dynamic_values,
    device const Bls12_381Fr* challenges,
    device const Bls12_381Fr* l_values,
    device Bls12_381Fr* numerator,
    constant MidnightGraphEvalParameters& parameters,
    uint row)
{
    if (row >= parameters.len) {
        return;
    }

    const Bls12_381Fr zero = midnight_graph_zero();
    Bls12_381Fr accumulator = zero;
    if (parameters.post_kind != MIDNIGHT_GRAPH_POST_STORE
        || parameters.previous_mode != 0) {
        accumulator = numerator[row];
    }
    // PreviousValue is meaningful only while custom graphs fold multiple
    // proof instances. Lookup and trash graph oracles receive zero even though
    // their post-processing continues from the resident numerator.
    const Bls12_381Fr graph_previous =
        parameters.post_kind == MIDNIGHT_GRAPH_POST_STORE
            && parameters.previous_mode != 0
        ? accumulator
        : zero;
    const Bls12_381Fr graph_result = midnight_graph_evaluate_operations<ScratchCapacity>(
        constants,
        rotations,
        operations,
        horner_parts,
        fixed_values,
        dynamic_values,
        challenges,
        graph_previous,
        row,
        parameters);

    if (parameters.post_kind == MIDNIGHT_GRAPH_POST_STORE) {
        numerator[row] = graph_result;
        return;
    }

    const Bls12_381Fr one = midnight_graph_one();
    const Bls12_381Fr l0 = l_values[row];
    const Bls12_381Fr l_last = l_values[
        static_cast<ulong>(parameters.stride) + row];
    const Bls12_381Fr l_active = l_values[
        2UL * static_cast<ulong>(parameters.stride) + row];

    if (parameters.post_kind == MIDNIGHT_GRAPH_POST_LOOKUP) {
        const uint next_row = midnight_graph_rotated_row(
            row, 1, parameters.rotation_scale, parameters.len);
        const uint previous_row = midnight_graph_rotated_row(
            row, -1, parameters.rotation_scale, parameters.len);
        const Bls12_381Fr product = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.lookup_product_column,
            row,
            parameters.stride);
        const Bls12_381Fr product_next = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.lookup_product_column,
            next_row,
            parameters.stride);
        const Bls12_381Fr input = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.lookup_input_column,
            row,
            parameters.stride);
        const Bls12_381Fr input_previous = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.lookup_input_column,
            previous_row,
            parameters.stride);
        const Bls12_381Fr table = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.lookup_table_column,
            row,
            parameters.stride);
        const Bls12_381Fr input_minus_table = fr_sub(input, table);

        Bls12_381Fr value = accumulator;
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(fr_sub(one, product), l0));
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(fr_sub(fr_montgomery_mul(product, product), product), l_last));

        const Bls12_381Fr left = fr_montgomery_mul(
            fr_montgomery_mul(product_next, fr_add(input, parameters.beta)),
            fr_add(table, parameters.gamma));
        const Bls12_381Fr right = fr_montgomery_mul(product, graph_result);
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(fr_sub(left, right), l_active));
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(input_minus_table, l0));
        value = midnight_graph_accumulate(
            value,
            parameters.y,
            fr_montgomery_mul(
                fr_montgomery_mul(input_minus_table, fr_sub(input, input_previous)),
                l_active));
        numerator[row] = value;
        return;
    }

    if (parameters.post_kind == MIDNIGHT_GRAPH_POST_TRASH) {
        const Bls12_381Fr selector = midnight_graph_dynamic_value(
            fixed_values,
            parameters.trash_selector_fixed_column,
            row,
            parameters.stride);
        const Bls12_381Fr trash = midnight_graph_dynamic_value(
            dynamic_values,
            parameters.trash_column,
            row,
            parameters.stride);
        const Bls12_381Fr term = fr_sub(
            graph_result,
            fr_montgomery_mul(fr_sub(one, selector), trash));
        numerator[row] = midnight_graph_accumulate(accumulator, parameters.y, term);
    }
}

#define MIDNIGHT_GRAPH_KERNEL(name, capacity)                                      \
kernel void name(                                                                 \
    device const Bls12_381Fr* constants [[buffer(0)]],                            \
    device const int* rotations [[buffer(1)]],                                    \
    device const MidnightGraphOperation* operations [[buffer(2)]],                \
    device const MidnightGraphValueRef* horner_parts [[buffer(3)]],               \
    device const Bls12_381Fr* fixed_values [[buffer(4)]],                         \
    device const Bls12_381Fr* dynamic_values [[buffer(5)]],                       \
    device const Bls12_381Fr* challenges [[buffer(6)]],                           \
    device const Bls12_381Fr* l_values [[buffer(7)]],                             \
    device Bls12_381Fr* numerator [[buffer(8)]],                                  \
    constant MidnightGraphEvalParameters& parameters [[buffer(9)]],               \
    uint row [[thread_position_in_grid]])                                          \
{                                                                                 \
    midnight_graph_evaluate_row<capacity>(                                        \
        constants, rotations, operations, horner_parts, fixed_values,             \
        dynamic_values, challenges, l_values, numerator, parameters, row);         \
}

MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_8, MIDNIGHT_GRAPH_SCRATCH_8)
MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_16, MIDNIGHT_GRAPH_SCRATCH_16)
MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_32, MIDNIGHT_GRAPH_SCRATCH_32)
MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_64, MIDNIGHT_GRAPH_SCRATCH_64)
MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_128, MIDNIGHT_GRAPH_SCRATCH_128)
MIDNIGHT_GRAPH_KERNEL(midnight_bls12_381_graph_evaluate_256, MIDNIGHT_GRAPH_SCRATCH_256)

kernel void midnight_bls12_381_graph_abi_probe(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }

    output[0] = MIDNIGHT_GRAPH_ABI_VERSION;
    output[1] = sizeof(MidnightGraphValueRef);
    output[2] = alignof(MidnightGraphValueRef);
    output[3] = __builtin_offsetof(MidnightGraphValueRef, kind);
    output[4] = __builtin_offsetof(MidnightGraphValueRef, a);
    output[5] = __builtin_offsetof(MidnightGraphValueRef, b);
    output[6] = sizeof(MidnightGraphOperation);
    output[7] = alignof(MidnightGraphOperation);
    output[8] = __builtin_offsetof(MidnightGraphOperation, opcode);
    output[9] = __builtin_offsetof(MidnightGraphOperation, target);
    output[10] = __builtin_offsetof(MidnightGraphOperation, parts_offset);
    output[11] = __builtin_offsetof(MidnightGraphOperation, parts_len);
    output[12] = __builtin_offsetof(MidnightGraphOperation, lhs);
    output[13] = __builtin_offsetof(MidnightGraphOperation, rhs);
    output[14] = sizeof(MidnightGraphEvalParameters);
    output[15] = alignof(MidnightGraphEvalParameters);
    output[16] = __builtin_offsetof(MidnightGraphEvalParameters, len);
    output[17] = __builtin_offsetof(MidnightGraphEvalParameters, stride);
    output[18] = __builtin_offsetof(MidnightGraphEvalParameters, rotation_scale);
    output[19] = __builtin_offsetof(MidnightGraphEvalParameters, post_kind);
    output[20] = __builtin_offsetof(MidnightGraphEvalParameters, constants_offset);
    output[21] = __builtin_offsetof(MidnightGraphEvalParameters, constants_count);
    output[22] = __builtin_offsetof(MidnightGraphEvalParameters, rotations_offset);
    output[23] = __builtin_offsetof(MidnightGraphEvalParameters, rotations_count);
    output[24] = __builtin_offsetof(MidnightGraphEvalParameters, operations_offset);
    output[25] = __builtin_offsetof(MidnightGraphEvalParameters, operations_count);
    output[26] = __builtin_offsetof(MidnightGraphEvalParameters, parts_offset);
    output[27] = __builtin_offsetof(MidnightGraphEvalParameters, parts_count);
    output[28] = __builtin_offsetof(MidnightGraphEvalParameters, fixed_columns);
    output[29] = __builtin_offsetof(MidnightGraphEvalParameters, advice_offset);
    output[30] = __builtin_offsetof(MidnightGraphEvalParameters, advice_columns);
    output[31] = __builtin_offsetof(MidnightGraphEvalParameters, instance_offset);
    output[32] = __builtin_offsetof(MidnightGraphEvalParameters, instance_columns);
    output[33] = __builtin_offsetof(MidnightGraphEvalParameters, challenge_count);
    output[34] = __builtin_offsetof(MidnightGraphEvalParameters, previous_mode);
    output[35] = __builtin_offsetof(MidnightGraphEvalParameters, lookup_product_column);
    output[36] = __builtin_offsetof(MidnightGraphEvalParameters, lookup_input_column);
    output[37] = __builtin_offsetof(MidnightGraphEvalParameters, lookup_table_column);
    output[38] = __builtin_offsetof(MidnightGraphEvalParameters, trash_column);
    output[39] = __builtin_offsetof(MidnightGraphEvalParameters, trash_selector_fixed_column);
    output[40] = __builtin_offsetof(MidnightGraphEvalParameters, beta);
    output[41] = __builtin_offsetof(MidnightGraphEvalParameters, gamma);
    output[42] = __builtin_offsetof(MidnightGraphEvalParameters, theta);
    output[43] = __builtin_offsetof(MidnightGraphEvalParameters, trash_challenge);
    output[44] = __builtin_offsetof(MidnightGraphEvalParameters, y);

    output[45] = MIDNIGHT_GRAPH_VALUE_CONSTANT;
    output[46] = MIDNIGHT_GRAPH_VALUE_INTERMEDIATE;
    output[47] = MIDNIGHT_GRAPH_VALUE_FIXED;
    output[48] = MIDNIGHT_GRAPH_VALUE_ADVICE;
    output[49] = MIDNIGHT_GRAPH_VALUE_INSTANCE;
    output[50] = MIDNIGHT_GRAPH_VALUE_CHALLENGE;
    output[51] = MIDNIGHT_GRAPH_VALUE_BETA;
    output[52] = MIDNIGHT_GRAPH_VALUE_GAMMA;
    output[53] = MIDNIGHT_GRAPH_VALUE_THETA;
    output[54] = MIDNIGHT_GRAPH_VALUE_TRASH_CHALLENGE;
    output[55] = MIDNIGHT_GRAPH_VALUE_Y;
    output[56] = MIDNIGHT_GRAPH_VALUE_PREVIOUS;
    output[57] = MIDNIGHT_GRAPH_OP_ADD;
    output[58] = MIDNIGHT_GRAPH_OP_SUB;
    output[59] = MIDNIGHT_GRAPH_OP_MUL;
    output[60] = MIDNIGHT_GRAPH_OP_SQUARE;
    output[61] = MIDNIGHT_GRAPH_OP_DOUBLE;
    output[62] = MIDNIGHT_GRAPH_OP_NEGATE;
    output[63] = MIDNIGHT_GRAPH_OP_HORNER;
    output[64] = MIDNIGHT_GRAPH_OP_STORE;
    output[65] = MIDNIGHT_GRAPH_POST_STORE;
    output[66] = MIDNIGHT_GRAPH_POST_LOOKUP;
    output[67] = MIDNIGHT_GRAPH_POST_TRASH;
    output[68] = MIDNIGHT_GRAPH_SCRATCH_8;
    output[69] = MIDNIGHT_GRAPH_SCRATCH_16;
    output[70] = MIDNIGHT_GRAPH_SCRATCH_32;
    output[71] = MIDNIGHT_GRAPH_SCRATCH_64;
    output[72] = MIDNIGHT_GRAPH_SCRATCH_128;
    output[73] = MIDNIGHT_GRAPH_SCRATCH_256;

    ulong fingerprint = 0xcbf29ce484222325UL;
    for (uint word = 0; word < 74; ++word) {
        fingerprint = midnight_graph_abi_mix(fingerprint, output[word]);
    }
    output[74] = fingerprint;
}

#undef MIDNIGHT_GRAPH_KERNEL

#endif
