// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_NTT_METAL
#define MIDNIGHT_BLS12_381_NTT_METAL

// This phase file intentionally relies on field.metal being included first by
// the single root translation unit. The pass layout follows the mixed-radix
// Stockham autosort used by ec-gpu-gen 0.7.1, specialized here for a batched,
// resident Metal buffer ABI and natural-order inputs and outputs.

constant uint MIDNIGHT_NTT_FLAG_NORMALIZE = 1U;
constant uint MIDNIGHT_NTT_FLAG_COSET_EGRESS = 2U;
constant ulong MIDNIGHT_NTT_PARAMETER_ABI_VERSION = 2UL;

struct MidnightNttPassParameters {
    uint n;
    uint source_stride;
    uint destination_stride;
    uint log_p;
    uint degree;
    uint max_degree;
    uint batch_count;
    uint flags;
    uint reserved[2];
    Bls12_381Fr one;
    Bls12_381Fr inverse_size;
    Bls12_381Fr coset_generator;
    Bls12_381Fr inverse_coset_generator;
};

struct MidnightNttCosetParameters {
    uint input_length;
    uint output_length;
    uint source_stride;
    uint destination_stride;
    uint batch_count;
    uint reserved;
    Bls12_381Fr coset_generator;
    Bls12_381Fr inverse_coset_generator;
};

inline ulong midnight_ntt_abi_mix(ulong fingerprint, ulong value)
{
    // FNV-1a over fixed-width layout words. This is a diagnostic layout
    // fingerprint, not a cryptographic digest.
    return (fingerprint ^ value) * 0x00000100000001b3UL;
}

kernel void midnight_bls12_381_ntt_parameter_abi_probe(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }

    output[0] = MIDNIGHT_NTT_PARAMETER_ABI_VERSION;
    output[1] = sizeof(MidnightNttPassParameters);
    output[2] = alignof(MidnightNttPassParameters);
    output[3] = __builtin_offsetof(MidnightNttPassParameters, n);
    output[4] = __builtin_offsetof(MidnightNttPassParameters, source_stride);
    output[5] = __builtin_offsetof(MidnightNttPassParameters, destination_stride);
    output[6] = __builtin_offsetof(MidnightNttPassParameters, log_p);
    output[7] = __builtin_offsetof(MidnightNttPassParameters, degree);
    output[8] = __builtin_offsetof(MidnightNttPassParameters, max_degree);
    output[9] = __builtin_offsetof(MidnightNttPassParameters, batch_count);
    output[10] = __builtin_offsetof(MidnightNttPassParameters, flags);
    output[11] = __builtin_offsetof(MidnightNttPassParameters, reserved);
    output[12] = __builtin_offsetof(MidnightNttPassParameters, one);
    output[13] = __builtin_offsetof(MidnightNttPassParameters, inverse_size);
    output[14] = __builtin_offsetof(MidnightNttPassParameters, coset_generator);
    output[15] = __builtin_offsetof(MidnightNttPassParameters, inverse_coset_generator);
    output[16] = sizeof(MidnightNttCosetParameters);
    output[17] = alignof(MidnightNttCosetParameters);
    output[18] = __builtin_offsetof(MidnightNttCosetParameters, input_length);
    output[19] = __builtin_offsetof(MidnightNttCosetParameters, output_length);
    output[20] = __builtin_offsetof(MidnightNttCosetParameters, source_stride);
    output[21] = __builtin_offsetof(MidnightNttCosetParameters, destination_stride);
    output[22] = __builtin_offsetof(MidnightNttCosetParameters, batch_count);
    output[23] = __builtin_offsetof(MidnightNttCosetParameters, reserved);
    output[24] = __builtin_offsetof(MidnightNttCosetParameters, coset_generator);
    output[25] = __builtin_offsetof(MidnightNttCosetParameters, inverse_coset_generator);

    ulong fingerprint = 0xcbf29ce484222325UL;
    for (uint index = 0; index < 26; ++index) {
        fingerprint = midnight_ntt_abi_mix(fingerprint, output[index]);
    }
    output[26] = fingerprint;
}

inline Bls12_381Fr midnight_fr_zero()
{
    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = 0;
    }
    return result;
}

inline Bls12_381Fr midnight_fr_pow_lookup(
    device const Bls12_381Fr* squares,
    uint exponent,
    Bls12_381Fr one)
{
    Bls12_381Fr result = one;
    uint bit = 0;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = fr_montgomery_mul(result, squares[bit]);
        }
        exponent >>= 1;
        ++bit;
    }
    return result;
}

inline uint midnight_reverse_low_bits(uint value, uint bit_count)
{
    uint reversed = 0;
    for (uint bit = 0; bit < bit_count; ++bit) {
        reversed = (reversed << 1) | (value & 1U);
        value >>= 1;
    }
    return reversed;
}

inline Bls12_381Fr midnight_ntt_finish_value(
    Bls12_381Fr value,
    uint coefficient_index,
    constant MidnightNttPassParameters& parameters)
{
    if ((parameters.flags & MIDNIGHT_NTT_FLAG_NORMALIZE) != 0) {
        value = fr_montgomery_mul(value, parameters.inverse_size);
    }
    if ((parameters.flags & MIDNIGHT_NTT_FLAG_COSET_EGRESS) != 0) {
        const uint coset_index = coefficient_index % 3U;
        if (coset_index == 1U) {
            value = fr_montgomery_mul(value, parameters.inverse_coset_generator);
        } else if (coset_index == 2U) {
            value = fr_montgomery_mul(value, parameters.coset_generator);
        }
    }
    return value;
}

kernel void midnight_bls12_381_ntt_stockham(
    device const Bls12_381Fr* source [[buffer(0)]],
    device Bls12_381Fr* destination [[buffer(1)]],
    device const Bls12_381Fr* pq [[buffer(2)]],
    device const Bls12_381Fr* omega_squares [[buffer(3)]],
    constant MidnightNttPassParameters& parameters [[buffer(4)]],
    threadgroup Bls12_381Fr* scratch [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    const uint groups_per_batch = parameters.n >> parameters.degree;
    if (group.x >= groups_per_batch || group.y >= parameters.batch_count) {
        return;
    }

    const uint index = group.x;
    const uint count = 1U << parameters.degree;
    const uint half_count = count >> 1;
    const uint t = parameters.n >> parameters.degree;
    const uint p = 1U << parameters.log_p;
    const uint k = index & (p - 1U);
    // The host dispatches count/2 lanes, so each lane owns exactly two input
    // values and one butterfly in every round.
    const uint lane_begin = lane << 1;
    const ulong source_batch_base =
        static_cast<ulong>(group.y) * static_cast<ulong>(parameters.source_stride);
    const ulong destination_batch_base =
        static_cast<ulong>(group.y) * static_cast<ulong>(parameters.destination_stride);

    const uint twiddle_exponent =
        (parameters.n >> parameters.log_p >> parameters.degree) * k;
    const Bls12_381Fr twiddle = midnight_fr_pow_lookup(
        omega_squares,
        twiddle_exponent,
        parameters.one);
    const ulong first_source_index = source_batch_base
        + static_cast<ulong>(index)
        + static_cast<ulong>(lane_begin) * static_cast<ulong>(t);
    const ulong second_source_index = first_source_index + static_cast<ulong>(t);
    if (parameters.log_p == 0) {
        // p == 1 makes k and the pass twiddle zero/one respectively.  This is
        // the first pass of every transform, so avoid four Montgomery
        // multiplications by one per lane.
        scratch[lane_begin] = source[first_source_index];
        scratch[lane_begin + 1U] = source[second_source_index];
    } else {
        // `twiddle` is root^twiddle_exponent.  Starting a lane at `lane_begin`
        // therefore needs root^(twiddle_exponent*lane_begin), which the
        // existing powers-of-two table can construct directly.
        const uint lane_twiddle_exponent = twiddle_exponent * lane_begin;
        const Bls12_381Fr power = midnight_fr_pow_lookup(
            omega_squares,
            lane_twiddle_exponent,
            parameters.one);
        scratch[lane_begin] = fr_montgomery_mul(power, source[first_source_index]);
        scratch[lane_begin + 1U] = fr_montgomery_mul(
            fr_montgomery_mul(power, twiddle),
            source[second_source_index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pq_shift = parameters.max_degree - parameters.degree;
    for (uint round = 0; round < parameters.degree; ++round) {
        const uint bit = half_count >> round;
        for (uint local_index = lane; local_index < lane + 1U; ++local_index) {
            const uint diagonal = local_index & (bit - 1U);
            const uint first = (local_index << 1) - diagonal;
            const uint second = first + bit;
            const Bls12_381Fr first_value = scratch[first];
            scratch[first] = fr_add(first_value, scratch[second]);
            Bls12_381Fr second_value = fr_sub(first_value, scratch[second]);
            if (diagonal != 0) {
                const uint pq_index = diagonal << round << pq_shift;
                second_value = fr_montgomery_mul(pq[pq_index], second_value);
            }
            scratch[second] = second_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint destination_base = ((index - k) << parameters.degree) + k;
    for (uint local_index = lane; local_index < lane + 1U; ++local_index) {
        const uint first_local = midnight_reverse_low_bits(local_index, parameters.degree);
        const uint second_local =
            midnight_reverse_low_bits(local_index + half_count, parameters.degree);
        const uint first_index = destination_base + local_index * p;
        const uint second_index = destination_base + (local_index + half_count) * p;
        destination[destination_batch_base + first_index] = midnight_ntt_finish_value(
            scratch[first_local],
            first_index,
            parameters);
        destination[destination_batch_base + second_index] = midnight_ntt_finish_value(
            scratch[second_local],
            second_index,
            parameters);
    }
}

kernel void midnight_bls12_381_ntt_identity(
    device const Bls12_381Fr* source [[buffer(0)]],
    device Bls12_381Fr* destination [[buffer(1)]],
    constant MidnightNttPassParameters& parameters [[buffer(2)]],
    uint2 index [[thread_position_in_grid]])
{
    if (index.x >= parameters.n || index.y >= parameters.batch_count) {
        return;
    }
    const ulong source_batch_base =
        static_cast<ulong>(index.y) * static_cast<ulong>(parameters.source_stride);
    const ulong destination_batch_base =
        static_cast<ulong>(index.y) * static_cast<ulong>(parameters.destination_stride);
    destination[destination_batch_base + index.x] = midnight_ntt_finish_value(
        source[source_batch_base + index.x],
        index.x,
        parameters);
}

kernel void midnight_bls12_381_ntt_coset_prepare(
    device const Bls12_381Fr* source [[buffer(0)]],
    device Bls12_381Fr* destination [[buffer(1)]],
    constant MidnightNttCosetParameters& parameters [[buffer(2)]],
    uint2 index [[thread_position_in_grid]])
{
    if (index.x >= parameters.output_length || index.y >= parameters.batch_count) {
        return;
    }

    Bls12_381Fr value = midnight_fr_zero();
    const ulong source_batch_base =
        static_cast<ulong>(index.y) * static_cast<ulong>(parameters.source_stride);
    const ulong destination_batch_base =
        static_cast<ulong>(index.y) * static_cast<ulong>(parameters.destination_stride);
    if (index.x < parameters.input_length) {
        value = source[source_batch_base + index.x];
        const uint coset_index = index.x % 3U;
        if (coset_index == 1U) {
            value = fr_montgomery_mul(value, parameters.coset_generator);
        } else if (coset_index == 2U) {
            value = fr_montgomery_mul(value, parameters.inverse_coset_generator);
        }
    }
    destination[destination_batch_base + index.x] = value;
}

#endif
