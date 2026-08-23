// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "bls12_381_u32_constants.metal"

using namespace metal;

#ifndef BLS12_381_U32_FIELD_ELEMENT_TYPE
#define BLS12_381_U32_FIELD_ELEMENT_TYPE
struct U32FieldElement {
    uint limbs[12];
};
#endif

struct U32FieldHalf {
    uint limbs[6];
};

struct U32FieldParameters {
    uint logical_fields;
    uint iterations;
    uint reserved0;
    uint reserved1;
};

struct U32FieldAccumulator {
    ulong value;
    uint high;
};

inline uint u32_field_side_mask(uint side)
{
    return 0U - side;
}

inline uint u32_field_role_select(
    uint lane_zero,
    uint lane_one,
    uint side_mask)
{
    return (lane_zero & ~side_mask) | (lane_one & side_mask);
}

inline uint u32_field_assign_owner(
    uint previous,
    uint value,
    uint side,
    uint owner)
{
    const uint owner_mask =
        0U - static_cast<uint>(side == owner);
    return (previous & ~owner_mask) | (value & owner_mask);
}

inline U32FieldAccumulator u32_field_accumulator_from_carry(
    uint carry_low,
    uint carry_high,
    uint side_mask)
{
    U32FieldAccumulator result;
    result.value =
        static_cast<ulong>(carry_low & ~side_mask) |
        (static_cast<ulong>(carry_high & ~side_mask) << 32);
    result.high = 0U;
    return result;
}

inline void u32_field_accumulate_product(
    thread U32FieldAccumulator& accumulator,
    uint lhs,
    uint rhs)
{
    const ulong product =
        static_cast<ulong>(lhs) * static_cast<ulong>(rhs);
    const ulong next = accumulator.value + product;
    accumulator.high += static_cast<uint>(next < accumulator.value);
    accumulator.value = next;
}

inline uint3 u32_field_combine_accumulator(
    U32FieldAccumulator accumulator)
{
    const uint4 local = uint4(
        static_cast<uint>(accumulator.value),
        static_cast<uint>(accumulator.value >> 32),
        accumulator.high,
        0U);
    const uint4 peer = simd_shuffle_xor(local, 1);

    const uint low = local.x + peer.x;
    const uint low_carry = static_cast<uint>(low < local.x);
    const uint middle_base = local.y + peer.y;
    const uint middle_base_carry =
        static_cast<uint>(middle_base < local.y);
    const uint middle = middle_base + low_carry;
    const uint middle_carry = static_cast<uint>(middle < middle_base);
    const uint high =
        local.z + peer.z + (middle_base_carry | middle_carry);
    return uint3(low, middle, high);
}

inline uint u32_field_subtract_flags(
    uint value,
    uint side,
    uint pair_index)
{
    const uint modulus =
        BLS12_381_U32_FP_MODULUS[2U * pair_index + side];
    return static_cast<uint>(value < modulus) |
        (static_cast<uint>(value == modulus) << 1);
}

inline uint u32_field_subtract_pair(
    uint value,
    uint local_flags,
    uint peer_flags,
    uint side,
    uint pair_index,
    thread uint& borrow)
{
    const uint side_mask = u32_field_side_mask(side);
    const uint even_flags = u32_field_role_select(
        local_flags, peer_flags, side_mask);
    const uint odd_flags = u32_field_role_select(
        peer_flags, local_flags, side_mask);
    const uint even_borrow =
        (even_flags & 1U) | (((even_flags >> 1) & 1U) & borrow);
    const uint incoming = u32_field_role_select(
        borrow, even_borrow, side_mask);
    const uint modulus =
        BLS12_381_U32_FP_MODULUS[2U * pair_index + side];
    const uint difference = value - modulus - incoming;
    borrow =
        (odd_flags & 1U) | (((odd_flags >> 1) & 1U) & even_borrow);
    return difference;
}

inline uint u32_field_pair_high(uint value, uint side)
{
    const uint peer = simd_shuffle_xor(value, 1);
    const uint mask = u32_field_side_mask(side);
    return (peer & ~mask) | (value & mask);
}

inline uint u32_field_pair_low(uint value, uint side)
{
    const uint peer = simd_shuffle_xor(value, 1);
    const uint mask = u32_field_side_mask(side);
    return (value & ~mask) | (peer & mask);
}

inline uint u32_field_block_multiply_accumulate(
    U32FieldHalf multiplicand,
    uint multiplier,
    thread U32FieldHalf& accumulator)
{
    ulong carry = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        const ulong value =
            static_cast<ulong>(multiplicand.limbs[index]) *
                static_cast<ulong>(multiplier) +
            static_cast<ulong>(accumulator.limbs[index]) + carry;
        accumulator.limbs[index] = static_cast<uint>(value);
        carry = value >> 32;
    }
    return static_cast<uint>(carry);
}

inline uint u32_field_add_incoming_carry(
    thread U32FieldHalf& value,
    uint high,
    uint incoming)
{
    ulong accumulated =
        static_cast<ulong>(value.limbs[0]) + static_cast<ulong>(incoming);
    value.limbs[0] = static_cast<uint>(accumulated);
    uint carry = static_cast<uint>(accumulated >> 32);
#pragma unroll
    for (uint index = 1; index < 6; ++index) {
        accumulated =
            static_cast<ulong>(value.limbs[index]) + static_cast<ulong>(carry);
        value.limbs[index] = static_cast<uint>(accumulated);
        carry = static_cast<uint>(accumulated >> 32);
    }

    // A six-word block plus a 32-bit incoming carry is strictly below 2^224,
    // so the adjusted high word cannot itself overflow.
    return high + carry;
}

inline uint u32_field_block_subtract(
    U32FieldHalf value,
    U32FieldHalf modulus,
    thread U32FieldHalf& difference)
{
    uint borrow = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        const uint first = value.limbs[index] - modulus.limbs[index];
        const uint first_borrow = static_cast<uint>(
            value.limbs[index] < modulus.limbs[index]);
        const uint result = first - borrow;
        const uint second_borrow = static_cast<uint>(first < borrow);
        difference.limbs[index] = result;
        borrow = first_borrow | second_borrow;
    }
    return borrow;
}

inline uint u32_field_subtract_incoming_borrow(
    thread U32FieldHalf& value,
    uint block_borrow,
    uint incoming)
{
    uint result = value.limbs[0] - incoming;
    uint borrow = static_cast<uint>(value.limbs[0] < incoming);
    value.limbs[0] = result;
#pragma unroll
    for (uint index = 1; index < 6; ++index) {
        result = value.limbs[index] - borrow;
        const uint next_borrow = static_cast<uint>(value.limbs[index] < borrow);
        value.limbs[index] = result;
        borrow = next_borrow;
    }
    return block_borrow | borrow;
}

inline U32FieldHalf u32_field_half(
    U32FieldElement value,
    uint side)
{
    U32FieldHalf result;
    const uint offset = side * 6;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = value.limbs[offset + index];
    }
    return result;
}

inline U32FieldHalf u32_field_parity_half(
    U32FieldElement value,
    uint side)
{
    U32FieldHalf result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = value.limbs[2U * index + side];
    }
    return result;
}

inline U32FieldHalf u32_field_modulus_half(uint side)
{
    U32FieldHalf result;
    const uint offset = side * 6;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = BLS12_381_U32_FP_MODULUS[offset + index];
    }
    return result;
}

#include "bls12_381_u32_fips.metal"
#include "bls12_381_u32_named_fios.metal"
#include "bls12_381_u32_named_karatsuba_l4.metal"
#include "bls12_381_u32_named_blocked128_karatsuba_l4.metal"

inline U32FieldHalf u32_field_montgomery_multiply(
    U32FieldHalf lhs,
    U32FieldElement rhs,
    uint side)
{
    U32FieldHalf temporary = {{0U, 0U, 0U, 0U, 0U, 0U}};
    const U32FieldHalf modulus = u32_field_modulus_half(side);
    const uint side_mask = u32_field_side_mask(side);
    uint top = 0;

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 12; ++rhs_index) {
        uint block_high = u32_field_block_multiply_accumulate(
            lhs, rhs.limbs[rhs_index], temporary);
        const uint incoming_product =
            simd_shuffle_xor(block_high, 1) & side_mask;
        block_high = u32_field_add_incoming_carry(
            temporary, block_high, incoming_product);
        const uint product_high = u32_field_pair_high(block_high, side);

        const ulong product_top =
            static_cast<ulong>(top) + static_cast<ulong>(product_high);
        const uint top_word = static_cast<uint>(product_top);
        const uint extra_word = static_cast<uint>(product_top >> 32);

        const uint low_word = u32_field_pair_low(temporary.limbs[0], side);
        const uint reduction = low_word * BLS12_381_U32_FP_INV;
        block_high = u32_field_block_multiply_accumulate(
            modulus, reduction, temporary);
        const uint incoming_reduction =
            simd_shuffle_xor(block_high, 1) & side_mask;
        block_high = u32_field_add_incoming_carry(
            temporary, block_high, incoming_reduction);
        const uint reduction_high = u32_field_pair_high(block_high, side);

        const ulong reduction_top =
            static_cast<ulong>(top_word) +
            static_cast<ulong>(reduction_high);
        const uint next_last = static_cast<uint>(reduction_top);
        const uint next_top =
            extra_word + static_cast<uint>(reduction_top >> 32);

        const uint peer_first = simd_shuffle_xor(temporary.limbs[0], 1);
#pragma unroll
        for (uint index = 0; index < 5; ++index) {
            temporary.limbs[index] = temporary.limbs[index + 1];
        }
        temporary.limbs[5] =
            (peer_first & ~side_mask) | (next_last & side_mask);
        top = next_top;
    }

    const U32FieldHalf original = temporary;
    U32FieldHalf subtracted;
    uint block_borrow = u32_field_block_subtract(
        original, modulus, subtracted);
    const uint incoming_borrow =
        simd_shuffle_xor(block_borrow, 1) & side_mask;
    block_borrow = u32_field_subtract_incoming_borrow(
        subtracted, block_borrow, incoming_borrow);
    const uint final_borrow = u32_field_pair_high(block_borrow, side);
    const uint use_subtracted = 0U - (final_borrow ^ 1U);

#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        temporary.limbs[index] =
            (original.limbs[index] & ~use_subtracted) |
            (subtracted.limbs[index] & use_subtracted);
    }
    return temporary;
}

kernel void bls12_381_metal_msm_u32_field_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(U32FieldElement);
    output[2] = alignof(U32FieldElement);
    output[3] = sizeof(U32FieldParameters);
    output[4] = alignof(U32FieldParameters);
    output[5] = __builtin_offsetof(U32FieldParameters, iterations);
    output[6] = __builtin_offsetof(U32FieldParameters, logical_fields);
    output[7] = 2UL;
    output[8] = 12UL;
    output[9] = 384UL;
    output[10] = static_cast<ulong>(BLS12_381_U32_FP_INV);
    output[11] = static_cast<ulong>(BLS12_381_U32_FP_MODULUS[0]);
    output[12] = static_cast<ulong>(BLS12_381_U32_FP_MODULUS[11]);
    output[13] = static_cast<ulong>(BLS12_381_U32_FIELD_ALGORITHM_VERSION);
    output[14] = BLS12_381_U32_FIELD_ABI_FINGERPRINT;
    output[15] = sizeof(uint);
}

kernel void bls12_381_metal_msm_u32_field_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint field_index = thread_index >> 1;
    const uint side = simd_lane & 1U;

    // Physical input is padded to complete SIMD32 groups. Every peer remains
    // active through all shuffles; only stores are suppressed for padding.
    U32FieldHalf accumulator = u32_field_parity_half(lhs[field_index], side);
    const U32FieldHalf multiplier =
        u32_field_parity_half(rhs[field_index], side);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulator = u32_field_fips_montgomery_multiply(
            accumulator, multiplier, side);
    }

    if (field_index < parameters.logical_fields) {
#pragma unroll
        for (uint index = 0; index < 6; ++index) {
            output[field_index].limbs[2U * index + side] =
                accumulator.limbs[index];
        }
    }
}

kernel void bls12_381_metal_msm_u32_named_field_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(U32FieldElement);
    output[2] = alignof(U32FieldElement);
    output[3] = sizeof(U32FieldParameters);
    output[4] = alignof(U32FieldParameters);
    output[5] = __builtin_offsetof(U32FieldParameters, iterations);
    output[6] = __builtin_offsetof(U32FieldParameters, logical_fields);
    output[7] = 1UL;
    output[8] = 12UL;
    output[9] = 384UL;
    output[10] = static_cast<ulong>(BLS12_381_U32_FP_INV);
    output[11] = static_cast<ulong>(BLS12_381_U32_FP_MODULUS[0]);
    output[12] = static_cast<ulong>(BLS12_381_U32_FP_MODULUS[11]);
    output[13] =
        static_cast<ulong>(BLS12_381_U32_NAMED_FIELD_ALGORITHM_VERSION);
    output[14] = BLS12_381_U32_NAMED_FIELD_ABI_FINGERPRINT;
    output[15] = static_cast<ulong>(BLS12_381_U32_NAMED_FIELD_VARIANT_COUNT);
}

kernel void bls12_381_metal_msm_u32_named_fios_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint field_index [[thread_position_in_grid]])
{
    if (field_index >= parameters.logical_fields) {
        return;
    }

    U32NamedFieldElement accumulator =
        u32_named_field_from_abi(lhs[field_index]);
    const U32NamedFieldElement multiplier =
        u32_named_field_from_abi(rhs[field_index]);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulator = u32_named_fios_montgomery_multiply(
            multiplier, accumulator);
    }
    output[field_index] = u32_named_field_to_abi(accumulator);
}

kernel void bls12_381_metal_msm_u32_named_fios_reversed_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint field_index [[thread_position_in_grid]])
{
    if (field_index >= parameters.logical_fields) {
        return;
    }

    U32NamedFieldElement accumulator =
        u32_named_field_from_abi(lhs[field_index]);
    const U32NamedFieldElement multiplier =
        u32_named_field_from_abi(rhs[field_index]);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulator = u32_named_fios_montgomery_multiply(
            accumulator, multiplier);
    }
    output[field_index] = u32_named_field_to_abi(accumulator);
}

kernel void bls12_381_metal_msm_u32_named_karatsuba_l4_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(U32FieldElement);
    output[2] = alignof(U32FieldElement);
    output[3] = sizeof(U32FieldParameters);
    output[4] = alignof(U32FieldParameters);
    output[5] = __builtin_offsetof(U32FieldParameters, iterations);
    output[6] = __builtin_offsetof(U32FieldParameters, logical_fields);
    output[7] = 1UL;
    output[8] = 12UL;
    output[9] = 384UL;
    output[10] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12);
    output[11] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC);
    output[12] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_LAZY_MULTIPLE);
    output[13] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_ALGORITHM_VERSION);
    output[14] = BLS12_381_U32_KARATSUBA_L4_ABI_FINGERPRINT;
    output[15] = BLS12_381_U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT;
    output[16] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS);
    output[17] = static_cast<ulong>(
        BLS12_381_U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT);
}

kernel void bls12_381_metal_msm_u32_named_karatsuba_l4_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint field_index [[thread_position_in_grid]])
{
    if (field_index >= parameters.logical_fields) {
        return;
    }

    U32NamedFieldElement accumulator =
        u32_named_field_from_abi(lhs[field_index]);
    const U32NamedFieldElement multiplier =
        u32_named_field_from_abi(rhs[field_index]);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulator = u32_named_karatsuba_l4_redc(
            accumulator, multiplier);
    }
    output[field_index] = u32_named_field_to_abi(accumulator);
}

kernel void bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(U32FieldElement);
    output[2] = alignof(U32FieldElement);
    output[3] = sizeof(U32FieldParameters);
    output[4] = alignof(U32FieldParameters);
    output[5] = __builtin_offsetof(U32FieldParameters, iterations);
    output[6] = __builtin_offsetof(U32FieldParameters, logical_fields);
    output[7] = 1UL;
    output[8] = 12UL;
    output[9] = 384UL;
    output[10] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4);
    output[11] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC);
    output[12] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS);
    output[13] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ROUNDS);
    output[14] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS);
    output[15] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE);
    output[16] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION);
    output[17] = BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT;
    output[18] =
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT;
    output[19] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS);
    output[20] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS);
    output[21] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS);
    output[22] = static_cast<ulong>(
        BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT);
}

kernel void bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint field_index [[thread_position_in_grid]])
{
    if (field_index >= parameters.logical_fields) {
        return;
    }

    U32NamedFieldElement accumulator =
        u32_named_field_from_abi(lhs[field_index]);
    const U32NamedFieldElement multiplier =
        u32_named_field_from_abi(rhs[field_index]);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulator = u32_named_blocked128_karatsuba_l4_redc(
            accumulator, multiplier);
    }
    output[field_index] = u32_named_field_to_abi(accumulator);
}

kernel void bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_rotating_chain(
    device const U32FieldElement* lhs [[buffer(0)]],
    device const U32FieldElement* rhs [[buffer(1)]],
    device U32FieldElement* output [[buffer(2)]],
    constant U32FieldParameters& parameters [[buffer(3)]],
    uint field_index [[thread_position_in_grid]])
{
    if (field_index >= parameters.logical_fields) {
        return;
    }

    U32NamedFieldElement left =
        u32_named_field_from_abi(lhs[field_index]);
    U32NamedFieldElement right =
        u32_named_field_from_abi(rhs[field_index]);
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        const U32NamedFieldElement product =
            u32_named_blocked128_karatsuba_l4_redc(left, right);
        left = right;
        right = product;
    }
    output[field_index] = u32_named_field_to_abi(right);
}
