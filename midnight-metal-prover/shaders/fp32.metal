// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_FP32_METAL
#define MIDNIGHT_BLS12_381_FP32_METAL

#include "field.metal"

// Experimental diagnostic representation only. Limbs are little-endian 32-bit
// words of the same R=2^384 Montgomery integer used by the public six-ulong Fp
// ABI. Existing field, NTT, G1, and proof kernels never use this type.
struct Bls12_381Fp32 {
    uint limbs[12];
};

inline Bls12_381Fp32 fp32_from_public(Bls12_381Fp value)
{
    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[2 * index] = static_cast<uint>(value.limbs[index]);
        result.limbs[2 * index + 1] =
            static_cast<uint>(value.limbs[index] >> 32);
    }
    return result;
}

inline Bls12_381Fp fp32_to_public(Bls12_381Fp32 value)
{
    Bls12_381Fp result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        const ulong low = static_cast<ulong>(value.limbs[2 * index]);
        const ulong high =
            static_cast<ulong>(value.limbs[2 * index + 1]) << 32;
        result.limbs[index] = low | high;
    }
    return result;
}

inline uint fp32_add_with_carry(
    uint lhs,
    uint rhs,
    thread uint& carry)
{
    const uint first = lhs + rhs;
    const uint first_carry = static_cast<uint>(first < lhs);
    const uint result = first + carry;
    const uint second_carry = static_cast<uint>(result < first);
    carry = first_carry | second_carry;
    return result;
}

inline uint fp32_subtract_with_borrow(
    uint lhs,
    uint rhs,
    thread uint& borrow)
{
    const uint first = lhs - rhs;
    const uint first_borrow = static_cast<uint>(lhs < rhs);
    const uint result = first - borrow;
    const uint second_borrow = static_cast<uint>(first < borrow);
    borrow = first_borrow | second_borrow;
    return result;
}

inline uint fp32_select_with_mask(
    uint when_clear,
    uint when_set,
    uint mask)
{
    return (when_clear & ~mask) | (when_set & mask);
}

inline Bls12_381Fp32 fp32_reduce_once(Bls12_381Fp32 value)
{
    Bls12_381Fp32 subtracted;
    uint borrow = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        subtracted.limbs[index] = fp32_subtract_with_borrow(
            value.limbs[index],
            MIDNIGHT_BLS12_381_FP32_MODULUS[index],
            borrow);
    }

    const uint use_subtracted = 0U - (borrow ^ 1U);
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        value.limbs[index] = fp32_select_with_mask(
            value.limbs[index], subtracted.limbs[index], use_subtracted);
    }
    return value;
}

inline Bls12_381Fp32 fp32_add(Bls12_381Fp32 lhs, Bls12_381Fp32 rhs)
{
    Bls12_381Fp32 result;
    uint carry = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] =
            fp32_add_with_carry(lhs.limbs[index], rhs.limbs[index], carry);
    }
    return fp32_reduce_once(result);
}

inline Bls12_381Fp32 fp32_sub(Bls12_381Fp32 lhs, Bls12_381Fp32 rhs)
{
    Bls12_381Fp32 result;
    uint borrow = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = fp32_subtract_with_borrow(
            lhs.limbs[index], rhs.limbs[index], borrow);
    }

    const uint add_modulus = 0U - borrow;
    uint carry = 0;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = fp32_add_with_carry(
            result.limbs[index],
            MIDNIGHT_BLS12_381_FP32_MODULUS[index] & add_modulus,
            carry);
    }
    return result;
}

inline uint fp32_multiply_accumulate(
    uint lhs,
    uint rhs,
    uint addend,
    thread uint& carry)
{
    uint low = lhs * rhs;
    uint high = mulhi(lhs, rhs);

    uint next = low + addend;
    high += static_cast<uint>(next < low);
    low = next;

    next = low + carry;
    high += static_cast<uint>(next < low);
    carry = high;
    return next;
}

inline uint fp32_add_carry_word(uint value, thread uint& carry)
{
    const uint result = value + carry;
    carry = static_cast<uint>(result < value);
    return result;
}

inline Bls12_381Fp32 fp32_montgomery_mul(
    Bls12_381Fp32 lhs,
    Bls12_381Fp32 rhs)
{
    // 12-limb CIOS with 32-bit low products and mulhi. The fixed loop shape
    // matches the public 6x64 path while preserving R=2^384 exactly.
    uint temporary[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 12; ++rhs_index) {
        uint carry = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 12; ++lhs_index) {
            temporary[lhs_index] = fp32_multiply_accumulate(
                lhs.limbs[lhs_index],
                rhs.limbs[rhs_index],
                temporary[lhs_index],
                carry);
        }
        temporary[12] = fp32_add_carry_word(temporary[12], carry);
        temporary[13] = carry;

        carry = 0;
        const uint reduction =
            temporary[0] * MIDNIGHT_BLS12_381_FP32_INV;
        (void)fp32_multiply_accumulate(
            reduction,
            MIDNIGHT_BLS12_381_FP32_MODULUS[0],
            temporary[0],
            carry);
#pragma unroll
        for (uint index = 1; index < 12; ++index) {
            temporary[index - 1] = fp32_multiply_accumulate(
                reduction,
                MIDNIGHT_BLS12_381_FP32_MODULUS[index],
                temporary[index],
                carry);
        }
        temporary[11] = fp32_add_carry_word(temporary[12], carry);
        temporary[12] = temporary[13] + carry;
    }

    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = temporary[index];
    }
    return fp32_reduce_once(result);
}

inline Bls12_381Fp32 fp32_montgomery_mul_wide(
    Bls12_381Fp32 lhs,
    Bls12_381Fp32 rhs)
{
    // Sibling 12-limb CIOS experiment. Each 32x32 product and accumulated
    // limb/carry are represented exactly in one ulong before splitting at bit
    // 32. The field representation and fixed final reduction are unchanged.
    uint temporary[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 12; ++rhs_index) {
        ulong carry = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 12; ++lhs_index) {
            const ulong accumulated =
                static_cast<ulong>(lhs.limbs[lhs_index]) *
                    static_cast<ulong>(rhs.limbs[rhs_index]) +
                static_cast<ulong>(temporary[lhs_index]) + carry;
            temporary[lhs_index] = static_cast<uint>(accumulated);
            carry = accumulated >> 32;
        }
        ulong top = static_cast<ulong>(temporary[12]) + carry;
        temporary[12] = static_cast<uint>(top);
        temporary[13] = static_cast<uint>(top >> 32);

        const uint reduction =
            temporary[0] * MIDNIGHT_BLS12_381_FP32_INV;
        ulong accumulated =
            static_cast<ulong>(reduction) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary[0]);
        carry = accumulated >> 32;
#pragma unroll
        for (uint index = 1; index < 12; ++index) {
            accumulated =
                static_cast<ulong>(reduction) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary[index]) + carry;
            temporary[index - 1] = static_cast<uint>(accumulated);
            carry = accumulated >> 32;
        }
        top = static_cast<ulong>(temporary[12]) + carry;
        temporary[11] = static_cast<uint>(top);
        temporary[12] = temporary[13] + static_cast<uint>(top >> 32);
    }

    Bls12_381Fp32 result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = temporary[index];
    }
    return fp32_reduce_once(result);
}

inline void fp32_montgomery_mul_wide_2(
    Bls12_381Fp32 lhs0,
    Bls12_381Fp32 rhs0,
    Bls12_381Fp32 lhs1,
    Bls12_381Fp32 rhs1,
    thread Bls12_381Fp32& output0,
    thread Bls12_381Fp32& output1)
{
    // Explicitly alternate two independent CIOS states at every limb so this
    // diagnostic measures available integer ILP rather than relying on the
    // compiler to schedule across two complete inlined multiplications.
    uint temporary0[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    uint temporary1[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 12; ++rhs_index) {
        ulong carry0 = 0;
        ulong carry1 = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 12; ++lhs_index) {
            const ulong accumulated0 =
                static_cast<ulong>(lhs0.limbs[lhs_index]) *
                    static_cast<ulong>(rhs0.limbs[rhs_index]) +
                static_cast<ulong>(temporary0[lhs_index]) + carry0;
            const ulong accumulated1 =
                static_cast<ulong>(lhs1.limbs[lhs_index]) *
                    static_cast<ulong>(rhs1.limbs[rhs_index]) +
                static_cast<ulong>(temporary1[lhs_index]) + carry1;
            temporary0[lhs_index] = static_cast<uint>(accumulated0);
            temporary1[lhs_index] = static_cast<uint>(accumulated1);
            carry0 = accumulated0 >> 32;
            carry1 = accumulated1 >> 32;
        }
        ulong top0 = static_cast<ulong>(temporary0[12]) + carry0;
        ulong top1 = static_cast<ulong>(temporary1[12]) + carry1;
        temporary0[12] = static_cast<uint>(top0);
        temporary1[12] = static_cast<uint>(top1);
        temporary0[13] = static_cast<uint>(top0 >> 32);
        temporary1[13] = static_cast<uint>(top1 >> 32);

        const uint reduction0 =
            temporary0[0] * MIDNIGHT_BLS12_381_FP32_INV;
        const uint reduction1 =
            temporary1[0] * MIDNIGHT_BLS12_381_FP32_INV;
        ulong accumulated0 =
            static_cast<ulong>(reduction0) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary0[0]);
        ulong accumulated1 =
            static_cast<ulong>(reduction1) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary1[0]);
        carry0 = accumulated0 >> 32;
        carry1 = accumulated1 >> 32;
#pragma unroll
        for (uint index = 1; index < 12; ++index) {
            accumulated0 =
                static_cast<ulong>(reduction0) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary0[index]) + carry0;
            accumulated1 =
                static_cast<ulong>(reduction1) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary1[index]) + carry1;
            temporary0[index - 1] = static_cast<uint>(accumulated0);
            temporary1[index - 1] = static_cast<uint>(accumulated1);
            carry0 = accumulated0 >> 32;
            carry1 = accumulated1 >> 32;
        }
        top0 = static_cast<ulong>(temporary0[12]) + carry0;
        top1 = static_cast<ulong>(temporary1[12]) + carry1;
        temporary0[11] = static_cast<uint>(top0);
        temporary1[11] = static_cast<uint>(top1);
        temporary0[12] = temporary0[13] + static_cast<uint>(top0 >> 32);
        temporary1[12] = temporary1[13] + static_cast<uint>(top1 >> 32);
    }

    Bls12_381Fp32 result0;
    Bls12_381Fp32 result1;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result0.limbs[index] = temporary0[index];
        result1.limbs[index] = temporary1[index];
    }
    output0 = fp32_reduce_once(result0);
    output1 = fp32_reduce_once(result1);
}

inline void fp32_montgomery_mul_wide_3(
    Bls12_381Fp32 lhs0,
    Bls12_381Fp32 rhs0,
    Bls12_381Fp32 lhs1,
    Bls12_381Fp32 rhs1,
    Bls12_381Fp32 lhs2,
    Bls12_381Fp32 rhs2,
    thread Bls12_381Fp32& output0,
    thread Bls12_381Fp32& output1,
    thread Bls12_381Fp32& output2)
{
    uint temporary0[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    uint temporary1[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    uint temporary2[14] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 12; ++rhs_index) {
        ulong carry0 = 0;
        ulong carry1 = 0;
        ulong carry2 = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 12; ++lhs_index) {
            const ulong accumulated0 =
                static_cast<ulong>(lhs0.limbs[lhs_index]) *
                    static_cast<ulong>(rhs0.limbs[rhs_index]) +
                static_cast<ulong>(temporary0[lhs_index]) + carry0;
            const ulong accumulated1 =
                static_cast<ulong>(lhs1.limbs[lhs_index]) *
                    static_cast<ulong>(rhs1.limbs[rhs_index]) +
                static_cast<ulong>(temporary1[lhs_index]) + carry1;
            const ulong accumulated2 =
                static_cast<ulong>(lhs2.limbs[lhs_index]) *
                    static_cast<ulong>(rhs2.limbs[rhs_index]) +
                static_cast<ulong>(temporary2[lhs_index]) + carry2;
            temporary0[lhs_index] = static_cast<uint>(accumulated0);
            temporary1[lhs_index] = static_cast<uint>(accumulated1);
            temporary2[lhs_index] = static_cast<uint>(accumulated2);
            carry0 = accumulated0 >> 32;
            carry1 = accumulated1 >> 32;
            carry2 = accumulated2 >> 32;
        }
        ulong top0 = static_cast<ulong>(temporary0[12]) + carry0;
        ulong top1 = static_cast<ulong>(temporary1[12]) + carry1;
        ulong top2 = static_cast<ulong>(temporary2[12]) + carry2;
        temporary0[12] = static_cast<uint>(top0);
        temporary1[12] = static_cast<uint>(top1);
        temporary2[12] = static_cast<uint>(top2);
        temporary0[13] = static_cast<uint>(top0 >> 32);
        temporary1[13] = static_cast<uint>(top1 >> 32);
        temporary2[13] = static_cast<uint>(top2 >> 32);

        const uint reduction0 =
            temporary0[0] * MIDNIGHT_BLS12_381_FP32_INV;
        const uint reduction1 =
            temporary1[0] * MIDNIGHT_BLS12_381_FP32_INV;
        const uint reduction2 =
            temporary2[0] * MIDNIGHT_BLS12_381_FP32_INV;
        ulong accumulated0 =
            static_cast<ulong>(reduction0) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary0[0]);
        ulong accumulated1 =
            static_cast<ulong>(reduction1) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary1[0]);
        ulong accumulated2 =
            static_cast<ulong>(reduction2) *
                static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[0]) +
            static_cast<ulong>(temporary2[0]);
        carry0 = accumulated0 >> 32;
        carry1 = accumulated1 >> 32;
        carry2 = accumulated2 >> 32;
#pragma unroll
        for (uint index = 1; index < 12; ++index) {
            accumulated0 =
                static_cast<ulong>(reduction0) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary0[index]) + carry0;
            accumulated1 =
                static_cast<ulong>(reduction1) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary1[index]) + carry1;
            accumulated2 =
                static_cast<ulong>(reduction2) *
                    static_cast<ulong>(MIDNIGHT_BLS12_381_FP32_MODULUS[index]) +
                static_cast<ulong>(temporary2[index]) + carry2;
            temporary0[index - 1] = static_cast<uint>(accumulated0);
            temporary1[index - 1] = static_cast<uint>(accumulated1);
            temporary2[index - 1] = static_cast<uint>(accumulated2);
            carry0 = accumulated0 >> 32;
            carry1 = accumulated1 >> 32;
            carry2 = accumulated2 >> 32;
        }
        top0 = static_cast<ulong>(temporary0[12]) + carry0;
        top1 = static_cast<ulong>(temporary1[12]) + carry1;
        top2 = static_cast<ulong>(temporary2[12]) + carry2;
        temporary0[11] = static_cast<uint>(top0);
        temporary1[11] = static_cast<uint>(top1);
        temporary2[11] = static_cast<uint>(top2);
        temporary0[12] = temporary0[13] + static_cast<uint>(top0 >> 32);
        temporary1[12] = temporary1[13] + static_cast<uint>(top1 >> 32);
        temporary2[12] = temporary2[13] + static_cast<uint>(top2 >> 32);
    }

    Bls12_381Fp32 result0;
    Bls12_381Fp32 result1;
    Bls12_381Fp32 result2;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result0.limbs[index] = temporary0[index];
        result1.limbs[index] = temporary1[index];
        result2.limbs[index] = temporary2[index];
    }
    output0 = fp32_reduce_once(result0);
    output1 = fp32_reduce_once(result1);
    output2 = fp32_reduce_once(result2);
}

kernel void midnight_bls12_381_fp32_from_public(
    device const Bls12_381Fp* input [[buffer(0)]],
    device Bls12_381Fp32* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_from_public(input[index]);
    }
}

kernel void midnight_bls12_381_fp32_to_public(
    device const Bls12_381Fp32* input [[buffer(0)]],
    device Bls12_381Fp* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_to_public(input[index]);
    }
}

kernel void midnight_bls12_381_fp32_add(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_to_public(fp32_add(
            fp32_from_public(lhs[index]), fp32_from_public(rhs[index])));
    }
}

kernel void midnight_bls12_381_fp32_sub(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_to_public(fp32_sub(
            fp32_from_public(lhs[index]), fp32_from_public(rhs[index])));
    }
}

kernel void midnight_bls12_381_fp32_mul(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_to_public(fp32_montgomery_mul(
            fp32_from_public(lhs[index]), fp32_from_public(rhs[index])));
    }
}

kernel void midnight_bls12_381_fp32_square(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    (void)rhs;
    if (index < length) {
        const Bls12_381Fp32 value = fp32_from_public(lhs[index]);
        output[index] =
            fp32_to_public(fp32_montgomery_mul(value, value));
    }
}

kernel void midnight_bls12_381_fp32_wide_mul(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp32_to_public(fp32_montgomery_mul_wide(
            fp32_from_public(lhs[index]), fp32_from_public(rhs[index])));
    }
}

kernel void midnight_bls12_381_fp32_wide_square(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    (void)rhs;
    if (index < length) {
        const Bls12_381Fp32 value = fp32_from_public(lhs[index]);
        output[index] =
            fp32_to_public(fp32_montgomery_mul_wide(value, value));
    }
}

kernel void midnight_bls12_381_fp32_mul_chain(
    device const Bls12_381Fp32* lhs [[buffer(0)]],
    device const Bls12_381Fp32* rhs [[buffer(1)]],
    device Bls12_381Fp32* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& iterations [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        Bls12_381Fp32 accumulator = lhs[index];
        const Bls12_381Fp32 multiplier = rhs[index];
        for (uint iteration = 0; iteration < iterations; ++iteration) {
            accumulator = fp32_montgomery_mul(accumulator, multiplier);
        }
        output[index] = accumulator;
    }
}

kernel void midnight_bls12_381_fp32_wide_mul_chain(
    device const Bls12_381Fp32* lhs [[buffer(0)]],
    device const Bls12_381Fp32* rhs [[buffer(1)]],
    device Bls12_381Fp32* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& iterations [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        Bls12_381Fp32 accumulator = lhs[index];
        const Bls12_381Fp32 multiplier = rhs[index];
        for (uint iteration = 0; iteration < iterations; ++iteration) {
            accumulator =
                fp32_montgomery_mul_wide(accumulator, multiplier);
        }
        output[index] = accumulator;
    }
}

kernel void midnight_bls12_381_fp32_wide_mul_chain_2(
    device const Bls12_381Fp32* lhs [[buffer(0)]],
    device const Bls12_381Fp32* rhs [[buffer(1)]],
    device Bls12_381Fp32* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& iterations [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        Bls12_381Fp32 accumulator0 = lhs[index];
        Bls12_381Fp32 accumulator1 = rhs[index];
        const Bls12_381Fp32 multiplier0 = rhs[index];
        const Bls12_381Fp32 multiplier1 = lhs[index];
        for (uint iteration = 0; iteration < iterations; ++iteration) {
            Bls12_381Fp32 next0;
            Bls12_381Fp32 next1;
            fp32_montgomery_mul_wide_2(
                accumulator0,
                multiplier0,
                accumulator1,
                multiplier1,
                next0,
                next1);
            accumulator0 = next0;
            accumulator1 = next1;
        }
        // Both chains contribute to observable output, preventing dead-code
        // removal while keeping the timed loop representative of independent
        // group-formula multiplications.
        output[index] = fp32_add(accumulator0, accumulator1);
    }
}

kernel void midnight_bls12_381_fp32_wide_mul_chain_3(
    device const Bls12_381Fp32* lhs [[buffer(0)]],
    device const Bls12_381Fp32* rhs [[buffer(1)]],
    device Bls12_381Fp32* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& iterations [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        Bls12_381Fp32 accumulator0 = lhs[index];
        Bls12_381Fp32 accumulator1 = rhs[index];
        Bls12_381Fp32 accumulator2 = fp32_add(lhs[index], rhs[index]);
        const Bls12_381Fp32 multiplier0 = rhs[index];
        const Bls12_381Fp32 multiplier1 = lhs[index];
        const Bls12_381Fp32 multiplier2 = fp32_sub(lhs[index], rhs[index]);
        for (uint iteration = 0; iteration < iterations; ++iteration) {
            Bls12_381Fp32 next0;
            Bls12_381Fp32 next1;
            Bls12_381Fp32 next2;
            fp32_montgomery_mul_wide_3(
                accumulator0,
                multiplier0,
                accumulator1,
                multiplier1,
                accumulator2,
                multiplier2,
                next0,
                next1,
                next2);
            accumulator0 = next0;
            accumulator1 = next1;
            accumulator2 = next2;
        }
        output[index] =
            fp32_add(fp32_add(accumulator0, accumulator1), accumulator2);
    }
}

#endif
