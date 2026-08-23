// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_FIELD_METAL
#define MIDNIGHT_BLS12_381_FIELD_METAL

#include <metal_stdlib>
#include "bls12_381_constants.metal"

using namespace metal;

struct Bls12_381Fr {
    ulong limbs[4];
};

struct Bls12_381Fp {
    ulong limbs[6];
};

struct Bls12_381G1Affine {
    Bls12_381Fp x;
    Bls12_381Fp y;
};

struct Bls12_381G1Projective {
    Bls12_381Fp x;
    Bls12_381Fp y;
    Bls12_381Fp z;
};

// Metal-only complete-formula representation. Unlike the public
// BLST-compatible Jacobian ABI above, this homogeneous point represents the
// affine coordinates (x / z, y / z). Its canonical identity is (0, 1, 0).
struct Bls12_381G1Homogeneous {
    Bls12_381Fp x;
    Bls12_381Fp y;
    Bls12_381Fp z;
};

inline ulong add_with_carry(ulong lhs, ulong rhs, thread ulong& carry)
{
    const ulong first = lhs + rhs;
    const ulong first_carry = static_cast<ulong>(first < lhs);
    const ulong result = first + carry;
    const ulong second_carry = static_cast<ulong>(result < first);
    carry = first_carry | second_carry;
    return result;
}

inline ulong subtract_with_borrow(ulong lhs, ulong rhs, thread ulong& borrow)
{
    const ulong first = lhs - rhs;
    const ulong first_borrow = static_cast<ulong>(lhs < rhs);
    const ulong result = first - borrow;
    const ulong second_borrow = static_cast<ulong>(first < borrow);
    borrow = first_borrow | second_borrow;
    return result;
}

inline ulong select_with_mask(ulong when_clear, ulong when_set, ulong mask)
{
    return (when_clear & ~mask) | (when_set & mask);
}

inline Bls12_381Fr fr_reduce_once(Bls12_381Fr value)
{
    Bls12_381Fr subtracted;
    ulong borrow = 0;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        subtracted.limbs[index] = subtract_with_borrow(
            value.limbs[index],
            MIDNIGHT_BLS12_381_FR_MODULUS[index],
            borrow);
    }

    // borrow=0 selects value-r; borrow=1 retains value. The arithmetic mask
    // keeps reduction on one fixed limb schedule for every input value.
    const ulong use_subtracted = 0UL - (borrow ^ 1UL);
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        value.limbs[index] = select_with_mask(
            value.limbs[index], subtracted.limbs[index], use_subtracted);
    }
    return value;
}

inline Bls12_381Fr fr_add(Bls12_381Fr lhs, Bls12_381Fr rhs)
{
    Bls12_381Fr result;
    ulong carry = 0;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = add_with_carry(lhs.limbs[index], rhs.limbs[index], carry);
    }

    // Reduced Fr values satisfy lhs + rhs < 2r < 2^256, so the final carry
    // is zero. One fixed subtract-and-select restores the reduced invariant.
    return fr_reduce_once(result);
}

inline Bls12_381Fr fr_sub(Bls12_381Fr lhs, Bls12_381Fr rhs)
{
    Bls12_381Fr result;
    ulong borrow = 0;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = subtract_with_borrow(lhs.limbs[index], rhs.limbs[index], borrow);
    }

    const ulong add_modulus = 0UL - borrow;
    ulong carry = 0;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = add_with_carry(
            result.limbs[index],
            MIDNIGHT_BLS12_381_FR_MODULUS[index] & add_modulus,
            carry);
    }
    return result;
}

inline ulong multiply_accumulate(
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

inline ulong add_carry_word(ulong value, thread ulong& carry)
{
    const ulong result = value + carry;
    carry = static_cast<ulong>(result < value);
    return result;
}

inline Bls12_381Fr fr_montgomery_mul(Bls12_381Fr lhs, Bls12_381Fr rhs)
{
    // Coarsely integrated operand scanning (CIOS). Inputs and output use the
    // existing BLST/sppark Montgomery representation with R = 2^256.
    ulong temporary[6] = {0, 0, 0, 0, 0, 0};

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 4; ++rhs_index) {
        ulong carry = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 4; ++lhs_index) {
            temporary[lhs_index] = multiply_accumulate(
                lhs.limbs[lhs_index],
                rhs.limbs[rhs_index],
                temporary[lhs_index],
                carry);
        }
        temporary[4] = add_carry_word(temporary[4], carry);
        temporary[5] = carry;

        carry = 0;
        const ulong reduction = temporary[0] * MIDNIGHT_BLS12_381_FR_INV;
        (void)multiply_accumulate(
            reduction,
            MIDNIGHT_BLS12_381_FR_MODULUS[0],
            temporary[0],
            carry);
#pragma unroll
        for (uint index = 1; index < 4; ++index) {
            temporary[index - 1] = multiply_accumulate(
                reduction,
                MIDNIGHT_BLS12_381_FR_MODULUS[index],
                temporary[index],
                carry);
        }
        temporary[3] = add_carry_word(temporary[4], carry);
        temporary[4] = temporary[5] + carry;
    }

    Bls12_381Fr result;
#pragma unroll
    for (uint index = 0; index < 4; ++index) {
        result.limbs[index] = temporary[index];
    }
    return fr_reduce_once(result);
}

inline Bls12_381Fp fp_reduce_once(Bls12_381Fp value)
{
    Bls12_381Fp subtracted;
    ulong borrow = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        subtracted.limbs[index] = subtract_with_borrow(
            value.limbs[index],
            MIDNIGHT_BLS12_381_FP_MODULUS[index],
            borrow);
    }

    const ulong use_subtracted = 0UL - (borrow ^ 1UL);
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        value.limbs[index] = select_with_mask(
            value.limbs[index], subtracted.limbs[index], use_subtracted);
    }
    return value;
}

inline Bls12_381Fp fp_add(Bls12_381Fp lhs, Bls12_381Fp rhs)
{
    Bls12_381Fp result;
    ulong carry = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = add_with_carry(lhs.limbs[index], rhs.limbs[index], carry);
    }

    // Reduced Fp values satisfy lhs + rhs < 2p < 2^384.
    return fp_reduce_once(result);
}

inline Bls12_381Fp fp_sub(Bls12_381Fp lhs, Bls12_381Fp rhs)
{
    Bls12_381Fp result;
    ulong borrow = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = subtract_with_borrow(lhs.limbs[index], rhs.limbs[index], borrow);
    }

    const ulong add_modulus = 0UL - borrow;
    ulong carry = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = add_with_carry(
            result.limbs[index],
            MIDNIGHT_BLS12_381_FP_MODULUS[index] & add_modulus,
            carry);
    }
    return result;
}

inline Bls12_381Fp fp_montgomery_mul(Bls12_381Fp lhs, Bls12_381Fp rhs)
{
    // CIOS Montgomery multiplication with R = 2^384. The low-level carry and
    // multiply-accumulate helpers are shared with Fr; field widths stay explicit.
    ulong temporary[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (uint rhs_index = 0; rhs_index < 6; ++rhs_index) {
        ulong carry = 0;
#pragma unroll
        for (uint lhs_index = 0; lhs_index < 6; ++lhs_index) {
            temporary[lhs_index] = multiply_accumulate(
                lhs.limbs[lhs_index],
                rhs.limbs[rhs_index],
                temporary[lhs_index],
                carry);
        }
        temporary[6] = add_carry_word(temporary[6], carry);
        temporary[7] = carry;

        carry = 0;
        const ulong reduction = temporary[0] * MIDNIGHT_BLS12_381_FP_INV;
        (void)multiply_accumulate(
            reduction,
            MIDNIGHT_BLS12_381_FP_MODULUS[0],
            temporary[0],
            carry);
#pragma unroll
        for (uint index = 1; index < 6; ++index) {
            temporary[index - 1] = multiply_accumulate(
                reduction,
                MIDNIGHT_BLS12_381_FP_MODULUS[index],
                temporary[index],
                carry);
        }
        temporary[5] = add_carry_word(temporary[6], carry);
        temporary[6] = temporary[7] + carry;
    }

    Bls12_381Fp result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = temporary[index];
    }
    return fp_reduce_once(result);
}

inline ulong midnight_bls12_381_layout_abi_mix(ulong fingerprint, ulong value)
{
    // FNV-1a over fixed-width layout words. This is a pinned diagnostic
    // fingerprint, not a cryptographic digest.
    return (fingerprint ^ value) * 0x00000100000001b3UL;
}

kernel void midnight_bls12_381_abi_probe(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }

    output[0] = MIDNIGHT_BLS12_381_ABI_VERSION;
    output[1] = sizeof(Bls12_381Fr);
    output[2] = alignof(Bls12_381Fr);
    output[3] = __builtin_offsetof(Bls12_381Fr, limbs);
    output[4] = sizeof(Bls12_381Fp);
    output[5] = alignof(Bls12_381Fp);
    output[6] = __builtin_offsetof(Bls12_381Fp, limbs);
    output[7] = sizeof(Bls12_381G1Affine);
    output[8] = alignof(Bls12_381G1Affine);
    output[9] = __builtin_offsetof(Bls12_381G1Affine, x);
    output[10] = __builtin_offsetof(Bls12_381G1Affine, y);
    output[11] = sizeof(Bls12_381G1Projective);
    output[12] = alignof(Bls12_381G1Projective);
    output[13] = __builtin_offsetof(Bls12_381G1Projective, x);
    output[14] = __builtin_offsetof(Bls12_381G1Projective, y);
    output[15] = __builtin_offsetof(Bls12_381G1Projective, z);
    output[16] = sizeof(Bls12_381G1Homogeneous);
    output[17] = alignof(Bls12_381G1Homogeneous);
    output[18] = __builtin_offsetof(Bls12_381G1Homogeneous, x);
    output[19] = __builtin_offsetof(Bls12_381G1Homogeneous, y);
    output[20] = __builtin_offsetof(Bls12_381G1Homogeneous, z);

    ulong layout_fingerprint = 0xcbf29ce484222325UL;
    for (uint index = 0; index < 21; ++index) {
        layout_fingerprint = midnight_bls12_381_layout_abi_mix(
            layout_fingerprint, output[index]);
    }
    output[21] = layout_fingerprint;
    output[22] = MIDNIGHT_BLS12_381_CONSTANTS_FINGERPRINT;
}

kernel void midnight_bls12_381_fr64_add(
    device const Bls12_381Fr* lhs [[buffer(0)]],
    device const Bls12_381Fr* rhs [[buffer(1)]],
    device Bls12_381Fr* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fr_add(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fr64_sub(
    device const Bls12_381Fr* lhs [[buffer(0)]],
    device const Bls12_381Fr* rhs [[buffer(1)]],
    device Bls12_381Fr* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fr_sub(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fr64_mul(
    device const Bls12_381Fr* lhs [[buffer(0)]],
    device const Bls12_381Fr* rhs [[buffer(1)]],
    device Bls12_381Fr* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fr_montgomery_mul(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fr64_square(
    device const Bls12_381Fr* lhs [[buffer(0)]],
    device const Bls12_381Fr* rhs [[buffer(1)]],
    device Bls12_381Fr* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    (void)rhs;
    if (index < length) {
        output[index] = fr_montgomery_mul(lhs[index], lhs[index]);
    }
}

kernel void midnight_bls12_381_fp64_add(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp_add(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fp64_sub(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp_sub(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fp64_mul(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = fp_montgomery_mul(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_fp64_square(
    device const Bls12_381Fp* lhs [[buffer(0)]],
    device const Bls12_381Fp* rhs [[buffer(1)]],
    device Bls12_381Fp* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    (void)rhs;
    if (index < length) {
        output[index] = fp_montgomery_mul(lhs[index], lhs[index]);
    }
}

#endif
