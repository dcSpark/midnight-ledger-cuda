// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// The public point layout remains six little-endian u64 Montgomery limbs per
// coordinate. Metal views the same 48 bytes as twelve u32 limbs and executes
// the generated named-register FIOS schedule validated by the standalone
// field gate. This avoids private limb arrays in the multiplication hot path.

#ifndef BLS12_381_METAL_MSM_FIELD
#define BLS12_381_METAL_MSM_FIELD

#include <metal_stdlib>
#include "bls12_381_u32_constants.metal"
#include "bls12_381_msm_constants.metal"

using namespace metal;

#ifndef BLS12_381_U32_FIELD_ELEMENT_TYPE
#define BLS12_381_U32_FIELD_ELEMENT_TYPE
struct U32FieldElement {
    uint limbs[12];
};
#endif

#include "bls12_381_u32_named_fios.metal"

// Eight-byte alignment preserves the Rust/BLST six-u64 point ABI while the
// limbs themselves use native u32 arithmetic on little-endian Apple GPUs.
struct alignas(8) Bls12381MetalMsmFp {
    uint limbs[12];
};

struct Bls12381MetalMsmAffine {
    Bls12381MetalMsmFp x;
    Bls12381MetalMsmFp y;
};

struct Bls12381MetalMsmJacobian {
    Bls12381MetalMsmFp x;
    Bls12381MetalMsmFp y;
    Bls12381MetalMsmFp z;
};

// Complete formulas represent (x/z, y/z); identity is (0, R, 0).
struct Bls12381MetalMsmHomogeneous {
    Bls12381MetalMsmFp x;
    Bls12381MetalMsmFp y;
    Bls12381MetalMsmFp z;
};

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_zero()
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = 0U;
    }
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_one()
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const ulong limb = BLS12_381_METAL_MSM_FP_R[index >> 1U];
        result.limbs[index] = (index & 1U) == 0U
            ? static_cast<uint>(limb)
            : static_cast<uint>(limb >> 32);
    }
    return result;
}

inline __attribute__((always_inline))
ulong bls12_381_msm_fp_exact_zero_mask(Bls12381MetalMsmFp value)
{
    uint zero_aggregate = 0U;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        zero_aggregate |= value.limbs[index];
    }
    return 0UL - static_cast<ulong>(zero_aggregate == 0U);
}

inline __attribute__((always_inline))
ulong bls12_381_msm_fp_zero_mask(Bls12381MetalMsmFp value)
{
    uint zero_aggregate = 0U;
    uint modulus_aggregate = 0U;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        zero_aggregate |= value.limbs[index];
        modulus_aggregate |= value.limbs[index] ^
            BLS12_381_U32_FP_MODULUS[index];
    }
    return 0UL - static_cast<ulong>(
        zero_aggregate == 0U || modulus_aggregate == 0U);
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_select(
    Bls12381MetalMsmFp when_clear,
    Bls12381MetalMsmFp when_set,
    ulong mask)
{
    Bls12381MetalMsmFp result;
    const uint word_mask = static_cast<uint>(mask);
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] =
            (when_clear.limbs[index] & ~word_mask) |
            (when_set.limbs[index] & word_mask);
    }
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_reduce_once(Bls12381MetalMsmFp value)
{
    Bls12381MetalMsmFp difference;
    uint borrow = 0U;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const uint first = value.limbs[index] - BLS12_381_U32_FP_MODULUS[index];
        const uint first_borrow = static_cast<uint>(
            value.limbs[index] < BLS12_381_U32_FP_MODULUS[index]);
        difference.limbs[index] = first - borrow;
        const uint second_borrow = static_cast<uint>(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    return bls12_381_msm_fp_select(
        value, difference, 0UL - static_cast<ulong>(borrow ^ 1U));
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_add(
    Bls12381MetalMsmFp lhs,
    Bls12381MetalMsmFp rhs)
{
    Bls12381MetalMsmFp result;
    ulong carry = 0UL;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const ulong sum = static_cast<ulong>(lhs.limbs[index]) +
            static_cast<ulong>(rhs.limbs[index]) + carry;
        result.limbs[index] = static_cast<uint>(sum);
        carry = sum >> 32;
    }
    Bls12381MetalMsmFp difference;
    uint borrow = 0U;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const uint first = result.limbs[index] -
            BLS12_381_U32_FP_MODULUS_X2[index];
        const uint first_borrow = static_cast<uint>(
            result.limbs[index] < BLS12_381_U32_FP_MODULUS_X2[index]);
        difference.limbs[index] = first - borrow;
        const uint second_borrow = static_cast<uint>(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    return bls12_381_msm_fp_select(
        result, difference, 0UL - static_cast<ulong>(borrow ^ 1U));
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_sub(
    Bls12381MetalMsmFp lhs,
    Bls12381MetalMsmFp rhs)
{
    Bls12381MetalMsmFp result;
    uint borrow = 0U;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const uint first = lhs.limbs[index] - rhs.limbs[index];
        const uint first_borrow = static_cast<uint>(lhs.limbs[index] < rhs.limbs[index]);
        result.limbs[index] = first - borrow;
        const uint second_borrow = static_cast<uint>(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    const uint add_modulus = 0U - borrow;
    ulong carry = 0UL;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        const ulong sum = static_cast<ulong>(result.limbs[index]) +
            static_cast<ulong>(BLS12_381_U32_FP_MODULUS_X2[index] & add_modulus) + carry;
        result.limbs[index] = static_cast<uint>(sum);
        carry = sum >> 32;
    }
    return result;
}

Bls12381MetalMsmFp bls12_381_msm_fp_negate(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_sub(bls12_381_msm_fp_zero(), value);
}

inline __attribute__((always_inline))
U32FieldElement bls12_381_msm_fp_to_u32_abi(Bls12381MetalMsmFp value)
{
    U32FieldElement result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = value.limbs[index];
    }
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_from_u32_abi(U32FieldElement value)
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0; index < 12; ++index) {
        result.limbs[index] = value.limbs[index];
    }
    return result;
}

inline Bls12381MetalMsmFp bls12_381_msm_fp_compact_reduce_once(
    Bls12381MetalMsmFp value)
{
    Bls12381MetalMsmFp subtracted;
    uint borrow = 0U;
#pragma unroll
    for (uint index = 0U; index < 12U; ++index) {
        const uint first = value.limbs[index] -
            BLS12_381_U32_FP_MODULUS[index];
        const uint first_borrow = static_cast<uint>(
            value.limbs[index] < BLS12_381_U32_FP_MODULUS[index]);
        subtracted.limbs[index] = first - borrow;
        const uint second_borrow = static_cast<uint>(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    const uint use_subtracted = 0U - (borrow ^ 1U);
#pragma unroll
    for (uint index = 0U; index < 12U; ++index) {
        value.limbs[index] =
            (value.limbs[index] & ~use_subtracted) |
            (subtracted.limbs[index] & use_subtracted);
    }
    return value;
}

__attribute__((noinline))
Bls12381MetalMsmFp bls12_381_msm_fp_mul(
    Bls12381MetalMsmFp lhs,
    Bls12381MetalMsmFp rhs)
{
    uint temporary[14] = {
        0U, 0U, 0U, 0U, 0U, 0U, 0U,
        0U, 0U, 0U, 0U, 0U, 0U, 0U,
    };

#pragma unroll
    for (uint rhs_index = 0U; rhs_index < 12U; ++rhs_index) {
        ulong carry = 0UL;
#pragma unroll
        for (uint lhs_index = 0U; lhs_index < 12U; ++lhs_index) {
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

        const uint reduction = temporary[0] * BLS12_381_U32_FP_INV;
        ulong accumulated =
            static_cast<ulong>(reduction) *
                static_cast<ulong>(BLS12_381_U32_FP_MODULUS[0]) +
            static_cast<ulong>(temporary[0]);
        carry = accumulated >> 32;
#pragma unroll
        for (uint index = 1U; index < 12U; ++index) {
            accumulated =
                static_cast<ulong>(reduction) *
                    static_cast<ulong>(BLS12_381_U32_FP_MODULUS[index]) +
                static_cast<ulong>(temporary[index]) + carry;
            temporary[index - 1U] = static_cast<uint>(accumulated);
            carry = accumulated >> 32;
        }
        top = static_cast<ulong>(temporary[12]) + carry;
        temporary[11] = static_cast<uint>(top);
        temporary[12] = temporary[13] + static_cast<uint>(top >> 32);
    }

    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0U; index < 12U; ++index) {
        result.limbs[index] = temporary[index];
    }
    // Inputs and outputs use a redundant [0, 2p) Montgomery representation.
    // Since 4p < 2^384, Montgomery reduction of two such inputs is again
    // strictly below 2p. Deferring the canonical subtraction removes one
    // twelve-limb pass from every point-arithmetic multiplication.
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_square(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_mul(value, value);
}

#endif
