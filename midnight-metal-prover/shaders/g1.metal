// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef MIDNIGHT_BLS12_381_G1_METAL
#define MIDNIGHT_BLS12_381_G1_METAL

#include "field.metal"

// BLS12-381 G1 is y^2 = x^3 + 4, so a=0 and 3b=12. Arithmetic uses the
// complete homogeneous formulas from Renes-Costello-Batina Algorithms 7, 8,
// and 9
// (ePrint 2015/1060), following the vendored midnight-curves 0.3.1 and BLST
// 0.3.16 operation order. The public buffer ABI remains BLST-compatible
// Jacobian; conversion to the Metal-only homogeneous representation happens at
// the kernel boundary. See PROVENANCE.md for exact revisions and licenses.
//
// The formula schedule has no exceptional-case branches. The underlying field
// reduction code has not received a side-channel audit, so these remain
// correctness probes and are not routed into proof execution or MSM yet.

inline Bls12_381Fp fp_zero()
{
    Bls12_381Fp result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = 0;
    }
    return result;
}

inline Bls12_381Fp fp_montgomery_one()
{
    Bls12_381Fp result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] = MIDNIGHT_BLS12_381_FP_R[index];
    }
    return result;
}

inline bool fp_is_zero(Bls12_381Fp value)
{
    ulong aggregate = 0;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        aggregate |= value.limbs[index];
    }
    return aggregate == 0;
}

inline Bls12_381Fp fp_square(Bls12_381Fp value)
{
    return fp_montgomery_mul(value, value);
}

inline Bls12_381Fp fp_double(Bls12_381Fp value)
{
    return fp_add(value, value);
}

inline Bls12_381Fp fp_triple(Bls12_381Fp value)
{
    return fp_add(fp_double(value), value);
}

inline Bls12_381Fp fp_times_eight(Bls12_381Fp value)
{
    return fp_double(fp_double(fp_double(value)));
}

inline Bls12_381Fp fp_mul_by_3b(Bls12_381Fp value)
{
    // b=4, hence 3b=12. Four reduced additions are cheaper than a general
    // six-limb Montgomery multiplication by the curve constant.
    const Bls12_381Fp four = fp_double(fp_double(value));
    const Bls12_381Fp eight = fp_double(four);
    return fp_add(eight, four);
}

inline ulong fp_zero_mask(Bls12_381Fp value)
{
    return 0UL - static_cast<ulong>(fp_is_zero(value));
}

inline Bls12_381Fp fp_select_mask(
    Bls12_381Fp when_clear,
    Bls12_381Fp when_set,
    ulong mask)
{
    Bls12_381Fp result;
#pragma unroll
    for (uint index = 0; index < 6; ++index) {
        result.limbs[index] =
            (when_clear.limbs[index] & ~mask) | (when_set.limbs[index] & mask);
    }
    return result;
}

inline Bls12_381Fp fp_inverse(Bls12_381Fp value)
{
    // Fermat inversion, value^(p-2), uses a fixed public exponent and loop
    // shape. This is intentionally a simple correctness implementation; a
    // separately reviewed addition chain or batch inversion can replace it.
    Bls12_381Fp result = fp_montgomery_one();
    for (int bit = 383; bit >= 0; --bit) {
        result = fp_square(result);
        const uint word_index = static_cast<uint>(bit) >> 6;
        const uint bit_index = static_cast<uint>(bit) & 63;
        if (((MIDNIGHT_BLS12_381_FP_MODULUS_MINUS_TWO[word_index] >> bit_index) & 1UL) != 0) {
            result = fp_montgomery_mul(result, value);
        }
    }
    return result;
}

inline Bls12_381G1Homogeneous g1_homogeneous_identity()
{
    Bls12_381G1Homogeneous result;
    result.x = fp_zero();
    result.y = fp_montgomery_one();
    result.z = fp_zero();
    return result;
}

inline ulong g1_affine_identity_mask(Bls12_381G1Affine point)
{
    return fp_zero_mask(point.x) & fp_zero_mask(point.y);
}

inline Bls12_381G1Homogeneous g1_canonicalize_homogeneous_infinity(
    Bls12_381G1Homogeneous point)
{
    const ulong infinity = fp_zero_mask(point.z);
    point.x = fp_select_mask(point.x, fp_zero(), infinity);
    point.y = fp_select_mask(point.y, fp_montgomery_one(), infinity);
    point.z = fp_select_mask(point.z, fp_zero(), infinity);
    return point;
}

inline Bls12_381G1Homogeneous g1_jacobian_to_homogeneous(
    Bls12_381G1Projective point)
{
    // BLST ec_ops.h POINT_TO_PROJECTIVE_IMPL. For finite points:
    // (Xj, Yj, Zj) -> (Xj*Zj, Yj, Zj^3). The Y mask turns every
    // noncanonical Jacobian Z=0 representation into homogeneous (0,1,0).
    const ulong infinity = fp_zero_mask(point.z);
    const Bls12_381Fp z_squared = fp_square(point.z);
    Bls12_381G1Homogeneous result;
    result.x = fp_montgomery_mul(point.x, point.z);
    result.y = fp_select_mask(point.y, fp_montgomery_one(), infinity);
    result.z = fp_montgomery_mul(z_squared, point.z);
    return g1_canonicalize_homogeneous_infinity(result);
}

inline Bls12_381G1Projective g1_homogeneous_to_jacobian(
    Bls12_381G1Homogeneous point)
{
    // BLST ec_ops.h POINT_PROJ_TO_JACOBIAN_IMPL. The explicit final mask
    // preserves the public all-zero Jacobian infinity encoding even if a future
    // caller supplies a noncanonical homogeneous Z=0 point.
    const ulong infinity = fp_zero_mask(point.z);
    const Bls12_381Fp z_squared = fp_square(point.z);
    Bls12_381G1Projective result;
    result.x = fp_montgomery_mul(point.x, point.z);
    result.y = fp_montgomery_mul(point.y, z_squared);
    result.z = point.z;
    result.x = fp_select_mask(result.x, fp_zero(), infinity);
    result.y = fp_select_mask(result.y, fp_zero(), infinity);
    result.z = fp_select_mask(result.z, fp_zero(), infinity);
    return result;
}

inline Bls12_381G1Homogeneous g1_complete_add_homogeneous(
    Bls12_381G1Homogeneous lhs,
    Bls12_381G1Homogeneous rhs)
{
    // Renes-Costello-Batina Algorithm 7, a=0. Exactly 12 general field
    // multiplications and 27 field additions/subtractions when multiplication
    // by 3b=12 uses fp_mul_by_3b's four-addition chain.
    Bls12_381Fp t0 = fp_montgomery_mul(lhs.x, rhs.x);       // 1
    Bls12_381Fp t1 = fp_montgomery_mul(lhs.y, rhs.y);       // 2
    Bls12_381Fp t2 = fp_montgomery_mul(lhs.z, rhs.z);       // 3
    Bls12_381Fp t3 = fp_add(lhs.x, lhs.y);                  // 4
    Bls12_381Fp t4 = fp_add(rhs.x, rhs.y);                  // 5
    t3 = fp_montgomery_mul(t3, t4);                         // 6
    t4 = fp_add(t0, t1);                                   // 7
    t3 = fp_sub(t3, t4);                                   // 8
    t4 = fp_add(lhs.y, lhs.z);                              // 9
    Bls12_381Fp t5 = fp_add(rhs.y, rhs.z);                  // 10
    t4 = fp_montgomery_mul(t4, t5);                         // 11
    t5 = fp_add(t1, t2);                                   // 12
    t4 = fp_sub(t4, t5);                                   // 13
    t5 = fp_add(lhs.x, lhs.z);                              // 14
    Bls12_381Fp y3 = fp_add(rhs.x, rhs.z);                  // 15
    t5 = fp_montgomery_mul(t5, y3);                         // 16
    y3 = fp_add(t0, t2);                                   // 17
    y3 = fp_sub(t5, y3);                                   // 18
    t0 = fp_triple(t0);                                    // 19-20
    t2 = fp_mul_by_3b(t2);                                 // 21
    Bls12_381Fp z3 = fp_add(t1, t2);                       // 22
    t1 = fp_sub(t1, t2);                                   // 23
    y3 = fp_mul_by_3b(y3);                                 // 24
    Bls12_381Fp x3 = fp_montgomery_mul(t4, y3);             // 25
    t2 = fp_montgomery_mul(t3, t1);                        // 26
    x3 = fp_sub(t2, x3);                                   // 27
    y3 = fp_montgomery_mul(y3, t0);                        // 28
    t1 = fp_montgomery_mul(t1, z3);                        // 29
    y3 = fp_add(t1, y3);                                   // 30
    t0 = fp_montgomery_mul(t0, t3);                        // 31
    z3 = fp_montgomery_mul(z3, t4);                        // 32
    z3 = fp_add(z3, t0);                                   // 33

    Bls12_381G1Homogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return g1_canonicalize_homogeneous_infinity(result);
}

inline Bls12_381G1Homogeneous g1_complete_mixed_add_homogeneous(
    Bls12_381G1Homogeneous lhs,
    Bls12_381G1Affine rhs)
{
    // Renes-Costello-Batina Algorithm 8, a=0, with BLST's all-zero affine
    // infinity mask. Exactly 11 general field multiplications and 21 field
    // additions/subtractions when multiplication by 3b=12 uses
    // fp_mul_by_3b's four-addition chain.
    const ulong rhs_infinity = g1_affine_identity_mask(rhs);
    Bls12_381Fp t0 = fp_montgomery_mul(lhs.x, rhs.x);       // 1
    Bls12_381Fp t1 = fp_montgomery_mul(lhs.y, rhs.y);       // 2
    Bls12_381Fp t3 = fp_add(lhs.x, lhs.y);                  // 3
    Bls12_381Fp t4 = fp_add(rhs.x, rhs.y);                  // 4
    t3 = fp_montgomery_mul(t3, t4);                         // 5
    t4 = fp_add(t0, t1);                                   // 6
    t3 = fp_sub(t3, t4);                                   // 7
    t4 = fp_montgomery_mul(rhs.y, lhs.z);                   // 8
    t4 = fp_add(t4, lhs.y);                                // 9
    Bls12_381Fp y3 = fp_montgomery_mul(rhs.x, lhs.z);       // 10
    y3 = fp_add(y3, lhs.x);                                // 11
    t0 = fp_triple(t0);                                    // 12-13
    Bls12_381Fp t2 = fp_mul_by_3b(lhs.z);                   // 14
    Bls12_381Fp z3 = fp_add(t1, t2);                       // 15
    t1 = fp_sub(t1, t2);                                   // 16
    y3 = fp_mul_by_3b(y3);                                 // 17
    Bls12_381Fp x3 = fp_montgomery_mul(t4, y3);             // 18
    t2 = fp_montgomery_mul(t3, t1);                        // 19
    x3 = fp_sub(t2, x3);                                   // 20
    y3 = fp_montgomery_mul(y3, t0);                        // 21
    t1 = fp_montgomery_mul(t1, z3);                        // 22
    y3 = fp_add(t1, y3);                                   // 23
    t0 = fp_montgomery_mul(t0, t3);                        // 24
    z3 = fp_montgomery_mul(z3, t4);                        // 25
    z3 = fp_add(z3, t0);                                   // 26

    Bls12_381G1Homogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result = g1_canonicalize_homogeneous_infinity(result);
    result.x = fp_select_mask(result.x, lhs.x, rhs_infinity);
    result.y = fp_select_mask(result.y, lhs.y, rhs_infinity);
    result.z = fp_select_mask(result.z, lhs.z, rhs_infinity);
    return result;
}

inline Bls12_381G1Homogeneous g1_complete_double_homogeneous(
    Bls12_381G1Homogeneous point)
{
    // Renes-Costello-Batina Algorithm 9, a=0, reordered as in BLST so the
    // destination may alias the input. Exactly 6M + 2S + 13A.
    Bls12_381Fp t0 = fp_square(point.y);                    // 1
    Bls12_381Fp t1 = fp_montgomery_mul(point.y, point.z);   // 5
    Bls12_381Fp t2 = fp_square(point.z);                    // 6
    const Bls12_381Fp t3 = fp_montgomery_mul(point.x, point.y); // 16
    Bls12_381Fp z3 = fp_times_eight(t0);                    // 2-4
    t2 = fp_mul_by_3b(t2);                                 // 7
    Bls12_381Fp x3 = fp_montgomery_mul(t2, z3);             // 8
    Bls12_381Fp y3 = fp_add(t0, t2);                       // 9
    z3 = fp_montgomery_mul(t1, z3);                        // 10
    t2 = fp_triple(t2);                                    // 11-12
    t0 = fp_sub(t0, t2);                                   // 13
    y3 = fp_montgomery_mul(t0, y3);                        // 14
    y3 = fp_add(x3, y3);                                   // 15
    x3 = fp_montgomery_mul(t0, t3);                        // 17
    x3 = fp_double(x3);                                    // 18

    Bls12_381G1Homogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return g1_canonicalize_homogeneous_infinity(result);
}

inline Bls12_381G1Projective g1_add(
    Bls12_381G1Projective lhs,
    Bls12_381G1Projective rhs)
{
    return g1_homogeneous_to_jacobian(g1_complete_add_homogeneous(
        g1_jacobian_to_homogeneous(lhs),
        g1_jacobian_to_homogeneous(rhs)));
}

inline Bls12_381G1Projective g1_double(Bls12_381G1Projective point)
{
    return g1_homogeneous_to_jacobian(g1_complete_double_homogeneous(
        g1_jacobian_to_homogeneous(point)));
}

inline Bls12_381G1Projective g1_mixed_add(
    Bls12_381G1Projective lhs,
    Bls12_381G1Affine rhs)
{
    return g1_homogeneous_to_jacobian(g1_complete_mixed_add_homogeneous(
        g1_jacobian_to_homogeneous(lhs), rhs));
}

inline Bls12_381G1Affine g1_to_affine(Bls12_381G1Projective point)
{
    const ulong infinity = fp_zero_mask(point.z);
    const Bls12_381Fp z_inverse = fp_inverse(point.z);
    const Bls12_381Fp z_inverse_squared = fp_square(z_inverse);
    const Bls12_381Fp z_inverse_cubed =
        fp_montgomery_mul(z_inverse_squared, z_inverse);
    Bls12_381G1Affine result;
    result.x = fp_montgomery_mul(point.x, z_inverse_squared);
    result.y = fp_montgomery_mul(point.y, z_inverse_cubed);
    result.x = fp_select_mask(result.x, fp_zero(), infinity);
    result.y = fp_select_mask(result.y, fp_zero(), infinity);
    return result;
}

kernel void midnight_bls12_381_g1_add(
    device const Bls12_381G1Projective* lhs [[buffer(0)]],
    device const Bls12_381G1Projective* rhs [[buffer(1)]],
    device Bls12_381G1Projective* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_add(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_g1_homogeneous_fill_identity(
    device Bls12_381G1Homogeneous* output [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_homogeneous_identity();
    }
}

kernel void midnight_bls12_381_g1_jacobian_to_homogeneous(
    device const Bls12_381G1Projective* input [[buffer(0)]],
    device Bls12_381G1Homogeneous* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_jacobian_to_homogeneous(input[index]);
    }
}

kernel void midnight_bls12_381_g1_homogeneous_to_jacobian(
    device const Bls12_381G1Homogeneous* input [[buffer(0)]],
    device Bls12_381G1Projective* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_homogeneous_to_jacobian(input[index]);
    }
}

kernel void midnight_bls12_381_g1_complete_add_chain(
    device const Bls12_381G1Homogeneous* lhs [[buffer(0)]],
    device const Bls12_381G1Homogeneous* rhs [[buffer(1)]],
    device Bls12_381G1Homogeneous* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& additions [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        Bls12_381G1Homogeneous accumulator = lhs[index];
        const Bls12_381G1Homogeneous addend = rhs[index];
        for (uint iteration = 0; iteration < additions; ++iteration) {
            accumulator = g1_complete_add_homogeneous(accumulator, addend);
        }
        output[index] = accumulator;
    }
}

kernel void midnight_bls12_381_g1_double(
    device const Bls12_381G1Projective* input [[buffer(0)]],
    device Bls12_381G1Projective* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_double(input[index]);
    }
}

kernel void midnight_bls12_381_g1_mixed_add(
    device const Bls12_381G1Projective* lhs [[buffer(0)]],
    device const Bls12_381G1Affine* rhs [[buffer(1)]],
    device Bls12_381G1Projective* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_mixed_add(lhs[index], rhs[index]);
    }
}

kernel void midnight_bls12_381_g1_to_affine(
    device const Bls12_381G1Projective* input [[buffer(0)]],
    device Bls12_381G1Affine* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < length) {
        output[index] = g1_to_affine(input[index]);
    }
}

#endif
