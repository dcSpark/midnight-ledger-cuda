// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Complete homogeneous BLS12-381 G1 formulas extracted from the validated
// midnight-metal-prover implementation. The schedules are RCB Algorithms 7,
// 8, and 9 (ePrint 2015/1060), following BLST 0.3.16's operation order.

#ifndef BLS12_381_METAL_MSM_G1
#define BLS12_381_METAL_MSM_G1

#include "msm_field.metal"

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_double(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_add(value, value);
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_triple(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_add(bls12_381_msm_fp_double(value), value);
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_times_eight(Bls12381MetalMsmFp value)
{
    return bls12_381_msm_fp_double(
        bls12_381_msm_fp_double(bls12_381_msm_fp_double(value)));
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_mul_by_3b(Bls12381MetalMsmFp value)
{
    // BLS12-381 G1 has b=4, so 3b=12.
    const Bls12381MetalMsmFp four =
        bls12_381_msm_fp_double(bls12_381_msm_fp_double(value));
    return bls12_381_msm_fp_add(bls12_381_msm_fp_double(four), four);
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_g1_identity()
{
    Bls12381MetalMsmHomogeneous result;
    result.x = bls12_381_msm_fp_zero();
    result.y = bls12_381_msm_fp_one();
    result.z = bls12_381_msm_fp_zero();
    return result;
}

inline __attribute__((always_inline))
ulong bls12_381_msm_g1_affine_identity_mask(Bls12381MetalMsmAffine point)
{
    return bls12_381_msm_fp_exact_zero_mask(point.x) &
        bls12_381_msm_fp_exact_zero_mask(point.y);
}

inline __attribute__((always_inline))
Bls12381MetalMsmAffine bls12_381_msm_g1_signed_affine(
    Bls12381MetalMsmAffine point,
    uint flags)
{
    const ulong negate_mask = 0UL - static_cast<ulong>(flags & 1U);
    point.y = bls12_381_msm_fp_select(
        point.y, bls12_381_msm_fp_negate(point.y), negate_mask);
    return point;
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_g1_canonicalize_identity(
    Bls12381MetalMsmHomogeneous point)
{
    const ulong identity = bls12_381_msm_fp_zero_mask(point.z);
    point.x = bls12_381_msm_fp_select(
        point.x, bls12_381_msm_fp_zero(), identity);
    point.y = bls12_381_msm_fp_select(
        point.y, bls12_381_msm_fp_one(), identity);
    point.z = bls12_381_msm_fp_select(
        point.z, bls12_381_msm_fp_zero(), identity);
    return point;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_g1_to_jacobian(
    Bls12381MetalMsmHomogeneous point)
{
    const ulong identity = bls12_381_msm_fp_zero_mask(point.z);
    const Bls12381MetalMsmFp z_squared = bls12_381_msm_fp_square(point.z);
    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_mul(point.x, point.z);
    result.y = bls12_381_msm_fp_mul(point.y, z_squared);
    result.z = point.z;
    result.x = bls12_381_msm_fp_select(
        result.x, bls12_381_msm_fp_zero(), identity);
    result.y = bls12_381_msm_fp_select(
        result.y, bls12_381_msm_fp_zero(), identity);
    result.z = bls12_381_msm_fp_select(
        result.z, bls12_381_msm_fp_zero(), identity);
    result.x = bls12_381_msm_fp_reduce_once(result.x);
    result.y = bls12_381_msm_fp_reduce_once(result.y);
    result.z = bls12_381_msm_fp_reduce_once(result.z);
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_g1_add(
    Bls12381MetalMsmHomogeneous lhs,
    Bls12381MetalMsmHomogeneous rhs)
{
    Bls12381MetalMsmFp t0 = bls12_381_msm_fp_mul(lhs.x, rhs.x);
    Bls12381MetalMsmFp t1 = bls12_381_msm_fp_mul(lhs.y, rhs.y);
    Bls12381MetalMsmFp t2 = bls12_381_msm_fp_mul(lhs.z, rhs.z);
    Bls12381MetalMsmFp t3 = bls12_381_msm_fp_add(lhs.x, lhs.y);
    Bls12381MetalMsmFp t4 = bls12_381_msm_fp_add(rhs.x, rhs.y);
    t3 = bls12_381_msm_fp_mul(t3, t4);
    t4 = bls12_381_msm_fp_add(t0, t1);
    t3 = bls12_381_msm_fp_sub(t3, t4);
    t4 = bls12_381_msm_fp_add(lhs.y, lhs.z);
    Bls12381MetalMsmFp t5 = bls12_381_msm_fp_add(rhs.y, rhs.z);
    t4 = bls12_381_msm_fp_mul(t4, t5);
    t5 = bls12_381_msm_fp_add(t1, t2);
    t4 = bls12_381_msm_fp_sub(t4, t5);
    t5 = bls12_381_msm_fp_add(lhs.x, lhs.z);
    Bls12381MetalMsmFp y3 = bls12_381_msm_fp_add(rhs.x, rhs.z);
    t5 = bls12_381_msm_fp_mul(t5, y3);
    y3 = bls12_381_msm_fp_add(t0, t2);
    y3 = bls12_381_msm_fp_sub(t5, y3);
    t0 = bls12_381_msm_fp_triple(t0);
    t2 = bls12_381_msm_fp_mul_by_3b(t2);
    Bls12381MetalMsmFp z3 = bls12_381_msm_fp_add(t1, t2);
    t1 = bls12_381_msm_fp_sub(t1, t2);
    y3 = bls12_381_msm_fp_mul_by_3b(y3);
    Bls12381MetalMsmFp x3 = bls12_381_msm_fp_mul(t4, y3);
    t2 = bls12_381_msm_fp_mul(t3, t1);
    x3 = bls12_381_msm_fp_sub(t2, x3);
    y3 = bls12_381_msm_fp_mul(y3, t0);
    t1 = bls12_381_msm_fp_mul(t1, z3);
    y3 = bls12_381_msm_fp_add(t1, y3);
    t0 = bls12_381_msm_fp_mul(t0, t3);
    z3 = bls12_381_msm_fp_mul(z3, t4);
    z3 = bls12_381_msm_fp_add(z3, t0);

    Bls12381MetalMsmHomogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return bls12_381_msm_g1_canonicalize_identity(result);
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_g1_mixed_add(
    Bls12381MetalMsmHomogeneous lhs,
    Bls12381MetalMsmAffine rhs)
{
    const ulong rhs_identity = bls12_381_msm_g1_affine_identity_mask(rhs);
    Bls12381MetalMsmFp t0 = bls12_381_msm_fp_mul(lhs.x, rhs.x);
    Bls12381MetalMsmFp t1 = bls12_381_msm_fp_mul(lhs.y, rhs.y);
    Bls12381MetalMsmFp t3 = bls12_381_msm_fp_add(lhs.x, lhs.y);
    Bls12381MetalMsmFp t4 = bls12_381_msm_fp_add(rhs.x, rhs.y);
    t3 = bls12_381_msm_fp_mul(t3, t4);
    t4 = bls12_381_msm_fp_add(t0, t1);
    t3 = bls12_381_msm_fp_sub(t3, t4);
    t4 = bls12_381_msm_fp_mul(rhs.y, lhs.z);
    t4 = bls12_381_msm_fp_add(t4, lhs.y);
    Bls12381MetalMsmFp y3 = bls12_381_msm_fp_mul(rhs.x, lhs.z);
    y3 = bls12_381_msm_fp_add(y3, lhs.x);
    t0 = bls12_381_msm_fp_triple(t0);
    Bls12381MetalMsmFp t2 = bls12_381_msm_fp_mul_by_3b(lhs.z);
    Bls12381MetalMsmFp z3 = bls12_381_msm_fp_add(t1, t2);
    t1 = bls12_381_msm_fp_sub(t1, t2);
    y3 = bls12_381_msm_fp_mul_by_3b(y3);
    Bls12381MetalMsmFp x3 = bls12_381_msm_fp_mul(t4, y3);
    t2 = bls12_381_msm_fp_mul(t3, t1);
    x3 = bls12_381_msm_fp_sub(t2, x3);
    y3 = bls12_381_msm_fp_mul(y3, t0);
    t1 = bls12_381_msm_fp_mul(t1, z3);
    y3 = bls12_381_msm_fp_add(t1, y3);
    t0 = bls12_381_msm_fp_mul(t0, t3);
    z3 = bls12_381_msm_fp_mul(z3, t4);
    z3 = bls12_381_msm_fp_add(z3, t0);

    Bls12381MetalMsmHomogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result = bls12_381_msm_g1_canonicalize_identity(result);
    result.x = bls12_381_msm_fp_select(result.x, lhs.x, rhs_identity);
    result.y = bls12_381_msm_fp_select(result.y, lhs.y, rhs_identity);
    result.z = bls12_381_msm_fp_select(result.z, lhs.z, rhs_identity);
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmHomogeneous bls12_381_msm_g1_double(
    Bls12381MetalMsmHomogeneous point)
{
    Bls12381MetalMsmFp t0 = bls12_381_msm_fp_square(point.y);
    Bls12381MetalMsmFp t1 = bls12_381_msm_fp_mul(point.y, point.z);
    Bls12381MetalMsmFp t2 = bls12_381_msm_fp_square(point.z);
    const Bls12381MetalMsmFp t3 = bls12_381_msm_fp_mul(point.x, point.y);
    Bls12381MetalMsmFp z3 = bls12_381_msm_fp_times_eight(t0);
    t2 = bls12_381_msm_fp_mul_by_3b(t2);
    Bls12381MetalMsmFp x3 = bls12_381_msm_fp_mul(t2, z3);
    Bls12381MetalMsmFp y3 = bls12_381_msm_fp_add(t0, t2);
    z3 = bls12_381_msm_fp_mul(t1, z3);
    t2 = bls12_381_msm_fp_triple(t2);
    t0 = bls12_381_msm_fp_sub(t0, t2);
    y3 = bls12_381_msm_fp_mul(t0, y3);
    y3 = bls12_381_msm_fp_add(x3, y3);
    x3 = bls12_381_msm_fp_mul(t0, t3);
    x3 = bls12_381_msm_fp_double(x3);

    Bls12381MetalMsmHomogeneous result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return bls12_381_msm_g1_canonicalize_identity(result);
}

#endif
