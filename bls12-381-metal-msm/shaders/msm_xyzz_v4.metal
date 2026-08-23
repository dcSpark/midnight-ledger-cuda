// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Branch-completed XYZZ arithmetic for the GPU-routed reducer.  The
// representation stores affine x = X/ZZ and y = Y/ZZZ, avoiding repeated Z
// squaring in mixed and full additions.  The schedules are the standard
// madd/add/dbl-2008-s formulas specialized to BLS12-381's a=0 curve.

#ifndef BLS12_381_METAL_MSM_XYZZ_V4
#define BLS12_381_METAL_MSM_XYZZ_V4

#include "msm_parallel_jacobian_v2.metal"

struct Bls12381MetalMsmXyzzV4 {
    Bls12381MetalMsmFp x;
    Bls12381MetalMsmFp y;
    Bls12381MetalMsmFp zz;
    Bls12381MetalMsmFp zzz;
};

static_assert(sizeof(Bls12381MetalMsmXyzzV4) == 192, "XYZZ v4 ABI");

constant uint BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4 = 0U;
constant uint BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_AFFINE_V4 = 1U;
constant uint BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_PROJECTIVE_V4 = 2U;

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_identity_v4()
{
    Bls12381MetalMsmXyzzV4 result;
    result.x = bls12_381_msm_fp_zero();
    result.y = bls12_381_msm_fp_one();
    result.zz = bls12_381_msm_fp_zero();
    result.zzz = bls12_381_msm_fp_zero();
    return result;
}

inline __attribute__((always_inline))
bool bls12_381_msm_xyzz_is_identity_v4(Bls12381MetalMsmXyzzV4 point)
{
    return bls12_381_msm_jacobian_fp_is_exact_zero_v2(point.zz) &&
        bls12_381_msm_jacobian_fp_is_exact_zero_v2(point.zzz);
}

inline __attribute__((always_inline))
Bls12381MetalMsmFp bls12_381_msm_fp_simd_shuffle_down_v4(
    Bls12381MetalMsmFp value,
    ushort delta)
{
    Bls12381MetalMsmFp result;
#pragma unroll
    for (uint index = 0U; index < 12U; ++index) {
        result.limbs[index] = simd_shuffle_down(value.limbs[index], delta);
    }
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_simd_shuffle_down_v4(
    Bls12381MetalMsmXyzzV4 point,
    ushort delta)
{
    Bls12381MetalMsmXyzzV4 result;
    result.x = bls12_381_msm_fp_simd_shuffle_down_v4(point.x, delta);
    result.y = bls12_381_msm_fp_simd_shuffle_down_v4(point.y, delta);
    result.zz = bls12_381_msm_fp_simd_shuffle_down_v4(point.zz, delta);
    result.zzz = bls12_381_msm_fp_simd_shuffle_down_v4(point.zzz, delta);
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_from_affine_v4(
    Bls12381MetalMsmAffine point)
{
    if (bls12_381_msm_g1_affine_identity_mask(point) != 0UL) {
        return bls12_381_msm_xyzz_identity_v4();
    }
    Bls12381MetalMsmXyzzV4 result;
    result.x = point.x;
    result.y = point.y;
    result.zz = bls12_381_msm_fp_one();
    result.zzz = bls12_381_msm_fp_one();
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_from_nonidentity_affine_v4(
    Bls12381MetalMsmAffine point)
{
    Bls12381MetalMsmXyzzV4 result;
    result.x = point.x;
    result.y = point.y;
    result.zz = bls12_381_msm_fp_one();
    result.zzz = bls12_381_msm_fp_one();
    return result;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_double_v4(
    Bls12381MetalMsmXyzzV4 point)
{
    if (bls12_381_msm_xyzz_is_identity_v4(point) ||
        bls12_381_msm_jacobian_fp_is_exact_zero_v2(point.y)) {
        return bls12_381_msm_xyzz_identity_v4();
    }

    Bls12381MetalMsmFp u = bls12_381_msm_fp_double(point.y);
    Bls12381MetalMsmFp v = bls12_381_msm_fp_square(u);
    u = bls12_381_msm_fp_mul(u, v);
    Bls12381MetalMsmFp s = bls12_381_msm_fp_mul(point.x, v);
    point.zz = bls12_381_msm_fp_mul(point.zz, v);
    point.y = bls12_381_msm_fp_mul(point.y, u);
    point.zzz = bls12_381_msm_fp_mul(point.zzz, u);
    u = bls12_381_msm_fp_triple(bls12_381_msm_fp_square(point.x));
    point.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_square(u), bls12_381_msm_fp_double(s));
    s = bls12_381_msm_fp_mul(u, bls12_381_msm_fp_sub(s, point.x));
    point.y = bls12_381_msm_fp_sub(s, point.y);
    return point;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_mixed_add_nonidentity_v4(
    Bls12381MetalMsmXyzzV4 lhs,
    Bls12381MetalMsmAffine rhs);

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_mixed_add_v4(
    Bls12381MetalMsmXyzzV4 lhs,
    Bls12381MetalMsmAffine rhs)
{
    if (bls12_381_msm_g1_affine_identity_mask(rhs) != 0UL) {
        return lhs;
    }
    if (bls12_381_msm_xyzz_is_identity_v4(lhs)) {
        return bls12_381_msm_xyzz_from_affine_v4(rhs);
    }
    return bls12_381_msm_xyzz_mixed_add_nonidentity_v4(lhs, rhs);
}

// Prepared fixed bases are validated as non-identity before upload, and a
// nonzero power-of-two shift cannot turn a prime-order G1 point into identity.
// Retain every lhs and exceptional-case check while avoiding a redundant scan
// of both affine coordinates for each routed record.
inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_mixed_add_nonidentity_v4(
    Bls12381MetalMsmXyzzV4 lhs,
    Bls12381MetalMsmAffine rhs)
{
    if (bls12_381_msm_xyzz_is_identity_v4(lhs)) {
        return bls12_381_msm_xyzz_from_nonidentity_affine_v4(rhs);
    }

    Bls12381MetalMsmFp r = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(rhs.y, lhs.zzz), lhs.y);
    Bls12381MetalMsmFp p = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(rhs.x, lhs.zz), lhs.x);
    if (bls12_381_msm_jacobian_fp_is_zero_v2(p)) {
        return bls12_381_msm_jacobian_fp_is_zero_v2(r)
            ? bls12_381_msm_xyzz_double_v4(lhs)
            : bls12_381_msm_xyzz_identity_v4();
    }

    Bls12381MetalMsmFp pp = bls12_381_msm_fp_square(p);
    p = bls12_381_msm_fp_mul(p, pp);
    lhs.y = bls12_381_msm_fp_mul(lhs.y, p);
    lhs.zz = bls12_381_msm_fp_mul(lhs.zz, pp);
    lhs.zzz = bls12_381_msm_fp_mul(lhs.zzz, p);
    pp = bls12_381_msm_fp_mul(lhs.x, pp);
    lhs.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(bls12_381_msm_fp_square(r), p),
        bls12_381_msm_fp_double(pp));
    pp = bls12_381_msm_fp_mul(r, bls12_381_msm_fp_sub(pp, lhs.x));
    lhs.y = bls12_381_msm_fp_sub(pp, lhs.y);
    return lhs;
}

// Fixed-base reducers track both identity and the one-point affine state. The
// latter removes four field multiplications from the lane's first addition;
// later additions avoid rescanning all 24 ZZ/ZZZ limbs.
inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_mixed_add_tracked_v4(
    Bls12381MetalMsmXyzzV4 lhs,
    Bls12381MetalMsmAffine rhs,
    thread uint& accumulator_state)
{
    if (accumulator_state ==
        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4) {
        accumulator_state =
            BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_AFFINE_V4;
        return bls12_381_msm_xyzz_from_nonidentity_affine_v4(rhs);
    }

    const bool affine_lhs = accumulator_state ==
        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_AFFINE_V4;
    Bls12381MetalMsmFp r = affine_lhs
        ? bls12_381_msm_fp_sub(rhs.y, lhs.y)
        : bls12_381_msm_fp_sub(
            bls12_381_msm_fp_mul(rhs.y, lhs.zzz), lhs.y);
    Bls12381MetalMsmFp p = affine_lhs
        ? bls12_381_msm_fp_sub(rhs.x, lhs.x)
        : bls12_381_msm_fp_sub(
            bls12_381_msm_fp_mul(rhs.x, lhs.zz), lhs.x);
    if (bls12_381_msm_jacobian_fp_is_zero_v2(p)) {
        if (bls12_381_msm_jacobian_fp_is_zero_v2(r)) {
            if (affine_lhs) {
                if (bls12_381_msm_jacobian_fp_is_exact_zero_v2(lhs.y)) {
                    accumulator_state =
                        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4;
                    return bls12_381_msm_xyzz_identity_v4();
                }
                Bls12381MetalMsmFp u = bls12_381_msm_fp_double(lhs.y);
                const Bls12381MetalMsmFp v = bls12_381_msm_fp_square(u);
                u = bls12_381_msm_fp_mul(u, v);
                Bls12381MetalMsmFp s = bls12_381_msm_fp_mul(lhs.x, v);
                lhs.zz = v;
                lhs.y = bls12_381_msm_fp_mul(lhs.y, u);
                lhs.zzz = u;
                u = bls12_381_msm_fp_triple(
                    bls12_381_msm_fp_square(lhs.x));
                lhs.x = bls12_381_msm_fp_sub(
                    bls12_381_msm_fp_square(u),
                    bls12_381_msm_fp_double(s));
                s = bls12_381_msm_fp_mul(
                    u, bls12_381_msm_fp_sub(s, lhs.x));
                lhs.y = bls12_381_msm_fp_sub(s, lhs.y);
                accumulator_state =
                    BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_PROJECTIVE_V4;
                return lhs;
            }
            lhs = bls12_381_msm_xyzz_double_v4(lhs);
            accumulator_state = bls12_381_msm_xyzz_is_identity_v4(lhs)
                ? BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4
                : BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_PROJECTIVE_V4;
            return lhs;
        }
        accumulator_state =
            BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_IDENTITY_V4;
        return bls12_381_msm_xyzz_identity_v4();
    }

    Bls12381MetalMsmFp pp = bls12_381_msm_fp_square(p);
    p = bls12_381_msm_fp_mul(p, pp);
    lhs.y = bls12_381_msm_fp_mul(lhs.y, p);
    lhs.zz = affine_lhs ? pp : bls12_381_msm_fp_mul(lhs.zz, pp);
    lhs.zzz = affine_lhs ? p : bls12_381_msm_fp_mul(lhs.zzz, p);
    pp = bls12_381_msm_fp_mul(lhs.x, pp);
    lhs.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(bls12_381_msm_fp_square(r), p),
        bls12_381_msm_fp_double(pp));
    pp = bls12_381_msm_fp_mul(r, bls12_381_msm_fp_sub(pp, lhs.x));
    lhs.y = bls12_381_msm_fp_sub(pp, lhs.y);
    accumulator_state =
        BLS12_381_METAL_MSM_XYZZ_ACCUMULATOR_PROJECTIVE_V4;
    return lhs;
}

inline __attribute__((always_inline))
Bls12381MetalMsmXyzzV4 bls12_381_msm_xyzz_add_v4(
    Bls12381MetalMsmXyzzV4 lhs,
    Bls12381MetalMsmXyzzV4 rhs)
{
    if (bls12_381_msm_xyzz_is_identity_v4(lhs)) {
        return rhs;
    }
    if (bls12_381_msm_xyzz_is_identity_v4(rhs)) {
        return lhs;
    }

    Bls12381MetalMsmFp u = bls12_381_msm_fp_mul(lhs.x, rhs.zz);
    Bls12381MetalMsmFp s = bls12_381_msm_fp_mul(lhs.y, rhs.zzz);
    Bls12381MetalMsmFp p = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(rhs.x, lhs.zz), u);
    Bls12381MetalMsmFp r = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_mul(rhs.y, lhs.zzz), s);
    if (bls12_381_msm_jacobian_fp_is_zero_v2(p)) {
        return bls12_381_msm_jacobian_fp_is_zero_v2(r)
            ? bls12_381_msm_xyzz_double_v4(lhs)
            : bls12_381_msm_xyzz_identity_v4();
    }

    Bls12381MetalMsmFp pp = bls12_381_msm_fp_square(p);
    p = bls12_381_msm_fp_mul(p, pp);
    lhs.zz = bls12_381_msm_fp_mul(lhs.zz, rhs.zz);
    lhs.zz = bls12_381_msm_fp_mul(lhs.zz, pp);
    lhs.zzz = bls12_381_msm_fp_mul(lhs.zzz, rhs.zzz);
    lhs.zzz = bls12_381_msm_fp_mul(lhs.zzz, p);
    lhs.y = bls12_381_msm_fp_mul(s, p);
    pp = bls12_381_msm_fp_mul(u, pp);
    lhs.x = bls12_381_msm_fp_sub(
        bls12_381_msm_fp_sub(bls12_381_msm_fp_square(r), p),
        bls12_381_msm_fp_double(pp));
    pp = bls12_381_msm_fp_mul(r, bls12_381_msm_fp_sub(pp, lhs.x));
    lhs.y = bls12_381_msm_fp_sub(pp, lhs.y);
    return lhs;
}

inline __attribute__((always_inline))
Bls12381MetalMsmJacobian bls12_381_msm_xyzz_to_jacobian_v4(
    Bls12381MetalMsmXyzzV4 point)
{
    if (bls12_381_msm_xyzz_is_identity_v4(point)) {
        return bls12_381_msm_jacobian_identity_v2();
    }
    Bls12381MetalMsmJacobian result;
    result.x = bls12_381_msm_fp_mul(point.x, point.zz);
    result.y = bls12_381_msm_fp_mul(point.y, point.zzz);
    result.z = point.zz;
    result.x = bls12_381_msm_fp_reduce_once(result.x);
    result.y = bls12_381_msm_fp_reduce_once(result.y);
    result.z = bls12_381_msm_fp_reduce_once(result.z);
    return result;
}

#endif
