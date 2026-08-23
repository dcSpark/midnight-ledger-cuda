// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct CarryNormalizedGateInput {
    float4 lhs;
    float4 rhs;
};

struct CarryNormalizedGateOutput {
    float4 accumulators[8];
};

struct CarryNormalizedGateParameters {
    uint length;
    uint iterations;
    uint reserved0;
    uint reserved1;
};

constant ulong CARRY_NORMALIZED_GATE_ABI_FINGERPRINT = 0x56b27f5b4ad9e13dUL;
constant ulong U32_LOW_LIMB_GATE_ABI_FINGERPRINT = 0xa51330b19d22e047UL;

kernel void bls12_381_metal_msm_product_issue_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(CarryNormalizedGateInput);
    output[2] = alignof(CarryNormalizedGateInput);
    output[3] = __builtin_offsetof(CarryNormalizedGateInput, rhs);
    output[4] = sizeof(CarryNormalizedGateOutput);
    output[5] = sizeof(CarryNormalizedGateParameters);
    output[6] = 32UL;
    output[7] = CARRY_NORMALIZED_GATE_ABI_FINGERPRINT;
}

kernel void bls12_381_metal_msm_precise_fp32_product_issue(
    device const CarryNormalizedGateInput* inputs [[buffer(0)]],
    device CarryNormalizedGateOutput* outputs [[buffer(1)]],
    constant CarryNormalizedGateParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.length) {
        return;
    }

    const float4 lhs = inputs[index].lhs;
    const float4 rhs = inputs[index].rhs;
    float4 accumulators[8] = {
        lhs + rhs,
        lhs - rhs,
        lhs.yzwx + rhs.wxyz,
        lhs.zwxy - rhs.yzwx,
        lhs.wxyz + rhs.zwxy,
        lhs.yxwz - rhs.zwxy,
        lhs.xwzy + rhs.yxwz,
        lhs.zyxw - rhs.xwzy,
    };

#pragma clang loop unroll(disable)
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulators[0] = fma(lhs, rhs, accumulators[0]);
        accumulators[1] = fma(lhs.yzwx, rhs.zwxy, accumulators[1]);
        accumulators[2] = fma(lhs.zwxy, rhs.wxyz, accumulators[2]);
        accumulators[3] = fma(lhs.wxyz, rhs.yzwx, accumulators[3]);
        accumulators[4] = fma(lhs.yxwz, rhs.xwzy, accumulators[4]);
        accumulators[5] = fma(lhs.xwzy, rhs.zyxw, accumulators[5]);
        accumulators[6] = fma(lhs.zyxw, rhs.yxwz, accumulators[6]);
        accumulators[7] = fma(lhs.wzyx, rhs.wzyx, accumulators[7]);
    }

#pragma unroll
    for (uint accumulator = 0; accumulator < 8; ++accumulator) {
        outputs[index].accumulators[accumulator] = accumulators[accumulator];
    }
}

struct U32LowLimbGateInput {
    uint4 lhs;
    uint4 rhs;
};

struct U32LowLimbGateOutput {
    uint4 accumulators[8];
};

kernel void bls12_381_metal_msm_u32_low_limb_product_issue_abi(
    device ulong* output [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) {
        return;
    }
    output[0] = 1UL;
    output[1] = sizeof(U32LowLimbGateInput);
    output[2] = alignof(U32LowLimbGateInput);
    output[3] = __builtin_offsetof(U32LowLimbGateInput, rhs);
    output[4] = sizeof(U32LowLimbGateOutput);
    output[5] = sizeof(CarryNormalizedGateParameters);
    output[6] = 32UL;
    output[7] = U32_LOW_LIMB_GATE_ABI_FINGERPRINT;
}

kernel void bls12_381_metal_msm_u32_low_limb_product_issue(
    device const U32LowLimbGateInput* inputs [[buffer(0)]],
    device U32LowLimbGateOutput* outputs [[buffer(1)]],
    constant CarryNormalizedGateParameters& parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= parameters.length) {
        return;
    }

    const uint4 lhs = inputs[index].lhs;
    const uint4 rhs = inputs[index].rhs;
    uint4 accumulators[8] = {
        lhs + rhs,
        lhs - rhs,
        lhs.yzwx + rhs.wxyz,
        lhs.zwxy - rhs.yzwx,
        lhs.wxyz + rhs.zwxy,
        lhs.yxwz - rhs.zwxy,
        lhs.xwzy + rhs.yxwz,
        lhs.zyxw - rhs.xwzy,
    };

#pragma clang loop unroll(disable)
    for (uint iteration = 0; iteration < parameters.iterations; ++iteration) {
        accumulators[0] = lhs * rhs + accumulators[0];
        accumulators[1] = lhs.yzwx * rhs.zwxy + accumulators[1];
        accumulators[2] = lhs.zwxy * rhs.wxyz + accumulators[2];
        accumulators[3] = lhs.wxyz * rhs.yzwx + accumulators[3];
        accumulators[4] = lhs.yxwz * rhs.xwzy + accumulators[4];
        accumulators[5] = lhs.xwzy * rhs.zyxw + accumulators[5];
        accumulators[6] = lhs.zyxw * rhs.yxwz + accumulators[6];
        accumulators[7] = lhs.wzyx * rhs + accumulators[7];
    }

#pragma unroll
    for (uint accumulator = 0; accumulator < 8; ++accumulator) {
        outputs[index].accumulators[accumulator] = accumulators[accumulator];
    }
}
