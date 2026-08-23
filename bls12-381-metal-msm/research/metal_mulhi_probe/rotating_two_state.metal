// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

struct BenchParams {
    uint thread_count;
    uint iterations;
};

inline uint2 step_mulhi(uint2 state)
{
    const uint low = state.x * state.y;
    const uint high = metal::mulhi(state.x, state.y);
    const uint next = (low ^ metal::rotate(high, 13u)) + 0x243f6a88u;
    return uint2(state.y, next);
}

inline uint2 step_widened(uint2 state)
{
    const ulong product = ulong(state.x) * ulong(state.y);
    const uint low = uint(product);
    const uint high = uint(product >> 32);
    const uint next = (low ^ metal::rotate(high, 13u)) + 0x243f6a88u;
    return uint2(state.y, next);
}

kernel void bench_rotating_mulhi(
    device const uint2 *seed [[buffer(0)]],
    device uint2 *output [[buffer(1)]],
    constant BenchParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.thread_count) {
        return;
    }
    uint2 state = seed[gid];
    for (uint i = 0; i < params.iterations; ++i) {
        state = step_mulhi(state);
    }
    output[gid] = state;
}

kernel void bench_rotating_widened(
    device const uint2 *seed [[buffer(0)]],
    device uint2 *output [[buffer(1)]],
    constant BenchParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.thread_count) {
        return;
    }
    uint2 state = seed[gid];
    for (uint i = 0; i < params.iterations; ++i) {
        state = step_widened(state);
    }
    output[gid] = state;
}
