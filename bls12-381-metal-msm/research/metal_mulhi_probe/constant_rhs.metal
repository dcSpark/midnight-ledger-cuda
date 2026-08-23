// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

struct BenchParams {
    uint thread_count;
    uint iterations;
};

kernel void check_mulhi(
    device const uint2 *input [[buffer(0)]],
    device uint2 *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    const uint a = input[gid].x;
    const uint b = input[gid].y;
    output[gid] = uint2(a * b, metal::mulhi(a, b));
}

kernel void check_widened(
    device const uint2 *input [[buffer(0)]],
    device uint2 *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    const uint a = input[gid].x;
    const uint b = input[gid].y;
    const ulong product = ulong(a) * ulong(b);
    output[gid] = uint2(uint(product), uint(product >> 32));
}

inline uint step_mulhi(uint x)
{
    constexpr uint multiplier = 0x9e3779b1u;
    constexpr uint additive = 0x243f6a88u;
    const uint low = x * multiplier;
    const uint high = metal::mulhi(x, multiplier);
    return (low ^ metal::rotate(high, 13u)) + additive;
}

inline uint step_widened(uint x)
{
    constexpr ulong multiplier = 0x9e3779b1ul;
    constexpr uint additive = 0x243f6a88u;
    const ulong product = ulong(x) * multiplier;
    const uint low = uint(product);
    const uint high = uint(product >> 32);
    return (low ^ metal::rotate(high, 13u)) + additive;
}

kernel void bench_mulhi(
    device const uint *seed [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant BenchParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.thread_count) {
        return;
    }
    uint state = seed[gid];
    for (uint i = 0; i < params.iterations; ++i) {
        state = step_mulhi(state);
    }
    output[gid] = state;
}

kernel void bench_widened(
    device const uint *seed [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant BenchParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.thread_count) {
        return;
    }
    uint state = seed[gid];
    for (uint i = 0; i < params.iterations; ++i) {
        state = step_widened(state);
    }
    output[gid] = state;
}
