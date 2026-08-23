// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>

using namespace metal;

/// Deterministic adapter boundary used before proof-internal kernels are trusted.
kernel void midnight_proof_byte_digest(
    device const uchar* bytes [[buffer(0)]],
    constant ulong& length [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0) {
        return;
    }

    constexpr ulong fnv_prime = 0x100000001b3UL;
    uchar xor_byte = 0;
    ulong sum = 0;
    ulong rolling = 0;
    for (ulong index = 0; index < length; ++index) {
        const uchar byte = bytes[index];
        xor_byte ^= byte;
        sum += static_cast<ulong>(byte);
        rolling = rolling * fnv_prime + static_cast<ulong>(byte) + index;
    }

    output[0] = length;
    output[1] = static_cast<ulong>(xor_byte);
    output[2] = sum;
    output[3] = rolling;
}

