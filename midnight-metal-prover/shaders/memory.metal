// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>

using namespace metal;

/// Clears exactly the number of words dispatched by the host.
///
/// Proof arenas will use this primitive before recycling secret-bearing shared
/// allocations. The digest smoke path also exercises it to verify queue order.
kernel void midnight_clear_words(
    device ulong* words [[buffer(0)]],
    uint word_index [[thread_position_in_grid]])
{
    words[word_index] = 0;
}
