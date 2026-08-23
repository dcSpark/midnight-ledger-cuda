// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// One root translation unit keeps shared BLS12-381 types and field helpers
// identical across all kernels while preserving phase-oriented source files.
#include "proof_digest.metal"
#include "memory.metal"
#include "field.metal"
#include "expression.metal"
#include "fp32.metal"
#include "batch_affine.metal"
#include "flat_segmented.metal"
#include "ntt.metal"
#include "quotient.metal"
#include "multiopen.metal"
#include "g1.metal"
