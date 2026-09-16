// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cuda_host::EmbeddedModuleError;

pub use proof_digest_kernels::LoadedModule as ProofDigestKernelModule;

pub fn load_proof_digest_kernels(
    ctx: &Arc<CudaContext>,
) -> Result<ProofDigestKernelModule, EmbeddedModuleError> {
    proof_digest_kernels::load(ctx)
}

pub fn launch_proof_byte_digest(
    module: &ProofDigestKernelModule,
    stream: &Arc<CudaStream>,
    bytes: &DeviceBuffer<u8>,
    partials: &mut DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    const CHUNK_BYTES: usize = 1024;
    let chunks = bytes.len().div_ceil(CHUNK_BYTES);
    if chunks == 0 {
        return module.proof_byte_digest_empty(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            out,
        );
    }

    module.proof_byte_digest_partials(
        stream,
        LaunchConfig::for_num_elems(chunks as u32),
        bytes,
        CHUNK_BYTES,
        partials,
    )?;
    module.proof_byte_digest_finalize(
        stream,
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        partials,
        chunks,
        out,
    )
}

#[cuda_module]
pub mod proof_digest_kernels {
    use super::*;

    #[kernel]
    pub fn proof_byte_digest_empty(mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() != 0 || out.len() < 4 {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(0) = 0;
            *out.get_unchecked_mut(1) = 0;
            *out.get_unchecked_mut(2) = 0;
            *out.get_unchecked_mut(3) = 0;
        }
    }

    #[kernel]
    pub fn proof_byte_digest_partials(
        bytes: &[u8],
        chunk_bytes: usize,
        mut partials: DisjointSlice<u64>,
    ) {
        const FNV64_PRIME: u64 = 0x100_0000_01b3;
        const PARTIAL_WORDS: usize = 5;

        let chunk = thread::index_1d().get();
        let start = chunk * chunk_bytes;
        if start >= bytes.len() {
            return;
        }
        let mut end = start + chunk_bytes;
        if end > bytes.len() {
            end = bytes.len();
        }
        let base = chunk * PARTIAL_WORDS;
        if base + PARTIAL_WORDS > partials.len() {
            return;
        }

        let mut len = 0u64;
        let mut xor = 0u64;
        let mut sum64 = 0u64;
        let mut rolling = 0u64;
        let mut pow = 1u64;
        let mut idx = start;
        while idx < end {
            let byte = bytes[idx] as u64;
            xor ^= byte;
            sum64 = sum64.wrapping_add(byte);
            rolling = rolling
                .wrapping_mul(FNV64_PRIME)
                .wrapping_add(byte)
                .wrapping_add(idx as u64);
            pow = pow.wrapping_mul(FNV64_PRIME);
            len += 1;
            idx += 1;
        }

        unsafe {
            *partials.get_unchecked_mut(base) = len;
            *partials.get_unchecked_mut(base + 1) = xor;
            *partials.get_unchecked_mut(base + 2) = sum64;
            *partials.get_unchecked_mut(base + 3) = rolling;
            *partials.get_unchecked_mut(base + 4) = pow;
        }
    }

    #[kernel]
    pub fn proof_byte_digest_finalize(
        partials: &[u64],
        chunks: usize,
        mut out: DisjointSlice<u64>,
    ) {
        const PARTIAL_WORDS: usize = 5;

        if thread::index_1d().get() != 0 || out.len() < 4 {
            return;
        }

        let mut len = 0u64;
        let mut xor = 0u64;
        let mut sum64 = 0u64;
        let mut rolling = 0u64;
        let mut chunk = 0usize;
        while chunk < chunks {
            let base = chunk * PARTIAL_WORDS;
            if base + PARTIAL_WORDS > partials.len() {
                return;
            }
            let chunk_len = partials[base];
            let chunk_xor = partials[base + 1];
            let chunk_sum64 = partials[base + 2];
            let chunk_rolling = partials[base + 3];
            let chunk_pow = partials[base + 4];

            len = len.wrapping_add(chunk_len);
            xor ^= chunk_xor;
            sum64 = sum64.wrapping_add(chunk_sum64);
            rolling = rolling.wrapping_mul(chunk_pow).wrapping_add(chunk_rolling);
            chunk += 1;
        }

        unsafe {
            *out.get_unchecked_mut(0) = len;
            *out.get_unchecked_mut(1) = xor;
            *out.get_unchecked_mut(2) = sum64;
            *out.get_unchecked_mut(3) = rolling;
        }
    }
}
