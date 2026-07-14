// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

use std::process::ExitCode;
use std::time::Instant;

use proof_server_cuda::{ProofDigestGpu, proof_byte_digest};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let mut bytes = deterministic_bytes(1 << 24);
    let gpu = ProofDigestGpu::new().map_err(|e| e.to_string())?;
    gpu.digest(&bytes).map_err(|e| e.to_string())?;

    let cpu_start = Instant::now();
    let cpu = proof_byte_digest(&bytes);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

    let gpu_start = Instant::now();
    let gpu_digest = gpu.digest(&bytes).map_err(|e| e.to_string())?;
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1e3;

    if cpu != gpu_digest {
        return Err(format!("digest mismatch: cpu={cpu:?} gpu={gpu_digest:?}"));
    }

    let middle = bytes.len() / 2;
    bytes[middle] ^= 1;
    let changed = gpu.digest(&bytes).map_err(|e| e.to_string())?;
    if changed == gpu_digest {
        return Err("digest did not change after input mutation".into());
    }

    println!(
        "adapter_digest=ok bytes={} cpu_ms={:.3} gpu_ms={:.3} speedup={:.3}x digest=len:{} xor:{} sum64:{} rolling:{}",
        bytes.len(),
        cpu_ms,
        gpu_ms,
        cpu_ms / gpu_ms,
        gpu_digest.len,
        gpu_digest.xor,
        gpu_digest.sum64,
        gpu_digest.rolling
    );
    Ok(())
}

fn deterministic_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x5eed_cafe_d15c_a11du64;
    (0..len)
        .map(|idx| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state.wrapping_add(idx as u64) & 0xff) as u8
        })
        .collect()
}
