// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

//! CUDA adapter crate for proof-server.
//!
//! The crate is linked by `midnight-proof-server` behind its `cuda` feature.
//! The `cuda-oxide` feature is reserved for actual NVIDIA cuda-oxide kernels
//! and should be built through the cargo-oxide toolchain. Plain cargo builds use
//! the host-stub adapter below so the backend path, parity gate, and profiling
//! scripts can compile everywhere without pretending the prover is accelerated.

pub const CUDA_OXIDE_REV: &str = "baa1773ad834e6869f5341ff6dd0ba1b48c3e1b8";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterMode {
    HostStub,
    CudaOxide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdapterStatus {
    mode: AdapterMode,
    proof_backend_available: bool,
    proof_backend_unavailable_reason: &'static str,
}

impl AdapterStatus {
    pub fn mode(self) -> AdapterMode {
        self.mode
    }

    pub fn proof_backend_available(self) -> bool {
        self.proof_backend_available
    }

    pub fn proof_backend_unavailable_reason(self) -> &'static str {
        self.proof_backend_unavailable_reason
    }
}

pub fn adapter_status() -> AdapterStatus {
    AdapterStatus {
        mode: adapter_mode(),
        proof_backend_available: false,
        proof_backend_unavailable_reason: "proof-server CUDA proof adapter is linked, but proof execution kernels are not implemented",
    }
}

#[cfg(feature = "cuda-oxide")]
fn adapter_mode() -> AdapterMode {
    AdapterMode::CudaOxide
}

#[cfg(not(feature = "cuda-oxide"))]
fn adapter_mode() -> AdapterMode {
    AdapterMode::HostStub
}

/// Deterministic first adapter boundary for CPU/GPU parity gates.
///
/// Today this is a host-stub digest over final proof bytes. The first real CUDA
/// kernel should preserve this contract exactly before any prover-internal
/// acceleration is trusted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProofByteDigest {
    pub len: usize,
    pub xor: u8,
    pub sum64: u64,
    pub rolling: u64,
}

pub fn proof_byte_digest(bytes: &[u8]) -> ProofByteDigest {
    let mut out = ProofByteDigest {
        len: bytes.len(),
        ..ProofByteDigest::default()
    };
    for (idx, byte) in bytes.iter().copied().enumerate() {
        out.xor ^= byte;
        out.sum64 = out.sum64.wrapping_add(byte as u64);
        out.rolling = out
            .rolling
            .wrapping_mul(0x100_0000_01b3)
            .wrapping_add(byte as u64)
            .wrapping_add(idx as u64);
    }
    out
}

#[cfg(feature = "cuda-oxide")]
pub mod kernels;

#[cfg(feature = "cuda-oxide")]
#[derive(Debug)]
pub enum CudaAdapterError {
    Driver(cuda_core::DriverError),
    Module(cuda_host::EmbeddedModuleError),
}

#[cfg(feature = "cuda-oxide")]
impl std::fmt::Display for CudaAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::Module(err) => write!(f, "CUDA module error: {err}"),
        }
    }
}

#[cfg(feature = "cuda-oxide")]
impl std::error::Error for CudaAdapterError {}

#[cfg(feature = "cuda-oxide")]
impl From<cuda_core::DriverError> for CudaAdapterError {
    fn from(value: cuda_core::DriverError) -> Self {
        Self::Driver(value)
    }
}

#[cfg(feature = "cuda-oxide")]
impl From<cuda_host::EmbeddedModuleError> for CudaAdapterError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        Self::Module(value)
    }
}

#[cfg(feature = "cuda-oxide")]
pub fn proof_byte_digest_cuda(bytes: &[u8]) -> Result<ProofByteDigest, CudaAdapterError> {
    ProofDigestGpu::new()?.digest(bytes)
}

#[cfg(feature = "cuda-oxide")]
pub struct ProofDigestGpu {
    stream: std::sync::Arc<cuda_core::CudaStream>,
    module: kernels::proof_digest::ProofDigestKernelModule,
}

#[cfg(feature = "cuda-oxide")]
impl ProofDigestGpu {
    pub fn new() -> Result<Self, CudaAdapterError> {
        let ctx = std::sync::Arc::new(cuda_core::CudaContext::new(0)?);
        let stream = ctx.new_stream()?;
        let module = kernels::proof_digest::load_proof_digest_kernels(&ctx)?;
        Ok(Self { stream, module })
    }

    pub fn digest(&self, bytes: &[u8]) -> Result<ProofByteDigest, CudaAdapterError> {
        const CHUNK_BYTES: usize = 1024;
        const PARTIAL_WORDS: usize = 5;
        let chunks = bytes.len().div_ceil(CHUNK_BYTES);
        let bytes_dev = cuda_core::DeviceBuffer::from_host(&self.stream, bytes)?;
        let mut partials_dev =
            cuda_core::DeviceBuffer::zeroed(&self.stream, chunks.max(1) * PARTIAL_WORDS)?;
        let mut out_dev = cuda_core::DeviceBuffer::zeroed(&self.stream, 4)?;
        kernels::proof_digest::launch_proof_byte_digest(
            &self.module,
            &self.stream,
            &bytes_dev,
            &mut partials_dev,
            &mut out_dev,
        )?;

        let mut words = [0u64; 4];
        out_dev.copy_to_host(&self.stream, &mut words)?;
        Ok(ProofByteDigest {
            len: words[0] as usize,
            xor: words[1] as u8,
            sum64: words[2],
            rolling: words[3],
        })
    }
}

#[cfg(feature = "cuda-oxide")]
pub fn proof_byte_digest_cuda_one_shot(bytes: &[u8]) -> Result<ProofByteDigest, CudaAdapterError> {
    let ctx = std::sync::Arc::new(cuda_core::CudaContext::new(0)?);
    let stream = ctx.new_stream()?;
    let module = kernels::proof_digest::load_proof_digest_kernels(&ctx)?;
    const CHUNK_BYTES: usize = 1024;
    const PARTIAL_WORDS: usize = 5;
    let chunks = bytes.len().div_ceil(CHUNK_BYTES);
    let bytes_dev = cuda_core::DeviceBuffer::from_host(&stream, bytes)?;
    let mut partials_dev = cuda_core::DeviceBuffer::zeroed(&stream, chunks.max(1) * PARTIAL_WORDS)?;
    let mut out_dev = cuda_core::DeviceBuffer::zeroed(&stream, 4)?;
    kernels::proof_digest::launch_proof_byte_digest(
        &module,
        &stream,
        &bytes_dev,
        &mut partials_dev,
        &mut out_dev,
    )?;

    let mut words = [0u64; 4];
    out_dev.copy_to_host(&stream, &mut words)?;
    Ok(ProofByteDigest {
        len: words[0] as usize,
        xor: words[1] as u8,
        sum64: words[2],
        rolling: words[3],
    })
}

#[cfg(feature = "cuda-oxide")]
pub mod cuda_oxide {
    pub fn enabled() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_stub_reports_proof_backend_unavailable() {
        let status = adapter_status();

        assert_eq!(status.mode(), AdapterMode::HostStub);
        assert!(!status.proof_backend_available());
        assert!(
            status
                .proof_backend_unavailable_reason()
                .contains("not implemented")
        );
    }

    #[test]
    fn proof_byte_digest_is_byte_sensitive() {
        let first = proof_byte_digest(b"proof-bytes");
        let same = proof_byte_digest(b"proof-bytes");
        let different = proof_byte_digest(b"proof-byteT");

        assert_eq!(first, same);
        assert_ne!(first, different);
        assert_eq!(first.len, 11);
    }
}
