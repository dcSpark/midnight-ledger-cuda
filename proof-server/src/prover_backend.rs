// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

use std::fmt;

use clap::ValueEnum;

/// Proof execution backend requested by proof-server.
///
/// CPU remains canonical. CUDA is an explicit integration point and must never
/// silently fall back to CPU when selected.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ProverBackend {
    #[default]
    Cpu,
    Cuda,
}

impl ProverBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    pub fn ensure_available(self) -> Result<(), ProverBackendError> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(feature = "gpu-blitzar")]
            Self::Cuda => Ok(()),
            #[cfg(not(feature = "gpu-blitzar"))]
            Self::Cuda => Err(ProverBackendError::BackendUnavailable {
                backend: self.as_str(),
                reason: cuda_unavailable_reason(),
            }),
        }
    }
}

impl fmt::Display for ProverBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProverBackendError {
    BackendUnavailable {
        backend: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for ProverBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable { backend, reason } => {
                write!(f, "prover backend '{backend}' is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for ProverBackendError {}

#[cfg(all(feature = "cuda", not(feature = "gpu-blitzar")))]
fn cuda_unavailable_reason() -> &'static str {
    proof_server_cuda::adapter_status().proof_backend_unavailable_reason()
}

#[cfg(all(not(feature = "cuda"), not(feature = "gpu-blitzar")))]
fn cuda_unavailable_reason() -> &'static str {
    "proof-server was compiled without the cuda feature"
}

/// Proof-server stages in dependency order for the resident single-proof path.
///
/// CPU owns decode, witness synthesis, transcript order, verification, and
/// encoding. Large polynomial state is GPU-owned from upload through multi-open.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaProofStage {
    KeyMaterial,
    IrDecode,
    DeviceContext,
    WitnessSynthesis,
    WitnessUpload,
    TraceArguments,
    QuotientNumerator,
    TranscriptEvaluations,
    MultiOpen,
    SelfVerify,
    Encode,
}

impl CudaProofStage {
    pub const SINGLE_PROOF_ORDER: [Self; 11] = [
        Self::KeyMaterial,
        Self::IrDecode,
        Self::DeviceContext,
        Self::WitnessSynthesis,
        Self::WitnessUpload,
        Self::TraceArguments,
        Self::QuotientNumerator,
        Self::TranscriptEvaluations,
        Self::MultiOpen,
        Self::SelfVerify,
        Self::Encode,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaResidencyPolicy {
    setup_h2d: bool,
    hot_loop_h2d: bool,
    hot_loop_d2h: bool,
    hot_loop_host_sync: bool,
    final_d2h: bool,
}

impl CudaResidencyPolicy {
    pub const PROOF_SERVER_SINGLE_PROOF: Self = Self {
        setup_h2d: true,
        hot_loop_h2d: false,
        hot_loop_d2h: false,
        hot_loop_host_sync: false,
        final_d2h: true,
    };

    pub fn setup_h2d(self) -> bool {
        self.setup_h2d
    }

    pub fn hot_loop_h2d(self) -> bool {
        self.hot_loop_h2d
    }

    pub fn hot_loop_d2h(self) -> bool {
        self.hot_loop_d2h
    }

    pub fn hot_loop_host_sync(self) -> bool {
        self.hot_loop_host_sync
    }

    pub fn final_d2h(self) -> bool {
        self.final_d2h
    }

    pub fn keeps_hot_loop_resident(self) -> bool {
        !self.hot_loop_h2d && !self.hot_loop_d2h && !self.hot_loop_host_sync
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaProofPlan {
    stages: [CudaProofStage; 11],
    residency: CudaResidencyPolicy,
}

impl CudaProofPlan {
    pub const SINGLE_PROOF_RESIDENT: Self = Self {
        stages: CudaProofStage::SINGLE_PROOF_ORDER,
        residency: CudaResidencyPolicy::PROOF_SERVER_SINGLE_PROOF,
    };

    pub fn stages(self) -> [CudaProofStage; 11] {
        self.stages
    }

    pub fn residency(self) -> CudaResidencyPolicy {
        self.residency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "gpu-blitzar"))]
    #[test]
    fn cuda_backend_is_explicit_and_unavailable_until_kernels_exist() {
        let err = ProverBackend::Cuda
            .ensure_available()
            .expect_err("CUDA should not silently fall back to CPU");

        assert!(matches!(
            err,
            ProverBackendError::BackendUnavailable {
                backend: "cuda",
                ..
            }
        ));
    }

    #[cfg(feature = "gpu-blitzar")]
    #[test]
    fn cuda_backend_is_available_with_gpu_blitzar() {
        ProverBackend::Cuda
            .ensure_available()
            .expect("gpu-blitzar provides the resident CUDA prover backend");
    }

    #[test]
    fn resident_policy_keeps_hot_loop_on_device() {
        let policy = CudaResidencyPolicy::PROOF_SERVER_SINGLE_PROOF;

        assert!(policy.setup_h2d());
        assert!(policy.final_d2h());
        assert!(!policy.hot_loop_h2d());
        assert!(!policy.hot_loop_d2h());
        assert!(!policy.hot_loop_host_sync());
        assert!(policy.keeps_hot_loop_resident());
    }

    #[test]
    fn single_proof_plan_names_stage_order() {
        assert_eq!(
            CudaProofPlan::SINGLE_PROOF_RESIDENT.stages(),
            [
                CudaProofStage::KeyMaterial,
                CudaProofStage::IrDecode,
                CudaProofStage::DeviceContext,
                CudaProofStage::WitnessSynthesis,
                CudaProofStage::WitnessUpload,
                CudaProofStage::TraceArguments,
                CudaProofStage::QuotientNumerator,
                CudaProofStage::TranscriptEvaluations,
                CudaProofStage::MultiOpen,
                CudaProofStage::SelfVerify,
                CudaProofStage::Encode,
            ]
        );
    }
}
