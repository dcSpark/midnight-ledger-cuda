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
/// CPU remains canonical. Accelerator backends are explicit and must never
/// silently fall back to CPU when selected.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ProverBackend {
    #[default]
    Cpu,
    Cuda,
    Metal,
    MetalHybrid,
}

impl ProverBackend {
    /// Stable command-line and diagnostic name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::MetalHybrid => "metal-hybrid",
        }
    }

    /// Resolves a requested label to a concrete proof execution route.
    ///
    /// This is the authority for proof selection. Runtime or smoke-kernel
    /// readiness is insufficient: a backend is accepted only after its native
    /// proof path is connected here. Consequently Metal cannot fall through to
    /// the canonical CPU path while its implementation is under construction.
    pub fn execution_route(self) -> Result<ProofExecutionRoute, ProverBackendError> {
        match self {
            Self::Cpu => Ok(ProofExecutionRoute::Cpu),
            #[cfg(feature = "gpu-blitzar")]
            Self::Cuda => Ok(ProofExecutionRoute::Cuda),
            #[cfg(not(feature = "gpu-blitzar"))]
            Self::Cuda => Err(ProverBackendError::BackendUnavailable {
                backend: self.as_str(),
                reason: cuda_unavailable_reason().to_owned(),
            }),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal => Err(ProverBackendError::BackendUnavailable {
                backend: self.as_str(),
                reason: metal_execution_route_unavailable_reason(),
            }),
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            Self::Metal => Err(ProverBackendError::BackendUnavailable {
                backend: self.as_str(),
                reason: metal_unavailable_reason().to_owned(),
            }),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::MetalHybrid => {
                let status = midnight_gpu_prover::metal::status();
                if status.runtime() == midnight_gpu_prover::RuntimeAvailability::Available {
                    Ok(ProofExecutionRoute::MetalHybrid)
                } else {
                    Err(ProverBackendError::BackendUnavailable {
                        backend: self.as_str(),
                        reason: status
                            .unavailable_reason()
                            .unwrap_or("Metal runtime is unavailable")
                            .to_owned(),
                    })
                }
            }
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            Self::MetalHybrid => Err(ProverBackendError::BackendUnavailable {
                backend: self.as_str(),
                reason: metal_unavailable_reason().to_owned(),
            }),
        }
    }

    /// Enters the selected proof route for the lifetime of the returned guard.
    ///
    /// Every proof entry point must hold this guard across the complete native
    /// proof operation. CUDA and Metal support are compiled in separate platform
    /// lanes. This source-level guard acquires one global proof permit and enables
    /// only the selected lane's hook; entering CPU disables that hook and preserves
    /// the canonical implementation.
    pub fn enter_proof_execution(self) -> Result<ProofExecutionGuard, ProverBackendError> {
        let route = self.execution_route()?;
        #[cfg(any(feature = "gpu-blitzar", feature = "metal"))]
        let accelerator_guard = transient_crypto::proofs::ProofAcceleratorGuard::set(
            route == ProofExecutionRoute::Cuda,
            route == ProofExecutionRoute::MetalHybrid,
        );

        Ok(ProofExecutionGuard {
            route,
            #[cfg(any(feature = "gpu-blitzar", feature = "metal"))]
            _accelerator_guard: accelerator_guard,
        })
    }
}

/// Concrete proof routes implemented by proof-server.
///
/// The hybrid route is intentionally named separately from `Metal`. Metal owns
/// NTTs, fixed-base MSM, quotient construction and commitments, and the resident
/// multi-open polynomial/MSM stages. CPU still owns witness synthesis, transcript
/// orchestration, and one combined quotient-opening materialization. Once an
/// operation selects Metal, failure aborts instead of retrying on CPU. The pure
/// `Metal` label remains unavailable until that polynomial materialization and
/// the remaining hot-loop host ownership are removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofExecutionRoute {
    /// Canonical CPU prover with accelerator hooks disabled.
    Cpu,
    /// Existing resident CUDA/Blitzar prover path.
    Cuda,
    /// Mostly accelerated CPU+Metal proof with explicit remaining host seams.
    MetalHybrid,
}

impl ProofExecutionRoute {
    /// Stable benchmark and diagnostic name for the concrete route.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::MetalHybrid => "metal-hybrid",
        }
    }

    /// Whether independent proof jobs may run concurrently on this route.
    ///
    /// CUDA owns bounded execution lanes. Metal uses bounded worker queues and
    /// serialized device ownership while allowing independent CPU preparation
    /// to overlap. CPU remains the serial reference.
    pub const fn supports_parallel_proof_jobs(self) -> bool {
        matches!(self, Self::Cuda | Self::MetalHybrid)
    }
}

impl fmt::Display for ProofExecutionRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Scope guard for a resolved proof execution route.
#[must_use = "the execution guard must live across the entire proof operation"]
pub struct ProofExecutionGuard {
    route: ProofExecutionRoute,
    #[cfg(any(feature = "gpu-blitzar", feature = "metal"))]
    _accelerator_guard: transient_crypto::proofs::ProofAcceleratorGuard,
}

impl ProofExecutionGuard {
    /// Concrete route held by this guard.
    pub const fn route(&self) -> ProofExecutionRoute {
        self.route
    }
}

impl fmt::Display for ProverBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProverBackendError {
    BackendUnavailable {
        backend: &'static str,
        reason: String,
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

#[cfg(all(
    feature = "metal",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
fn metal_unavailable_reason() -> &'static str {
    "the Metal proof backend requires Apple silicon macOS"
}

#[cfg(not(feature = "metal"))]
fn metal_unavailable_reason() -> &'static str {
    "proof-server was compiled without the metal feature"
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_execution_route_unavailable_reason() -> String {
    let status = midnight_gpu_prover::metal::status();
    let detail = status
        .unavailable_reason()
        .unwrap_or("Metal runtime and kernels report ready");
    format!("{detail}; proof-server has no native Metal proof execution route")
}

/// Target proof-server stages for the fully resident single-proof path.
///
/// CPU owns decode, witness synthesis, transcript order, verification, and
/// encoding. Large polynomial state is GPU-owned from upload through multi-open.
/// This target contract is stricter than today's `metal-hybrid` route, which
/// still materializes the combined quotient opening and some multi-open inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcceleratorProofStage {
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

impl AcceleratorProofStage {
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

/// Host/device ownership policy for one accelerated proof request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceleratorResidencyPolicy {
    setup_h2d: bool,
    hot_loop_h2d: bool,
    hot_loop_d2h: bool,
    hot_loop_host_sync: bool,
    final_d2h: bool,
}

impl AcceleratorResidencyPolicy {
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

/// Backend-neutral stage order and residency contract for an accelerated proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceleratorProofPlan {
    stages: [AcceleratorProofStage; 11],
    residency: AcceleratorResidencyPolicy,
}

impl AcceleratorProofPlan {
    pub const SINGLE_PROOF_RESIDENT: Self = Self {
        stages: AcceleratorProofStage::SINGLE_PROOF_ORDER,
        residency: AcceleratorResidencyPolicy::PROOF_SERVER_SINGLE_PROOF,
    };

    pub fn stages(self) -> [AcceleratorProofStage; 11] {
        self.stages
    }

    pub fn residency(self) -> AcceleratorResidencyPolicy {
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
            .execution_route()
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
        let route = ProverBackend::Cuda
            .execution_route()
            .expect("gpu-blitzar provides the resident CUDA prover backend");

        assert_eq!(route, ProofExecutionRoute::Cuda);
    }

    #[test]
    fn metal_backend_has_no_route_until_native_provider_is_connected() {
        let err = match ProverBackend::Metal.enter_proof_execution() {
            Ok(_) => panic!("Metal should not silently fall back to CPU"),
            Err(error) => error,
        };

        let ProverBackendError::BackendUnavailable { backend, reason } = err;
        assert_eq!(backend, "metal");
        assert!(!reason.is_empty());
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        assert!(reason.contains("no native Metal proof execution route"));
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_hybrid_route_tracks_runtime_availability() {
        let status = midnight_gpu_prover::metal::status();
        let route = ProverBackend::MetalHybrid.execution_route();

        match status.runtime() {
            midnight_gpu_prover::RuntimeAvailability::Available => {
                assert_eq!(route, Ok(ProofExecutionRoute::MetalHybrid));
            }
            midnight_gpu_prover::RuntimeAvailability::Unavailable => {
                let ProverBackendError::BackendUnavailable { backend, reason } =
                    route.expect_err("an unavailable Metal runtime must fail closed");
                assert_eq!(backend, "metal-hybrid");
                assert!(!reason.is_empty());
            }
            midnight_gpu_prover::RuntimeAvailability::Unprobed => {
                panic!("the Metal status probe must resolve runtime availability");
            }
        }
    }

    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn metal_hybrid_route_is_unavailable_without_apple_metal_support() {
        let ProverBackendError::BackendUnavailable { backend, reason } = ProverBackend::MetalHybrid
            .execution_route()
            .expect_err("Metal hybrid must not silently fall back to CPU");

        assert_eq!(backend, "metal-hybrid");
        assert!(!reason.is_empty());
    }

    #[test]
    fn resident_policy_keeps_hot_loop_on_device() {
        let policy = AcceleratorResidencyPolicy::PROOF_SERVER_SINGLE_PROOF;

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
            AcceleratorProofPlan::SINGLE_PROOF_RESIDENT.stages(),
            [
                AcceleratorProofStage::KeyMaterial,
                AcceleratorProofStage::IrDecode,
                AcceleratorProofStage::DeviceContext,
                AcceleratorProofStage::WitnessSynthesis,
                AcceleratorProofStage::WitnessUpload,
                AcceleratorProofStage::TraceArguments,
                AcceleratorProofStage::QuotientNumerator,
                AcceleratorProofStage::TranscriptEvaluations,
                AcceleratorProofStage::MultiOpen,
                AcceleratorProofStage::SelfVerify,
                AcceleratorProofStage::Encode,
            ]
        );
    }

    #[test]
    fn cpu_enters_only_the_canonical_cpu_route() {
        let guard = ProverBackend::Cpu
            .enter_proof_execution()
            .expect("CPU route is always present");

        assert_eq!(guard.route(), ProofExecutionRoute::Cpu);
        assert!(!guard.route().supports_parallel_proof_jobs());
        assert!(ProofExecutionRoute::Cuda.supports_parallel_proof_jobs());
        assert!(ProofExecutionRoute::MetalHybrid.supports_parallel_proof_jobs());
    }
}
