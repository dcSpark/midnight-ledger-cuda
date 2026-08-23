// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{
    Bls12_381AbiProbe, Bls12_381Fp, Bls12_381Fp64ProbeResult, Bls12_381Fr,
    Bls12_381Fr64ProbeResult, Bls12_381G1Affine, Bls12_381G1ChainProbeResult,
    Bls12_381G1ProbeResult, Bls12_381G1Projective, Bls12_381MsmBasis, Bls12_381MsmWindow,
    GraphAbiProbe, MetalActivity, MetalDeviceInfo, MetalError, MetalKeyContextIdentity,
    MetalKeyContextInput, MetalKeyContextInputV2, MetalQuotientCpuMsmLease,
    MetalQuotientEpochRequest, MetalStatus, NttDirection, NttDomainSpec, ProofByteDigest,
};

const UNAVAILABLE_REASON: &str =
    "this build target does not provide the embedded Apple Metal runtime";

/// API-compatible unavailable session for non-Apple and shaderless builds.
pub struct MetalSession;

/// Unconstructible immutable-key handle for builds without the Metal runtime.
#[derive(Clone, Debug)]
pub struct MetalKeyContext {
    identity: MetalKeyContextIdentity,
    graph_count: usize,
    immutable_bytes: usize,
}

impl MetalKeyContext {
    /// Cannot return on an unsupported target because no context can exist.
    pub fn identity(&self) -> &MetalKeyContextIdentity {
        &self.identity
    }

    /// Cannot return on an unsupported target because no context can exist.
    pub fn graph_count(&self) -> usize {
        self.graph_count
    }

    /// Cannot return on an unsupported target because no context can exist.
    pub fn immutable_bytes(&self) -> usize {
        self.immutable_bytes
    }
}

impl MetalSession {
    /// Returns an explicit unavailable error; no CPU implementation is selected.
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without preparing an SRS or creating a CPU substitute.
    pub fn prepare_bls12_381_msm_srs(
        &self,
        _fingerprint: [u8; 32],
        _coefficient: &[Bls12_381G1Affine],
        _lagrange: &[Bls12_381G1Affine],
    ) -> Result<(), MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without consulting or creating an MSM SRS cache.
    pub fn bls12_381_msm_srs_is_prepared(
        &self,
        _fingerprint: [u8; 32],
    ) -> Result<bool, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without executing or falling back to a CPU MSM.
    pub fn bls12_381_msm(
        &self,
        _fingerprint: [u8; 32],
        _basis: Bls12_381MsmBasis,
        _scalars: &[Bls12_381Fr],
        _window: Bls12_381MsmWindow,
    ) -> Result<Bls12_381G1Projective, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without executing or falling back to CPU batch MSMs.
    pub fn bls12_381_msm_batch(
        &self,
        _fingerprint: [u8; 32],
        _basis: Bls12_381MsmBasis,
        _scalar_batches: &[&[Bls12_381Fr]],
        _window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without validating, allocating, or caching key data.
    pub fn get_or_create_key_context(
        &self,
        _input: MetalKeyContextInput<'_>,
    ) -> Result<MetalKeyContext, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without validating, allocating, or caching complete
    /// schema-v2 proving-key or SRS data.
    pub fn get_or_create_key_context_v2(
        &self,
        _input: MetalKeyContextInputV2<'_>,
    ) -> Result<MetalKeyContext, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable because no key context can exist on this target.
    pub fn validate_key_context(&self, _context: &MetalKeyContext) -> Result<(), MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns unavailable without invoking the CPU MSM callback.
    pub fn with_diagnostic_quotient_epoch<T, F>(
        &self,
        _context: &MetalKeyContext,
        _request: MetalQuotientEpochRequest<'_>,
        _with_cpu_msm: F,
    ) -> Result<T, MetalError>
    where
        F: for<'lease> FnOnce(MetalQuotientCpuMsmLease<'lease>) -> T,
    {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error; no CPU digest is substituted.
    pub fn proof_byte_digest(&self, _bytes: &[u8]) -> Result<ProofByteDigest, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of fabricating an ABI probe.
    pub fn bls12_381_abi_probe(&self) -> Result<Bls12_381AbiProbe, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of fabricating a graph ABI probe.
    pub fn graph_abi_probe(&self) -> Result<GraphAbiProbe, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU field work.
    pub fn probe_bls12_381_fr64(
        &self,
        _lhs: &[Bls12_381Fr],
        _rhs: &[Bls12_381Fr],
    ) -> Result<Bls12_381Fr64ProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU field work.
    pub fn probe_bls12_381_fp64(
        &self,
        _lhs: &[Bls12_381Fp],
        _rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU field work.
    pub fn probe_bls12_381_fp32(
        &self,
        _lhs: &[Bls12_381Fp],
        _rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU field work.
    pub fn probe_bls12_381_fp32_wide(
        &self,
        _lhs: &[Bls12_381Fp],
        _rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU curve work.
    pub fn probe_bls12_381_g1(
        &self,
        _projective: &[Bls12_381G1Projective],
        _affine: &[Bls12_381G1Affine],
    ) -> Result<Bls12_381G1ProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running CPU curve work.
    pub fn probe_bls12_381_g1_complete_add(
        &self,
        _lhs: &[Bls12_381G1Projective],
        _rhs: &[Bls12_381G1Projective],
    ) -> Result<Vec<Bls12_381G1Affine>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of timing CPU curve work.
    pub fn probe_bls12_381_g1_complete_add_chain(
        &self,
        _lhs: &[Bls12_381G1Projective],
        _rhs: &[Bls12_381G1Projective],
        _additions_per_chain: u32,
        _timed_dispatches: u32,
    ) -> Result<Bls12_381G1ChainProbeResult, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of fabricating GPU storage.
    pub fn probe_bls12_381_g1_homogeneous_identity_fill(
        &self,
        _count: usize,
    ) -> Result<bool, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running a CPU NTT.
    pub fn probe_bls12_381_ntt(
        &self,
        _values: &[Bls12_381Fr],
        _domain: NttDomainSpec,
        _direction: NttDirection,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of executing a CPU NTT.
    pub fn execute_bls12_381_ntt(
        &self,
        _values: &[Bls12_381Fr],
        _log_size: u32,
        _direction: NttDirection,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of running a CPU roundtrip.
    pub fn probe_bls12_381_ntt_roundtrip(
        &self,
        _values: &[Bls12_381Fr],
        _domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of extending coefficients.
    pub fn probe_bls12_381_coeff_to_extended(
        &self,
        _coefficients: &[Bls12_381Fr],
        _coefficients_per_batch: usize,
        _extended_domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of extending coefficients.
    pub fn execute_bls12_381_coeff_to_extended(
        &self,
        _coefficients: &[Bls12_381Fr],
        _coefficients_per_batch: usize,
        _extended_log_size: u32,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of interpolating a coset NTT.
    pub fn probe_bls12_381_extended_to_coeff(
        &self,
        _evaluations: &[Bls12_381Fr],
        _extended_domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error instead of clearing on the CPU.
    pub fn clear_words_probe(&self, _words: &[u64]) -> Result<Vec<u64>, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns an explicit unavailable error because no device exists.
    pub fn device_info(&self) -> Result<MetalDeviceInfo, MetalError> {
        Err(MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        })
    }

    /// Returns empty counters for the unavailable session surface.
    pub fn activity(&self) -> MetalActivity {
        MetalActivity::default()
    }

    /// Has no effect because an unavailable session cannot record activity.
    pub fn reset_activity(&self) {}
}

pub(crate) fn probe_status() -> MetalStatus {
    MetalStatus::unavailable(
        false,
        &MetalError::Unavailable {
            reason: UNAVAILABLE_REASON,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_abi_probe_does_not_fabricate_host_layouts() {
        let error = MetalSession
            .graph_abi_probe()
            .expect_err("an unsupported target must fail closed");
        assert!(matches!(error, MetalError::Unavailable { .. }));
    }
}
