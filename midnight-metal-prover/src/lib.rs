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

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![forbid(unsafe_op_in_unsafe_fn)]

//! Native Metal runtime for Midnight proof acceleration.
//!
//! The first accelerated boundaries are a deterministic digest of completed
//! proof bytes plus correctness probes for raw BLS12-381 Fr/Fp/G1 arithmetic
//! and resident scalar-field NTTs. They are intentionally separate from
//! proof-backend readiness: working primitives prove that the Metal toolchain
//! and arithmetic are usable, not that proof generation runs on Metal.

use std::{fmt, marker::PhantomData, sync::OnceLock};

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
use bls12_381_metal_msm::ResidentMontgomeryScalars;

#[allow(dead_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bls12_381_constants.rs"));
}

mod key_context;
pub use key_context::{
    METAL_KEY_CONTEXT_SCHEMA_VERSION, METAL_KEY_CONTEXT_SCHEMA_VERSION_V2, MetalGraphDescriptor,
    MetalGraphFamilySections, MetalGraphOperation, MetalGraphValueRef, MetalKeyContextIdentity,
    MetalKeyContextInput, MetalKeyContextInputV2, MetalResidentColumnRef,
    MetalResidentDomainIdentity,
};

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
mod metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
use metal as platform;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders)))]
mod unsupported;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders)))]
use unsupported as platform;

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
pub use platform::MetalMultiopenArena;
pub use platform::{MetalKeyContext, MetalSession};

/// Raw BLS12-381 scalar-field element used by BLST and Metal buffers.
///
/// Limbs are little-endian, reduced, and in Montgomery form. This is an
/// internal-memory ABI, not a canonical scalar serialization. Its wire layout
/// also matches the independently built CUDA reference backend; no CUDA code is
/// linked into this crate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Bls12_381Fr {
    /// Four little-endian 64-bit Montgomery limbs.
    pub limbs: [u64; 4],
}

/// Host polynomials and compact protocol metadata for one resident Metal
/// multi-opening argument.
///
/// Every polynomial has the same coefficient length. `q_offsets` partitions
/// `q_poly_indices` and `q_coefficients` into the already-sorted point-set
/// order required by the protocol. `point_offsets` partitions `points` in that
/// same order, and `f_weights` contains one power of the second multi-open
/// challenge per q-polynomial.
#[derive(Clone, Copy, Debug)]
pub struct MetalMultiopenRequest<'a> {
    polynomials: &'a [&'a [Bls12_381Fr]],
    q_offsets: &'a [u32],
    q_poly_indices: &'a [u32],
    q_coefficients: &'a [Bls12_381Fr],
    point_offsets: &'a [u32],
    points: &'a [Bls12_381Fr],
    f_weights: &'a [Bls12_381Fr],
}

impl<'a> MetalMultiopenRequest<'a> {
    /// Creates a multi-open request. The session performs complete shape,
    /// index, and field-reduction validation before allocating secret buffers.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        polynomials: &'a [&'a [Bls12_381Fr]],
        q_offsets: &'a [u32],
        q_poly_indices: &'a [u32],
        q_coefficients: &'a [Bls12_381Fr],
        point_offsets: &'a [u32],
        points: &'a [Bls12_381Fr],
        f_weights: &'a [Bls12_381Fr],
    ) -> Self {
        Self {
            polynomials,
            q_offsets,
            q_poly_indices,
            q_coefficients,
            point_offsets,
            points,
            f_weights,
        }
    }

    /// Equal-length coefficient polynomials entering the argument.
    pub const fn polynomials(self) -> &'a [&'a [Bls12_381Fr]] {
        self.polynomials
    }

    /// Prefix offsets into the q-polynomial source-entry arrays.
    pub const fn q_offsets(self) -> &'a [u32] {
        self.q_offsets
    }

    /// Source polynomial index for every q-polynomial entry.
    pub const fn q_poly_indices(self) -> &'a [u32] {
        self.q_poly_indices
    }

    /// Linear-combination coefficient for every q-polynomial entry.
    pub const fn q_coefficients(self) -> &'a [Bls12_381Fr] {
        self.q_coefficients
    }

    /// Prefix offsets partitioning the flattened point array.
    pub const fn point_offsets(self) -> &'a [u32] {
        self.point_offsets
    }

    /// Flattened point sets in sorted q-polynomial order.
    pub const fn points(self) -> &'a [Bls12_381Fr] {
        self.points
    }

    /// Powers used to combine divided q-polynomials into f.
    pub const fn f_weights(self) -> &'a [Bls12_381Fr] {
        self.f_weights
    }
}

/// Ordered borrowed inputs for the diagnostic resident quotient epoch.
///
/// Every coefficient column has exactly the base-domain length carried by the
/// supplied schema-v2 [`MetalKeyContext`]. Columns are ordered as advice,
/// instance, permutation products, lookup products, lookup permuted inputs,
/// lookup permuted tables, then trash. Keeping the columns borrowed and
/// separate lets the proof-facing adapter avoid constructing one concatenated
/// host vector.
#[derive(Clone, Copy, Debug)]
pub struct MetalQuotientEpochRequest<'a> {
    /// Individually allocated coefficient columns in canonical protocol order.
    pub coefficient_columns: &'a [&'a [Bls12_381Fr]],
    /// Number of entries in [`Self::coefficient_columns`].
    pub total_columns: u32,
    /// First advice column in the ordered dynamic arena.
    pub advice_offset: u32,
    /// Number of compact advice columns.
    pub advice_columns: u32,
    /// First instance column in the ordered dynamic arena.
    pub instance_offset: u32,
    /// Number of compact instance columns.
    pub instance_columns: u32,
    /// First permutation-product column in the ordered dynamic arena.
    pub permutation_offset: u32,
    /// Number of permutation-product sets.
    pub permutation_sets: u32,
    /// First lookup-product column in the ordered dynamic arena.
    pub lookup_product_offset: u32,
    /// First lookup-input column in the ordered dynamic arena.
    pub lookup_input_offset: u32,
    /// First lookup-table column in the ordered dynamic arena.
    pub lookup_table_offset: u32,
    /// First trash-product column in the ordered dynamic arena.
    pub trash_offset: u32,
    /// Transcript challenges addressable by the compiled expression graphs.
    pub challenges: &'a [Bls12_381Fr],
    /// Permutation and lookup challenge beta.
    pub beta: Bls12_381Fr,
    /// Permutation and lookup challenge gamma.
    pub gamma: Bls12_381Fr,
    /// Lookup compression challenge theta.
    pub theta: Bls12_381Fr,
    /// Trash argument challenge.
    pub trash_challenge: Bls12_381Fr,
    /// Quotient-expression aggregation challenge.
    pub y: Bls12_381Fr,
    /// Existing prover-sampled inter-piece blinds in ascending boundary order.
    pub quotient_blinds: &'a [Bls12_381Fr],
}

/// Callback-scoped CPU view of blinded quotient pieces produced by Metal.
///
/// The view is deliberately not cloneable and exposes no Metal object. Its
/// backing secret allocation remains pinned until the callback supplied to
/// [`MetalSession::with_diagnostic_quotient_epoch`] returns.
pub struct MetalQuotientCpuMsmLease<'a> {
    values: &'a [Bls12_381Fr],
    piece_len: usize,
    piece_count: usize,
    key_context_identity: &'a MetalKeyContextIdentity,
    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    resident_scalars: ResidentMontgomeryScalars,
}

impl<'a> MetalQuotientCpuMsmLease<'a> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    pub(crate) fn new(
        values: &'a [Bls12_381Fr],
        piece_len: usize,
        piece_count: usize,
        key_context_identity: &'a MetalKeyContextIdentity,
        resident_scalars: ResidentMontgomeryScalars,
    ) -> Self {
        Self {
            values,
            piece_len,
            piece_count,
            key_context_identity,
            resident_scalars,
        }
    }

    /// Exact `piece_len * piece_count` CPU-visible raw scalar slice.
    pub const fn as_raw_fr(&self) -> &'a [Bls12_381Fr] {
        self.values
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    pub(crate) const fn resident_scalars(&self) -> &ResidentMontgomeryScalars {
        &self.resident_scalars
    }

    /// Number of coefficients, including the boundary blind slot, per piece.
    pub const fn piece_len(&self) -> usize {
        self.piece_len
    }

    /// Number of quotient pieces.
    pub const fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// Complete immutable key, domain, SRS, ABI, metallib, and device identity.
    pub const fn key_context_identity(&self) -> &MetalKeyContextIdentity {
        self.key_context_identity
    }
}

/// Callback-scoped resident quotient pieces for the standalone Metal MSM worker.
///
/// The lease owns only a retained view; it exposes no CPU pointer. The quotient
/// producer has completed before construction, the MSM call is synchronous, and
/// the originating session erases the allocation after the callback returns.
#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
#[must_use = "the resident quotient lease must be consumed inside its callback"]
pub struct MetalQuotientResidentMsmLease<'a> {
    scalars: ResidentMontgomeryScalars,
    continuation: platform::QuotientContinuationBuffers,
    piece_len: usize,
    piece_count: usize,
    key_context_identity: &'a MetalKeyContextIdentity,
    scope: PhantomData<&'a mut ()>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
impl fmt::Debug for MetalQuotientResidentMsmLease<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalQuotientResidentMsmLease")
            .field("piece_len", &self.piece_len)
            .field("piece_count", &self.piece_count)
            .field("device_registry_id", &self.scalars.device_registry_id())
            .finish_non_exhaustive()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
impl<'a> MetalQuotientResidentMsmLease<'a> {
    pub(crate) fn new(
        scalars: ResidentMontgomeryScalars,
        continuation: platform::QuotientContinuationBuffers,
        piece_len: usize,
        piece_count: usize,
        key_context_identity: &'a MetalKeyContextIdentity,
    ) -> Self {
        Self {
            scalars,
            continuation,
            piece_len,
            piece_count,
            key_context_identity,
            scope: PhantomData,
        }
    }

    /// Number of resident coefficients in each quotient piece.
    pub const fn piece_len(&self) -> usize {
        self.piece_len
    }

    /// Number of ordered resident quotient pieces.
    pub const fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// Complete immutable key, domain, SRS, ABI, metallib, and device identity.
    pub const fn key_context_identity(&self) -> &MetalKeyContextIdentity {
        self.key_context_identity
    }
}

/// Callback-scoped resident coefficient polynomial accepted by the Metal MSM
/// worker without scalar readback.
#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
#[must_use = "the resident polynomial lease must be committed inside its arena callback"]
pub struct MetalResidentPolynomialMsmLease<'a> {
    scalars: ResidentMontgomeryScalars,
    len: usize,
    session_id: u64,
    scope: PhantomData<&'a mut ()>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
impl fmt::Debug for MetalResidentPolynomialMsmLease<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalResidentPolynomialMsmLease")
            .field("len", &self.len)
            .field("device_registry_id", &self.scalars.device_registry_id())
            .finish_non_exhaustive()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
impl<'a> MetalResidentPolynomialMsmLease<'a> {
    pub(crate) fn new(scalars: ResidentMontgomeryScalars, len: usize, session_id: u64) -> Self {
        Self {
            scalars,
            len,
            session_id,
            scope: PhantomData,
        }
    }

    pub(crate) const fn scalars(&self) -> &ResidentMontgomeryScalars {
        &self.scalars
    }

    pub(crate) const fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Number of coefficient scalars in the resident polynomial.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the resident polynomial contains no coefficient scalars.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Direction of a resident BLS12-381 scalar-field NTT.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NttDirection {
    /// Evaluate coefficients with the domain's primitive root.
    Forward,
    /// Interpolate evaluations with the inverse root and one final `1/n` scale.
    Inverse,
}

/// Host-validated constants for one power-of-two BLS12-381 Fr domain.
///
/// Every field value uses the same raw reduced Montgomery representation as
/// [`Bls12_381Fr`]. A session validates the root orders, inverse pairs, inverse
/// size, and order-three coset generator before caching a device plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NttDomainSpec {
    /// Base-two logarithm of the transform size.
    pub log_size: u32,
    /// Primitive `2^log_size` root of unity.
    pub root: Bls12_381Fr,
    /// Multiplicative inverse of [`Self::root`].
    pub inverse_root: Bls12_381Fr,
    /// Montgomery representation of `1 / 2^log_size`.
    pub inverse_size: Bls12_381Fr,
    /// Nontrivial order-three generator used for coset ingress.
    pub coset_generator: Bls12_381Fr,
    /// Multiplicative inverse of [`Self::coset_generator`].
    pub inverse_coset_generator: Bls12_381Fr,
}

/// Raw BLS12-381 base-field element used by BLST and Metal buffers.
///
/// Limbs are little-endian, reduced, and in Montgomery form.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Bls12_381Fp {
    /// Six little-endian 64-bit Montgomery limbs.
    pub limbs: [u64; 6],
}

/// Raw BLS12-381 G1 affine point for BLST and Metal buffers.
///
/// Both coordinates contain little-endian Montgomery base-field limbs. The
/// affine identity is encoded with both coordinates zero. The wire layout also
/// matches the independently built CUDA reference ABI without linking CUDA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Bls12_381G1Affine {
    /// Affine x-coordinate in Montgomery form.
    pub x: [u64; 6],
    /// Affine y-coordinate in Montgomery form.
    pub y: [u64; 6],
}

/// Raw BLS12-381 G1 Jacobian point for BLST and Metal buffers.
///
/// All coordinates contain little-endian Montgomery base-field limbs. Any
/// point with a zero z-coordinate represents the identity. The wire layout also
/// matches the independently built CUDA reference ABI without linking CUDA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Bls12_381G1Projective {
    /// Jacobian x-coordinate in Montgomery form.
    pub x: [u64; 6],
    /// Jacobian y-coordinate in Montgomery form.
    pub y: [u64; 6],
    /// Jacobian z-coordinate in Montgomery form.
    pub z: [u64; 6],
}

/// Prepared G1 basis selected for a standalone Metal MSM request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Bls12_381MsmBasis {
    /// Monomial coefficient SRS prefix.
    Coefficient,
    /// Domain-specific Lagrange SRS basis.
    Lagrange,
}

/// Signed Booth window supported by the standalone Metal MSM engine.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Bls12_381MsmWindow {
    /// Six-bit windows.
    W6,
    /// Seven-bit windows.
    W7,
    /// Eight-bit windows.
    W8,
    /// Nine-bit windows.
    W9,
    /// Ten-bit windows.
    W10,
    /// Eleven-bit pre-shifted fixed-base windows.
    FixedW11,
    /// Twelve-bit pre-shifted fixed-base windows.
    FixedW12,
    /// Thirteen-bit pre-shifted fixed-base windows.
    FixedW13,
    /// Fourteen-bit pre-shifted fixed-base windows.
    FixedW14,
    /// Fifteen-bit pre-shifted fixed-base windows.
    FixedW15,
    /// Sixteen-bit pre-shifted fixed-base windows.
    FixedW16,
}

/// Layout and generated-constant values reported by the compiled Metal shader.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381AbiProbe {
    /// Version of the shared Rust/Metal ABI contract.
    pub abi_version: u64,
    /// Metal `sizeof(Bls12_381Fr)`.
    pub fr_size: u64,
    /// Metal `alignof(Bls12_381Fr)`.
    pub fr_alignment: u64,
    /// Metal offset of `Bls12_381Fr::limbs`.
    pub fr_limbs_offset: u64,
    /// Metal `sizeof(Bls12_381Fp)`.
    pub fp_size: u64,
    /// Metal `alignof(Bls12_381Fp)`.
    pub fp_alignment: u64,
    /// Metal offset of `Bls12_381Fp::limbs`.
    pub fp_limbs_offset: u64,
    /// Metal `sizeof(Bls12_381G1Affine)`.
    pub g1_affine_size: u64,
    /// Metal `alignof(Bls12_381G1Affine)`.
    pub g1_affine_alignment: u64,
    /// Metal offset of `Bls12_381G1Affine::x`.
    pub g1_affine_x_offset: u64,
    /// Metal offset of `Bls12_381G1Affine::y`.
    pub g1_affine_y_offset: u64,
    /// Metal `sizeof(Bls12_381G1Projective)`.
    pub g1_projective_size: u64,
    /// Metal `alignof(Bls12_381G1Projective)`.
    pub g1_projective_alignment: u64,
    /// Metal offset of `Bls12_381G1Projective::x`.
    pub g1_projective_x_offset: u64,
    /// Metal offset of `Bls12_381G1Projective::y`.
    pub g1_projective_y_offset: u64,
    /// Metal offset of `Bls12_381G1Projective::z`.
    pub g1_projective_z_offset: u64,
    /// Metal `sizeof` for the internal complete-formula homogeneous point.
    pub g1_homogeneous_size: u64,
    /// Metal `alignof` for the internal complete-formula homogeneous point.
    pub g1_homogeneous_alignment: u64,
    /// Metal offset of the internal homogeneous x-coordinate.
    pub g1_homogeneous_x_offset: u64,
    /// Metal offset of the internal homogeneous y-coordinate.
    pub g1_homogeneous_y_offset: u64,
    /// Metal offset of the internal homogeneous z-coordinate.
    pub g1_homogeneous_z_offset: u64,
    /// Pinned FNV-1a fingerprint of the preceding layout words.
    pub layout_fingerprint: u64,
    /// Fingerprint of every generated Fr/Fp Montgomery constant.
    pub constants_fingerprint: u64,
}

/// Expression-kernel layout and tag values reported by the compiled shader.
///
/// Layout arrays begin with `sizeof` and `alignof`, followed by member offsets
/// in declaration order. The value-reference members are `kind`, `a`, `b`;
/// operation members are `opcode`, `target`, `parts_offset`, `parts_len`,
/// `lhs`, `rhs`; and parameter members follow the complete declaration order
/// frozen in `expression.metal`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphAbiProbe {
    /// Version of the expression shader ABI.
    pub abi_version: u64,
    /// `GraphValueRef` size, alignment, and three member offsets.
    pub value_ref_layout: [u64; 5],
    /// `GraphOperation` size, alignment, and six member offsets.
    pub operation_layout: [u64; 8],
    /// `GraphEvalParameters` size, alignment, and all 29 member offsets.
    pub parameters_layout: [u64; 31],
    /// Numeric tags for constant through previous-value sources.
    pub value_tags: [u64; 12],
    /// Numeric tags for add through store operations.
    pub operation_tags: [u64; 8],
    /// Numeric tags for store, lookup, and trash post-processing.
    pub post_kinds: [u64; 3],
    /// Private-field scratch capacities with compiled evaluator entry points.
    pub scratch_classes: [u64; 6],
    /// FNV-1a fingerprint of all preceding probe words.
    pub layout_fingerprint: u64,
}

/// Batched correctness-probe outputs for the four initial Fr64 operations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381Fr64ProbeResult {
    /// Pairwise modular sums.
    pub add: Vec<Bls12_381Fr>,
    /// Pairwise modular differences.
    pub sub: Vec<Bls12_381Fr>,
    /// Pairwise Montgomery products.
    pub mul: Vec<Bls12_381Fr>,
    /// Montgomery squares of the left-hand inputs.
    pub square: Vec<Bls12_381Fr>,
}

/// Batched correctness-probe outputs for the four initial Fp64 operations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381Fp64ProbeResult {
    /// Pairwise modular sums.
    pub add: Vec<Bls12_381Fp>,
    /// Pairwise modular differences.
    pub sub: Vec<Bls12_381Fp>,
    /// Pairwise Montgomery products.
    pub mul: Vec<Bls12_381Fp>,
    /// Montgomery squares of the left-hand inputs.
    pub square: Vec<Bls12_381Fp>,
}

/// Canonical affine outputs from the initial batched G1 correctness probe.
///
/// Affine infinity is encoded with both coordinates zero. The probe accepts
/// raw reduced Montgomery coordinates and does not validate curve or subgroup
/// membership.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381G1ProbeResult {
    /// Canonical affine form of each projective input.
    pub normalized: Vec<Bls12_381G1Affine>,
    /// Canonical affine form after projective doubling.
    pub doubled: Vec<Bls12_381G1Affine>,
    /// Canonical affine form after adding the paired affine input.
    pub mixed_sum: Vec<Bls12_381G1Affine>,
}

/// Output and isolated timing from a homogeneous complete-add chain probe.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381G1ChainProbeResult {
    /// Public-Jacobian form of the final homogeneous accumulator values.
    pub projective: Vec<Bls12_381G1Projective>,
    /// Wall time covering only the complete-add command submissions and waits.
    pub timed_duration: std::time::Duration,
    /// Number of command buffers included in [`Self::timed_duration`].
    pub timed_commands: u32,
    /// Number of homogeneous-chain dispatches included in the timed command.
    pub timed_dispatches: u32,
}

/// Deterministic summary used to compare CPU and accelerator adapter output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProofByteDigest {
    /// Number of proof bytes consumed.
    pub len: usize,
    /// XOR of every proof byte.
    pub xor: u8,
    /// Wrapping sum of every proof byte.
    pub sum64: u64,
    /// Position-sensitive wrapping polynomial digest.
    pub rolling: u64,
}

/// Cumulative Metal activity since session creation or the last reset.
///
/// Host read/write counters record logical CPU-visible buffer accesses on
/// unified memory. They must not be interpreted as PCIe transfer volume.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalActivity {
    /// Command buffers created by the session.
    pub command_buffers: u64,
    /// Compute dispatches encoded by the session.
    pub dispatches: u64,
    /// Explicit waits caused by a CPU dependency.
    pub host_waits: u64,
    /// Shared or private Metal allocations created by the session.
    pub allocations: u64,
    /// Cumulative bytes allocated by the session.
    pub allocated_bytes: u64,
    /// Bytes explicitly written through a CPU-visible mapping.
    pub host_write_bytes: u64,
    /// Bytes explicitly read through a CPU-visible mapping.
    pub host_read_bytes: u64,
    /// Bytes moved by Metal blit encoders.
    pub blit_bytes: u64,
    /// Allocation bytes currently reported by the Metal device.
    pub current_allocated_bytes: u64,
    /// Command buffers that completed with a Metal error.
    pub execution_failures: u64,
    /// Standalone MSM command submissions attempted on Metal after coalescing.
    pub msm_metal_submissions: u64,
    /// MSM outputs successfully produced by Metal.
    pub msm_metal_outputs: u64,
    /// Logical scalar positions submitted to Metal MSMs, including zeros.
    pub msm_metal_logical_scalars: u64,
    /// Nonzero scalar positions submitted to Metal MSMs.
    pub msm_metal_nonzero_scalars: u64,
    /// Standalone Metal MSM submissions that failed closed.
    pub msm_metal_failures: u64,
    /// Nanoseconds spent in the serialized standalone Metal MSM engine.
    pub msm_metal_service_ns: u64,
    /// Nanoseconds the worker intentionally held submissions for compatible batching.
    pub msm_metal_coalesce_ns: u64,
    /// Cumulative caller-thread nanoseconds converting Montgomery scalars to canonical limbs.
    ///
    /// Multiple proof threads can overlap, so this is work rather than a wall-time interval.
    pub msm_metal_canonicalization_ns: u64,
    /// MSM outputs deliberately assigned to the bounded CPU partition.
    ///
    /// These are policy decisions made before Metal selection, not fallbacks.
    pub msm_cpu_outputs: u64,
    /// Logical scalar positions assigned to the bounded CPU MSM partition.
    pub msm_cpu_logical_scalars: u64,
    /// Nonzero scalar positions assigned to the bounded CPU MSM partition.
    pub msm_cpu_nonzero_scalars: u64,
    /// Nanoseconds spent executing CPU MSMs while holding the hybrid partition lock.
    pub msm_cpu_service_ns: u64,
    /// Cumulative caller-thread nanoseconds waiting to acquire the CPU MSM lock.
    ///
    /// Waits from multiple proof threads can overlap, so this is not wall time.
    pub msm_cpu_lock_wait_ns: u64,
    /// Exact-size secret-buffer pool checkouts served without allocation.
    pub pool_hits: u64,
    /// Exact-size secret-buffer pool checkouts requiring allocation.
    pub pool_misses: u64,
    /// Cleared secret buffers returned to the bounded pool.
    pub pool_returns: u64,
    /// Command buffers created specifically for secret erasure.
    pub clear_commands: u64,
    /// Secret-erasure kernel dispatches encoded by the session.
    pub clear_dispatches: u64,
    /// Host waits used to prove secret-erasure command completion.
    pub clear_waits: u64,
    /// Full-capacity secret bytes whose GPU clear completed successfully.
    pub cleared_bytes: u64,
    /// Secret-erasure attempts that could not be proven complete.
    pub clear_failures: u64,
    /// Transitions that permanently poisoned the session.
    pub poison_events: u64,
    /// Request-scoped resident arena identities issued by the session.
    pub resident_generations: u64,
    /// Resident handles rejected because their arena or generation was stale.
    pub stale_handle_rejections: u64,
    /// Explicit resident-buffer materializations to host memory.
    pub materializations: u64,
    /// Materializations containing a complete resident polynomial batch.
    pub whole_polynomial_materializations: u64,
    /// Secret allocation bytes currently owned by the session.
    pub secret_live_bytes: u64,
    /// Maximum secret allocation bytes concurrently owned by the session.
    pub secret_peak_live_bytes: u64,
    /// Cleared secret bytes currently cached for exact-size reuse.
    pub pooled_secret_bytes: u64,
    /// Secret bytes retained because their clear could not be proven complete.
    pub quarantined_secret_bytes: u64,
    /// Immutable key-context cache lookups served by a live context.
    pub key_cache_hits: u64,
    /// Immutable key-context cache lookups requiring a physical build.
    pub key_cache_misses: u64,
    /// Immutable key contexts successfully built by the session.
    pub key_context_builds: u64,
    /// Immutable key-context builds that failed after validation.
    pub key_context_build_failures: u64,
    /// Diagnostic resident quotient epochs that reached and returned from the
    /// callback-scoped CPU MSM boundary.
    pub quotient_epochs: u64,
    /// Nanoseconds spent in successful diagnostic quotient epochs, measured
    /// from request validation through mandatory secret-buffer cleanup.
    pub quotient_epoch_elapsed_ns: u64,
    /// Diagnostic resident quotient epochs rejected or failed after entry.
    pub quotient_epoch_failures: u64,
    /// Nanoseconds spent in diagnostic quotient epochs that failed after
    /// entry, including any cleanup attempted before returning the error.
    pub quotient_epoch_failure_elapsed_ns: u64,
    /// Explicit resident-buffer materializations attributed to quotient epochs.
    pub quotient_materializations: u64,
    /// Whole-polynomial materializations attributed to quotient epochs. The
    /// discarded shadow prefix requires this to remain zero.
    pub quotient_whole_polynomial_materializations: u64,
    /// Callback-scoped CPU MSM leases issued by diagnostic quotient epochs.
    pub cpu_msm_leases: u64,
    /// Exact quotient-scalar bytes exposed through CPU MSM leases.
    pub cpu_msm_lease_bytes: u64,
    /// Callback-scoped quotient leases consumed without CPU scalar access.
    pub resident_msm_leases: u64,
    /// Exact quotient-scalar bytes retained through resident MSM leases.
    pub resident_msm_lease_bytes: u64,
    /// Immutable key bytes currently owned by live contexts.
    pub immutable_key_live_bytes: u64,
    /// Maximum immutable key bytes concurrently owned by live contexts.
    pub immutable_key_peak_bytes: u64,
    /// Number of immutable key contexts with at least one strong handle.
    pub live_key_contexts: u64,
}

/// Stable device properties useful for diagnostics and benchmark records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceInfo {
    /// Human-readable Metal device name.
    pub name: String,
    /// Stable registry identity reported by Metal, or zero when unavailable.
    ///
    /// A zero value must never be replaced with the device name for cache
    /// identity; caches keyed by such a device remain local to one session.
    pub registry_id: u64,
    /// Whether the device reports unified CPU/GPU memory.
    pub unified_memory: bool,
    /// Driver-recommended maximum working-set size.
    pub recommended_working_set_bytes: u64,
}

/// Result of probing the Metal runtime and the proof implementation separately.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalStatus {
    runtime_compiled: bool,
    runtime_available: bool,
    proof_backend_available: bool,
    device: Option<MetalDeviceInfo>,
    runtime_unavailable_reason: Option<String>,
    proof_backend_unavailable_reason: &'static str,
}

impl MetalStatus {
    /// Whether this crate was built with its embedded Metal shader library.
    pub fn runtime_compiled(&self) -> bool {
        self.runtime_compiled
    }

    /// Whether a device, queue, shader library, and smoke pipelines loaded.
    pub fn runtime_available(&self) -> bool {
        self.runtime_available
    }

    /// Whether the complete proof path is implemented by Metal.
    pub fn proof_backend_available(&self) -> bool {
        self.proof_backend_available
    }

    /// Information about the probed device, when runtime initialization worked.
    pub fn device(&self) -> Option<&MetalDeviceInfo> {
        self.device.as_ref()
    }

    /// Why runtime initialization failed, if it did.
    pub fn runtime_unavailable_reason(&self) -> Option<&str> {
        self.runtime_unavailable_reason.as_deref()
    }

    /// Why the complete proof backend is unavailable.
    pub fn proof_backend_unavailable_reason(&self) -> &'static str {
        self.proof_backend_unavailable_reason
    }

    /// The first applicable runtime or proof-backend failure reason.
    pub fn unavailable_reason(&self) -> Option<&str> {
        self.runtime_unavailable_reason()
            .or(Some(self.proof_backend_unavailable_reason))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    pub(crate) fn runtime_ready(device: MetalDeviceInfo) -> Self {
        Self {
            runtime_compiled: true,
            runtime_available: true,
            proof_backend_available: false,
            device: Some(device),
            runtime_unavailable_reason: None,
            proof_backend_unavailable_reason: "Metal proof execution kernels are not implemented",
        }
    }

    pub(crate) fn unavailable(runtime_compiled: bool, error: &MetalError) -> Self {
        Self {
            runtime_compiled,
            runtime_available: false,
            proof_backend_available: false,
            device: None,
            runtime_unavailable_reason: Some(error.to_string()),
            proof_backend_unavailable_reason: "Metal proof execution kernels are not implemented",
        }
    }
}

/// Errors from explicit Metal initialization, encoding, and execution.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    /// Metal cannot be used on the current build target.
    #[error("Metal is unavailable: {reason}")]
    Unavailable {
        /// Target-specific explanation.
        reason: &'static str,
    },
    /// No default Metal device could be created.
    #[error("Metal device creation failed")]
    Device,
    /// The selected device could not create a command queue.
    #[error("Metal command queue creation failed")]
    Queue,
    /// The embedded shader library could not be loaded.
    #[error("Metal shader library load failed: {0}")]
    Library(String),
    /// A required shader function is absent from the embedded library.
    #[error("Metal shader function {0} is missing")]
    Function(&'static str),
    /// A compute pipeline could not be created.
    #[error("Metal compute pipeline creation failed for {name}: {reason}")]
    Pipeline {
        /// Shader function name.
        name: &'static str,
        /// Driver-provided failure detail.
        reason: String,
    },
    /// A buffer allocation failed.
    #[error("Metal buffer allocation failed for {bytes} bytes")]
    Buffer {
        /// Requested byte length.
        bytes: usize,
    },
    /// An input length cannot be represented by the shader ABI.
    #[error("Metal input length {0} exceeds the shader ABI")]
    LengthOverflow(usize),
    /// Two paired batched inputs have different element counts.
    #[error("Metal batch length mismatch: left has {left} elements, right has {right}")]
    LengthMismatch {
        /// Left-hand input length.
        left: usize,
        /// Right-hand input length.
        right: usize,
    },
    /// A CPU-side mapping attempted to cross a buffer boundary.
    #[error("Metal buffer access of {requested} bytes exceeds {available} allocated bytes")]
    BufferBounds {
        /// Requested byte length.
        requested: usize,
        /// Allocated byte length.
        available: usize,
    },
    /// A command buffer could not be created.
    #[error("Metal command buffer creation failed")]
    CommandBuffer,
    /// A compute encoder could not be created.
    #[error("Metal compute encoder creation failed")]
    Encoder,
    /// An explicit threadgroup geometry is invalid for the selected pipeline.
    #[error("Metal dispatch geometry is invalid: {0}")]
    Dispatch(&'static str),
    /// A submitted command failed on the device.
    #[error("Metal command execution failed: {0}")]
    Execution(String),
    /// Device output violated the documented digest ABI.
    #[error("Metal digest output is invalid: {0}")]
    DigestAbi(&'static str),
    /// Device and host disagree about the BLS12-381 raw-memory contract.
    #[error("Metal BLS12-381 ABI validation failed: {0}")]
    Bls12_381Abi(String),
    /// Device and host disagree about an NTT parameter-block layout.
    #[error("Metal BLS12-381 NTT parameter ABI validation failed: {0}")]
    NttAbi(String),
    /// Device and host disagree about the expression-kernel ABI.
    #[error("Metal graph-expression ABI validation failed: {0}")]
    GraphAbi(String),
    /// A diagnostic graph-evaluation request violates the checked shader contract.
    #[error("Metal graph-expression input is invalid: {0}")]
    GraphEvaluation(String),
    /// A diagnostic resident quotient request or epoch transition is invalid.
    #[error("Metal diagnostic quotient epoch is invalid: {0}")]
    QuotientEpoch(String),
    /// A resident KZG multi-opening request or transition is invalid.
    #[error("Metal resident multi-open is invalid: {0}")]
    Multiopen(String),
    /// Standalone Metal MSM preparation, routing, execution, or worker failure.
    #[error("Metal BLS12-381 MSM failed: {0}")]
    Msm(String),
    /// Immutable graph/fixed key-context input or cache state is invalid.
    #[error("Metal immutable key context is invalid: {0}")]
    KeyContext(String),
    /// An immutable key context belongs to a different Metal session.
    #[error("Metal immutable key context belongs to a different session")]
    ForeignKeyContext,
    /// A supplied scalar-field NTT domain violates its algebraic contract.
    #[error("Metal BLS12-381 NTT domain is invalid: {0}")]
    NttDomain(&'static str),
    /// NTT inputs do not contain complete fixed-size batches.
    #[error(
        "Metal BLS12-381 NTT input has {elements} elements, not complete {batch_len}-element batches"
    )]
    NttBatch {
        /// Total input element count.
        elements: usize,
        /// Required elements per batch.
        batch_len: usize,
    },
    /// A coefficient batch cannot be embedded in the requested extended domain.
    #[error(
        "Metal coefficient batch size {coefficients_per_batch} cannot embed {elements} elements in extended size {extended_len}"
    )]
    NttCoefficientBatch {
        /// Total coefficient element count.
        elements: usize,
        /// Coefficients in each packed input polynomial.
        coefficients_per_batch: usize,
        /// Elements in each extended-domain polynomial.
        extended_len: usize,
    },
    /// An opaque resident batch handle is stale or inconsistent with its arena.
    #[error("Metal resident scalar-field state is invalid: {0}")]
    ResidentState(&'static str),
    /// A prior lifecycle failure made further work unsafe.
    #[error("Metal session is poisoned: {reason}")]
    SessionPoisoned {
        /// First lifecycle failure that made reuse unsafe.
        reason: String,
    },
}

/// Probes runtime/device availability without claiming proof acceleration.
pub fn metal_status() -> MetalStatus {
    static STATUS: OnceLock<MetalStatus> = OnceLock::new();
    STATUS.get_or_init(platform::probe_status).clone()
}

/// Computes the canonical CPU reference for the Metal adapter smoke boundary.
pub fn proof_byte_digest_cpu(bytes: &[u8]) -> ProofByteDigest {
    const FNV64_PRIME: u64 = 0x100_0000_01b3;

    let mut digest = ProofByteDigest {
        len: bytes.len(),
        ..ProofByteDigest::default()
    };
    for (index, byte) in bytes.iter().copied().enumerate() {
        digest.xor ^= byte;
        digest.sum64 = digest.sum64.wrapping_add(u64::from(byte));
        digest.rolling = digest
            .rolling
            .wrapping_mul(FNV64_PRIME)
            .wrapping_add(u64::from(byte))
            .wrapping_add(index as u64);
    }
    digest
}

static SHARED_SESSION: OnceLock<MetalSession> = OnceLock::new();

fn shared_session() -> Result<&'static MetalSession, MetalError> {
    if let Some(session) = SHARED_SESSION.get() {
        return Ok(session);
    }
    let candidate = MetalSession::new()?;
    let _ = SHARED_SESSION.set(candidate);
    Ok(SHARED_SESSION
        .get()
        .expect("shared Metal session was initialized"))
}

/// Returns the process Metal session for the feature-gated quotient diagnostic.
///
/// This is intentionally a narrow integration seam for the proof adapter. It
/// does not enable a production proof route, and callers must still use the
/// checked diagnostic quotient API rather than issuing arbitrary Metal work.
#[doc(hidden)]
pub fn diagnostic_shared_session() -> Result<&'static MetalSession, MetalError> {
    shared_session()
}

/// Executes packed canonical-domain BLS12-381 Fr NTTs without CPU fallback.
pub fn bls12_381_ntt_metal(
    values: &[Bls12_381Fr],
    log_size: u32,
    direction: NttDirection,
) -> Result<Vec<Bls12_381Fr>, MetalError> {
    shared_session()?.execute_bls12_381_ntt(values, log_size, direction)
}

/// Executes a packed coefficient-to-extended-coset transform on Metal.
pub fn bls12_381_coeff_to_extended_metal(
    coefficients: &[Bls12_381Fr],
    coefficients_per_batch: usize,
    extended_log_size: u32,
) -> Result<Vec<Bls12_381Fr>, MetalError> {
    shared_session()?.execute_bls12_381_coeff_to_extended(
        coefficients,
        coefficients_per_batch,
        extended_log_size,
    )
}

/// Returns process-session activity, or zeros before the first native call.
pub fn shared_metal_activity() -> MetalActivity {
    SHARED_SESSION
        .get()
        .map(MetalSession::activity)
        .unwrap_or_default()
}

/// Resets process-session diagnostic counters without creating a Metal device.
pub fn reset_shared_metal_activity() {
    if let Some(session) = SHARED_SESSION.get() {
        session.reset_activity();
    }
}

/// Records one intentional CPU MSM partition decision for route accounting.
///
/// This internal integration hook does not execute an MSM and must be called
/// only after the CPU result succeeds. It distinguishes a configured hybrid
/// policy from a post-selection Metal fallback.
#[doc(hidden)]
pub fn record_cpu_msm_partition(logical_scalars: usize, nonzero_scalars: usize) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    metal::record_cpu_msm_partition(logical_scalars, nonzero_scalars);
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders)))]
    {
        let _ = (logical_scalars, nonzero_scalars);
    }
}

/// Records timing for one lock acquisition covering one or more intentional CPU MSMs.
///
/// This diagnostic hook records no secret values. `service_ns` is lock-held
/// time, while `lock_wait_ns` is cumulative caller wait and may overlap across
/// proof threads.
#[doc(hidden)]
pub fn record_cpu_msm_timing(lock_wait_ns: u64, service_ns: u64) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders))]
    metal::record_cpu_msm_timing(lock_wait_ns, service_ns);
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", midnight_metal_shaders)))]
    {
        let _ = (lock_wait_ns, service_ns);
    }
}

/// Computes a digest on the process Metal session without CPU fallback.
pub fn proof_byte_digest_metal(bytes: &[u8]) -> Result<ProofByteDigest, MetalError> {
    shared_session()?.proof_byte_digest(bytes)
}

impl fmt::Display for ProofByteDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "len={} xor={:#04x} sum64={} rolling={:#018x}",
            self.len, self.xor, self.sum64, self.rolling
        )
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::*;

    #[test]
    fn cpu_digest_is_deterministic_and_position_sensitive() {
        let first = proof_byte_digest_cpu(b"proof-bytes");
        let same = proof_byte_digest_cpu(b"proof-bytes");
        let changed = proof_byte_digest_cpu(b"proof-byteT");
        let reordered = proof_byte_digest_cpu(b"bytes-proof");

        assert_eq!(first, same);
        assert_ne!(first, changed);
        assert_ne!(first, reordered);
        assert_eq!(first.len, 11);
    }

    #[test]
    fn status_never_claims_full_proof_execution() {
        let status = metal_status();

        assert!(!status.proof_backend_available());
        assert!(status.unavailable_reason().is_some());
    }

    #[test]
    fn bls12_381_raw_abi_matches_cuda_and_blst() {
        assert_eq!(size_of::<Bls12_381Fr>(), 32);
        assert_eq!(align_of::<Bls12_381Fr>(), 8);
        assert_eq!(size_of::<Bls12_381Fp>(), 48);
        assert_eq!(align_of::<Bls12_381Fp>(), 8);
        assert_eq!(size_of::<Bls12_381G1Affine>(), 96);
        assert_eq!(align_of::<Bls12_381G1Affine>(), 8);
        assert_eq!(size_of::<Bls12_381G1Projective>(), 144);
        assert_eq!(align_of::<Bls12_381G1Projective>(), 8);
    }

    #[test]
    fn generated_montgomery_constants_match_blst() {
        let mut fr_one = blst::blst_fr::default();
        let canonical_fr_one = [1, 0, 0, 0];
        // SAFETY: the output and four-limb canonical input are valid BLST
        // objects for the duration of the call.
        unsafe {
            blst::blst_fr_from_uint64(&mut fr_one, canonical_fr_one.as_ptr());
        }
        assert_eq!(fr_one.l, generated::FR_R);

        let mut fr_r2 = blst::blst_fr::default();
        // SAFETY: generated `FR_R` is a four-limb canonical integer and both
        // pointers remain valid for the duration of the call.
        unsafe {
            blst::blst_fr_from_uint64(&mut fr_r2, generated::FR_R.as_ptr());
        }
        assert_eq!(fr_r2.l, generated::FR_R2);
        assert_eq!(
            generated::FR_MODULUS[0].wrapping_mul(generated::FR_INV),
            u64::MAX
        );

        let mut fp_one = blst::blst_fp::default();
        let canonical_fp_one = [1, 0, 0, 0, 0, 0];
        // SAFETY: the output and six-limb canonical input are valid BLST
        // objects for the duration of the call.
        unsafe {
            blst::blst_fp_from_uint64(&mut fp_one, canonical_fp_one.as_ptr());
        }
        assert_eq!(fp_one.l, generated::FP_R);

        let mut fp_r2 = blst::blst_fp::default();
        // SAFETY: generated `FP_R` is a six-limb canonical integer and both
        // pointers remain valid for the duration of the call.
        unsafe {
            blst::blst_fp_from_uint64(&mut fp_r2, generated::FP_R.as_ptr());
        }
        assert_eq!(fp_r2.l, generated::FP_R2);
        assert_eq!(
            generated::FP_MODULUS[0].wrapping_mul(generated::FP_INV),
            u64::MAX
        );

        for index in 0..6 {
            assert_eq!(
                generated::FP32_MODULUS[2 * index],
                generated::FP_MODULUS[index] as u32
            );
            assert_eq!(
                generated::FP32_MODULUS[2 * index + 1],
                (generated::FP_MODULUS[index] >> 32) as u32
            );
            assert_eq!(generated::FP32_R[2 * index], generated::FP_R[index] as u32);
            assert_eq!(
                generated::FP32_R[2 * index + 1],
                (generated::FP_R[index] >> 32) as u32
            );
            assert_eq!(
                generated::FP32_R2[2 * index],
                generated::FP_R2[index] as u32
            );
            assert_eq!(
                generated::FP32_R2[2 * index + 1],
                (generated::FP_R2[index] >> 32) as u32
            );
        }
        assert_eq!(generated::FP32_INV, generated::FP_INV as u32);
        assert_eq!(
            generated::FP32_MODULUS[0].wrapping_mul(generated::FP32_INV),
            u32::MAX
        );
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn non_apple_surface_fails_explicitly() {
        let error = proof_byte_digest_metal(b"proof").expect_err("Metal must be unavailable");

        assert!(matches!(error, MetalError::Unavailable { .. }));
        assert!(!metal_status().runtime_available());
    }
}
