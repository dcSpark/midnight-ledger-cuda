// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

//! Backend-neutral entry point for the resident BLS12-381 prover.
//!
//! The proof crates use a compile-time selected implementation. This avoids
//! threading trait objects and backend-associated buffer types through PLONK
//! while keeping the alternative, separately built CUDA and Metal lanes behind
//! one target-selected API boundary. They never form one runtime backend.

#![deny(missing_docs)]
#![deny(unreachable_pub)]

/// Checked expression graph ABI shared by resident backends.
pub mod expression;
/// Backend-neutral immutable proving-key and SRS preparation.
pub mod prepared_key;

pub use prepared_key::{
    PreparedKeyError, PreparedResidentProvingKey, PreparedResidentProvingKeyInput,
    PreparedResidentSrs, RawG1Affine, ResidentColumnKind, ResidentColumnRef,
    ResidentDomainManifest, ResidentDomainManifestInput,
};

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!(
    "midnight-gpu-prover implementation selectors are mutually exclusive: CUDA is a separate Linux reference backend, while the macOS development route is CPU+Metal; do not enable `cuda` and `metal` together"
);

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
static NEXT_PROOF_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Raw BLS12-381 scalar-field element shared by resident backend ABIs.
///
/// Limbs are little-endian, reduced, and in Montgomery form. This is an
/// internal-memory ABI, not a canonical scalar serialization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct RawFr {
    /// Four little-endian 64-bit Montgomery limbs.
    pub limbs: [u64; 4],
}

/// Semantic class of one coefficient-polynomial group entering the quotient epoch.
///
/// The declaration order is the protocol upload order. It is deliberately not
/// caller-configurable: changing it would also change the dynamic-column offsets
/// consumed by expression and permutation kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotientCoefficientSourceKind {
    /// Advice polynomials in the proving key's compact advice-column order.
    Advice,
    /// Instance polynomials in the proving key's compact instance-column order.
    Instance,
    /// Permutation product polynomials in set order.
    PermutationProducts,
    /// Lookup product polynomials in lookup-argument order.
    LookupProducts,
    /// Permuted lookup-input polynomials in lookup-argument order.
    LookupPermutedInputs,
    /// Permuted lookup-table polynomials in lookup-argument order.
    LookupPermutedTables,
    /// Trash polynomials in trash-argument order.
    Trash,
}

/// One borrowed coefficient-polynomial group in the fixed quotient source order.
#[derive(Clone, Copy, Debug)]
pub struct QuotientCoefficientSourceGroup<'a> {
    kind: QuotientCoefficientSourceKind,
    columns: &'a [&'a [RawFr]],
}

impl<'a> QuotientCoefficientSourceGroup<'a> {
    /// Semantic source class for these columns.
    pub const fn kind(self) -> QuotientCoefficientSourceKind {
        self.kind
    }

    /// Separately allocated coefficient columns, each with base-domain length.
    ///
    /// Keeping these as borrowed slices lets the Metal backend copy each column
    /// directly into its mapped request arena without constructing a concatenated
    /// host field vector.
    pub const fn columns(self) -> &'a [&'a [RawFr]] {
        self.columns
    }
}

/// Borrowed dynamic coefficient columns for one diagnostic quotient epoch.
///
/// Every category has a named field and [`Self::groups_in_protocol_order`]
/// emits the one permitted layout. The facade never concatenates the field
/// elements on the host; only a small vector of slice metadata may be built by
/// the concrete adapter.
#[derive(Clone, Copy, Debug)]
pub struct QuotientCoefficientSources<'a> {
    advice: &'a [&'a [RawFr]],
    instance: &'a [&'a [RawFr]],
    permutation_products: &'a [&'a [RawFr]],
    lookup_products: &'a [&'a [RawFr]],
    lookup_permuted_inputs: &'a [&'a [RawFr]],
    lookup_permuted_tables: &'a [&'a [RawFr]],
    trash: &'a [&'a [RawFr]],
}

impl<'a> QuotientCoefficientSources<'a> {
    /// Creates the seven protocol-classified dynamic source groups.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        advice: &'a [&'a [RawFr]],
        instance: &'a [&'a [RawFr]],
        permutation_products: &'a [&'a [RawFr]],
        lookup_products: &'a [&'a [RawFr]],
        lookup_permuted_inputs: &'a [&'a [RawFr]],
        lookup_permuted_tables: &'a [&'a [RawFr]],
        trash: &'a [&'a [RawFr]],
    ) -> Self {
        Self {
            advice,
            instance,
            permutation_products,
            lookup_products,
            lookup_permuted_inputs,
            lookup_permuted_tables,
            trash,
        }
    }

    /// Returns all groups in the exact dynamic-column order consumed by Metal.
    pub const fn groups_in_protocol_order(self) -> [QuotientCoefficientSourceGroup<'a>; 7] {
        [
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::Advice,
                columns: self.advice,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::Instance,
                columns: self.instance,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::PermutationProducts,
                columns: self.permutation_products,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::LookupProducts,
                columns: self.lookup_products,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::LookupPermutedInputs,
                columns: self.lookup_permuted_inputs,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::LookupPermutedTables,
                columns: self.lookup_permuted_tables,
            },
            QuotientCoefficientSourceGroup {
                kind: QuotientCoefficientSourceKind::Trash,
                columns: self.trash,
            },
        ]
    }
}

/// Challenges, dynamic columns, and quotient blinds for one proof's diagnostic epoch.
///
/// The immutable proving key and SRS are retained here so the concrete adapter
/// can bind the request to the exact schema-v2 Metal context before allocating
/// or submitting any proof-secret work.
#[derive(Clone, Copy, Debug)]
pub struct DiagnosticQuotientEpochInput<'a> {
    key: &'a PreparedResidentProvingKey,
    srs: &'a PreparedResidentSrs,
    sources: QuotientCoefficientSources<'a>,
    challenges: &'a [RawFr],
    beta: RawFr,
    gamma: RawFr,
    theta: RawFr,
    trash_challenge: RawFr,
    y: RawFr,
    quotient_blinds: &'a [RawFr],
}

impl<'a> DiagnosticQuotientEpochInput<'a> {
    /// Creates a proof-scoped diagnostic quotient request.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        key: &'a PreparedResidentProvingKey,
        srs: &'a PreparedResidentSrs,
        sources: QuotientCoefficientSources<'a>,
        challenges: &'a [RawFr],
        beta: RawFr,
        gamma: RawFr,
        theta: RawFr,
        trash_challenge: RawFr,
        y: RawFr,
        quotient_blinds: &'a [RawFr],
    ) -> Self {
        Self {
            key,
            srs,
            sources,
            challenges,
            beta,
            gamma,
            theta,
            trash_challenge,
            y,
            quotient_blinds,
        }
    }

    /// Prepared proving key whose complete fingerprint must match the Metal context.
    pub const fn key(self) -> &'a PreparedResidentProvingKey {
        self.key
    }

    /// Prepared SRS whose complete fingerprint must match the Metal context.
    pub const fn srs(self) -> &'a PreparedResidentSrs {
        self.srs
    }

    /// Protocol-classified dynamic coefficient columns.
    pub const fn sources(self) -> QuotientCoefficientSources<'a> {
        self.sources
    }

    /// Proof challenges addressed by compiled expression graphs.
    pub const fn challenges(self) -> &'a [RawFr] {
        self.challenges
    }

    /// Permutation and lookup challenge beta.
    pub const fn beta(self) -> RawFr {
        self.beta
    }

    /// Permutation and lookup challenge gamma.
    pub const fn gamma(self) -> RawFr {
        self.gamma
    }

    /// Lookup compression challenge theta.
    pub const fn theta(self) -> RawFr {
        self.theta
    }

    /// Trash compression challenge.
    pub const fn trash_challenge(self) -> RawFr {
        self.trash_challenge
    }

    /// Numerator identity-combination challenge.
    pub const fn y(self) -> RawFr {
        self.y
    }

    /// The exactly `quotient_piece_count - 1` blinds sampled by the CPU prover.
    pub const fn quotient_blinds(self) -> &'a [RawFr] {
        self.quotient_blinds
    }
}

/// Borrowed graph/fixed proving-key material prepared outside a proof request.
///
/// This first immutable-key slice contains only public graph metadata, compact
/// fixed extended-coset columns, and the `l0 | l_last | l_active` selectors.
/// It contains no advice, instance, quotient, multi-open, or MSM secret state.
#[derive(Clone, Copy, Debug)]
pub struct GraphFixedKeyInput<'a> {
    n: u32,
    n_ext: u32,
    graphs: &'a expression::CompiledProtocolGraphFamily,
    fixed_cosets: &'a [RawFr],
    lagrange_selectors: &'a [RawFr],
}

impl<'a> GraphFixedKeyInput<'a> {
    /// Creates immutable graph/fixed key material for backend preparation.
    pub const fn new(
        n: u32,
        n_ext: u32,
        graphs: &'a expression::CompiledProtocolGraphFamily,
        fixed_cosets: &'a [RawFr],
        lagrange_selectors: &'a [RawFr],
    ) -> Self {
        Self {
            n,
            n_ext,
            graphs,
            fixed_cosets,
            lagrange_selectors,
        }
    }

    /// Base polynomial-domain length.
    pub const fn n(self) -> u32 {
        self.n
    }

    /// Extended polynomial-domain length.
    pub const fn n_ext(self) -> u32 {
        self.n_ext
    }

    /// Canonically packed neutral graph family.
    pub const fn graphs(self) -> &'a expression::CompiledProtocolGraphFamily {
        self.graphs
    }

    /// Protocol-category ranges derived while packing the graph family.
    pub const fn sections(self) -> expression::GraphFamilySections {
        self.graphs.sections()
    }

    /// Fixed extended-coset columns in compact-map order.
    pub const fn fixed_cosets(self) -> &'a [RawFr] {
        self.fixed_cosets
    }

    /// Packed `l0`, `l_last`, and `l_active` extended-domain columns.
    pub const fn lagrange_selectors(self) -> &'a [RawFr] {
        self.lagrange_selectors
    }
}

/// Failure from the backend-neutral resident proof boundary.
#[derive(Debug, Eq, PartialEq)]
pub enum ResidentError {
    /// No concrete resident backend feature was selected.
    BackendNotSelected,
    /// The selected backend cannot run on this build target or process.
    BackendUnavailable(&'static str),
    /// The backend exists, but its full resident proof contract is incomplete.
    BackendIncomplete(&'static str),
    /// A concrete backend rejected an operation.
    Backend(String),
    /// The monotonic proof-arena identity space was exhausted.
    ArenaIdentityExhausted,
    /// The monotonic allocation identity space was exhausted.
    AllocationIdentityExhausted,
    /// The total requested element count overflowed `usize`.
    LengthOverflow,
    /// A resident token belongs to a different proof arena.
    ForeignBuffer,
    /// A resident token no longer names the allocation generation it was issued for.
    StaleBuffer,
    /// A resident token's logical range exceeds its allocation.
    BufferBounds,
}

impl std::fmt::Display for ResidentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotSelected => formatter.write_str("no resident GPU backend is selected"),
            Self::BackendUnavailable(reason) => {
                write!(formatter, "resident GPU backend is unavailable: {reason}")
            }
            Self::BackendIncomplete(reason) => {
                write!(formatter, "resident GPU backend is incomplete: {reason}")
            }
            Self::Backend(reason) => write!(formatter, "resident GPU backend failed: {reason}"),
            Self::ArenaIdentityExhausted => {
                formatter.write_str("resident proof-arena identity space is exhausted")
            }
            Self::AllocationIdentityExhausted => {
                formatter.write_str("resident allocation identity space is exhausted")
            }
            Self::LengthOverflow => formatter.write_str("resident element count overflowed usize"),
            Self::ForeignBuffer => {
                formatter.write_str("resident buffer belongs to a different proof arena")
            }
            Self::StaleBuffer => formatter.write_str("resident buffer generation is stale"),
            Self::BufferBounds => {
                formatter.write_str("resident buffer range exceeds its allocation")
            }
        }
    }
}

impl std::error::Error for ResidentError {}

/// Thread-bound owner of one backend proof-execution scope.
///
/// CUDA uses this scope to reserve a private sppark execution lane and pin the
/// MSM profile selected from the polynomial-domain size. Metal deliberately
/// rejects acquisition until its resident proof route is implemented.
#[must_use = "dropping the execution guard releases the proof execution scope"]
pub struct ExecutionGuard {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    _cuda: cuda_backend::SpparkExecutionGuard,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl std::fmt::Debug for ExecutionGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionGuard")
            .finish_non_exhaustive()
    }
}

impl ExecutionGuard {
    /// Acquires the selected backend's proof-execution scope.
    ///
    /// `affinity_key` preserves backend-local context affinity across proofs;
    /// `domain_log_size` selects the immutable MSM tuning profile for the
    /// scope. Diagnostic CUDA overrides are resolved before lane acquisition.
    pub fn acquire(affinity_key: usize, domain_log_size: u32) -> Result<Self, ResidentError> {
        let _ = (affinity_key, domain_log_size);
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let config = cuda_backend::Bls12_381MsmConfig::for_k(domain_log_size)
                .with_diagnostic_overrides()
                .map_err(|error| ResidentError::Backend(error.to_string()))?;
            let cuda = cuda_backend::SpparkExecutionGuard::acquire_with_affinity_and_config(
                affinity_key,
                config,
            )
            .map_err(|error| ResidentError::Backend(error.to_string()))?;
            return Ok(Self {
                _cuda: cuda,
                _thread_bound: std::marker::PhantomData,
            });
        }

        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            return Err(ResidentError::BackendIncomplete(
                "Metal resident proof execution is not implemented",
            ));
        }

        #[cfg(all(
            any(feature = "cuda", feature = "metal"),
            not(any(
                all(feature = "cuda", target_os = "linux"),
                all(feature = "metal", target_os = "macos", target_arch = "aarch64")
            ))
        ))]
        {
            return Err(ResidentError::BackendUnavailable(
                "the selected resident backend does not support this target",
            ));
        }

        #[allow(unreachable_code)]
        Err(ResidentError::BackendNotSelected)
    }
}

#[derive(Debug)]
struct AllocationRecord {
    allocation_id: u64,
    generation: u64,
    len: usize,
}

#[derive(Debug)]
enum BackendArena {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    Cuda(Vec<cuda_backend::Bls12_381FrDeviceVec>),
    #[cfg(test)]
    Test(Vec<Box<[RawFr]>>),
    #[allow(dead_code)]
    Unsupported,
}

impl BackendArena {
    fn upload(&mut self, _values: &[RawFr]) -> Result<(), ResidentError> {
        match self {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            Self::Cuda(allocations) => {
                let allocation =
                    cuda_backend::Bls12_381FrDeviceVec::upload(cuda_raw_slice(_values))
                        .map_err(|error| ResidentError::Backend(error.to_string()))?;
                allocations.push(allocation);
                Ok(())
            }
            #[cfg(test)]
            Self::Test(allocations) => {
                allocations.push(_values.to_vec().into_boxed_slice());
                Ok(())
            }
            Self::Unsupported => Err(ResidentError::BackendUnavailable(
                "the resident arena has no concrete backend",
            )),
        }
    }

    fn upload_many(&mut self, _values: &[&[RawFr]]) -> Result<(), ResidentError> {
        match self {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            Self::Cuda(allocations) => {
                let raw = _values
                    .iter()
                    .map(|values| cuda_raw_slice(values))
                    .collect::<Vec<_>>();
                let allocation = cuda_backend::Bls12_381FrDeviceVec::upload_many(&raw)
                    .map_err(|error| ResidentError::Backend(error.to_string()))?;
                allocations.push(allocation);
                Ok(())
            }
            #[cfg(test)]
            Self::Test(allocations) => {
                allocations.push(
                    _values
                        .iter()
                        .flat_map(|values| values.iter().copied())
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                Ok(())
            }
            Self::Unsupported => Err(ResidentError::BackendUnavailable(
                "the resident arena has no concrete backend",
            )),
        }
    }
}

/// Opaque, non-owning token for scalar-field data owned by a [`ProofArena`].
///
/// The token exposes no pointer, queue, download, or backend arithmetic. Every
/// consuming arena operation validates its arena, allocation, generation, and
/// logical bounds before delegating to a backend.
#[derive(Debug)]
pub struct ResidentFrBuffer {
    arena_id: u64,
    slot: usize,
    allocation_id: u64,
    generation: u64,
    len: usize,
}

impl ResidentFrBuffer {
    /// Logical number of scalar-field elements represented by this token.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the token represents an empty scalar-field buffer.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Request-scoped owner for resident proof-secret allocations.
///
/// The arena is the only object allowed to create or consume buffer tokens.
/// Dropping or explicitly finishing it releases all backend allocations. The
/// current CUDA adapter delegates allocation to sppark; Metal construction is
/// deliberately fail-closed until the general proof arena is connected to the
/// already implemented Metal secret-buffer lifecycle.
#[derive(Debug)]
pub struct ProofArena {
    arena_id: u64,
    next_allocation_id: u64,
    allocations: Vec<AllocationRecord>,
    backend: BackendArena,
}

impl ProofArena {
    /// Creates an arena for the backend selected by the target and feature set.
    pub fn for_current_proof() -> Result<Self, ResidentError> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            return Self::with_backend(BackendArena::Cuda(Vec::new()));
        }

        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            return Err(ResidentError::BackendIncomplete(
                "Metal resident proof-arena routing is not implemented",
            ));
        }

        #[cfg(all(
            any(feature = "cuda", feature = "metal"),
            not(any(
                all(feature = "cuda", target_os = "linux"),
                all(feature = "metal", target_os = "macos", target_arch = "aarch64")
            ))
        ))]
        {
            return Err(ResidentError::BackendUnavailable(
                "the selected resident backend does not support this target",
            ));
        }

        #[allow(unreachable_code)]
        Err(ResidentError::BackendNotSelected)
    }

    #[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
    fn with_backend(backend: BackendArena) -> Result<Self, ResidentError> {
        let arena_id = NEXT_PROOF_ARENA_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| ResidentError::ArenaIdentityExhausted)?;
        Ok(Self {
            arena_id,
            next_allocation_id: 1,
            allocations: Vec::new(),
            backend,
        })
    }

    #[cfg(test)]
    fn for_test() -> Self {
        Self::with_backend(BackendArena::Test(Vec::new()))
            .expect("test arena identity should be available")
    }

    /// Uploads one contiguous scalar-field slice into this arena.
    pub fn upload(&mut self, values: &[RawFr]) -> Result<ResidentFrBuffer, ResidentError> {
        let allocation_id = self.reserve_allocation_id()?;
        self.backend.upload(values)?;
        Ok(self.record_allocation(allocation_id, values.len()))
    }

    /// Uploads several slices as one contiguous resident allocation.
    ///
    /// CUDA converts only the small outer list of slice references; scalar
    /// contents are not copied into an intermediate host vector.
    pub fn upload_many(&mut self, values: &[&[RawFr]]) -> Result<ResidentFrBuffer, ResidentError> {
        let len = values.iter().try_fold(0usize, |len, values| {
            len.checked_add(values.len())
                .ok_or(ResidentError::LengthOverflow)
        })?;
        let allocation_id = self.reserve_allocation_id()?;
        self.backend.upload_many(values)?;
        Ok(self.record_allocation(allocation_id, len))
    }

    fn reserve_allocation_id(&mut self) -> Result<u64, ResidentError> {
        let allocation_id = self.next_allocation_id;
        self.next_allocation_id = allocation_id
            .checked_add(1)
            .ok_or(ResidentError::AllocationIdentityExhausted)?;
        Ok(allocation_id)
    }

    fn record_allocation(&mut self, allocation_id: u64, len: usize) -> ResidentFrBuffer {
        let slot = self.allocations.len();
        // This bounded scaffold never reuses an arena allocation. The first
        // pooling/retire operation must increment this generation before a
        // slot can be issued again; validation is already prepared for it.
        let generation = 1;
        self.allocations.push(AllocationRecord {
            allocation_id,
            generation,
            len,
        });
        ResidentFrBuffer {
            arena_id: self.arena_id,
            slot,
            allocation_id,
            generation,
            len,
        }
    }

    /// Validates that a buffer token still names live data in this arena.
    pub fn validate(&self, buffer: &ResidentFrBuffer) -> Result<(), ResidentError> {
        if buffer.arena_id != self.arena_id {
            return Err(ResidentError::ForeignBuffer);
        }
        let allocation = self
            .allocations
            .get(buffer.slot)
            .ok_or(ResidentError::StaleBuffer)?;
        if allocation.allocation_id != buffer.allocation_id
            || allocation.generation != buffer.generation
        {
            return Err(ResidentError::StaleBuffer);
        }
        if buffer.len > allocation.len {
            return Err(ResidentError::BufferBounds);
        }
        Ok(())
    }

    /// Finishes the request and releases every resident allocation.
    ///
    /// Backend-specific synchronization, erasure, quarantine, and poison rules
    /// are part of each arena implementation. The CUDA adapter currently keeps
    /// its established sppark destruction behavior; Metal remains unavailable
    /// here until it can preserve the stricter Metal lifecycle contract.
    pub fn finish(self) -> Result<(), ResidentError> {
        Ok(())
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_raw_slice(values: &[RawFr]) -> &[cuda_backend::Bls12_381Fr] {
    const {
        assert!(std::mem::size_of::<RawFr>() == std::mem::size_of::<cuda_backend::Bls12_381Fr>());
        assert!(std::mem::align_of::<RawFr>() == std::mem::align_of::<cuda_backend::Bls12_381Fr>());
    }
    // SAFETY: both types are repr(C), contain exactly four u64 limbs with the
    // same Montgomery semantics, and their size/alignment are asserted above.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<cuda_backend::Bls12_381Fr>(),
            values.len(),
        )
    }
}

/// Compile-time accelerator implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    /// Native CUDA/sppark implementation.
    Cuda,
    /// Native Apple Metal implementation.
    Metal,
}

/// Result of probing an accelerator runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeAvailability {
    /// The facade does not eagerly initialize this runtime.
    Unprobed,
    /// Runtime initialization and the required smoke pipelines succeeded.
    Available,
    /// The runtime cannot be used by this process.
    Unavailable,
}

/// Static capabilities of a compiled backend implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    full_proof_pipeline: bool,
    resident_polynomials: bool,
}

impl BackendCapabilities {
    /// Whether all proof stages required by proof-server are implemented.
    pub const fn full_proof_pipeline(self) -> bool {
        self.full_proof_pipeline
    }

    /// Whether proof polynomials can remain device-resident across stages.
    pub const fn resident_polynomials(self) -> bool {
        self.resident_polynomials
    }
}

/// Neutral status returned by every target backend module.
///
/// `execution_routed` is deliberately separate from kernel readiness. A smoke
/// runtime or even a complete kernel set must not make a caller label CPU work
/// as accelerated before the proof entry point has an explicit route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendStatus {
    kind: BackendKind,
    runtime: RuntimeAvailability,
    capabilities: BackendCapabilities,
    execution_routed: bool,
    unavailable_reason: Option<String>,
}

impl BackendStatus {
    /// Backend described by this status.
    pub const fn kind(&self) -> BackendKind {
        self.kind
    }

    /// Runtime probe result.
    pub const fn runtime(&self) -> RuntimeAvailability {
        self.runtime
    }

    /// Static capabilities of the compiled implementation.
    pub const fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    /// Whether the application has an explicit proof execution route.
    pub const fn execution_routed(&self) -> bool {
        self.execution_routed
    }

    /// Whether selecting this backend may execute a full proof.
    ///
    /// An unprobed runtime can still be routed: its first native operation is
    /// responsible for reporting initialization failure. An unavailable
    /// runtime, incomplete proof pipeline, or missing route always rejects the
    /// backend without fallback.
    pub const fn proof_execution_available(&self) -> bool {
        self.execution_routed
            && self.capabilities.full_proof_pipeline
            && !matches!(self.runtime, RuntimeAvailability::Unavailable)
    }

    /// Explanation for an unavailable runtime, pipeline, or execution route.
    pub fn unavailable_reason(&self) -> Option<&str> {
        self.unavailable_reason.as_deref()
    }
}

/// The backend selected for this target and feature set.
///
/// CUDA and Metal are platform-exclusive in the first implementation: Linux
/// selects CUDA and Apple silicon macOS selects Metal.
pub const fn selected_backend() -> Option<BackendKind> {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        return Some(BackendKind::Cuda);
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        return Some(BackendKind::Metal);
    }
    #[allow(unreachable_code)]
    None
}

/// Status of the backend selected for this target and feature set.
pub fn selected_backend_status() -> Option<BackendStatus> {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        return Some(cuda::status());
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        return Some(metal::status());
    }
    #[allow(unreachable_code)]
    None
}

/// CUDA implementation selected on Linux.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod cuda {
    // Compatibility surface for the existing separate CUDA route. New generic
    // resident proof code must use the root ProofArena contract instead.
    pub use cuda_backend::*;

    use crate::{BackendCapabilities, BackendKind, BackendStatus, RuntimeAvailability};

    const EXECUTION_ROUTE_UNAVAILABLE: &str =
        "the accelerator facade does not own the existing CUDA proof route";

    /// Reports the CUDA adapter surface currently owned by this facade.
    ///
    /// CUDA device initialization remains lazy in sppark, so runtime
    /// availability is unprobed. The patched proof crates and proof-server do
    /// have a CUDA route, but it is deliberately not advertised here until the
    /// complete route and its counters are owned behind this neutral boundary.
    pub fn status() -> BackendStatus {
        BackendStatus {
            kind: BackendKind::Cuda,
            runtime: RuntimeAvailability::Unprobed,
            capabilities: BackendCapabilities {
                full_proof_pipeline: false,
                resident_polynomials: false,
            },
            execution_routed: false,
            unavailable_reason: Some(EXECUTION_ROUTE_UNAVAILABLE.to_owned()),
        }
    }
}

/// Metal implementation selected on Apple silicon macOS.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal {
    // Compatibility surface for current Metal probes and the CPU+Metal NTT
    // bridge. Production resident proof code must use the root arena contract.
    pub use metal_backend::{
        Bls12_381Fr, Bls12_381G1Projective, Bls12_381MsmBasis, Bls12_381MsmWindow, MetalActivity,
        MetalError, MetalKeyContext, MetalSession, NttDirection, ProofByteDigest,
        bls12_381_coeff_to_extended_metal, bls12_381_ntt_metal, diagnostic_shared_session,
        proof_byte_digest_metal, record_cpu_msm_partition, record_cpu_msm_timing,
        reset_shared_metal_activity, shared_metal_activity,
    };

    use crate::{
        BackendCapabilities, BackendKind, BackendStatus, DiagnosticQuotientEpochInput,
        GraphFixedKeyInput, PreparedResidentProvingKey, PreparedResidentSrs, RawFr, RawG1Affine,
        ResidentColumnKind, ResidentError, RuntimeAvailability,
    };

    const EXECUTION_ROUTE_UNAVAILABLE: &str =
        "Metal proof execution is not routed into the proof pipeline";

    /// Validates and caches the immutable graph/fixed slice of a proving key.
    ///
    /// Conversion is deliberately fieldwise: the facade never transmutes a
    /// cross-crate Rust type or exposes a Metal buffer. This preparation step
    /// does not acquire a proof execution guard or make Metal proof routing
    /// available.
    pub fn prepare_graph_fixed_key_context(
        session: &MetalSession,
        input: GraphFixedKeyInput<'_>,
    ) -> Result<MetalKeyContext, ResidentError> {
        preflight_graph_fixed_key_input(input)?;
        let graph_arrays = ConvertedMetalGraphArrays::new(input.graphs());
        let fixed_cosets = metal_fr_vec(input.fixed_cosets());
        let lagrange_selectors = metal_fr_vec(input.lagrange_selectors());

        session
            .get_or_create_key_context(metal_graph_fixed_input(
                input.n(),
                input.n_ext(),
                input.graphs(),
                &graph_arrays,
                &fixed_cosets,
                &lagrange_selectors,
            ))
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Validates and caches a complete immutable proving-key and SRS context.
    ///
    /// The prepared key and SRS remain backend-neutral owners. This boundary
    /// copies every cross-crate field explicitly into the Metal transport ABI;
    /// it does not transmute Rust types, acquire a proof execution guard, or
    /// make Metal proof routing available.
    pub fn prepare_resident_key_context(
        session: &MetalSession,
        key: &PreparedResidentProvingKey,
        srs: &PreparedResidentSrs,
    ) -> Result<MetalKeyContext, ResidentError> {
        // This mismatch must fail before conversion and before any method on
        // the Metal session can allocate, validate, or touch its key cache.
        if key.domain().n() != srs.n() {
            return Err(invalid_resident_key(
                "the proving-key domain and SRS prefix lengths differ",
            ));
        }

        let graph_fixed = GraphFixedKeyInput::new(
            key.domain().n(),
            key.domain().n_ext(),
            key.graphs(),
            key.fixed_cosets_compact(),
            key.extended_selectors(),
        );
        preflight_graph_fixed_key_input(graph_fixed)?;

        let graph_arrays = ConvertedMetalGraphArrays::new(key.graphs());
        let fixed_cosets = metal_fr_vec(key.fixed_cosets_compact());
        let extended_selectors = metal_fr_vec(key.extended_selectors());
        let fixed_lagrange = metal_fr_vec(key.fixed_lagrange_compact());
        let permutation_cosets = metal_fr_vec(key.permutation_cosets());
        let permutation_lagrange = metal_fr_vec(key.permutation_lagrange());
        let coefficient_columns = metal_fr_vec(key.coefficient_columns());
        let vanishing_inverses = metal_fr_vec(key.domain().vanishing_inverses());
        let permutation_columns = key
            .permutation_columns()
            .iter()
            .copied()
            .map(metal_column_ref)
            .collect::<Vec<_>>();
        let coefficient_bases = key_srs_bases(srs.coefficient_bases());
        let lagrange_bases = key_srs_bases(srs.lagrange_bases());
        let domain = key.domain();

        let context = session
            .get_or_create_key_context_v2(metal_backend::MetalKeyContextInputV2 {
                graph_fixed: metal_graph_fixed_input(
                    domain.n(),
                    domain.n_ext(),
                    key.graphs(),
                    &graph_arrays,
                    &fixed_cosets,
                    &extended_selectors,
                ),
                domain: metal_backend::MetalResidentDomainIdentity {
                    neutral_schema_version: domain.schema_version(),
                    neutral_fingerprint: domain.fingerprint(),
                    n: domain.n(),
                    k: domain.k(),
                    n_ext: domain.n_ext(),
                    extended_k: domain.extended_k(),
                    constraint_system_degree: domain.constraint_system_degree(),
                    quotient_piece_count: domain.quotient_piece_count(),
                    blinding_factors: domain.blinding_factors(),
                    usable_rows: domain.usable_rows(),
                    rotation_scale: domain.rotation_scale(),
                    last_rotation: domain.last_rotation(),
                    permutation_chunk_len: domain.permutation_chunk_len(),
                    base_ntt: metal_ntt_domain(
                        domain.k(),
                        domain.omega(),
                        domain.omega_inv(),
                        domain.n_inv(),
                        domain.coset_generator(),
                        domain.inverse_coset_generator(),
                    ),
                    extended_ntt: metal_ntt_domain(
                        domain.extended_k(),
                        domain.extended_omega(),
                        domain.extended_omega_inv(),
                        domain.n_ext_inv(),
                        domain.coset_generator(),
                        domain.inverse_coset_generator(),
                    ),
                    barycentric_weight: metal_fr(domain.barycentric_weight()),
                    delta: metal_fr(domain.delta()),
                    vanishing_inverses: &vanishing_inverses,
                },
                prepared_key_schema_version: key.schema_version(),
                prepared_key_fingerprint: key.fingerprint(),
                prepared_srs_schema_version: srs.schema_version(),
                prepared_srs_n: srs.n(),
                prepared_srs_fingerprint: srs.fingerprint(),
                coefficient_srs_digest: srs.coefficient_digest(),
                lagrange_srs_digest: srs.lagrange_digest(),
                permutation_columns: &permutation_columns,
                trash_selector_fixed_indices: key.trash_selector_fixed_indices(),
                fixed_lagrange: &fixed_lagrange,
                permutation_cosets: &permutation_cosets,
                permutation_lagrange: &permutation_lagrange,
                coefficient_columns: &coefficient_columns,
                fixed_coefficient_count: key.fixed_coefficient_count(),
                permutation_coefficient_count: key.permutation_coefficient_count(),
                coefficient_bases: &coefficient_bases,
                lagrange_bases: &lagrange_bases,
            })
            .map_err(|error| ResidentError::Backend(error.to_string()))?;
        session
            .prepare_bls12_381_msm_srs(srs.fingerprint(), &coefficient_bases, &lagrange_bases)
            .map_err(|error| ResidentError::Backend(error.to_string()))?;
        Ok(context)
    }

    /// Prepares both fixed SRS bases for the standalone, thread-affine Metal MSM worker.
    pub fn prepare_msm_srs(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
    ) -> Result<(), ResidentError> {
        if session
            .bls12_381_msm_srs_is_prepared(srs.fingerprint())
            .map_err(|error| ResidentError::Backend(error.to_string()))?
        {
            return Ok(());
        }
        let coefficient = key_srs_bases(srs.coefficient_bases());
        let lagrange = key_srs_bases(srs.lagrange_bases());
        session
            .prepare_bls12_381_msm_srs(srs.fingerprint(), &coefficient, &lagrange)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Executes one fixed-basis Metal MSM with no CPU curve-arithmetic fallback.
    pub fn msm_prepared(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
        basis: Bls12_381MsmBasis,
        scalars: &[RawFr],
        window: Bls12_381MsmWindow,
    ) -> Result<Bls12_381G1Projective, ResidentError> {
        if scalars.len() > srs.n() as usize {
            return Err(ResidentError::BufferBounds);
        }
        prepare_msm_srs(session, srs)?;
        session
            .bls12_381_msm(srs.fingerprint(), basis, metal_fr_slice(scalars), window)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Executes same-basis fixed-prefix MSMs in one Metal submission.
    pub fn msm_prepared_batch(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
        basis: Bls12_381MsmBasis,
        scalar_batches: &[&[RawFr]],
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, ResidentError> {
        if scalar_batches.is_empty()
            || scalar_batches
                .iter()
                .any(|scalars| scalars.len() > srs.n() as usize)
        {
            return Err(ResidentError::BufferBounds);
        }
        prepare_msm_srs(session, srs)?;
        let batches = scalar_batches
            .iter()
            .map(|scalars| metal_fr_slice(scalars))
            .collect::<Vec<_>>();
        session
            .bls12_381_msm_batch(srs.fingerprint(), basis, &batches, window)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Host coefficient polynomials and compact metadata for one resident
    /// Metal KZG multi-opening argument.
    #[derive(Clone, Copy, Debug)]
    pub struct ResidentMultiopenRequest<'a> {
        polynomials: &'a [&'a [RawFr]],
        q_offsets: &'a [u32],
        q_poly_indices: &'a [u32],
        q_coefficients: &'a [RawFr],
        point_offsets: &'a [u32],
        points: &'a [RawFr],
        f_weights: &'a [RawFr],
    }

    impl<'a> ResidentMultiopenRequest<'a> {
        /// Creates a request in sorted point-set order. The concrete runtime
        /// validates every extent, index, offset, and field element before use.
        #[allow(clippy::too_many_arguments)]
        pub const fn new(
            polynomials: &'a [&'a [RawFr]],
            q_offsets: &'a [u32],
            q_poly_indices: &'a [u32],
            q_coefficients: &'a [RawFr],
            point_offsets: &'a [u32],
            points: &'a [RawFr],
            f_weights: &'a [RawFr],
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
        pub const fn polynomials(self) -> &'a [&'a [RawFr]] {
            self.polynomials
        }

        /// Prefix offsets into q source indices and coefficients.
        pub const fn q_offsets(self) -> &'a [u32] {
            self.q_offsets
        }

        /// Source polynomial index for every q linear-combination term.
        pub const fn q_poly_indices(self) -> &'a [u32] {
            self.q_poly_indices
        }

        /// Coefficient for every q linear-combination term.
        pub const fn q_coefficients(self) -> &'a [RawFr] {
            self.q_coefficients
        }

        /// Prefix offsets into the flattened division-point array.
        pub const fn point_offsets(self) -> &'a [u32] {
            self.point_offsets
        }

        /// Flattened division points in sorted q order.
        pub const fn points(self) -> &'a [RawFr] {
            self.points
        }

        /// Powers of the second multi-open challenge used to construct f.
        pub const fn f_weights(self) -> &'a [RawFr] {
            self.f_weights
        }
    }

    /// Opaque callback-scoped coefficient polynomial accepted by resident MSM.
    #[must_use = "the resident polynomial lease must be committed inside its callback"]
    pub struct ResidentMultiopenMsmLease<'a> {
        backend: metal_backend::MetalResidentPolynomialMsmLease<'a>,
    }

    impl std::fmt::Debug for ResidentMultiopenMsmLease<'_> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("ResidentMultiopenMsmLease")
                .field("len", &self.len())
                .finish_non_exhaustive()
        }
    }

    impl ResidentMultiopenMsmLease<'_> {
        /// Number of resident coefficient scalars.
        pub const fn len(&self) -> usize {
            self.backend.len()
        }

        /// Whether the resident polynomial is empty.
        pub const fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    /// Callback-scoped owner of resident q, f, final, and π polynomials.
    pub struct ResidentMultiopenArena<'arena, 'session> {
        backend: &'arena mut metal_backend::MetalMultiopenArena<'session>,
    }

    impl std::fmt::Debug for ResidentMultiopenArena<'_, '_> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("ResidentMultiopenArena")
                .field("polynomial_len", &self.polynomial_len())
                .field("q_count", &self.q_count())
                .finish_non_exhaustive()
        }
    }

    impl ResidentMultiopenArena<'_, '_> {
        /// Number of coefficients in every resident polynomial slot.
        pub const fn polynomial_len(&self) -> usize {
            self.backend.polynomial_len()
        }

        /// Number of sorted q-polynomials.
        pub const fn q_count(&self) -> usize {
            self.backend.q_count()
        }

        /// Borrows resident f for immediate fixed-basis commitment.
        pub fn f_polynomial(&self) -> Result<ResidentMultiopenMsmLease<'_>, ResidentError> {
            self.backend
                .f_polynomial()
                .map(|backend| ResidentMultiopenMsmLease { backend })
                .map_err(|error| ResidentError::Backend(error.to_string()))
        }

        /// Evaluates all original q-polynomials and reads back only the compact
        /// evaluation vector required by the transcript.
        pub fn evaluate_q(&mut self, point: RawFr) -> Result<Vec<RawFr>, ResidentError> {
            self.backend
                .evaluate_q(metal_fr(point))
                .map(|values| {
                    values
                        .into_iter()
                        .map(|value| RawFr { limbs: value.limbs })
                        .collect()
                })
                .map_err(|error| ResidentError::Backend(error.to_string()))
        }

        /// Constructs π from original q, resident f, and transcript challenges,
        /// then borrows it for immediate fixed-basis commitment.
        pub fn compute_pi(
            &mut self,
            x3: RawFr,
            combination_weights: &[RawFr],
        ) -> Result<ResidentMultiopenMsmLease<'_>, ResidentError> {
            self.backend
                .compute_pi(metal_fr(x3), metal_fr_slice(combination_weights))
                .map(|backend| ResidentMultiopenMsmLease { backend })
                .map_err(|error| ResidentError::Backend(error.to_string()))
        }
    }

    /// Builds q and f on Metal and executes a transcript callback while all
    /// full polynomials remain resident. Every secret allocation is erased
    /// before this function returns, including callback error and panic paths.
    pub fn with_resident_multiopen<T>(
        session: &MetalSession,
        request: ResidentMultiopenRequest<'_>,
        with_arena: impl FnOnce(&mut ResidentMultiopenArena<'_, '_>) -> Result<T, ResidentError>,
    ) -> Result<T, ResidentError> {
        let polynomials = request
            .polynomials()
            .iter()
            .copied()
            .map(metal_fr_slice)
            .collect::<Vec<_>>();
        let backend_request = metal_backend::MetalMultiopenRequest::new(
            &polynomials,
            request.q_offsets(),
            request.q_poly_indices(),
            metal_fr_slice(request.q_coefficients()),
            request.point_offsets(),
            metal_fr_slice(request.points()),
            metal_fr_slice(request.f_weights()),
        );
        let callback_result = session
            .with_bls12_381_multiopen(backend_request, |backend| {
                let mut arena = ResidentMultiopenArena { backend };
                Ok(with_arena(&mut arena))
            })
            .map_err(|error| ResidentError::Backend(error.to_string()))?;
        callback_result
    }

    /// Commits one callback-scoped multi-open polynomial without scalar readback.
    pub fn msm_prepared_resident_multiopen(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
        basis: Bls12_381MsmBasis,
        lease: &ResidentMultiopenMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Bls12_381G1Projective, ResidentError> {
        if lease.len() != srs.n() as usize {
            return Err(ResidentError::BufferBounds);
        }
        prepare_msm_srs(session, srs)?;
        session
            .bls12_381_msm_resident_polynomial(srs.fingerprint(), basis, &lease.backend, window)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Callback-scoped opaque quotient buffer accepted only by resident Metal MSM.
    #[must_use = "the resident quotient lease must be consumed inside its callback"]
    pub struct ResidentQuotientMsmLease<'a> {
        backend: metal_backend::MetalQuotientResidentMsmLease<'a>,
        prepared_key_fingerprint: [u8; 32],
        prepared_srs_fingerprint: [u8; 32],
    }

    impl std::fmt::Debug for ResidentQuotientMsmLease<'_> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("ResidentQuotientMsmLease")
                .field("piece_len", &self.piece_len())
                .field("piece_count", &self.piece_count())
                .finish_non_exhaustive()
        }
    }

    impl ResidentQuotientMsmLease<'_> {
        /// Base-domain scalar count in every resident quotient piece.
        pub const fn piece_len(&self) -> usize {
            self.backend.piece_len()
        }

        /// Number of ordered resident quotient pieces.
        pub const fn piece_count(&self) -> usize {
            self.backend.piece_count()
        }

        /// Prepared proving-key fingerprint bound by the originating context.
        pub const fn prepared_key_fingerprint(&self) -> [u8; 32] {
            self.prepared_key_fingerprint
        }

        /// Prepared SRS fingerprint bound by the originating context.
        pub const fn prepared_srs_fingerprint(&self) -> [u8; 32] {
            self.prepared_srs_fingerprint
        }
    }

    /// Commits all pieces in a resident quotient lease with no scalar readback.
    pub fn msm_prepared_resident_quotient_batch(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
        basis: Bls12_381MsmBasis,
        lease: &ResidentQuotientMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, ResidentError> {
        if lease.piece_len() != srs.n() as usize
            || lease.piece_count() == 0
            || lease.prepared_srs_fingerprint() != srs.fingerprint()
        {
            return Err(ResidentError::BufferBounds);
        }
        prepare_msm_srs(session, srs)?;
        session
            .bls12_381_msm_resident_quotient_batch(srs.fingerprint(), basis, &lease.backend, window)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Combines resident quotient pieces and reads back one opening polynomial.
    ///
    /// This is a transitional interoperability boundary for a host multi-open
    /// continuation. Individual quotient pieces remain opaque and no CPU field
    /// arithmetic is used for their Horner combination.
    pub fn materialize_resident_quotient_combination(
        session: &MetalSession,
        lease: &ResidentQuotientMsmLease<'_>,
        splitting_factor: RawFr,
    ) -> Result<Vec<RawFr>, ResidentError> {
        session
            .materialize_bls12_381_quotient_combination(&lease.backend, metal_fr(splitting_factor))
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| RawFr { limbs: value.limbs })
                    .collect()
            })
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Callback-scoped view of quotient limbs synchronized for intentional CPU BLST MSM.
    ///
    /// This type is deliberately non-`Clone` and contains no Metal object. The
    /// borrowed values are the backend's mapped quotient allocation in reduced,
    /// little-endian Montgomery form; no host polynomial vector is created at
    /// this boundary. "Zero-copy" ends at this `RawFr` lease. The current CPU
    /// MSM implementation subsequently allocates filtered scalar/base vectors
    /// and BLST performs canonical-byte and digit preprocessing. Those secret
    /// CPU copies are not zeroized, so this diagnostic remains restricted to an
    /// explicitly enabled trusted-local test profile; production use requires
    /// either zeroization of that preprocessing or an approved equivalent gate.
    /// Its higher-ranked callback cannot return a value borrowed from the lease:
    ///
    /// ```compile_fail
    /// use midnight_gpu_prover::metal::DiagnosticQuotientCpuMsmLease;
    ///
    /// fn scoped<T>(
    ///     callback: impl for<'lease> FnOnce(DiagnosticQuotientCpuMsmLease<'lease>) -> T,
    /// ) {
    ///     unimplemented!()
    /// }
    ///
    /// scoped(|lease| lease.piece(0).expect("piece exists"));
    /// ```
    #[must_use = "the quotient lease must be consumed inside its callback"]
    pub struct DiagnosticQuotientCpuMsmLease<'a> {
        backend: Option<metal_backend::MetalQuotientCpuMsmLease<'a>>,
        values: &'a [RawFr],
        piece_len: usize,
        piece_count: usize,
        prepared_key_fingerprint: [u8; 32],
        prepared_srs_fingerprint: [u8; 32],
    }

    impl std::fmt::Debug for DiagnosticQuotientCpuMsmLease<'_> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("DiagnosticQuotientCpuMsmLease")
                .field("piece_len", &self.piece_len)
                .field("piece_count", &self.piece_count)
                .finish_non_exhaustive()
        }
    }

    impl<'a> DiagnosticQuotientCpuMsmLease<'a> {
        /// Base-domain scalar count in every quotient limb.
        pub const fn piece_len(&self) -> usize {
            self.piece_len
        }

        /// Number of ordered quotient limbs.
        pub const fn piece_count(&self) -> usize {
            self.piece_count
        }

        /// Total scalar count visible through this lease.
        pub const fn element_len(&self) -> usize {
            self.values.len()
        }

        /// Total logical CPU-visible byte count of this lease.
        pub const fn byte_len(&self) -> usize {
            std::mem::size_of_val(self.values)
        }

        /// Returns one quotient limb without copying its coefficients.
        pub fn piece(&self, index: usize) -> Result<&'a [RawFr], ResidentError> {
            if index >= self.piece_count {
                return Err(ResidentError::BufferBounds);
            }
            let start = index
                .checked_mul(self.piece_len)
                .ok_or(ResidentError::LengthOverflow)?;
            let end = start
                .checked_add(self.piece_len)
                .ok_or(ResidentError::LengthOverflow)?;
            self.values
                .get(start..end)
                .ok_or(ResidentError::BufferBounds)
        }

        /// Prepared proving-key fingerprint bound by the originating Metal context.
        pub const fn prepared_key_fingerprint(&self) -> [u8; 32] {
            self.prepared_key_fingerprint
        }

        /// Prepared SRS fingerprint bound by the originating Metal context.
        pub const fn prepared_srs_fingerprint(&self) -> [u8; 32] {
            self.prepared_srs_fingerprint
        }
    }

    /// Commits a CPU-visible diagnostic quotient lease through its resident buffer.
    ///
    /// This transitional operation avoids CPU canonicalization and scalar upload
    /// for commitment while preserving the explicit host continuation needed by
    /// the current evaluation and multi-open implementation.
    pub fn msm_prepared_resident_diagnostic_quotient_batch(
        session: &MetalSession,
        srs: &PreparedResidentSrs,
        basis: Bls12_381MsmBasis,
        lease: &DiagnosticQuotientCpuMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, ResidentError> {
        if lease.piece_len() != srs.n() as usize
            || lease.piece_count() == 0
            || lease.prepared_srs_fingerprint() != srs.fingerprint()
        {
            return Err(ResidentError::BufferBounds);
        }
        let backend = lease.backend.as_ref().ok_or_else(|| {
            ResidentError::Backend("diagnostic quotient lease has no resident Metal view".into())
        })?;
        prepare_msm_srs(session, srs)?;
        session
            .bls12_381_msm_resident_cpu_quotient_batch(srs.fingerprint(), basis, backend, window)
            .map_err(|error| ResidentError::Backend(error.to_string()))
    }

    /// Executes the feature-gated diagnostic numerator-to-quotient Metal epoch.
    ///
    /// The facade validates the complete schema-v2 key/SRS/domain binding and
    /// every dynamic source shape before calling the backend. Source columns
    /// remain separate borrowed slices and are copied directly into the mapped
    /// Metal arena in protocol order. The callback is the sole intentional
    /// whole-polynomial CPU boundary and runs only after command completion.
    /// Its zero-copy guarantee covers only the mapped `RawFr` lease, not the
    /// existing CPU MSM's scalar/base filtering or BLST digit preprocessing.
    ///
    /// This diagnostic does not enable production proof routing. Once selected,
    /// any Metal or callback error is returned; it never retries on the CPU.
    pub fn with_diagnostic_quotient_epoch<T>(
        session: &MetalSession,
        context: &MetalKeyContext,
        input: DiagnosticQuotientEpochInput<'_>,
        with_cpu_msm: impl for<'lease> FnOnce(
            DiagnosticQuotientCpuMsmLease<'lease>,
        ) -> Result<T, ResidentError>,
    ) -> Result<T, ResidentError> {
        let layout = validate_diagnostic_quotient_epoch(context, input)?;
        let mut coefficient_columns = Vec::with_capacity(layout.total_columns as usize);
        for group in input.sources().groups_in_protocol_order() {
            coefficient_columns.extend(group.columns().iter().copied().map(metal_fr_slice));
        }
        if coefficient_columns.len() != layout.total_columns as usize {
            return Err(invalid_quotient_epoch(
                "dynamic source count changed after preflight",
            ));
        }

        let challenges = metal_fr_slice(input.challenges());
        let quotient_blinds = metal_fr_slice(input.quotient_blinds());
        let prepared_key_fingerprint = input.key().fingerprint();
        let prepared_srs_fingerprint = input.srs().fingerprint();
        session
            .with_diagnostic_quotient_epoch(
                context,
                metal_backend::MetalQuotientEpochRequest {
                    coefficient_columns: &coefficient_columns,
                    total_columns: layout.total_columns,
                    advice_offset: layout.advice_offset,
                    advice_columns: layout.advice_columns,
                    instance_offset: layout.instance_offset,
                    instance_columns: layout.instance_columns,
                    permutation_offset: layout.permutation_offset,
                    permutation_sets: layout.permutation_sets,
                    lookup_product_offset: layout.lookup_product_offset,
                    lookup_input_offset: layout.lookup_input_offset,
                    lookup_table_offset: layout.lookup_table_offset,
                    trash_offset: layout.trash_offset,
                    challenges,
                    beta: metal_fr(input.beta()),
                    gamma: metal_fr(input.gamma()),
                    theta: metal_fr(input.theta()),
                    trash_challenge: metal_fr(input.trash_challenge()),
                    y: metal_fr(input.y()),
                    quotient_blinds,
                },
                |lease| {
                    if lease.key_context_identity() != context.identity()
                        || lease.piece_len() != input.key().domain().n() as usize
                        || lease.piece_count()
                            != input.key().domain().quotient_piece_count() as usize
                    {
                        return Err(invalid_quotient_epoch(
                            "backend CPU-MSM lease does not match its validated request",
                        ));
                    }
                    let values = raw_fr_slice(lease.as_raw_fr());
                    let expected = lease
                        .piece_len()
                        .checked_mul(lease.piece_count())
                        .ok_or(ResidentError::LengthOverflow)?;
                    if values.len() != expected {
                        return Err(invalid_quotient_epoch(
                            "backend CPU-MSM lease has an inconsistent extent",
                        ));
                    }
                    let piece_len = lease.piece_len();
                    let piece_count = lease.piece_count();
                    with_cpu_msm(DiagnosticQuotientCpuMsmLease {
                        backend: Some(lease),
                        values,
                        piece_len,
                        piece_count,
                        prepared_key_fingerprint,
                        prepared_srs_fingerprint,
                    })
                },
            )
            .map_err(|error| ResidentError::Backend(error.to_string()))?
    }

    /// Executes the quotient epoch and lends its pieces to resident Metal MSM.
    ///
    /// The callback receives no scalar slice or Metal object. Its only supported
    /// commitment operation is [`msm_prepared_resident_quotient_batch`], which
    /// synchronously completes before the producer erases the quotient arena.
    pub fn with_resident_quotient_epoch<T>(
        session: &MetalSession,
        context: &MetalKeyContext,
        input: DiagnosticQuotientEpochInput<'_>,
        with_msm: impl for<'lease> FnOnce(ResidentQuotientMsmLease<'lease>) -> Result<T, ResidentError>,
    ) -> Result<T, ResidentError> {
        let layout = validate_diagnostic_quotient_epoch(context, input)?;
        let mut coefficient_columns = Vec::with_capacity(layout.total_columns as usize);
        for group in input.sources().groups_in_protocol_order() {
            coefficient_columns.extend(group.columns().iter().copied().map(metal_fr_slice));
        }
        if coefficient_columns.len() != layout.total_columns as usize {
            return Err(invalid_quotient_epoch(
                "dynamic source count changed after preflight",
            ));
        }

        let challenges = metal_fr_slice(input.challenges());
        let quotient_blinds = metal_fr_slice(input.quotient_blinds());
        let prepared_key_fingerprint = input.key().fingerprint();
        let prepared_srs_fingerprint = input.srs().fingerprint();
        session
            .with_resident_quotient_epoch(
                context,
                metal_backend::MetalQuotientEpochRequest {
                    coefficient_columns: &coefficient_columns,
                    total_columns: layout.total_columns,
                    advice_offset: layout.advice_offset,
                    advice_columns: layout.advice_columns,
                    instance_offset: layout.instance_offset,
                    instance_columns: layout.instance_columns,
                    permutation_offset: layout.permutation_offset,
                    permutation_sets: layout.permutation_sets,
                    lookup_product_offset: layout.lookup_product_offset,
                    lookup_input_offset: layout.lookup_input_offset,
                    lookup_table_offset: layout.lookup_table_offset,
                    trash_offset: layout.trash_offset,
                    challenges,
                    beta: metal_fr(input.beta()),
                    gamma: metal_fr(input.gamma()),
                    theta: metal_fr(input.theta()),
                    trash_challenge: metal_fr(input.trash_challenge()),
                    y: metal_fr(input.y()),
                    quotient_blinds,
                },
                |lease| {
                    if lease.key_context_identity() != context.identity()
                        || lease.piece_len() != input.key().domain().n() as usize
                        || lease.piece_count()
                            != input.key().domain().quotient_piece_count() as usize
                    {
                        return Err(invalid_quotient_epoch(
                            "backend resident MSM lease does not match its validated request",
                        ));
                    }
                    with_msm(ResidentQuotientMsmLease {
                        backend: lease,
                        prepared_key_fingerprint,
                        prepared_srs_fingerprint,
                    })
                },
            )
            .map_err(|error| ResidentError::Backend(error.to_string()))?
    }

    #[derive(Clone, Copy, Debug)]
    struct DiagnosticQuotientLayout {
        total_columns: u32,
        advice_offset: u32,
        advice_columns: u32,
        instance_offset: u32,
        instance_columns: u32,
        permutation_offset: u32,
        permutation_sets: u32,
        lookup_product_offset: u32,
        lookup_input_offset: u32,
        lookup_table_offset: u32,
        trash_offset: u32,
    }

    fn validate_diagnostic_quotient_epoch(
        context: &MetalKeyContext,
        input: DiagnosticQuotientEpochInput<'_>,
    ) -> Result<DiagnosticQuotientLayout, ResidentError> {
        let key = input.key();
        let srs = input.srs();
        let domain = key.domain();
        let identity = context.identity();
        if srs.n() != domain.n()
            || identity.n() != domain.n()
            || identity.n_ext() != domain.n_ext()
            || identity.domain_fingerprint() != Some(domain.fingerprint())
            || identity.prepared_key_fingerprint() != Some(key.fingerprint())
            || identity.prepared_srs_fingerprint() != Some(srs.fingerprint())
            || identity.graph_family_fingerprint() != key.graphs().packed().fingerprint()
        {
            return Err(invalid_quotient_epoch(
                "schema-v2 context, proving key, SRS, or exact domain fingerprint differs",
            ));
        }

        let n = domain.n() as usize;
        let n_ext = domain.n_ext() as usize;
        let quotient_piece_count = domain.quotient_piece_count() as usize;
        let quotient_elements = quotient_piece_count
            .checked_mul(n)
            .ok_or(ResidentError::LengthOverflow)?;
        if quotient_piece_count < 2 || quotient_elements > n_ext {
            return Err(invalid_quotient_epoch(
                "quotient pieces do not fit the exact extended domain",
            ));
        }
        if input.quotient_blinds().len() != quotient_piece_count - 1 {
            return Err(invalid_quotient_epoch(
                "quotient blind count must be exactly quotient_piece_count - 1",
            ));
        }
        if input.challenges().len() < key.graphs().packed().challenge_bound() as usize {
            return Err(invalid_quotient_epoch(
                "proof challenge slice is shorter than the compiled graph bound",
            ));
        }

        let source_groups = input.sources().groups_in_protocol_order();
        let permutation_sets = key
            .permutation_columns()
            .len()
            .div_ceil(domain.permutation_chunk_len() as usize);
        let lookup_count = key.graphs().sections().lookups() as usize;
        let expected_counts = [
            key.graphs().packed().advice_compact_to_logical().len(),
            key.graphs().packed().instance_compact_to_logical().len(),
            permutation_sets,
            lookup_count,
            lookup_count,
            lookup_count,
            key.graphs().sections().trash() as usize,
        ];
        for (group, expected_count) in source_groups.into_iter().zip(expected_counts) {
            if group.columns().len() != expected_count {
                return Err(invalid_quotient_epoch(&format!(
                    "{:?} source count differs from the prepared proving key",
                    group.kind()
                )));
            }
            for (column_index, column) in group.columns().iter().enumerate() {
                if column.len() != n {
                    return Err(invalid_quotient_epoch(&format!(
                        "{:?} column {column_index} does not have base-domain length",
                        group.kind()
                    )));
                }
                validate_reduced_raw_fr("dynamic coefficient column", column)?;
            }
        }
        validate_reduced_raw_fr("proof challenge", input.challenges())?;
        validate_reduced_raw_fr(
            "quotient scalar challenge",
            &[
                input.beta(),
                input.gamma(),
                input.theta(),
                input.trash_challenge(),
                input.y(),
            ],
        )?;
        validate_reduced_raw_fr("quotient blind", input.quotient_blinds())?;

        let checked_counts = expected_counts
            .into_iter()
            .map(|count| {
                u32::try_from(count).map_err(|_| {
                    invalid_quotient_epoch("dynamic source count exceeds the backend u32 ABI")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let advice_columns = checked_counts[0];
        let instance_columns = checked_counts[1];
        let permutation_sets = checked_counts[2];
        let lookup_products = checked_counts[3];
        let lookup_permuted_inputs = checked_counts[4];
        let lookup_permuted_tables = checked_counts[5];
        let trash = checked_counts[6];

        let advice_offset = 0u32;
        let instance_offset = advice_offset
            .checked_add(advice_columns)
            .ok_or(ResidentError::LengthOverflow)?;
        let permutation_offset = instance_offset
            .checked_add(instance_columns)
            .ok_or(ResidentError::LengthOverflow)?;
        let lookup_product_offset = permutation_offset
            .checked_add(permutation_sets)
            .ok_or(ResidentError::LengthOverflow)?;
        let lookup_input_offset = lookup_product_offset
            .checked_add(lookup_products)
            .ok_or(ResidentError::LengthOverflow)?;
        let lookup_table_offset = lookup_input_offset
            .checked_add(lookup_permuted_inputs)
            .ok_or(ResidentError::LengthOverflow)?;
        let trash_offset = lookup_table_offset
            .checked_add(lookup_permuted_tables)
            .ok_or(ResidentError::LengthOverflow)?;
        let total_columns = trash_offset
            .checked_add(trash)
            .ok_or(ResidentError::LengthOverflow)?;

        Ok(DiagnosticQuotientLayout {
            total_columns,
            advice_offset,
            advice_columns,
            instance_offset,
            instance_columns,
            permutation_offset,
            permutation_sets,
            lookup_product_offset,
            lookup_input_offset,
            lookup_table_offset,
            trash_offset,
        })
    }

    const BLS12_381_FR_MODULUS: [u64; 4] = [
        0xffff_ffff_0000_0001,
        0x53bd_a402_fffe_5bfe,
        0x3339_d808_09a1_d805,
        0x73ed_a753_299d_7d48,
    ];

    fn validate_reduced_raw_fr(label: &str, values: &[RawFr]) -> Result<(), ResidentError> {
        if let Some(index) = values.iter().position(|value| !raw_fr_is_reduced(*value)) {
            return Err(invalid_quotient_epoch(&format!(
                "{label} {index} is not a reduced BLS12-381 scalar"
            )));
        }
        Ok(())
    }

    fn raw_fr_is_reduced(value: RawFr) -> bool {
        for index in (0..4).rev() {
            if value.limbs[index] < BLS12_381_FR_MODULUS[index] {
                return true;
            }
            if value.limbs[index] > BLS12_381_FR_MODULUS[index] {
                return false;
            }
        }
        false
    }

    #[allow(unsafe_code)]
    fn metal_fr_slice(values: &[RawFr]) -> &[metal_backend::Bls12_381Fr] {
        assert_eq!(
            std::mem::size_of::<RawFr>(),
            std::mem::size_of::<metal_backend::Bls12_381Fr>()
        );
        assert_eq!(
            std::mem::align_of::<RawFr>(),
            std::mem::align_of::<metal_backend::Bls12_381Fr>()
        );
        // SAFETY: both repr(C) types are exactly four little-endian u64
        // Montgomery limbs; the returned slice cannot outlive `values`.
        unsafe {
            std::slice::from_raw_parts(
                values.as_ptr().cast::<metal_backend::Bls12_381Fr>(),
                values.len(),
            )
        }
    }

    #[allow(unsafe_code)]
    fn raw_fr_slice(values: &[metal_backend::Bls12_381Fr]) -> &[RawFr] {
        assert_eq!(
            std::mem::size_of::<RawFr>(),
            std::mem::size_of::<metal_backend::Bls12_381Fr>()
        );
        assert_eq!(
            std::mem::align_of::<RawFr>(),
            std::mem::align_of::<metal_backend::Bls12_381Fr>()
        );
        // SAFETY: the backend lease exposes the same checked four-u64
        // Montgomery ABI and pins its allocation for this returned lifetime.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<RawFr>(), values.len()) }
    }

    fn invalid_quotient_epoch(reason: &str) -> ResidentError {
        ResidentError::Backend(format!(
            "invalid diagnostic resident quotient epoch: {reason}"
        ))
    }

    #[cfg(test)]
    mod diagnostic_quotient_adapter_tests {
        use super::*;

        #[test]
        fn cpu_msm_lease_returns_zero_copy_piece_views() {
            let values = [
                RawFr { limbs: [1; 4] },
                RawFr { limbs: [2; 4] },
                RawFr { limbs: [3; 4] },
                RawFr { limbs: [4; 4] },
            ];
            let lease = DiagnosticQuotientCpuMsmLease {
                backend: None,
                values: &values,
                piece_len: 2,
                piece_count: 2,
                prepared_key_fingerprint: [5; 32],
                prepared_srs_fingerprint: [6; 32],
            };

            let first = lease.piece(0).expect("borrow first quotient piece");
            let second = lease.piece(1).expect("borrow second quotient piece");
            assert_eq!(first.as_ptr(), values.as_ptr());
            assert_eq!(second.as_ptr(), values[2..].as_ptr());
            assert_eq!(lease.element_len(), 4);
            assert_eq!(lease.byte_len(), 4 * std::mem::size_of::<RawFr>());
            assert_eq!(lease.piece(2), Err(ResidentError::BufferBounds));
        }
    }

    struct ConvertedMetalGraphArrays {
        descriptors: Vec<metal_backend::MetalGraphDescriptor>,
        constants: Vec<metal_backend::Bls12_381Fr>,
        operations: Vec<metal_backend::MetalGraphOperation>,
        horner_parts: Vec<metal_backend::MetalGraphValueRef>,
    }

    impl ConvertedMetalGraphArrays {
        fn new(graphs: &crate::expression::CompiledProtocolGraphFamily) -> Self {
            let graphs = graphs.packed();
            let descriptors = graphs
                .descriptors()
                .iter()
                .map(|descriptor| metal_backend::MetalGraphDescriptor {
                    constants_offset: descriptor.constants_offset(),
                    constants_count: descriptor.constants_count(),
                    rotations_offset: descriptor.rotations_offset(),
                    rotations_count: descriptor.rotations_count(),
                    operations_offset: descriptor.operations_offset(),
                    operations_count: descriptor.operations_count(),
                    parts_offset: descriptor.parts_offset(),
                    parts_count: descriptor.parts_count(),
                    scratch_capacity: descriptor.scratch_class().capacity(),
                })
                .collect();
            let constants = graphs.constants().iter().copied().map(metal_fr).collect();
            let operations = graphs
                .operations()
                .iter()
                .map(|operation| metal_backend::MetalGraphOperation {
                    opcode: operation.opcode,
                    target: operation.target,
                    parts_offset: operation.parts_offset,
                    parts_len: operation.parts_len,
                    lhs: metal_value_ref(operation.lhs),
                    rhs: metal_value_ref(operation.rhs),
                })
                .collect();
            let horner_parts = graphs
                .horner_parts()
                .iter()
                .copied()
                .map(metal_value_ref)
                .collect();
            Self {
                descriptors,
                constants,
                operations,
                horner_parts,
            }
        }
    }

    fn metal_graph_fixed_input<'a>(
        n: u32,
        n_ext: u32,
        graphs: &'a crate::expression::CompiledProtocolGraphFamily,
        arrays: &'a ConvertedMetalGraphArrays,
        fixed_cosets: &'a [metal_backend::Bls12_381Fr],
        lagrange_selectors: &'a [metal_backend::Bls12_381Fr],
    ) -> metal_backend::MetalKeyContextInput<'a> {
        let packed = graphs.packed();
        let sections = graphs.sections();
        metal_backend::MetalKeyContextInput {
            n,
            n_ext,
            graph_family_fingerprint_version: crate::expression::GRAPH_FAMILY_FINGERPRINT_VERSION,
            graph_family_fingerprint: packed.fingerprint(),
            sections: metal_backend::MetalGraphFamilySections {
                custom: sections.custom(),
                lookups: sections.lookups(),
                lookup_inputs: sections.lookup_inputs(),
                lookup_tables: sections.lookup_tables(),
                trash: sections.trash(),
            },
            descriptors: &arrays.descriptors,
            constants: &arrays.constants,
            rotations: packed.rotations(),
            operations: &arrays.operations,
            horner_parts: &arrays.horner_parts,
            fixed_compact_to_logical: packed.fixed_compact_to_logical(),
            advice_compact_to_logical: packed.advice_compact_to_logical(),
            instance_compact_to_logical: packed.instance_compact_to_logical(),
            challenge_bound: packed.challenge_bound(),
            fixed_cosets,
            lagrange_selectors,
        }
    }

    fn metal_ntt_domain(
        log_size: u32,
        root: RawFr,
        inverse_root: RawFr,
        inverse_size: RawFr,
        coset_generator: RawFr,
        inverse_coset_generator: RawFr,
    ) -> metal_backend::NttDomainSpec {
        metal_backend::NttDomainSpec {
            log_size,
            root: metal_fr(root),
            inverse_root: metal_fr(inverse_root),
            inverse_size: metal_fr(inverse_size),
            coset_generator: metal_fr(coset_generator),
            inverse_coset_generator: metal_fr(inverse_coset_generator),
        }
    }

    fn metal_column_ref(column: crate::ResidentColumnRef) -> metal_backend::MetalResidentColumnRef {
        let kind = match column.kind() {
            ResidentColumnKind::Fixed => 0,
            ResidentColumnKind::Advice => 1,
            ResidentColumnKind::Instance => 2,
        };
        metal_backend::MetalResidentColumnRef {
            kind,
            index: column.index(),
        }
    }

    fn metal_fr_vec(values: &[RawFr]) -> Vec<metal_backend::Bls12_381Fr> {
        values.iter().copied().map(metal_fr).collect()
    }

    const fn metal_fr(value: RawFr) -> metal_backend::Bls12_381Fr {
        metal_backend::Bls12_381Fr { limbs: value.limbs }
    }

    fn key_srs_bases(points: &[RawG1Affine]) -> Vec<metal_backend::Bls12_381G1Affine> {
        points
            .iter()
            .map(|point| metal_backend::Bls12_381G1Affine {
                x: point.x,
                y: point.y,
            })
            .collect()
    }

    fn preflight_graph_fixed_key_input(input: GraphFixedKeyInput<'_>) -> Result<(), ResidentError> {
        let n = input.n();
        let n_ext = input.n_ext();
        if !n.is_power_of_two()
            || !n_ext.is_power_of_two()
            || n_ext < n
            || !n_ext.is_multiple_of(n)
            || n_ext / n > i32::MAX as u32
        {
            return Err(invalid_graph_fixed_key(
                "n and n_ext must be supported power-of-two domains",
            ));
        }

        let sections = input.sections();
        if sections.custom() > 1 {
            return Err(invalid_graph_fixed_key(
                "the custom graph count must be zero or one",
            ));
        }
        if sections.lookups() != sections.lookup_inputs()
            || sections.lookups() != sections.lookup_tables()
        {
            return Err(invalid_graph_fixed_key(
                "lookup graph section counts must agree",
            ));
        }
        let descriptor_count = sections
            .checked_total()
            .and_then(|count| usize::try_from(count).ok());
        if descriptor_count != Some(input.graphs().packed().descriptors().len()) {
            return Err(invalid_graph_fixed_key(
                "graph sections must exactly cover the descriptor family",
            ));
        }

        let n_ext = usize::try_from(n_ext)
            .map_err(|_| invalid_graph_fixed_key("n_ext does not fit usize"))?;
        let expected_fixed = input
            .graphs()
            .packed()
            .fixed_compact_to_logical()
            .len()
            .checked_mul(n_ext)
            .ok_or_else(|| invalid_graph_fixed_key("fixed-coset extent overflows usize"))?;
        if input.fixed_cosets().len() != expected_fixed {
            return Err(invalid_graph_fixed_key(
                "fixed-coset storage does not match the compact fixed-column map",
            ));
        }
        if u32::try_from(input.fixed_cosets().len()).is_err() {
            return Err(invalid_graph_fixed_key(
                "fixed-coset storage exceeds the backend u32 ABI",
            ));
        }
        let expected_selectors = 3usize
            .checked_mul(n_ext)
            .ok_or_else(|| invalid_graph_fixed_key("selector extent overflows usize"))?;
        if input.lagrange_selectors().len() != expected_selectors {
            return Err(invalid_graph_fixed_key(
                "selector storage must contain exactly l0, l_last, and l_active",
            ));
        }
        if u32::try_from(input.lagrange_selectors().len()).is_err() {
            return Err(invalid_graph_fixed_key(
                "selector storage exceeds the backend u32 ABI",
            ));
        }
        Ok(())
    }

    fn invalid_graph_fixed_key(reason: &str) -> ResidentError {
        ResidentError::Backend(format!(
            "invalid immutable graph/fixed key context: {reason}"
        ))
    }

    fn invalid_resident_key(reason: &str) -> ResidentError {
        ResidentError::Backend(format!(
            "invalid immutable resident proving-key context: {reason}"
        ))
    }

    fn metal_value_ref(
        value: crate::expression::GraphValueRef,
    ) -> metal_backend::MetalGraphValueRef {
        metal_backend::MetalGraphValueRef {
            kind: value.kind,
            a: value.a,
            b: value.b,
        }
    }

    /// Reports Metal runtime, proof-pipeline, and routing readiness through the
    /// same neutral contract as CUDA.
    pub fn status() -> BackendStatus {
        let metal = metal_backend::metal_status();
        let runtime = if metal.runtime_available() {
            RuntimeAvailability::Available
        } else {
            RuntimeAvailability::Unavailable
        };
        let full_proof_pipeline = metal.proof_backend_available();
        let unavailable_reason = metal
            .runtime_unavailable_reason()
            .map(str::to_owned)
            .or_else(|| {
                (!full_proof_pipeline).then(|| metal.proof_backend_unavailable_reason().to_owned())
            })
            .or_else(|| Some(EXECUTION_ROUTE_UNAVAILABLE.to_owned()));

        BackendStatus {
            kind: BackendKind::Metal,
            runtime,
            capabilities: BackendCapabilities {
                full_proof_pipeline,
                resident_polynomials: full_proof_pipeline,
            },
            // This changes only when proof-server has a native Metal execution
            // route. Kernel/runtime readiness alone must never unlock it.
            execution_routed: false,
            unavailable_reason,
        }
    }
}

/// Concrete implementation for the backend selected by the build target.
///
/// This compatibility module supports existing backend-specific diagnostics.
/// New resident proof code should use [`ProofArena`], [`ResidentFrBuffer`], and
/// semantic root-level operations instead of importing concrete backend APIs.
#[cfg(any(
    all(feature = "cuda", target_os = "linux"),
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
pub mod target {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub use crate::cuda::*;
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub use crate::metal::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_fr_abi_is_four_aligned_u64_limbs() {
        assert_eq!(std::mem::size_of::<RawFr>(), 32);
        assert_eq!(std::mem::align_of::<RawFr>(), std::mem::align_of::<u64>());
        let value = RawFr {
            limbs: [1, 2, 3, 4],
        };
        assert_eq!(value.limbs, [1, 2, 3, 4]);
    }

    #[test]
    fn quotient_coefficient_sources_have_one_fixed_protocol_order() {
        let advice_column = [RawFr { limbs: [1; 4] }];
        let instance_column = [RawFr { limbs: [2; 4] }];
        let permutation_column = [RawFr { limbs: [3; 4] }];
        let lookup_product_column = [RawFr { limbs: [4; 4] }];
        let lookup_input_column = [RawFr { limbs: [5; 4] }];
        let lookup_table_column = [RawFr { limbs: [6; 4] }];
        let trash_column = [RawFr { limbs: [7; 4] }];
        let advice = [advice_column.as_slice()];
        let instance = [instance_column.as_slice()];
        let permutation = [permutation_column.as_slice()];
        let lookup_products = [lookup_product_column.as_slice()];
        let lookup_permuted_inputs = [lookup_input_column.as_slice()];
        let lookup_permuted_tables = [lookup_table_column.as_slice()];
        let trash = [trash_column.as_slice()];
        let sources = QuotientCoefficientSources::new(
            &advice,
            &instance,
            &permutation,
            &lookup_products,
            &lookup_permuted_inputs,
            &lookup_permuted_tables,
            &trash,
        );

        let groups = sources.groups_in_protocol_order();
        assert_eq!(
            groups.map(QuotientCoefficientSourceGroup::kind),
            [
                QuotientCoefficientSourceKind::Advice,
                QuotientCoefficientSourceKind::Instance,
                QuotientCoefficientSourceKind::PermutationProducts,
                QuotientCoefficientSourceKind::LookupProducts,
                QuotientCoefficientSourceKind::LookupPermutedInputs,
                QuotientCoefficientSourceKind::LookupPermutedTables,
                QuotientCoefficientSourceKind::Trash,
            ]
        );
        assert_eq!(
            groups.map(|group| group.columns()[0][0].limbs[0]),
            [1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn authentic_five_proof_quotient_leases_total_thirteen_mib() {
        let proof_shapes = [
            (8_192usize, 4usize),
            (32_768, 4),
            (32_768, 4),
            (16_384, 4),
            (16_384, 4),
        ];
        let element_count = proof_shapes
            .into_iter()
            .map(|(n, piece_count)| n * piece_count)
            .sum::<usize>();
        assert_eq!(element_count, 425_984);
        assert_eq!(
            element_count * std::mem::size_of::<RawFr>(),
            13 * 1024 * 1024
        );
    }

    #[test]
    fn proof_arena_rejects_foreign_stale_and_out_of_bounds_tokens() {
        let values = [RawFr { limbs: [7; 4] }, RawFr { limbs: [9; 4] }];
        let mut first = ProofArena::for_test();
        let mut token = first.upload(&values).expect("upload test values");
        assert_eq!(token.len(), values.len());
        assert!(!token.is_empty());
        first.validate(&token).expect("fresh token must validate");

        let second = ProofArena::for_test();
        assert_eq!(second.validate(&token), Err(ResidentError::ForeignBuffer));

        token.len += 1;
        assert_eq!(first.validate(&token), Err(ResidentError::BufferBounds));
        token.len -= 1;

        first.allocations[token.slot].generation += 1;
        assert_eq!(first.validate(&token), Err(ResidentError::StaleBuffer));
    }

    #[test]
    fn proof_arena_upload_many_preserves_order_and_owns_storage() {
        let first = [RawFr { limbs: [1; 4] }, RawFr { limbs: [2; 4] }];
        let second = [RawFr { limbs: [3; 4] }];
        let mut arena = ProofArena::for_test();
        let token = arena
            .upload_many(&[&first, &second])
            .expect("upload several slices");
        assert_eq!(token.len(), 3);
        arena
            .validate(&token)
            .expect("combined token must validate");
        let BackendArena::Test(allocations) = &arena.backend else {
            panic!("test arena must use the test backend")
        };
        assert_eq!(
            allocations[token.slot].as_ref(),
            &[first[0], first[1], second[0]]
        );
        arena.finish().expect("finish test arena");
    }

    #[test]
    fn allocation_identity_exhaustion_precedes_backend_upload() {
        let mut arena = ProofArena::for_test();
        arena.next_allocation_id = u64::MAX;

        let error = arena
            .upload(&[RawFr { limbs: [1; 4] }])
            .expect_err("exhausted identity space must reject before upload");
        assert_eq!(error, ResidentError::AllocationIdentityExhausted);
        assert!(arena.allocations.is_empty());
        let BackendArena::Test(allocations) = &arena.backend else {
            panic!("test arena must use the test backend")
        };
        assert!(allocations.is_empty());
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_raw_slice_is_zero_copy_and_layout_identical() {
        let values = [RawFr {
            limbs: [11, 12, 13, 14],
        }];
        let raw = cuda_raw_slice(&values);
        assert_eq!(raw.as_ptr().cast::<RawFr>(), values.as_ptr());
        assert_eq!(raw[0].limbs, values[0].limbs);
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    #[test]
    fn no_backend_is_selected_without_a_feature() {
        assert_eq!(selected_backend(), None);
        assert_eq!(selected_backend_status(), None);
        assert_eq!(
            ExecutionGuard::acquire(7, 13).expect_err("no backend guard must fail closed"),
            ResidentError::BackendNotSelected
        );
        assert_eq!(
            ProofArena::for_current_proof().expect_err("no backend must fail closed"),
            ResidentError::BackendNotSelected
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_selects_metal_without_routing_cpu_as_metal() {
        assert_eq!(selected_backend(), Some(BackendKind::Metal));
        assert_eq!(
            std::mem::size_of::<RawFr>(),
            std::mem::size_of::<metal_backend::Bls12_381Fr>()
        );
        assert_eq!(
            std::mem::align_of::<RawFr>(),
            std::mem::align_of::<metal_backend::Bls12_381Fr>()
        );

        let status = selected_backend_status().expect("Metal status should be selected");
        assert_eq!(status.kind(), BackendKind::Metal);
        assert!(!status.execution_routed());
        assert!(!status.proof_execution_available());
        assert!(status.unavailable_reason().is_some());
        assert!(matches!(
            ExecutionGuard::acquire(7, 13),
            Err(ResidentError::BackendIncomplete(_))
        ));
        let status_after_guard =
            selected_backend_status().expect("Metal status should remain selected");
        assert!(!status_after_guard.execution_routed());
        assert!(!status_after_guard.proof_execution_available());
        assert!(matches!(
            ProofArena::for_current_proof(),
            Err(ResidentError::BackendIncomplete(_))
        ));
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_facade_prepares_exact_neutral_graph_fixed_key_identity() {
        use crate::expression::{
            CompiledGraph, CompiledProtocolGraphFamily, GraphColumnRequirements, GraphOp,
            GraphOpcode, GraphSourceCounts, GraphValueKind, GraphValueRef,
        };

        let graph = CompiledGraph::try_new(
            vec![RawFr::default()].into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            vec![GraphOp::new(
                GraphOpcode::Store,
                0,
                0,
                0,
                GraphValueRef::new(GraphValueKind::Constant, 0, 0),
                GraphValueRef::default(),
            )]
            .into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphSourceCounts::default(),
        )
        .expect("compile facade key-context graph");
        let required_fixed = [2];
        let family = CompiledProtocolGraphFamily::try_new_with_required_columns(
            Some(graph),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphColumnRequirements::new(&required_fixed, &[], &[]),
        )
        .expect("pack facade key-context graph family");
        let fixed_cosets = [RawFr::default(); 4];
        let selectors = [RawFr::default(); 12];
        let input = GraphFixedKeyInput::new(2, 4, &family, &fixed_cosets, &selectors);
        let session = metal_backend::MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();

        let invalid_domain = GraphFixedKeyInput::new(3, 4, &family, &fixed_cosets, &selectors);
        let error = metal::prepare_graph_fixed_key_context(&session, invalid_domain)
            .expect_err("invalid domain must fail facade preflight");
        assert!(error.to_string().contains("power-of-two domains"));

        let short_selectors = [RawFr::default(); 11];
        let invalid_selectors =
            GraphFixedKeyInput::new(2, 4, &family, &fixed_cosets, &short_selectors);
        let error = metal::prepare_graph_fixed_key_context(&session, invalid_selectors)
            .expect_err("invalid selector extent must fail facade preflight");
        assert!(error.to_string().contains("selector storage"));
        let preflight_activity = session.activity();
        assert_eq!(preflight_activity.allocations, 0);
        assert_eq!(preflight_activity.host_write_bytes, 0);
        assert_eq!(preflight_activity.key_cache_hits, 0);
        assert_eq!(preflight_activity.key_cache_misses, 0);
        assert_eq!(preflight_activity.key_context_builds, 0);

        let first = metal::prepare_graph_fixed_key_context(&session, input)
            .expect("prepare graph/fixed key context through facade");
        let same = metal::prepare_graph_fixed_key_context(&session, input)
            .expect("reuse graph/fixed key context through facade");
        assert_eq!(first.identity(), same.identity());
        assert_eq!(
            first.identity().graph_family_fingerprint(),
            family.packed().fingerprint()
        );
        assert_eq!(first.identity().n(), 2);
        assert_eq!(first.identity().n_ext(), 4);
        let activity = session.activity();
        assert_eq!(activity.key_cache_misses, 1);
        assert_eq!(activity.key_cache_hits, 1);
        assert_eq!(activity.key_context_builds, 1);
        assert_eq!(activity.live_key_contexts, 1);
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resident_facade_rejects_pk_srs_mismatch_before_metal_activity() {
        let (key, _) = representative_resident_key_and_srs();
        let mismatched_srs = representative_resident_srs(4, false);
        let session = metal_backend::MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();

        let error = metal::prepare_resident_key_context(&session, &key, &mismatched_srs)
            .expect_err("domain/SRS mismatch must fail before Metal import");
        assert!(error.to_string().contains("SRS prefix lengths differ"));

        let activity = session.activity();
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.host_write_bytes, 0);
        assert_eq!(activity.key_cache_hits, 0);
        assert_eq!(activity.key_cache_misses, 0);
        assert_eq!(activity.key_context_builds, 0);
        assert_eq!(activity.live_key_contexts, 0);
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resident_facade_reuses_the_complete_context() {
        let (key, srs) = representative_resident_key_and_srs();
        let session = metal_backend::MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();

        let first = metal::prepare_resident_key_context(&session, &key, &srs)
            .expect("prepare complete resident key context");
        let same = metal::prepare_resident_key_context(&session, &key, &srs)
            .expect("reuse complete resident key context");
        assert_eq!(first.identity(), same.identity());

        let activity = session.activity();
        assert_eq!(activity.key_cache_misses, 1);
        assert_eq!(activity.key_cache_hits, 1);
        assert_eq!(activity.key_context_builds, 1);
        assert_eq!(activity.live_key_contexts, 1);
        assert_eq!(activity.command_buffers, 0);
        assert_eq!(activity.dispatches, 0);
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resident_facade_binds_lagrange_srs_independently() {
        let (key, baseline_srs) = representative_resident_key_and_srs();
        let changed_srs = representative_resident_srs(2, true);
        assert_eq!(
            baseline_srs.coefficient_digest(),
            changed_srs.coefficient_digest()
        );
        assert_ne!(
            baseline_srs.lagrange_digest(),
            changed_srs.lagrange_digest()
        );

        let session = metal_backend::MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let baseline = metal::prepare_resident_key_context(&session, &key, &baseline_srs)
            .expect("prepare baseline resident SRS context");
        let changed = metal::prepare_resident_key_context(&session, &key, &changed_srs)
            .expect("prepare Lagrange-only changed SRS context");

        assert_ne!(baseline.identity(), changed.identity());
        assert_eq!(
            baseline.identity().prepared_key_fingerprint(),
            changed.identity().prepared_key_fingerprint()
        );
        assert_ne!(
            baseline.identity().prepared_srs_fingerprint(),
            changed.identity().prepared_srs_fingerprint()
        );
        let activity = session.activity();
        assert_eq!(activity.key_cache_misses, 2);
        assert_eq!(activity.key_context_builds, 2);
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_resident_facade_preserves_neutral_fingerprints_and_ntt_fields() {
        let (key, srs) = representative_resident_key_and_srs();
        let session = metal_backend::MetalSession::new().expect("initialize Metal runtime");
        let context = metal::prepare_resident_key_context(&session, &key, &srs)
            .expect("prepare complete resident key context");
        let identity = context.identity();

        assert_eq!(
            identity.schema_version(),
            metal_backend::METAL_KEY_CONTEXT_SCHEMA_VERSION_V2
        );
        assert_eq!(
            identity.graph_family_fingerprint(),
            key.graphs().packed().fingerprint()
        );
        assert_eq!(
            identity.domain_fingerprint(),
            Some(key.domain().fingerprint())
        );
        assert_eq!(identity.prepared_key_fingerprint(), Some(key.fingerprint()));
        assert_eq!(identity.prepared_srs_fingerprint(), Some(srs.fingerprint()));

        let base_ntt = identity
            .base_ntt_domain()
            .expect("v2 identity must retain the exact base NTT domain");
        assert_eq!(base_ntt.log_size, key.domain().k());
        assert_eq!(base_ntt.root.limbs, key.domain().omega().limbs);
        assert_eq!(base_ntt.inverse_root.limbs, key.domain().omega_inv().limbs);
        assert_eq!(base_ntt.inverse_size.limbs, key.domain().n_inv().limbs);
        assert_eq!(
            base_ntt.coset_generator.limbs,
            key.domain().coset_generator().limbs
        );
        assert_eq!(
            base_ntt.inverse_coset_generator.limbs,
            key.domain().inverse_coset_generator().limbs
        );

        let extended_ntt = identity
            .extended_ntt_domain()
            .expect("v2 identity must retain the exact extended NTT domain");
        assert_eq!(extended_ntt.log_size, key.domain().extended_k());
        assert_eq!(extended_ntt.root.limbs, key.domain().extended_omega().limbs);
        assert_eq!(
            extended_ntt.inverse_root.limbs,
            key.domain().extended_omega_inv().limbs
        );
        assert_eq!(
            extended_ntt.inverse_size.limbs,
            key.domain().n_ext_inv().limbs
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn representative_resident_key_and_srs() -> (PreparedResidentProvingKey, PreparedResidentSrs) {
        use crate::expression::{
            CompiledGraph, CompiledProtocolGraphFamily, GraphColumnRequirements, GraphOp,
            GraphOpcode, GraphSourceCounts, GraphValueKind, GraphValueRef,
        };

        let graph = CompiledGraph::try_new(
            vec![RawFr::default()].into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            vec![GraphOp::new(
                GraphOpcode::Store,
                0,
                0,
                0,
                GraphValueRef::new(GraphValueKind::Constant, 0, 0),
                GraphValueRef::default(),
            )]
            .into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphSourceCounts::default(),
        )
        .expect("compile representative resident graph");
        let graphs = CompiledProtocolGraphFamily::try_new_with_required_columns(
            Some(graph),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            Vec::new().into_boxed_slice(),
            GraphColumnRequirements::new(&[2], &[7], &[11]),
        )
        .expect("pack representative resident graph family");
        let domain = representative_resident_domain();
        let zero = RawFr::default();
        let key = PreparedResidentProvingKey::try_new(PreparedResidentProvingKeyInput {
            domain,
            graphs,
            permutation_columns: vec![
                ResidentColumnRef::new(ResidentColumnKind::Fixed, 0),
                ResidentColumnRef::new(ResidentColumnKind::Advice, 0),
                ResidentColumnRef::new(ResidentColumnKind::Instance, 0),
            ]
            .into_boxed_slice(),
            trash_selector_fixed_indices: Vec::new().into_boxed_slice(),
            fixed_cosets_compact: vec![zero; 4].into_boxed_slice(),
            fixed_lagrange_compact: vec![zero; 2].into_boxed_slice(),
            permutation_cosets: vec![zero; 12].into_boxed_slice(),
            permutation_lagrange: vec![zero; 6].into_boxed_slice(),
            // Three logical fixed columns followed by three permutation
            // columns, each with the base-domain stride two.
            coefficient_columns: vec![zero; 12].into_boxed_slice(),
            extended_selectors: vec![zero; 12].into_boxed_slice(),
        })
        .expect("prepare representative resident proving key");
        let srs = representative_resident_srs(2, false);
        (key, srs)
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn representative_resident_domain() -> ResidentDomainManifest {
        ResidentDomainManifest::try_new(ResidentDomainManifestInput {
            k: 1,
            extended_k: 2,
            constraint_system_degree: 3,
            blinding_factors: 0,
            omega: test_raw_fr([
                0xffff_fffd_0000_0003,
                0xfb38_ec08_fffb_13fc,
                0x99ad_8818_1ce5_880f,
                0x5bc8_f5f9_7cd8_77d8,
            ]),
            omega_inv: test_raw_fr([
                0xffff_fffd_0000_0003,
                0xfb38_ec08_fffb_13fc,
                0x99ad_8818_1ce5_880f,
                0x5bc8_f5f9_7cd8_77d8,
            ]),
            extended_omega: test_raw_fr([
                0xf3b0_5674_aa89_cfb1,
                0x072f_0140_6006_b9fe,
                0xce9a_0dbf_2566_7a26,
                0x4d2c_e405_2d59_8374,
            ]),
            extended_omega_inv: test_raw_fr([
                0x0c4f_a98a_5576_3050,
                0x4c8e_a2c2_9ff7_a200,
                0x649f_ca48_e43b_5ddf,
                0x26c0_c34d_fc43_f9d3,
            ]),
            n_inv: test_raw_fr([
                0x0000_0000_ffff_ffff,
                0xac42_5bfd_0001_a401,
                0xccc6_27f7_f65e_27fa,
                0x0c12_58ac_d662_82b7,
            ]),
            n_ext_inv: test_raw_fr([0, 0, 0, 0x4000_0000_0000_0000]),
            coset_generator: test_raw_fr([
                0x92d9_090b_0930_11d2,
                0xfc9c_bd71_9d6a_a073,
                0xc1f1_4ef0_cd65_a1a6,
                0x017f_6d35_e72f_cdeb,
            ]),
            inverse_coset_generator: test_raw_fr([
                0x6d26_f6f1_f6cf_ee31,
                0xfe9c_2e97_6290_7389,
                0xd7bc_3927_4f7f_e668,
                0x5a49_88c3_95a8_a9ec,
            ]),
            barycentric_weight: test_raw_fr([
                0x0000_0000_ffff_ffff,
                0xac42_5bfd_0001_a401,
                0xccc6_27f7_f65e_27fa,
                0x0c12_58ac_d662_82b7,
            ]),
            vanishing_inverses: vec![
                test_raw_fr([
                    0xdb9d_adad_0310_05f2,
                    0x6e86_6f29_df21_5acf,
                    0x84f2_e505_a6a4_55e9,
                    0x45bc_ae2b_848c_9659,
                ]),
                test_raw_fr([
                    0x6d26_f6f1_f6cf_ee31,
                    0xfe9c_2e97_6290_7389,
                    0xd7bc_3927_4f7f_e668,
                    0x5a49_88c3_95a8_a9ec,
                ]),
            ]
            .into_boxed_slice(),
            delta: test_raw_fr([
                0x70e3_10d3_d146_f96a,
                0x4b64_c089_19e2_99e6,
                0x51e1_1418_6a8b_970d,
                0x6185_d066_27c0_67cb,
            ]),
        })
        .expect("prepare representative exact resident domain")
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn representative_resident_srs(n: u32, reverse_lagrange: bool) -> PreparedResidentSrs {
        // Transport fixture only: every base is reduced, nonidentity, on-curve,
        // and in the prime subgroup. Repeated +/- generators do not claim the
        // powers-of-tau relationship of a production trusted setup.
        let generator = RawG1Affine {
            x: [
                0x5cb3_8790_fd53_0c16,
                0x7817_fc67_9976_fff5,
                0x154f_95c7_143b_a1c1,
                0xf0ae_6acd_f3d0_e747,
                0xedce_6ecc_21db_f440,
                0x1201_7741_9e0b_fb75,
            ],
            y: [
                0xbaac_93d5_0ce7_2271,
                0x8c22_631a_7918_fd8e,
                0xdd59_5f13_5707_25ce,
                0x51ac_5829_5040_5194,
                0x0e1c_8c3f_ad00_59c0,
                0x0bbc_3efc_5008_a26a,
            ],
        };
        let negative_generator = RawG1Affine {
            x: generator.x,
            y: [
                0xff52_6c2a_f318_883a,
                0x9289_9ce4_383b_0270,
                0x89d7_738d_9fa9_d055,
                0x12ca_f35b_a344_c12a,
                0x3cff_1b76_964b_5317,
                0x0e44_d2ed_e977_4430,
            ],
        };
        let coefficient_bases = (0..n)
            .map(|index| {
                if index.is_multiple_of(2) {
                    generator
                } else {
                    negative_generator
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let lagrange_bases = (0..n)
            .map(|index| {
                let even = index.is_multiple_of(2) ^ reverse_lagrange;
                if even { generator } else { negative_generator }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        PreparedResidentSrs::try_new(n, coefficient_bases, lagrange_bases)
            .expect("prepare representative subgroup-valid nonidentity SRS bases")
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    const fn test_raw_fr(limbs: [u64; 4]) -> RawFr {
        RawFr { limbs }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    #[ignore = "requires a provisioned NVIDIA CUDA runtime"]
    fn cuda_execution_guard_owns_and_restores_lane() {
        assert_eq!(cuda_backend::current_execution_lane(), 0);
        let guard = ExecutionGuard::acquire(0, 13).expect("acquire CUDA execution guard");
        assert_ne!(cuda_backend::current_execution_lane(), 0);
        drop(guard);
        assert_eq!(cuda_backend::current_execution_lane(), 0);
    }
}
