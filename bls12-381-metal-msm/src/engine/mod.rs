// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Synchronous, fail-closed Apple Metal MSM execution.

extern crate std;

mod abi;
mod batch_affine_plan;
// The complete unsafe surface is isolated in this audited Objective-C and
// shared-memory boundary; public planning and value types remain safe Rust.
#[allow(unsafe_code)]
mod sys;

use core::fmt;
use core::sync::atomic::{Ordering, compiler_fence};
use std::format;
use std::rc::Rc;
use std::string::{String, ToString};
use std::vec::Vec;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use zeroize::Zeroizing;

use crate::msm_api::{
    CanonicalScalar, FixedWindowBits, MsmJobV1, RawFrMontgomery, RawG1Affine, RawG1Jacobian,
    WindowBits,
};
use crate::router::{RoutePlan, route_jobs};

use self::abi::{
    BucketRun, BucketTileV1, CollectDescriptor, EMPTY_BUCKET, FIXED_ROUTE_ABI_VERSION_V10,
    FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10, FIXED_ROUTE_THREADS_PER_GROUP_V10,
    FixedRouteParametersV10, GPU_ROUTED_DENSE_ABI_VERSION_V4, GpuAffine, GpuFp, GpuJacobian,
    GpuRouteJobV3, GpuRouteParametersV3, GpuScalar, PARALLEL_BUCKET_TILE_RECORDS,
    PARALLEL_JACOBIAN_LARGE_THREADS, PARALLEL_JACOBIAN_LARGE_TILE_RECORDS,
    PARALLEL_JACOBIAN_SMALL_THREADS, PARALLEL_JACOBIAN_SMALL_TILE_RECORDS, SignedRecord, TileRunV1,
    WindowDescriptor, fixed_route_buckets_per_group_v10,
};
use self::batch_affine_plan::{BatchAffineLevelPlan, compile_balanced_run_plan};
use self::sys::{ExecutionPlan, MetalContext, PreparedBaseBuffer, SysError};

const FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

// The larger launch first wins once each lane can amortize roughly three to
// four mixed additions. Keep this route-shape decision out of the shader ABI.
const JACOBIAN_LARGE_GEOMETRY_MIN_RECORDS_PER_RUN: usize = 192;

/// Stable identity of the Metal device owned by an engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceIdentity {
    name: String,
    registry_id: u64,
}

impl MetalDeviceIdentity {
    /// Human-readable Metal device name reported by the operating system.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Registry identifier used to distinguish physical Metal devices.
    #[must_use]
    pub const fn registry_id(&self) -> u64 {
        self.registry_id
    }
}

/// Side-channel observation profile of an execution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMsmObservationProfile {
    /// Version 1 is restricted to a trusted local device and process.
    ///
    /// CPU Booth routing, the number of routed records, non-empty bucket
    /// occupancy, and serial per-bucket work are scalar-dependent. Buffer
    /// contents are erased, but this profile does not promise constant timing,
    /// traffic, allocation size, or GPU occupancy to a hostile co-tenant.
    TrustedLocalDataDependentV1,
}

/// Metal bucket-reduction architecture selected for an engine instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMsmReductionStrategy {
    /// Globally batches affine denominators at every balanced tree level.
    GlobalBatchAffineV1,
    /// Reduces bounded projective tiles cooperatively inside threadgroups.
    ParallelProjectiveV1,
    /// Uses branch-completed fast Jacobian formulas inside cooperative tiles.
    ParallelBranchCompletedJacobianV2,
    /// Recodes and routes scalars on Metal, then reduces each dense bucket in
    /// one cooperative branch-completed Jacobian group.
    GpuRoutedDenseJacobianV3,
    /// Reuses the validated Metal routing path with XYZZ bucket arithmetic,
    /// retaining both Z squared and Z cubed across additions.
    GpuRoutedDenseXyzzV4,
    /// Retains XYZZ arithmetic and gives a full threadgroup to highly skewed
    /// adjacent bucket pairs while preserving the v4 dense comparator.
    GpuRoutedAdaptiveXyzzV5,
    /// Keeps v5 for ordinary buckets, but splits long runs into compact,
    /// indirectly dispatched XYZZ tiles before recombining their partials.
    GpuRoutedTiledXyzzV6,
    /// Packs ordinary buckets by window width while retaining v6's compact
    /// indirect tiling for scalar distributions with long bucket runs.
    GpuRoutedPackedTiledXyzzV7,
    /// Splits medium-heavy buckets into 2048-record tiles so structured proof
    /// polynomials do not leave long serial chains inside a bucket lane.
    GpuRoutedFineTiledXyzzV8,
    /// Batches the first affine addition in every 128-record tile behind one
    /// global inversion, then completes tile and bucket reduction in XYZZ.
    GpuRoutedBatchAffineXyzzV9,
}

/// Why a base was rejected before it reached Metal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedBaseIssue {
    /// The x-coordinate is not a reduced BLS12-381 base-field residue.
    XCoordinateNotReduced,
    /// The y-coordinate is not a reduced BLS12-381 base-field residue.
    YCoordinateNotReduced,
    /// The all-zero affine identity encoding is not a usable prepared base.
    Identity,
    /// BLST's independent predicate rejected the point as off-curve.
    NotOnCurve,
    /// BLST's independent predicate rejected the point as outside prime-order G1.
    NotInPrimeSubgroup,
}

impl fmt::Display for PreparedBaseIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::XCoordinateNotReduced => formatter.write_str("x-coordinate is not reduced"),
            Self::YCoordinateNotReduced => formatter.write_str("y-coordinate is not reduced"),
            Self::Identity => formatter.write_str("affine identity is not a prepared base"),
            Self::NotOnCurve => formatter.write_str("point is not on BLS12-381 G1"),
            Self::NotInPrimeSubgroup => {
                formatter.write_str("point is not in the BLS12-381 G1 prime subgroup")
            }
        }
    }
}

/// A preparation, routing, allocation, ABI, execution, or erasure failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum MetalMsmError {
    /// Device, library, pipeline, or shader-ABI initialization failed.
    Initialization(String),
    /// A prepared base failed host-side semantic validation.
    InvalidPreparedBase {
        /// Index in the caller's prepared-base slice.
        index: usize,
        /// Exact validation predicate that failed.
        issue: PreparedBaseIssue,
    },
    /// Preparing an empty base set is not supported by the version-1 engine.
    EmptyPreparedBases,
    /// Prepared bases were created by another engine instance.
    PreparedBasesEngineMismatch,
    /// A batch or ragged submission did not contain an MSM job.
    EmptySubmission,
    /// A prefix batch contains more scalars than prepared bases.
    ScalarBatchTooLong {
        /// Batch index supplied by the caller.
        batch_index: usize,
        /// Number of canonical scalars in that batch.
        scalar_count: usize,
        /// Number of bases available to the engine.
        prepared_base_count: usize,
    },
    /// An owned host array or versioned ABI count overflowed.
    Capacity(&'static str),
    /// An external Montgomery-scalar buffer has an invalid extent or alignment.
    InvalidResidentScalarBuffer(String),
    /// An external Montgomery-scalar buffer belongs to another Metal device.
    ResidentScalarDeviceMismatch {
        /// Registry identifier of the engine device.
        expected_registry_id: u64,
        /// Registry identifier of the external buffer's device.
        actual_registry_id: u64,
    },
    /// Deterministic Booth recoding or routing rejected the request.
    Routing(String),
    /// Independent runtime translation rejected a malformed route.
    InvalidRoute(&'static str),
    /// Metal execution failed; the engine never falls back to CPU curve arithmetic.
    Execution(String),
}

impl fmt::Display for MetalMsmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization(reason) => {
                write!(formatter, "Metal MSM initialization failed: {reason}")
            }
            Self::InvalidPreparedBase { index, issue } => {
                write!(formatter, "prepared base {index} is invalid: {issue}")
            }
            Self::EmptyPreparedBases => formatter.write_str("prepared-base set must not be empty"),
            Self::PreparedBasesEngineMismatch => formatter
                .write_str("prepared bases belong to a different Metal MSM engine instance"),
            Self::EmptySubmission => formatter.write_str("Metal MSM submission has no jobs"),
            Self::ScalarBatchTooLong { batch_index, scalar_count, prepared_base_count } => write!(
                formatter,
                "scalar batch {batch_index} has {scalar_count} terms but only {prepared_base_count} bases are prepared"
            ),
            Self::Capacity(label) => {
                write!(formatter, "Metal MSM {label} exceeds host or ABI capacity")
            }
            Self::InvalidResidentScalarBuffer(reason) => {
                write!(formatter, "invalid resident Montgomery-scalar buffer: {reason}")
            }
            Self::ResidentScalarDeviceMismatch { expected_registry_id, actual_registry_id } => {
                write!(
                    formatter,
                    "resident Montgomery-scalar buffer device {actual_registry_id:#018x} differs from engine device {expected_registry_id:#018x}"
                )
            }
            Self::Routing(reason) => write!(formatter, "Metal MSM routing failed: {reason}"),
            Self::InvalidRoute(reason) => write!(formatter, "Metal MSM route is invalid: {reason}"),
            Self::Execution(reason) => write!(formatter, "Metal MSM execution failed: {reason}"),
        }
    }
}

impl std::error::Error for MetalMsmError {}

/// Long-lived Apple Metal device, queue, and MSM pipeline collection.
///
/// The engine is synchronous and serializes submissions so secret-buffer
/// erasure has a single auditable lifecycle. Any unconfirmed erasure poisons
/// the shared context; later calls fail instead of falling back to CPU. Version
/// 1 is intentionally thread-affine; a server should own it on one Metal worker
/// thread instead of asserting undocumented Objective-C `Send`/`Sync` traits.
#[derive(Clone)]
pub struct MetalMsmEngine {
    context: Rc<MetalContext>,
    identity: Rc<MetalDeviceIdentity>,
    observation_profile: MetalMsmObservationProfile,
    reduction_strategy: MetalMsmReductionStrategy,
}

impl fmt::Debug for MetalMsmEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalMsmEngine")
            .field("device", &self.identity)
            .field("observation_profile", &self.observation_profile)
            .field("reduction_strategy", &self.reduction_strategy)
            .finish()
    }
}

/// Immutable, validated G1 bases uploaded for reuse by one engine instance.
#[derive(Clone)]
pub struct PreparedBases {
    context: Rc<MetalContext>,
    allocation: Rc<PreparedBaseBuffer>,
    identity: Rc<MetalDeviceIdentity>,
}

/// Immutable fixed bases expanded by window powers and retained on one device.
///
/// Preparation is intentionally explicit because it trades roughly
/// `windows_per_scalar` times the affine-base storage for fewer point additions
/// on every subsequent MSM. The expanded allocation is reusable and is not
/// rebuilt during submission.
#[derive(Clone)]
pub struct PreparedFixedBases {
    context: Rc<MetalContext>,
    allocation: Rc<PreparedBaseBuffer>,
    identity: Rc<MetalDeviceIdentity>,
    base_count: usize,
    window_bits: FixedWindowBits,
}

/// Retained view of reduced BLS12-381 Fr values in Montgomery form.
///
/// The producer must complete every write before submitting this view to an
/// MSM, and must not mutate the covered bytes until that synchronous call
/// returns. The library retains but never erases the caller-owned allocation;
/// only its private canonical copy and routing scratch are securely cleared.
#[derive(Clone)]
pub struct ResidentMontgomeryScalars {
    inner: self::sys::ResidentScalarBuffer,
}

impl fmt::Debug for ResidentMontgomeryScalars {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentMontgomeryScalars")
            .field("len", &self.len())
            .field("byte_offset", &self.byte_offset())
            .field("device_registry_id", &self.device_registry_id())
            .finish_non_exhaustive()
    }
}

impl ResidentMontgomeryScalars {
    /// Retains a synchronized Metal buffer range without reading it on the CPU.
    ///
    /// `byte_offset` must be a multiple of the 32-byte scalar representation,
    /// and the exact `len * 32` byte range must fit in `buffer`. Each value is
    /// checked for reduction and converted from Montgomery form by Metal before
    /// recoding. Construction does not wait for producer commands.
    pub fn from_synchronized_mtl_buffer(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        byte_offset: usize,
        len: usize,
    ) -> Result<Self, MetalMsmError> {
        self::sys::ResidentScalarBuffer::new(buffer, byte_offset, len)
            .map(|inner| Self { inner })
            .map_err(|error| MetalMsmError::InvalidResidentScalarBuffer(error.to_string()))
    }

    /// Number of Montgomery residues in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the view contains no residues.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Byte offset of the first residue in the retained buffer.
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.inner.byte_offset()
    }

    /// Registry identifier of the Metal device that owns the buffer.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.inner.registry_id()
    }
}

impl fmt::Debug for PreparedFixedBases {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedFixedBases")
            .field("base_count", &self.base_count)
            .field("window_bits", &self.window_bits)
            .field("expanded_base_count", &self.allocation.len())
            .field("device", &self.identity)
            .finish_non_exhaustive()
    }
}

impl PreparedFixedBases {
    /// Number of original bases represented by the expanded allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.base_count
    }

    /// Returns whether no original bases are represented.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.base_count == 0
    }

    /// Window width baked into the expanded allocation.
    #[must_use]
    pub const fn window_bits(&self) -> FixedWindowBits {
        self.window_bits
    }

    /// Number of affine points retained after window expansion.
    #[must_use]
    pub fn expanded_len(&self) -> usize {
        self.allocation.len()
    }

    /// Device that owns this prepared allocation.
    #[must_use]
    pub fn device_identity(&self) -> &MetalDeviceIdentity {
        &self.identity
    }
}

impl fmt::Debug for PreparedBases {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedBases")
            .field("base_count", &self.len())
            .field("device", &self.identity)
            .finish_non_exhaustive()
    }
}

impl PreparedBases {
    /// Number of validated affine bases in the retained allocation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocation.len()
    }

    /// Returns whether no bases are prepared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device that owns this prepared allocation.
    #[must_use]
    pub fn device_identity(&self) -> &MetalDeviceIdentity {
        &self.identity
    }
}

impl MetalMsmEngine {
    /// Loads the embedded metallib after an explicit observation-profile opt-in.
    ///
    /// Requiring the profile at construction prevents the first data-dependent
    /// implementation from being enabled without acknowledging its trusted-local
    /// deployment restriction.
    pub fn new(observation_profile: MetalMsmObservationProfile) -> Result<Self, MetalMsmError> {
        Self::new_with_strategy(observation_profile, MetalMsmReductionStrategy::GlobalBatchAffineV1)
    }

    /// Loads the embedded metallib with an explicit reduction architecture.
    pub fn new_with_strategy(
        observation_profile: MetalMsmObservationProfile,
        reduction_strategy: MetalMsmReductionStrategy,
    ) -> Result<Self, MetalMsmError> {
        match observation_profile {
            MetalMsmObservationProfile::TrustedLocalDataDependentV1 => {}
        }
        let context = Rc::new(MetalContext::new().map_err(initialization_error)?);
        let identity = Rc::new(MetalDeviceIdentity {
            name: context.device_name(),
            registry_id: context.registry_id(),
        });
        Ok(Self { context, identity, observation_profile, reduction_strategy })
    }

    /// Device used by this engine and all prepared allocations it creates.
    #[must_use]
    pub fn device_identity(&self) -> &MetalDeviceIdentity {
        &self.identity
    }

    /// Declares the observable-work profile of this implementation.
    #[must_use]
    pub const fn observation_profile(&self) -> MetalMsmObservationProfile {
        self.observation_profile
    }

    /// Bucket-reduction architecture used by this engine.
    #[must_use]
    pub const fn reduction_strategy(&self) -> MetalMsmReductionStrategy {
        self.reduction_strategy
    }

    /// Validates and uploads finite, reduced, on-curve prime-subgroup bases.
    pub fn prepare_bases(&self, bases: &[RawG1Affine]) -> Result<PreparedBases, MetalMsmError> {
        validate_prepared_bases(bases)?;
        let mut gpu_bases = Vec::new();
        gpu_bases
            .try_reserve_exact(bases.len())
            .map_err(|_| MetalMsmError::Capacity("prepared-base allocation"))?;
        gpu_bases.extend(bases.iter().copied().map(gpu_affine));
        let allocation = self.context.upload_bases(&gpu_bases).map_err(execution_error)?;
        Ok(PreparedBases {
            context: self.context.clone(),
            allocation: Rc::new(allocation),
            identity: self.identity.clone(),
        })
    }

    /// Validates, window-expands, and uploads bases for the fixed-base MSM path.
    ///
    /// Layer `j` stores `2^(j*w) * bases[i]`, where `w` is `window_bits`.
    /// The CPU preparation cost is amortized across every MSM that reuses the
    /// returned allocation; submissions perform recoding, routing, reduction,
    /// and weighted integration on Metal.
    pub fn prepare_fixed_bases(
        &self,
        bases: &[RawG1Affine],
        window_bits: FixedWindowBits,
    ) -> Result<PreparedFixedBases, MetalMsmError> {
        validate_prepared_bases(bases)?;
        let gpu_bases = expand_fixed_bases(bases, window_bits)?;
        let allocation = self.context.upload_bases(&gpu_bases).map_err(execution_error)?;
        Ok(PreparedFixedBases {
            context: self.context.clone(),
            allocation: Rc::new(allocation),
            identity: self.identity.clone(),
            base_count: bases.len(),
            window_bits,
        })
    }

    /// Computes one prefix MSM with an explicit Booth window width.
    ///
    /// `scalars[i]` is paired with prepared base `i`. Empty scalars produce the
    /// identity. The method never performs CPU curve arithmetic.
    pub fn msm(
        &self,
        prepared: &PreparedBases,
        scalars: &[CanonicalScalar],
        window_bits: WindowBits,
    ) -> Result<RawG1Jacobian, MetalMsmError> {
        self.check_prepared(prepared)?;
        if scalars.len() > prepared.len() {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: 0,
                scalar_count: scalars.len(),
                prepared_base_count: prepared.len(),
            });
        }
        let term_count = checked_u32(scalars.len(), "single-MSM term count")?;
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count, output_index: 0 }];
        let mut outputs = self.execute_jobs(prepared, scalars, &jobs, window_bits)?;
        outputs.pop().ok_or(MetalMsmError::InvalidRoute("single MSM returned no output"))
    }

    /// Computes a batch of prefix MSMs over the same prepared bases.
    ///
    /// Batches may have different lengths. Scalar data is copied only to build
    /// the contiguous routing input and is wiped with volatile stores afterward.
    pub fn msm_batch(
        &self,
        prepared: &PreparedBases,
        scalar_batches: &[&[CanonicalScalar]],
        window_bits: WindowBits,
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_prepared(prepared)?;
        if scalar_batches.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        let total_scalars = scalar_batches.iter().try_fold(0usize, |total, batch| {
            total.checked_add(batch.len()).ok_or(MetalMsmError::Capacity("batch scalar count"))
        })?;
        let mut scalars = SensitiveScalars::with_capacity(total_scalars)?;
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(scalar_batches.len())
            .map_err(|_| MetalMsmError::Capacity("batch job allocation"))?;
        for (batch_index, batch) in scalar_batches.iter().enumerate() {
            if batch.len() > prepared.len() {
                return Err(MetalMsmError::ScalarBatchTooLong {
                    batch_index,
                    scalar_count: batch.len(),
                    prepared_base_count: prepared.len(),
                });
            }
            let scalar_offset = checked_u32(scalars.len(), "batch scalar offset")?;
            let term_count = checked_u32(batch.len(), "batch term count")?;
            let output_index = checked_u32(batch_index, "batch output index")?;
            scalars.extend_from_slice(batch);
            jobs.push(MsmJobV1 { base_offset: 0, scalar_offset, term_count, output_index });
        }
        self.execute_jobs(prepared, scalars.as_slice(), &jobs, window_bits)
    }

    /// Computes caller-described ragged MSMs over shared base and scalar arrays.
    ///
    /// Job output indices must be a permutation of `0..jobs.len()`. Range and
    /// density checks are performed by both the public router and private Metal
    /// translation. This is the direct path for authentic heterogeneous batches.
    pub fn msm_ragged(
        &self,
        prepared: &PreparedBases,
        scalars: &[CanonicalScalar],
        jobs: &[MsmJobV1],
        window_bits: WindowBits,
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_prepared(prepared)?;
        if jobs.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        self.execute_jobs(prepared, scalars, jobs, window_bits)
    }

    /// Computes one MSM using a reusable pre-shifted fixed-base allocation.
    pub fn msm_fixed(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[CanonicalScalar],
    ) -> Result<RawG1Jacobian, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if scalars.len() > prepared.len() {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: 0,
                scalar_count: scalars.len(),
                prepared_base_count: prepared.len(),
            });
        }
        let term_count = checked_u32(scalars.len(), "single fixed MSM term count")?;
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count, output_index: 0 }];
        let mut outputs = self.execute_fixed_jobs(prepared, scalars, &jobs)?;
        outputs.pop().ok_or(MetalMsmError::InvalidRoute("single fixed MSM returned no output"))
    }

    /// Computes a ragged batch of prefix MSMs using shared pre-shifted bases.
    pub fn msm_fixed_batch(
        &self,
        prepared: &PreparedFixedBases,
        scalar_batches: &[&[CanonicalScalar]],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if scalar_batches.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        let total_scalars = scalar_batches.iter().try_fold(0usize, |total, batch| {
            total
                .checked_add(batch.len())
                .ok_or(MetalMsmError::Capacity("fixed batch scalar count"))
        })?;
        let mut scalars = SensitiveScalars::with_capacity(total_scalars)?;
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(scalar_batches.len())
            .map_err(|_| MetalMsmError::Capacity("fixed batch job allocation"))?;
        for (batch_index, batch) in scalar_batches.iter().enumerate() {
            if batch.len() > prepared.len() {
                return Err(MetalMsmError::ScalarBatchTooLong {
                    batch_index,
                    scalar_count: batch.len(),
                    prepared_base_count: prepared.len(),
                });
            }
            let scalar_offset = checked_u32(scalars.len(), "fixed batch scalar offset")?;
            let term_count = checked_u32(batch.len(), "fixed batch term count")?;
            let output_index = checked_u32(batch_index, "fixed batch output index")?;
            scalars.extend_from_slice(batch);
            jobs.push(MsmJobV1 { base_offset: 0, scalar_offset, term_count, output_index });
        }
        self.execute_fixed_jobs(prepared, scalars.as_slice(), &jobs)
    }

    /// Computes caller-described ragged MSMs using shared pre-shifted bases.
    pub fn msm_fixed_ragged(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[CanonicalScalar],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if jobs.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        self.execute_fixed_jobs(prepared, scalars, jobs)
    }

    /// Computes one fixed-base MSM from host-resident Montgomery residues.
    ///
    /// The input is uploaded without CPU field conversion. Metal validates
    /// reduction and converts it into private canonical scratch before
    /// recoding; that scratch and the temporary upload are securely erased
    /// before this synchronous call returns.
    pub fn msm_fixed_montgomery(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[RawFrMontgomery],
    ) -> Result<RawG1Jacobian, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if scalars.len() > prepared.len() {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: 0,
                scalar_count: scalars.len(),
                prepared_base_count: prepared.len(),
            });
        }
        let term_count = checked_u32(scalars.len(), "single Montgomery fixed MSM term count")?;
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count, output_index: 0 }];
        let mut outputs = self.execute_fixed_montgomery_jobs(prepared, scalars, &jobs)?;
        outputs
            .pop()
            .ok_or(MetalMsmError::InvalidRoute("single Montgomery fixed MSM returned no output"))
    }

    /// Computes prefix MSMs from host-resident Montgomery scalar batches.
    ///
    /// Batches may have different lengths. Their temporary contiguous copy is
    /// wiped after the synchronous submission. Call
    /// [`Self::msm_fixed_montgomery_ragged`] when the producer already owns one
    /// contiguous input arena.
    pub fn msm_fixed_montgomery_batch(
        &self,
        prepared: &PreparedFixedBases,
        scalar_batches: &[&[RawFrMontgomery]],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if scalar_batches.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        let total_scalars = scalar_batches.iter().try_fold(0usize, |total, batch| {
            total
                .checked_add(batch.len())
                .ok_or(MetalMsmError::Capacity("Montgomery fixed batch scalar count"))
        })?;
        let mut scalars =
            SensitiveVec::with_capacity(total_scalars, "Montgomery fixed batch scalar allocation")?;
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(scalar_batches.len())
            .map_err(|_| MetalMsmError::Capacity("Montgomery fixed batch job allocation"))?;
        for (batch_index, batch) in scalar_batches.iter().enumerate() {
            if batch.len() > prepared.len() {
                return Err(MetalMsmError::ScalarBatchTooLong {
                    batch_index,
                    scalar_count: batch.len(),
                    prepared_base_count: prepared.len(),
                });
            }
            let scalar_offset = checked_u32(scalars.len(), "Montgomery fixed batch offset")?;
            let term_count = checked_u32(batch.len(), "Montgomery fixed batch term count")?;
            let output_index = checked_u32(batch_index, "Montgomery fixed batch output index")?;
            scalars.extend_from_slice(batch);
            jobs.push(MsmJobV1 { base_offset: 0, scalar_offset, term_count, output_index });
        }
        self.execute_fixed_montgomery_jobs(prepared, scalars.as_slice(), &jobs)
    }

    /// Computes caller-described ragged fixed-base MSMs from one host arena of
    /// raw Montgomery residues.
    ///
    /// Jobs and residues are independently validated before and during Metal
    /// execution. No CPU canonicalization or curve arithmetic is performed.
    pub fn msm_fixed_montgomery_ragged(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[RawFrMontgomery],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if jobs.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        self.execute_fixed_montgomery_jobs(prepared, scalars, jobs)
    }

    /// Computes one fixed-base MSM directly from synchronized Montgomery residues.
    pub fn msm_fixed_resident(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &ResidentMontgomeryScalars,
    ) -> Result<RawG1Jacobian, MetalMsmError> {
        if scalars.len() > prepared.len() {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: 0,
                scalar_count: scalars.len(),
                prepared_base_count: prepared.len(),
            });
        }
        let term_count = checked_u32(scalars.len(), "single resident fixed MSM term count")?;
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count, output_index: 0 }];
        let mut outputs = self.execute_fixed_resident_jobs(prepared, &[scalars], &jobs)?;
        outputs
            .pop()
            .ok_or(MetalMsmError::InvalidRoute("single resident fixed MSM returned no output"))
    }

    /// Computes prefix MSMs from independent synchronized Montgomery-buffer views.
    ///
    /// Each view is paired with the same prepared-base prefix. Views may have
    /// different lengths, including zero, and are canonicalized into private
    /// scratch by Metal in the same command as scalar recoding and routing.
    pub fn msm_fixed_resident_batch(
        &self,
        prepared: &PreparedFixedBases,
        scalar_batches: &[ResidentMontgomeryScalars],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        if scalar_batches.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(scalar_batches.len())
            .map_err(|_| MetalMsmError::Capacity("resident fixed batch job allocation"))?;
        let mut scalar_offset = 0usize;
        for (batch_index, batch) in scalar_batches.iter().enumerate() {
            if batch.len() > prepared.len() {
                return Err(MetalMsmError::ScalarBatchTooLong {
                    batch_index,
                    scalar_count: batch.len(),
                    prepared_base_count: prepared.len(),
                });
            }
            let term_count = checked_u32(batch.len(), "resident fixed batch term count")?;
            jobs.push(MsmJobV1 {
                base_offset: 0,
                scalar_offset: checked_u32(scalar_offset, "resident fixed batch scalar offset")?,
                term_count,
                output_index: checked_u32(batch_index, "resident fixed batch output index")?,
            });
            scalar_offset = scalar_offset
                .checked_add(batch.len())
                .ok_or(MetalMsmError::Capacity("resident fixed batch scalar count"))?;
        }
        let sources = scalar_batches.iter().collect::<Vec<_>>();
        self.execute_fixed_resident_jobs(prepared, &sources, &jobs)
    }

    /// Computes caller-described ragged fixed-base MSMs from one resident view.
    ///
    /// Jobs address the view through `scalar_offset` and may select matching
    /// prepared-base ranges through `base_offset`. Output indices must be a
    /// permutation of `0..jobs.len()`.
    pub fn msm_fixed_resident_ragged(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &ResidentMontgomeryScalars,
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        if jobs.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        self.execute_fixed_resident_jobs(prepared, &[scalars], jobs)
    }

    /// Computes ragged fixed-base MSMs over concatenated resident scalar views.
    ///
    /// Job scalar offsets address the logical concatenation of `scalar_sources`
    /// in slice order. Metal copies and canonicalizes every synchronized view
    /// into private scratch in the same command as routing; scalar contents are
    /// never materialized on the host. Output indices must be a permutation of
    /// `0..jobs.len()`.
    pub fn msm_fixed_resident_sources_ragged(
        &self,
        prepared: &PreparedFixedBases,
        scalar_sources: &[ResidentMontgomeryScalars],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        if scalar_sources.is_empty() || jobs.is_empty() {
            return Err(MetalMsmError::EmptySubmission);
        }
        let sources = scalar_sources.iter().collect::<Vec<_>>();
        self.execute_fixed_resident_jobs(prepared, &sources, jobs)
    }

    fn execute_fixed_resident_jobs(
        &self,
        prepared: &PreparedFixedBases,
        sources: &[&ResidentMontgomeryScalars],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        for source in sources {
            if source.device_registry_id() != self.identity.registry_id() {
                return Err(MetalMsmError::ResidentScalarDeviceMismatch {
                    expected_registry_id: self.identity.registry_id(),
                    actual_registry_id: source.device_registry_id(),
                });
            }
        }
        let execution = translate_fixed_resident_route(
            prepared.len(),
            prepared.expanded_len(),
            sources,
            jobs,
            prepared.window_bits(),
        )?;
        self.context
            .execute_fixed_gpu_routed(&prepared.allocation, execution.borrowed())
            .map(|outputs| outputs.into_iter().map(raw_jacobian).collect())
            .map_err(execution_error)
    }

    fn execute_fixed_jobs(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[CanonicalScalar],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        let execution = translate_fixed_route(
            prepared.len(),
            prepared.expanded_len(),
            scalars,
            jobs,
            prepared.window_bits(),
        )?;
        self.context
            .execute_fixed_gpu_routed(&prepared.allocation, execution.borrowed())
            .map(|outputs| outputs.into_iter().map(raw_jacobian).collect())
            .map_err(execution_error)
    }

    fn execute_fixed_montgomery_jobs(
        &self,
        prepared: &PreparedFixedBases,
        scalars: &[RawFrMontgomery],
        jobs: &[MsmJobV1],
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_fixed_prepared(prepared)?;
        let execution = translate_fixed_montgomery_route(
            prepared.len(),
            prepared.expanded_len(),
            scalars,
            jobs,
            prepared.window_bits(),
        )?;
        self.context
            .execute_fixed_gpu_routed(&prepared.allocation, execution.borrowed())
            .map(|outputs| outputs.into_iter().map(raw_jacobian).collect())
            .map_err(execution_error)
    }

    fn execute_jobs(
        &self,
        prepared: &PreparedBases,
        scalars: &[CanonicalScalar],
        jobs: &[MsmJobV1],
        window_bits: WindowBits,
    ) -> Result<Vec<RawG1Jacobian>, MetalMsmError> {
        self.check_prepared(prepared)?;
        if matches!(
            self.reduction_strategy,
            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
                | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
                | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
                | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
                | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
                | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
                | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
        ) {
            let execution = translate_gpu_route(
                prepared.len(),
                scalars,
                jobs,
                window_bits,
                self.reduction_strategy,
            )?;
            return self
                .context
                .execute_gpu_routed(&prepared.allocation, execution.borrowed())
                .map(|outputs| outputs.into_iter().map(raw_jacobian).collect())
                .map_err(execution_error);
        }
        let mut route = route_jobs(prepared.len(), scalars, jobs, window_bits)
            .map_err(|error| MetalMsmError::Routing(format!("{error}")))?;
        let execution = match translate_route(prepared.len(), &route, self.reduction_strategy) {
            Ok(execution) => execution,
            Err(error) => {
                route.erase();
                return Err(error);
            }
        };
        let result = self
            .context
            .execute(&prepared.allocation, execution.borrowed())
            .map(|outputs| outputs.into_iter().map(raw_jacobian).collect())
            .map_err(execution_error);
        route.erase();
        result
    }

    fn check_prepared(&self, prepared: &PreparedBases) -> Result<(), MetalMsmError> {
        if Rc::ptr_eq(&self.context, &prepared.context) {
            Ok(())
        } else {
            Err(MetalMsmError::PreparedBasesEngineMismatch)
        }
    }

    fn check_fixed_prepared(&self, prepared: &PreparedFixedBases) -> Result<(), MetalMsmError> {
        if Rc::ptr_eq(&self.context, &prepared.context) {
            Ok(())
        } else {
            Err(MetalMsmError::PreparedBasesEngineMismatch)
        }
    }
}

struct SensitiveGpuRoutedPlan {
    reduction_strategy: MetalMsmReductionStrategy,
    scalars: SensitiveVec<GpuScalar>,
    point_indices: SensitiveVec<u32>,
    jobs: SensitiveVec<GpuRouteJobV3>,
    parameters: GpuRouteParametersV3,
    packing_eligible: bool,
    output_count: usize,
    window_point_count: usize,
}

impl SensitiveGpuRoutedPlan {
    fn borrowed(&self) -> self::sys::GpuRoutedExecutionPlan<'_> {
        self::sys::GpuRoutedExecutionPlan {
            reduction_strategy: self.reduction_strategy,
            scalars: self.scalars.as_slice(),
            point_indices: self.point_indices.as_slice(),
            jobs: self.jobs.as_slice(),
            parameters: self.parameters,
            packing_eligible: self.packing_eligible,
            output_count: self.output_count,
            window_point_count: self.window_point_count,
        }
    }
}

impl Drop for SensitiveGpuRoutedPlan {
    fn drop(&mut self) {
        self.parameters = GpuRouteParametersV3::default();
        self.packing_eligible = false;
        self.output_count = 0;
        self.window_point_count = 0;
    }
}

struct SensitiveCompactedFixedGpuRoutedPlan<T: Copy + Default> {
    scalars: SensitiveVec<T>,
    point_indices: SensitiveVec<u32>,
    jobs: SensitiveVec<GpuRouteJobV3>,
    parameters: FixedRouteParametersV10,
    output_count: usize,
}

struct SensitiveFixedGpuRoutedPlan(SensitiveCompactedFixedGpuRoutedPlan<GpuScalar>);

impl SensitiveFixedGpuRoutedPlan {
    fn borrowed(&self) -> self::sys::FixedGpuRoutedExecutionPlan<'_> {
        self::sys::FixedGpuRoutedExecutionPlan {
            scalar_input: self::sys::FixedScalarInput::Host {
                scalars: self.0.scalars.as_slice(),
                point_indices: self.0.point_indices.as_slice(),
            },
            jobs: self.0.jobs.as_slice(),
            parameters: self.0.parameters,
            output_count: self.0.output_count,
        }
    }
}

impl<T: Copy + Default> Drop for SensitiveCompactedFixedGpuRoutedPlan<T> {
    fn drop(&mut self) {
        self.parameters = FixedRouteParametersV10::default();
        self.output_count = 0;
    }
}

struct SensitiveResidentFixedGpuRoutedPlan<'a> {
    sources: Vec<self::sys::ResidentScalarSource<'a>>,
    route: SensitiveUncompactedFixedRoute,
}

struct SensitiveMontgomeryFixedGpuRoutedPlan {
    scalars: SensitiveVec<RawFrMontgomery>,
    route: SensitiveUncompactedFixedRoute,
}

struct SensitiveUncompactedFixedRoute {
    jobs: SensitiveVec<GpuRouteJobV3>,
    parameters: FixedRouteParametersV10,
    output_count: usize,
}

impl SensitiveResidentFixedGpuRoutedPlan<'_> {
    fn borrowed(&self) -> self::sys::FixedGpuRoutedExecutionPlan<'_> {
        self::sys::FixedGpuRoutedExecutionPlan {
            scalar_input: self::sys::FixedScalarInput::Resident(&self.sources),
            jobs: self.route.jobs.as_slice(),
            parameters: self.route.parameters,
            output_count: self.route.output_count,
        }
    }
}

impl SensitiveMontgomeryFixedGpuRoutedPlan {
    fn borrowed(&self) -> self::sys::FixedGpuRoutedExecutionPlan<'_> {
        self::sys::FixedGpuRoutedExecutionPlan {
            scalar_input: self::sys::FixedScalarInput::HostMontgomery(self.scalars.as_slice()),
            jobs: self.route.jobs.as_slice(),
            parameters: self.route.parameters,
            output_count: self.route.output_count,
        }
    }
}

impl Drop for SensitiveUncompactedFixedRoute {
    fn drop(&mut self) {
        self.parameters = FixedRouteParametersV10::default();
        self.output_count = 0;
    }
}

struct SensitiveExecutionPlan {
    reduction_strategy: MetalMsmReductionStrategy,
    parallel_threads: usize,
    parallel_tile_records: u32,
    records: SensitiveVec<SignedRecord>,
    affine_records: SensitiveVec<SignedRecord>,
    affine_levels: SensitiveVec<BatchAffineLevelPlan>,
    affine_destination_map: SensitiveVec<u32>,
    affine_max_pairs: usize,
    affine_max_tree_slots: usize,
    runs: SensitiveVec<BucketRun>,
    tiles: SensitiveVec<BucketTileV1>,
    tile_runs: SensitiveVec<TileRunV1>,
    dense_map: SensitiveVec<u32>,
    windows: SensitiveVec<WindowDescriptor>,
    collects: SensitiveVec<CollectDescriptor>,
    output_count: usize,
}

impl SensitiveExecutionPlan {
    fn borrowed(&self) -> ExecutionPlan<'_> {
        ExecutionPlan {
            reduction_strategy: self.reduction_strategy,
            parallel_threads: self.parallel_threads,
            parallel_tile_records: self.parallel_tile_records,
            records: self.records.as_slice(),
            affine_records: self.affine_records.as_slice(),
            affine_levels: self.affine_levels.as_slice(),
            affine_destination_map: self.affine_destination_map.as_slice(),
            affine_max_pairs: self.affine_max_pairs,
            affine_max_tree_slots: self.affine_max_tree_slots,
            runs: self.runs.as_slice(),
            tiles: self.tiles.as_slice(),
            tile_runs: self.tile_runs.as_slice(),
            dense_map: self.dense_map.as_slice(),
            windows: self.windows.as_slice(),
            window_point_count: self.windows.len(),
            collects: self.collects.as_slice(),
            output_count: self.output_count,
        }
    }
}

impl Drop for SensitiveExecutionPlan {
    fn drop(&mut self) {
        self.parallel_threads = 0;
        self.parallel_tile_records = 0;
        self.affine_max_pairs = 0;
        self.affine_max_tree_slots = 0;
        self.output_count = 0;
    }
}

struct SensitiveVec<T: Copy + Default>(Vec<T>);

impl<T: Copy + Default> SensitiveVec<T> {
    fn with_capacity(capacity: usize, label: &'static str) -> Result<Self, MetalMsmError> {
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|_| MetalMsmError::Capacity(label))?;
        Ok(Self(values))
    }

    fn filled(len: usize, value: T, label: &'static str) -> Result<Self, MetalMsmError> {
        let mut values = Self::with_capacity(len, label)?;
        values.0.resize(len, value);
        Ok(values)
    }

    fn push(&mut self, value: T) {
        self.0.push(value);
    }

    fn extend_from_slice(&mut self, values: &[T]) {
        self.0.extend_from_slice(values);
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T: Copy + Default> core::ops::Deref for SensitiveVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Copy + Default> core::ops::DerefMut for SensitiveVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Copy + Default> Drop for SensitiveVec<T> {
    fn drop(&mut self) {
        wipe_values(&mut self.0);
    }
}

struct SensitiveScalars(Vec<CanonicalScalar>);

impl SensitiveScalars {
    fn with_capacity(capacity: usize) -> Result<Self, MetalMsmError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| MetalMsmError::Capacity("batch scalar allocation"))?;
        Ok(Self(values))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn extend_from_slice(&mut self, values: &[CanonicalScalar]) {
        self.0.extend_from_slice(values);
    }

    fn as_slice(&self) -> &[CanonicalScalar] {
        &self.0
    }
}

impl Drop for SensitiveScalars {
    fn drop(&mut self) {
        wipe_values(&mut self.0);
    }
}

fn translate_route(
    prepared_base_count: usize,
    route: &RoutePlan,
    reduction_strategy: MetalMsmReductionStrategy,
) -> Result<SensitiveExecutionPlan, MetalMsmError> {
    if route.prepared_base_count() as usize != prepared_base_count {
        return Err(MetalMsmError::InvalidRoute("prepared-base count changed after routing"));
    }
    let output_count = route.output_count() as usize;
    if output_count == 0 || route.jobs().len() != output_count {
        return Err(MetalMsmError::InvalidRoute("route has no dense output jobs"));
    }
    let windows_per_scalar = route.windows_per_scalar() as usize;
    let buckets_per_window = route.buckets_per_window() as usize;
    let window_point_count = output_count
        .checked_mul(windows_per_scalar)
        .ok_or(MetalMsmError::Capacity("window-point count"))?;
    let expected_dense_count = window_point_count
        .checked_mul(buckets_per_window)
        .ok_or(MetalMsmError::Capacity("dense-bucket count"))?;
    if route.dense_bucket_count() as usize != expected_dense_count {
        return Err(MetalMsmError::InvalidRoute("dense bucket count is inconsistent"));
    }

    let mut output_seen = try_default_vec::<bool>(output_count, "output validation")?;
    for job in route.jobs() {
        let output = job.output_index as usize;
        if output >= output_count || output_seen[output] {
            return Err(MetalMsmError::InvalidRoute("job outputs are not a permutation"));
        }
        output_seen[output] = true;
    }

    let mut records =
        SensitiveVec::with_capacity(route.records().len(), "private record allocation")?;
    for record in route.records() {
        if record.point_index() as usize >= prepared_base_count {
            return Err(MetalMsmError::InvalidRoute("routed point exceeds prepared bases"));
        }
        records.push(SignedRecord {
            point_index: record.point_index(),
            flags: u32::from(record.is_negative()),
        });
    }

    let (
        affine_records,
        affine_levels,
        affine_destination_map,
        affine_max_pairs,
        affine_max_tree_slots,
    ) = match reduction_strategy {
        MetalMsmReductionStrategy::GlobalBatchAffineV1 => {
            let run_lengths = Zeroizing::new(
                route.runs().iter().map(|run| run.record_count as usize).collect::<Vec<_>>(),
            );
            let affine_plan = compile_balanced_run_plan(&run_lengths).map_err(|_| {
                MetalMsmError::InvalidRoute("balanced affine topology rejected the route")
            })?;
            if affine_plan.leaf_count as usize != records.len()
                || affine_plan.root_run_indices.len() != route.runs().len()
            {
                return Err(MetalMsmError::InvalidRoute(
                    "balanced affine topology does not match routed records",
                ));
            }
            let mut affine_records =
                SensitiveVec::with_capacity(records.len(), "affine record-order allocation")?;
            for &source_index in affine_plan.initial_source_order.iter() {
                let record = records.as_slice().get(source_index as usize).ok_or(
                    MetalMsmError::InvalidRoute(
                        "balanced affine source order exceeds routed records",
                    ),
                )?;
                affine_records.push(*record);
            }
            let mut affine_levels =
                SensitiveVec::with_capacity(affine_plan.levels.len(), "affine level allocation")?;
            for &level in affine_plan.levels.iter() {
                affine_levels.push(level);
            }
            let mut affine_destination_map = SensitiveVec::with_capacity(
                affine_plan.destination_map.len(),
                "affine destination-map allocation",
            )?;
            for &destination in affine_plan.destination_map.iter() {
                affine_destination_map.push(destination);
            }
            let affine_max_pairs =
                affine_plan.levels.iter().map(|level| level.pair_count as usize).max().unwrap_or(0);
            let affine_max_tree_slots =
                affine_plan.levels.iter().try_fold(0usize, |maximum, level| {
                    Ok::<_, MetalMsmError>(
                        maximum.max(batch_affine_product_tree_slots(level.pair_count)?),
                    )
                })?;
            (
                affine_records,
                affine_levels,
                affine_destination_map,
                affine_max_pairs,
                affine_max_tree_slots,
            )
        }
        MetalMsmReductionStrategy::ParallelProjectiveV1
        | MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2
        | MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
        | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
        | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
        | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
        | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
        | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
        | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => (
            SensitiveVec::with_capacity(0, "empty affine record allocation")?,
            SensitiveVec::with_capacity(0, "empty affine level allocation")?,
            SensitiveVec::with_capacity(0, "empty affine map allocation")?,
            0,
            0,
        ),
    };

    let (parallel_threads, tile_records) =
        parallel_geometry(reduction_strategy, records.len(), route.runs().len());
    let tile_count = route.runs().iter().try_fold(0usize, |total, run| {
        total
            .checked_add((run.record_count as usize).div_ceil(tile_records as usize))
            .ok_or(MetalMsmError::Capacity("parallel bucket tile count"))
    })?;
    let mut runs = SensitiveVec::with_capacity(route.runs().len(), "private run allocation")?;
    let mut tiles = SensitiveVec::with_capacity(tile_count, "bucket tile allocation")?;
    let mut tile_runs = SensitiveVec::with_capacity(route.runs().len(), "tile run allocation")?;
    let mut dense_map =
        SensitiveVec::filled(expected_dense_count, EMPTY_BUCKET, "dense-map allocation")?;
    let mut expected_record_offset = 0usize;
    let mut previous_dense = None;
    for (run_index, run) in route.runs().iter().enumerate() {
        if run.reserved != 0 || run.record_count == 0 {
            return Err(MetalMsmError::InvalidRoute("portable bucket run is malformed"));
        }
        let record_offset = run.record_offset as usize;
        let record_end = record_offset
            .checked_add(run.record_count as usize)
            .ok_or(MetalMsmError::InvalidRoute("portable run range overflows"))?;
        if record_offset != expected_record_offset || record_end > records.len() {
            return Err(MetalMsmError::InvalidRoute(
                "portable runs do not partition routed records",
            ));
        }
        let dense_index = run.dense_bucket_index as usize;
        if dense_index >= dense_map.len()
            || previous_dense.is_some_and(|previous| dense_index <= previous)
            || dense_map[dense_index] != EMPTY_BUCKET
        {
            return Err(MetalMsmError::InvalidRoute(
                "portable runs are not uniquely ordered by dense bucket",
            ));
        }
        dense_map[dense_index] = checked_u32(run_index, "private run index")?;
        let tile_offset = checked_u32(tiles.len(), "parallel tile offset")?;
        let mut remaining = run.record_count;
        let mut tile_record_offset = run.record_offset;
        while remaining != 0 {
            let record_count = remaining.min(tile_records);
            tiles.push(BucketTileV1 {
                record_offset: tile_record_offset,
                record_count,
                run_index: checked_u32(run_index, "parallel tile run index")?,
                reserved: 0,
            });
            tile_record_offset = tile_record_offset
                .checked_add(record_count)
                .ok_or(MetalMsmError::Capacity("parallel tile record offset"))?;
            remaining -= record_count;
        }
        tile_runs.push(TileRunV1 {
            tile_offset,
            tile_count: checked_u32(tiles.len(), "parallel tile count")?
                .checked_sub(tile_offset)
                .ok_or(MetalMsmError::InvalidRoute("parallel tile span underflow"))?,
            reserved0: 0,
            reserved1: 0,
        });
        runs.push(BucketRun {
            record_offset: run.record_offset,
            record_count: run.record_count,
            reserved0: 0,
            reserved1: 0,
        });
        expected_record_offset = record_end;
        previous_dense = Some(dense_index);
    }
    if expected_record_offset != records.len() {
        return Err(MetalMsmError::InvalidRoute("portable runs do not consume routed records"));
    }

    let mut windows =
        SensitiveVec::with_capacity(window_point_count, "window descriptor allocation")?;
    for window_index in 0..window_point_count {
        windows.push(WindowDescriptor {
            dense_map_offset: checked_u32(
                window_index
                    .checked_mul(buckets_per_window)
                    .ok_or(MetalMsmError::Capacity("window dense offset"))?,
                "window dense offset",
            )?,
            bucket_count: checked_u32(buckets_per_window, "window bucket count")?,
            window_point_index: checked_u32(window_index, "window point index")?,
            reserved: 0,
        });
    }

    let mut collects =
        SensitiveVec::with_capacity(output_count, "collection descriptor allocation")?;
    for output_index in 0..output_count {
        collects.push(CollectDescriptor {
            window_point_offset: checked_u32(
                output_index
                    .checked_mul(windows_per_scalar)
                    .ok_or(MetalMsmError::Capacity("collection window offset"))?,
                "collection window offset",
            )?,
            window_count: checked_u32(windows_per_scalar, "collection window count")?,
            window_bits: route.window_bits().get(),
            output_index: checked_u32(output_index, "collection output index")?,
        });
    }

    if tiles.len() != tile_count || tile_runs.len() != runs.len() {
        return Err(MetalMsmError::InvalidRoute(
            "parallel bucket descriptors do not cover the route",
        ));
    }

    Ok(SensitiveExecutionPlan {
        reduction_strategy,
        parallel_threads,
        parallel_tile_records: tile_records,
        records,
        affine_records,
        affine_levels,
        affine_destination_map,
        affine_max_pairs,
        affine_max_tree_slots,
        runs,
        tiles,
        tile_runs,
        dense_map,
        windows,
        collects,
        output_count,
    })
}

fn translate_gpu_route(
    prepared_base_count: usize,
    scalars: &[CanonicalScalar],
    jobs: &[MsmJobV1],
    window_bits: WindowBits,
    reduction_strategy: MetalMsmReductionStrategy,
) -> Result<SensitiveGpuRoutedPlan, MetalMsmError> {
    if !matches!(
        reduction_strategy,
        MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
            | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
    ) {
        return Err(MetalMsmError::InvalidRoute(
            "GPU-routed translation requires a GPU-routed strategy",
        ));
    }
    if jobs.is_empty() {
        return Err(MetalMsmError::EmptySubmission);
    }
    let base_count = checked_u32(prepared_base_count, "GPU-routed prepared-base count")?;
    let input_scalar_count = checked_u32(scalars.len(), "GPU-routed input scalar count")?;
    let output_count = jobs.len();
    let output_count_u32 = checked_u32(output_count, "GPU-routed output count")?;
    let windows = window_bits.windows_per_scalar();
    let buckets = window_bits.buckets_per_window();
    let dense_bucket_count = output_count_u32
        .checked_mul(windows)
        .and_then(|value| value.checked_mul(buckets))
        .ok_or(MetalMsmError::Capacity("GPU-routed dense bucket count"))?;
    let window_point_count = output_count
        .checked_mul(windows as usize)
        .ok_or(MetalMsmError::Capacity("GPU-routed window-point count"))?;

    let mut outputs_seen = try_default_vec::<bool>(output_count, "GPU-routed output validation")?;
    let mut gpu_scalars =
        SensitiveVec::with_capacity(scalars.len(), "GPU-routed scalar allocation")?;
    let mut point_indices =
        SensitiveVec::with_capacity(scalars.len(), "GPU-routed point-index allocation")?;
    let mut gpu_jobs = SensitiveVec::with_capacity(jobs.len(), "GPU-routed job allocation")?;
    let mut record_segment_offset = 0_u32;
    let mut max_terms_per_job = 0_u32;
    let mut packing_eligible = true;
    for (job_index, job) in jobs.iter().copied().enumerate() {
        let output_index = job.output_index as usize;
        if output_index >= output_count || outputs_seen[output_index] {
            return Err(MetalMsmError::InvalidRoute(
                "GPU-routed job outputs are not a permutation",
            ));
        }
        outputs_seen[output_index] = true;
        let base_end = job
            .base_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("GPU-routed base range overflows"))?;
        if base_end > base_count {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: job_index,
                scalar_count: job.term_count as usize,
                prepared_base_count,
            });
        }
        let scalar_end = job
            .scalar_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("GPU-routed scalar range overflows"))?;
        if scalar_end > input_scalar_count {
            return Err(MetalMsmError::InvalidRoute(
                "GPU-routed scalar range exceeds the submission",
            ));
        }
        let scalar_range = job.scalar_offset as usize..scalar_end as usize;
        let job_scalars = &scalars[scalar_range];
        let nonzero = job_scalars.iter().filter(|scalar| !scalar.is_zero()).count();
        // Sparse proof polynomials produce a few disproportionately heavy
        // buckets. Preserve v5/v6 reduction unless every output is at least
        // 75% nonzero; this profile already permits scalar-dependent routing.
        packing_eligible &=
            nonzero.saturating_mul(4) >= (job.term_count as usize).saturating_mul(3);
        let compact_scalar_offset =
            checked_u32(gpu_scalars.len(), "GPU-routed compact scalar offset")?;
        for (local_index, scalar) in job_scalars.iter().copied().enumerate() {
            if scalar.is_zero() {
                continue;
            }
            let local_index = checked_u32(local_index, "GPU-routed local point index")?;
            let point_index = job
                .base_offset
                .checked_add(local_index)
                .ok_or(MetalMsmError::Capacity("GPU-routed compact point index"))?;
            gpu_scalars.push(GpuScalar { limbs: scalar.to_le_limbs() });
            point_indices.push(point_index);
        }
        let compact_term_count = checked_u32(
            gpu_scalars.len() - compact_scalar_offset as usize,
            "GPU-routed compact term count",
        )?;
        gpu_jobs.push(GpuRouteJobV3 {
            base_offset: job.base_offset,
            scalar_offset: compact_scalar_offset,
            term_count: compact_term_count,
            output_index: job.output_index,
            record_segment_offset,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        });
        record_segment_offset = record_segment_offset
            .checked_add(
                compact_term_count
                    .checked_mul(windows)
                    .ok_or(MetalMsmError::Capacity("GPU-routed record segment"))?,
            )
            .ok_or(MetalMsmError::Capacity("GPU-routed record count"))?;
        max_terms_per_job = max_terms_per_job.max(compact_term_count);
    }

    let scalar_count = checked_u32(gpu_scalars.len(), "GPU-routed compact scalar count")?;
    let threads_per_bucket =
        if reduction_strategy == MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 {
            PARALLEL_JACOBIAN_LARGE_THREADS
        } else {
            parallel_geometry(
                reduction_strategy,
                record_segment_offset as usize,
                dense_bucket_count as usize,
            )
            .0
        };
    let parameters = GpuRouteParametersV3 {
        abi_version: GPU_ROUTED_DENSE_ABI_VERSION_V4,
        base_count,
        scalar_count,
        job_count: output_count_u32,
        output_count: output_count_u32,
        window_bits: window_bits.get(),
        windows_per_scalar: windows,
        buckets_per_window: buckets,
        dense_bucket_count,
        max_record_count: record_segment_offset,
        max_terms_per_job,
        threads_per_group: threads_per_bucket as u32,
        buckets_per_group: match (
            reduction_strategy,
            window_bits,
            output_count_u32,
            packing_eligible,
        ) {
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, WindowBits::W9, _, _) => 4,
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, WindowBits::W10, _, _) => 8,
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, WindowBits::W8, 4.., true) => 8,
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, WindowBits::W9, 4.., true) => 4,
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, WindowBits::W10, 2.., true) => {
                8
            }
            _ => 2,
        },
        reserved0: 0,
        reserved1: 0,
        reserved2: 0,
    };
    Ok(SensitiveGpuRoutedPlan {
        reduction_strategy,
        scalars: gpu_scalars,
        point_indices,
        jobs: gpu_jobs,
        parameters,
        packing_eligible,
        output_count,
        window_point_count,
    })
}

fn translate_fixed_route(
    prepared_base_count: usize,
    shifted_base_count: usize,
    scalars: &[CanonicalScalar],
    jobs: &[MsmJobV1],
    window_bits: FixedWindowBits,
) -> Result<SensitiveFixedGpuRoutedPlan, MetalMsmError> {
    translate_compacted_fixed_route(
        prepared_base_count,
        shifted_base_count,
        scalars,
        jobs,
        window_bits,
        |scalar| (!scalar.is_zero()).then(|| GpuScalar { limbs: scalar.to_le_limbs() }),
    )
    .map(SensitiveFixedGpuRoutedPlan)
}

fn translate_compacted_fixed_route<T, U>(
    prepared_base_count: usize,
    shifted_base_count: usize,
    scalars: &[T],
    jobs: &[MsmJobV1],
    window_bits: FixedWindowBits,
    mut retain: impl FnMut(T) -> Option<U>,
) -> Result<SensitiveCompactedFixedGpuRoutedPlan<U>, MetalMsmError>
where
    T: Copy,
    U: Copy + Default,
{
    if jobs.is_empty() {
        return Err(MetalMsmError::EmptySubmission);
    }
    let base_count = checked_u32(prepared_base_count, "fixed prepared-base count")?;
    let shifted_base_count = checked_u32(shifted_base_count, "fixed shifted-base count")?;
    let input_scalar_count = checked_u32(scalars.len(), "fixed input scalar count")?;
    let output_count = jobs.len();
    let output_count_u32 = checked_u32(output_count, "fixed output count")?;
    let windows = window_bits.windows_per_scalar();
    let buckets = window_bits.buckets_per_output();
    let expected_shifted = base_count
        .checked_mul(windows)
        .ok_or(MetalMsmError::Capacity("fixed shifted-base count"))?;
    if shifted_base_count != expected_shifted {
        return Err(MetalMsmError::InvalidRoute(
            "fixed prepared-base expansion does not match its window width",
        ));
    }
    let dense_bucket_count = output_count_u32
        .checked_mul(buckets)
        .ok_or(MetalMsmError::Capacity("fixed dense bucket count"))?;
    let integration_block_buckets = FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10;
    if !buckets.is_multiple_of(integration_block_buckets) {
        return Err(MetalMsmError::InvalidRoute(
            "fixed bucket count is not divisible by the integration block",
        ));
    }
    let integration_blocks_per_output = buckets / integration_block_buckets;

    let mut outputs_seen = try_default_vec::<bool>(output_count, "fixed output validation")?;
    let mut gpu_scalars = SensitiveVec::with_capacity(scalars.len(), "fixed scalar allocation")?;
    let mut point_indices =
        SensitiveVec::with_capacity(scalars.len(), "fixed point-index allocation")?;
    let mut gpu_jobs = SensitiveVec::with_capacity(jobs.len(), "fixed job allocation")?;
    let mut record_segment_offset = 0_u32;
    let mut max_terms_per_job = 0_u32;
    for (job_index, job) in jobs.iter().copied().enumerate() {
        let output_index = job.output_index as usize;
        if output_index >= output_count || outputs_seen[output_index] {
            return Err(MetalMsmError::InvalidRoute("fixed job outputs are not a permutation"));
        }
        outputs_seen[output_index] = true;
        let base_end = job
            .base_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("fixed base range overflows"))?;
        if base_end > base_count {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: job_index,
                scalar_count: job.term_count as usize,
                prepared_base_count,
            });
        }
        let scalar_end = job
            .scalar_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("fixed scalar range overflows"))?;
        if scalar_end > input_scalar_count {
            return Err(MetalMsmError::InvalidRoute("fixed scalar range exceeds the submission"));
        }
        let compact_scalar_offset = checked_u32(gpu_scalars.len(), "fixed compact scalar offset")?;
        for (local_index, scalar) in
            scalars[job.scalar_offset as usize..scalar_end as usize].iter().copied().enumerate()
        {
            let Some(scalar) = retain(scalar) else {
                continue;
            };
            let point_index = job
                .base_offset
                .checked_add(checked_u32(local_index, "fixed local point index")?)
                .ok_or(MetalMsmError::Capacity("fixed compact point index"))?;
            gpu_scalars.push(scalar);
            point_indices.push(point_index);
        }
        let compact_term_count = checked_u32(
            gpu_scalars.len() - compact_scalar_offset as usize,
            "fixed compact term count",
        )?;
        gpu_jobs.push(GpuRouteJobV3 {
            base_offset: job.base_offset,
            scalar_offset: compact_scalar_offset,
            term_count: compact_term_count,
            output_index: job.output_index,
            record_segment_offset,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        });
        record_segment_offset = record_segment_offset
            .checked_add(
                compact_term_count
                    .checked_mul(windows)
                    .ok_or(MetalMsmError::Capacity("fixed record segment"))?,
            )
            .ok_or(MetalMsmError::Capacity("fixed record count"))?;
        max_terms_per_job = max_terms_per_job.max(compact_term_count);
    }

    let parameters = FixedRouteParametersV10 {
        abi_version: FIXED_ROUTE_ABI_VERSION_V10,
        base_count,
        shifted_base_count,
        scalar_count: checked_u32(gpu_scalars.len(), "fixed compact scalar count")?,
        job_count: output_count_u32,
        output_count: output_count_u32,
        window_bits: window_bits.get(),
        windows_per_scalar: windows,
        buckets_per_output: buckets,
        dense_bucket_count,
        max_record_count: record_segment_offset,
        max_terms_per_job,
        threads_per_group: FIXED_ROUTE_THREADS_PER_GROUP_V10,
        buckets_per_group: fixed_route_buckets_per_group_v10(window_bits.get(), output_count_u32),
        integration_block_buckets,
        integration_blocks_per_output,
    };
    Ok(SensitiveCompactedFixedGpuRoutedPlan {
        scalars: gpu_scalars,
        point_indices,
        jobs: gpu_jobs,
        parameters,
        output_count,
    })
}

fn translate_fixed_resident_route<'a>(
    prepared_base_count: usize,
    shifted_base_count: usize,
    resident_sources: &[&'a ResidentMontgomeryScalars],
    jobs: &[MsmJobV1],
    window_bits: FixedWindowBits,
) -> Result<SensitiveResidentFixedGpuRoutedPlan<'a>, MetalMsmError> {
    if resident_sources.is_empty() {
        return Err(MetalMsmError::InvalidResidentScalarBuffer(
            "submission has no resident sources".into(),
        ));
    }
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(resident_sources.len())
        .map_err(|_| MetalMsmError::Capacity("resident scalar source allocation"))?;
    let mut input_scalar_count = 0usize;
    for source in resident_sources {
        let destination_offset =
            checked_u32(input_scalar_count, "resident scalar source destination offset")?;
        sources.push(self::sys::ResidentScalarSource { buffer: &source.inner, destination_offset });
        input_scalar_count = input_scalar_count
            .checked_add(source.len())
            .ok_or(MetalMsmError::Capacity("resident scalar source count"))?;
    }
    let route = translate_uncompacted_fixed_route(
        prepared_base_count,
        shifted_base_count,
        input_scalar_count,
        jobs,
        window_bits,
    )?;
    Ok(SensitiveResidentFixedGpuRoutedPlan { sources, route })
}

fn translate_fixed_montgomery_route(
    prepared_base_count: usize,
    shifted_base_count: usize,
    scalars: &[RawFrMontgomery],
    jobs: &[MsmJobV1],
    window_bits: FixedWindowBits,
) -> Result<SensitiveMontgomeryFixedGpuRoutedPlan, MetalMsmError> {
    let route = translate_uncompacted_fixed_route(
        prepared_base_count,
        shifted_base_count,
        scalars.len(),
        jobs,
        window_bits,
    )?;
    let mut copied =
        SensitiveVec::with_capacity(scalars.len(), "fixed Montgomery scalar allocation")?;
    copied.extend_from_slice(scalars);
    Ok(SensitiveMontgomeryFixedGpuRoutedPlan { scalars: copied, route })
}

fn translate_uncompacted_fixed_route(
    prepared_base_count: usize,
    shifted_base_count: usize,
    input_scalar_count: usize,
    jobs: &[MsmJobV1],
    window_bits: FixedWindowBits,
) -> Result<SensitiveUncompactedFixedRoute, MetalMsmError> {
    if jobs.is_empty() {
        return Err(MetalMsmError::EmptySubmission);
    }
    let base_count = checked_u32(prepared_base_count, "uncompacted fixed prepared-base count")?;
    let shifted_base_count =
        checked_u32(shifted_base_count, "uncompacted fixed shifted-base count")?;
    let windows = window_bits.windows_per_scalar();
    let buckets = window_bits.buckets_per_output();
    let expected_shifted = base_count
        .checked_mul(windows)
        .ok_or(MetalMsmError::Capacity("uncompacted fixed shifted-base count"))?;
    if shifted_base_count != expected_shifted {
        return Err(MetalMsmError::InvalidRoute(
            "uncompacted fixed prepared-base expansion does not match its window width",
        ));
    }
    let input_scalar_count_u32 =
        checked_u32(input_scalar_count, "uncompacted fixed input scalar count")?;
    let output_count = jobs.len();
    let output_count_u32 = checked_u32(output_count, "uncompacted fixed output count")?;
    let dense_bucket_count = output_count_u32
        .checked_mul(buckets)
        .ok_or(MetalMsmError::Capacity("uncompacted fixed dense bucket count"))?;
    let integration_block_buckets = FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10;
    if !buckets.is_multiple_of(integration_block_buckets) {
        return Err(MetalMsmError::InvalidRoute(
            "uncompacted fixed bucket count is not divisible by the integration block",
        ));
    }
    let integration_blocks_per_output = buckets / integration_block_buckets;

    let mut outputs_seen =
        try_default_vec::<bool>(output_count, "uncompacted fixed output validation")?;
    let mut gpu_jobs = SensitiveVec::with_capacity(jobs.len(), "uncompacted fixed job allocation")?;
    let mut record_segment_offset = 0_u32;
    let mut max_terms_per_job = 0_u32;
    for (job_index, job) in jobs.iter().copied().enumerate() {
        let output_index = job.output_index as usize;
        if output_index >= output_count || outputs_seen[output_index] {
            return Err(MetalMsmError::InvalidRoute(
                "uncompacted fixed job outputs are not a permutation",
            ));
        }
        outputs_seen[output_index] = true;
        let base_end = job
            .base_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("uncompacted fixed base range overflows"))?;
        if base_end > base_count {
            return Err(MetalMsmError::ScalarBatchTooLong {
                batch_index: job_index,
                scalar_count: job.term_count as usize,
                prepared_base_count,
            });
        }
        let scalar_end = job
            .scalar_offset
            .checked_add(job.term_count)
            .ok_or(MetalMsmError::InvalidRoute("uncompacted fixed scalar range overflows"))?;
        if scalar_end > input_scalar_count_u32 {
            return Err(MetalMsmError::InvalidRoute(
                "uncompacted fixed scalar range exceeds the submission",
            ));
        }
        gpu_jobs.push(GpuRouteJobV3 {
            base_offset: job.base_offset,
            scalar_offset: job.scalar_offset,
            term_count: job.term_count,
            output_index: job.output_index,
            record_segment_offset,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        });
        record_segment_offset = record_segment_offset
            .checked_add(
                job.term_count
                    .checked_mul(windows)
                    .ok_or(MetalMsmError::Capacity("uncompacted fixed record segment"))?,
            )
            .ok_or(MetalMsmError::Capacity("uncompacted fixed record count"))?;
        max_terms_per_job = max_terms_per_job.max(job.term_count);
    }

    let parameters = FixedRouteParametersV10 {
        abi_version: FIXED_ROUTE_ABI_VERSION_V10,
        base_count,
        shifted_base_count,
        scalar_count: input_scalar_count_u32,
        job_count: output_count_u32,
        output_count: output_count_u32,
        window_bits: window_bits.get(),
        windows_per_scalar: windows,
        buckets_per_output: buckets,
        dense_bucket_count,
        max_record_count: record_segment_offset,
        max_terms_per_job,
        threads_per_group: FIXED_ROUTE_THREADS_PER_GROUP_V10,
        buckets_per_group: fixed_route_buckets_per_group_v10(window_bits.get(), output_count_u32),
        integration_block_buckets,
        integration_blocks_per_output,
    };
    Ok(SensitiveUncompactedFixedRoute { jobs: gpu_jobs, parameters, output_count })
}

fn parallel_geometry(
    reduction_strategy: MetalMsmReductionStrategy,
    record_count: usize,
    run_count: usize,
) -> (usize, u32) {
    if matches!(
        reduction_strategy,
        MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2
            | MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
            | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
    ) && run_count != 0
        && record_count.div_ceil(run_count) >= JACOBIAN_LARGE_GEOMETRY_MIN_RECORDS_PER_RUN
    {
        (PARALLEL_JACOBIAN_LARGE_THREADS, PARALLEL_JACOBIAN_LARGE_TILE_RECORDS)
    } else if matches!(
        reduction_strategy,
        MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2
            | MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
            | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
    ) {
        (PARALLEL_JACOBIAN_SMALL_THREADS, PARALLEL_JACOBIAN_SMALL_TILE_RECORDS)
    } else {
        (PARALLEL_JACOBIAN_SMALL_THREADS, PARALLEL_BUCKET_TILE_RECORDS)
    }
}

fn batch_affine_product_tree_slots(pair_count: u32) -> Result<usize, MetalMsmError> {
    if pair_count == 0 {
        return Err(MetalMsmError::InvalidRoute("balanced affine level has no point pairs"));
    }
    let mut slots = pair_count as usize;
    let mut count = pair_count;
    while count > 1 {
        count = count.div_ceil(2);
        slots = slots
            .checked_add(count as usize)
            .ok_or(MetalMsmError::Capacity("affine product-tree slots"))?;
    }
    Ok(slots)
}

fn try_default_vec<T: Default + Clone>(
    len: usize,
    label: &'static str,
) -> Result<Vec<T>, MetalMsmError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| MetalMsmError::Capacity(label))?;
    values.resize(len, T::default());
    Ok(values)
}

#[allow(unsafe_code)]
fn expand_fixed_bases(
    bases: &[RawG1Affine],
    window_bits: FixedWindowBits,
) -> Result<Vec<GpuAffine>, MetalMsmError> {
    let windows = window_bits.windows_per_scalar() as usize;
    let expanded_count = bases
        .len()
        .checked_mul(windows)
        .ok_or(MetalMsmError::Capacity("fixed prepared-base expansion"))?;
    checked_u32(expanded_count, "fixed prepared-base expansion")?;

    let mut projective = Vec::new();
    projective
        .try_reserve_exact(bases.len())
        .map_err(|_| MetalMsmError::Capacity("fixed projective preparation"))?;
    for point in bases {
        let affine = blst::blst_p1_affine {
            x: blst::blst_fp { l: point.x.limbs },
            y: blst::blst_fp { l: point.y.limbs },
        };
        let mut value = blst::blst_p1::default();
        // SAFETY: `validate_prepared_bases` established a finite on-curve G1
        // affine value, and `value` is a distinct initialized output.
        unsafe { blst::blst_p1_from_affine(&mut value, &affine) };
        projective.push(value);
    }

    let mut pointers = Vec::new();
    pointers
        .try_reserve_exact(projective.len())
        .map_err(|_| MetalMsmError::Capacity("fixed batch-normalization pointers"))?;
    pointers.extend(projective.iter().map(core::ptr::from_ref));
    let mut affine_layer =
        try_default_vec::<blst::blst_p1_affine>(bases.len(), "fixed batch-normalization output")?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(expanded_count)
        .map_err(|_| MetalMsmError::Capacity("fixed prepared-base allocation"))?;

    for layer in 0..windows {
        // SAFETY: every pointer targets a live element of the fixed-capacity
        // `projective` vector, and `affine_layer` has one output slot per input.
        unsafe {
            blst::blst_p1s_to_affine(affine_layer.as_mut_ptr(), pointers.as_ptr(), projective.len())
        };
        expanded.extend(affine_layer.iter().map(|point| GpuAffine {
            x: GpuFp { limbs: point.x.l },
            y: GpuFp { limbs: point.y.l },
        }));

        if layer + 1 == windows {
            break;
        }
        for _ in 0..window_bits.get() {
            for point in &mut projective {
                let mut doubled = blst::blst_p1::default();
                // SAFETY: input and output are initialized and disjoint. G1 is
                // closed under doubling, so every next layer remains valid.
                unsafe { blst::blst_p1_double(&mut doubled, point) };
                *point = doubled;
            }
        }
    }
    Ok(expanded)
}

#[allow(unsafe_code)]
fn validate_prepared_bases(bases: &[RawG1Affine]) -> Result<(), MetalMsmError> {
    if bases.is_empty() {
        return Err(MetalMsmError::EmptyPreparedBases);
    }
    for (index, point) in bases.iter().copied().enumerate() {
        if greater_or_equal(&point.x.limbs, &FP_MODULUS) {
            return Err(MetalMsmError::InvalidPreparedBase {
                index,
                issue: PreparedBaseIssue::XCoordinateNotReduced,
            });
        }
        if greater_or_equal(&point.y.limbs, &FP_MODULUS) {
            return Err(MetalMsmError::InvalidPreparedBase {
                index,
                issue: PreparedBaseIssue::YCoordinateNotReduced,
            });
        }
        if point.x.limbs == [0; 6] && point.y.limbs == [0; 6] {
            return Err(MetalMsmError::InvalidPreparedBase {
                index,
                issue: PreparedBaseIssue::Identity,
            });
        }
        let affine = blst::blst_p1_affine {
            x: blst::blst_fp { l: point.x.limbs },
            y: blst::blst_fp { l: point.y.limbs },
        };
        // SAFETY: coordinates are fully initialized reduced BLST Montgomery
        // values. The first predicate validates curve membership before the
        // subgroup predicate is evaluated.
        if !unsafe { blst::blst_p1_affine_on_curve(&affine) } {
            return Err(MetalMsmError::InvalidPreparedBase {
                index,
                issue: PreparedBaseIssue::NotOnCurve,
            });
        }
        // SAFETY: `affine` is initialized and passed BLST's on-curve predicate.
        if !unsafe { blst::blst_p1_affine_in_g1(&affine) } {
            return Err(MetalMsmError::InvalidPreparedBase {
                index,
                issue: PreparedBaseIssue::NotInPrimeSubgroup,
            });
        }
    }
    Ok(())
}

fn greater_or_equal(lhs: &[u64; 6], rhs: &[u64; 6]) -> bool {
    for index in (0..lhs.len()).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn gpu_affine(point: RawG1Affine) -> GpuAffine {
    GpuAffine { x: GpuFp { limbs: point.x.limbs }, y: GpuFp { limbs: point.y.limbs } }
}

fn raw_jacobian(point: GpuJacobian) -> RawG1Jacobian {
    RawG1Jacobian {
        x: crate::msm_api::RawFpMontgomery { limbs: point.x.limbs },
        y: crate::msm_api::RawFpMontgomery { limbs: point.y.limbs },
        z: crate::msm_api::RawFpMontgomery { limbs: point.z.limbs },
    }
}

fn checked_u32(value: usize, label: &'static str) -> Result<u32, MetalMsmError> {
    u32::try_from(value).map_err(|_| MetalMsmError::Capacity(label))
}

fn initialization_error(error: SysError) -> MetalMsmError {
    MetalMsmError::Initialization(format!("{error}"))
}

fn execution_error(error: SysError) -> MetalMsmError {
    MetalMsmError::Execution(format!("{error}"))
}

#[allow(unsafe_code)]
fn wipe_values<T: Copy + Default>(values: &mut [T]) {
    for value in values {
        // SAFETY: each pointer is derived from a unique mutable slice element
        // and receives a fully initialized value of the same type.
        unsafe { core::ptr::write_volatile(value, T::default()) };
    }
    compiler_fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinate_reduction_is_strict() {
        let mut below = FP_MODULUS;
        below[0] -= 1;
        assert!(!greater_or_equal(&below, &FP_MODULUS));
        assert!(greater_or_equal(&FP_MODULUS, &FP_MODULUS));
        assert!(greater_or_equal(&[u64::MAX; 6], &FP_MODULUS));
    }

    #[test]
    fn translation_expands_negative_bit_and_dense_map() {
        let scalar = CanonicalScalar::from_le_limbs([255, 0, 0, 0]).unwrap();
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 1, output_index: 0 }];
        let mut route = route_jobs(1, &[scalar], &jobs, WindowBits::W8).unwrap();
        let execution =
            translate_route(1, &route, MetalMsmReductionStrategy::GlobalBatchAffineV1).unwrap();

        assert_eq!(execution.records.len(), 2);
        assert!(execution.records.iter().any(|record| record.flags == 1));
        assert!(execution.records.iter().all(|record| record.point_index == 0));
        assert_eq!(execution.dense_map.iter().filter(|entry| **entry != EMPTY_BUCKET).count(), 2);
        route.erase();
    }

    #[test]
    fn parallel_translation_omits_affine_topology() {
        let scalar = CanonicalScalar::from_le_limbs([255, 0, 0, 0]).unwrap();
        let jobs = [MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 1, output_index: 0 }];
        let mut route = route_jobs(1, &[scalar], &jobs, WindowBits::W8).unwrap();
        let execution =
            translate_route(1, &route, MetalMsmReductionStrategy::ParallelProjectiveV1).unwrap();

        assert!(execution.affine_records.is_empty());
        assert!(execution.affine_levels.is_empty());
        assert!(execution.affine_destination_map.is_empty());
        assert_eq!(execution.affine_max_pairs, 0);
        assert_eq!(execution.affine_max_tree_slots, 0);
        assert!(!execution.tiles.is_empty());
        route.erase();
    }

    #[test]
    fn gpu_translation_compacts_zero_scalars_without_reindexing_bases() {
        let scalars = [
            CanonicalScalar::ZERO,
            CanonicalScalar::from_le_limbs([1, 0, 0, 0]).unwrap(),
            CanonicalScalar::ZERO,
            CanonicalScalar::from_le_limbs([2, 0, 0, 0]).unwrap(),
        ];
        let jobs = [MsmJobV1 { base_offset: 5, scalar_offset: 0, term_count: 4, output_index: 0 }];
        let execution = translate_gpu_route(
            10,
            &scalars,
            &jobs,
            WindowBits::W6,
            MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8,
        )
        .unwrap();

        assert_eq!(execution.scalars.len(), 2);
        assert_eq!(execution.point_indices.as_slice(), &[6, 8]);
        assert_eq!(execution.jobs[0].scalar_offset, 0);
        assert_eq!(execution.jobs[0].term_count, 2);
        assert_eq!(execution.parameters.scalar_count, 2);
        assert_eq!(execution.parameters.max_record_count, 2 * 43);
        assert!(!execution.packing_eligible);
    }

    #[test]
    fn gpu_translation_preserves_all_zero_job_as_identity() {
        let scalars = [CanonicalScalar::ZERO; 3];
        let jobs = [MsmJobV1 { base_offset: 2, scalar_offset: 0, term_count: 3, output_index: 0 }];
        let execution = translate_gpu_route(
            8,
            &scalars,
            &jobs,
            WindowBits::W6,
            MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8,
        )
        .unwrap();

        assert!(execution.scalars.is_empty());
        assert!(execution.point_indices.is_empty());
        assert_eq!(execution.jobs[0].term_count, 0);
        assert_eq!(execution.parameters.scalar_count, 0);
        assert_eq!(execution.parameters.max_record_count, 0);
    }

    #[test]
    fn jacobian_geometry_tracks_routed_bucket_occupancy() {
        assert_eq!(
            parallel_geometry(MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2, 191, 1,),
            (PARALLEL_JACOBIAN_SMALL_THREADS, PARALLEL_JACOBIAN_SMALL_TILE_RECORDS)
        );
        assert_eq!(
            parallel_geometry(MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2, 192, 1,),
            (PARALLEL_JACOBIAN_LARGE_THREADS, PARALLEL_JACOBIAN_LARGE_TILE_RECORDS)
        );
        assert_eq!(
            parallel_geometry(MetalMsmReductionStrategy::ParallelProjectiveV1, usize::MAX, 1),
            (PARALLEL_JACOBIAN_SMALL_THREADS, PARALLEL_BUCKET_TILE_RECORDS)
        );
    }

    #[test]
    fn invalid_identity_is_rejected_before_metal() {
        let result = validate_prepared_bases(&[RawG1Affine::default()]);
        assert!(matches!(
            result,
            Err(MetalMsmError::InvalidPreparedBase {
                index: 0,
                issue: PreparedBaseIssue::Identity
            })
        ));
    }
}
