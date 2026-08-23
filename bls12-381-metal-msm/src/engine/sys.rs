// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Audited Objective-C and shared-memory boundary for the standalone MSM engine.

use std::ffi::c_void;
use std::fmt;
use std::format;
use std::mem::{size_of, size_of_val};
use std::ptr::NonNull;
use std::string::{String, ToString};
use std::sync::Mutex;
use std::vec;
use std::vec::Vec;

#[cfg(feature = "metal-diagnostics")]
use std::time::{Duration, Instant};

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
#[cfg(feature = "metal-diagnostics")]
use objc2_foundation::NSData;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
};

use crate::msm_api::RawFrMontgomery;
#[cfg(feature = "metal-diagnostics")]
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLStorageMode,
};

use super::MetalMsmReductionStrategy;
use super::abi::{
    BatchAffineGatherParameters, BatchAffineLevelParameters, BatchAffineProductParameters,
    BatchAffineStatus, BucketRun, BucketTileV1, CollectDescriptor, CollectParameters,
    FIXED_ROUTE_ABI_VERSION_V10, FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10,
    FIXED_ROUTE_LONG_BUCKET_RECORDS_V10, FIXED_ROUTE_LONG_THREADS_PER_GROUP_V10,
    FIXED_ROUTE_THREADS_PER_GROUP_V10, FIXED_ROUTE_TILE_RECORDS_V10, FixedRouteParametersV10,
    GPU_ROUTED_DENSE_ABI_VERSION_V4, GpuAffine, GpuFp, GpuJacobian, GpuRouteJobV3,
    GpuRouteParametersV3, GpuScalar, GpuXyzz, IntegrationParameters, KernelStatus,
    PARALLEL_BUCKET_TILE_RECORDS, PARALLEL_JACOBIAN_LARGE_THREADS,
    PARALLEL_JACOBIAN_LARGE_TILE_RECORDS, PARALLEL_JACOBIAN_SMALL_THREADS,
    PARALLEL_JACOBIAN_SMALL_TILE_RECORDS, ParallelBucketParametersV1, ResidentScalarCopyV1,
    SignedRecord, TileRunV1, WindowDescriptor, fixed_route_buckets_per_group_v10,
    fixed_route_integration_blocks_per_group_v10,
};
use super::batch_affine_plan::BatchAffineLevelPlan;

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bls12-381-metal-msm.metallib"));

const ABI_FUNCTION: &str = "bls12_381_metal_msm_msm_abi";
const AFFINE_GATHER_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_gather_signed_leaves";
const AFFINE_PREPARE_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_prepare_pairs";
const AFFINE_REDUCE_PRODUCTS_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_reduce_products";
const AFFINE_INVERT_ROOT_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_invert_product_root";
const AFFINE_EXPAND_INVERSES_FUNCTION: &str =
    "bls12_381_metal_msm_batch_affine_expand_product_inverses";
const AFFINE_FINISH_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_finish_pairs";
const AFFINE_COPY_CARRIES_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_copy_carries";
const AFFINE_EXPORT_ROOTS_FUNCTION: &str = "bls12_381_metal_msm_batch_affine_export_roots";
const PARALLEL_BUCKET_TILES_FUNCTION: &str = "bls12_381_metal_msm_bucket_tiles_parallel_v1";
const PARALLEL_BUCKET_PARTIALS_FUNCTION: &str =
    "bls12_381_metal_msm_bucket_tile_partials_parallel_v1";
const PARALLEL_JACOBIAN_SMALL_TILES_FUNCTION: &str =
    "bls12_381_metal_msm_bucket_tiles_jacobian_small_v2";
const PARALLEL_JACOBIAN_SMALL_PARTIALS_FUNCTION: &str =
    "bls12_381_metal_msm_bucket_partials_jacobian_small_v2";
const PARALLEL_JACOBIAN_LARGE_TILES_FUNCTION: &str =
    "bls12_381_metal_msm_bucket_tiles_jacobian_large_v2";
const PARALLEL_JACOBIAN_LARGE_PARTIALS_FUNCTION: &str =
    "bls12_381_metal_msm_bucket_partials_jacobian_large_v2";
const PARALLEL_JACOBIAN_INTEGRATE_FUNCTION: &str =
    "bls12_381_metal_msm_integrate_windows_jacobian_v2";
const PARALLEL_JACOBIAN_COLLECT_FUNCTION: &str = "bls12_381_metal_msm_collect_jacobian_v2";
const GPU_ROUTE_ABI_FUNCTION: &str = "bls12_381_metal_msm_gpu_route_abi_v4";
const GPU_ROUTE_HISTOGRAM_FUNCTION: &str = "bls12_381_metal_msm_gpu_histogram_v4";
const GPU_ROUTE_SCAN_FUNCTION: &str = "bls12_381_metal_msm_gpu_scan_buckets_v4";
const GPU_ROUTE_SCATTER_FUNCTION: &str = "bls12_381_metal_msm_gpu_scatter_v4";
const GPU_ROUTE_REDUCE_SMALL_FUNCTION: &str = "bls12_381_metal_msm_gpu_reduce_dense_small_v3";
const GPU_ROUTE_REDUCE_LARGE_FUNCTION: &str = "bls12_381_metal_msm_gpu_reduce_dense_large_v3";
const GPU_ROUTE_INTEGRATE_FUNCTION: &str = "bls12_381_metal_msm_gpu_integrate_dense_v3";
const GPU_ROUTE_COLLECT_FUNCTION: &str = "bls12_381_metal_msm_gpu_collect_dense_v3";
const XYZZ_ABI_FUNCTION: &str = "bls12_381_metal_msm_xyzz_abi_v4";
const GPU_ROUTE_XYZZ_REDUCE_SMALL_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_v4";
const GPU_ROUTE_XYZZ_REDUCE_LARGE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_v4";
const GPU_ROUTE_XYZZ_REDUCE_SMALL_PACK4_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_pack4_v4";
const GPU_ROUTE_XYZZ_REDUCE_LARGE_PACK4_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_pack4_v4";
const GPU_ROUTE_XYZZ_REDUCE_SMALL_PACK8_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_small_pack8_v4";
const GPU_ROUTE_XYZZ_REDUCE_LARGE_PACK8_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_dense_xyzz_large_pack8_v4";
const GPU_ROUTE_XYZZ_INTEGRATE_FUNCTION: &str = "bls12_381_metal_msm_gpu_integrate_dense_xyzz_v4";
const GPU_ROUTE_XYZZ_COLLECT_FUNCTION: &str = "bls12_381_metal_msm_gpu_collect_dense_xyzz_v4";
const GPU_ROUTE_ADAPTIVE_XYZZ_REDUCE_SMALL_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_adaptive_xyzz_small_v5";
const GPU_ROUTE_ADAPTIVE_XYZZ_REDUCE_LARGE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_adaptive_xyzz_large_v5";
const GPU_ROUTE_TILED_XYZZ_BUILD_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_build_long_tiles_xyzz_v6";
const GPU_ROUTE_TILED_XYZZ_FINALIZE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_finalize_long_tiles_xyzz_v6";
const GPU_ROUTE_TILED_XYZZ_REDUCE_SHORT_SMALL_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_small_v6";
const GPU_ROUTE_TILED_XYZZ_REDUCE_SHORT_LARGE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_large_v6";
const GPU_ROUTE_TILED_XYZZ_REDUCE_TILES_SMALL_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_long_tiles_xyzz_small_v6";
const GPU_ROUTE_TILED_XYZZ_REDUCE_TILES_LARGE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_long_tiles_xyzz_large_v6";
const GPU_ROUTE_TILED_XYZZ_COMBINE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_combine_long_tiles_xyzz_v6";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK4_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack4_v7";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK4_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack4_v7";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK8_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack8_v7";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK8_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack8_v7";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK16_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_small_pack16_v7";
const GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK16_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_large_pack16_v7";
const GPU_ROUTE_FINE_TILED_XYZZ_BUILD_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_build_fine_tiles_xyzz_v8";
const GPU_ROUTE_FINE_TILED_XYZZ_REDUCE_SHORT_SMALL_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_small_v8";
const GPU_ROUTE_FINE_TILED_XYZZ_REDUCE_SHORT_LARGE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_reduce_short_xyzz_large_v8";
const GPU_ROUTE_BATCH_AFFINE_XYZZ_BUILD_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_build_batch_affine_tiles_v9";
const GPU_ROUTE_BATCH_AFFINE_XYZZ_INITIALIZE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_initialize_batch_affine_products_v9";
const GPU_ROUTE_BATCH_AFFINE_XYZZ_PREPARE_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_prepare_batch_affine_tiles_v9";
const GPU_ROUTE_BATCH_AFFINE_XYZZ_FINISH_FUNCTION: &str =
    "bls12_381_metal_msm_gpu_finish_batch_affine_tiles_v9";
const FIXED_ROUTE_ABI_FUNCTION: &str = "bls12_381_metal_msm_fixed_route_abi_v10";
const FIXED_ROUTE_HISTOGRAM_FUNCTION: &str = "bls12_381_metal_msm_fixed_histogram_v10";
const FIXED_ROUTE_SCAN_FUNCTION: &str = "bls12_381_metal_msm_fixed_scan_buckets_v10";
const FIXED_ROUTE_BUILD_TILES_FUNCTION: &str = "bls12_381_metal_msm_fixed_build_long_tiles_v10";
const FIXED_ROUTE_FINALIZE_TILES_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_finalize_long_tiles_v10";
const FIXED_ROUTE_SCATTER_FUNCTION: &str = "bls12_381_metal_msm_fixed_scatter_v10";
const FIXED_ROUTE_REDUCE_FUNCTION: &str = "bls12_381_metal_msm_fixed_reduce_xyzz_v10";
const FIXED_ROUTE_REDUCE_TILES_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_reduce_long_tiles_xyzz_v10";
const FIXED_ROUTE_COMBINE_TILES_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_combine_long_tiles_xyzz_v10";
const FIXED_ROUTE_INTEGRATE_BLOCKS_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_integrate_blocks_v10";
const FIXED_ROUTE_FINISH_INTEGRATION_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_finish_integration_v10";
const RESIDENT_SCALAR_ABI_FUNCTION: &str = "bls12_381_metal_msm_resident_scalar_abi_v1";
const RESIDENT_SCALAR_CANONICALIZE_FUNCTION: &str =
    "bls12_381_metal_msm_resident_montgomery_to_canonical_v1";
const FIXED_ROUTE_RESIDENT_SCATTER_FUNCTION: &str =
    "bls12_381_metal_msm_fixed_scatter_resident_v10";
const INTEGRATE_FUNCTION: &str = "bls12_381_metal_msm_integrate_windows_serial";
const COLLECT_FUNCTION: &str = "bls12_381_metal_msm_collect_serial";
const PARALLEL_BUCKET_ABI_VERSION_V1: u32 = 1;
const PARALLEL_JACOBIAN_ABI_VERSION_V2: u32 = 2;
const PARALLEL_BUCKET_THREADS: usize = 32;
const GPU_ROUTE_TILED_XYZZ_CAPACITY_DIVISOR: usize = 2_048;
const GPU_ROUTE_FINE_TILED_XYZZ_CAPACITY_DIVISOR: usize = 1_024;
const GPU_ROUTE_BATCH_AFFINE_XYZZ_TILE_RECORDS: usize = 128;
const MAX_CLEARED_SCRATCH_BUFFERS: usize = 128;
const MAX_CLEARED_SCRATCH_BYTES: usize = 512 * 1024 * 1024;

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
#[cfg(feature = "metal-diagnostics")]
type CounterSampleBuffer = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

#[cfg(feature = "metal-diagnostics")]
const FIXED_ROUTE_PHASES_V10: [&str; 11] = [
    "canonicalize",
    "histogram",
    "scan",
    "build_tiles",
    "finalize_tiles",
    "scatter",
    "reduce_short",
    "reduce_long",
    "combine_long",
    "integrate_blocks",
    "finish_integration",
];

/// Private shader plan after explicit portable-to-device ABI translation.
pub(super) struct ExecutionPlan<'a> {
    pub(super) reduction_strategy: MetalMsmReductionStrategy,
    pub(super) parallel_threads: usize,
    pub(super) parallel_tile_records: u32,
    pub(super) records: &'a [SignedRecord],
    pub(super) affine_records: &'a [SignedRecord],
    pub(super) affine_levels: &'a [BatchAffineLevelPlan],
    pub(super) affine_destination_map: &'a [u32],
    pub(super) affine_max_pairs: usize,
    pub(super) affine_max_tree_slots: usize,
    pub(super) runs: &'a [BucketRun],
    pub(super) tiles: &'a [BucketTileV1],
    pub(super) tile_runs: &'a [TileRunV1],
    pub(super) dense_map: &'a [u32],
    pub(super) windows: &'a [WindowDescriptor],
    pub(super) window_point_count: usize,
    pub(super) collects: &'a [CollectDescriptor],
    pub(super) output_count: usize,
}

/// Private Metal-side routing plan after independent host validation.
pub(super) struct GpuRoutedExecutionPlan<'a> {
    pub(super) reduction_strategy: MetalMsmReductionStrategy,
    pub(super) scalars: &'a [GpuScalar],
    pub(super) point_indices: &'a [u32],
    pub(super) jobs: &'a [GpuRouteJobV3],
    pub(super) parameters: GpuRouteParametersV3,
    pub(super) packing_eligible: bool,
    pub(super) output_count: usize,
    pub(super) window_point_count: usize,
}

/// Private pre-shifted fixed-base plan after independent host validation.
pub(super) struct FixedGpuRoutedExecutionPlan<'a> {
    pub(super) scalar_input: FixedScalarInput<'a>,
    pub(super) jobs: &'a [GpuRouteJobV3],
    pub(super) parameters: FixedRouteParametersV10,
    pub(super) output_count: usize,
}

/// Host-uploaded canonical scalars or synchronized external Montgomery views.
pub(super) enum FixedScalarInput<'a> {
    Host { scalars: &'a [GpuScalar], point_indices: &'a [u32] },
    HostMontgomery(&'a [RawFrMontgomery]),
    Resident(&'a [ResidentScalarSource<'a>]),
}

/// One external source and its contiguous destination in canonical scratch.
#[derive(Clone, Copy)]
pub(super) struct ResidentScalarSource<'a> {
    pub(super) buffer: &'a ResidentScalarBuffer,
    pub(super) destination_offset: u32,
}

/// Opaque retained Metal buffer range transferred to the engine worker.
#[derive(Clone)]
pub(super) struct ResidentScalarBuffer {
    raw: Buffer,
    byte_offset: usize,
    len: usize,
    registry_id: u64,
}

// SAFETY: MTLBuffer resources are designed for cross-thread command encoding.
// This wrapper exposes no CPU pointer or mutation, carries only an immutable
// range, and is consumed behind the engine's serialized execution boundary.
unsafe impl Send for ResidentScalarBuffer {}

impl ResidentScalarBuffer {
    pub(super) fn new(raw: Buffer, byte_offset: usize, len: usize) -> Result<Self, SysError> {
        if !byte_offset.is_multiple_of(size_of::<GpuScalar>()) {
            return Err(SysError::InvalidPlan(
                "resident scalar byte offset is not 32-byte aligned",
            ));
        }
        let byte_len = len
            .checked_mul(size_of::<GpuScalar>())
            .ok_or(SysError::LengthOverflow("resident scalar byte length"))?;
        let end = byte_offset
            .checked_add(byte_len)
            .ok_or(SysError::LengthOverflow("resident scalar byte range"))?;
        if end > raw.length() {
            return Err(SysError::InvalidPlan("resident scalar range exceeds its Metal buffer"));
        }
        let registry_id = raw.device().registryID();
        Ok(Self { raw, byte_offset, len, registry_id })
    }

    pub(super) const fn len(&self) -> usize {
        self.len
    }

    pub(super) const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub(super) const fn registry_id(&self) -> u64 {
        self.registry_id
    }
}

/// Retained, immutable prepared-base allocation.
pub(super) struct PreparedBaseBuffer {
    raw: Buffer,
    logical_len: usize,
}

impl PreparedBaseBuffer {
    pub(super) const fn len(&self) -> usize {
        self.logical_len
    }
}

#[derive(Debug)]
pub(super) enum SysError {
    DeviceUnavailable,
    QueueUnavailable,
    Library(String),
    MissingFunction(&'static str),
    Pipeline { name: &'static str, reason: String },
    Allocation { label: &'static str, bytes: usize },
    LengthOverflow(&'static str),
    Abi(String),
    CommandBuffer,
    Encoder(&'static str),
    Execution(String),
    InvalidPlan(&'static str),
    BatchAffineStatus(BatchAffineStatus),
    KernelStatus(KernelStatus),
    Erasure(String),
    Poisoned(String),
}

impl fmt::Display for SysError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnavailable => formatter.write_str("no Apple Metal device is available"),
            Self::QueueUnavailable => formatter.write_str("Metal command queue creation failed"),
            Self::Library(reason) => {
                write!(formatter, "embedded Metal library load failed: {reason}")
            }
            Self::MissingFunction(name) => {
                write!(formatter, "embedded Metal function {name} is missing")
            }
            Self::Pipeline { name, reason } => {
                write!(formatter, "Metal pipeline {name} creation failed: {reason}")
            }
            Self::Allocation { label, bytes } => {
                write!(formatter, "Metal allocation {label} ({bytes} bytes) failed")
            }
            Self::LengthOverflow(label) => write!(formatter, "{label} exceeds the Metal MSM ABI"),
            Self::Abi(reason) => write!(formatter, "Metal MSM ABI mismatch: {reason}"),
            Self::CommandBuffer => formatter.write_str("Metal command buffer creation failed"),
            Self::Encoder(stage) => {
                write!(formatter, "Metal {stage} command encoder creation failed")
            }
            Self::Execution(reason) => write!(formatter, "Metal MSM command failed: {reason}"),
            Self::InvalidPlan(reason) => write!(formatter, "invalid Metal MSM plan: {reason}"),
            Self::BatchAffineStatus(status) => write!(
                formatter,
                "Metal MSM batch-affine reducer rejected its input (flags={:#x}, record_errors={}, destination_errors={}, reserved={:#x})",
                status.flags, status.record_errors, status.destination_errors, status.reserved
            ),
            Self::KernelStatus(status) => write!(
                formatter,
                "Metal MSM kernel rejected its input (flags={:#x}, record_errors={}, descriptor_errors={}, reserved={})",
                status.flags, status.record_errors, status.descriptor_errors, status.reserved
            ),
            Self::Erasure(reason) => {
                write!(formatter, "Metal MSM scratch erasure failed: {reason}")
            }
            Self::Poisoned(reason) => write!(formatter, "Metal MSM engine is poisoned: {reason}"),
        }
    }
}

impl std::error::Error for SysError {}

struct Pipelines {
    affine_gather: Pipeline,
    affine_prepare: Pipeline,
    affine_reduce_products: Pipeline,
    affine_invert_root: Pipeline,
    affine_expand_inverses: Pipeline,
    affine_finish: Pipeline,
    affine_copy_carries: Pipeline,
    affine_export_roots: Pipeline,
    parallel_bucket_tiles: Pipeline,
    parallel_bucket_partials: Pipeline,
    parallel_jacobian_small_tiles: Pipeline,
    parallel_jacobian_small_partials: Pipeline,
    parallel_jacobian_large_tiles: Pipeline,
    parallel_jacobian_large_partials: Pipeline,
    parallel_jacobian_integrate: Pipeline,
    parallel_jacobian_collect: Pipeline,
    gpu_route_histogram: Pipeline,
    gpu_route_scan: Pipeline,
    gpu_route_scatter: Pipeline,
    gpu_route_reduce_small: Pipeline,
    gpu_route_reduce_large: Pipeline,
    gpu_route_integrate: Pipeline,
    gpu_route_collect: Pipeline,
    gpu_route_xyzz_reduce_small: Pipeline,
    gpu_route_xyzz_reduce_large: Pipeline,
    gpu_route_xyzz_reduce_small_pack4: Pipeline,
    gpu_route_xyzz_reduce_large_pack4: Pipeline,
    gpu_route_xyzz_reduce_small_pack8: Pipeline,
    gpu_route_xyzz_reduce_large_pack8: Pipeline,
    gpu_route_adaptive_xyzz_reduce_small: Pipeline,
    gpu_route_adaptive_xyzz_reduce_large: Pipeline,
    gpu_route_tiled_xyzz_build: Pipeline,
    gpu_route_tiled_xyzz_finalize: Pipeline,
    gpu_route_tiled_xyzz_reduce_short_small: Pipeline,
    gpu_route_tiled_xyzz_reduce_short_large: Pipeline,
    gpu_route_tiled_xyzz_reduce_tiles_small: Pipeline,
    gpu_route_tiled_xyzz_reduce_tiles_large: Pipeline,
    gpu_route_tiled_xyzz_combine: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_small_pack4: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_large_pack4: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_small_pack8: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_large_pack8: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_small_pack16: Pipeline,
    gpu_route_packed_tiled_xyzz_reduce_large_pack16: Pipeline,
    gpu_route_fine_tiled_xyzz_build: Pipeline,
    gpu_route_fine_tiled_xyzz_reduce_short_small: Pipeline,
    gpu_route_fine_tiled_xyzz_reduce_short_large: Pipeline,
    gpu_route_batch_affine_xyzz_build: Pipeline,
    gpu_route_batch_affine_xyzz_initialize: Pipeline,
    gpu_route_batch_affine_xyzz_prepare: Pipeline,
    gpu_route_batch_affine_xyzz_finish: Pipeline,
    fixed_route_histogram: Pipeline,
    fixed_route_scan: Pipeline,
    fixed_route_build_tiles: Pipeline,
    fixed_route_finalize_tiles: Pipeline,
    fixed_route_scatter: Pipeline,
    fixed_route_resident_scatter: Pipeline,
    resident_scalar_canonicalize: Pipeline,
    fixed_route_reduce: Pipeline,
    fixed_route_reduce_tiles: Pipeline,
    fixed_route_combine_tiles: Pipeline,
    fixed_route_integrate_blocks: Pipeline,
    fixed_route_finish_integration: Pipeline,
    gpu_route_xyzz_integrate: Pipeline,
    gpu_route_xyzz_collect: Pipeline,
    integrate: Pipeline,
    collect: Pipeline,
}

struct GpuRoutedBuffers<'a> {
    scalars: &'a ProtocolObject<dyn MTLBuffer>,
    point_indices: &'a ProtocolObject<dyn MTLBuffer>,
    jobs: &'a ProtocolObject<dyn MTLBuffer>,
    histogram: &'a ProtocolObject<dyn MTLBuffer>,
    offsets: &'a ProtocolObject<dyn MTLBuffer>,
    cursors: &'a ProtocolObject<dyn MTLBuffer>,
    records: &'a ProtocolObject<dyn MTLBuffer>,
    bucket_roots: &'a ProtocolObject<dyn MTLBuffer>,
    window_points: &'a ProtocolObject<dyn MTLBuffer>,
    outputs: &'a ProtocolObject<dyn MTLBuffer>,
    status: &'a ProtocolObject<dyn MTLBuffer>,
    tiled: Option<GpuRoutedTiledBuffers<'a>>,
    batch_affine: Option<GpuRoutedBatchAffineBuffers<'a>>,
}

#[derive(Clone, Copy)]
struct GpuRoutedTiledBuffers<'a> {
    tile_offsets: &'a ProtocolObject<dyn MTLBuffer>,
    tile_counts: &'a ProtocolObject<dyn MTLBuffer>,
    tiles: &'a ProtocolObject<dyn MTLBuffer>,
    tile_roots: &'a ProtocolObject<dyn MTLBuffer>,
    indirect_groups: &'a ProtocolObject<dyn MTLBuffer>,
    tile_capacity: u32,
}

#[derive(Clone, Copy)]
struct FixedGpuRoutedTiledBuffers<'a> {
    tile_offsets: &'a ProtocolObject<dyn MTLBuffer>,
    tile_counts: &'a ProtocolObject<dyn MTLBuffer>,
    tiles: &'a ProtocolObject<dyn MTLBuffer>,
    tile_roots: &'a ProtocolObject<dyn MTLBuffer>,
    indirect_groups: &'a ProtocolObject<dyn MTLBuffer>,
    tile_capacity: u32,
}

#[derive(Clone, Copy)]
struct GpuRoutedBatchAffineBuffers<'a> {
    product_tree: &'a ProtocolObject<dyn MTLBuffer>,
    status: &'a ProtocolObject<dyn MTLBuffer>,
}

struct ParallelProjectiveBuffers<'a> {
    records: &'a ProtocolObject<dyn MTLBuffer>,
    tiles: &'a ProtocolObject<dyn MTLBuffer>,
    tile_runs: &'a ProtocolObject<dyn MTLBuffer>,
    tile_roots: &'a ProtocolObject<dyn MTLBuffer>,
    run_roots: &'a ProtocolObject<dyn MTLBuffer>,
    dense_map: &'a ProtocolObject<dyn MTLBuffer>,
    windows: &'a ProtocolObject<dyn MTLBuffer>,
    window_points: &'a ProtocolObject<dyn MTLBuffer>,
    collects: &'a ProtocolObject<dyn MTLBuffer>,
    outputs: &'a ProtocolObject<dyn MTLBuffer>,
    status: &'a ProtocolObject<dyn MTLBuffer>,
}

#[derive(Default)]
struct Lifecycle {
    poisoned: Option<String>,
    quarantine: Option<Vec<Buffer>>,
    cleared_pool: Vec<Buffer>,
    cleared_pool_bytes: usize,
}

impl Drop for Lifecycle {
    fn drop(&mut self) {
        if let Some(buffers) = self.quarantine.take() {
            // An unconfirmed clear must not return the allocation to a driver
            // pool merely because the poisoned engine was dropped and rebuilt.
            // This catastrophic path intentionally retains it until process exit.
            core::mem::forget(buffers);
        }
    }
}

/// Owns every request allocation from the moment it is created.
///
/// The guard stays armed until a confirmed clear command completes. This is
/// deliberately independent of the compute result: allocation failures and
/// command-encoding failures must not turn into secret-dependent early exits.
struct ScratchGuard<'a> {
    context: &'a MetalContext,
    buffers: Vec<ScratchBuffer>,
    armed: bool,
}

struct ScratchBuffer {
    raw: Buffer,
    used_bytes: usize,
}

impl<'a> ScratchGuard<'a> {
    fn new(context: &'a MetalContext, capacity: usize) -> Result<Self, SysError> {
        let mut buffers = Vec::new();
        buffers.try_reserve_exact(capacity).map_err(|_| SysError::Allocation {
            label: "request scratch handle list",
            bytes: capacity.saturating_mul(size_of::<Buffer>()),
        })?;
        Ok(Self { context, buffers, armed: true })
    }

    fn retain(&mut self, buffer: Buffer) -> Buffer {
        let used_bytes = buffer.length();
        self.buffers.push(ScratchBuffer { raw: buffer.clone(), used_bytes });
        buffer
    }

    fn retain_used(&mut self, buffer: Buffer, used_bytes: usize) -> Buffer {
        debug_assert!(used_bytes <= buffer.length());
        self.buffers.push(ScratchBuffer { raw: buffer.clone(), used_bytes });
        buffer
    }

    fn take_raw_buffers(&mut self) -> Vec<Buffer> {
        core::mem::take(&mut self.buffers).into_iter().map(|buffer| buffer.raw).collect()
    }

    fn upload_slice<T: Copy>(
        &mut self,
        values: &[T],
        label: &'static str,
    ) -> Result<Buffer, SysError> {
        let bytes = size_of_val(values);
        if bytes == 0 {
            return Err(SysError::InvalidPlan("a request allocation must not be empty"));
        }
        let raw = match self.context.take_cleared_scratch(bytes, label)? {
            Some(raw) => raw,
            None => new_shared_buffer(&self.context.device, bytes, label)?,
        };
        copy_values(&raw, values);
        Ok(self.retain_used(raw, bytes))
    }

    fn upload_slice_or_word<T: Copy>(
        &mut self,
        values: &[T],
        label: &'static str,
    ) -> Result<Buffer, SysError> {
        if values.is_empty() {
            return self.zeroed::<u32>(1, label);
        }
        self.upload_slice(values, label)
    }

    fn zeroed<T>(&mut self, len: usize, label: &'static str) -> Result<Buffer, SysError> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or(SysError::LengthOverflow(label))?;
        if bytes == 0 {
            return Err(SysError::InvalidPlan("zero-length device output"));
        }
        let raw = match self.context.take_cleared_scratch(bytes, label)? {
            Some(raw) => raw,
            None => zeroed_buffer(&self.context.device, bytes, label)?,
        };
        Ok(self.retain_used(raw, bytes))
    }

    fn finish(mut self) -> Result<(), SysError> {
        match self.context.erase_scratch(&self.buffers) {
            Ok(()) => {
                let buffers = self.take_raw_buffers();
                self.context.recycle_cleared_scratch(buffers);
                self.armed = false;
                Ok(())
            }
            Err(error) => {
                let reason = error.to_string();
                let buffers = self.take_raw_buffers();
                self.context.poison_and_quarantine(reason.clone(), buffers);
                self.armed = false;
                Err(SysError::Erasure(reason))
            }
        }
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        if !self.armed || self.buffers.is_empty() {
            return;
        }
        match self.context.erase_scratch(&self.buffers) {
            Ok(()) => {
                let buffers = self.take_raw_buffers();
                self.context.recycle_cleared_scratch(buffers);
            }
            Err(error) => {
                let reason = error.to_string();
                let buffers = self.take_raw_buffers();
                self.context.poison_and_quarantine(reason, buffers);
            }
        }
        self.armed = false;
    }
}

/// Long-lived device, queue, library pipelines, and serialized secret lifecycle.
pub(super) struct MetalContext {
    device: Device,
    queue: Queue,
    pipelines: Pipelines,
    execution: Mutex<()>,
    lifecycle: Mutex<Lifecycle>,
}

impl MetalContext {
    pub(super) fn new() -> Result<Self, SysError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(SysError::DeviceUnavailable)?;
        let queue = device.newCommandQueue().ok_or(SysError::QueueUnavailable)?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.compute")));
        let library = device
            .newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB))
            .map_err(|error| SysError::Library(format!("{error:?}")))?;
        let abi = load_pipeline(&device, &library, ABI_FUNCTION)?;
        let gpu_route_abi = load_pipeline(&device, &library, GPU_ROUTE_ABI_FUNCTION)?;
        let fixed_route_abi = load_pipeline(&device, &library, FIXED_ROUTE_ABI_FUNCTION)?;
        let resident_scalar_abi = load_pipeline(&device, &library, RESIDENT_SCALAR_ABI_FUNCTION)?;
        let xyzz_abi = load_pipeline(&device, &library, XYZZ_ABI_FUNCTION)?;
        let pipelines = Pipelines {
            affine_gather: load_pipeline(&device, &library, AFFINE_GATHER_FUNCTION)?,
            affine_prepare: load_pipeline(&device, &library, AFFINE_PREPARE_FUNCTION)?,
            affine_reduce_products: load_pipeline(
                &device,
                &library,
                AFFINE_REDUCE_PRODUCTS_FUNCTION,
            )?,
            affine_invert_root: load_pipeline(&device, &library, AFFINE_INVERT_ROOT_FUNCTION)?,
            affine_expand_inverses: load_pipeline(
                &device,
                &library,
                AFFINE_EXPAND_INVERSES_FUNCTION,
            )?,
            affine_finish: load_pipeline(&device, &library, AFFINE_FINISH_FUNCTION)?,
            affine_copy_carries: load_pipeline(&device, &library, AFFINE_COPY_CARRIES_FUNCTION)?,
            affine_export_roots: load_pipeline(&device, &library, AFFINE_EXPORT_ROOTS_FUNCTION)?,
            parallel_bucket_tiles: load_pipeline(
                &device,
                &library,
                PARALLEL_BUCKET_TILES_FUNCTION,
            )?,
            parallel_bucket_partials: load_pipeline(
                &device,
                &library,
                PARALLEL_BUCKET_PARTIALS_FUNCTION,
            )?,
            parallel_jacobian_small_tiles: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_SMALL_TILES_FUNCTION,
            )?,
            parallel_jacobian_small_partials: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_SMALL_PARTIALS_FUNCTION,
            )?,
            parallel_jacobian_large_tiles: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_LARGE_TILES_FUNCTION,
            )?,
            parallel_jacobian_large_partials: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_LARGE_PARTIALS_FUNCTION,
            )?,
            parallel_jacobian_integrate: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_INTEGRATE_FUNCTION,
            )?,
            parallel_jacobian_collect: load_pipeline(
                &device,
                &library,
                PARALLEL_JACOBIAN_COLLECT_FUNCTION,
            )?,
            gpu_route_histogram: load_pipeline(&device, &library, GPU_ROUTE_HISTOGRAM_FUNCTION)?,
            gpu_route_scan: load_pipeline(&device, &library, GPU_ROUTE_SCAN_FUNCTION)?,
            gpu_route_scatter: load_pipeline(&device, &library, GPU_ROUTE_SCATTER_FUNCTION)?,
            gpu_route_reduce_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_REDUCE_SMALL_FUNCTION,
            )?,
            gpu_route_reduce_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_REDUCE_LARGE_FUNCTION,
            )?,
            gpu_route_integrate: load_pipeline(&device, &library, GPU_ROUTE_INTEGRATE_FUNCTION)?,
            gpu_route_collect: load_pipeline(&device, &library, GPU_ROUTE_COLLECT_FUNCTION)?,
            gpu_route_xyzz_reduce_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_SMALL_FUNCTION,
            )?,
            gpu_route_xyzz_reduce_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_LARGE_FUNCTION,
            )?,
            gpu_route_xyzz_reduce_small_pack4: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_SMALL_PACK4_FUNCTION,
            )?,
            gpu_route_xyzz_reduce_large_pack4: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_LARGE_PACK4_FUNCTION,
            )?,
            gpu_route_xyzz_reduce_small_pack8: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_SMALL_PACK8_FUNCTION,
            )?,
            gpu_route_xyzz_reduce_large_pack8: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_REDUCE_LARGE_PACK8_FUNCTION,
            )?,
            gpu_route_adaptive_xyzz_reduce_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_ADAPTIVE_XYZZ_REDUCE_SMALL_FUNCTION,
            )?,
            gpu_route_adaptive_xyzz_reduce_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_ADAPTIVE_XYZZ_REDUCE_LARGE_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_build: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_BUILD_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_finalize: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_FINALIZE_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_reduce_short_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_REDUCE_SHORT_SMALL_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_reduce_short_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_REDUCE_SHORT_LARGE_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_reduce_tiles_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_REDUCE_TILES_SMALL_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_reduce_tiles_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_REDUCE_TILES_LARGE_FUNCTION,
            )?,
            gpu_route_tiled_xyzz_combine: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_TILED_XYZZ_COMBINE_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_small_pack4: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK4_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_large_pack4: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK4_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_small_pack8: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK8_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_large_pack8: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK8_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_small_pack16: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_SMALL_PACK16_FUNCTION,
            )?,
            gpu_route_packed_tiled_xyzz_reduce_large_pack16: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_PACKED_TILED_XYZZ_REDUCE_LARGE_PACK16_FUNCTION,
            )?,
            gpu_route_fine_tiled_xyzz_build: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_FINE_TILED_XYZZ_BUILD_FUNCTION,
            )?,
            gpu_route_fine_tiled_xyzz_reduce_short_small: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_FINE_TILED_XYZZ_REDUCE_SHORT_SMALL_FUNCTION,
            )?,
            gpu_route_fine_tiled_xyzz_reduce_short_large: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_FINE_TILED_XYZZ_REDUCE_SHORT_LARGE_FUNCTION,
            )?,
            gpu_route_batch_affine_xyzz_build: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_BATCH_AFFINE_XYZZ_BUILD_FUNCTION,
            )?,
            gpu_route_batch_affine_xyzz_initialize: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_BATCH_AFFINE_XYZZ_INITIALIZE_FUNCTION,
            )?,
            gpu_route_batch_affine_xyzz_prepare: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_BATCH_AFFINE_XYZZ_PREPARE_FUNCTION,
            )?,
            gpu_route_batch_affine_xyzz_finish: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_BATCH_AFFINE_XYZZ_FINISH_FUNCTION,
            )?,
            fixed_route_histogram: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_HISTOGRAM_FUNCTION,
            )?,
            fixed_route_scan: load_pipeline(&device, &library, FIXED_ROUTE_SCAN_FUNCTION)?,
            fixed_route_build_tiles: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_BUILD_TILES_FUNCTION,
            )?,
            fixed_route_finalize_tiles: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_FINALIZE_TILES_FUNCTION,
            )?,
            fixed_route_scatter: load_pipeline(&device, &library, FIXED_ROUTE_SCATTER_FUNCTION)?,
            fixed_route_resident_scatter: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_RESIDENT_SCATTER_FUNCTION,
            )?,
            resident_scalar_canonicalize: load_pipeline(
                &device,
                &library,
                RESIDENT_SCALAR_CANONICALIZE_FUNCTION,
            )?,
            fixed_route_reduce: load_pipeline(&device, &library, FIXED_ROUTE_REDUCE_FUNCTION)?,
            fixed_route_reduce_tiles: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_REDUCE_TILES_FUNCTION,
            )?,
            fixed_route_combine_tiles: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_COMBINE_TILES_FUNCTION,
            )?,
            fixed_route_integrate_blocks: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_INTEGRATE_BLOCKS_FUNCTION,
            )?,
            fixed_route_finish_integration: load_pipeline(
                &device,
                &library,
                FIXED_ROUTE_FINISH_INTEGRATION_FUNCTION,
            )?,
            gpu_route_xyzz_integrate: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_INTEGRATE_FUNCTION,
            )?,
            gpu_route_xyzz_collect: load_pipeline(
                &device,
                &library,
                GPU_ROUTE_XYZZ_COLLECT_FUNCTION,
            )?,
            integrate: load_pipeline(&device, &library, INTEGRATE_FUNCTION)?,
            collect: load_pipeline(&device, &library, COLLECT_FUNCTION)?,
        };
        validate_abi(&device, &queue, &abi)?;
        validate_gpu_route_abi(&device, &queue, &gpu_route_abi)?;
        validate_fixed_route_abi(&device, &queue, &fixed_route_abi)?;
        validate_resident_scalar_abi(&device, &queue, &resident_scalar_abi)?;
        validate_xyzz_abi(&device, &queue, &xyzz_abi)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            execution: Mutex::new(()),
            lifecycle: Mutex::new(Lifecycle::default()),
        })
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn upload_bases(&self, bases: &[GpuAffine]) -> Result<PreparedBaseBuffer, SysError> {
        if bases.is_empty() {
            return Err(SysError::InvalidPlan("prepared bases must not be empty"));
        }
        u32::try_from(bases.len()).map_err(|_| SysError::LengthOverflow("prepared base count"))?;
        let raw = buffer_from_slice(&self.device, bases, "prepared-bases")?;
        Ok(PreparedBaseBuffer { raw, logical_len: bases.len() })
    }

    pub(super) fn execute(
        &self,
        bases: &PreparedBaseBuffer,
        plan: ExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| SysError::Poisoned("host execution mutex was poisoned".into()))?;
        self.ensure_healthy()?;
        validate_execution_plan(bases.len(), &plan)?;

        match plan.reduction_strategy {
            MetalMsmReductionStrategy::GlobalBatchAffineV1 => {
                self.execute_batch_affine(bases, plan)
            }
            MetalMsmReductionStrategy::ParallelProjectiveV1
            | MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2 => {
                self.execute_parallel_projective(bases, plan)
            }
            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
            | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => {
                Err(SysError::InvalidPlan("GPU-routed strategy reached the CPU-routed executor"))
            }
        }
    }

    pub(super) fn execute_gpu_routed(
        &self,
        bases: &PreparedBaseBuffer,
        plan: GpuRoutedExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| SysError::Poisoned("host execution mutex was poisoned".into()))?;
        self.ensure_healthy()?;
        validate_gpu_routed_plan(bases.len(), &plan)?;
        self.execute_gpu_routed_inner(bases, plan)
    }

    pub(super) fn execute_fixed_gpu_routed(
        &self,
        shifted_bases: &PreparedBaseBuffer,
        plan: FixedGpuRoutedExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| SysError::Poisoned("host execution mutex was poisoned".into()))?;
        self.ensure_healthy()?;
        validate_fixed_gpu_routed_plan(self.registry_id(), shifted_bases.len(), &plan)?;

        #[cfg(feature = "metal-diagnostics")]
        let request_started = Instant::now();
        #[cfg(feature = "metal-diagnostics")]
        let allocation_started = Instant::now();
        let parameters = plan.parameters;
        let externally_resident = matches!(plan.scalar_input, FixedScalarInput::Resident(_));
        let mut scratch = ScratchGuard::new(self, if externally_resident { 16 } else { 17 })?;
        let (scalars, point_indices, montgomery_input) = match &plan.scalar_input {
            FixedScalarInput::Host { scalars, point_indices } => (
                scratch.upload_slice_or_word(scalars, "request.fixed-scalars")?,
                Some(scratch.upload_slice_or_word(point_indices, "request.fixed-point-indices")?),
                None,
            ),
            FixedScalarInput::HostMontgomery(values) => (
                scratch.zeroed::<GpuScalar>(
                    (parameters.scalar_count as usize).max(1),
                    "request.fixed-host-montgomery-canonical-scalars",
                )?,
                None,
                Some(
                    scratch
                        .upload_slice_or_word(values, "request.fixed-host-montgomery-scalars")?,
                ),
            ),
            FixedScalarInput::Resident(_) => (
                scratch.zeroed::<GpuScalar>(
                    (parameters.scalar_count as usize).max(1),
                    "request.fixed-resident-canonical-scalars",
                )?,
                None,
                None,
            ),
        };
        let jobs = scratch.upload_slice(plan.jobs, "request.fixed-jobs")?;
        let histogram = scratch
            .zeroed::<u32>(parameters.dense_bucket_count as usize, "request.fixed-histogram")?;
        let offsets = scratch
            .zeroed::<u32>(parameters.dense_bucket_count as usize, "request.fixed-offsets")?;
        let cursors = scratch
            .zeroed::<u32>(parameters.dense_bucket_count as usize, "request.fixed-cursors")?;
        let records = scratch.zeroed::<SignedRecord>(
            (parameters.max_record_count as usize).max(1),
            "request.fixed-records",
        )?;
        let bucket_roots = scratch.zeroed::<GpuXyzz>(
            parameters.dense_bucket_count as usize,
            "request.fixed-bucket-roots",
        )?;
        let tile_capacity = fixed_route_tiled_capacity(parameters.max_record_count)?;
        let tile_offsets = scratch
            .zeroed::<u32>(parameters.dense_bucket_count as usize, "request.fixed-tile-offsets")?;
        let tile_counts = scratch
            .zeroed::<u32>(parameters.dense_bucket_count as usize, "request.fixed-tile-counts")?;
        let tiles =
            scratch.zeroed::<BucketTileV1>(tile_capacity as usize, "request.fixed-tiles")?;
        let tile_roots =
            scratch.zeroed::<GpuXyzz>(tile_capacity as usize, "request.fixed-tile-roots")?;
        let indirect_groups =
            scratch.upload_slice(&[0_u32, 1_u32, 1_u32], "request.fixed-indirect-groups")?;
        let block_count = plan
            .output_count
            .checked_mul(parameters.integration_blocks_per_output as usize)
            .ok_or(SysError::LengthOverflow("fixed integration block count"))?;
        let block_sums =
            scratch.zeroed::<GpuXyzz>(block_count, "request.fixed-integration-block-sums")?;
        let block_weighted =
            scratch.zeroed::<GpuXyzz>(block_count, "request.fixed-integration-block-weighted")?;
        let outputs = scratch.zeroed::<GpuJacobian>(plan.output_count, "request.fixed-outputs")?;
        let status = scratch.zeroed::<KernelStatus>(1, "request.fixed-status")?;
        #[cfg(feature = "metal-diagnostics")]
        let allocation_elapsed = allocation_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let compute_started = Instant::now();
        let compute = self.encode_fixed_gpu_routed(
            shifted_bases,
            &plan,
            &scalars,
            point_indices.as_deref(),
            montgomery_input.as_deref(),
            &jobs,
            &histogram,
            &offsets,
            &cursors,
            &records,
            &bucket_roots,
            FixedGpuRoutedTiledBuffers {
                tile_offsets: &tile_offsets,
                tile_counts: &tile_counts,
                tiles: &tiles,
                tile_roots: &tile_roots,
                indirect_groups: &indirect_groups,
                tile_capacity,
            },
            &block_sums,
            &block_weighted,
            &outputs,
            &status,
        );
        #[cfg(feature = "metal-diagnostics")]
        let compute_elapsed = compute_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let read_started = Instant::now();
        let result = match compute {
            Ok(()) => {
                let kernel_status = read_one::<KernelStatus>(&status);
                if kernel_status == KernelStatus::default() {
                    Ok(read_values::<GpuJacobian>(&outputs, plan.output_count))
                } else {
                    Err(SysError::KernelStatus(kernel_status))
                }
            }
            Err(error) => Err(error),
        };
        #[cfg(feature = "metal-diagnostics")]
        let read_elapsed = read_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let erasure_started = Instant::now();
        let erasure = scratch.finish();
        #[cfg(feature = "metal-diagnostics")]
        trace_request_timing(
            allocation_elapsed,
            compute_elapsed,
            read_elapsed,
            erasure_started.elapsed(),
            request_started.elapsed(),
        );
        match erasure {
            Ok(()) => result,
            Err(error) => Err(error),
        }
    }

    fn execute_gpu_routed_inner(
        &self,
        bases: &PreparedBaseBuffer,
        plan: GpuRoutedExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let request_started = Instant::now();

        let tiled = matches!(
            plan.reduction_strategy,
            MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
                | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
                | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
                | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
        );
        let batch_affine =
            plan.reduction_strategy == MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9;
        let mut scratch = ScratchGuard::new(
            self,
            if batch_affine {
                18
            } else if tiled {
                16
            } else {
                11
            },
        )?;
        let scalars = scratch.upload_slice_or_word(plan.scalars, "request.gpu-route-scalars")?;
        let point_indices =
            scratch.upload_slice_or_word(plan.point_indices, "request.gpu-route-point-indices")?;
        let jobs = scratch.upload_slice(plan.jobs, "request.gpu-route-jobs")?;
        let dense_count = plan.parameters.dense_bucket_count as usize;
        let histogram = scratch.zeroed::<u32>(dense_count, "request.gpu-route-histogram")?;
        let offsets = scratch.zeroed::<u32>(dense_count, "request.gpu-route-offsets")?;
        let cursors = scratch.zeroed::<u32>(dense_count, "request.gpu-route-cursors")?;
        let records = scratch.zeroed::<SignedRecord>(
            (plan.parameters.max_record_count as usize).max(1),
            "request.gpu-route-records",
        )?;
        let (bucket_roots, window_points) = match plan.reduction_strategy {
            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3 => (
                scratch.zeroed::<GpuJacobian>(
                    dense_count,
                    "request.gpu-route-jacobian-bucket-roots",
                )?,
                scratch.zeroed::<GpuJacobian>(
                    plan.window_point_count,
                    "request.gpu-route-jacobian-window-points",
                )?,
            ),
            MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => (
                scratch.zeroed::<GpuXyzz>(dense_count, "request.gpu-route-xyzz-bucket-roots")?,
                scratch.zeroed::<GpuXyzz>(
                    plan.window_point_count,
                    "request.gpu-route-xyzz-window-points",
                )?,
            ),
            _ => {
                return Err(SysError::InvalidPlan(
                    "non-GPU strategy reached GPU-routed allocation",
                ));
            }
        };
        let outputs =
            scratch.zeroed::<GpuJacobian>(plan.output_count, "request.gpu-route-outputs")?;
        let status = scratch.zeroed::<KernelStatus>(1, "request.gpu-route-status")?;
        let tile_capacity = tiled
            .then(|| {
                gpu_route_tiled_xyzz_capacity(
                    plan.reduction_strategy,
                    plan.parameters.max_record_count,
                    plan.parameters.dense_bucket_count,
                )
            })
            .transpose()?;
        let tile_offsets = tile_capacity
            .map(|_| scratch.zeroed::<u32>(dense_count, "request.gpu-route-v6-tile-offsets"))
            .transpose()?;
        let tile_counts = tile_capacity
            .map(|_| scratch.zeroed::<u32>(dense_count, "request.gpu-route-v6-tile-counts"))
            .transpose()?;
        let tiles = tile_capacity
            .map(|capacity| {
                scratch.zeroed::<BucketTileV1>(capacity as usize, "request.gpu-route-v6-tiles")
            })
            .transpose()?;
        let tile_roots = tile_capacity
            .map(|capacity| {
                scratch.zeroed::<GpuXyzz>(capacity as usize, "request.gpu-route-v6-tile-roots")
            })
            .transpose()?;
        let indirect_groups = tile_capacity
            .map(|_| {
                scratch.upload_slice(&[0_u32, 1_u32, 1_u32], "request.gpu-route-v6-indirect-groups")
            })
            .transpose()?;
        let batch_affine_tree = if batch_affine {
            Some(product_tree_encoding(tile_capacity.ok_or(SysError::InvalidPlan(
                "batch-affine XYZZ strategy has no tile capacity",
            ))?)?)
        } else {
            None
        };
        let batch_affine_product_tree = batch_affine_tree
            .as_ref()
            .map(|tree| scratch.zeroed::<GpuFp>(tree.slots, "request.gpu-route-v9-product-tree"))
            .transpose()?;
        let batch_affine_status = batch_affine
            .then(|| scratch.zeroed::<BatchAffineStatus>(1, "request.gpu-route-v9-affine-status"))
            .transpose()?;

        #[cfg(feature = "metal-diagnostics")]
        let allocation_elapsed = request_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let compute_started = Instant::now();
        let buffers = GpuRoutedBuffers {
            scalars: &scalars,
            point_indices: &point_indices,
            jobs: &jobs,
            histogram: &histogram,
            offsets: &offsets,
            cursors: &cursors,
            records: &records,
            bucket_roots: &bucket_roots,
            window_points: &window_points,
            outputs: &outputs,
            status: &status,
            tiled: match (
                tile_offsets.as_deref(),
                tile_counts.as_deref(),
                tiles.as_deref(),
                tile_roots.as_deref(),
                indirect_groups.as_deref(),
                tile_capacity,
            ) {
                (
                    Some(tile_offsets),
                    Some(tile_counts),
                    Some(tiles),
                    Some(tile_roots),
                    Some(indirect_groups),
                    Some(tile_capacity),
                ) => Some(GpuRoutedTiledBuffers {
                    tile_offsets,
                    tile_counts,
                    tiles,
                    tile_roots,
                    indirect_groups,
                    tile_capacity,
                }),
                (None, None, None, None, None, None) => None,
                _ => return Err(SysError::InvalidPlan("incomplete tiled XYZZ allocation")),
            },
            batch_affine: match (
                batch_affine_product_tree.as_deref(),
                batch_affine_status.as_deref(),
            ) {
                (Some(product_tree), Some(status)) => {
                    Some(GpuRoutedBatchAffineBuffers { product_tree, status })
                }
                (None, None) => None,
                _ => {
                    return Err(SysError::InvalidPlan("incomplete batch-affine XYZZ allocation"));
                }
            },
        };
        let compute = self.encode_gpu_routed(bases, &plan, buffers);
        #[cfg(feature = "metal-diagnostics")]
        let compute_elapsed = compute_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let read_started = Instant::now();
        let result = match compute {
            Ok(()) => {
                let batch_affine_kernel_status = batch_affine_status
                    .as_deref()
                    .map(read_one::<BatchAffineStatus>)
                    .unwrap_or_default();
                let kernel_status = read_one::<KernelStatus>(&status);
                if batch_affine_kernel_status != BatchAffineStatus::default() {
                    Err(SysError::BatchAffineStatus(batch_affine_kernel_status))
                } else if kernel_status == KernelStatus::default() {
                    Ok(read_values::<GpuJacobian>(&outputs, plan.output_count))
                } else {
                    Err(SysError::KernelStatus(kernel_status))
                }
            }
            Err(error) => Err(error),
        };
        #[cfg(feature = "metal-diagnostics")]
        let read_elapsed = read_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let erasure_started = Instant::now();
        let erasure = scratch.finish();
        #[cfg(feature = "metal-diagnostics")]
        trace_request_timing(
            allocation_elapsed,
            compute_elapsed,
            read_elapsed,
            erasure_started.elapsed(),
            request_started.elapsed(),
        );
        match erasure {
            Ok(()) => result,
            Err(error) => Err(error),
        }
    }

    fn execute_batch_affine(
        &self,
        bases: &PreparedBaseBuffer,
        plan: ExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let request_started = Instant::now();

        let mut scratch = ScratchGuard::new(self, 15)?;
        let affine_records = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.affine_records,
            "request.affine-records",
        )?);
        let affine_destination_map = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.affine_destination_map,
            "request.affine-destination-map",
        )?);
        let points_a = scratch.retain(zeroed_buffer_for_values::<GpuAffine>(
            &self.device,
            plan.affine_records.len().max(1),
            "request.affine-points-a",
        )?);
        let points_b = scratch.retain(zeroed_buffer_for_values::<GpuAffine>(
            &self.device,
            plan.affine_records.len().max(1),
            "request.affine-points-b",
        )?);
        let numerators = scratch.retain(zeroed_buffer_for_values::<GpuFp>(
            &self.device,
            plan.affine_max_pairs.max(1),
            "request.affine-numerators",
        )?);
        let cases = scratch.retain(zeroed_buffer_for_values::<u32>(
            &self.device,
            plan.affine_max_pairs.max(1),
            "request.affine-cases",
        )?);
        let product_tree = scratch.retain(zeroed_buffer_for_values::<GpuFp>(
            &self.device,
            plan.affine_max_tree_slots.max(1),
            "request.affine-product-tree",
        )?);
        let run_roots = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.runs.len().max(1),
            "request.run-roots",
        )?);
        let dense_map = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.dense_map,
            "request.dense-map",
        )?);
        let windows = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.windows,
            "request.windows",
        )?);
        let window_points = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.window_point_count,
            "request.window-points",
        )?);
        let collects = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.collects,
            "request.collects",
        )?);
        let outputs = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.output_count,
            "request.outputs",
        )?);
        let affine_status = scratch.retain(zeroed_buffer_for_values::<BatchAffineStatus>(
            &self.device,
            1,
            "request.affine-status",
        )?);
        let status = scratch.retain(zeroed_buffer_for_values::<KernelStatus>(
            &self.device,
            1,
            "request.status",
        )?);

        #[cfg(feature = "metal-diagnostics")]
        let allocation_elapsed = request_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let compute_started = Instant::now();
        let compute = self.encode_compute(
            bases,
            &plan,
            &affine_records,
            &affine_destination_map,
            &points_a,
            &points_b,
            &numerators,
            &cases,
            &product_tree,
            &run_roots,
            &dense_map,
            &windows,
            &window_points,
            &collects,
            &outputs,
            &affine_status,
            &status,
        );
        #[cfg(feature = "metal-diagnostics")]
        let compute_elapsed = compute_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let read_started = Instant::now();
        let result = match compute {
            Ok(()) => {
                let batch_affine_status = read_one::<BatchAffineStatus>(&affine_status);
                let kernel_status = read_one::<KernelStatus>(&status);
                if batch_affine_status != BatchAffineStatus::default() {
                    Err(SysError::BatchAffineStatus(batch_affine_status))
                } else if kernel_status == KernelStatus::default() {
                    Ok(read_values::<GpuJacobian>(&outputs, plan.output_count))
                } else {
                    Err(SysError::KernelStatus(kernel_status))
                }
            }
            Err(error) => Err(error),
        };
        #[cfg(feature = "metal-diagnostics")]
        let read_elapsed = read_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let erasure_started = Instant::now();
        let erasure = scratch.finish();
        #[cfg(feature = "metal-diagnostics")]
        trace_request_timing(
            allocation_elapsed,
            compute_elapsed,
            read_elapsed,
            erasure_started.elapsed(),
            request_started.elapsed(),
        );
        match erasure {
            Ok(()) => result,
            Err(error) => Err(error),
        }
    }

    fn execute_parallel_projective(
        &self,
        bases: &PreparedBaseBuffer,
        plan: ExecutionPlan<'_>,
    ) -> Result<Vec<GpuJacobian>, SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let request_started = Instant::now();

        let mut scratch = ScratchGuard::new(self, 11)?;
        let records = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.records,
            "request.parallel-records",
        )?);
        let tiles = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.tiles,
            "request.parallel-tiles",
        )?);
        let tile_runs = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.tile_runs,
            "request.parallel-tile-runs",
        )?);
        let tile_roots = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.tiles.len().max(1),
            "request.parallel-tile-roots",
        )?);
        let run_roots = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.runs.len().max(1),
            "request.run-roots",
        )?);
        let dense_map = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.dense_map,
            "request.dense-map",
        )?);
        let windows = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.windows,
            "request.windows",
        )?);
        let window_points = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.window_point_count,
            "request.window-points",
        )?);
        let collects = scratch.retain(buffer_from_slice_or_word(
            &self.device,
            plan.collects,
            "request.collects",
        )?);
        let outputs = scratch.retain(zeroed_buffer_for_values::<GpuJacobian>(
            &self.device,
            plan.output_count,
            "request.outputs",
        )?);
        let status = scratch.retain(zeroed_buffer_for_values::<KernelStatus>(
            &self.device,
            1,
            "request.status",
        )?);

        #[cfg(feature = "metal-diagnostics")]
        let allocation_elapsed = request_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let compute_started = Instant::now();
        let buffers = ParallelProjectiveBuffers {
            records: &records,
            tiles: &tiles,
            tile_runs: &tile_runs,
            tile_roots: &tile_roots,
            run_roots: &run_roots,
            dense_map: &dense_map,
            windows: &windows,
            window_points: &window_points,
            collects: &collects,
            outputs: &outputs,
            status: &status,
        };
        let compute = self.encode_parallel_projective(bases, &plan, buffers);
        #[cfg(feature = "metal-diagnostics")]
        let compute_elapsed = compute_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let read_started = Instant::now();
        let result = match compute {
            Ok(()) => {
                let kernel_status = read_one::<KernelStatus>(&status);
                if kernel_status == KernelStatus::default() {
                    Ok(read_values::<GpuJacobian>(&outputs, plan.output_count))
                } else {
                    Err(SysError::KernelStatus(kernel_status))
                }
            }
            Err(error) => Err(error),
        };
        #[cfg(feature = "metal-diagnostics")]
        let read_elapsed = read_started.elapsed();

        #[cfg(feature = "metal-diagnostics")]
        let erasure_started = Instant::now();
        let erasure = scratch.finish();
        #[cfg(feature = "metal-diagnostics")]
        trace_request_timing(
            allocation_elapsed,
            compute_elapsed,
            read_elapsed,
            erasure_started.elapsed(),
            request_started.elapsed(),
        );
        match erasure {
            Ok(()) => result,
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_fixed_gpu_routed(
        &self,
        shifted_bases: &PreparedBaseBuffer,
        plan: &FixedGpuRoutedExecutionPlan<'_>,
        scalars: &ProtocolObject<dyn MTLBuffer>,
        point_indices: Option<&ProtocolObject<dyn MTLBuffer>>,
        montgomery_input: Option<&ProtocolObject<dyn MTLBuffer>>,
        jobs: &ProtocolObject<dyn MTLBuffer>,
        histogram: &ProtocolObject<dyn MTLBuffer>,
        offsets: &ProtocolObject<dyn MTLBuffer>,
        cursors: &ProtocolObject<dyn MTLBuffer>,
        records: &ProtocolObject<dyn MTLBuffer>,
        bucket_roots: &ProtocolObject<dyn MTLBuffer>,
        tiled: FixedGpuRoutedTiledBuffers<'_>,
        block_sums: &ProtocolObject<dyn MTLBuffer>,
        block_weighted: &ProtocolObject<dyn MTLBuffer>,
        outputs: &ProtocolObject<dyn MTLBuffer>,
        status: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<(), SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let encoding_started = Instant::now();
        let parameters = plan.parameters;
        let command = retained_command(&self.queue, "bls12-381-metal-msm.fixed-base-v10")?;
        #[cfg(feature = "metal-diagnostics")]
        let phase_samples = new_fixed_route_phase_samples(&self.device);
        #[cfg(feature = "metal-diagnostics")]
        let mut encoder = fixed_route_phase_encoder(&command, phase_samples.as_deref(), 0)?;
        #[cfg(not(feature = "metal-diagnostics"))]
        let encoder =
            command.computeCommandEncoder().ok_or(SysError::Encoder("fixed-base MSM v10"))?;

        encoder.setComputePipelineState(&self.pipelines.resident_scalar_canonicalize);
        match &plan.scalar_input {
            FixedScalarInput::HostMontgomery(values) if !values.is_empty() => {
                let copy = ResidentScalarCopyV1 {
                    source_count: checked_u32(values.len(), "host Montgomery scalar count")?,
                    destination_offset: 0,
                    destination_capacity: parameters.scalar_count,
                    reserved: 0,
                };
                bind_buffer(
                    &encoder,
                    montgomery_input.ok_or(SysError::InvalidPlan(
                        "host Montgomery input allocation is missing",
                    ))?,
                    0,
                );
                bind_buffer(&encoder, scalars, 1);
                bind_buffer(&encoder, status, 2);
                bind_bytes(&encoder, &copy, 3);
                dispatch(&encoder, &self.pipelines.resident_scalar_canonicalize, values.len());
            }
            FixedScalarInput::Resident(sources) => {
                for source in *sources {
                    if source.buffer.len() == 0 {
                        continue;
                    }
                    let copy = ResidentScalarCopyV1 {
                        source_count: checked_u32(
                            source.buffer.len(),
                            "resident scalar source count",
                        )?,
                        destination_offset: source.destination_offset,
                        destination_capacity: parameters.scalar_count,
                        reserved: 0,
                    };
                    bind_buffer_offset(
                        &encoder,
                        &source.buffer.raw,
                        source.buffer.byte_offset(),
                        0,
                    );
                    bind_buffer(&encoder, scalars, 1);
                    bind_buffer(&encoder, status, 2);
                    bind_bytes(&encoder, &copy, 3);
                    dispatch(
                        &encoder,
                        &self.pipelines.resident_scalar_canonicalize,
                        source.buffer.len(),
                    );
                }
            }
            FixedScalarInput::Host { .. } | FixedScalarInput::HostMontgomery(_) => {}
        }
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 1)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_histogram);
        bind_buffer(&encoder, scalars, 0);
        bind_buffer(&encoder, jobs, 1);
        bind_buffer(&encoder, histogram, 2);
        bind_buffer(&encoder, status, 3);
        bind_bytes(&encoder, &parameters, 4);
        dispatch_3d(
            &encoder,
            &self.pipelines.fixed_route_histogram,
            (parameters.max_terms_per_job as usize).max(1),
            parameters.job_count as usize,
            parameters.windows_per_scalar as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 2)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_scan);
        bind_buffer(&encoder, jobs, 0);
        bind_buffer(&encoder, histogram, 1);
        bind_buffer(&encoder, offsets, 2);
        bind_buffer(&encoder, cursors, 3);
        bind_buffer(&encoder, status, 4);
        bind_bytes(&encoder, &parameters, 5);
        dispatch(&encoder, &self.pipelines.fixed_route_scan, plan.jobs.len());
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 3)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_build_tiles);
        bind_buffer(&encoder, histogram, 0);
        bind_buffer(&encoder, offsets, 1);
        bind_buffer(&encoder, tiled.tile_offsets, 2);
        bind_buffer(&encoder, tiled.tile_counts, 3);
        bind_buffer(&encoder, tiled.tiles, 4);
        bind_buffer(&encoder, tiled.indirect_groups, 5);
        bind_buffer(&encoder, status, 6);
        bind_bytes(&encoder, &parameters, 7);
        bind_bytes(&encoder, &tiled.tile_capacity, 8);
        dispatch(
            &encoder,
            &self.pipelines.fixed_route_build_tiles,
            parameters.dense_bucket_count as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 4)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_finalize_tiles);
        bind_buffer(&encoder, tiled.indirect_groups, 0);
        bind_buffer(&encoder, status, 1);
        bind_bytes(&encoder, &tiled.tile_capacity, 2);
        dispatch(&encoder, &self.pipelines.fixed_route_finalize_tiles, 1);
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 5)?;

        let scatter = if point_indices.is_some() {
            &self.pipelines.fixed_route_scatter
        } else {
            &self.pipelines.fixed_route_resident_scatter
        };
        encoder.setComputePipelineState(scatter);
        bind_buffer(&encoder, scalars, 0);
        bind_buffer(&encoder, jobs, 1);
        bind_buffer(&encoder, histogram, 2);
        bind_buffer(&encoder, offsets, 3);
        bind_buffer(&encoder, cursors, 4);
        bind_buffer(&encoder, records, 5);
        bind_buffer(&encoder, status, 6);
        bind_bytes(&encoder, &parameters, 7);
        if let Some(point_indices) = point_indices {
            bind_buffer(&encoder, point_indices, 8);
        }
        dispatch_3d(
            &encoder,
            scatter,
            (parameters.max_terms_per_job as usize).max(1),
            parameters.job_count as usize,
            parameters.windows_per_scalar as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 6)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_reduce);
        bind_buffer(&encoder, &shifted_bases.raw, 0);
        bind_buffer(&encoder, records, 1);
        bind_buffer(&encoder, histogram, 2);
        bind_buffer(&encoder, offsets, 3);
        bind_buffer(&encoder, bucket_roots, 4);
        bind_buffer(&encoder, status, 5);
        bind_bytes(&encoder, &parameters, 6);
        dispatch_threadgroups(
            &encoder,
            (parameters.dense_bucket_count as usize)
                .div_ceil(parameters.buckets_per_group as usize),
            parameters.threads_per_group as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 7)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_reduce_tiles);
        bind_buffer(&encoder, &shifted_bases.raw, 0);
        bind_buffer(&encoder, records, 1);
        bind_buffer(&encoder, tiled.tiles, 2);
        bind_buffer(&encoder, tiled.tile_roots, 3);
        bind_buffer(&encoder, status, 4);
        bind_bytes(&encoder, &parameters, 5);
        bind_bytes(&encoder, &tiled.tile_capacity, 6);
        // The build/finalize pair bounds the indirect width before this
        // retained buffer is consumed by the tile reducer.
        unsafe {
            encoder
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    tiled.indirect_groups,
                    0,
                    MTLSize {
                        width: FIXED_ROUTE_LONG_THREADS_PER_GROUP_V10 as usize,
                        height: 1,
                        depth: 1,
                    },
                );
        }
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 8)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_combine_tiles);
        bind_buffer(&encoder, tiled.tile_offsets, 0);
        bind_buffer(&encoder, tiled.tile_counts, 1);
        bind_buffer(&encoder, tiled.tile_roots, 2);
        bind_buffer(&encoder, bucket_roots, 3);
        bind_buffer(&encoder, status, 4);
        bind_bytes(&encoder, &parameters, 5);
        bind_bytes(&encoder, &tiled.tile_capacity, 6);
        dispatch(
            &encoder,
            &self.pipelines.fixed_route_combine_tiles,
            parameters.dense_bucket_count as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 9)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_integrate_blocks);
        bind_buffer(&encoder, bucket_roots, 0);
        bind_buffer(&encoder, block_sums, 1);
        bind_buffer(&encoder, block_weighted, 2);
        bind_buffer(&encoder, status, 3);
        bind_bytes(&encoder, &parameters, 4);
        dispatch_threadgroups(
            &encoder,
            (plan.output_count * parameters.integration_blocks_per_output as usize)
                / fixed_route_integration_blocks_per_group_v10(
                    parameters.window_bits,
                    parameters.output_count,
                ) as usize,
            parameters.threads_per_group as usize,
        );
        #[cfg(feature = "metal-diagnostics")]
        advance_fixed_route_phase(&command, &mut encoder, phase_samples.as_deref(), 10)?;

        encoder.setComputePipelineState(&self.pipelines.fixed_route_finish_integration);
        bind_buffer(&encoder, block_sums, 0);
        bind_buffer(&encoder, block_weighted, 1);
        bind_buffer(&encoder, outputs, 2);
        bind_buffer(&encoder, status, 3);
        bind_bytes(&encoder, &parameters, 4);
        dispatch_threadgroups(&encoder, plan.output_count, parameters.threads_per_group as usize);
        encoder.endEncoding();

        #[cfg(feature = "metal-diagnostics")]
        let encoding_elapsed = encoding_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let timing_command = command.clone();
        #[cfg(feature = "metal-diagnostics")]
        let wait_started = Instant::now();
        let result = submit_and_wait(command);
        #[cfg(feature = "metal-diagnostics")]
        let gpu_elapsed = completed_gpu_duration(&timing_command);
        #[cfg(feature = "metal-diagnostics")]
        if result.is_ok() {
            trace_fixed_route_phases(phase_samples.as_deref(), gpu_elapsed);
        }
        #[cfg(feature = "metal-diagnostics")]
        trace_compute_timing(encoding_elapsed, wait_started.elapsed(), gpu_elapsed);
        result
    }

    fn encode_gpu_routed(
        &self,
        bases: &PreparedBaseBuffer,
        plan: &GpuRoutedExecutionPlan<'_>,
        buffers: GpuRoutedBuffers<'_>,
    ) -> Result<(), SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let encoding_started = Instant::now();
        let parameters = plan.parameters;
        let command = retained_command(&self.queue, "bls12-381-metal-msm.gpu-route-dense")?;
        let encoder =
            command.computeCommandEncoder().ok_or(SysError::Encoder("GPU-routed dense MSM"))?;

        encoder.setComputePipelineState(&self.pipelines.gpu_route_histogram);
        bind_buffer(&encoder, buffers.scalars, 0);
        bind_buffer(&encoder, buffers.jobs, 1);
        bind_buffer(&encoder, buffers.histogram, 2);
        bind_buffer(&encoder, buffers.status, 3);
        bind_bytes(&encoder, &parameters, 4);
        dispatch_3d(
            &encoder,
            &self.pipelines.gpu_route_histogram,
            (parameters.max_terms_per_job as usize).max(1),
            parameters.job_count as usize,
            parameters.windows_per_scalar as usize,
        );

        encoder.setComputePipelineState(&self.pipelines.gpu_route_scan);
        bind_buffer(&encoder, buffers.jobs, 0);
        bind_buffer(&encoder, buffers.histogram, 1);
        bind_buffer(&encoder, buffers.offsets, 2);
        bind_buffer(&encoder, buffers.cursors, 3);
        bind_buffer(&encoder, buffers.status, 4);
        bind_bytes(&encoder, &parameters, 5);
        dispatch(
            &encoder,
            &self.pipelines.gpu_route_scan,
            plan.jobs.len() * parameters.windows_per_scalar as usize,
        );

        if let Some(tiled) = buffers.tiled {
            let build_pipeline = match plan.reduction_strategy {
                MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => {
                    &self.pipelines.gpu_route_batch_affine_xyzz_build
                }
                MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8 => {
                    &self.pipelines.gpu_route_fine_tiled_xyzz_build
                }
                MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
                | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7 => {
                    &self.pipelines.gpu_route_tiled_xyzz_build
                }
                _ => {
                    return Err(SysError::InvalidPlan(
                        "non-tiled strategy reached tiled XYZZ build",
                    ));
                }
            };
            encoder.setComputePipelineState(build_pipeline);
            bind_buffer(&encoder, buffers.histogram, 0);
            bind_buffer(&encoder, buffers.offsets, 1);
            bind_buffer(&encoder, tiled.tile_offsets, 2);
            bind_buffer(&encoder, tiled.tile_counts, 3);
            bind_buffer(&encoder, tiled.tiles, 4);
            bind_buffer(&encoder, tiled.indirect_groups, 5);
            bind_buffer(&encoder, buffers.status, 6);
            bind_bytes(&encoder, &parameters, 7);
            bind_bytes(&encoder, &tiled.tile_capacity, 8);
            dispatch(&encoder, build_pipeline, parameters.dense_bucket_count as usize);

            encoder.setComputePipelineState(&self.pipelines.gpu_route_tiled_xyzz_finalize);
            bind_buffer(&encoder, tiled.indirect_groups, 0);
            bind_buffer(&encoder, buffers.status, 1);
            bind_bytes(&encoder, &tiled.tile_capacity, 2);
            dispatch(&encoder, &self.pipelines.gpu_route_tiled_xyzz_finalize, 1);
        }

        encoder.setComputePipelineState(&self.pipelines.gpu_route_scatter);
        bind_buffer(&encoder, buffers.scalars, 0);
        bind_buffer(&encoder, buffers.jobs, 1);
        bind_buffer(&encoder, buffers.histogram, 2);
        bind_buffer(&encoder, buffers.offsets, 3);
        bind_buffer(&encoder, buffers.cursors, 4);
        bind_buffer(&encoder, buffers.records, 5);
        bind_buffer(&encoder, buffers.status, 6);
        bind_bytes(&encoder, &parameters, 7);
        bind_buffer(&encoder, buffers.point_indices, 8);
        dispatch_3d(
            &encoder,
            &self.pipelines.gpu_route_scatter,
            (parameters.max_terms_per_job as usize).max(1),
            parameters.job_count as usize,
            parameters.windows_per_scalar as usize,
        );

        let (reduce_small, reduce_large, integrate_pipeline, collect_pipeline) = match plan
            .reduction_strategy
        {
            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3 => (
                &self.pipelines.gpu_route_reduce_small,
                &self.pipelines.gpu_route_reduce_large,
                &self.pipelines.gpu_route_integrate,
                &self.pipelines.gpu_route_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4 => (
                &self.pipelines.gpu_route_xyzz_reduce_small,
                &self.pipelines.gpu_route_xyzz_reduce_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5 => (
                &self.pipelines.gpu_route_adaptive_xyzz_reduce_small,
                &self.pipelines.gpu_route_adaptive_xyzz_reduce_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6 => (
                &self.pipelines.gpu_route_tiled_xyzz_reduce_short_small,
                &self.pipelines.gpu_route_tiled_xyzz_reduce_short_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7 => (
                &self.pipelines.gpu_route_tiled_xyzz_reduce_short_small,
                &self.pipelines.gpu_route_tiled_xyzz_reduce_short_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8 => (
                &self.pipelines.gpu_route_fine_tiled_xyzz_reduce_short_small,
                &self.pipelines.gpu_route_fine_tiled_xyzz_reduce_short_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => (
                &self.pipelines.gpu_route_fine_tiled_xyzz_reduce_short_small,
                &self.pipelines.gpu_route_fine_tiled_xyzz_reduce_short_large,
                &self.pipelines.gpu_route_xyzz_integrate,
                &self.pipelines.gpu_route_xyzz_collect,
            ),
            _ => {
                return Err(SysError::InvalidPlan("non-GPU strategy reached GPU-routed encoder"));
            }
        };
        let (reduce_pipeline, threads_per_group) = match (
            plan.reduction_strategy,
            parameters.buckets_per_group,
            parameters.threads_per_group,
        ) {
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 4, 32) => {
                (&self.pipelines.gpu_route_xyzz_reduce_small_pack4, 32)
            }
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 4, 64) => {
                (&self.pipelines.gpu_route_xyzz_reduce_large_pack4, 64)
            }
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 8, 32) => {
                (&self.pipelines.gpu_route_xyzz_reduce_small_pack8, 32)
            }
            (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 8, 64) => {
                (&self.pipelines.gpu_route_xyzz_reduce_large_pack8, 64)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 4, 32) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_small_pack4, 32)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 4, 64) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_large_pack4, 64)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 8, 32) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_small_pack8, 32)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 8, 64) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_large_pack8, 64)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 16, 32) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_small_pack16, 32)
            }
            (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 16, 64) => {
                (&self.pipelines.gpu_route_packed_tiled_xyzz_reduce_large_pack16, 64)
            }
            (_, 2, 32) => (reduce_small, 32),
            (_, 2, 64) => (reduce_large, 64),
            _ => {
                return Err(SysError::InvalidPlan(
                    "GPU-routed bucket packing geometry is unsupported",
                ));
            }
        };
        if plan.reduction_strategy == MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 {
            let tiled = buffers
                .tiled
                .ok_or(SysError::InvalidPlan("batch-affine XYZZ execution has no tile buffers"))?;
            let batch_affine = buffers
                .batch_affine
                .ok_or(SysError::InvalidPlan("batch-affine XYZZ execution has no product tree"))?;
            let tree = product_tree_encoding(tiled.tile_capacity)?;

            encoder.setComputePipelineState(&self.pipelines.gpu_route_batch_affine_xyzz_initialize);
            bind_buffer(&encoder, batch_affine.product_tree, 0);
            bind_bytes(&encoder, &tiled.tile_capacity, 1);
            dispatch(
                &encoder,
                &self.pipelines.gpu_route_batch_affine_xyzz_initialize,
                tiled.tile_capacity as usize,
            );

            encoder.setComputePipelineState(&self.pipelines.gpu_route_batch_affine_xyzz_prepare);
            bind_buffer(&encoder, &bases.raw, 0);
            bind_buffer(&encoder, buffers.records, 1);
            bind_buffer(&encoder, tiled.tiles, 2);
            bind_buffer(&encoder, batch_affine.product_tree, 3);
            bind_buffer(&encoder, buffers.status, 4);
            bind_bytes(&encoder, &parameters, 5);
            bind_bytes(&encoder, &tiled.tile_capacity, 6);
            // SAFETY: the build/finalize kernels bound the indirect width by
            // tile_capacity before this dispatch consumes the retained buffer.
            unsafe {
                encoder
                    .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        tiled.indirect_groups,
                        0,
                        MTLSize { width: 64, height: 1, depth: 1 },
                    );
            }

            for reduction in &tree.reductions[..tree.reduction_count] {
                encoder.setComputePipelineState(&self.pipelines.affine_reduce_products);
                bind_buffer(&encoder, batch_affine.product_tree, 0);
                bind_buffer(&encoder, batch_affine.status, 1);
                bind_bytes(&encoder, reduction, 2);
                dispatch(
                    &encoder,
                    &self.pipelines.affine_reduce_products,
                    reduction.output_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.pipelines.affine_invert_root);
            bind_buffer(&encoder, batch_affine.product_tree, 0);
            bind_buffer(&encoder, batch_affine.status, 1);
            bind_bytes(&encoder, &tree.root, 2);
            dispatch(&encoder, &self.pipelines.affine_invert_root, 1);

            for reduction in tree.reductions[..tree.reduction_count].iter().rev() {
                let expansion = BatchAffineProductParameters {
                    input_offset: reduction.output_offset,
                    input_count: reduction.output_count,
                    output_offset: reduction.input_offset,
                    output_count: reduction.input_count,
                };
                encoder.setComputePipelineState(&self.pipelines.affine_expand_inverses);
                bind_buffer(&encoder, batch_affine.product_tree, 0);
                bind_buffer(&encoder, batch_affine.status, 1);
                bind_bytes(&encoder, &expansion, 2);
                dispatch(
                    &encoder,
                    &self.pipelines.affine_expand_inverses,
                    expansion.input_count as usize,
                );
            }

            encoder.setComputePipelineState(&self.pipelines.gpu_route_batch_affine_xyzz_finish);
            bind_buffer(&encoder, &bases.raw, 0);
            bind_buffer(&encoder, buffers.records, 1);
            bind_buffer(&encoder, tiled.tiles, 2);
            bind_buffer(&encoder, batch_affine.product_tree, 3);
            bind_buffer(&encoder, tiled.tile_roots, 4);
            bind_buffer(&encoder, buffers.status, 5);
            bind_bytes(&encoder, &parameters, 6);
            bind_bytes(&encoder, &tiled.tile_capacity, 7);
            // SAFETY: identical bounded indirect geometry to the prepare pass.
            unsafe {
                encoder
                    .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        tiled.indirect_groups,
                        0,
                        MTLSize { width: 64, height: 1, depth: 1 },
                    );
            }

            encoder.setComputePipelineState(&self.pipelines.gpu_route_tiled_xyzz_combine);
            bind_buffer(&encoder, tiled.tile_offsets, 0);
            bind_buffer(&encoder, tiled.tile_counts, 1);
            bind_buffer(&encoder, tiled.tile_roots, 2);
            bind_buffer(&encoder, buffers.bucket_roots, 3);
            bind_buffer(&encoder, buffers.status, 4);
            bind_bytes(&encoder, &parameters, 5);
            bind_bytes(&encoder, &tiled.tile_capacity, 6);
            dispatch(
                &encoder,
                &self.pipelines.gpu_route_tiled_xyzz_combine,
                parameters.dense_bucket_count as usize,
            );
        } else {
            if buffers.batch_affine.is_some() {
                return Err(SysError::InvalidPlan(
                    "non-batch-affine strategy received a product tree",
                ));
            }
            encoder.setComputePipelineState(reduce_pipeline);
            bind_buffer(&encoder, &bases.raw, 0);
            bind_buffer(&encoder, buffers.records, 1);
            bind_buffer(&encoder, buffers.histogram, 2);
            bind_buffer(&encoder, buffers.offsets, 3);
            bind_buffer(&encoder, buffers.bucket_roots, 4);
            bind_buffer(&encoder, buffers.status, 5);
            bind_bytes(&encoder, &parameters, 6);
            dispatch_threadgroups(
                &encoder,
                (parameters.dense_bucket_count as usize)
                    .div_ceil(parameters.buckets_per_group as usize),
                threads_per_group,
            );

            if let Some(tiled) = buffers.tiled {
                let tile_reduce_pipeline = match threads_per_group {
                    32 => &self.pipelines.gpu_route_tiled_xyzz_reduce_tiles_small,
                    64 => &self.pipelines.gpu_route_tiled_xyzz_reduce_tiles_large,
                    _ => {
                        return Err(SysError::InvalidPlan(
                            "tiled XYZZ threadgroup geometry is unsupported",
                        ));
                    }
                };
                encoder.setComputePipelineState(tile_reduce_pipeline);
                bind_buffer(&encoder, &bases.raw, 0);
                bind_buffer(&encoder, buffers.records, 1);
                bind_buffer(&encoder, tiled.tiles, 2);
                bind_buffer(&encoder, tiled.tile_roots, 3);
                bind_buffer(&encoder, buffers.status, 4);
                bind_bytes(&encoder, &parameters, 5);
                bind_bytes(&encoder, &tiled.tile_capacity, 6);
                // SAFETY: the retained three-u32 buffer is initialized as Metal's
                // indirect threadgroup argument, finalized on this encoder before
                // use, and bounded by the allocated tile capacity in both kernels.
                unsafe {
                    encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            tiled.indirect_groups,
                            0,
                            MTLSize { width: threads_per_group, height: 1, depth: 1 },
                        );
                }

                encoder.setComputePipelineState(&self.pipelines.gpu_route_tiled_xyzz_combine);
                bind_buffer(&encoder, tiled.tile_offsets, 0);
                bind_buffer(&encoder, tiled.tile_counts, 1);
                bind_buffer(&encoder, tiled.tile_roots, 2);
                bind_buffer(&encoder, buffers.bucket_roots, 3);
                bind_buffer(&encoder, buffers.status, 4);
                bind_bytes(&encoder, &parameters, 5);
                bind_bytes(&encoder, &tiled.tile_capacity, 6);
                dispatch(
                    &encoder,
                    &self.pipelines.gpu_route_tiled_xyzz_combine,
                    parameters.dense_bucket_count as usize,
                );
            }
        }

        encoder.setComputePipelineState(integrate_pipeline);
        bind_buffer(&encoder, buffers.bucket_roots, 0);
        bind_buffer(&encoder, buffers.window_points, 1);
        bind_buffer(&encoder, buffers.status, 2);
        bind_bytes(&encoder, &parameters, 3);
        dispatch(&encoder, integrate_pipeline, plan.window_point_count);

        encoder.setComputePipelineState(collect_pipeline);
        bind_buffer(&encoder, buffers.window_points, 0);
        bind_buffer(&encoder, buffers.outputs, 1);
        bind_buffer(&encoder, buffers.status, 2);
        bind_bytes(&encoder, &parameters, 3);
        dispatch(&encoder, collect_pipeline, plan.output_count);
        encoder.endEncoding();

        #[cfg(feature = "metal-diagnostics")]
        let encoding_elapsed = encoding_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let timing_command = command.clone();
        #[cfg(feature = "metal-diagnostics")]
        let wait_started = Instant::now();
        let result = submit_and_wait(command);
        #[cfg(feature = "metal-diagnostics")]
        trace_compute_timing(
            encoding_elapsed,
            wait_started.elapsed(),
            completed_gpu_duration(&timing_command),
        );
        result
    }

    fn encode_parallel_projective(
        &self,
        bases: &PreparedBaseBuffer,
        plan: &ExecutionPlan<'_>,
        buffers: ParallelProjectiveBuffers<'_>,
    ) -> Result<(), SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let encoding_started = Instant::now();
        let (abi_version, tile_pipeline, partial_pipeline, integrate_pipeline, collect_pipeline) =
            match plan.reduction_strategy {
                MetalMsmReductionStrategy::ParallelProjectiveV1 => (
                    PARALLEL_BUCKET_ABI_VERSION_V1,
                    &self.pipelines.parallel_bucket_tiles,
                    &self.pipelines.parallel_bucket_partials,
                    &self.pipelines.integrate,
                    &self.pipelines.collect,
                ),
                MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2 => {
                    let (tiles, partials) =
                        match (plan.parallel_threads, plan.parallel_tile_records) {
                            (
                                PARALLEL_JACOBIAN_SMALL_THREADS,
                                PARALLEL_JACOBIAN_SMALL_TILE_RECORDS,
                            ) => (
                                &self.pipelines.parallel_jacobian_small_tiles,
                                &self.pipelines.parallel_jacobian_small_partials,
                            ),
                            (
                                PARALLEL_JACOBIAN_LARGE_THREADS,
                                PARALLEL_JACOBIAN_LARGE_TILE_RECORDS,
                            ) => (
                                &self.pipelines.parallel_jacobian_large_tiles,
                                &self.pipelines.parallel_jacobian_large_partials,
                            ),
                            _ => {
                                return Err(SysError::InvalidPlan(
                                    "parallel Jacobian launch geometry is unsupported",
                                ));
                            }
                        };
                    (
                        PARALLEL_JACOBIAN_ABI_VERSION_V2,
                        tiles,
                        partials,
                        &self.pipelines.parallel_jacobian_integrate,
                        &self.pipelines.parallel_jacobian_collect,
                    )
                }
                MetalMsmReductionStrategy::GlobalBatchAffineV1 => {
                    return Err(SysError::InvalidPlan(
                        "batch-affine plan reached the parallel encoder",
                    ));
                }
                MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
                | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
                | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
                | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
                | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
                | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
                | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => {
                    return Err(SysError::InvalidPlan(
                        "GPU-routed plan reached the CPU-routed parallel encoder",
                    ));
                }
            };
        let threads_per_threadgroup = plan.parallel_threads;
        let records_per_tile = plan.parallel_tile_records;
        let command = retained_command(&self.queue, "bls12-381-metal-msm.parallel-projective")?;
        let encoder =
            command.computeCommandEncoder().ok_or(SysError::Encoder("parallel projective MSM"))?;

        if !plan.records.is_empty() {
            let parameters = ParallelBucketParametersV1 {
                abi_version,
                base_count: checked_u32(bases.len(), "prepared base count")?,
                record_count: checked_u32(plan.records.len(), "parallel record count")?,
                tile_count: checked_u32(plan.tiles.len(), "parallel tile count")?,
                run_count: checked_u32(plan.runs.len(), "parallel run count")?,
                threads_per_threadgroup: threads_per_threadgroup as u32,
                max_records_per_tile: records_per_tile,
                reserved: 0,
            };
            encoder.setComputePipelineState(tile_pipeline);
            bind_buffer(&encoder, &bases.raw, 0);
            bind_buffer(&encoder, buffers.records, 1);
            bind_buffer(&encoder, buffers.tiles, 2);
            bind_buffer(&encoder, buffers.tile_roots, 3);
            bind_buffer(&encoder, buffers.status, 4);
            bind_bytes(&encoder, &parameters, 5);
            dispatch_threadgroups(&encoder, plan.tiles.len(), threads_per_threadgroup);

            encoder.setComputePipelineState(partial_pipeline);
            bind_buffer(&encoder, buffers.tile_roots, 0);
            bind_buffer(&encoder, buffers.tiles, 1);
            bind_buffer(&encoder, buffers.tile_runs, 2);
            bind_buffer(&encoder, buffers.run_roots, 3);
            bind_buffer(&encoder, buffers.status, 4);
            bind_bytes(&encoder, &parameters, 5);
            dispatch_threadgroups(&encoder, plan.runs.len(), threads_per_threadgroup);
        }

        let integration = IntegrationParameters {
            dense_map_count: checked_u32(plan.dense_map.len(), "dense bucket count")?,
            run_count: checked_u32(plan.runs.len(), "bucket run count")?,
            window_count: checked_u32(plan.windows.len(), "window descriptor count")?,
            window_point_count: checked_u32(plan.window_point_count, "window point count")?,
        };
        encoder.setComputePipelineState(integrate_pipeline);
        bind_buffer(&encoder, buffers.run_roots, 0);
        bind_buffer(&encoder, buffers.dense_map, 1);
        bind_buffer(&encoder, buffers.windows, 2);
        bind_buffer(&encoder, buffers.window_points, 3);
        bind_buffer(&encoder, buffers.status, 4);
        bind_bytes(&encoder, &integration, 5);
        dispatch(&encoder, integrate_pipeline, plan.windows.len());

        let collection = CollectParameters {
            window_point_count: checked_u32(plan.window_point_count, "window point count")?,
            descriptor_count: checked_u32(plan.collects.len(), "collect descriptor count")?,
            output_count: checked_u32(plan.output_count, "MSM output count")?,
            reserved: 0,
        };
        encoder.setComputePipelineState(collect_pipeline);
        bind_buffer(&encoder, buffers.window_points, 0);
        bind_buffer(&encoder, buffers.collects, 1);
        bind_buffer(&encoder, buffers.outputs, 2);
        bind_buffer(&encoder, buffers.status, 3);
        bind_bytes(&encoder, &collection, 4);
        dispatch(&encoder, collect_pipeline, plan.collects.len());
        encoder.endEncoding();

        #[cfg(feature = "metal-diagnostics")]
        let encoding_elapsed = encoding_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let timing_command = command.clone();
        #[cfg(feature = "metal-diagnostics")]
        let wait_started = Instant::now();
        let result = submit_and_wait(command);
        #[cfg(feature = "metal-diagnostics")]
        trace_compute_timing(
            encoding_elapsed,
            wait_started.elapsed(),
            completed_gpu_duration(&timing_command),
        );
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_compute(
        &self,
        bases: &PreparedBaseBuffer,
        plan: &ExecutionPlan<'_>,
        affine_records: &ProtocolObject<dyn MTLBuffer>,
        affine_destination_map: &ProtocolObject<dyn MTLBuffer>,
        points_a: &ProtocolObject<dyn MTLBuffer>,
        points_b: &ProtocolObject<dyn MTLBuffer>,
        numerators: &ProtocolObject<dyn MTLBuffer>,
        cases: &ProtocolObject<dyn MTLBuffer>,
        product_tree: &ProtocolObject<dyn MTLBuffer>,
        run_roots: &ProtocolObject<dyn MTLBuffer>,
        dense_map: &ProtocolObject<dyn MTLBuffer>,
        windows: &ProtocolObject<dyn MTLBuffer>,
        window_points: &ProtocolObject<dyn MTLBuffer>,
        collects: &ProtocolObject<dyn MTLBuffer>,
        outputs: &ProtocolObject<dyn MTLBuffer>,
        affine_status: &ProtocolObject<dyn MTLBuffer>,
        status: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<(), SysError> {
        #[cfg(feature = "metal-diagnostics")]
        let encoding_started = Instant::now();
        let command = retained_command(&self.queue, "bls12-381-metal-msm.execute")?;
        let encoder = command.computeCommandEncoder().ok_or(SysError::Encoder("MSM"))?;

        if !plan.affine_records.is_empty() {
            let gather = BatchAffineGatherParameters {
                record_offset: 0,
                record_count: checked_u32(plan.affine_records.len(), "affine record count")?,
                srs_count: checked_u32(bases.len(), "prepared base count")?,
                reserved: 0,
            };
            encoder.setComputePipelineState(&self.pipelines.affine_gather);
            bind_buffer(&encoder, &bases.raw, 0);
            bind_buffer(&encoder, affine_records, 1);
            bind_buffer(&encoder, points_a, 2);
            bind_buffer(&encoder, affine_status, 3);
            bind_bytes(&encoder, &gather, 4);
            dispatch(&encoder, &self.pipelines.affine_gather, plan.affine_records.len());

            let mut source_is_a = true;
            for level in plan.affine_levels {
                let parameters = BatchAffineLevelParameters {
                    destination_map_offset: level.destination_map_offset,
                    pair_count: level.pair_count,
                    carry_count: level.carry_count,
                    destination_len: level.destination_count,
                };
                let tree = product_tree_encoding(level.pair_count)?;
                let (source, destination) =
                    if source_is_a { (points_a, points_b) } else { (points_b, points_a) };

                encoder.setComputePipelineState(&self.pipelines.affine_prepare);
                bind_buffer(&encoder, source, 0);
                bind_buffer(&encoder, numerators, 1);
                bind_buffer(&encoder, cases, 2);
                bind_buffer(&encoder, product_tree, 3);
                bind_buffer(&encoder, affine_status, 4);
                bind_bytes(&encoder, &parameters, 5);
                dispatch(&encoder, &self.pipelines.affine_prepare, level.pair_count as usize);

                for reduction in &tree.reductions[..tree.reduction_count] {
                    encoder.setComputePipelineState(&self.pipelines.affine_reduce_products);
                    bind_buffer(&encoder, product_tree, 0);
                    bind_buffer(&encoder, affine_status, 1);
                    bind_bytes(&encoder, reduction, 2);
                    dispatch(
                        &encoder,
                        &self.pipelines.affine_reduce_products,
                        reduction.output_count as usize,
                    );
                }

                encoder.setComputePipelineState(&self.pipelines.affine_invert_root);
                bind_buffer(&encoder, product_tree, 0);
                bind_buffer(&encoder, affine_status, 1);
                bind_bytes(&encoder, &tree.root, 2);
                dispatch(&encoder, &self.pipelines.affine_invert_root, 1);

                for reduction in tree.reductions[..tree.reduction_count].iter().rev() {
                    let expansion = BatchAffineProductParameters {
                        input_offset: reduction.output_offset,
                        input_count: reduction.output_count,
                        output_offset: reduction.input_offset,
                        output_count: reduction.input_count,
                    };
                    encoder.setComputePipelineState(&self.pipelines.affine_expand_inverses);
                    bind_buffer(&encoder, product_tree, 0);
                    bind_buffer(&encoder, affine_status, 1);
                    bind_bytes(&encoder, &expansion, 2);
                    dispatch(
                        &encoder,
                        &self.pipelines.affine_expand_inverses,
                        expansion.input_count as usize,
                    );
                }

                encoder.setComputePipelineState(&self.pipelines.affine_finish);
                bind_buffer(&encoder, source, 0);
                bind_buffer(&encoder, numerators, 1);
                bind_buffer(&encoder, cases, 2);
                bind_buffer(&encoder, product_tree, 3);
                bind_buffer(&encoder, affine_destination_map, 4);
                bind_buffer(&encoder, destination, 5);
                bind_buffer(&encoder, affine_status, 6);
                bind_bytes(&encoder, &parameters, 7);
                dispatch(&encoder, &self.pipelines.affine_finish, level.pair_count as usize);

                if level.carry_count != 0 {
                    encoder.setComputePipelineState(&self.pipelines.affine_copy_carries);
                    bind_buffer(&encoder, source, 0);
                    bind_buffer(&encoder, affine_destination_map, 1);
                    bind_buffer(&encoder, destination, 2);
                    bind_buffer(&encoder, affine_status, 3);
                    bind_bytes(&encoder, &parameters, 4);
                    dispatch(
                        &encoder,
                        &self.pipelines.affine_copy_carries,
                        level.carry_count as usize,
                    );
                }
                source_is_a = !source_is_a;
            }

            let root_source = if source_is_a { points_a } else { points_b };
            let root_count = checked_u32(plan.runs.len(), "bucket run count")?;
            encoder.setComputePipelineState(&self.pipelines.affine_export_roots);
            bind_buffer(&encoder, root_source, 0);
            bind_buffer(&encoder, run_roots, 1);
            bind_buffer(&encoder, affine_status, 2);
            bind_bytes(&encoder, &root_count, 3);
            dispatch(&encoder, &self.pipelines.affine_export_roots, plan.runs.len());
        }

        let integration = IntegrationParameters {
            dense_map_count: checked_u32(plan.dense_map.len(), "dense bucket count")?,
            run_count: checked_u32(plan.runs.len(), "bucket run count")?,
            window_count: checked_u32(plan.windows.len(), "window descriptor count")?,
            window_point_count: checked_u32(plan.window_point_count, "window point count")?,
        };
        encoder.setComputePipelineState(&self.pipelines.integrate);
        bind_buffer(&encoder, run_roots, 0);
        bind_buffer(&encoder, dense_map, 1);
        bind_buffer(&encoder, windows, 2);
        bind_buffer(&encoder, window_points, 3);
        bind_buffer(&encoder, status, 4);
        bind_bytes(&encoder, &integration, 5);
        dispatch(&encoder, &self.pipelines.integrate, plan.windows.len());

        let collection = CollectParameters {
            window_point_count: checked_u32(plan.window_point_count, "window point count")?,
            descriptor_count: checked_u32(plan.collects.len(), "collect descriptor count")?,
            output_count: checked_u32(plan.output_count, "MSM output count")?,
            reserved: 0,
        };
        encoder.setComputePipelineState(&self.pipelines.collect);
        bind_buffer(&encoder, window_points, 0);
        bind_buffer(&encoder, collects, 1);
        bind_buffer(&encoder, outputs, 2);
        bind_buffer(&encoder, status, 3);
        bind_bytes(&encoder, &collection, 4);
        dispatch(&encoder, &self.pipelines.collect, plan.collects.len());
        encoder.endEncoding();
        #[cfg(feature = "metal-diagnostics")]
        let encoding_elapsed = encoding_started.elapsed();
        #[cfg(feature = "metal-diagnostics")]
        let timing_command = command.clone();
        #[cfg(feature = "metal-diagnostics")]
        let wait_started = Instant::now();
        let result = submit_and_wait(command);
        #[cfg(feature = "metal-diagnostics")]
        trace_compute_timing(
            encoding_elapsed,
            wait_started.elapsed(),
            completed_gpu_duration(&timing_command),
        );
        result
    }

    fn erase_scratch(&self, scratch: &[ScratchBuffer]) -> Result<(), SysError> {
        if scratch.is_empty() {
            return Ok(());
        }
        for buffer in scratch {
            let bytes = buffer.used_bytes;
            if bytes == 0 || bytes > buffer.raw.length() || !bytes.is_multiple_of(size_of::<u32>())
            {
                return Err(SysError::InvalidPlan(
                    "scratch erasure range must be nonempty, in bounds, and word aligned",
                ));
            }
            checked_u32(bytes / size_of::<u32>(), "scratch word count")?;
        }
        let command = retained_command(&self.queue, "bls12-381-metal-msm.erase")?;
        let encoder = command.blitCommandEncoder().ok_or(SysError::Encoder("erasure"))?;
        for buffer in scratch {
            encoder.fillBuffer_range_value(&buffer.raw, NSRange::new(0, buffer.used_bytes), 0);
        }
        encoder.endEncoding();
        submit_and_wait(command)
    }

    fn take_cleared_scratch(
        &self,
        bytes: usize,
        label: &'static str,
    ) -> Result<Option<Buffer>, SysError> {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| SysError::Poisoned("host lifecycle mutex was poisoned".into()))?;
        if let Some(reason) = &lifecycle.poisoned {
            return Err(SysError::Poisoned(reason.clone()));
        }
        let Some(position) = lifecycle
            .cleared_pool
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.length() >= bytes)
            .min_by_key(|(_, buffer)| buffer.length())
            .map(|(position, _)| position)
        else {
            return Ok(None);
        };
        let buffer = lifecycle.cleared_pool.swap_remove(position);
        lifecycle.cleared_pool_bytes = lifecycle.cleared_pool_bytes.saturating_sub(buffer.length());
        buffer.setLabel(Some(&NSString::from_str(label)));
        Ok(Some(buffer))
    }

    fn recycle_cleared_scratch(&self, buffers: Vec<Buffer>) {
        let mut lifecycle =
            self.lifecycle.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for buffer in buffers {
            let bytes = buffer.length();
            if bytes > MAX_CLEARED_SCRATCH_BYTES {
                continue;
            }
            while (!lifecycle.cleared_pool.is_empty())
                && (lifecycle.cleared_pool.len() >= MAX_CLEARED_SCRATCH_BUFFERS
                    || lifecycle.cleared_pool_bytes.saturating_add(bytes)
                        > MAX_CLEARED_SCRATCH_BYTES)
            {
                let evicted = lifecycle.cleared_pool.swap_remove(0);
                lifecycle.cleared_pool_bytes =
                    lifecycle.cleared_pool_bytes.saturating_sub(evicted.length());
            }
            if lifecycle.cleared_pool.len() < MAX_CLEARED_SCRATCH_BUFFERS
                && lifecycle.cleared_pool_bytes.saturating_add(bytes) <= MAX_CLEARED_SCRATCH_BYTES
            {
                lifecycle.cleared_pool_bytes = lifecycle.cleared_pool_bytes.saturating_add(bytes);
                lifecycle.cleared_pool.push(buffer);
            }
        }
    }

    fn ensure_healthy(&self) -> Result<(), SysError> {
        let lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| SysError::Poisoned("host lifecycle mutex was poisoned".into()))?;
        match &lifecycle.poisoned {
            Some(reason) => Err(SysError::Poisoned(reason.clone())),
            None => Ok(()),
        }
    }

    fn poison_and_quarantine(&self, reason: String, buffers: Vec<Buffer>) {
        let mut lifecycle = match self.lifecycle.lock() {
            Ok(lifecycle) => lifecycle,
            Err(poisoned) => poisoned.into_inner(),
        };
        lifecycle.poisoned = Some(reason);
        if lifecycle.quarantine.is_none() {
            lifecycle.quarantine = Some(buffers);
        } else {
            // Execution is serialized and a poisoned context rejects every
            // later submission, so this branch is defensive only. Leaking is
            // preferable to releasing allocations whose erasure is unconfirmed.
            core::mem::forget(buffers);
        }
    }
}

struct ProductTreeEncoding {
    reductions: [BatchAffineProductParameters; u32::BITS as usize],
    reduction_count: usize,
    root: BatchAffineProductParameters,
    slots: usize,
}

fn product_tree_encoding(pair_count: u32) -> Result<ProductTreeEncoding, SysError> {
    if pair_count == 0 {
        return Err(SysError::InvalidPlan("batch-affine reduction level has no pairs"));
    }
    let mut reductions = [BatchAffineProductParameters::default(); u32::BITS as usize];
    let mut reduction_count = 0usize;
    let mut input_offset = 0u32;
    let mut input_count = pair_count;
    let mut next_offset = pair_count;
    while input_count > 1 {
        let output_count = input_count.div_ceil(2);
        let output_offset = next_offset;
        next_offset = next_offset
            .checked_add(output_count)
            .ok_or(SysError::LengthOverflow("batch-affine product tree"))?;
        let slot = reductions
            .get_mut(reduction_count)
            .ok_or(SysError::InvalidPlan("batch-affine product tree is too deep"))?;
        *slot =
            BatchAffineProductParameters { input_offset, input_count, output_offset, output_count };
        reduction_count += 1;
        input_offset = output_offset;
        input_count = output_count;
    }
    Ok(ProductTreeEncoding {
        reductions,
        reduction_count,
        root: BatchAffineProductParameters {
            input_offset,
            input_count: 1,
            output_offset: input_offset,
            output_count: 1,
        },
        slots: next_offset as usize,
    })
}

fn validate_fixed_gpu_routed_plan(
    expected_registry_id: u64,
    shifted_base_count: usize,
    plan: &FixedGpuRoutedExecutionPlan<'_>,
) -> Result<(), SysError> {
    let parameters = plan.parameters;
    let (scalar_count, point_indices) = match &plan.scalar_input {
        FixedScalarInput::Host { scalars, point_indices } => {
            if point_indices.len() != scalars.len() {
                return Err(SysError::InvalidPlan(
                    "fixed host scalar and point-index counts differ",
                ));
            }
            (scalars.len(), Some(*point_indices))
        }
        FixedScalarInput::HostMontgomery(scalars) => (scalars.len(), None),
        FixedScalarInput::Resident(sources) => {
            if sources.is_empty() {
                return Err(SysError::InvalidPlan("fixed resident source list is empty"));
            }
            let mut expected_offset = 0usize;
            for source in *sources {
                if source.buffer.registry_id() != expected_registry_id
                    || source.destination_offset as usize != expected_offset
                {
                    return Err(SysError::InvalidPlan(
                        "fixed resident source device or destination offset is inconsistent",
                    ));
                }
                expected_offset = expected_offset
                    .checked_add(source.buffer.len())
                    .ok_or(SysError::LengthOverflow("fixed resident scalar count"))?;
            }
            (expected_offset, None)
        }
    };
    let expected_windows = match parameters.window_bits {
        11 => 24,
        12 => 22,
        13 => 20,
        14 => 19,
        15 => 18,
        16 => 16,
        _ => return Err(SysError::InvalidPlan("fixed window width is unsupported")),
    };
    let expected_buckets = 1_u32 << (parameters.window_bits - 1);
    let expected_shifted = parameters
        .base_count
        .checked_mul(expected_windows)
        .ok_or(SysError::InvalidPlan("fixed shifted-base count overflows"))?;
    let expected_dense = parameters
        .output_count
        .checked_mul(expected_buckets)
        .ok_or(SysError::InvalidPlan("fixed dense-bucket count overflows"))?;
    if parameters.abi_version != FIXED_ROUTE_ABI_VERSION_V10
        || parameters.windows_per_scalar != expected_windows
        || parameters.buckets_per_output != expected_buckets
        || parameters.shifted_base_count != expected_shifted
        || parameters.shifted_base_count as usize != shifted_base_count
        || parameters.scalar_count as usize != scalar_count
        || parameters.job_count as usize != plan.jobs.len()
        || parameters.output_count as usize != plan.output_count
        || parameters.job_count != parameters.output_count
        || plan.jobs.is_empty()
        || parameters.dense_bucket_count != expected_dense
        || parameters.threads_per_group != FIXED_ROUTE_THREADS_PER_GROUP_V10
        || parameters.buckets_per_group
            != fixed_route_buckets_per_group_v10(parameters.window_bits, parameters.output_count)
        || parameters.integration_block_buckets != FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10
        || !expected_buckets.is_multiple_of(parameters.integration_block_buckets)
        || parameters.integration_blocks_per_output
            != expected_buckets / parameters.integration_block_buckets
    {
        return Err(SysError::InvalidPlan("fixed route topology is inconsistent"));
    }

    let mut outputs_seen = vec![false; plan.output_count];
    let mut expected_segment_offset = 0_u32;
    let mut maximum_terms = 0_u32;
    for job in plan.jobs {
        if job.reserved0 != 0 || job.reserved1 != 0 || job.reserved2 != 0 {
            return Err(SysError::InvalidPlan("fixed route job has reserved bits"));
        }
        let output = job.output_index as usize;
        if output >= outputs_seen.len() || outputs_seen[output] {
            return Err(SysError::InvalidPlan("fixed route outputs are not a permutation"));
        }
        outputs_seen[output] = true;
        if job.record_segment_offset != expected_segment_offset
            || !range_valid(job.base_offset, job.term_count, parameters.base_count)
            || !range_valid(job.scalar_offset, job.term_count, parameters.scalar_count)
        {
            return Err(SysError::InvalidPlan("fixed route job range is malformed"));
        }
        let scalar_start = job.scalar_offset as usize;
        let scalar_end = scalar_start
            .checked_add(job.term_count as usize)
            .ok_or(SysError::InvalidPlan("fixed route scalar range overflows"))?;
        if let Some(point_indices) = point_indices
            && point_indices[scalar_start..scalar_end]
                .iter()
                .any(|&index| index >= parameters.base_count)
        {
            return Err(SysError::InvalidPlan("fixed compact point index exceeds prepared bases"));
        }
        expected_segment_offset = expected_segment_offset
            .checked_add(
                job.term_count
                    .checked_mul(expected_windows)
                    .ok_or(SysError::InvalidPlan("fixed route record segment overflows"))?,
            )
            .ok_or(SysError::InvalidPlan("fixed route record count overflows"))?;
        maximum_terms = maximum_terms.max(job.term_count);
    }
    if expected_segment_offset != parameters.max_record_count
        || maximum_terms != parameters.max_terms_per_job
    {
        return Err(SysError::InvalidPlan("fixed route scratch bounds are inconsistent"));
    }
    plan.output_count
        .checked_mul(parameters.integration_blocks_per_output as usize)
        .ok_or(SysError::InvalidPlan("fixed integration block count overflows"))?;
    Ok(())
}

fn validate_gpu_routed_plan(
    base_count: usize,
    plan: &GpuRoutedExecutionPlan<'_>,
) -> Result<(), SysError> {
    if !matches!(
        plan.reduction_strategy,
        MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
            | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
            | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
            | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
            | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
            | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
            | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
    ) {
        return Err(SysError::InvalidPlan("GPU-routed plan has a non-GPU reduction strategy"));
    }
    let parameters = plan.parameters;
    if parameters.abi_version != GPU_ROUTED_DENSE_ABI_VERSION_V4
        || parameters.reserved0 != 0
        || parameters.reserved1 != 0
        || parameters.reserved2 != 0
    {
        return Err(SysError::InvalidPlan("GPU-routed parameter ABI is malformed"));
    }
    let expected_windows = match parameters.window_bits {
        6 => 43,
        7 => 37,
        8 => 32,
        9 => 29,
        10 => 26,
        _ => return Err(SysError::InvalidPlan("GPU-routed window width is unsupported")),
    };
    let expected_buckets = 1_u32 << (parameters.window_bits - 1);
    if parameters.windows_per_scalar != expected_windows
        || parameters.buckets_per_window != expected_buckets
        || parameters.base_count as usize != base_count
        || parameters.scalar_count as usize != plan.scalars.len()
        || plan.point_indices.len() != plan.scalars.len()
        || parameters.job_count as usize != plan.jobs.len()
        || parameters.output_count as usize != plan.output_count
        || parameters.job_count != parameters.output_count
        || plan.jobs.is_empty()
    {
        return Err(SysError::InvalidPlan("GPU-routed plan counts are inconsistent"));
    }
    let expected_dense = parameters
        .output_count
        .checked_mul(expected_windows)
        .and_then(|value| value.checked_mul(expected_buckets))
        .ok_or(SysError::InvalidPlan("GPU-routed dense bucket count overflows"))?;
    let packing_eligible = plan.packing_eligible;
    if parameters.dense_bucket_count != expected_dense
        || plan.window_point_count
            != plan
                .output_count
                .checked_mul(expected_windows as usize)
                .ok_or(SysError::InvalidPlan("GPU-routed window-point count overflows"))?
        || !matches!(parameters.threads_per_group, 32 | 64)
        || parameters.buckets_per_group
            != match (
                plan.reduction_strategy,
                parameters.window_bits,
                parameters.output_count,
                packing_eligible,
            ) {
                (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 10, _, _) => 8,
                (MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4, 9, _, _) => 4,
                (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 8, 4.., true) => 8,
                (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 9, 4.., true) => 4,
                (MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7, 10, 2.., true) => 8,
                _ => 2,
            }
    {
        return Err(SysError::InvalidPlan("GPU-routed topology is inconsistent"));
    }

    let mut outputs_seen = vec![false; plan.output_count];
    let mut expected_segment_offset = 0_u32;
    let mut maximum_terms = 0_u32;
    for job in plan.jobs {
        if job.reserved0 != 0 || job.reserved1 != 0 || job.reserved2 != 0 {
            return Err(SysError::InvalidPlan("GPU-routed job has reserved bits"));
        }
        let output = job.output_index as usize;
        if output >= outputs_seen.len() || outputs_seen[output] {
            return Err(SysError::InvalidPlan("GPU-routed outputs are not a permutation"));
        }
        outputs_seen[output] = true;
        if job.record_segment_offset != expected_segment_offset
            || !range_valid(job.base_offset, job.term_count, parameters.base_count)
            || !range_valid(job.scalar_offset, job.term_count, parameters.scalar_count)
        {
            return Err(SysError::InvalidPlan("GPU-routed job range is malformed"));
        }
        expected_segment_offset = expected_segment_offset
            .checked_add(
                job.term_count
                    .checked_mul(expected_windows)
                    .ok_or(SysError::InvalidPlan("GPU-routed record segment overflows"))?,
            )
            .ok_or(SysError::InvalidPlan("GPU-routed record count overflows"))?;
        maximum_terms = maximum_terms.max(job.term_count);
    }
    if expected_segment_offset != parameters.max_record_count
        || maximum_terms != parameters.max_terms_per_job
    {
        return Err(SysError::InvalidPlan("GPU-routed scratch bounds are inconsistent"));
    }
    for scalar in plan.scalars {
        if scalar.limbs == [0; 4] || !scalar_limbs_are_canonical(scalar.limbs) {
            return Err(SysError::InvalidPlan(
                "GPU-routed compact scalar is zero or non-canonical",
            ));
        }
    }
    if plan.point_indices.iter().any(|&point_index| point_index as usize >= base_count) {
        return Err(SysError::InvalidPlan("GPU-routed compact point index exceeds prepared bases"));
    }
    Ok(())
}

fn range_valid(offset: u32, count: u32, total: u32) -> bool {
    offset <= total && count <= total - offset
}

fn scalar_limbs_are_canonical(limbs: [u64; 4]) -> bool {
    for index in (0..limbs.len()).rev() {
        if limbs[index] != crate::msm_api::BLS12_381_SCALAR_MODULUS[index] {
            return limbs[index] < crate::msm_api::BLS12_381_SCALAR_MODULUS[index];
        }
    }
    false
}

fn validate_execution_plan(base_count: usize, plan: &ExecutionPlan<'_>) -> Result<(), SysError> {
    if base_count == 0 {
        return Err(SysError::InvalidPlan("prepared base allocation is empty"));
    }
    if plan.windows.is_empty() || plan.window_point_count == 0 {
        return Err(SysError::InvalidPlan("window descriptors must not be empty"));
    }
    if plan.collects.is_empty() || plan.output_count == 0 {
        return Err(SysError::InvalidPlan("collection descriptors must not be empty"));
    }
    if plan.window_point_count != plan.windows.len() {
        return Err(SysError::InvalidPlan(
            "every window descriptor must own exactly one window point",
        ));
    }
    if plan.output_count != plan.collects.len() {
        return Err(SysError::InvalidPlan(
            "every collection descriptor must own exactly one output",
        ));
    }
    if plan.runs.len() > plan.dense_map.len() {
        return Err(SysError::InvalidPlan("there are more runs than dense buckets"));
    }
    for record in plan.records {
        if record.point_index as usize >= base_count
            || record.flags & !super::abi::SIGNED_RECORD_NEGATE != 0
        {
            return Err(SysError::InvalidPlan("signed record is malformed"));
        }
    }
    match plan.reduction_strategy {
        MetalMsmReductionStrategy::GlobalBatchAffineV1 => {
            validate_affine_execution_plan(base_count, plan)?;
        }
        MetalMsmReductionStrategy::ParallelProjectiveV1
        | MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2 => {
            if !plan.affine_records.is_empty()
                || !plan.affine_levels.is_empty()
                || !plan.affine_destination_map.is_empty()
                || plan.affine_max_pairs != 0
                || plan.affine_max_tree_slots != 0
            {
                return Err(SysError::InvalidPlan(
                    "parallel projective plan contains batch-affine topology",
                ));
            }
        }
        MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
        | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
        | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
        | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
        | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
        | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
        | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => {
            return Err(SysError::InvalidPlan(
                "GPU-routed strategy reached CPU-routed plan validation",
            ));
        }
    }
    let mut expected_record_offset = 0usize;
    for run in plan.runs {
        if run.reserved0 != 0 || run.reserved1 != 0 || run.record_count == 0 {
            return Err(SysError::InvalidPlan("bucket run has reserved bits or no records"));
        }
        let offset = usize::try_from(run.record_offset)
            .map_err(|_| SysError::InvalidPlan("bucket run offset does not fit usize"))?;
        if offset != expected_record_offset {
            return Err(SysError::InvalidPlan(
                "bucket runs do not exactly partition routed records",
            ));
        }
        let end = offset
            .checked_add(run.record_count as usize)
            .ok_or(SysError::InvalidPlan("bucket run range overflows"))?;
        if end > plan.records.len() {
            return Err(SysError::InvalidPlan("bucket run exceeds routed records"));
        }
        expected_record_offset = end;
    }
    if expected_record_offset != plan.records.len() {
        return Err(SysError::InvalidPlan("bucket runs do not consume every routed record"));
    }
    if plan.tile_runs.len() != plan.runs.len() {
        return Err(SysError::InvalidPlan("parallel tile-run count does not match bucket runs"));
    }
    let valid_geometry = match plan.reduction_strategy {
        MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2 => {
            (plan.parallel_threads == PARALLEL_JACOBIAN_SMALL_THREADS
                && plan.parallel_tile_records == PARALLEL_JACOBIAN_SMALL_TILE_RECORDS)
                || (plan.parallel_threads == PARALLEL_JACOBIAN_LARGE_THREADS
                    && plan.parallel_tile_records == PARALLEL_JACOBIAN_LARGE_TILE_RECORDS)
        }
        MetalMsmReductionStrategy::GlobalBatchAffineV1
        | MetalMsmReductionStrategy::ParallelProjectiveV1 => {
            plan.parallel_threads == PARALLEL_BUCKET_THREADS
                && plan.parallel_tile_records == PARALLEL_BUCKET_TILE_RECORDS
        }
        MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
        | MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
        | MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
        | MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
        | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
        | MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
        | MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => false,
    };
    if !valid_geometry {
        return Err(SysError::InvalidPlan("parallel launch geometry is unsupported"));
    }
    let maximum_tile_records = plan.parallel_tile_records;
    let mut expected_tile_offset = 0usize;
    for (run_index, (run, tile_run)) in plan.runs.iter().zip(plan.tile_runs).enumerate() {
        if tile_run.reserved0 != 0 || tile_run.reserved1 != 0 || tile_run.tile_count == 0 {
            return Err(SysError::InvalidPlan("parallel tile run is malformed"));
        }
        let tile_offset = tile_run.tile_offset as usize;
        let tile_end = tile_offset
            .checked_add(tile_run.tile_count as usize)
            .ok_or(SysError::InvalidPlan("parallel tile span overflows"))?;
        if tile_offset != expected_tile_offset || tile_end > plan.tiles.len() {
            return Err(SysError::InvalidPlan("parallel tile runs do not partition all tiles"));
        }
        let run_end = (run.record_offset as usize)
            .checked_add(run.record_count as usize)
            .ok_or(SysError::InvalidPlan("parallel bucket run range overflows"))?;
        let mut expected_tile_record = run.record_offset as usize;
        for tile in &plan.tiles[tile_offset..tile_end] {
            if tile.reserved != 0
                || tile.run_index as usize != run_index
                || tile.record_count == 0
                || tile.record_count > maximum_tile_records
                || tile.record_offset as usize != expected_tile_record
            {
                return Err(SysError::InvalidPlan("parallel bucket tile is malformed"));
            }
            expected_tile_record = expected_tile_record
                .checked_add(tile.record_count as usize)
                .ok_or(SysError::InvalidPlan("parallel tile record range overflows"))?;
            if expected_tile_record > run_end {
                return Err(SysError::InvalidPlan("parallel bucket tile exceeds its run"));
            }
        }
        if expected_tile_record != run_end {
            return Err(SysError::InvalidPlan("parallel bucket tiles do not partition their run"));
        }
        expected_tile_offset = tile_end;
    }
    if expected_tile_offset != plan.tiles.len() {
        return Err(SysError::InvalidPlan("parallel tile runs do not consume every tile"));
    }
    let mut expected_run_index = 0usize;
    for &run_index in plan.dense_map {
        if run_index != u32::MAX && run_index as usize >= plan.runs.len() {
            return Err(SysError::InvalidPlan("dense bucket maps outside run roots"));
        }
        if run_index != u32::MAX {
            if run_index as usize != expected_run_index {
                return Err(SysError::InvalidPlan(
                    "dense buckets do not reference every run exactly once in order",
                ));
            }
            expected_run_index += 1;
        }
    }
    if expected_run_index != plan.runs.len() {
        return Err(SysError::InvalidPlan("a bucket run is absent from the dense bucket map"));
    }
    let mut expected_dense_offset = 0usize;
    for (window_index, window) in plan.windows.iter().enumerate() {
        if window.reserved != 0 || window.bucket_count == 0 {
            return Err(SysError::InvalidPlan("window descriptor is malformed"));
        }
        let offset = usize::try_from(window.dense_map_offset)
            .map_err(|_| SysError::InvalidPlan("window dense offset does not fit usize"))?;
        if offset != expected_dense_offset || window.window_point_index as usize != window_index {
            return Err(SysError::InvalidPlan(
                "window descriptors do not exactly partition dense buckets",
            ));
        }
        let end = offset
            .checked_add(window.bucket_count as usize)
            .ok_or(SysError::InvalidPlan("window dense range overflows"))?;
        if end > plan.dense_map.len()
            || window.window_point_index as usize >= plan.window_point_count
        {
            return Err(SysError::InvalidPlan("window descriptor exceeds plan bounds"));
        }
        expected_dense_offset = end;
    }
    if expected_dense_offset != plan.dense_map.len() {
        return Err(SysError::InvalidPlan(
            "window descriptors do not consume the dense bucket map",
        ));
    }
    let mut expected_window_offset = 0usize;
    for (collect_index, collect) in plan.collects.iter().enumerate() {
        if !matches!(collect.window_bits, 6..=10) || collect.window_count == 0 {
            return Err(SysError::InvalidPlan("collection descriptor is malformed"));
        }
        let expected_window_count = 255_u32 / collect.window_bits + 1;
        if collect.window_count != expected_window_count {
            return Err(SysError::InvalidPlan(
                "collection window count does not match its window width",
            ));
        }
        let offset = usize::try_from(collect.window_point_offset)
            .map_err(|_| SysError::InvalidPlan("collection offset does not fit usize"))?;
        if offset != expected_window_offset || collect.output_index as usize != collect_index {
            return Err(SysError::InvalidPlan(
                "collection descriptors do not exactly partition windows and outputs",
            ));
        }
        let end = offset
            .checked_add(collect.window_count as usize)
            .ok_or(SysError::InvalidPlan("collection window range overflows"))?;
        if end > plan.window_point_count || collect.output_index as usize >= plan.output_count {
            return Err(SysError::InvalidPlan("collection descriptor exceeds plan bounds"));
        }
        let expected_bucket_count = 1_u32 << (collect.window_bits - 1);
        if plan.windows[offset..end]
            .iter()
            .any(|window| window.bucket_count != expected_bucket_count)
        {
            return Err(SysError::InvalidPlan(
                "window bucket count does not match its collection width",
            ));
        }
        expected_window_offset = end;
    }
    if expected_window_offset != plan.window_point_count {
        return Err(SysError::InvalidPlan(
            "collection descriptors do not cover every window and output",
        ));
    }
    checked_u32(plan.records.len(), "routed record count")?;
    checked_u32(plan.affine_records.len(), "affine record count")?;
    checked_u32(plan.affine_destination_map.len(), "affine destination-map count")?;
    checked_u32(plan.runs.len(), "bucket run count")?;
    checked_u32(plan.tiles.len(), "bucket tile count")?;
    checked_u32(plan.tile_runs.len(), "tile run count")?;
    checked_u32(plan.dense_map.len(), "dense bucket count")?;
    checked_u32(plan.windows.len(), "window descriptor count")?;
    checked_u32(plan.window_point_count, "window point count")?;
    checked_u32(plan.collects.len(), "collect descriptor count")?;
    checked_u32(plan.output_count, "MSM output count")?;
    Ok(())
}

fn validate_affine_execution_plan(
    base_count: usize,
    plan: &ExecutionPlan<'_>,
) -> Result<(), SysError> {
    if plan.affine_records.len() != plan.records.len() {
        return Err(SysError::InvalidPlan("batch-affine record permutation has the wrong length"));
    }
    for record in plan.affine_records {
        if record.point_index as usize >= base_count
            || record.flags & !super::abi::SIGNED_RECORD_NEGATE != 0
        {
            return Err(SysError::InvalidPlan("batch-affine signed record is malformed"));
        }
    }
    if plan.affine_records.is_empty() {
        if !plan.runs.is_empty()
            || !plan.affine_levels.is_empty()
            || !plan.affine_destination_map.is_empty()
            || plan.affine_max_pairs != 0
            || plan.affine_max_tree_slots != 0
        {
            return Err(SysError::InvalidPlan("empty batch-affine input has nonempty topology"));
        }
        return Ok(());
    }
    if plan.runs.is_empty() {
        return Err(SysError::InvalidPlan("nonempty batch-affine input has no bucket roots"));
    }

    let mut expected_input = plan.affine_records.len();
    let mut expected_map_offset = 0usize;
    let mut maximum_pairs = 0usize;
    let mut maximum_tree_slots = 0usize;
    for level in plan.affine_levels {
        let pair_count = level.pair_count as usize;
        let carry_count = level.carry_count as usize;
        let destination_count = level.destination_count as usize;
        let source_count = pair_count
            .checked_mul(2)
            .and_then(|paired| paired.checked_add(carry_count))
            .ok_or(SysError::InvalidPlan("batch-affine level source count overflows"))?;
        if level.input_count as usize != expected_input
            || pair_count == 0
            || source_count != expected_input
            || destination_count != pair_count + carry_count
            || level.destination_map_offset as usize != expected_map_offset
        {
            return Err(SysError::InvalidPlan("batch-affine level topology is inconsistent"));
        }
        let map_end = expected_map_offset
            .checked_add(destination_count)
            .ok_or(SysError::InvalidPlan("batch-affine destination-map range overflows"))?;
        let level_map = plan
            .affine_destination_map
            .get(expected_map_offset..map_end)
            .ok_or(SysError::InvalidPlan("batch-affine destination-map range is missing"))?;
        if level_map.iter().any(|&destination| destination as usize >= destination_count) {
            return Err(SysError::InvalidPlan(
                "batch-affine destination-map value is out of range",
            ));
        }
        let tree = product_tree_encoding(level.pair_count)?;
        maximum_pairs = maximum_pairs.max(pair_count);
        maximum_tree_slots = maximum_tree_slots.max(tree.slots);
        expected_input = destination_count;
        expected_map_offset = map_end;
    }
    if expected_input != plan.runs.len() {
        return Err(SysError::InvalidPlan("batch-affine terminal roots do not match bucket runs"));
    }
    if expected_map_offset != plan.affine_destination_map.len() {
        return Err(SysError::InvalidPlan(
            "batch-affine levels do not consume the destination map",
        ));
    }
    if maximum_pairs != plan.affine_max_pairs || maximum_tree_slots != plan.affine_max_tree_slots {
        return Err(SysError::InvalidPlan("batch-affine scratch bounds do not match the topology"));
    }
    Ok(())
}

fn load_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
) -> Result<Pipeline, SysError> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or(SysError::MissingFunction(name))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| SysError::Pipeline { name, reason: format!("{error:?}") })
}

fn retained_command(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    label: &str,
) -> Result<Command, SysError> {
    let command = queue.commandBuffer().ok_or(SysError::CommandBuffer)?;
    if !command.retainedReferences() {
        return Err(SysError::Execution(
            "standalone MSM requires retained-reference command buffers".into(),
        ));
    }
    command.setLabel(Some(&NSString::from_str(label)));
    Ok(command)
}

fn submit_and_wait(command: Command) -> Result<(), SysError> {
    command.commit();
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        return Err(SysError::Execution(format!("{error:?}")));
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(SysError::Execution(format!("unexpected completion status {status:?}")));
    }
    Ok(())
}

#[cfg(feature = "metal-diagnostics")]
fn new_fixed_route_phase_samples(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<CounterSampleBuffer> {
    if !fixed_phase_trace_enabled()
        || !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
    {
        return None;
    }
    let counter_sets = device.counterSets()?;
    // SAFETY: this is an immutable Metal framework string constant.
    let timestamp_name = unsafe { MTLCommonCounterSetTimestamp };
    let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
    for index in 0..counter_sets.count() {
        let candidate = counter_sets.objectAtIndex(index);
        let candidate_name = candidate.name();
        if &*candidate_name == timestamp_name {
            timestamp_set = Some(candidate);
            break;
        }
    }
    let timestamp_set = timestamp_set?;
    let descriptor = MTLCounterSampleBufferDescriptor::new();
    descriptor.setCounterSet(Some(&timestamp_set));
    descriptor.setLabel(&NSString::from_str("bls12-381-metal-msm.fixed-base-v10.phases"));
    descriptor.setStorageMode(MTLStorageMode::Shared);
    // SAFETY: each profiled compute pass records one start/end pair.
    unsafe { descriptor.setSampleCount(FIXED_ROUTE_PHASES_V10.len() * 2) };
    device.newCounterSampleBufferWithDescriptor_error(&descriptor).ok()
}

#[cfg(feature = "metal-diagnostics")]
fn fixed_route_phase_encoder(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
    phase: usize,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, SysError> {
    let Some(samples) = samples else {
        return command.computeCommandEncoder().ok_or(SysError::Encoder("fixed-base MSM v10"));
    };
    let descriptor = MTLComputePassDescriptor::new();
    // SAFETY: Metal compute passes expose one counter attachment and the
    // sample buffer contains two slots for every phase.
    let attachment = unsafe { descriptor.sampleBufferAttachments().objectAtIndexedSubscript(0) };
    attachment.setSampleBuffer(Some(samples));
    unsafe {
        attachment.setStartOfEncoderSampleIndex(phase * 2);
        attachment.setEndOfEncoderSampleIndex(phase * 2 + 1);
    }
    command
        .computeCommandEncoderWithDescriptor(&descriptor)
        .ok_or(SysError::Encoder("profiled fixed-base MSM v10"))
}

#[cfg(feature = "metal-diagnostics")]
fn advance_fixed_route_phase(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    encoder: &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
    next_phase: usize,
) -> Result<(), SysError> {
    if samples.is_none() {
        return Ok(());
    }
    encoder.endEncoding();
    *encoder = fixed_route_phase_encoder(command, samples, next_phase)?;
    Ok(())
}

#[cfg(feature = "metal-diagnostics")]
fn trace_fixed_route_phases(
    samples: Option<&ProtocolObject<dyn MTLCounterSampleBuffer>>,
    gpu_elapsed: Option<Duration>,
) {
    if !fixed_phase_trace_enabled() {
        return;
    }
    let Some(samples) = samples else {
        std::eprintln!(
            "{{\"schema\":\"bls12-381-metal-msm/timing/fixed-phases-v1\",\"available\":false}}"
        );
        return;
    };
    // SAFETY: the command buffer has completed and the shared sample buffer is
    // immutable while the returned NSData is inspected.
    let Some(data) =
        (unsafe { samples.resolveCounterRange(NSRange::new(0, FIXED_ROUTE_PHASES_V10.len() * 2)) })
    else {
        return;
    };
    let Some(timestamps) = fixed_route_timestamps(&data) else {
        return;
    };
    let total_ticks = (0..FIXED_ROUTE_PHASES_V10.len())
        .map(|index| timestamps[index * 2 + 1].saturating_sub(timestamps[index * 2]))
        .sum::<u64>();
    if total_ticks == 0 {
        return;
    }
    let gpu_ms = gpu_elapsed.map(milliseconds);
    let phases = FIXED_ROUTE_PHASES_V10
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let ticks = timestamps[index * 2 + 1].saturating_sub(timestamps[index * 2]);
            let calibrated_ms = gpu_ms.map(|total| total * ticks as f64 / total_ticks as f64);
            format!(
                "\"{name}\":{{\"ticks\":{ticks},\"calibrated_ms\":{}}}",
                calibrated_ms
                    .map_or_else(|| String::from("null"), |duration| format!("{duration:.6}"),)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    std::eprintln!(
        "{{\"schema\":\"bls12-381-metal-msm/timing/fixed-phases-v1\",\"available\":true,\"tick_sum\":{total_ticks},\"gpu_ms\":{},\"phases\":{{{phases}}}}}",
        gpu_ms.map_or_else(|| String::from("null"), |duration| format!("{duration:.6}")),
    );
}

#[cfg(feature = "metal-diagnostics")]
fn fixed_route_timestamps(data: &NSData) -> Option<Vec<u64>> {
    // SAFETY: NSData is immutable for this scope; values are copied before it is released.
    let bytes = unsafe { data.as_bytes_unchecked() };
    let count = FIXED_ROUTE_PHASES_V10.len() * 2;
    if bytes.len() < count * size_of::<u64>() {
        return None;
    }
    let timestamps = bytes
        .chunks_exact(size_of::<u64>())
        .take(count)
        .map(|bytes| u64::from_ne_bytes(bytes.try_into().expect("counter timestamp width")))
        .collect::<Vec<_>>();
    (!timestamps.iter().any(|timestamp| *timestamp == u64::MAX)).then_some(timestamps)
}

#[cfg(feature = "metal-diagnostics")]
fn trace_compute_timing(encoding: Duration, wait: Duration, gpu: Option<Duration>) {
    if !timing_trace_enabled() {
        return;
    }
    std::eprintln!(
        "{{\"schema\":\"bls12-381-metal-msm/timing/compute-v1\",\"encoding_ms\":{:.6},\"wait_ms\":{:.6},\"gpu_ms\":{}}}",
        milliseconds(encoding),
        milliseconds(wait),
        gpu.map_or_else(
            || String::from("null"),
            |duration| format!("{:.6}", milliseconds(duration)),
        )
    );
}

#[cfg(feature = "metal-diagnostics")]
fn trace_request_timing(
    allocation: Duration,
    compute: Duration,
    read: Duration,
    erasure: Duration,
    total: Duration,
) {
    if !timing_trace_enabled() {
        return;
    }
    std::eprintln!(
        "{{\"schema\":\"bls12-381-metal-msm/timing/request-v1\",\"allocation_ms\":{:.6},\"compute_ms\":{:.6},\"read_ms\":{:.6},\"erasure_ms\":{:.6},\"total_ms\":{:.6}}}",
        milliseconds(allocation),
        milliseconds(compute),
        milliseconds(read),
        milliseconds(erasure),
        milliseconds(total)
    );
}

#[cfg(feature = "metal-diagnostics")]
fn timing_trace_enabled() -> bool {
    std::env::var_os("BLS12_381_METAL_MSM_TRACE_TIMING").is_some()
}

#[cfg(feature = "metal-diagnostics")]
fn fixed_phase_trace_enabled() -> bool {
    std::env::var_os("BLS12_381_METAL_MSM_TRACE_PHASES").is_some()
}

#[cfg(feature = "metal-diagnostics")]
fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(feature = "metal-diagnostics")]
fn completed_gpu_duration(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Option<Duration> {
    // SAFETY: these read-only selectors are part of MTLCommandBuffer and are
    // queried only after synchronous completion.
    let start: f64 = unsafe { objc2::msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { objc2::msg_send![command, GPUEndTime] };
    (start.is_finite() && end.is_finite() && end >= start && start > 0.0)
        .then(|| Duration::from_secs_f64(end - start))
}

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    elements: usize,
) {
    debug_assert!(elements != 0);
    let width = elements.min(pipeline.maxTotalThreadsPerThreadgroup()).max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: elements, height: 1, depth: 1 },
        MTLSize { width, height: 1, depth: 1 },
    );
}

fn dispatch_3d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
    height: usize,
    depth: usize,
) {
    debug_assert!(width != 0 && height != 0 && depth != 0);
    let threads = width.min(pipeline.maxTotalThreadsPerThreadgroup()).max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width, height, depth },
        MTLSize { width: threads, height: 1, depth: 1 },
    );
}

fn dispatch_threadgroups(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: usize,
    threads: usize,
) {
    debug_assert!(groups != 0 && threads != 0);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: groups, height: 1, depth: 1 },
        MTLSize { width: threads, height: 1, depth: 1 },
    );
}

fn gpu_route_tiled_xyzz_capacity(
    strategy: MetalMsmReductionStrategy,
    max_record_count: u32,
    dense_bucket_count: u32,
) -> Result<u32, SysError> {
    if strategy == MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 {
        // For bucket populations c_i, sum ceil(c_i / 128) is bounded by
        // ceil(sum(c_i) / 128) plus the number of non-empty buckets.
        let capacity = (max_record_count as usize)
            .div_ceil(GPU_ROUTE_BATCH_AFFINE_XYZZ_TILE_RECORDS)
            .checked_add(dense_bucket_count as usize)
            .ok_or(SysError::LengthOverflow("batch-affine XYZZ tile capacity"))?
            .max(1);
        return u32::try_from(capacity)
            .map_err(|_| SysError::LengthOverflow("batch-affine XYZZ tile capacity"));
    }
    let divisor = match strategy {
        MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
        | MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7 => {
            GPU_ROUTE_TILED_XYZZ_CAPACITY_DIVISOR
        }
        MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8 => {
            GPU_ROUTE_FINE_TILED_XYZZ_CAPACITY_DIVISOR
        }
        _ => return Err(SysError::InvalidPlan("non-tiled strategy requested tile capacity")),
    };
    // Both schedules tile only above twice their divisor and use tiles of that
    // same size. Therefore ceil(bucket/divisor/2) <= bucket/divisor, summed
    // over every tiled bucket, without a data-dependent host readback.
    let capacity = (max_record_count as usize).div_ceil(divisor).max(1);
    u32::try_from(capacity).map_err(|_| SysError::LengthOverflow("tiled XYZZ capacity"))
}

fn fixed_route_tiled_capacity(max_record_count: u32) -> Result<u32, SysError> {
    // Each tiled bucket contributes one rounding term and contains more than
    // the threshold, bounding their count without a histogram readback.
    let records = max_record_count as usize;
    let capacity = records
        .div_ceil(FIXED_ROUTE_TILE_RECORDS_V10 as usize)
        .checked_add(records / (FIXED_ROUTE_LONG_BUCKET_RECORDS_V10 as usize + 1))
        .ok_or(SysError::LengthOverflow("fixed-route tile capacity"))?
        .max(1);
    u32::try_from(capacity).map_err(|_| SysError::LengthOverflow("fixed-route tile capacity"))
}

fn new_shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    label: &'static str,
) -> Result<Buffer, SysError> {
    let raw = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(SysError::Allocation { label, bytes })?;
    raw.setLabel(Some(&NSString::from_str(label)));
    Ok(raw)
}

fn buffer_from_slice<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    values: &[T],
    label: &'static str,
) -> Result<Buffer, SysError> {
    let bytes = size_of_val(values);
    if bytes == 0 {
        return Err(SysError::InvalidPlan("a persistent allocation must not be empty"));
    }
    let raw = new_shared_buffer(device, bytes, label)?;
    copy_values(&raw, values);
    Ok(raw)
}

fn buffer_from_slice_or_word<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    values: &[T],
    label: &'static str,
) -> Result<Buffer, SysError> {
    if values.is_empty() {
        return zeroed_buffer(device, size_of::<u32>(), label);
    }
    buffer_from_slice(device, values, label)
}

fn zeroed_buffer_for_values<T>(
    device: &ProtocolObject<dyn MTLDevice>,
    len: usize,
    label: &'static str,
) -> Result<Buffer, SysError> {
    let bytes = len.checked_mul(size_of::<T>()).ok_or(SysError::LengthOverflow(label))?;
    if bytes == 0 {
        return Err(SysError::InvalidPlan("zero-length device output"));
    }
    zeroed_buffer(device, bytes, label)
}

fn zeroed_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    label: &'static str,
) -> Result<Buffer, SysError> {
    let raw = new_shared_buffer(device, bytes, label)?;
    // SAFETY: this is a fresh shared allocation with at least `bytes` CPU-visible bytes.
    unsafe { std::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, bytes) };
    Ok(raw)
}

fn copy_values<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) {
    // SAFETY: every caller allocates exactly `size_of_val(values)` shared bytes;
    // all types crossing this boundary are checked repr(C) integer PODs.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            size_of_val(values),
        )
    };
}

fn read_one<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>) -> T {
    // SAFETY: the completed device command initialized a correctly sized shared
    // integer POD allocation and Metal mappings satisfy the POD alignment.
    unsafe { buffer.contents().as_ptr().cast::<T>().read() }
}

fn read_values<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
    let mut values = Vec::<T>::with_capacity(len);
    // SAFETY: the completed device command initialized `len` POD values, the
    // allocation is sized for them, and `set_len` follows the complete copy.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<T>(),
            values.as_mut_ptr(),
            len,
        );
        values.set_len(len);
    }
    values
}

fn bind_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    index: usize,
) {
    // SAFETY: the retained-reference command buffer retains this zero-offset
    // allocation through synchronous completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
}

fn bind_buffer_offset(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    index: usize,
) {
    // SAFETY: resident views validate this offset and extent at construction;
    // the retained caller buffer remains live through synchronous completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), byte_offset, index) };
}

fn bind_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    // SAFETY: `value` is a fully initialized repr(C) integer parameter block
    // copied by Metal during command encoding.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(value).cast::<c_void>(),
            size_of::<T>(),
            index,
        )
    };
}

fn checked_u32(value: usize, label: &'static str) -> Result<u32, SysError> {
    u32::try_from(value).map_err(|_| SysError::LengthOverflow(label))
}

const EXPECTED_ABI: [u64; 68] = [
    1,
    0xd599_f628_1415_d2c5,
    48,
    8,
    96,
    8,
    0,
    48,
    144,
    8,
    0,
    48,
    96,
    144,
    8,
    0,
    48,
    96,
    8,
    4,
    0,
    4,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    16,
    4,
    0,
    4,
    8,
    12,
    0xa393_9e5a_6148_f9b7,
    u32::MAX as u64,
    1,
    0xff,
];

const EXPECTED_GPU_ROUTE_ABI_V4: [u64; 16] =
    [4, 32, 8, 32, 4, 16, 64, 4, 4, 16, 20, 32, 36, 40, 44, 48];
const EXPECTED_FIXED_ROUTE_ABI_V10: [u64; 18] = [
    10,
    32,
    8,
    32,
    4,
    16,
    64,
    4,
    4,
    8,
    20,
    24,
    36,
    40,
    56,
    60,
    FIXED_ROUTE_TILE_RECORDS_V10 as u64,
    FIXED_ROUTE_LONG_BUCKET_RECORDS_V10 as u64,
];
const EXPECTED_RESIDENT_SCALAR_ABI_V1: [u64; 13] = [
    1,
    32,
    8,
    16,
    4,
    4,
    8,
    12,
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
    0xffff_fffe_ffff_ffff,
];
const EXPECTED_XYZZ_ABI_V4: [u64; 3] = [4, 192, 8];

fn validate_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), SysError> {
    let output = zeroed_buffer_for_values::<u64>(device, EXPECTED_ABI.len(), "msm.abi")?;
    let command = retained_command(queue, "bls12-381-metal-msm.abi")?;
    let encoder = command.computeCommandEncoder().ok_or(SysError::Encoder("ABI"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    dispatch(&encoder, pipeline, 1);
    encoder.endEncoding();
    submit_and_wait(command)?;
    let actual = read_values::<u64>(&output, EXPECTED_ABI.len());
    if actual != EXPECTED_ABI {
        return Err(SysError::Abi(format!(
            "expected {EXPECTED_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_gpu_route_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), SysError> {
    let output = zeroed_buffer_for_values::<u64>(
        device,
        EXPECTED_GPU_ROUTE_ABI_V4.len(),
        "msm.gpu-route-v4.abi",
    )?;
    let command = retained_command(queue, "bls12-381-metal-msm.gpu-route-v4.abi")?;
    let encoder = command.computeCommandEncoder().ok_or(SysError::Encoder("GPU route ABI"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    dispatch(&encoder, pipeline, 1);
    encoder.endEncoding();
    submit_and_wait(command)?;
    let actual = read_values::<u64>(&output, EXPECTED_GPU_ROUTE_ABI_V4.len());
    if actual != EXPECTED_GPU_ROUTE_ABI_V4 {
        return Err(SysError::Abi(format!(
            "expected GPU route v4 {EXPECTED_GPU_ROUTE_ABI_V4:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_fixed_route_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), SysError> {
    let output = zeroed_buffer_for_values::<u64>(
        device,
        EXPECTED_FIXED_ROUTE_ABI_V10.len(),
        "msm.fixed-route-v10.abi",
    )?;
    let command = retained_command(queue, "bls12-381-metal-msm.fixed-route-v10.abi")?;
    let encoder = command.computeCommandEncoder().ok_or(SysError::Encoder("fixed route ABI"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    dispatch(&encoder, pipeline, 1);
    encoder.endEncoding();
    submit_and_wait(command)?;
    let actual = read_values::<u64>(&output, EXPECTED_FIXED_ROUTE_ABI_V10.len());
    if actual != EXPECTED_FIXED_ROUTE_ABI_V10 {
        return Err(SysError::Abi(format!(
            "expected fixed route v10 {EXPECTED_FIXED_ROUTE_ABI_V10:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_resident_scalar_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), SysError> {
    let output = zeroed_buffer_for_values::<u64>(
        device,
        EXPECTED_RESIDENT_SCALAR_ABI_V1.len(),
        "msm.resident-scalar-v1.abi",
    )?;
    let command = retained_command(queue, "bls12-381-metal-msm.resident-scalar-v1.abi")?;
    let encoder =
        command.computeCommandEncoder().ok_or(SysError::Encoder("resident scalar ABI"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    dispatch(&encoder, pipeline, 1);
    encoder.endEncoding();
    submit_and_wait(command)?;
    let actual = read_values::<u64>(&output, EXPECTED_RESIDENT_SCALAR_ABI_V1.len());
    if actual != EXPECTED_RESIDENT_SCALAR_ABI_V1 {
        return Err(SysError::Abi(format!(
            "expected resident scalar v1 {EXPECTED_RESIDENT_SCALAR_ABI_V1:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_xyzz_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), SysError> {
    let output =
        zeroed_buffer_for_values::<u64>(device, EXPECTED_XYZZ_ABI_V4.len(), "msm.xyzz-v4.abi")?;
    let command = retained_command(queue, "bls12-381-metal-msm.xyzz-v4.abi")?;
    let encoder = command.computeCommandEncoder().ok_or(SysError::Encoder("XYZZ ABI"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    dispatch(&encoder, pipeline, 1);
    encoder.endEncoding();
    submit_and_wait(command)?;
    let actual = read_values::<u64>(&output, EXPECTED_XYZZ_ABI_V4.len());
    if actual != EXPECTED_XYZZ_ABI_V4 {
        return Err(SysError::Abi(format!(
            "expected XYZZ v4 {EXPECTED_XYZZ_ABI_V4:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::abi::EMPTY_BUCKET;
    use super::*;
    use std::vec;

    #[test]
    fn plan_validation_rejects_invalid_record_flags_before_submission() {
        let records = [SignedRecord { point_index: 0, flags: 2 }];
        let runs = [BucketRun { record_offset: 0, record_count: 1, reserved0: 0, reserved1: 0 }];
        let tiles = [BucketTileV1 { record_offset: 0, record_count: 1, run_index: 0, reserved: 0 }];
        let tile_runs = [TileRunV1 { tile_offset: 0, tile_count: 1, reserved0: 0, reserved1: 0 }];
        let mut dense_map = vec![EMPTY_BUCKET; 32 * 128];
        dense_map[0] = 0;
        let windows = windows(0, 0, 32, 128);
        let collects = [CollectDescriptor {
            window_point_offset: 0,
            window_count: 32,
            window_bits: 8,
            output_index: 0,
        }];
        let plan = ExecutionPlan {
            reduction_strategy: MetalMsmReductionStrategy::GlobalBatchAffineV1,
            parallel_threads: PARALLEL_BUCKET_THREADS,
            parallel_tile_records: PARALLEL_BUCKET_TILE_RECORDS,
            records: &records,
            affine_records: &records,
            affine_levels: &[],
            affine_destination_map: &[],
            affine_max_pairs: 0,
            affine_max_tree_slots: 0,
            runs: &runs,
            tiles: &tiles,
            tile_runs: &tile_runs,
            dense_map: &dense_map,
            windows: &windows,
            window_point_count: windows.len(),
            collects: &collects,
            output_count: 1,
        };

        assert!(matches!(
            validate_execution_plan(1, &plan),
            Err(SysError::InvalidPlan("signed record is malformed"))
        ));
    }

    #[test]
    fn plan_validation_accepts_mixed_window_profiles() {
        let first_dense_count = 32 * 128;
        let second_dense_count = 26 * 512;
        let dense_map = vec![EMPTY_BUCKET; first_dense_count + second_dense_count];
        let mut window_descriptors = windows(0, 0, 32, 128);
        window_descriptors.extend(windows(first_dense_count, 32, 26, 512));
        let collects = [
            CollectDescriptor {
                window_point_offset: 0,
                window_count: 32,
                window_bits: 8,
                output_index: 0,
            },
            CollectDescriptor {
                window_point_offset: 32,
                window_count: 26,
                window_bits: 10,
                output_index: 1,
            },
        ];
        let plan = ExecutionPlan {
            reduction_strategy: MetalMsmReductionStrategy::GlobalBatchAffineV1,
            parallel_threads: PARALLEL_BUCKET_THREADS,
            parallel_tile_records: PARALLEL_BUCKET_TILE_RECORDS,
            records: &[],
            affine_records: &[],
            affine_levels: &[],
            affine_destination_map: &[],
            affine_max_pairs: 0,
            affine_max_tree_slots: 0,
            runs: &[],
            tiles: &[],
            tile_runs: &[],
            dense_map: &dense_map,
            windows: &window_descriptors,
            window_point_count: window_descriptors.len(),
            collects: &collects,
            output_count: 2,
        };

        validate_execution_plan(1, &plan).unwrap();
    }

    fn windows(
        first_dense_offset: usize,
        first_window_index: usize,
        count: usize,
        buckets: usize,
    ) -> Vec<WindowDescriptor> {
        (0..count)
            .map(|index| WindowDescriptor {
                dense_map_offset: u32::try_from(first_dense_offset + index * buckets).unwrap(),
                bucket_count: u32::try_from(buckets).unwrap(),
                window_point_index: u32::try_from(first_window_index + index).unwrap(),
                reserved: 0,
            })
            .collect()
    }
}
