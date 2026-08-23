// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Rust side of the versioned standalone MSM shader ABI.

/// Raw Montgomery Fp coordinate used by the Metal kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuFp {
    pub(super) limbs: [u64; 6],
}

/// Raw affine base used by prepared-base buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuAffine {
    pub(super) x: GpuFp,
    pub(super) y: GpuFp,
}

/// Compact BLST-compatible Jacobian output.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuJacobian {
    pub(super) x: GpuFp,
    pub(super) y: GpuFp,
    pub(super) z: GpuFp,
}

/// Internal XYZZ point used only to size Metal scratch allocations.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuXyzz {
    pub(super) x: GpuFp,
    pub(super) y: GpuFp,
    pub(super) zz: GpuFp,
    pub(super) zzz: GpuFp,
}

/// One routed signed base reference. Only bit zero of `flags` is defined.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SignedRecord {
    pub(super) point_index: u32,
    pub(super) flags: u32,
}

/// Canonical little-endian scalar uploaded for Metal-side Booth recoding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuScalar {
    pub(super) limbs: [u64; 4],
}

/// One independently validated ragged job for GPU routing ABI version 3.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuRouteJobV3 {
    pub(super) base_offset: u32,
    pub(super) scalar_offset: u32,
    pub(super) term_count: u32,
    pub(super) output_index: u32,
    pub(super) record_segment_offset: u32,
    pub(super) reserved0: u32,
    pub(super) reserved1: u32,
    pub(super) reserved2: u32,
}

/// Bounds and topology for GPU recoding, routing, and dense reduction v3.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GpuRouteParametersV3 {
    pub(super) abi_version: u32,
    pub(super) base_count: u32,
    pub(super) scalar_count: u32,
    pub(super) job_count: u32,
    pub(super) output_count: u32,
    pub(super) window_bits: u32,
    pub(super) windows_per_scalar: u32,
    pub(super) buckets_per_window: u32,
    pub(super) dense_bucket_count: u32,
    pub(super) max_record_count: u32,
    pub(super) max_terms_per_job: u32,
    pub(super) threads_per_group: u32,
    pub(super) buckets_per_group: u32,
    pub(super) reserved0: u32,
    pub(super) reserved1: u32,
    pub(super) reserved2: u32,
}

/// Bounds and topology for pre-shifted fixed-base routing ABI version 10.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FixedRouteParametersV10 {
    pub(super) abi_version: u32,
    pub(super) base_count: u32,
    pub(super) shifted_base_count: u32,
    pub(super) scalar_count: u32,
    pub(super) job_count: u32,
    pub(super) output_count: u32,
    pub(super) window_bits: u32,
    pub(super) windows_per_scalar: u32,
    pub(super) buckets_per_output: u32,
    pub(super) dense_bucket_count: u32,
    pub(super) max_record_count: u32,
    pub(super) max_terms_per_job: u32,
    pub(super) threads_per_group: u32,
    pub(super) buckets_per_group: u32,
    pub(super) integration_block_buckets: u32,
    pub(super) integration_blocks_per_output: u32,
}

/// One checked copy from an external Montgomery-scalar view into canonical scratch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ResidentScalarCopyV1 {
    pub(super) source_count: u32,
    pub(super) destination_offset: u32,
    pub(super) destination_capacity: u32,
    pub(super) reserved: u32,
}

/// One contiguous run of records reduced into one homogeneous point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BucketRun {
    pub(super) record_offset: u32,
    pub(super) record_count: u32,
    pub(super) reserved0: u32,
    pub(super) reserved1: u32,
}

/// One contiguous record tile reduced by a 128-thread threadgroup.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BucketTileV1 {
    pub(super) record_offset: u32,
    pub(super) record_count: u32,
    pub(super) run_index: u32,
    pub(super) reserved: u32,
}

/// Contiguous tile span belonging to one dense bucket run.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct TileRunV1 {
    pub(super) tile_offset: u32,
    pub(super) tile_count: u32,
    pub(super) reserved0: u32,
    pub(super) reserved1: u32,
}

/// Bounds and frozen launch topology for parallel bucket reduction v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ParallelBucketParametersV1 {
    pub(super) abi_version: u32,
    pub(super) base_count: u32,
    pub(super) record_count: u32,
    pub(super) tile_count: u32,
    pub(super) run_count: u32,
    pub(super) threads_per_threadgroup: u32,
    pub(super) max_records_per_tile: u32,
    pub(super) reserved: u32,
}

/// Bounds for gathering preordered signed affine leaves.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BatchAffineGatherParameters {
    pub(super) record_offset: u32,
    pub(super) record_count: u32,
    pub(super) srs_count: u32,
    pub(super) reserved: u32,
}

/// One balanced affine point-reduction level.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BatchAffineLevelParameters {
    pub(super) destination_map_offset: u32,
    pub(super) pair_count: u32,
    pub(super) carry_count: u32,
    pub(super) destination_len: u32,
}

/// One product-tree reduction or inverse-expansion pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BatchAffineProductParameters {
    pub(super) input_offset: u32,
    pub(super) input_count: u32,
    pub(super) output_offset: u32,
    pub(super) output_count: u32,
}

/// Atomic status words written by the batch-affine reducer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BatchAffineStatus {
    pub(super) flags: u32,
    pub(super) record_errors: u32,
    pub(super) destination_errors: u32,
    pub(super) reserved: u32,
}

/// Bounds supplied to the serial bucket-reduction kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ReduceParameters {
    pub(super) base_count: u32,
    pub(super) record_count: u32,
    pub(super) run_count: u32,
    pub(super) reserved: u32,
}

/// Dense-bucket segment and destination for one MSM window.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct WindowDescriptor {
    pub(super) dense_map_offset: u32,
    pub(super) bucket_count: u32,
    pub(super) window_point_index: u32,
    pub(super) reserved: u32,
}

/// Bounds supplied to the descending bucket-integration kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct IntegrationParameters {
    pub(super) dense_map_count: u32,
    pub(super) run_count: u32,
    pub(super) window_count: u32,
    pub(super) window_point_count: u32,
}

/// Contiguous low-to-high window span collected into one output.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CollectDescriptor {
    pub(super) window_point_offset: u32,
    pub(super) window_count: u32,
    pub(super) window_bits: u32,
    pub(super) output_index: u32,
}

/// Bounds supplied to the high-to-low window-collection kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CollectParameters {
    pub(super) window_point_count: u32,
    pub(super) descriptor_count: u32,
    pub(super) output_count: u32,
    pub(super) reserved: u32,
}

/// Four atomic words populated by validation in every MSM kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct KernelStatus {
    pub(super) flags: u32,
    pub(super) record_errors: u32,
    pub(super) descriptor_errors: u32,
    pub(super) reserved: u32,
}

pub(super) const SIGNED_RECORD_NEGATE: u32 = 1;
pub(super) const EMPTY_BUCKET: u32 = u32::MAX;
pub(super) const PARALLEL_BUCKET_TILE_RECORDS: u32 = 128;
pub(super) const PARALLEL_JACOBIAN_SMALL_THREADS: usize = 32;
pub(super) const PARALLEL_JACOBIAN_SMALL_TILE_RECORDS: u32 = 128;
pub(super) const PARALLEL_JACOBIAN_LARGE_THREADS: usize = 64;
pub(super) const PARALLEL_JACOBIAN_LARGE_TILE_RECORDS: u32 = 256;
pub(super) const GPU_ROUTED_DENSE_ABI_VERSION_V4: u32 = 4;
pub(super) const FIXED_ROUTE_ABI_VERSION_V10: u32 = 10;
pub(super) const FIXED_ROUTE_THREADS_PER_GROUP_V10: u32 = 64;
pub(super) const FIXED_ROUTE_LONG_THREADS_PER_GROUP_V10: u32 = 32;
pub(super) const FIXED_ROUTE_BUCKETS_PER_GROUP_V10: u32 = 8;
pub(super) const FIXED_ROUTE_BATCHED_BUCKETS_PER_GROUP_V10: u32 = 8;
pub(super) const FIXED_ROUTE_INTEGRATION_BLOCK_BUCKETS_V10: u32 = 64;
pub(super) const FIXED_ROUTE_TILE_RECORDS_V10: u32 = 2048;
pub(super) const FIXED_ROUTE_LONG_BUCKET_RECORDS_V10: u32 = 512;

pub(super) const fn fixed_route_buckets_per_group_v10(window_bits: u32, output_count: u32) -> u32 {
    if window_bits >= 13 && output_count >= 4 {
        FIXED_ROUTE_BATCHED_BUCKETS_PER_GROUP_V10
    } else {
        FIXED_ROUTE_BUCKETS_PER_GROUP_V10
    }
}

pub(super) const fn fixed_route_integration_blocks_per_group_v10(
    window_bits: u32,
    output_count: u32,
) -> u32 {
    if window_bits >= 13 && output_count >= 8 {
        16
    } else if window_bits >= 13 && output_count >= 4 {
        8
    } else {
        4
    }
}

const _: [(); 48] = [(); core::mem::size_of::<GpuFp>()];
const _: [(); 96] = [(); core::mem::size_of::<GpuAffine>()];
const _: [(); 144] = [(); core::mem::size_of::<GpuJacobian>()];
const _: [(); 192] = [(); core::mem::size_of::<GpuXyzz>()];
const _: [(); 8] = [(); core::mem::size_of::<SignedRecord>()];
const _: [(); 32] = [(); core::mem::size_of::<GpuScalar>()];
const _: [(); 32] = [(); core::mem::size_of::<GpuRouteJobV3>()];
const _: [(); 64] = [(); core::mem::size_of::<GpuRouteParametersV3>()];
const _: [(); 64] = [(); core::mem::size_of::<FixedRouteParametersV10>()];
const _: [(); 16] = [(); core::mem::size_of::<BucketRun>()];
const _: [(); 16] = [(); core::mem::size_of::<BucketTileV1>()];
const _: [(); 16] = [(); core::mem::size_of::<TileRunV1>()];
const _: [(); 32] = [(); core::mem::size_of::<ParallelBucketParametersV1>()];
const _: [(); 16] = [(); core::mem::size_of::<BatchAffineGatherParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<BatchAffineLevelParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<BatchAffineProductParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<BatchAffineStatus>()];
const _: [(); 16] = [(); core::mem::size_of::<ReduceParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<WindowDescriptor>()];
const _: [(); 16] = [(); core::mem::size_of::<IntegrationParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<CollectDescriptor>()];
const _: [(); 16] = [(); core::mem::size_of::<CollectParameters>()];
const _: [(); 16] = [(); core::mem::size_of::<KernelStatus>()];
