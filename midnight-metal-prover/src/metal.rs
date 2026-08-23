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

//! Device, queue, pipeline, command, and shared-buffer ownership.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::NonNull;
use std::sync::{
    Arc, Mutex, MutexGuard, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

use bls12_381_metal_msm::ResidentMontgomeryScalars;
use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
#[cfg(feature = "metal-diagnostics")]
use objc2_foundation::NSRange;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
};
#[cfg(feature = "metal-diagnostics")]
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLStorageMode,
};
use sha2::{Digest, Sha256};

use crate::{
    Bls12_381AbiProbe, Bls12_381Fp, Bls12_381Fp64ProbeResult, Bls12_381Fr,
    Bls12_381Fr64ProbeResult, Bls12_381G1Affine, Bls12_381G1ChainProbeResult,
    Bls12_381G1ProbeResult, Bls12_381G1Projective, Bls12_381MsmBasis, Bls12_381MsmWindow,
    GraphAbiProbe, MetalActivity, MetalDeviceInfo, MetalError, MetalGraphDescriptor,
    MetalGraphFamilySections, MetalGraphOperation, MetalGraphValueRef, MetalKeyContextIdentity,
    MetalKeyContextInput, MetalKeyContextInputV2, MetalMultiopenRequest, MetalQuotientCpuMsmLease,
    MetalQuotientEpochRequest, MetalQuotientResidentMsmLease, MetalResidentColumnRef,
    MetalResidentPolynomialMsmLease, MetalStatus, NttDirection, NttDomainSpec, ProofByteDigest,
    generated,
    key_context::{
        ValidatedMetalKeyContextInput,
        v2::{
            ValidatedMetalKeyContextInputV2, prevalidate_metal_key_context_input_v2,
            validate_prevalidated_metal_key_context_srs_subgroups_v2,
        },
        validate_metal_key_context_input,
    },
};

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/midnight-metal.metallib"));
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
const KEY_CONTEXT_WARM_MAX_ENTRIES: usize = 2;
const VALIDATED_SRS_MAX_ENTRIES: usize = 8;

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
#[cfg(feature = "metal-diagnostics")]
type CounterSampleBuffer = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

#[cfg(test)]
#[path = "batch_affine_authentic.rs"]
mod batch_affine_authentic;
#[cfg(test)]
#[path = "batch_affine_benchmark.rs"]
mod batch_affine_benchmark;
#[cfg(test)]
#[path = "batch_affine_ceiling.rs"]
mod batch_affine_ceiling;
#[cfg(test)]
#[path = "batch_affine_manifest.rs"]
mod batch_affine_manifest;
#[cfg(test)]
#[path = "batch_affine_probe.rs"]
mod batch_affine_probe;
#[cfg(test)]
#[path = "batch_affine_replay.rs"]
mod batch_affine_replay;
#[cfg(test)]
#[path = "flat_segmented_authentic.rs"]
mod flat_segmented_authentic;
#[cfg(test)]
#[path = "flat_segmented_benchmark.rs"]
mod flat_segmented_benchmark;
#[cfg(test)]
#[path = "flat_segmented_plan.rs"]
mod flat_segmented_plan;
#[cfg(test)]
#[path = "graph_probe.rs"]
mod graph_probe;
#[cfg(test)]
#[path = "key_context_metal_tests.rs"]
mod key_context_metal_tests;
#[path = "metal/msm_worker.rs"]
mod msm_worker;
#[path = "metal/multiopen.rs"]
mod multiopen;
pub use multiopen::MetalMultiopenArena;
#[cfg(test)]
#[path = "ntt_leaf_ceiling.rs"]
mod ntt_leaf_ceiling;
#[path = "metal/quotient_epoch.rs"]
mod quotient_epoch;
#[path = "metal/specialized_graph.rs"]
mod specialized_graph;
use specialized_graph::SpecializedGraphPipelines;

pub(super) fn record_cpu_msm_partition(logical_scalars: usize, nonzero_scalars: usize) {
    msm_worker::record_cpu_partition(logical_scalars, nonzero_scalars);
}

pub(super) fn record_cpu_msm_timing(lock_wait_ns: u64, service_ns: u64) {
    msm_worker::record_cpu_timing(lock_wait_ns, service_ns);
}

#[cfg(test)]
struct Fp32ChainBenchmark<'a> {
    pipeline: &'a Pipeline,
    variant: &'a str,
    iterations: u32,
    timed_dispatches: u32,
    threadgroup_width: usize,
}

const MAX_NTT_LOG_SIZE: u32 = 31;
const MAX_NTT_RADIX_LOG: u32 = 9;
const NTT_COSET_THREADS: usize = 128;
const BLS12_381_LAYOUT_ABI_WORDS: usize = 23;
const BLS12_381_LAYOUT_ABI_FINGERPRINT: u64 = 0x108c_b973_1d50_97c4;
const NTT_PARAMETER_ABI_VERSION: u64 = 2;
const NTT_PARAMETER_ABI_WORDS: usize = 27;
const NTT_PARAMETER_ABI_FINGERPRINT: u64 = 0x9e3b_3ec6_d4a6_b833;
const GRAPH_ABI_VERSION: u64 = 1;
const GRAPH_ABI_WORDS: usize = 75;
const GRAPH_ABI_FINGERPRINT: u64 = 0x1eeb_a113_151f_9303;
const GRAPH_SCRATCH_CLASSES: [usize; 6] = [8, 16, 32, 64, 128, 256];
const SECRET_BUFFER_POOL_MAX_BYTES: usize = 512 * 1024 * 1024;
#[cfg(test)]
const BATCH_AFFINE_ABI_VERSION: u64 = 1;
#[cfg(test)]
const BATCH_AFFINE_ABI_WORDS: usize = 32;
#[cfg(test)]
const BATCH_AFFINE_CEILING_ABI_VERSION: u64 = 1;
#[cfg(test)]
const BATCH_AFFINE_CEILING_ABI_WORDS: usize = 7;
#[cfg(test)]
const FLAT_SEGMENTED_ABI_VERSION: u64 = 1;
#[cfg(test)]
const FLAT_SEGMENTED_ABI_WORDS: usize = 17;

type GraphValueRefAbi = MetalGraphValueRef;
type GraphOperationAbi = MetalGraphOperation;

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)] // Host mirror is validated before graph dispatch is enabled.
struct GraphEvalParametersAbi {
    len: u32,
    stride: u32,
    rotation_scale: i32,
    post_kind: u32,
    constants_offset: u32,
    constants_count: u32,
    rotations_offset: u32,
    rotations_count: u32,
    operations_offset: u32,
    operations_count: u32,
    parts_offset: u32,
    parts_count: u32,
    fixed_columns: u32,
    advice_offset: u32,
    advice_columns: u32,
    instance_offset: u32,
    instance_columns: u32,
    challenge_count: u32,
    previous_mode: u32,
    lookup_product_column: u32,
    lookup_input_column: u32,
    lookup_table_column: u32,
    trash_column: u32,
    trash_selector_fixed_column: u32,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    theta: Bls12_381Fr,
    trash_challenge: Bls12_381Fr,
    y: Bls12_381Fr,
}

struct Field64Pipelines {
    add: Pipeline,
    sub: Pipeline,
    mul: Pipeline,
    square: Pipeline,
}

struct Fp32Pipelines {
    add: Pipeline,
    sub: Pipeline,
    mul: Pipeline,
    square: Pipeline,
    wide_mul: Pipeline,
    wide_square: Pipeline,
    #[cfg(test)]
    from_public: Pipeline,
    #[cfg(test)]
    to_public: Pipeline,
    #[cfg(test)]
    mul_chain: Pipeline,
    #[cfg(test)]
    wide_mul_chain: Pipeline,
    #[cfg(test)]
    wide_mul_chain_2: Pipeline,
    #[cfg(test)]
    wide_mul_chain_3: Pipeline,
}

struct FieldPipelines {
    abi_probe: Pipeline,
    fr64: Field64Pipelines,
    fp64: Field64Pipelines,
    fp32: Fp32Pipelines,
}

impl FieldPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(device, library, "midnight_bls12_381_abi_probe")?,
            fr64: Field64Pipelines {
                add: pipeline(device, library, "midnight_bls12_381_fr64_add")?,
                sub: pipeline(device, library, "midnight_bls12_381_fr64_sub")?,
                mul: pipeline(device, library, "midnight_bls12_381_fr64_mul")?,
                square: pipeline(device, library, "midnight_bls12_381_fr64_square")?,
            },
            fp64: Field64Pipelines {
                add: pipeline(device, library, "midnight_bls12_381_fp64_add")?,
                sub: pipeline(device, library, "midnight_bls12_381_fp64_sub")?,
                mul: pipeline(device, library, "midnight_bls12_381_fp64_mul")?,
                square: pipeline(device, library, "midnight_bls12_381_fp64_square")?,
            },
            fp32: Fp32Pipelines {
                add: pipeline(device, library, "midnight_bls12_381_fp32_add")?,
                sub: pipeline(device, library, "midnight_bls12_381_fp32_sub")?,
                mul: pipeline(device, library, "midnight_bls12_381_fp32_mul")?,
                square: pipeline(device, library, "midnight_bls12_381_fp32_square")?,
                wide_mul: pipeline(device, library, "midnight_bls12_381_fp32_wide_mul")?,
                wide_square: pipeline(device, library, "midnight_bls12_381_fp32_wide_square")?,
                #[cfg(test)]
                from_public: pipeline(device, library, "midnight_bls12_381_fp32_from_public")?,
                #[cfg(test)]
                to_public: pipeline(device, library, "midnight_bls12_381_fp32_to_public")?,
                #[cfg(test)]
                mul_chain: pipeline(device, library, "midnight_bls12_381_fp32_mul_chain")?,
                #[cfg(test)]
                wide_mul_chain: pipeline(
                    device,
                    library,
                    "midnight_bls12_381_fp32_wide_mul_chain",
                )?,
                #[cfg(test)]
                wide_mul_chain_2: pipeline(
                    device,
                    library,
                    "midnight_bls12_381_fp32_wide_mul_chain_2",
                )?,
                #[cfg(test)]
                wide_mul_chain_3: pipeline(
                    device,
                    library,
                    "midnight_bls12_381_fp32_wide_mul_chain_3",
                )?,
            },
        })
    }
}

struct GraphPipelines {
    abi_probe: Pipeline,
    evaluators: [GraphEvaluatorPipeline; 6],
}

struct GraphEvaluatorPipeline {
    scratch_capacity: usize,
    pipeline: Pipeline,
}

impl GraphPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(device, library, "midnight_bls12_381_graph_abi_probe")?,
            evaluators: [
                GraphEvaluatorPipeline {
                    scratch_capacity: 8,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_8")?,
                },
                GraphEvaluatorPipeline {
                    scratch_capacity: 16,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_16")?,
                },
                GraphEvaluatorPipeline {
                    scratch_capacity: 32,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_32")?,
                },
                GraphEvaluatorPipeline {
                    scratch_capacity: 64,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_64")?,
                },
                GraphEvaluatorPipeline {
                    scratch_capacity: 128,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_128")?,
                },
                GraphEvaluatorPipeline {
                    scratch_capacity: 256,
                    pipeline: pipeline(device, library, "midnight_bls12_381_graph_evaluate_256")?,
                },
            ],
        })
    }

    fn validate_pipeline_limits(&self) -> Result<(), MetalError> {
        for (evaluator, expected_capacity) in self.evaluators.iter().zip(GRAPH_SCRATCH_CLASSES) {
            let execution_width = evaluator.pipeline.threadExecutionWidth();
            let maximum_width = evaluator.pipeline.maxTotalThreadsPerThreadgroup();
            if evaluator.scratch_capacity != expected_capacity
                || execution_width == 0
                || maximum_width < execution_width
            {
                return Err(MetalError::Dispatch(
                    "graph evaluator pipeline reports invalid thread limits",
                ));
            }
        }
        Ok(())
    }

    fn evaluator(&self, scratch_capacity: usize) -> Option<&GraphEvaluatorPipeline> {
        self.evaluators
            .iter()
            .find(|candidate| candidate.scratch_capacity == scratch_capacity)
    }
}

struct QuotientPipelines {
    copy: Pipeline,
    combine_pieces: Pipeline,
    permutation: Pipeline,
    multiply_repeating: Pipeline,
    split_blind: Pipeline,
}

struct MultiopenPipelines {
    lincomb: Pipeline,
    division_summarize: Pipeline,
    division_scan_blocks: Pipeline,
    division_fill: Pipeline,
    combine: Pipeline,
    combine_final: Pipeline,
    eval_summarize: Pipeline,
    eval_finish: Pipeline,
    subtract_evaluation: Pipeline,
}

impl MultiopenPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            lincomb: pipeline(device, library, "midnight_bls12_381_multiopen_lincomb")?,
            division_summarize: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_division_summarize",
            )?,
            division_scan_blocks: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_division_scan_blocks",
            )?,
            division_fill: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_division_fill",
            )?,
            combine: pipeline(device, library, "midnight_bls12_381_multiopen_combine")?,
            combine_final: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_combine_final",
            )?,
            eval_summarize: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_eval_summarize",
            )?,
            eval_finish: pipeline(device, library, "midnight_bls12_381_multiopen_eval_finish")?,
            subtract_evaluation: pipeline(
                device,
                library,
                "midnight_bls12_381_multiopen_subtract_evaluation",
            )?,
        })
    }
}

impl QuotientPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            copy: pipeline(device, library, "midnight_bls12_381_quotient_copy")?,
            combine_pieces: pipeline(
                device,
                library,
                "midnight_bls12_381_quotient_combine_pieces",
            )?,
            permutation: pipeline(
                device,
                library,
                "midnight_bls12_381_quotient_permutation_accumulate",
            )?,
            multiply_repeating: pipeline(
                device,
                library,
                "midnight_bls12_381_quotient_multiply_repeating",
            )?,
            split_blind: pipeline(device, library, "midnight_bls12_381_quotient_split_blind")?,
        })
    }
}

struct G1Pipelines {
    add: Pipeline,
    complete_add_chain: Pipeline,
    double: Pipeline,
    from_homogeneous: Pipeline,
    identity_fill: Pipeline,
    mixed_add: Pipeline,
    to_homogeneous: Pipeline,
    to_affine: Pipeline,
}

impl G1Pipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            add: pipeline(device, library, "midnight_bls12_381_g1_add")?,
            complete_add_chain: pipeline(
                device,
                library,
                "midnight_bls12_381_g1_complete_add_chain",
            )?,
            double: pipeline(device, library, "midnight_bls12_381_g1_double")?,
            from_homogeneous: pipeline(
                device,
                library,
                "midnight_bls12_381_g1_homogeneous_to_jacobian",
            )?,
            identity_fill: pipeline(
                device,
                library,
                "midnight_bls12_381_g1_homogeneous_fill_identity",
            )?,
            mixed_add: pipeline(device, library, "midnight_bls12_381_g1_mixed_add")?,
            to_homogeneous: pipeline(
                device,
                library,
                "midnight_bls12_381_g1_jacobian_to_homogeneous",
            )?,
            to_affine: pipeline(device, library, "midnight_bls12_381_g1_to_affine")?,
        })
    }
}

struct NttPipelines {
    abi_probe: Pipeline,
    stockham: Pipeline,
    identity: Pipeline,
    coset_prepare: Pipeline,
}

impl NttPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(
                device,
                library,
                "midnight_bls12_381_ntt_parameter_abi_probe",
            )?,
            stockham: pipeline(device, library, "midnight_bls12_381_ntt_stockham")?,
            identity: pipeline(device, library, "midnight_bls12_381_ntt_identity")?,
            coset_prepare: pipeline(device, library, "midnight_bls12_381_ntt_coset_prepare")?,
        })
    }
}

#[cfg(test)]
#[allow(dead_code)]
struct BatchAffinePipelines {
    abi_probe: Pipeline,
    gather_signed_leaves: Pipeline,
    prepare_pairs: Pipeline,
    reduce_products: Pipeline,
    invert_product_root: Pipeline,
    expand_product_inverses: Pipeline,
    finish_pairs: Pipeline,
    copy_carries: Pipeline,
    export_roots: Pipeline,
}

#[cfg(test)]
impl BatchAffinePipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(device, library, "midnight_bls12_381_batch_affine_abi_probe")?,
            gather_signed_leaves: pipeline(
                device,
                library,
                "midnight_bls12_381_gather_signed_leaves",
            )?,
            prepare_pairs: pipeline(device, library, "midnight_bls12_381_prepare_pairs")?,
            reduce_products: pipeline(device, library, "midnight_bls12_381_reduce_products")?,
            invert_product_root: pipeline(
                device,
                library,
                "midnight_bls12_381_invert_product_root",
            )?,
            expand_product_inverses: pipeline(
                device,
                library,
                "midnight_bls12_381_expand_product_inverses",
            )?,
            finish_pairs: pipeline(device, library, "midnight_bls12_381_finish_pairs")?,
            copy_carries: pipeline(device, library, "midnight_bls12_381_copy_carries")?,
            export_roots: pipeline(device, library, "midnight_bls12_381_export_roots")?,
        })
    }
}

#[cfg(test)]
struct BatchAffineCeilingPipelines {
    abi_probe: Pipeline,
    execute: Pipeline,
}

#[cfg(test)]
impl BatchAffineCeilingPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(
                device,
                library,
                "midnight_bls12_381_batch_affine_ceiling_abi_probe",
            )?,
            execute: pipeline(
                device,
                library,
                "midnight_bls12_381_batch_affine_arithmetic_ceiling",
            )?,
        })
    }
}

#[cfg(test)]
struct FlatSegmentedPipelines {
    abi_probe: Pipeline,
    reduce: Pipeline,
}

#[cfg(test)]
impl FlatSegmentedPipelines {
    fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            abi_probe: pipeline(
                device,
                library,
                "midnight_bls12_381_flat_segmented_abi_probe",
            )?,
            reduce: pipeline(device, library, "midnight_bls12_381_flat_segmented_reduce")?,
        })
    }
}

#[derive(Default)]
struct ActivityCounters {
    command_buffers: AtomicU64,
    dispatches: AtomicU64,
    host_waits: AtomicU64,
    allocations: AtomicU64,
    allocated_bytes: AtomicU64,
    host_write_bytes: AtomicU64,
    host_read_bytes: AtomicU64,
    blit_bytes: AtomicU64,
    execution_failures: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
    pool_returns: AtomicU64,
    clear_commands: AtomicU64,
    clear_dispatches: AtomicU64,
    clear_waits: AtomicU64,
    cleared_bytes: AtomicU64,
    clear_failures: AtomicU64,
    poison_events: AtomicU64,
    resident_generations: AtomicU64,
    stale_handle_rejections: AtomicU64,
    materializations: AtomicU64,
    whole_polynomial_materializations: AtomicU64,
    secret_live_bytes: AtomicU64,
    secret_peak_live_bytes: AtomicU64,
    pooled_secret_bytes: AtomicU64,
    quarantined_secret_bytes: AtomicU64,
    key_cache_hits: AtomicU64,
    key_cache_misses: AtomicU64,
    key_context_builds: AtomicU64,
    key_context_build_failures: AtomicU64,
    quotient_epochs: AtomicU64,
    quotient_epoch_elapsed_ns: AtomicU64,
    quotient_epoch_failures: AtomicU64,
    quotient_epoch_failure_elapsed_ns: AtomicU64,
    quotient_materializations: AtomicU64,
    quotient_whole_polynomial_materializations: AtomicU64,
    cpu_msm_leases: AtomicU64,
    cpu_msm_lease_bytes: AtomicU64,
    resident_msm_leases: AtomicU64,
    resident_msm_lease_bytes: AtomicU64,
}

#[derive(Clone)]
struct TrackedBuffer {
    raw: Buffer,
    bytes: usize,
}

// SAFETY: Metal resource objects support ownership transfer between threads.
// This runtime serializes every mapping, encode, transition, and materialization
// through `MetalSession::execution`; the unsafe marker only permits the plan
// cache to follow the already-shareable session across callers. Command use
// additionally retains each raw buffer until completion.
unsafe impl Send for TrackedBuffer {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrBasis {
    Coefficient,
    Lagrange,
    ExtendedCoset,
    CosetCoefficient,
    StockhamWorking {
        direction: NttDirection,
        completed_log_p: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FrDomainTag {
    log_size: u32,
    basis: FrBasis,
    ntt_domain: Option<NttDomainSpec>,
}

impl FrDomainTag {
    const fn coefficient(log_size: u32) -> Self {
        Self {
            log_size,
            basis: FrBasis::Coefficient,
            ntt_domain: None,
        }
    }

    const fn for_ntt(domain: NttDomainSpec, basis: FrBasis) -> Self {
        Self {
            log_size: domain.log_size,
            basis,
            ntt_domain: Some(domain),
        }
    }

    fn element_count(self) -> Result<usize, MetalError> {
        1usize
            .checked_shl(self.log_size)
            .ok_or(MetalError::ResidentState(
                "resident Fr domain length overflows usize",
            ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ImmutableBufferSource {
    GraphDescriptors,
    GraphConstants,
    GraphRotations,
    GraphOperations,
    GraphHornerParts,
    FixedMap,
    AdviceMap,
    InstanceMap,
    FixedCosetCompact,
    Selectors,
    PermutationColumns,
    TrashSelectorFixedIndices,
    FixedLagrangeCompact,
    PermutationCosets,
    ExtendedCosetPoints,
    PermutationLagrange,
    CoefficientColumns,
    VanishingInverses,
    CoefficientBases,
    LagrangeBases,
}

/// Write-once, non-secret storage published only after its CPU upload finishes.
struct ImmutableBuffer {
    session_id: u64,
    context_id: u64,
    allocation_id: u64,
    source: ImmutableBufferSource,
    raw: Buffer,
    bytes: usize,
}

// SAFETY: immutable key buffers are mapped and written exactly once before the
// context is published. Raw handles are never exposed, and future command use
// is serialized by the owning session's execution mutex while commands retain
// the Metal resource through completion.
unsafe impl Send for ImmutableBuffer {}
// SAFETY: after publication no host or device operation mutates these buffers;
// concurrent handle access can observe only the immutable `bytes` metadata.
unsafe impl Sync for ImmutableBuffer {}

#[derive(Default)]
struct ImmutableKeyAccounting {
    live_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    live_contexts: AtomicU64,
}

/// Weak identity index plus a byte-bounded strong warm set.
///
/// The weak index deduplicates contexts held by callers. The warm LRU keeps
/// repeated proofs from rebuilding immutable device state after a short-lived
/// handle is dropped.
struct KeyContextCache {
    weak: HashMap<MetalKeyContextIdentity, Weak<MetalKeyContextInner>>,
    warm: VecDeque<(MetalKeyContextIdentity, Arc<MetalKeyContextInner>)>,
    warm_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl KeyContextCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            weak: HashMap::new(),
            warm: VecDeque::new(),
            warm_bytes: 0,
            max_entries: KEY_CONTEXT_WARM_MAX_ENTRIES,
            max_bytes,
        }
    }

    fn get(&mut self, identity: &MetalKeyContextIdentity) -> Option<Arc<MetalKeyContextInner>> {
        self.prune_dead_weak();
        let context = self.weak.get(identity).and_then(Weak::upgrade)?;
        self.touch(identity.clone(), context.clone());
        Some(context)
    }

    fn insert(&mut self, identity: MetalKeyContextIdentity, context: Arc<MetalKeyContextInner>) {
        self.weak.insert(identity.clone(), Arc::downgrade(&context));
        self.touch(identity, context);
    }

    fn touch(&mut self, identity: MetalKeyContextIdentity, context: Arc<MetalKeyContextInner>) {
        if let Some(position) = self
            .warm
            .iter()
            .position(|(candidate, _)| candidate == &identity)
            && self.warm.remove(position).is_some()
        {
            self.recompute_warm_bytes();
        }

        // An entry that cannot fit by itself must not become a strong cache
        // root. The weak index still deduplicates it while a caller owns it.
        if self.max_entries == 0 || context.immutable_bytes > self.max_bytes {
            self.prune_dead_weak();
            return;
        }

        self.warm.push_back((identity, context));
        self.recompute_warm_bytes();

        while self.warm.len() > self.max_entries || self.warm_bytes > self.max_bytes {
            if self.warm.pop_front().is_none() {
                break;
            }
            self.recompute_warm_bytes();
        }
        self.prune_dead_weak();
    }

    fn recompute_warm_bytes(&mut self) {
        let total = self.warm.iter().try_fold(0usize, |total, (_, context)| {
            total.checked_add(context.immutable_bytes)
        });
        if let Some(total) = total {
            self.warm_bytes = total;
        } else {
            // Accounting overflow makes budget enforcement unprovable. Drop
            // every strong cache root and retain only caller-owned weak entries.
            self.warm.clear();
            self.warm_bytes = 0;
        }
    }

    fn prune_dead_weak(&mut self) {
        self.weak.retain(|_, context| context.strong_count() != 0);
    }
}

/// Session-local bounded LRU of SRS prefixes that passed the complete BLST
/// on-curve and subgroup scan.
///
/// The key is a neutral fingerprint recomputed by v2 prevalidation over both
/// exact basis arrays and `n`; callers cannot nominate an arbitrary digest.
struct ValidatedSrsCache {
    fingerprints: VecDeque<[u8; 32]>,
    max_entries: usize,
}

impl ValidatedSrsCache {
    fn new(max_entries: usize) -> Self {
        Self {
            fingerprints: VecDeque::new(),
            max_entries,
        }
    }

    fn contains_and_touch(&mut self, fingerprint: [u8; 32]) -> bool {
        let Some(position) = self
            .fingerprints
            .iter()
            .position(|candidate| *candidate == fingerprint)
        else {
            return false;
        };
        let fingerprint = self
            .fingerprints
            .remove(position)
            .expect("located validated-SRS fingerprint remains present");
        self.fingerprints.push_back(fingerprint);
        true
    }

    fn insert_validated(&mut self, fingerprint: [u8; 32]) {
        if self.max_entries == 0 {
            return;
        }
        if self.contains_and_touch(fingerprint) {
            return;
        }
        self.fingerprints.push_back(fingerprint);
        while self.fingerprints.len() > self.max_entries {
            self.fingerprints.pop_front();
        }
    }
}

#[allow(dead_code)] // Buffers become encoder inputs when proof routing is implemented.
struct MetalKeyContextBuffers {
    descriptors: Option<ImmutableBuffer>,
    constants: Option<ImmutableBuffer>,
    rotations: Option<ImmutableBuffer>,
    operations: Option<ImmutableBuffer>,
    horner_parts: Option<ImmutableBuffer>,
    fixed_map: Option<ImmutableBuffer>,
    advice_map: Option<ImmutableBuffer>,
    instance_map: Option<ImmutableBuffer>,
    fixed_cosets: Option<ImmutableBuffer>,
    lagrange_selectors: Option<ImmutableBuffer>,
    permutation_columns: Option<ImmutableBuffer>,
    trash_selector_fixed_indices: Option<ImmutableBuffer>,
    fixed_lagrange: Option<ImmutableBuffer>,
    permutation_cosets: Option<ImmutableBuffer>,
    extended_coset_points: Option<ImmutableBuffer>,
    permutation_lagrange: Option<ImmutableBuffer>,
    coefficient_columns: Option<ImmutableBuffer>,
    vanishing_inverses: Option<ImmutableBuffer>,
    coefficient_bases: Option<ImmutableBuffer>,
    lagrange_bases: Option<ImmutableBuffer>,
}

impl MetalKeyContextBuffers {
    fn checked_bytes(&self) -> Result<usize, MetalError> {
        [
            self.descriptors.as_ref(),
            self.constants.as_ref(),
            self.rotations.as_ref(),
            self.operations.as_ref(),
            self.horner_parts.as_ref(),
            self.fixed_map.as_ref(),
            self.advice_map.as_ref(),
            self.instance_map.as_ref(),
            self.fixed_cosets.as_ref(),
            self.lagrange_selectors.as_ref(),
            self.permutation_columns.as_ref(),
            self.trash_selector_fixed_indices.as_ref(),
            self.fixed_lagrange.as_ref(),
            self.permutation_cosets.as_ref(),
            self.extended_coset_points.as_ref(),
            self.permutation_lagrange.as_ref(),
            self.coefficient_columns.as_ref(),
            self.vanishing_inverses.as_ref(),
            self.coefficient_bases.as_ref(),
            self.lagrange_bases.as_ref(),
        ]
        .into_iter()
        .flatten()
        .try_fold(0usize, |bytes, buffer| bytes.checked_add(buffer.bytes))
        .ok_or_else(|| MetalError::KeyContext("immutable buffer byte sum overflows usize".into()))
    }
}

/// Semantic Fr arrays currently resident in an immutable key context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Constructed by resident graph/source encoders in the next slice.
enum MetalKeyFrSource {
    FixedCosetCompact,
    Selectors,
    FixedLagrangeCompact,
    PermutationCosets,
    ExtendedCosetPoints,
    PermutationLagrange,
    CoefficientColumns,
}

impl MetalKeyFrSource {
    const fn buffer_source(self) -> ImmutableBufferSource {
        match self {
            Self::FixedCosetCompact => ImmutableBufferSource::FixedCosetCompact,
            Self::Selectors => ImmutableBufferSource::Selectors,
            Self::FixedLagrangeCompact => ImmutableBufferSource::FixedLagrangeCompact,
            Self::PermutationCosets => ImmutableBufferSource::PermutationCosets,
            Self::ExtendedCosetPoints => ImmutableBufferSource::ExtendedCosetPoints,
            Self::PermutationLagrange => ImmutableBufferSource::PermutationLagrange,
            Self::CoefficientColumns => ImmutableBufferSource::CoefficientColumns,
        }
    }
}

/// Checked backend-private view into immutable key-owned Fr columns.
///
/// Unlike request-secret arena views, immutable views need no write generation:
/// their context publishes buffers only after the sole host write completes and
/// exposes no mutation path. The context/session identity and bounds remain
/// explicit so a future encoder cannot combine a view with the wrong device or
/// proving key.
#[must_use = "immutable key views must be consumed by the owning Metal session"]
struct ImmutableFrView<'context> {
    session_id: u64,
    context_id: u64,
    source_id: MetalKeyFrSource,
    allocation_id: u64,
    buffer: &'context ImmutableBuffer,
    byte_offset: usize,
    logical_len: usize,
    stride: usize,
    batch_count: usize,
    domain: FrDomainTag,
}

impl<'context> ImmutableFrView<'context> {
    /// Selects consecutive columns while preserving their domain stride.
    #[allow(dead_code)] // Used by the resident graph/source encoder in the next slice.
    fn columns(&self, column_offset: usize, column_count: usize) -> Result<Self, MetalError> {
        let column_end =
            column_offset
                .checked_add(column_count)
                .ok_or(MetalError::ResidentState(
                    "immutable key-view column range overflows",
                ))?;
        if column_count == 0 || column_end > self.batch_count {
            return Err(MetalError::ResidentState(
                "immutable key-view column range escapes its source",
            ));
        }
        let relative_elements =
            column_offset
                .checked_mul(self.stride)
                .ok_or(MetalError::ResidentState(
                    "immutable key-view column offset overflows",
                ))?;
        let relative_bytes = relative_elements
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::ResidentState(
                "immutable key-view byte offset overflows",
            ))?;
        let byte_offset =
            self.byte_offset
                .checked_add(relative_bytes)
                .ok_or(MetalError::ResidentState(
                    "immutable key-view absolute offset overflows",
                ))?;
        validate_resident_fr_view_shape(
            self.buffer.bytes,
            byte_offset,
            self.logical_len,
            self.stride,
            column_count,
        )?;
        Ok(Self {
            session_id: self.session_id,
            context_id: self.context_id,
            source_id: self.source_id,
            allocation_id: self.allocation_id,
            buffer: self.buffer,
            byte_offset,
            logical_len: self.logical_len,
            stride: self.stride,
            batch_count: column_count,
            domain: self.domain,
        })
    }
}

/// Semantic periodic Fr tables owned by a schema-v2 key context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Consumed by the resident quotient encoder in a later slice.
enum MetalKeyPeriodicFrSource {
    VanishingInverses,
}

impl MetalKeyPeriodicFrSource {
    const fn buffer_source(self) -> ImmutableBufferSource {
        match self {
            Self::VanishingInverses => ImmutableBufferSource::VanishingInverses,
        }
    }
}

/// Checked view of one complete repeating table, deliberately distinct from a
/// polynomial column view. `period` is the number of values uploaded, not the
/// extended-domain length over which a future kernel repeats them.
#[must_use = "immutable periodic views must be consumed by the owning Metal session"]
#[allow(dead_code)] // Consumed by the resident quotient encoder in a later slice.
struct ImmutablePeriodicFrView<'context> {
    session_id: u64,
    context_id: u64,
    source_id: MetalKeyPeriodicFrSource,
    allocation_id: u64,
    buffer: &'context ImmutableBuffer,
    period: usize,
}

/// Basis carried by an immutable G1 SRS prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
enum G1Basis {
    Coefficient,
    Lagrange,
}

/// Semantic G1 arrays owned by a schema-v2 key context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
enum MetalKeyG1Source {
    CoefficientBases,
    LagrangeBases,
}

impl MetalKeyG1Source {
    const fn buffer_source(self) -> ImmutableBufferSource {
        match self {
            Self::CoefficientBases => ImmutableBufferSource::CoefficientBases,
            Self::LagrangeBases => ImmutableBufferSource::LagrangeBases,
        }
    }

    const fn basis(self) -> G1Basis {
        match self {
            Self::CoefficientBases => G1Basis::Coefficient,
            Self::LagrangeBases => G1Basis::Lagrange,
        }
    }
}

/// Checked backend-private view into a contiguous range of immutable G1 bases.
#[must_use = "immutable G1 views must be consumed by the owning Metal session"]
#[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
struct ImmutableG1View<'context> {
    session_id: u64,
    context_id: u64,
    source_id: MetalKeyG1Source,
    basis: G1Basis,
    allocation_id: u64,
    buffer: &'context ImmutableBuffer,
    byte_offset: usize,
    logical_len: usize,
}

#[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
impl<'context> ImmutableG1View<'context> {
    fn bases(&self, base_offset: usize, base_count: usize) -> Result<Self, MetalError> {
        if base_count == 0 {
            return Err(MetalError::ResidentState(
                "immutable G1 view range must be nonzero",
            ));
        }
        let relative_bytes = base_offset
            .checked_mul(size_of::<Bls12_381G1Affine>())
            .ok_or(MetalError::ResidentState(
                "immutable G1 view offset overflows bytes",
            ))?;
        let byte_offset =
            self.byte_offset
                .checked_add(relative_bytes)
                .ok_or(MetalError::ResidentState(
                    "immutable G1 view absolute offset overflows",
                ))?;
        let parent_bytes = self
            .logical_len
            .checked_mul(size_of::<Bls12_381G1Affine>())
            .and_then(|bytes| self.byte_offset.checked_add(bytes))
            .ok_or(MetalError::ResidentState(
                "immutable G1 parent view extent overflows",
            ))?;
        validate_immutable_g1_view_shape(parent_bytes, byte_offset, base_count)?;
        Ok(Self {
            session_id: self.session_id,
            context_id: self.context_id,
            source_id: self.source_id,
            basis: self.basis,
            allocation_id: self.allocation_id,
            buffer: self.buffer,
            byte_offset,
            logical_len: base_count,
        })
    }
}

struct MetalKeyContextInner {
    session_id: u64,
    context_id: u64,
    identity: MetalKeyContextIdentity,
    graph_count: usize,
    fixed_column_count: usize,
    advice_column_count: usize,
    instance_column_count: usize,
    permutation_column_count: usize,
    fixed_coefficient_count: usize,
    permutation_coefficient_count: usize,
    vanishing_period: usize,
    quotient_piece_count: usize,
    rotation_scale: i32,
    last_rotation: i32,
    permutation_chunk_len: usize,
    permutation_delta: Bls12_381Fr,
    sections: MetalGraphFamilySections,
    challenge_bound: u32,
    descriptors: Box<[MetalGraphDescriptor]>,
    specialized_graphs: SpecializedGraphPipelines,
    permutation_columns: Box<[MetalResidentColumnRef]>,
    trash_selector_fixed_indices: Box<[u32]>,
    immutable_bytes: usize,
    buffers: MetalKeyContextBuffers,
    accounting: Arc<ImmutableKeyAccounting>,
}

impl Drop for MetalKeyContextInner {
    fn drop(&mut self) {
        self.accounting
            .live_bytes
            .fetch_sub(self.immutable_bytes as u64, Ordering::Relaxed);
        self.accounting
            .live_contexts
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// Reusable immutable graph/fixed state owned by one Metal session.
///
/// The handle contains no proof-secret storage and does not make the Metal
/// proof backend executable. Its raw Metal buffers remain backend-private.
#[derive(Clone)]
pub struct MetalKeyContext {
    inner: Arc<MetalKeyContextInner>,
}

impl std::fmt::Debug for MetalKeyContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalKeyContext")
            .field("context_id", &self.inner.context_id)
            .field("identity", &self.inner.identity)
            .field("graph_count", &self.inner.graph_count)
            .field("fixed_column_count", &self.inner.fixed_column_count)
            .field("immutable_bytes", &self.inner.immutable_bytes)
            .finish_non_exhaustive()
    }
}

impl MetalKeyContext {
    /// Complete content, ABI, metallib, and device cache identity.
    pub fn identity(&self) -> &MetalKeyContextIdentity {
        &self.inner.identity
    }

    /// Number of graph descriptors owned by this context.
    pub fn graph_count(&self) -> usize {
        self.inner.graph_count
    }

    /// Protocol graph ranges retained for backend-private command encoding.
    #[allow(dead_code)] // Read by the resident graph encoder once proof routing is enabled.
    pub(crate) fn sections(&self) -> MetalGraphFamilySections {
        self.inner.sections
    }

    /// Exclusive proof-challenge bound retained for backend-private validation.
    #[allow(dead_code)] // Read by the resident graph encoder once proof routing is enabled.
    pub(crate) fn challenge_bound(&self) -> u32 {
        self.inner.challenge_bound
    }

    /// Logical bytes uploaded into immutable Metal allocations.
    pub fn immutable_bytes(&self) -> usize {
        self.inner.immutable_bytes
    }
}

/// Exclusive ownership of a request-secret allocation checked out from the pool.
///
/// Unlike immutable plan buffers, leases are deliberately not cloneable. A
/// command may retain the raw Metal object, but only the arena can recycle it.
struct SecretBufferLease {
    raw: Buffer,
    bytes: usize,
    allocation_id: u64,
    lease_generation: u64,
}

pub(super) struct QuotientContinuationBuffers {
    pieces: Buffer,
    scratch: Buffer,
    scratch_bytes: usize,
    session_id: u64,
}

// SAFETY: the session execution mutex serializes every mapping, command use,
// clear, and pool transition. Raw command ownership is separately retained.
unsafe impl Send for SecretBufferLease {}

#[derive(Debug)]
enum SessionHealth {
    Healthy,
    Clearing,
    Poisoned(String),
}

struct LifecycleState {
    health: SessionHealth,
    pool: HashMap<usize, Vec<SecretBufferLease>>,
    quarantine: Vec<SecretBufferLease>,
    pooled_bytes: usize,
    owned_secret_bytes: usize,
    quarantined_bytes: usize,
    next_allocation_id: u64,
    next_arena_id: u64,
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self {
            health: SessionHealth::Healthy,
            pool: HashMap::new(),
            quarantine: Vec::new(),
            pooled_bytes: 0,
            owned_secret_bytes: 0,
            quarantined_bytes: 0,
            next_allocation_id: 1,
            next_arena_id: 1,
        }
    }
}

#[derive(Clone)]
struct NttTwiddlePlan {
    pq: TrackedBuffer,
    omega_squares: TrackedBuffer,
}

#[derive(Clone)]
struct NttPlan {
    domain: NttDomainSpec,
    domain_len: usize,
    max_degree: u32,
    schedule: Box<[u32]>,
    forward: NttTwiddlePlan,
    inverse: NttTwiddlePlan,
}

/// Backend-private ownership for two fixed-capacity resident Fr allocations.
///
/// A view's generation prevents an encoder from accidentally consuming a
/// slot after a later ping-pong transition has logically overwritten it.
struct ResidentFrArena<'session> {
    session: &'session MetalSession,
    arena_id: u64,
    slots: Vec<SecretBufferLease>,
    write_generations: [u64; 2],
    slot_states: [Option<ResidentFrSlotState>; 2],
}

/// Authoritative interpretation published for one initialized slot generation.
///
/// Views may narrow this state to consecutive complete columns, but cannot
/// reinterpret its basis, exact NTT domain, stride, or pass progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentFrSlotState {
    write_generation: u64,
    byte_offset: usize,
    logical_len: usize,
    stride: usize,
    batch_count: usize,
    domain: FrDomainTag,
}

impl ResidentFrSlotState {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        write_generation: u64,
        byte_offset: usize,
        logical_len: usize,
        stride: usize,
        batch_count: usize,
        domain: FrDomainTag,
    ) -> Self {
        Self {
            write_generation,
            byte_offset,
            logical_len,
            stride,
            batch_count,
            domain,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_subview(
        self,
        write_generation: u64,
        byte_offset: usize,
        logical_len: usize,
        stride: usize,
        batch_count: usize,
        domain: FrDomainTag,
    ) -> Result<(), MetalError> {
        if self.write_generation != write_generation
            || self.logical_len != logical_len
            || self.stride != stride
            || self.domain != domain
        {
            return Err(MetalError::ResidentState(
                "resident view does not match the authoritative slot state",
            ));
        }
        let relative_bytes =
            byte_offset
                .checked_sub(self.byte_offset)
                .ok_or(MetalError::ResidentState(
                    "resident view begins before its authoritative slot state",
                ))?;
        let column_bytes =
            stride
                .checked_mul(size_of::<Bls12_381Fr>())
                .ok_or(MetalError::ResidentState(
                    "resident Fr column stride overflows bytes",
                ))?;
        if !relative_bytes.is_multiple_of(column_bytes) {
            return Err(MetalError::ResidentState(
                "resident view is not a complete-column subview of its slot state",
            ));
        }
        let column_offset = relative_bytes / column_bytes;
        let column_end =
            column_offset
                .checked_add(batch_count)
                .ok_or(MetalError::ResidentState(
                    "resident authoritative column range overflows",
                ))?;
        if batch_count == 0 || column_end > self.batch_count {
            return Err(MetalError::ResidentState(
                "resident view escapes its authoritative slot state",
            ));
        }
        Ok(())
    }
}

/// Generation-checked interpretation of part of one resident Fr allocation.
///
/// `logical_len` elements are addressable in each batch and consecutive batch
/// starts are `stride` elements apart. The descriptor is deliberately not
/// cloneable: additional aliases must be created through a checked arena
/// operation rather than by copying opaque allocation identity.
#[must_use = "resident views must be consumed or deliberately derived"]
struct ResidentFrView {
    session_id: u64,
    arena_id: u64,
    slot: usize,
    allocation_id: u64,
    lease_generation: u64,
    write_generation: u64,
    byte_offset: usize,
    logical_len: usize,
    stride: usize,
    batch_count: usize,
    domain: FrDomainTag,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NttPassParameters {
    n: u32,
    source_stride: u32,
    destination_stride: u32,
    log_p: u32,
    degree: u32,
    max_degree: u32,
    batch_count: u32,
    flags: u32,
    reserved: [u32; 2],
    one: Bls12_381Fr,
    inverse_size: Bls12_381Fr,
    coset_generator: Bls12_381Fr,
    inverse_coset_generator: Bls12_381Fr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NttCosetParameters {
    input_length: u32,
    output_length: u32,
    source_stride: u32,
    destination_stride: u32,
    batch_count: u32,
    reserved: u32,
    coset_generator: Bls12_381Fr,
    inverse_coset_generator: Bls12_381Fr,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffineFp32 {
    limbs: [u32; 12],
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffinePoint32 {
    x: BatchAffineFp32,
    y: BatchAffineFp32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffineGatherParameters {
    record_offset: u32,
    record_count: u32,
    srs_count: u32,
    reserved: u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffineLevelParameters {
    destination_map_offset: u32,
    pair_count: u32,
    carry_count: u32,
    destination_len: u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffineProductParameters {
    input_offset: u32,
    input_count: u32,
    output_offset: u32,
    output_count: u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BatchAffineStatus {
    flags: u32,
    record_errors: u32,
    destination_errors: u32,
    reserved: u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchAffineCeilingParameters {
    element_count: u32,
    iterations: u32,
    reserved_0: u32,
    reserved_1: u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct FlatSegmentedParameters {
    tile_count: u32,
    input_count: u32,
    output_count: u32,
    srs_count: u32,
    group_count: u32,
    input_kind: u32,
    group_width: u32,
    reserved: u32,
}

const _: [(); 168] = [(); size_of::<NttPassParameters>()];
const _: [(); 88] = [(); size_of::<NttCosetParameters>()];
#[cfg(test)]
const _: [(); 48] = [(); size_of::<BatchAffineFp32>()];
#[cfg(test)]
const _: [(); 96] = [(); size_of::<BatchAffinePoint32>()];
#[cfg(test)]
const _: [(); 16] = [(); size_of::<BatchAffineGatherParameters>()];
#[cfg(test)]
const _: [(); 16] = [(); size_of::<BatchAffineLevelParameters>()];
#[cfg(test)]
const _: [(); 16] = [(); size_of::<BatchAffineProductParameters>()];
#[cfg(test)]
const _: [(); 16] = [(); size_of::<BatchAffineStatus>()];
#[cfg(test)]
const _: [(); 16] = [(); size_of::<BatchAffineCeilingParameters>()];
#[cfg(test)]
const _: [(); 32] = [(); size_of::<FlatSegmentedParameters>()];

const NTT_FLAG_NORMALIZE: u32 = 1;
const NTT_FLAG_COSET_EGRESS: u32 = 2;

mod pod {
    pub(super) trait Sealed {}
}

/// Types whose complete in-memory byte representation is valid Metal input.
///
/// Implementations must have no padding, permit every bit pattern, and use the
/// same size, alignment, and endianness in Rust and Metal.
///
/// # Safety
///
/// Implementing this trait for a type with padding, invalid bit patterns, or a
/// Rust/Metal layout mismatch would make raw buffer upload or readback unsound.
unsafe trait MetalPod: Copy + pod::Sealed {}

impl pod::Sealed for u8 {}
// SAFETY: `u8` has no padding and every bit pattern is valid.
unsafe impl MetalPod for u8 {}
impl pod::Sealed for u32 {}
// SAFETY: `u32` has no padding and every bit pattern is valid.
unsafe impl MetalPod for u32 {}
impl pod::Sealed for i32 {}
// SAFETY: `i32` has no padding and every bit pattern is valid.
unsafe impl MetalPod for i32 {}
impl pod::Sealed for u64 {}
// SAFETY: `u64` has no padding and every bit pattern is valid.
unsafe impl MetalPod for u64 {}
impl pod::Sealed for Bls12_381Fr {}
// SAFETY: the repr(C) wrapper contains exactly four `u64` limbs with no padding.
unsafe impl MetalPod for Bls12_381Fr {}
impl pod::Sealed for Bls12_381Fp {}
// SAFETY: the repr(C) wrapper contains exactly six `u64` limbs with no padding.
unsafe impl MetalPod for Bls12_381Fp {}
impl pod::Sealed for Bls12_381G1Affine {}
// SAFETY: the repr(C) wrapper contains exactly twelve `u64` limbs with no padding.
unsafe impl MetalPod for Bls12_381G1Affine {}
impl pod::Sealed for Bls12_381G1Projective {}
// SAFETY: the repr(C) wrapper contains exactly eighteen `u64` limbs with no padding.
unsafe impl MetalPod for Bls12_381G1Projective {}
impl pod::Sealed for MetalGraphValueRef {}
// SAFETY: the repr(C) block is exactly three u32 values without padding.
unsafe impl MetalPod for MetalGraphValueRef {}
impl pod::Sealed for MetalGraphOperation {}
// SAFETY: startup ABI validation proves the repr(C) block has no padding.
unsafe impl MetalPod for MetalGraphOperation {}
impl pod::Sealed for MetalGraphDescriptor {}
// SAFETY: the repr(C) block is exactly nine u32 values without padding.
unsafe impl MetalPod for MetalGraphDescriptor {}
impl pod::Sealed for MetalResidentColumnRef {}
// SAFETY: the repr(C) block is exactly two u32 values without padding.
unsafe impl MetalPod for MetalResidentColumnRef {}
#[cfg(test)]
impl pod::Sealed for GraphEvalParametersAbi {}
#[cfg(test)]
// SAFETY: startup ABI validation proves the repr(C) block has no padding.
unsafe impl MetalPod for GraphEvalParametersAbi {}

/// A command plus every resource whose lifetime must extend through execution.
struct PendingCommand {
    command: CommandBuffer,
    _resources: Vec<Buffer>,
}

/// A committed command that still owns every unretained encoder resource.
///
/// Keeping this state distinct from `PendingCommand` makes submission and the
/// first real host dependency separate operations without weakening lifetimes.
#[must_use = "a submitted Metal command must be joined before resources can be recycled"]
struct SubmittedCommand {
    command: CommandBuffer,
    _resources: Vec<Buffer>,
}

/// Long-lived owner of one Metal device, one ordered queue, and loaded pipelines.
///
/// One execution mutex covers command encoding, submission, waiting, and CPU
/// readback. Queue order defines dependencies inside that serialized region.
/// Callers receive a result only after the driver reports completion.
pub struct MetalSession {
    session_id: u64,
    next_key_context_id: AtomicU64,
    next_immutable_allocation_id: AtomicU64,
    device_registry_id: u64,
    metallib_fingerprint: [u8; 32],
    device: Device,
    queue: Queue,
    digest: Pipeline,
    clear_words: Pipeline,
    field: FieldPipelines,
    graph: GraphPipelines,
    quotient: QuotientPipelines,
    multiopen: MultiopenPipelines,
    g1: G1Pipelines,
    ntt: NttPipelines,
    #[cfg(test)]
    batch_affine: BatchAffinePipelines,
    #[cfg(test)]
    batch_affine_ceiling: BatchAffineCeilingPipelines,
    #[cfg(test)]
    flat_segmented: FlatSegmentedPipelines,
    ntt_plans: Mutex<HashMap<NttDomainSpec, NttPlan>>,
    key_contexts: Mutex<KeyContextCache>,
    validated_srs: Mutex<ValidatedSrsCache>,
    immutable_key_accounting: Arc<ImmutableKeyAccounting>,
    execution: Mutex<()>,
    lifecycle: Mutex<LifecycleState>,
    activity: ActivityCounters,
    inject_next_clear_failure: AtomicBool,
    #[cfg(test)]
    srs_subgroup_validations: AtomicU64,
}

fn validate_resident_fr_view_shape(
    available_bytes: usize,
    byte_offset: usize,
    logical_len: usize,
    stride: usize,
    batch_count: usize,
) -> Result<(), MetalError> {
    if !byte_offset.is_multiple_of(size_of::<Bls12_381Fr>()) {
        return Err(MetalError::ResidentState(
            "resident view offset is not on an Fr element boundary",
        ));
    }
    if logical_len == 0 || stride == 0 || batch_count == 0 {
        return Err(MetalError::ResidentState(
            "resident view dimensions must be nonzero",
        ));
    }
    if logical_len > stride {
        return Err(MetalError::ResidentState(
            "resident view logical length exceeds its stride",
        ));
    }
    let span_elements = (batch_count - 1)
        .checked_mul(stride)
        .and_then(|span| span.checked_add(logical_len))
        .ok_or(MetalError::ResidentState(
            "resident view element span overflows",
        ))?;
    let span_bytes =
        span_elements
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::ResidentState(
                "resident view byte span overflows",
            ))?;
    let requested = byte_offset
        .checked_add(span_bytes)
        .ok_or(MetalError::ResidentState(
            "resident view end offset overflows",
        ))?;
    if requested > available_bytes {
        return Err(MetalError::BufferBounds {
            requested,
            available: available_bytes,
        });
    }
    Ok(())
}

fn validate_immutable_g1_view_shape(
    available_bytes: usize,
    byte_offset: usize,
    logical_len: usize,
) -> Result<(), MetalError> {
    if !byte_offset.is_multiple_of(size_of::<Bls12_381G1Affine>()) {
        return Err(MetalError::ResidentState(
            "immutable G1 view offset is not on an affine-point boundary",
        ));
    }
    if logical_len == 0 {
        return Err(MetalError::ResidentState(
            "immutable G1 view length must be nonzero",
        ));
    }
    let requested = logical_len
        .checked_mul(size_of::<Bls12_381G1Affine>())
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(MetalError::ResidentState(
            "immutable G1 view extent overflows",
        ))?;
    if requested > available_bytes {
        return Err(MetalError::BufferBounds {
            requested,
            available: available_bytes,
        });
    }
    Ok(())
}

fn validate_fr_domain_shape(domain: FrDomainTag, logical_len: usize) -> Result<(), MetalError> {
    if domain.element_count()? != logical_len {
        return Err(MetalError::ResidentState(
            "resident Fr basis/domain does not match its logical length",
        ));
    }
    match (domain.basis, domain.ntt_domain) {
        (FrBasis::Coefficient, None) => {}
        (FrBasis::Coefficient, Some(_)) => {
            return Err(MetalError::ResidentState(
                "coefficient basis must not claim an evaluation-domain plan",
            ));
        }
        (
            FrBasis::Lagrange
            | FrBasis::ExtendedCoset
            | FrBasis::CosetCoefficient
            | FrBasis::StockhamWorking { .. },
            Some(spec),
        ) if spec.log_size == domain.log_size => {
            validate_ntt_domain(spec)?;
        }
        _ => {
            return Err(MetalError::ResidentState(
                "evaluation or NTT-working basis lacks its exact domain plan",
            ));
        }
    }
    if let FrBasis::StockhamWorking {
        completed_log_p, ..
    } = domain.basis
        && (completed_log_p == 0 || completed_log_p >= domain.log_size)
    {
        return Err(MetalError::ResidentState(
            "Stockham working state has an invalid completed-pass boundary",
        ));
    }
    Ok(())
}

fn validate_complete_fr_column_alignment(
    byte_offset: usize,
    stride: usize,
) -> Result<(), MetalError> {
    let column_bytes =
        stride
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::ResidentState(
                "resident Fr column stride overflows bytes",
            ))?;
    if !byte_offset.is_multiple_of(column_bytes) {
        return Err(MetalError::ResidentState(
            "resident Fr view is not aligned to a complete column",
        ));
    }
    Ok(())
}

impl ResidentFrArena<'_> {
    fn validate(&self, view: &ResidentFrView) -> Result<(), MetalError> {
        if view.session_id != self.session.session_id {
            self.session
                .activity
                .stale_handle_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(MetalError::ResidentState(
                "view belongs to a different Metal session",
            ));
        }
        if view.arena_id != self.arena_id {
            self.session
                .activity
                .stale_handle_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(MetalError::ResidentState(
                "view belongs to a different resident arena",
            ));
        }
        if view.slot >= self.slots.len() {
            return Err(MetalError::ResidentState("slot index is out of range"));
        }
        if view.write_generation == 0 || view.lease_generation == 0 {
            return Err(MetalError::ResidentState(
                "generation zero does not identify initialized resident data",
            ));
        }
        let slot = &self.slots[view.slot];
        if slot.allocation_id != view.allocation_id
            || slot.lease_generation != view.lease_generation
            || self.write_generations[view.slot] != view.write_generation
        {
            self.session
                .activity
                .stale_handle_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(MetalError::ResidentState("view generation is stale"));
        }
        validate_resident_fr_view_shape(
            slot.bytes,
            view.byte_offset,
            view.logical_len,
            view.stride,
            view.batch_count,
        )?;
        validate_complete_fr_column_alignment(view.byte_offset, view.stride)?;
        validate_fr_domain_shape(view.domain, view.logical_len)?;
        self.slot_states[view.slot]
            .ok_or(MetalError::ResidentState(
                "slot generation has no authoritative resident state",
            ))?
            .validate_subview(
                view.write_generation,
                view.byte_offset,
                view.logical_len,
                view.stride,
                view.batch_count,
                view.domain,
            )
    }

    #[allow(clippy::too_many_arguments)] // Every identity/layout field is validated at construction.
    fn view_for_slot(
        &self,
        slot: usize,
        write_generation: u64,
        byte_offset: usize,
        logical_len: usize,
        stride: usize,
        batch_count: usize,
        domain: FrDomainTag,
    ) -> Result<ResidentFrView, MetalError> {
        let allocation = self
            .slots
            .get(slot)
            .ok_or(MetalError::ResidentState("slot index is out of range"))?;
        if write_generation == 0 || self.write_generations.get(slot) != Some(&write_generation) {
            return Err(MetalError::ResidentState(
                "view does not identify the current initialized slot generation",
            ));
        }
        validate_resident_fr_view_shape(
            allocation.bytes,
            byte_offset,
            logical_len,
            stride,
            batch_count,
        )?;
        validate_complete_fr_column_alignment(byte_offset, stride)?;
        validate_fr_domain_shape(domain, logical_len)?;
        self.slot_states[slot]
            .ok_or(MetalError::ResidentState(
                "slot generation has no authoritative resident state",
            ))?
            .validate_subview(
                write_generation,
                byte_offset,
                logical_len,
                stride,
                batch_count,
                domain,
            )?;
        Ok(ResidentFrView {
            session_id: self.session.session_id,
            arena_id: self.arena_id,
            slot,
            allocation_id: allocation.allocation_id,
            lease_generation: allocation.lease_generation,
            write_generation,
            byte_offset,
            logical_len,
            stride,
            batch_count,
            domain,
        })
    }

    /// Selects complete consecutive polynomial columns without changing basis.
    #[allow(dead_code)] // Consumed by the first multi-column proof arena phase.
    fn columns(
        &self,
        parent: &ResidentFrView,
        batch_offset: usize,
        batch_count: usize,
    ) -> Result<ResidentFrView, MetalError> {
        self.validate(parent)?;
        let batch_end = batch_offset
            .checked_add(batch_count)
            .ok_or(MetalError::ResidentState("resident column range overflows"))?;
        if batch_count == 0 || batch_end > parent.batch_count {
            return Err(MetalError::ResidentState(
                "resident column range escapes its parent",
            ));
        }
        let relative_elements =
            batch_offset
                .checked_mul(parent.stride)
                .ok_or(MetalError::ResidentState(
                    "resident column offset overflows",
                ))?;
        let relative_bytes = relative_elements
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::ResidentState(
                "resident column byte offset overflows",
            ))?;
        let byte_offset =
            parent
                .byte_offset
                .checked_add(relative_bytes)
                .ok_or(MetalError::ResidentState(
                    "resident column byte offset overflows",
                ))?;
        self.view_for_slot(
            parent.slot,
            parent.write_generation,
            byte_offset,
            parent.logical_len,
            parent.stride,
            batch_count,
            parent.domain,
        )
    }

    fn transition(
        &mut self,
        input: &ResidentFrView,
        output_len: usize,
        output_stride: usize,
        output_domain: FrDomainTag,
    ) -> Result<ResidentFrView, MetalError> {
        self.validate(input)?;
        validate_complete_fr_column_alignment(input.byte_offset, input.stride)?;
        let input_column_bytes =
            input
                .stride
                .checked_mul(size_of::<Bls12_381Fr>())
                .ok_or(MetalError::ResidentState(
                    "resident transition input stride overflows bytes",
                ))?;
        let batch_offset = input.byte_offset / input_column_bytes;
        let byte_offset = batch_offset
            .checked_mul(output_stride)
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or(MetalError::ResidentState(
                "resident transition destination offset overflows",
            ))?;
        let slot = 1 - input.slot;
        let write_generation =
            self.write_generations[slot]
                .checked_add(1)
                .ok_or(MetalError::ResidentState(
                    "resident generation counter is exhausted",
                ))?;
        validate_resident_fr_view_shape(
            self.slots[slot].bytes,
            byte_offset,
            output_len,
            output_stride,
            input.batch_count,
        )?;
        validate_fr_domain_shape(output_domain, output_len)?;
        self.write_generations[slot] = write_generation;
        self.slot_states[slot] = Some(ResidentFrSlotState::new(
            write_generation,
            byte_offset,
            output_len,
            output_stride,
            input.batch_count,
            output_domain,
        ));
        Ok(ResidentFrView {
            session_id: self.session.session_id,
            arena_id: self.arena_id,
            slot,
            allocation_id: self.slots[slot].allocation_id,
            lease_generation: self.slots[slot].lease_generation,
            write_generation,
            byte_offset,
            logical_len: output_len,
            stride: output_stride,
            batch_count: input.batch_count,
            domain: output_domain,
        })
    }

    fn retained_resources(&self) -> Vec<Buffer> {
        self.slots.iter().map(|slot| slot.raw.clone()).collect()
    }

    fn finish<T>(mut self, operation: Result<T, MetalError>) -> Result<T, MetalError> {
        let cleanup = self.cleanup();
        match (operation, cleanup) {
            (result, Ok(())) => result,
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Err(operation), Err(cleanup)) => Err(MetalError::SessionPoisoned {
                reason: format!("{cleanup}; original operation error: {operation}"),
            }),
        }
    }

    fn cleanup(&mut self) -> Result<(), MetalError> {
        if self.slots.is_empty() {
            return Ok(());
        }
        self.session
            .clear_and_recycle_secret_buffers(std::mem::take(&mut self.slots))
    }
}

impl Drop for ResidentFrArena<'_> {
    fn drop(&mut self) {
        // An explicit finish reports cleanup failures. Drop covers early `?`
        // returns and unwinds; the cleanup path itself quarantines and poisons
        // the session if erasure cannot be proven complete.
        let _ = self.cleanup();
    }
}

impl MetalSession {
    /// Creates the device, ordered command queue, embedded library, and pipelines.
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::Device)?;
        let session_id = NEXT_SESSION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                MetalError::KeyContext("Metal session identity space is exhausted".into())
            })?;
        let device_registry_id = device.registryID();
        let metallib_fingerprint = Sha256::digest(METALLIB).into();
        let queue = device.newCommandQueue().ok_or(MetalError::Queue)?;
        queue.setLabel(Some(&NSString::from_str("midnight.prover.compute")));

        let library = device
            .newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB))
            .map_err(|error| MetalError::Library(format!("{error:?}")))?;
        let digest = pipeline(&device, &library, "midnight_proof_byte_digest")?;
        let clear_words = pipeline(&device, &library, "midnight_clear_words")?;
        let field = FieldPipelines::load(&device, &library)?;
        let graph = GraphPipelines::load(&device, &library)?;
        graph.validate_pipeline_limits()?;
        let quotient = QuotientPipelines::load(&device, &library)?;
        let multiopen = MultiopenPipelines::load(&device, &library)?;
        let g1 = G1Pipelines::load(&device, &library)?;
        let ntt = NttPipelines::load(&device, &library)?;
        #[cfg(test)]
        let batch_affine = BatchAffinePipelines::load(&device, &library)?;
        #[cfg(test)]
        let batch_affine_ceiling = BatchAffineCeilingPipelines::load(&device, &library)?;
        #[cfg(test)]
        let flat_segmented = FlatSegmentedPipelines::load(&device, &library)?;

        let key_context_warm_bytes =
            usize::try_from(device.recommendedMaxWorkingSetSize()).unwrap_or(usize::MAX) / 4;
        let session = Self {
            session_id,
            next_key_context_id: AtomicU64::new(1),
            next_immutable_allocation_id: AtomicU64::new(1),
            device_registry_id,
            metallib_fingerprint,
            device,
            queue,
            digest,
            clear_words,
            field,
            graph,
            quotient,
            multiopen,
            g1,
            ntt,
            #[cfg(test)]
            batch_affine,
            #[cfg(test)]
            batch_affine_ceiling,
            #[cfg(test)]
            flat_segmented,
            ntt_plans: Mutex::new(HashMap::new()),
            key_contexts: Mutex::new(KeyContextCache::new(key_context_warm_bytes)),
            validated_srs: Mutex::new(ValidatedSrsCache::new(VALIDATED_SRS_MAX_ENTRIES)),
            immutable_key_accounting: Arc::new(ImmutableKeyAccounting::default()),
            execution: Mutex::new(()),
            lifecycle: Mutex::new(LifecycleState::default()),
            activity: ActivityCounters::default(),
            inject_next_clear_failure: AtomicBool::new(false),
            #[cfg(test)]
            srs_subgroup_validations: AtomicU64::new(0),
        };
        let abi = session.bls12_381_abi_probe()?;
        validate_bls12_381_abi(abi)?;
        let ntt_abi = session.ntt_parameter_abi_probe()?;
        validate_ntt_parameter_abi(ntt_abi)?;
        let graph_abi = session.graph_abi_probe()?;
        validate_graph_abi(graph_abi)?;
        Ok(session)
    }

    /// Validates and prepares both fixed SRS bases in the thread-affine MSM worker.
    pub fn prepare_bls12_381_msm_srs(
        &self,
        fingerprint: [u8; 32],
        coefficient: &[Bls12_381G1Affine],
        lagrange: &[Bls12_381G1Affine],
    ) -> Result<(), MetalError> {
        msm_worker::prepare_srs(self.device_registry_id, fingerprint, coefficient, lagrange)
    }

    /// Reports whether this device's MSM worker already owns the content-addressed SRS.
    pub fn bls12_381_msm_srs_is_prepared(&self, fingerprint: [u8; 32]) -> Result<bool, MetalError> {
        msm_worker::is_prepared(self.device_registry_id, fingerprint)
    }

    /// Executes one fail-closed MSM against a previously prepared SRS basis.
    pub fn bls12_381_msm(
        &self,
        fingerprint: [u8; 32],
        basis: Bls12_381MsmBasis,
        scalars: &[Bls12_381Fr],
        window: Bls12_381MsmWindow,
    ) -> Result<Bls12_381G1Projective, MetalError> {
        msm_worker::execute(self.device_registry_id, fingerprint, basis, scalars, window)
    }

    /// Executes same-basis MSMs in one routed Metal submission.
    pub fn bls12_381_msm_batch(
        &self,
        fingerprint: [u8; 32],
        basis: Bls12_381MsmBasis,
        scalar_batches: &[&[Bls12_381Fr]],
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
        msm_worker::execute_batch(
            self.device_registry_id,
            fingerprint,
            basis,
            scalar_batches,
            window,
        )
    }

    /// Commits completed quotient pieces without mapping their scalar buffer on the CPU.
    pub fn bls12_381_msm_resident_quotient_batch(
        &self,
        fingerprint: [u8; 32],
        basis: Bls12_381MsmBasis,
        lease: &MetalQuotientResidentMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
        if lease.key_context_identity().device_registry_id() != self.device_registry_id
            || lease.scalars.device_registry_id() != self.device_registry_id
        {
            return Err(MetalError::ForeignKeyContext);
        }
        msm_worker::execute_resident_batch(
            self.device_registry_id,
            fingerprint,
            basis,
            lease.scalars.clone(),
            lease.piece_len(),
            lease.piece_count(),
            window,
        )
    }

    /// Commits a diagnostic CPU-visible quotient lease through its resident view.
    pub fn bls12_381_msm_resident_cpu_quotient_batch(
        &self,
        fingerprint: [u8; 32],
        basis: Bls12_381MsmBasis,
        lease: &MetalQuotientCpuMsmLease<'_>,
        window: Bls12_381MsmWindow,
    ) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
        if lease.key_context_identity().device_registry_id() != self.device_registry_id
            || lease.resident_scalars().device_registry_id() != self.device_registry_id
        {
            return Err(MetalError::ForeignKeyContext);
        }
        let lease_bytes = lease
            .piece_len()
            .checked_mul(lease.piece_count())
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or(MetalError::LengthOverflow(lease.piece_len()))?;
        self.activity
            .resident_msm_leases
            .fetch_add(1, Ordering::Relaxed);
        self.activity
            .resident_msm_lease_bytes
            .fetch_add(lease_bytes as u64, Ordering::Relaxed);
        msm_worker::execute_resident_batch(
            self.device_registry_id,
            fingerprint,
            basis,
            lease.resident_scalars().clone(),
            lease.piece_len(),
            lease.piece_count(),
            window,
        )
    }

    /// Validates, uploads, and caches low-level graph/fixed key arrays.
    ///
    /// This preparatory operation contains no proof-secret input and does not
    /// enable or route the incomplete Metal proof backend. Array shape and
    /// fingerprints are authoritative here, but protocol-category provenance
    /// is established by the classified `midnight-gpu-prover` facade before
    /// this low-level backend call.
    pub fn get_or_create_key_context(
        &self,
        input: MetalKeyContextInput<'_>,
    ) -> Result<MetalKeyContext, MetalError> {
        let validated = validate_metal_key_context_input(input)?;
        let identity = self.key_context_identity(&validated);
        let _execution = self.execution_guard();
        self.ensure_session_healthy()?;
        let mut cache = match self.key_contexts.lock() {
            Ok(cache) => cache,
            Err(_) => {
                let reason = "host panic poisoned the immutable key-context cache".to_owned();
                self.poison_session(reason.clone());
                self.key_contexts.clear_poison();
                return Err(MetalError::SessionPoisoned { reason });
            }
        };
        if let Some(context) = cache.get(&identity) {
            debug_assert_eq!(context.session_id, self.session_id);
            self.activity.key_cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(MetalKeyContext { inner: context });
        }

        self.activity
            .key_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        let context_id = match self.next_key_context_id() {
            Ok(context_id) => context_id,
            Err(error) => {
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        let context = match self.build_key_context(validated, identity.clone(), context_id) {
            Ok(context) => context,
            Err(error) => {
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        self.activity
            .key_context_builds
            .fetch_add(1, Ordering::Relaxed);
        cache.insert(identity, context.clone());
        Ok(MetalKeyContext { inner: context })
    }

    /// Validates, uploads, and caches the complete immutable schema-v2
    /// proving-key and SRS state.
    ///
    /// Cheap structural, field-reduction, exact-domain, and fingerprint checks
    /// run before cache lookup. The complete BLST on-curve/subgroup scan runs
    /// only for a context miss whose verified SRS fingerprint has not already
    /// passed in this session. This preparatory API does not enable proof
    /// routing or claim that the pure Metal backend is ready.
    pub fn get_or_create_key_context_v2(
        &self,
        input: MetalKeyContextInputV2<'_>,
    ) -> Result<MetalKeyContext, MetalError> {
        let validated = prevalidate_metal_key_context_input_v2(input)?;
        validate_ntt_domain_pair(
            validated.input.domain.base_ntt,
            validated.input.domain.extended_ntt,
        )?;
        let identity = self.key_context_identity_v2(&validated);

        let _execution = self.execution_guard();
        self.ensure_session_healthy()?;
        let mut cache = match self.key_contexts.lock() {
            Ok(cache) => cache,
            Err(_) => {
                let reason = "host panic poisoned the immutable key-context cache".to_owned();
                self.poison_session(reason.clone());
                self.key_contexts.clear_poison();
                return Err(MetalError::SessionPoisoned { reason });
            }
        };
        if let Some(context) = cache.get(&identity) {
            debug_assert_eq!(context.session_id, self.session_id);
            self.activity.key_cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(MetalKeyContext { inner: context });
        }

        self.activity
            .key_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        let srs_fingerprint = validated.input.prepared_srs_fingerprint;
        let mut validated_srs = match self.validated_srs.lock() {
            Ok(validated_srs) => validated_srs,
            Err(_) => {
                let reason = "host panic poisoned the validated-SRS cache".to_owned();
                self.poison_session(reason.clone());
                self.validated_srs.clear_poison();
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(MetalError::SessionPoisoned { reason });
            }
        };
        if !validated_srs.contains_and_touch(srs_fingerprint) {
            #[cfg(test)]
            self.srs_subgroup_validations
                .fetch_add(1, Ordering::Relaxed);
            if let Err(error) = validate_prevalidated_metal_key_context_srs_subgroups_v2(&validated)
            {
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
            // Publication follows, and never precedes, the successful complete
            // BLST scan above. The bounded cache stores no point data.
            validated_srs.insert_validated(srs_fingerprint);
        }
        drop(validated_srs);

        let context_id = match self.next_key_context_id() {
            Ok(context_id) => context_id,
            Err(error) => {
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        let context = match self.build_key_context_v2(validated, identity.clone(), context_id) {
            Ok(context) => context,
            Err(error) => {
                self.activity
                    .key_context_build_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        self.activity
            .key_context_builds
            .fetch_add(1, Ordering::Relaxed);
        cache.insert(identity, context.clone());
        Ok(MetalKeyContext { inner: context })
    }

    /// Rejects a key-context handle created by another Metal session.
    pub fn validate_key_context(&self, context: &MetalKeyContext) -> Result<(), MetalError> {
        if context.inner.session_id != self.session_id
            || context.inner.context_id == 0
            || context.inner.identity.device_registry_id() != self.device_registry_id
        {
            Err(MetalError::ForeignKeyContext)
        } else {
            Ok(())
        }
    }

    /// Resolves one immutable semantic Fr source without exposing its buffer.
    #[allow(dead_code)] // Consumed by resident proof encoders as they are connected.
    fn immutable_fr_view<'context>(
        &self,
        context: &'context MetalKeyContext,
        source: MetalKeyFrSource,
    ) -> Result<ImmutableFrView<'context>, MetalError> {
        let (buffer, logical_len, batch_count, domain) =
            self.immutable_fr_source_spec(context, source)?;
        validate_resident_fr_view_shape(buffer.bytes, 0, logical_len, logical_len, batch_count)?;
        let expected_bytes = logical_len
            .checked_mul(batch_count)
            .and_then(|elements| elements.checked_mul(size_of::<Bls12_381Fr>()))
            .ok_or(MetalError::ResidentState(
                "immutable key-view byte extent overflows",
            ))?;
        if buffer.bytes != expected_bytes {
            return Err(MetalError::ResidentState(
                "immutable key buffer does not match its validated column shape",
            ));
        }
        Ok(ImmutableFrView {
            session_id: self.session_id,
            context_id: context.inner.context_id,
            source_id: source,
            allocation_id: buffer.allocation_id,
            buffer,
            byte_offset: 0,
            logical_len,
            stride: logical_len,
            batch_count,
            domain,
        })
    }

    /// Revalidates an immutable view at the command-epoch boundary and retains
    /// its resource without exposing a raw Metal handle to the view API.
    #[allow(dead_code)] // Used by the resident graph/source encoder in the next slice.
    fn bind_immutable_fr_view(
        &self,
        context: &MetalKeyContext,
        view: &ImmutableFrView<'_>,
        expected_source: MetalKeyFrSource,
        resources: &mut Vec<Buffer>,
    ) -> Result<(), MetalError> {
        let (expected_buffer, canonical_len, source_columns, canonical_domain) =
            self.immutable_fr_source_spec(context, expected_source)?;
        if view.session_id != self.session_id
            || view.context_id != context.inner.context_id
            || view.allocation_id == 0
            || view.source_id != expected_source
            || view.domain != canonical_domain
            || !std::ptr::eq(view.buffer, expected_buffer)
            || view.buffer.session_id != self.session_id
            || view.buffer.context_id != context.inner.context_id
            || view.buffer.allocation_id != view.allocation_id
            || view.buffer.allocation_id != expected_buffer.allocation_id
            || view.buffer.source != expected_source.buffer_source()
        {
            return Err(MetalError::ResidentState(
                "immutable Fr view identity, source, or domain does not match its epoch binding",
            ));
        }
        validate_fr_domain_shape(view.domain, view.logical_len)?;
        validate_resident_fr_view_shape(
            view.buffer.bytes,
            view.byte_offset,
            view.logical_len,
            view.stride,
            view.batch_count,
        )?;
        if view.logical_len != canonical_len || view.stride != canonical_len {
            return Err(MetalError::ResidentState(
                "immutable Fr view does not cover complete canonical-domain columns",
            ));
        }
        validate_complete_fr_column_alignment(view.byte_offset, view.stride)?;
        let column_bytes = canonical_len.checked_mul(size_of::<Bls12_381Fr>()).ok_or(
            MetalError::ResidentState("immutable Fr column stride overflows bytes"),
        )?;
        let source_bytes =
            column_bytes
                .checked_mul(source_columns)
                .ok_or(MetalError::ResidentState(
                    "immutable Fr source extent overflows bytes",
                ))?;
        if view.buffer.bytes != source_bytes {
            return Err(MetalError::ResidentState(
                "immutable Fr buffer does not match its complete source extent",
            ));
        }
        resources.push(view.buffer.raw.clone());
        Ok(())
    }

    fn immutable_fr_source_spec<'context>(
        &self,
        context: &'context MetalKeyContext,
        source: MetalKeyFrSource,
    ) -> Result<(&'context ImmutableBuffer, usize, usize, FrDomainTag), MetalError> {
        self.validate_key_context(context)?;
        let identity = &context.inner.identity;
        let v2_base = identity.base_ntt_domain();
        let v2_extended = identity.extended_ntt_domain();
        let legacy_extended = || {
            let n_ext = identity.n_ext();
            bls12_381_ntt_domain(n_ext.ilog2())
        };
        let base_domain = || {
            v2_base.ok_or_else(|| {
                MetalError::KeyContext(
                    "requested immutable Fr source requires a schema-v2 key context".into(),
                )
            })
        };
        let extended_domain = || match (v2_base, v2_extended) {
            (Some(_), Some(extended)) => Ok(extended),
            (None, None) => legacy_extended(),
            _ => Err(MetalError::KeyContext(
                "key-context identity has an incomplete exact-domain pair".into(),
            )),
        };

        let (buffer, logical_len, batch_count, domain) = match source {
            MetalKeyFrSource::FixedCosetCompact => {
                let domain = extended_domain()?;
                (
                    context.inner.buffers.fixed_cosets.as_ref().ok_or_else(|| {
                        MetalError::KeyContext("key context has no fixed-coset columns".into())
                    })?,
                    1usize << domain.log_size,
                    context.inner.fixed_column_count,
                    FrDomainTag::for_ntt(domain, FrBasis::ExtendedCoset),
                )
            }
            MetalKeyFrSource::Selectors => {
                let domain = extended_domain()?;
                (
                    context
                        .inner
                        .buffers
                        .lagrange_selectors
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext("key context has no Lagrange selectors".into())
                        })?,
                    1usize << domain.log_size,
                    3,
                    FrDomainTag::for_ntt(domain, FrBasis::ExtendedCoset),
                )
            }
            MetalKeyFrSource::FixedLagrangeCompact => {
                let domain = base_domain()?;
                (
                    context
                        .inner
                        .buffers
                        .fixed_lagrange
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext(
                                "schema-v2 key context has no compact fixed-Lagrange columns"
                                    .into(),
                            )
                        })?,
                    1usize << domain.log_size,
                    context.inner.fixed_column_count,
                    FrDomainTag::for_ntt(domain, FrBasis::Lagrange),
                )
            }
            MetalKeyFrSource::PermutationCosets => {
                let domain = v2_extended.ok_or_else(|| {
                    MetalError::KeyContext(
                        "requested immutable Fr source requires a schema-v2 key context".into(),
                    )
                })?;
                (
                    context
                        .inner
                        .buffers
                        .permutation_cosets
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext(
                                "schema-v2 key context has no permutation cosets".into(),
                            )
                        })?,
                    1usize << domain.log_size,
                    context.inner.permutation_column_count,
                    FrDomainTag::for_ntt(domain, FrBasis::ExtendedCoset),
                )
            }
            MetalKeyFrSource::ExtendedCosetPoints => {
                let domain = v2_extended.ok_or_else(|| {
                    MetalError::KeyContext(
                        "requested immutable Fr source requires a schema-v2 key context".into(),
                    )
                })?;
                (
                    context
                        .inner
                        .buffers
                        .extended_coset_points
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext(
                                "schema-v2 key context has no extended-coset point table".into(),
                            )
                        })?,
                    1usize << domain.log_size,
                    1,
                    FrDomainTag::for_ntt(domain, FrBasis::ExtendedCoset),
                )
            }
            MetalKeyFrSource::PermutationLagrange => {
                let domain = base_domain()?;
                (
                    context
                        .inner
                        .buffers
                        .permutation_lagrange
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext(
                                "schema-v2 key context has no permutation-Lagrange columns".into(),
                            )
                        })?,
                    1usize << domain.log_size,
                    context.inner.permutation_column_count,
                    FrDomainTag::for_ntt(domain, FrBasis::Lagrange),
                )
            }
            MetalKeyFrSource::CoefficientColumns => {
                let domain = base_domain()?;
                let batch_count = context
                    .inner
                    .fixed_coefficient_count
                    .checked_add(context.inner.permutation_coefficient_count)
                    .ok_or(MetalError::ResidentState(
                        "schema-v2 coefficient-column count overflows usize",
                    ))?;
                (
                    context
                        .inner
                        .buffers
                        .coefficient_columns
                        .as_ref()
                        .ok_or_else(|| {
                            MetalError::KeyContext(
                                "schema-v2 key context has no coefficient columns".into(),
                            )
                        })?,
                    1usize << domain.log_size,
                    batch_count,
                    FrDomainTag::coefficient(domain.log_size),
                )
            }
        };
        Ok((buffer, logical_len, batch_count, domain))
    }

    /// Resolves the complete periodic vanishing-inverse table without
    /// pretending it is an extended-domain polynomial column.
    #[allow(dead_code)] // Consumed by the resident quotient encoder in a later slice.
    fn immutable_periodic_fr_view<'context>(
        &self,
        context: &'context MetalKeyContext,
        source: MetalKeyPeriodicFrSource,
    ) -> Result<ImmutablePeriodicFrView<'context>, MetalError> {
        self.validate_key_context(context)?;
        if context.inner.identity.base_ntt_domain().is_none()
            || context.inner.identity.extended_ntt_domain().is_none()
        {
            return Err(MetalError::KeyContext(
                "periodic immutable tables require a schema-v2 key context".into(),
            ));
        }
        let buffer = match source {
            MetalKeyPeriodicFrSource::VanishingInverses => context
                .inner
                .buffers
                .vanishing_inverses
                .as_ref()
                .ok_or_else(|| {
                    MetalError::KeyContext(
                        "schema-v2 key context has no vanishing-inverse table".into(),
                    )
                })?,
        };
        let expected_bytes = context
            .inner
            .vanishing_period
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::ResidentState(
                "periodic immutable Fr table extent overflows bytes",
            ))?;
        if context.inner.vanishing_period == 0 || buffer.bytes != expected_bytes {
            return Err(MetalError::ResidentState(
                "periodic immutable Fr buffer does not match its exact period",
            ));
        }
        Ok(ImmutablePeriodicFrView {
            session_id: self.session_id,
            context_id: context.inner.context_id,
            source_id: source,
            allocation_id: buffer.allocation_id,
            buffer,
            period: context.inner.vanishing_period,
        })
    }

    /// Revalidates a periodic table at a command-epoch boundary.
    #[allow(dead_code)] // Consumed by the resident quotient encoder in a later slice.
    fn bind_immutable_periodic_fr_view(
        &self,
        context: &MetalKeyContext,
        view: &ImmutablePeriodicFrView<'_>,
        expected_source: MetalKeyPeriodicFrSource,
        resources: &mut Vec<Buffer>,
    ) -> Result<(), MetalError> {
        let expected = self.immutable_periodic_fr_view(context, expected_source)?;
        if view.session_id != self.session_id
            || view.context_id != context.inner.context_id
            || view.source_id != expected_source
            || view.allocation_id == 0
            || view.allocation_id != expected.allocation_id
            || view.period != expected.period
            || !std::ptr::eq(view.buffer, expected.buffer)
            || view.buffer.session_id != self.session_id
            || view.buffer.context_id != context.inner.context_id
            || view.buffer.allocation_id != view.allocation_id
            || view.buffer.source != expected_source.buffer_source()
        {
            return Err(MetalError::ResidentState(
                "immutable periodic Fr view does not match its epoch binding",
            ));
        }
        resources.push(view.buffer.raw.clone());
        Ok(())
    }

    /// Resolves one exact schema-v2 G1 SRS basis without exposing its buffer.
    #[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
    fn immutable_g1_view<'context>(
        &self,
        context: &'context MetalKeyContext,
        source: MetalKeyG1Source,
    ) -> Result<ImmutableG1View<'context>, MetalError> {
        self.validate_key_context(context)?;
        let n = context.inner.identity.n() as usize;
        if context.inner.identity.prepared_srs_fingerprint().is_none() {
            return Err(MetalError::KeyContext(
                "immutable G1 SRS views require a schema-v2 key context".into(),
            ));
        }
        let buffer = match source {
            MetalKeyG1Source::CoefficientBases => context
                .inner
                .buffers
                .coefficient_bases
                .as_ref()
                .ok_or_else(|| {
                    MetalError::KeyContext(
                        "schema-v2 key context has no coefficient-basis SRS".into(),
                    )
                })?,
            MetalKeyG1Source::LagrangeBases => context
                .inner
                .buffers
                .lagrange_bases
                .as_ref()
                .ok_or_else(|| {
                    MetalError::KeyContext("schema-v2 key context has no Lagrange-basis SRS".into())
                })?,
        };
        validate_immutable_g1_view_shape(buffer.bytes, 0, n)?;
        let expected_bytes =
            n.checked_mul(size_of::<Bls12_381G1Affine>())
                .ok_or(MetalError::ResidentState(
                    "immutable G1 source extent overflows bytes",
                ))?;
        if buffer.bytes != expected_bytes {
            return Err(MetalError::ResidentState(
                "immutable G1 buffer does not match the prepared-SRS length",
            ));
        }
        Ok(ImmutableG1View {
            session_id: self.session_id,
            context_id: context.inner.context_id,
            source_id: source,
            basis: source.basis(),
            allocation_id: buffer.allocation_id,
            buffer,
            byte_offset: 0,
            logical_len: n,
        })
    }

    /// Revalidates an immutable G1 range at a command-epoch boundary.
    #[allow(dead_code)] // Reserved for a checked standalone-MSM adapter in a later slice.
    fn bind_immutable_g1_view(
        &self,
        context: &MetalKeyContext,
        view: &ImmutableG1View<'_>,
        expected_source: MetalKeyG1Source,
        resources: &mut Vec<Buffer>,
    ) -> Result<(), MetalError> {
        let expected = self.immutable_g1_view(context, expected_source)?;
        if view.session_id != self.session_id
            || view.context_id != context.inner.context_id
            || view.source_id != expected_source
            || view.basis != expected_source.basis()
            || view.allocation_id == 0
            || view.allocation_id != expected.allocation_id
            || !std::ptr::eq(view.buffer, expected.buffer)
            || view.buffer.session_id != self.session_id
            || view.buffer.context_id != context.inner.context_id
            || view.buffer.allocation_id != view.allocation_id
            || view.buffer.source != expected_source.buffer_source()
        {
            return Err(MetalError::ResidentState(
                "immutable G1 view identity, source, or basis does not match its epoch binding",
            ));
        }
        validate_immutable_g1_view_shape(view.buffer.bytes, view.byte_offset, view.logical_len)?;
        resources.push(view.buffer.raw.clone());
        Ok(())
    }

    fn key_context_identity(
        &self,
        validated: &ValidatedMetalKeyContextInput<'_>,
    ) -> MetalKeyContextIdentity {
        MetalKeyContextIdentity::new(
            validated.content_fingerprint,
            validated.input.graph_family_fingerprint,
            validated.input.n,
            validated.input.n_ext,
            GRAPH_ABI_VERSION,
            GRAPH_ABI_FINGERPRINT,
            BLS12_381_LAYOUT_ABI_FINGERPRINT,
            generated::BLS12_381_CONSTANTS_FINGERPRINT,
            self.metallib_fingerprint,
            self.device_registry_id,
        )
    }

    fn key_context_identity_v2(
        &self,
        validated: &ValidatedMetalKeyContextInputV2<'_>,
    ) -> MetalKeyContextIdentity {
        let input = validated.input;
        MetalKeyContextIdentity::new_v2(
            validated.content_fingerprint,
            input.graph_fixed.graph_family_fingerprint,
            input.domain.n,
            input.domain.n_ext,
            GRAPH_ABI_VERSION,
            GRAPH_ABI_FINGERPRINT,
            BLS12_381_LAYOUT_ABI_FINGERPRINT,
            generated::BLS12_381_CONSTANTS_FINGERPRINT,
            self.metallib_fingerprint,
            self.device_registry_id,
            input.domain.neutral_fingerprint,
            input.prepared_key_fingerprint,
            input.prepared_srs_fingerprint,
            input.domain.base_ntt,
            input.domain.extended_ntt,
            NTT_PARAMETER_ABI_VERSION,
            NTT_PARAMETER_ABI_FINGERPRINT,
        )
    }

    fn next_key_context_id(&self) -> Result<u64, MetalError> {
        self.next_key_context_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                MetalError::KeyContext("physical key-context identity space is exhausted".into())
            })
    }

    fn next_immutable_allocation_id(&self) -> Result<u64, MetalError> {
        self.next_immutable_allocation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                MetalError::KeyContext(
                    "immutable key-buffer allocation identity space is exhausted".into(),
                )
            })
    }

    fn build_key_context(
        &self,
        validated: ValidatedMetalKeyContextInput<'_>,
        identity: MetalKeyContextIdentity,
        context_id: u64,
    ) -> Result<Arc<MetalKeyContextInner>, MetalError> {
        let input = validated.input;
        let specialized_graphs = SpecializedGraphPipelines::disabled(input.descriptors.len());
        let buffers = MetalKeyContextBuffers {
            descriptors: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphDescriptors,
                input.descriptors,
                "midnight.key.graph_descriptors",
            )?,
            constants: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphConstants,
                input.constants,
                "midnight.key.graph_constants",
            )?,
            rotations: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphRotations,
                input.rotations,
                "midnight.key.graph_rotations",
            )?,
            operations: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphOperations,
                input.operations,
                "midnight.key.graph_operations",
            )?,
            horner_parts: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphHornerParts,
                input.horner_parts,
                "midnight.key.graph_horner_parts",
            )?,
            fixed_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::FixedMap,
                input.fixed_compact_to_logical,
                "midnight.key.fixed_map",
            )?,
            advice_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::AdviceMap,
                input.advice_compact_to_logical,
                "midnight.key.advice_map",
            )?,
            instance_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::InstanceMap,
                input.instance_compact_to_logical,
                "midnight.key.instance_map",
            )?,
            fixed_cosets: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::FixedCosetCompact,
                input.fixed_cosets,
                "midnight.key.fixed_cosets",
            )?,
            lagrange_selectors: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::Selectors,
                input.lagrange_selectors,
                "midnight.key.lagrange_selectors",
            )?,
            permutation_columns: None,
            trash_selector_fixed_indices: None,
            fixed_lagrange: None,
            permutation_cosets: None,
            extended_coset_points: None,
            permutation_lagrange: None,
            coefficient_columns: None,
            vanishing_inverses: None,
            coefficient_bases: None,
            lagrange_bases: None,
        };

        let immutable_bytes = u64::try_from(validated.immutable_bytes)
            .map_err(|_| MetalError::KeyContext("immutable byte count exceeds u64".into()))?;
        let live = self
            .immutable_key_accounting
            .live_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(immutable_bytes)
            })
            .map(|previous| previous + immutable_bytes)
            .map_err(|_| MetalError::KeyContext("immutable live-byte counter overflow".into()))?;
        if self
            .immutable_key_accounting
            .live_contexts
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |contexts| {
                contexts.checked_add(1)
            })
            .is_err()
        {
            self.immutable_key_accounting
                .live_bytes
                .fetch_sub(immutable_bytes, Ordering::Relaxed);
            return Err(MetalError::KeyContext(
                "immutable live-context counter overflow".into(),
            ));
        }
        self.immutable_key_accounting
            .peak_bytes
            .fetch_max(live, Ordering::Relaxed);
        Ok(Arc::new(MetalKeyContextInner {
            session_id: self.session_id,
            context_id,
            identity,
            graph_count: input.descriptors.len(),
            fixed_column_count: input.fixed_compact_to_logical.len(),
            advice_column_count: input.advice_compact_to_logical.len(),
            instance_column_count: input.instance_compact_to_logical.len(),
            permutation_column_count: 0,
            fixed_coefficient_count: 0,
            permutation_coefficient_count: 0,
            vanishing_period: 0,
            quotient_piece_count: 0,
            rotation_scale: 0,
            last_rotation: 0,
            permutation_chunk_len: 0,
            permutation_delta: Bls12_381Fr::default(),
            sections: input.sections,
            challenge_bound: input.challenge_bound,
            descriptors: input.descriptors.into(),
            specialized_graphs,
            permutation_columns: Box::new([]),
            trash_selector_fixed_indices: Box::new([]),
            immutable_bytes: validated.immutable_bytes,
            buffers,
            accounting: self.immutable_key_accounting.clone(),
        }))
    }

    fn build_key_context_v2(
        &self,
        validated: ValidatedMetalKeyContextInputV2<'_>,
        identity: MetalKeyContextIdentity,
        context_id: u64,
    ) -> Result<Arc<MetalKeyContextInner>, MetalError> {
        let input = validated.input;
        let graph = input.graph_fixed;
        let specialized_graphs = self.compile_specialized_graph_pipelines(graph)?;
        let extended_coset_points = if input.permutation_columns.is_empty() {
            Vec::new()
        } else {
            build_extended_coset_points(input.domain.extended_ntt, input.domain.n_ext as usize)?
        };
        let derived_immutable_bytes = extended_coset_points
            .len()
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or_else(|| {
                MetalError::KeyContext(
                    "derived extended-coset point table overflows immutable bytes".into(),
                )
            })?;
        let physical_immutable_bytes = validated
            .immutable_bytes
            .checked_add(derived_immutable_bytes)
            .ok_or_else(|| {
                MetalError::KeyContext("schema-v2 physical immutable bytes overflow usize".into())
            })?;
        let buffers = MetalKeyContextBuffers {
            descriptors: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphDescriptors,
                graph.descriptors,
                "midnight.key.graph_descriptors",
            )?,
            constants: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphConstants,
                graph.constants,
                "midnight.key.graph_constants",
            )?,
            rotations: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphRotations,
                graph.rotations,
                "midnight.key.graph_rotations",
            )?,
            operations: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphOperations,
                graph.operations,
                "midnight.key.graph_operations",
            )?,
            horner_parts: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::GraphHornerParts,
                graph.horner_parts,
                "midnight.key.graph_horner_parts",
            )?,
            fixed_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::FixedMap,
                graph.fixed_compact_to_logical,
                "midnight.key.fixed_map",
            )?,
            advice_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::AdviceMap,
                graph.advice_compact_to_logical,
                "midnight.key.advice_map",
            )?,
            instance_map: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::InstanceMap,
                graph.instance_compact_to_logical,
                "midnight.key.instance_map",
            )?,
            fixed_cosets: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::FixedCosetCompact,
                graph.fixed_cosets,
                "midnight.key.fixed_cosets",
            )?,
            lagrange_selectors: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::Selectors,
                graph.lagrange_selectors,
                "midnight.key.lagrange_selectors",
            )?,
            permutation_columns: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::PermutationColumns,
                input.permutation_columns,
                "midnight.key.permutation_columns",
            )?,
            trash_selector_fixed_indices: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::TrashSelectorFixedIndices,
                input.trash_selector_fixed_indices,
                "midnight.key.trash_selector_fixed_indices",
            )?,
            fixed_lagrange: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::FixedLagrangeCompact,
                input.fixed_lagrange,
                "midnight.key.fixed_lagrange",
            )?,
            permutation_cosets: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::PermutationCosets,
                input.permutation_cosets,
                "midnight.key.permutation_cosets",
            )?,
            extended_coset_points: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::ExtendedCosetPoints,
                &extended_coset_points,
                "midnight.key.extended_coset_points",
            )?,
            permutation_lagrange: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::PermutationLagrange,
                input.permutation_lagrange,
                "midnight.key.permutation_lagrange",
            )?,
            coefficient_columns: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::CoefficientColumns,
                input.coefficient_columns,
                "midnight.key.coefficient_columns",
            )?,
            vanishing_inverses: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::VanishingInverses,
                input.domain.vanishing_inverses,
                "midnight.key.vanishing_inverses",
            )?,
            coefficient_bases: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::CoefficientBases,
                input.coefficient_bases,
                "midnight.key.coefficient_bases",
            )?,
            lagrange_bases: self.immutable_buffer_from_slice(
                context_id,
                ImmutableBufferSource::LagrangeBases,
                input.lagrange_bases,
                "midnight.key.lagrange_bases",
            )?,
        };
        if buffers.checked_bytes()? != physical_immutable_bytes {
            return Err(MetalError::KeyContext(
                "schema-v2 immutable upload bytes differ from validated and derived bytes".into(),
            ));
        }

        let immutable_bytes = u64::try_from(physical_immutable_bytes)
            .map_err(|_| MetalError::KeyContext("immutable byte count exceeds u64".into()))?;
        let live = self
            .immutable_key_accounting
            .live_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(immutable_bytes)
            })
            .map(|previous| previous + immutable_bytes)
            .map_err(|_| MetalError::KeyContext("immutable live-byte counter overflow".into()))?;
        if self
            .immutable_key_accounting
            .live_contexts
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |contexts| {
                contexts.checked_add(1)
            })
            .is_err()
        {
            self.immutable_key_accounting
                .live_bytes
                .fetch_sub(immutable_bytes, Ordering::Relaxed);
            return Err(MetalError::KeyContext(
                "immutable live-context counter overflow".into(),
            ));
        }
        self.immutable_key_accounting
            .peak_bytes
            .fetch_max(live, Ordering::Relaxed);
        Ok(Arc::new(MetalKeyContextInner {
            session_id: self.session_id,
            context_id,
            identity,
            graph_count: graph.descriptors.len(),
            fixed_column_count: graph.fixed_compact_to_logical.len(),
            advice_column_count: graph.advice_compact_to_logical.len(),
            instance_column_count: graph.instance_compact_to_logical.len(),
            permutation_column_count: input.permutation_columns.len(),
            fixed_coefficient_count: input.fixed_coefficient_count as usize,
            permutation_coefficient_count: input.permutation_coefficient_count as usize,
            vanishing_period: input.domain.vanishing_inverses.len(),
            quotient_piece_count: input.domain.quotient_piece_count as usize,
            rotation_scale: input.domain.rotation_scale,
            last_rotation: input.domain.last_rotation,
            permutation_chunk_len: input.domain.permutation_chunk_len as usize,
            permutation_delta: input.domain.delta,
            sections: graph.sections,
            challenge_bound: graph.challenge_bound,
            descriptors: graph.descriptors.into(),
            specialized_graphs,
            permutation_columns: input.permutation_columns.into(),
            trash_selector_fixed_indices: input.trash_selector_fixed_indices.into(),
            immutable_bytes: physical_immutable_bytes,
            buffers,
            accounting: self.immutable_key_accounting.clone(),
        }))
    }

    fn immutable_buffer_from_slice<T: MetalPod>(
        &self,
        context_id: u64,
        source: ImmutableBufferSource,
        values: &[T],
        label: &str,
    ) -> Result<Option<ImmutableBuffer>, MetalError> {
        if values.is_empty() {
            return Ok(None);
        }
        let allocation_id = self.next_immutable_allocation_id()?;
        let buffer = self.buffer_from_slice(values, label)?;
        Ok(Some(ImmutableBuffer {
            session_id: self.session_id,
            context_id,
            allocation_id,
            source,
            raw: buffer.raw,
            bytes: buffer.bytes,
        }))
    }

    /// Computes the canonical digest on Metal and never falls back to CPU work.
    pub fn proof_byte_digest(&self, bytes: &[u8]) -> Result<ProofByteDigest, MetalError> {
        let _execution = self.execution_guard();
        let length =
            u64::try_from(bytes.len()).map_err(|_| MetalError::LengthOverflow(bytes.len()))?;
        let sentinel = [0u8];
        let input_bytes = if bytes.is_empty() {
            &sentinel[..]
        } else {
            bytes
        };
        let input = self.buffer_from_slice(input_bytes, "midnight.digest.input")?;
        let length = self.buffer_from_slice(&[length], "midnight.digest.length")?;
        let output = self.buffer(4 * size_of::<u64>(), "midnight.digest.output")?;

        let command = self.command_buffer("midnight.proof_byte_digest")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.clear_words);
        // SAFETY: the output allocation is retained until completion and its
        // zero offset is aligned for the shader's `ulong` pointer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.clear_words, 4);

        encoder.setComputePipelineState(&self.digest);
        // SAFETY: all three zero-offset buffers match their shader element
        // types and are retained by `PendingCommand` until execution completes.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&length.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
        }
        self.dispatch(&encoder, &self.digest, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![input.raw.clone(), length.raw.clone(), output.raw.clone()],
        })?;

        let words = self.read_words::<4>(&output)?;
        let len = usize::try_from(words[0])
            .map_err(|_| MetalError::DigestAbi("length does not fit usize"))?;
        let xor = u8::try_from(words[1])
            .map_err(|_| MetalError::DigestAbi("xor word does not fit u8"))?;
        if len != bytes.len() {
            return Err(MetalError::DigestAbi(
                "length differs from the submitted input",
            ));
        }
        Ok(ProofByteDigest {
            len,
            xor,
            sum64: words[2],
            rolling: words[3],
        })
    }

    /// Runs the generated BLS12-381 layout and constants probe on Metal.
    pub fn bls12_381_abi_probe(&self) -> Result<Bls12_381AbiProbe, MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            BLS12_381_LAYOUT_ABI_WORDS * size_of::<u64>(),
            "midnight.bls12_381.abi.output",
        )?;
        let command = self.command_buffer("midnight.bls12_381.abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.field.abi_probe);
        // SAFETY: `output` remains retained by `PendingCommand` until execution
        // completes, and offset zero satisfies the `ulong` buffer alignment.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.field.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;

        let words = self.read_words::<BLS12_381_LAYOUT_ABI_WORDS>(&output)?;
        Ok(Bls12_381AbiProbe {
            abi_version: words[0],
            fr_size: words[1],
            fr_alignment: words[2],
            fr_limbs_offset: words[3],
            fp_size: words[4],
            fp_alignment: words[5],
            fp_limbs_offset: words[6],
            g1_affine_size: words[7],
            g1_affine_alignment: words[8],
            g1_affine_x_offset: words[9],
            g1_affine_y_offset: words[10],
            g1_projective_size: words[11],
            g1_projective_alignment: words[12],
            g1_projective_x_offset: words[13],
            g1_projective_y_offset: words[14],
            g1_projective_z_offset: words[15],
            g1_homogeneous_size: words[16],
            g1_homogeneous_alignment: words[17],
            g1_homogeneous_x_offset: words[18],
            g1_homogeneous_y_offset: words[19],
            g1_homogeneous_z_offset: words[20],
            layout_fingerprint: words[21],
            constants_fingerprint: words[22],
        })
    }

    /// Runs the graph-expression layout and tag probe on Metal.
    pub fn graph_abi_probe(&self) -> Result<GraphAbiProbe, MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            GRAPH_ABI_WORDS * size_of::<u64>(),
            "midnight.graph.abi.output",
        )?;
        let command = self.command_buffer("midnight.graph.abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.graph.abi_probe);
        // SAFETY: the retained output is a zero-offset `ulong` array with one
        // element for every word emitted by the graph ABI probe.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.graph.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;
        let words = self.read_words::<GRAPH_ABI_WORDS>(&output)?;
        Ok(graph_abi_from_words(words))
    }

    fn ntt_parameter_abi_probe(&self) -> Result<[u64; NTT_PARAMETER_ABI_WORDS], MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            NTT_PARAMETER_ABI_WORDS * size_of::<u64>(),
            "midnight.ntt.parameter_abi.output",
        )?;
        let command = self.command_buffer("midnight.ntt.parameter_abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.ntt.abi_probe);
        // SAFETY: the output is a retained zero-offset u64 array large enough
        // for every layout word emitted by the parameter ABI probe.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.ntt.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;
        self.read_words::<NTT_PARAMETER_ABI_WORDS>(&output)
    }

    #[cfg(test)]
    fn batch_affine_abi_probe(&self) -> Result<[u64; BATCH_AFFINE_ABI_WORDS], MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            BATCH_AFFINE_ABI_WORDS * size_of::<u64>(),
            "midnight.batch_affine.abi.output",
        )?;
        let command = self.command_buffer("midnight.batch_affine.abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.batch_affine.abi_probe);
        // SAFETY: the retained output is a zero-offset u64 array with one slot
        // for every word in the diagnostic-only layout probe.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.batch_affine.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;
        self.read_words::<BATCH_AFFINE_ABI_WORDS>(&output)
    }

    #[cfg(test)]
    fn batch_affine_ceiling_abi_probe(
        &self,
    ) -> Result<[u64; BATCH_AFFINE_CEILING_ABI_WORDS], MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            BATCH_AFFINE_CEILING_ABI_WORDS * size_of::<u64>(),
            "midnight.batch_affine.ceiling_abi.output",
        )?;
        let command = self.command_buffer("midnight.batch_affine.ceiling_abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.batch_affine_ceiling.abi_probe);
        // SAFETY: the retained output is a zero-offset u64 array with one slot
        // for every word in the separately versioned ceiling-probe ABI.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.batch_affine_ceiling.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;
        self.read_words::<BATCH_AFFINE_CEILING_ABI_WORDS>(&output)
    }

    #[cfg(test)]
    fn flat_segmented_abi_probe(&self) -> Result<[u64; FLAT_SEGMENTED_ABI_WORDS], MetalError> {
        let _execution = self.execution_guard();
        let output = self.buffer(
            FLAT_SEGMENTED_ABI_WORDS * size_of::<u64>(),
            "midnight.flat_segmented.abi.output",
        )?;
        let command = self.command_buffer("midnight.flat_segmented.abi_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.flat_segmented.abi_probe);
        // SAFETY: the retained output is a zero-offset u64 array with one slot
        // for every word in the separately versioned diagnostic ABI probe.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.flat_segmented.abi_probe, 1);
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone()],
        })?;
        self.read_words::<FLAT_SEGMENTED_ABI_WORDS>(&output)
    }

    /// Executes batched Fr64 add, subtract, Montgomery multiply, and square probes.
    ///
    /// Inputs and outputs use the raw reduced Montgomery representation. This is
    /// a primitive correctness boundary and does not make the proof backend ready.
    pub fn probe_bls12_381_fr64(
        &self,
        lhs: &[Bls12_381Fr],
        rhs: &[Bls12_381Fr],
    ) -> Result<Bls12_381Fr64ProbeResult, MetalError> {
        let [add, sub, mul, square] = self.probe_field64(
            lhs,
            rhs,
            [
                &self.field.fr64.add,
                &self.field.fr64.sub,
                &self.field.fr64.mul,
                &self.field.fr64.square,
            ],
            "fr64",
        )?;
        Ok(Bls12_381Fr64ProbeResult {
            add,
            sub,
            mul,
            square,
        })
    }

    /// Executes batched Fp64 add, subtract, Montgomery multiply, and square probes.
    ///
    /// Inputs and outputs use the raw reduced Montgomery representation. This is
    /// a primitive correctness boundary and does not make the proof backend ready.
    pub fn probe_bls12_381_fp64(
        &self,
        lhs: &[Bls12_381Fp],
        rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        let [add, sub, mul, square] = self.probe_field64(
            lhs,
            rhs,
            [
                &self.field.fp64.add,
                &self.field.fp64.sub,
                &self.field.fp64.mul,
                &self.field.fp64.square,
            ],
            "fp64",
        )?;
        Ok(Bls12_381Fp64ProbeResult {
            add,
            sub,
            mul,
            square,
        })
    }

    /// Executes diagnostic 12x32-bit Fp add, subtract, multiply, and square.
    ///
    /// Public inputs and outputs retain the six-`u64` BLST/CUDA ABI. Each
    /// diagnostic kernel explicitly splits and rejoins the same R=2^384
    /// Montgomery bit pattern around the internal 12-`u32` arithmetic. No NTT,
    /// G1, proof, or existing Fp64 route selects this experiment.
    pub fn probe_bls12_381_fp32(
        &self,
        lhs: &[Bls12_381Fp],
        rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        let [add, sub, mul, square] = self.probe_field64(
            lhs,
            rhs,
            [
                &self.field.fp32.add,
                &self.field.fp32.sub,
                &self.field.fp32.mul,
                &self.field.fp32.square,
            ],
            "fp32",
        )?;
        Ok(Bls12_381Fp64ProbeResult {
            add,
            sub,
            mul,
            square,
        })
    }

    /// Executes the sibling wide-accumulator 12x32-bit Fp diagnostic.
    ///
    /// Addition and subtraction share the identical fixed-schedule Fp32
    /// kernels. Multiplication and square accumulate each 32x32 product, limb,
    /// and carry in one `u64` before splitting at bit 32. Public buffers remain
    /// the unchanged six-`u64`, R=2^384 Montgomery ABI.
    pub fn probe_bls12_381_fp32_wide(
        &self,
        lhs: &[Bls12_381Fp],
        rhs: &[Bls12_381Fp],
    ) -> Result<Bls12_381Fp64ProbeResult, MetalError> {
        let [add, sub, mul, square] = self.probe_field64(
            lhs,
            rhs,
            [
                &self.field.fp32.add,
                &self.field.fp32.sub,
                &self.field.fp32.wide_mul,
                &self.field.fp32.wide_square,
            ],
            "fp32_wide",
        )?;
        Ok(Bls12_381Fp64ProbeResult {
            add,
            sub,
            mul,
            square,
        })
    }

    /// Executes batched G1 normalization, projective doubling, and mixed addition.
    ///
    /// Every returned point is canonical affine, including the all-zero encoding
    /// for infinity. Inputs must contain reduced Montgomery coordinates; curve and
    /// subgroup membership are not checked. Complete formulas give the point
    /// operations fixed schedules, but the field implementation has not received
    /// a side-channel audit. This remains a correctness probe and is not routed
    /// into proof execution or witness-derived MSM work.
    pub fn probe_bls12_381_g1(
        &self,
        projective: &[Bls12_381G1Projective],
        affine: &[Bls12_381G1Affine],
    ) -> Result<Bls12_381G1ProbeResult, MetalError> {
        if projective.len() != affine.len() {
            return Err(MetalError::LengthMismatch {
                left: projective.len(),
                right: affine.len(),
            });
        }
        if projective.is_empty() {
            return Ok(Bls12_381G1ProbeResult::default());
        }

        let length = u32::try_from(projective.len())
            .map_err(|_| MetalError::LengthOverflow(projective.len()))?;
        let _execution = self.execution_guard();
        let projective_input =
            self.buffer_from_slice(projective, "midnight.g1.projective_input")?;
        let affine_input = self.buffer_from_slice(affine, "midnight.g1.affine_input")?;
        let length_buffer = self.buffer_from_slice(&[length], "midnight.g1.length")?;
        let projective_bytes = size_of_val(projective);
        let affine_bytes = size_of_val(affine);
        let doubled_projective = self.buffer(projective_bytes, "midnight.g1.doubled_projective")?;
        let mixed_projective = self.buffer(projective_bytes, "midnight.g1.mixed_projective")?;
        let normalized = self.buffer(affine_bytes, "midnight.g1.normalized")?;
        let doubled = self.buffer(affine_bytes, "midnight.g1.doubled")?;
        let mixed_sum = self.buffer(affine_bytes, "midnight.g1.mixed_sum")?;

        let command = self.command_buffer("midnight.bls12_381.g1_probe")?;
        let projective_encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        projective_encoder.setComputePipelineState(&self.g1.double);
        // SAFETY: every zero-offset buffer has the raw projective/u32 layout
        // required by the doubling kernel and is retained through completion.
        unsafe {
            projective_encoder.setBuffer_offset_atIndex(Some(&projective_input.raw), 0, 0);
            projective_encoder.setBuffer_offset_atIndex(Some(&doubled_projective.raw), 0, 1);
            projective_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
        }
        self.dispatch(&projective_encoder, &self.g1.double, projective.len());

        projective_encoder.setComputePipelineState(&self.g1.mixed_add);
        // SAFETY: every zero-offset buffer has the raw projective, affine, or
        // u32 layout required by mixed addition and is retained to completion.
        unsafe {
            projective_encoder.setBuffer_offset_atIndex(Some(&projective_input.raw), 0, 0);
            projective_encoder.setBuffer_offset_atIndex(Some(&affine_input.raw), 0, 1);
            projective_encoder.setBuffer_offset_atIndex(Some(&mixed_projective.raw), 0, 2);
            projective_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
        }
        self.dispatch(&projective_encoder, &self.g1.mixed_add, projective.len());
        projective_encoder.endEncoding();

        // A second encoder in the same command buffer executes after the first,
        // making both projective outputs visible to canonical normalization.
        let affine_encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        affine_encoder.setComputePipelineState(&self.g1.to_affine);
        for (input, output) in [
            (&projective_input.raw, &normalized.raw),
            (&doubled_projective.raw, &doubled.raw),
            (&mixed_projective.raw, &mixed_sum.raw),
        ] {
            // SAFETY: the input/output pairs have the raw projective/affine
            // layouts required by the kernel and remain retained until completion.
            unsafe {
                affine_encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                affine_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
                affine_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
            }
            self.dispatch(&affine_encoder, &self.g1.to_affine, projective.len());
        }
        affine_encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                projective_input.raw.clone(),
                affine_input.raw.clone(),
                length_buffer.raw.clone(),
                doubled_projective.raw.clone(),
                mixed_projective.raw.clone(),
                normalized.raw.clone(),
                doubled.raw.clone(),
                mixed_sum.raw.clone(),
            ],
        })?;

        Ok(Bls12_381G1ProbeResult {
            normalized: self.read_values(&normalized, projective.len())?,
            doubled: self.read_values(&doubled, projective.len())?,
            mixed_sum: self.read_values(&mixed_sum, projective.len())?,
        })
    }

    /// Executes complete batched projective-plus-projective G1 addition.
    ///
    /// Inputs use the public BLST-compatible Jacobian ABI and outputs are
    /// canonical affine points. The kernel converts to its Metal-only
    /// homogeneous representation, applies RCB Algorithm 7, converts back to
    /// Jacobian, and normalizes. This diagnostic is not a proof or MSM route.
    pub fn probe_bls12_381_g1_complete_add(
        &self,
        lhs: &[Bls12_381G1Projective],
        rhs: &[Bls12_381G1Projective],
    ) -> Result<Vec<Bls12_381G1Affine>, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        if lhs.is_empty() {
            return Ok(Vec::new());
        }

        let length = u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let _execution = self.execution_guard();
        let lhs_input = self.buffer_from_slice(lhs, "midnight.g1.add_lhs")?;
        let rhs_input = self.buffer_from_slice(rhs, "midnight.g1.add_rhs")?;
        let length_buffer = self.buffer_from_slice(&[length], "midnight.g1.add_length")?;
        let projective_output = self.buffer(size_of_val(lhs), "midnight.g1.add_projective")?;
        let affine_bytes = lhs
            .len()
            .checked_mul(size_of::<Bls12_381G1Affine>())
            .ok_or(MetalError::LengthOverflow(lhs.len()))?;
        let affine_output = self.buffer(affine_bytes, "midnight.g1.add_affine")?;

        let command = self.command_buffer("midnight.bls12_381.g1_complete_add")?;
        let add_encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        add_encoder.setComputePipelineState(&self.g1.add);
        // SAFETY: the retained zero-offset buffers have the exact raw Jacobian
        // and u32 layouts required by the complete-add kernel.
        unsafe {
            add_encoder.setBuffer_offset_atIndex(Some(&lhs_input.raw), 0, 0);
            add_encoder.setBuffer_offset_atIndex(Some(&rhs_input.raw), 0, 1);
            add_encoder.setBuffer_offset_atIndex(Some(&projective_output.raw), 0, 2);
            add_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
        }
        self.dispatch(&add_encoder, &self.g1.add, lhs.len());
        add_encoder.endEncoding();

        let affine_encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        affine_encoder.setComputePipelineState(&self.g1.to_affine);
        // SAFETY: the first encoder writes a retained Jacobian buffer that the
        // second encoder reads into an equally retained affine output buffer.
        unsafe {
            affine_encoder.setBuffer_offset_atIndex(Some(&projective_output.raw), 0, 0);
            affine_encoder.setBuffer_offset_atIndex(Some(&affine_output.raw), 0, 1);
            affine_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
        }
        self.dispatch(&affine_encoder, &self.g1.to_affine, lhs.len());
        affine_encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                lhs_input.raw.clone(),
                rhs_input.raw.clone(),
                length_buffer.raw.clone(),
                projective_output.raw.clone(),
                affine_output.raw.clone(),
            ],
        })?;
        self.read_values(&affine_output, lhs.len())
    }

    /// Measures repeated complete additions entirely in homogeneous storage.
    ///
    /// Setup converts both public-Jacobian inputs before timing. One timed
    /// command then executes `timed_dispatches` identical grids, with each
    /// thread retaining its accumulator for `additions_per_chain` Algorithm 7
    /// additions. Egress conversion and host readback happen after timing. This
    /// is a conservative arithmetic feasibility diagnostic, not a proof route.
    pub fn probe_bls12_381_g1_complete_add_chain(
        &self,
        lhs: &[Bls12_381G1Projective],
        rhs: &[Bls12_381G1Projective],
        additions_per_chain: u32,
        timed_dispatches: u32,
    ) -> Result<Bls12_381G1ChainProbeResult, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        if additions_per_chain == 0 {
            return Err(MetalError::Dispatch(
                "G1 complete-add chain length must be nonzero",
            ));
        }
        if timed_dispatches == 0 {
            return Err(MetalError::Dispatch(
                "G1 complete-add timed dispatch count must be nonzero",
            ));
        }
        if lhs.is_empty() {
            return Ok(Bls12_381G1ChainProbeResult::default());
        }

        let length = u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let bytes = size_of_val(lhs);
        let _execution = self.execution_guard();
        let lhs_input = self.buffer_from_slice(lhs, "midnight.g1.chain_lhs_jacobian")?;
        let rhs_input = self.buffer_from_slice(rhs, "midnight.g1.chain_rhs_jacobian")?;
        let length_buffer = self.buffer_from_slice(&[length], "midnight.g1.chain_length")?;
        let additions_buffer =
            self.buffer_from_slice(&[additions_per_chain], "midnight.g1.chain_additions")?;
        // Startup ABI validation proves that all three allocations also have
        // the internal homogeneous point's size and alignment.
        let lhs_homogeneous = self.buffer(bytes, "midnight.g1.chain_lhs_homogeneous")?;
        let rhs_homogeneous = self.buffer(bytes, "midnight.g1.chain_rhs_homogeneous")?;
        let output_homogeneous = self.buffer(bytes, "midnight.g1.chain_output_homogeneous")?;
        let output_projective = self.buffer(bytes, "midnight.g1.chain_output_jacobian")?;

        let setup = self.command_buffer("midnight.bls12_381.g1_chain_setup")?;
        let setup_encoder = setup.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        setup_encoder.setComputePipelineState(&self.g1.to_homogeneous);
        for (input, output) in [
            (&lhs_input.raw, &lhs_homogeneous.raw),
            (&rhs_input.raw, &rhs_homogeneous.raw),
        ] {
            // SAFETY: each input is public-Jacobian storage, each output is an
            // ABI-validated homogeneous allocation, and all buffers are retained.
            unsafe {
                setup_encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                setup_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
                setup_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
            }
            self.dispatch(&setup_encoder, &self.g1.to_homogeneous, lhs.len());
        }
        setup_encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command: setup,
            _resources: vec![
                lhs_input.raw.clone(),
                rhs_input.raw.clone(),
                length_buffer.raw.clone(),
                lhs_homogeneous.raw.clone(),
                rhs_homogeneous.raw.clone(),
            ],
        })?;

        let timed = self.command_buffer("midnight.bls12_381.g1_complete_add_chain")?;
        let timed_encoder = timed.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        timed_encoder.setComputePipelineState(&self.g1.complete_add_chain);
        // SAFETY: setup completed before this command was created; all buffers
        // have the homogeneous/u32 layouts required by the chain kernel.
        unsafe {
            timed_encoder.setBuffer_offset_atIndex(Some(&lhs_homogeneous.raw), 0, 0);
            timed_encoder.setBuffer_offset_atIndex(Some(&rhs_homogeneous.raw), 0, 1);
            timed_encoder.setBuffer_offset_atIndex(Some(&output_homogeneous.raw), 0, 2);
            timed_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
            timed_encoder.setBuffer_offset_atIndex(Some(&additions_buffer.raw), 0, 4);
        }
        for _ in 0..timed_dispatches {
            self.dispatch(&timed_encoder, &self.g1.complete_add_chain, lhs.len());
        }
        timed_encoder.endEncoding();
        let started = Instant::now();
        self.submit_and_wait(PendingCommand {
            command: timed,
            _resources: vec![
                lhs_homogeneous.raw.clone(),
                rhs_homogeneous.raw.clone(),
                output_homogeneous.raw.clone(),
                length_buffer.raw.clone(),
                additions_buffer.raw.clone(),
            ],
        })?;
        let timed_duration = started.elapsed();

        let egress = self.command_buffer("midnight.bls12_381.g1_chain_egress")?;
        let egress_encoder = egress.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        egress_encoder.setComputePipelineState(&self.g1.from_homogeneous);
        // SAFETY: the timed command completed its homogeneous output; the
        // retained destination has the public Jacobian layout.
        unsafe {
            egress_encoder.setBuffer_offset_atIndex(Some(&output_homogeneous.raw), 0, 0);
            egress_encoder.setBuffer_offset_atIndex(Some(&output_projective.raw), 0, 1);
            egress_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
        }
        self.dispatch(&egress_encoder, &self.g1.from_homogeneous, lhs.len());
        egress_encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command: egress,
            _resources: vec![
                output_homogeneous.raw.clone(),
                output_projective.raw.clone(),
                length_buffer.raw.clone(),
            ],
        })?;

        Ok(Bls12_381G1ChainProbeResult {
            projective: self.read_values(&output_projective, lhs.len())?,
            timed_duration,
            timed_commands: 1,
            timed_dispatches,
        })
    }

    /// Verifies the Metal-only homogeneous identity-fill kernel.
    ///
    /// Homogeneous infinity is `(0, R, 0)`, not an all-zero buffer. This probe
    /// fills fresh storage on the GPU and compares the raw representation on the
    /// host without adding the internal point type to the public ABI.
    pub fn probe_bls12_381_g1_homogeneous_identity_fill(
        &self,
        count: usize,
    ) -> Result<bool, MetalError> {
        if count == 0 {
            return Ok(true);
        }

        let length = u32::try_from(count).map_err(|_| MetalError::LengthOverflow(count))?;
        let bytes = count
            .checked_mul(size_of::<Bls12_381G1Projective>())
            .ok_or(MetalError::LengthOverflow(count))?;
        let _execution = self.execution_guard();
        let output = self.buffer(bytes, "midnight.g1.homogeneous_identity")?;
        let length_buffer =
            self.buffer_from_slice(&[length], "midnight.g1.homogeneous_identity_length")?;
        let command = self.command_buffer("midnight.bls12_381.g1_homogeneous_identity_fill")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.g1.identity_fill);
        // SAFETY: session initialization verifies that the internal homogeneous
        // layout has the public Jacobian layout's size/alignment, and both
        // retained zero-offset buffers satisfy the shader ABI.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 1);
        }
        self.dispatch(&encoder, &self.g1.identity_fill, count);
        encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![output.raw.clone(), length_buffer.raw.clone()],
        })?;

        let expected = Bls12_381G1Projective {
            x: [0; 6],
            y: generated::FP_R,
            z: [0; 6],
        };
        Ok(self
            .read_values::<Bls12_381G1Projective>(&output, count)?
            .into_iter()
            .all(|point| point == expected))
    }

    /// Executes natural-order batched Fr NTTs while keeping every pass resident.
    ///
    /// Packed input batches each contain exactly `2^domain.log_size` raw
    /// Montgomery elements. The only host read occurs after the submitted
    /// command has completed successfully. This remains a correctness probe and
    /// is not routed into proof execution.
    pub fn probe_bls12_381_ntt(
        &self,
        values: &[Bls12_381Fr],
        domain: NttDomainSpec,
        direction: NttDirection,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        let domain_len = validate_ntt_domain(domain)?;
        let batch_count = complete_ntt_batches(values.len(), domain_len)?;
        if batch_count == 0 {
            return self.empty_resident_output();
        }
        let input_domain = match direction {
            NttDirection::Forward => FrDomainTag::coefficient(domain.log_size),
            NttDirection::Inverse => FrDomainTag::for_ntt(domain, FrBasis::Lagrange),
        };

        self.with_resident_fr_arena(
            values,
            domain_len,
            domain_len,
            batch_count,
            input_domain,
            |arena, batch| {
                let plan = self.ntt_plan(domain)?;
                let command = self.command_buffer("midnight.bls12_381.ntt")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let mut resources = arena.retained_resources();
                let output = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    batch,
                    &plan,
                    direction,
                    false,
                )?;
                encoder.endEncoding();

                let submitted = self.submit(PendingCommand {
                    command,
                    _resources: resources,
                });
                self.wait_for_completion(submitted)?;
                self.materialize_resident_fr(arena, output)
            },
        )
    }

    /// Executes packed canonical-domain Fr transforms for an accelerator route.
    ///
    /// This host-round-trip boundary is intentionally narrower than the future
    /// resident prover. It derives all roots internally, rejects unsupported
    /// sizes, and never substitutes CPU arithmetic after Metal is selected.
    pub fn execute_bls12_381_ntt(
        &self,
        values: &[Bls12_381Fr],
        log_size: u32,
        direction: NttDirection,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        self.probe_bls12_381_ntt(values, bls12_381_ntt_domain(log_size)?, direction)
    }

    /// Executes forward and inverse NTTs in one resident command and reads once.
    ///
    /// This diagnostic makes ping-pong ownership, queue ordering, and exactly-once
    /// inverse normalization observable without exposing backend buffer handles.
    pub fn probe_bls12_381_ntt_roundtrip(
        &self,
        values: &[Bls12_381Fr],
        domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        let domain_len = validate_ntt_domain(domain)?;
        let batch_count = complete_ntt_batches(values.len(), domain_len)?;
        if batch_count == 0 {
            return self.empty_resident_output();
        }

        self.with_resident_fr_arena(
            values,
            domain_len,
            domain_len,
            batch_count,
            FrDomainTag::coefficient(domain.log_size),
            |arena, input| {
                let plan = self.ntt_plan(domain)?;
                let command = self.command_buffer("midnight.bls12_381.ntt_roundtrip")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let mut resources = arena.retained_resources();
                let evaluations = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    input,
                    &plan,
                    NttDirection::Forward,
                    false,
                )?;
                let coefficients = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    evaluations,
                    &plan,
                    NttDirection::Inverse,
                    false,
                )?;
                encoder.endEncoding();

                let submitted = self.submit(PendingCommand {
                    command,
                    _resources: resources,
                });
                self.wait_for_completion(submitted)?;
                self.materialize_resident_fr(arena, coefficients)
            },
        )
    }

    /// Zero-extends coefficient batches, enters the order-three coset, and NTTs.
    ///
    /// Each packed source polynomial has `coefficients_per_batch` elements and
    /// each result has the extended domain size. Coset ingress, zero fill, and
    /// every Stockham pass remain in the same command and arena.
    pub fn probe_bls12_381_coeff_to_extended(
        &self,
        coefficients: &[Bls12_381Fr],
        coefficients_per_batch: usize,
        extended_domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        let extended_len = validate_ntt_domain(extended_domain)?;
        if coefficients_per_batch == 0
            || !coefficients_per_batch.is_power_of_two()
            || coefficients_per_batch > extended_len
            || !coefficients.len().is_multiple_of(coefficients_per_batch)
        {
            return Err(MetalError::NttCoefficientBatch {
                elements: coefficients.len(),
                coefficients_per_batch,
                extended_len,
            });
        }
        let batch_count = coefficients.len() / coefficients_per_batch;
        if batch_count == 0 {
            return self.empty_resident_output();
        }

        self.with_resident_fr_arena(
            coefficients,
            coefficients_per_batch,
            extended_len,
            batch_count,
            FrDomainTag::coefficient(coefficients_per_batch.ilog2()),
            |arena, input| {
                let plan = self.ntt_plan(extended_domain)?;
                let command = self.command_buffer("midnight.bls12_381.coeff_to_extended")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let mut resources = arena.retained_resources();
                let coset_coefficients = self.encode_coset_prepare(
                    &encoder,
                    arena,
                    input,
                    coefficients_per_batch,
                    extended_len,
                    extended_domain,
                )?;
                let evaluations = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    coset_coefficients,
                    &plan,
                    NttDirection::Forward,
                    false,
                )?;
                encoder.endEncoding();

                let submitted = self.submit(PendingCommand {
                    command,
                    _resources: resources,
                });
                self.wait_for_completion(submitted)?;
                self.materialize_resident_fr(arena, evaluations)
            },
        )
    }

    /// Zero-extends packed coefficient polynomials and enters the canonical
    /// BLS12-381 order-three coset before a Metal NTT.
    pub fn execute_bls12_381_coeff_to_extended(
        &self,
        coefficients: &[Bls12_381Fr],
        coefficients_per_batch: usize,
        extended_log_size: u32,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        self.probe_bls12_381_coeff_to_extended(
            coefficients,
            coefficients_per_batch,
            bls12_381_ntt_domain(extended_log_size)?,
        )
    }

    /// Inverse-NTTs extended coset evaluations and exits the order-three coset.
    ///
    /// Inverse-size normalization and coset egress are fused into the final
    /// Stockham write, so neither can be repeated between resident phases.
    pub fn probe_bls12_381_extended_to_coeff(
        &self,
        evaluations: &[Bls12_381Fr],
        extended_domain: NttDomainSpec,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        let extended_len = validate_ntt_domain(extended_domain)?;
        let batch_count = complete_ntt_batches(evaluations.len(), extended_len)?;
        if batch_count == 0 {
            return self.empty_resident_output();
        }

        self.with_resident_fr_arena(
            evaluations,
            extended_len,
            extended_len,
            batch_count,
            FrDomainTag::for_ntt(extended_domain, FrBasis::ExtendedCoset),
            |arena, input| {
                let plan = self.ntt_plan(extended_domain)?;
                let command = self.command_buffer("midnight.bls12_381.extended_to_coeff")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let mut resources = arena.retained_resources();
                let coefficients = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    input,
                    &plan,
                    NttDirection::Inverse,
                    true,
                )?;
                encoder.endEncoding();

                let submitted = self.submit(PendingCommand {
                    command,
                    _resources: resources,
                });
                self.wait_for_completion(submitted)?;
                self.materialize_resident_fr(arena, coefficients)
            },
        )
    }

    fn probe_field64<T: MetalPod + Default>(
        &self,
        lhs: &[T],
        rhs: &[T],
        pipelines: [&Pipeline; 4],
        field_label: &str,
    ) -> Result<[Vec<T>; 4], MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        if lhs.is_empty() {
            return Ok(std::array::from_fn(|_| Vec::new()));
        }
        let length = u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let _execution = self.execution_guard();

        let lhs_buffer = self.buffer_from_slice(lhs, &format!("midnight.{field_label}.lhs"))?;
        let rhs_buffer = self.buffer_from_slice(rhs, &format!("midnight.{field_label}.rhs"))?;
        let length_buffer =
            self.buffer_from_slice(&[length], &format!("midnight.{field_label}.length"))?;
        let output_bytes = size_of_val(lhs);
        let add = self.buffer(output_bytes, &format!("midnight.{field_label}.add"))?;
        let sub = self.buffer(output_bytes, &format!("midnight.{field_label}.sub"))?;
        let mul = self.buffer(output_bytes, &format!("midnight.{field_label}.mul"))?;
        let square = self.buffer(output_bytes, &format!("midnight.{field_label}.square"))?;

        let command = self.command_buffer(&format!("midnight.bls12_381.{field_label}_probe"))?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        for (pipeline, output) in
            pipelines
                .into_iter()
                .zip([&add.raw, &sub.raw, &mul.raw, &square.raw])
        {
            encoder.setComputePipelineState(pipeline);
            // SAFETY: every bound buffer is retained through command completion,
            // has the exact raw field/u32 representation expected by the shader,
            // and offset zero satisfies each type's alignment.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&lhs_buffer.raw), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&rhs_buffer.raw), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
            }
            self.dispatch(&encoder, pipeline, lhs.len());
        }
        encoder.endEncoding();

        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                lhs_buffer.raw.clone(),
                rhs_buffer.raw.clone(),
                length_buffer.raw.clone(),
                add.raw.clone(),
                sub.raw.clone(),
                mul.raw.clone(),
                square.raw.clone(),
            ],
        })?;

        Ok([
            self.read_values(&add, lhs.len())?,
            self.read_values(&sub, lhs.len())?,
            self.read_values(&mul, lhs.len())?,
            self.read_values(&square, lhs.len())?,
        ])
    }

    #[cfg(test)]
    fn benchmark_field_pipeline<T: MetalPod + Default>(
        &self,
        lhs: &[T],
        rhs: &[T],
        pipeline: &Pipeline,
        label: &str,
        rounds: usize,
    ) -> Result<std::time::Duration, MetalError> {
        assert!(!lhs.is_empty(), "throughput batches must not be empty");
        assert!(rounds != 0, "throughput rounds must not be zero");
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        let length = u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let _execution = self.execution_guard();
        let lhs_buffer = self.buffer_from_slice(lhs, &format!("midnight.bench.{label}.lhs"))?;
        let rhs_buffer = self.buffer_from_slice(rhs, &format!("midnight.bench.{label}.rhs"))?;
        let length_buffer =
            self.buffer_from_slice(&[length], &format!("midnight.bench.{label}.length"))?;
        let output = self.buffer(size_of_val(lhs), &format!("midnight.bench.{label}.output"))?;
        let command = self.command_buffer(&format!("midnight.bench.{label}"))?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(pipeline);
        // SAFETY: the typed inputs, output, and length buffer remain retained
        // until all benchmark dispatches finish, and all offsets are zero.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&lhs_buffer.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&rhs_buffer.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
        }
        for _ in 0..rounds {
            self.dispatch(&encoder, pipeline, lhs.len());
        }
        encoder.endEncoding();

        let start = std::time::Instant::now();
        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![
                lhs_buffer.raw.clone(),
                rhs_buffer.raw.clone(),
                length_buffer.raw.clone(),
                output.raw.clone(),
            ],
        })?;
        let elapsed = start.elapsed();
        let _: Vec<T> = self.read_values(&output, 1)?;
        Ok(elapsed)
    }

    #[cfg(test)]
    fn benchmark_fp32_mul_chain(
        &self,
        lhs: &[Bls12_381Fp],
        rhs: &[Bls12_381Fp],
        benchmark: Fp32ChainBenchmark<'_>,
    ) -> Result<(Vec<Bls12_381Fp>, std::time::Duration), MetalError> {
        let Fp32ChainBenchmark {
            pipeline,
            variant,
            iterations,
            timed_dispatches,
            threadgroup_width,
        } = benchmark;
        assert!(!lhs.is_empty(), "Fp32 benchmark batches must not be empty");
        assert!(iterations != 0, "Fp32 chain iterations must not be zero");
        assert!(
            timed_dispatches != 0,
            "Fp32 timed dispatches must not be zero"
        );
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        if threadgroup_width == 0
            || threadgroup_width > pipeline.maxTotalThreadsPerThreadgroup()
            || !threadgroup_width.is_multiple_of(pipeline.threadExecutionWidth())
        {
            return Err(MetalError::Dispatch(
                "Fp32 threadgroup width violates the pipeline limits",
            ));
        }

        let length = u32::try_from(lhs.len()).map_err(|_| MetalError::LengthOverflow(lhs.len()))?;
        let bytes = size_of_val(lhs);
        let _execution = self.execution_guard();
        let lhs_public = self.buffer_from_slice(lhs, "midnight.fp32.chain_lhs_public")?;
        let rhs_public = self.buffer_from_slice(rhs, "midnight.fp32.chain_rhs_public")?;
        let length_buffer = self.buffer_from_slice(&[length], "midnight.fp32.chain_length")?;
        let iterations_buffer =
            self.buffer_from_slice(&[iterations], "midnight.fp32.chain_iterations")?;
        let lhs_internal = self.buffer(bytes, "midnight.fp32.chain_lhs_internal")?;
        let rhs_internal = self.buffer(bytes, "midnight.fp32.chain_rhs_internal")?;
        let output_internal = self.buffer(bytes, "midnight.fp32.chain_output_internal")?;
        let output_public = self.buffer(bytes, "midnight.fp32.chain_output_public")?;

        let setup = self.command_buffer("midnight.bls12_381.fp32_chain_setup")?;
        let setup_encoder = setup.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        setup_encoder.setComputePipelineState(&self.field.fp32.from_public);
        for (input, output) in [
            (&lhs_public.raw, &lhs_internal.raw),
            (&rhs_public.raw, &rhs_internal.raw),
        ] {
            // SAFETY: the source is public six-u64 Fp storage; the destination
            // is a same-byte-length internal 12-u32 allocation retained through
            // completion, and the length buffer contains one u32.
            unsafe {
                setup_encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                setup_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
                setup_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
            }
            self.dispatch(&setup_encoder, &self.field.fp32.from_public, lhs.len());
        }
        setup_encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command: setup,
            _resources: vec![
                lhs_public.raw.clone(),
                rhs_public.raw.clone(),
                length_buffer.raw.clone(),
                lhs_internal.raw.clone(),
                rhs_internal.raw.clone(),
            ],
        })?;

        let timed = self.command_buffer(&format!("midnight.bls12_381.{variant}_mul_chain"))?;
        let timed_encoder = timed.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        timed_encoder.setComputePipelineState(pipeline);
        // SAFETY: setup completed and all retained zero-offset buffers have the
        // internal Fp32 or u32 layouts required by the chain kernel.
        unsafe {
            timed_encoder.setBuffer_offset_atIndex(Some(&lhs_internal.raw), 0, 0);
            timed_encoder.setBuffer_offset_atIndex(Some(&rhs_internal.raw), 0, 1);
            timed_encoder.setBuffer_offset_atIndex(Some(&output_internal.raw), 0, 2);
            timed_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 3);
            timed_encoder.setBuffer_offset_atIndex(Some(&iterations_buffer.raw), 0, 4);
        }
        for _ in 0..timed_dispatches {
            timed_encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: lhs.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_width,
                    height: 1,
                    depth: 1,
                },
            );
            self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
        }
        timed_encoder.endEncoding();
        let started = Instant::now();
        self.submit_and_wait(PendingCommand {
            command: timed,
            _resources: vec![
                lhs_internal.raw.clone(),
                rhs_internal.raw.clone(),
                output_internal.raw.clone(),
                length_buffer.raw.clone(),
                iterations_buffer.raw.clone(),
            ],
        })?;
        let elapsed = started.elapsed();

        let egress = self.command_buffer("midnight.bls12_381.fp32_chain_egress")?;
        let egress_encoder = egress.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        egress_encoder.setComputePipelineState(&self.field.fp32.to_public);
        // SAFETY: the completed internal output and public destination are
        // retained and have the exact layouts required by the conversion kernel.
        unsafe {
            egress_encoder.setBuffer_offset_atIndex(Some(&output_internal.raw), 0, 0);
            egress_encoder.setBuffer_offset_atIndex(Some(&output_public.raw), 0, 1);
            egress_encoder.setBuffer_offset_atIndex(Some(&length_buffer.raw), 0, 2);
        }
        self.dispatch(&egress_encoder, &self.field.fp32.to_public, lhs.len());
        egress_encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command: egress,
            _resources: vec![
                output_internal.raw.clone(),
                output_public.raw.clone(),
                length_buffer.raw.clone(),
            ],
        })?;

        Ok((self.read_values(&output_public, lhs.len())?, elapsed))
    }

    #[cfg(test)]
    fn benchmark_ntt_roundtrips(
        &self,
        values: &[Bls12_381Fr],
        domain: NttDomainSpec,
        rounds: usize,
    ) -> Result<std::time::Duration, MetalError> {
        assert!(
            !values.is_empty(),
            "NTT throughput batches must not be empty"
        );
        assert!(rounds != 0, "NTT throughput rounds must not be zero");
        let domain_len = validate_ntt_domain(domain)?;
        let batch_count = complete_ntt_batches(values.len(), domain_len)?;
        let _execution = self.execution_guard();
        let plan = self.ntt_plan(domain)?;
        let (mut arena, mut batch) = self.upload_resident_fr(
            values,
            domain_len,
            domain_len,
            batch_count,
            FrDomainTag::coefficient(domain.log_size),
        )?;
        let operation = (|| {
            let command = self.command_buffer("midnight.bench.ntt_roundtrips")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            let mut resources = arena.retained_resources();
            for _ in 0..rounds {
                batch = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    &mut arena,
                    batch,
                    &plan,
                    NttDirection::Forward,
                    false,
                )?;
                batch = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    &mut arena,
                    batch,
                    &plan,
                    NttDirection::Inverse,
                    false,
                )?;
            }
            encoder.endEncoding();

            let start = std::time::Instant::now();
            let submitted = self.submit(PendingCommand {
                command,
                _resources: resources,
            });
            self.wait_for_completion(submitted)?;
            let elapsed = start.elapsed();
            if self.materialize_resident_fr(&arena, batch)? != values {
                return Err(MetalError::Execution(
                    "resident NTT benchmark roundtrip mismatch".to_owned(),
                ));
            }
            Ok(elapsed)
        })();
        arena.finish(operation)
    }

    fn ntt_plan(&self, domain: NttDomainSpec) -> Result<NttPlan, MetalError> {
        validate_ntt_domain(domain)?;
        {
            let plans = self
                .ntt_plans
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(plan) = plans.get(&domain) {
                return Ok(plan.clone());
            }
        }

        let candidate = NttPlan::create(self, domain)?;
        let mut plans = self
            .ntt_plans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(plans.entry(domain).or_insert(candidate).clone())
    }

    fn upload_resident_fr(
        &self,
        values: &[Bls12_381Fr],
        input_len: usize,
        capacity_stride: usize,
        batch_count: usize,
        domain: FrDomainTag,
    ) -> Result<(ResidentFrArena<'_>, ResidentFrView), MetalError> {
        if input_len == 0
            || input_len > capacity_stride
            || input_len.checked_mul(batch_count) != Some(values.len())
        {
            return Err(MetalError::ResidentState(
                "resident upload dimensions are inconsistent",
            ));
        }
        validate_fr_domain_shape(domain, input_len)?;
        if matches!(
            domain.basis,
            FrBasis::CosetCoefficient | FrBasis::StockhamWorking { .. }
        ) {
            return Err(MetalError::ResidentState(
                "host uploads cannot claim an internal NTT transition basis",
            ));
        }
        let elements = capacity_stride
            .checked_mul(batch_count)
            .ok_or(MetalError::LengthOverflow(capacity_stride))?;
        let bytes = elements
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::LengthOverflow(elements))?;
        let arena_id = self.next_resident_arena_id()?;
        let mut arena = ResidentFrArena {
            session: self,
            arena_id,
            slots: Vec::with_capacity(2),
            write_generations: [1, 0],
            slot_states: [None, None],
        };
        arena
            .slots
            .push(self.secret_buffer(bytes, "midnight.ntt.arena.0")?);
        arena
            .slots
            .push(self.secret_buffer(bytes, "midnight.ntt.arena.1")?);

        for batch in 0..batch_count {
            // SAFETY: the checked dimensions place every source batch inside
            // `values` and every destination batch inside the shared allocation.
            // `Bls12_381Fr` is a sealed, padding-free MetalPod type.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr().add(batch * input_len),
                    arena.slots[0]
                        .raw
                        .contents()
                        .as_ptr()
                        .cast::<Bls12_381Fr>()
                        .add(batch * input_len),
                    input_len,
                );
            }
        }
        self.activity
            .host_write_bytes
            .fetch_add(size_of_val(values) as u64, Ordering::Relaxed);

        arena.slot_states[0] = Some(ResidentFrSlotState::new(
            1,
            0,
            input_len,
            input_len,
            batch_count,
            domain,
        ));
        let view = arena.view_for_slot(0, 1, 0, input_len, input_len, batch_count, domain)?;
        Ok((arena, view))
    }

    fn with_resident_fr_arena<T, F>(
        &self,
        values: &[Bls12_381Fr],
        input_len: usize,
        capacity_stride: usize,
        batch_count: usize,
        domain: FrDomainTag,
        operation: F,
    ) -> Result<T, MetalError>
    where
        F: FnOnce(&mut ResidentFrArena<'_>, ResidentFrView) -> Result<T, MetalError>,
    {
        let execution = self.execution_guard();
        self.ensure_session_healthy()?;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let (mut arena, input) =
                self.upload_resident_fr(values, input_len, capacity_stride, batch_count, domain)?;
            let operation = operation(&mut arena, input);
            arena.finish(operation)
        }));
        drop(execution);
        match outcome {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }

    fn empty_resident_output<T>(&self) -> Result<Vec<T>, MetalError> {
        // Zero-work calls still participate in the process-level execution
        // epoch so a concurrent or prior erasure failure cannot be bypassed.
        let _execution = self.execution_guard();
        self.ensure_session_healthy()?;
        Ok(Vec::new())
    }

    fn encode_coset_prepare(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arena: &mut ResidentFrArena,
        input: ResidentFrView,
        input_len: usize,
        output_len: usize,
        domain: NttDomainSpec,
    ) -> Result<ResidentFrView, MetalError> {
        arena.validate(&input)?;
        if output_len == 0
            || !output_len.is_power_of_two()
            || input.logical_len != input_len
            || input.domain.basis != FrBasis::Coefficient
            || input.domain.element_count()? != input_len
            || domain.log_size != output_len.ilog2()
        {
            return Err(MetalError::ResidentState(
                "coset transition basis/domain does not match the resident batch",
            ));
        }
        let output = arena.transition(
            &input,
            output_len,
            output_len,
            FrDomainTag::for_ntt(domain, FrBasis::CosetCoefficient),
        )?;
        let parameters = NttCosetParameters {
            input_length: u32::try_from(input_len)
                .map_err(|_| MetalError::LengthOverflow(input_len))?,
            output_length: u32::try_from(output_len)
                .map_err(|_| MetalError::LengthOverflow(output_len))?,
            source_stride: u32::try_from(input.stride)
                .map_err(|_| MetalError::LengthOverflow(input.stride))?,
            destination_stride: u32::try_from(output.stride)
                .map_err(|_| MetalError::LengthOverflow(output.stride))?,
            batch_count: u32::try_from(input.batch_count)
                .map_err(|_| MetalError::LengthOverflow(input.batch_count))?,
            reserved: 0,
            coset_generator: domain.coset_generator,
            inverse_coset_generator: domain.inverse_coset_generator,
        };

        encoder.setComputePipelineState(&self.ntt.coset_prepare);
        // SAFETY: both arena slots are retained by the pending command, checked
        // view offsets lie on Fr element boundaries, and Metal copies
        // `parameters` during encoding. The repr(C) structure has no padding gap.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&arena.slots[input.slot].raw),
                input.byte_offset,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(&arena.slots[output.slot].raw),
                output.byte_offset,
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<NttCosetParameters>(),
                2,
            );
        }

        let threads = NTT_COSET_THREADS
            .min(self.ntt.coset_prepare.maxTotalThreadsPerThreadgroup())
            .max(1);
        let groups = output_len.div_ceil(threads);
        self.dispatch_threadgroups(
            encoder,
            &self.ntt.coset_prepare,
            MTLSize {
                width: groups,
                height: input.batch_count,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ntt(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        resources: &mut Vec<Buffer>,
        arena: &mut ResidentFrArena,
        input: ResidentFrView,
        plan: &NttPlan,
        direction: NttDirection,
        coset_egress: bool,
    ) -> Result<ResidentFrView, MetalError> {
        arena.validate(&input)?;
        if input.logical_len != plan.domain_len || input.domain.log_size != plan.domain.log_size {
            return Err(MetalError::ResidentState(
                "NTT plan and resident basis/domain lengths differ",
            ));
        }
        let output_domain = match (
            direction,
            coset_egress,
            input.domain.basis,
            input.domain.ntt_domain,
        ) {
            (NttDirection::Forward, false, FrBasis::Coefficient, None) => {
                FrDomainTag::for_ntt(plan.domain, FrBasis::Lagrange)
            }
            (NttDirection::Forward, false, FrBasis::CosetCoefficient, Some(domain))
                if domain == plan.domain =>
            {
                FrDomainTag::for_ntt(plan.domain, FrBasis::ExtendedCoset)
            }
            (NttDirection::Inverse, false, FrBasis::Lagrange, Some(domain))
                if domain == plan.domain =>
            {
                FrDomainTag::coefficient(plan.domain.log_size)
            }
            (NttDirection::Inverse, true, FrBasis::ExtendedCoset, Some(domain))
                if domain == plan.domain =>
            {
                FrDomainTag::coefficient(plan.domain.log_size)
            }
            _ => {
                return Err(MetalError::ResidentState(
                    "NTT direction/coset mode does not accept the resident Fr basis",
                ));
            }
        };

        // A size-one transform is still encoded as a resident device operation.
        // The inverse scale (which equals one) is deliberately applied here so
        // the exactly-once normalization contract has no edge-size exception.
        if plan.schedule.is_empty() {
            let output = arena.transition(&input, 1, 1, output_domain)?;
            let mut flags = 0;
            if direction == NttDirection::Inverse {
                flags |= NTT_FLAG_NORMALIZE;
            }
            if coset_egress {
                flags |= NTT_FLAG_COSET_EGRESS;
            }
            let parameters = NttPassParameters {
                n: 1,
                source_stride: u32::try_from(input.stride)
                    .map_err(|_| MetalError::LengthOverflow(input.stride))?,
                destination_stride: u32::try_from(output.stride)
                    .map_err(|_| MetalError::LengthOverflow(output.stride))?,
                log_p: 0,
                degree: 0,
                max_degree: 0,
                batch_count: u32::try_from(input.batch_count)
                    .map_err(|_| MetalError::LengthOverflow(input.batch_count))?,
                flags,
                reserved: [0; 2],
                one: Bls12_381Fr {
                    limbs: generated::FR_R,
                },
                inverse_size: plan.domain.inverse_size,
                coset_generator: plan.domain.coset_generator,
                inverse_coset_generator: plan.domain.inverse_coset_generator,
            };
            encoder.setComputePipelineState(&self.ntt.identity);
            // SAFETY: both resident Fr buffers are retained to completion and
            // Metal copies the fully initialized repr(C) parameter block.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&arena.slots[input.slot].raw),
                    input.byte_offset,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(&arena.slots[output.slot].raw),
                    output.byte_offset,
                    1,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&parameters).cast::<c_void>(),
                    size_of::<NttPassParameters>(),
                    2,
                );
            }
            self.dispatch_threadgroups(
                encoder,
                &self.ntt.identity,
                MTLSize {
                    width: 1,
                    height: input.batch_count,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )?;
            return Ok(output);
        }

        let twiddles = match direction {
            NttDirection::Forward => &plan.forward,
            NttDirection::Inverse => &plan.inverse,
        };
        resources.push(twiddles.pq.raw.clone());
        resources.push(twiddles.omega_squares.raw.clone());

        let n = u32::try_from(plan.domain_len)
            .map_err(|_| MetalError::LengthOverflow(plan.domain_len))?;
        let batch_count = u32::try_from(input.batch_count)
            .map_err(|_| MetalError::LengthOverflow(input.batch_count))?;
        let mut current = input;
        let mut log_p = 0;
        for (pass_index, degree) in plan.schedule.iter().copied().enumerate() {
            if pass_index != 0
                && current.domain
                    != FrDomainTag::for_ntt(
                        plan.domain,
                        FrBasis::StockhamWorking {
                            direction,
                            completed_log_p: log_p,
                        },
                    )
            {
                return Err(MetalError::ResidentState(
                    "Stockham working view does not continue the current plan and pass",
                ));
            }
            let is_last = pass_index + 1 == plan.schedule.len();
            let completed_log_p = log_p
                .checked_add(degree)
                .ok_or(MetalError::ResidentState("NTT pass progress overflows u32"))?;
            let pass_domain = if is_last {
                output_domain
            } else {
                FrDomainTag::for_ntt(
                    plan.domain,
                    FrBasis::StockhamWorking {
                        direction,
                        completed_log_p,
                    },
                )
            };
            let output =
                arena.transition(&current, plan.domain_len, plan.domain_len, pass_domain)?;
            let mut flags = 0;
            if is_last && direction == NttDirection::Inverse {
                flags |= NTT_FLAG_NORMALIZE;
            }
            if is_last && coset_egress {
                flags |= NTT_FLAG_COSET_EGRESS;
            }
            let parameters = NttPassParameters {
                n,
                source_stride: u32::try_from(current.stride)
                    .map_err(|_| MetalError::LengthOverflow(current.stride))?,
                destination_stride: u32::try_from(output.stride)
                    .map_err(|_| MetalError::LengthOverflow(output.stride))?,
                log_p,
                degree,
                max_degree: plan.max_degree,
                batch_count,
                flags,
                reserved: [0; 2],
                one: Bls12_381Fr {
                    limbs: generated::FR_R,
                },
                inverse_size: plan.domain.inverse_size,
                coset_generator: plan.domain.coset_generator,
                inverse_coset_generator: plan.domain.inverse_coset_generator,
            };

            self.encode_ntt_stockham_pass(
                encoder,
                &arena.slots[current.slot].raw,
                current.byte_offset,
                &arena.slots[output.slot].raw,
                output.byte_offset,
                twiddles,
                &parameters,
            )?;
            log_p = completed_log_p;
            current = output;
        }
        if log_p != plan.domain.log_size {
            return Err(MetalError::ResidentState(
                "NTT radix schedule does not cover the domain",
            ));
        }
        if current.domain != output_domain {
            return Err(MetalError::ResidentState(
                "NTT output did not reach its promised Fr basis/domain",
            ));
        }
        Ok(current)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ntt_stockham_pass(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        source: &ProtocolObject<dyn MTLBuffer>,
        source_offset: usize,
        destination: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: usize,
        twiddles: &NttTwiddlePlan,
        parameters: &NttPassParameters,
    ) -> Result<(), MetalError> {
        encoder.setComputePipelineState(&self.ntt.stockham);
        // SAFETY: callers retain the source, destination, and immutable
        // twiddle buffers through completion. Metal copies the complete ABI-v2
        // parameter block during encoding.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(source), source_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(destination), destination_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(&twiddles.pq.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&twiddles.omega_squares.raw), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::from(parameters).cast::<c_void>(),
                size_of::<NttPassParameters>(),
                4,
            );
        }

        let degree = parameters.degree;
        let radix = 1usize << degree;
        let scratch_bytes = radix
            .checked_mul(size_of::<Bls12_381Fr>())
            .ok_or(MetalError::Dispatch("threadgroup memory size overflow"))?;
        let total_threadgroup_bytes = scratch_bytes
            .checked_add(self.ntt.stockham.staticThreadgroupMemoryLength())
            .ok_or(MetalError::Dispatch("threadgroup memory size overflow"))?;
        if total_threadgroup_bytes > self.device.maxThreadgroupMemoryLength() {
            return Err(MetalError::Dispatch(
                "NTT scratch memory exceeds the device limit",
            ));
        }
        // SAFETY: binding zero is the kernel's sole dynamic threadgroup
        // allocation and `scratch_bytes` was checked against the device cap.
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(scratch_bytes, 0);
        }
        self.dispatch_threadgroups(
            encoder,
            &self.ntt.stockham,
            MTLSize {
                width: (parameters.n as usize) >> degree,
                height: parameters.batch_count as usize,
                depth: 1,
            },
            MTLSize {
                width: radix >> 1,
                height: 1,
                depth: 1,
            },
        )
    }

    fn materialize_resident_fr(
        &self,
        arena: &ResidentFrArena<'_>,
        view: ResidentFrView,
    ) -> Result<Vec<Bls12_381Fr>, MetalError> {
        arena.validate(&view)?;
        self.activity
            .materializations
            .fetch_add(1, Ordering::Relaxed);
        self.activity
            .whole_polynomial_materializations
            .fetch_add(1, Ordering::Relaxed);
        let elements = view
            .logical_len
            .checked_mul(view.batch_count)
            .ok_or(MetalError::LengthOverflow(view.logical_len))?;
        if view.byte_offset == 0 && view.logical_len == view.stride {
            return self.read_secret_values(&arena.slots[view.slot], elements);
        }
        let mut values = vec![Bls12_381Fr::default(); elements];
        for batch_index in 0..view.batch_count {
            // SAFETY: both source and destination ranges are covered by the
            // checked arena and packed output dimensions; Fr permits all bits.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    arena.slots[view.slot]
                        .raw
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(view.byte_offset)
                        .cast::<Bls12_381Fr>()
                        .add(batch_index * view.stride),
                    values.as_mut_ptr().add(batch_index * view.logical_len),
                    view.logical_len,
                );
            }
        }
        self.activity
            .host_read_bytes
            .fetch_add(size_of_val(values.as_slice()) as u64, Ordering::Relaxed);
        Ok(values)
    }

    /// Replaces every supplied word with zero using the Metal clear primitive.
    ///
    /// This diagnostic makes clear-kernel behavior independently testable; it
    /// does not fall back to a host-side fill.
    pub fn clear_words_probe(&self, words: &[u64]) -> Result<Vec<u64>, MetalError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let _execution = self.execution_guard();
        let buffer = self.buffer_from_slice(words, "midnight.clear_words.probe")?;
        let command = self.command_buffer("midnight.clear_words.probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;

        encoder.setComputePipelineState(&self.clear_words);
        // SAFETY: the buffer is a retained, zero-offset array of `u64` words
        // and remains alive until the command has completed.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buffer.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.clear_words, words.len());
        encoder.endEncoding();
        self.submit_and_wait(PendingCommand {
            command,
            _resources: vec![buffer.raw.clone()],
        })?;
        self.read_values(&buffer, words.len())
    }

    /// Returns stable properties of the selected Metal device.
    pub fn device_info(&self) -> Result<MetalDeviceInfo, MetalError> {
        Ok(MetalDeviceInfo {
            name: self.device.name().to_string(),
            registry_id: self.device.registryID(),
            unified_memory: self.device.hasUnifiedMemory(),
            recommended_working_set_bytes: self.device.recommendedMaxWorkingSetSize(),
        })
    }

    fn lifecycle_guard(&self) -> MutexGuard<'_, LifecycleState> {
        match self.lifecycle.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if !matches!(state.health, SessionHealth::Poisoned(_)) {
                    state.health = SessionHealth::Poisoned(
                        "host panic poisoned the Metal lifecycle mutex".to_owned(),
                    );
                    self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
                }
                self.lifecycle.clear_poison();
                state
            }
        }
    }

    fn ensure_session_healthy(&self) -> Result<(), MetalError> {
        let state = self.lifecycle_guard();
        match &state.health {
            SessionHealth::Healthy => Ok(()),
            SessionHealth::Clearing => Err(MetalError::SessionPoisoned {
                reason: "secret-buffer clearing is already in progress".to_owned(),
            }),
            SessionHealth::Poisoned(reason) => Err(MetalError::SessionPoisoned {
                reason: reason.clone(),
            }),
        }
    }

    fn poison_session(&self, reason: String) {
        let mut state = self.lifecycle_guard();
        if !matches!(state.health, SessionHealth::Poisoned(_)) {
            state.health = SessionHealth::Poisoned(reason);
            self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn next_resident_arena_id(&self) -> Result<u64, MetalError> {
        let mut state = self.lifecycle_guard();
        match &state.health {
            SessionHealth::Healthy => {}
            SessionHealth::Clearing => {
                return Err(MetalError::SessionPoisoned {
                    reason: "secret-buffer clearing is already in progress".to_owned(),
                });
            }
            SessionHealth::Poisoned(reason) => {
                return Err(MetalError::SessionPoisoned {
                    reason: reason.clone(),
                });
            }
        }
        let id = state.next_arena_id;
        let Some(next) = id.checked_add(1) else {
            let reason = "resident arena identity counter is exhausted".to_owned();
            state.health = SessionHealth::Poisoned(reason.clone());
            self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
            return Err(MetalError::SessionPoisoned { reason });
        };
        state.next_arena_id = next;
        self.activity
            .resident_generations
            .fetch_add(1, Ordering::Relaxed);
        Ok(id)
    }

    fn secret_buffer(&self, bytes: usize, label: &str) -> Result<SecretBufferLease, MetalError> {
        self.ensure_session_healthy()?;
        let pooled = {
            let mut state = self.lifecycle_guard();
            let lease = state.pool.get_mut(&bytes).and_then(Vec::pop);
            if let Some(mut lease) = lease {
                let Some(generation) = lease.lease_generation.checked_add(1) else {
                    let reason = "secret buffer lease generation counter is exhausted".to_owned();
                    state.quarantined_bytes = state.quarantined_bytes.saturating_add(lease.bytes);
                    state.pooled_bytes = state.pooled_bytes.saturating_sub(lease.bytes);
                    state.quarantine.push(lease);
                    state.health = SessionHealth::Poisoned(reason.clone());
                    self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
                    self.sync_lifecycle_gauges(&state);
                    return Err(MetalError::SessionPoisoned { reason });
                };
                lease.lease_generation = generation;
                state.pooled_bytes = state.pooled_bytes.saturating_sub(bytes);
                self.activity.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.sync_lifecycle_gauges(&state);
                Some(lease)
            } else {
                self.activity.pool_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        };

        if let Some(lease) = pooled {
            lease.raw.setLabel(Some(&NSString::from_str(label)));
            return Ok(lease);
        }

        let buffer = self.buffer(bytes, label)?;
        let mut state = self.lifecycle_guard();
        let allocation_id = state.next_allocation_id;
        let Some(next) = allocation_id.checked_add(1) else {
            let reason = "secret buffer allocation identity counter is exhausted".to_owned();
            state.health = SessionHealth::Poisoned(reason.clone());
            self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
            return Err(MetalError::SessionPoisoned { reason });
        };
        state.next_allocation_id = next;
        state.owned_secret_bytes = state
            .owned_secret_bytes
            .checked_add(bytes)
            .ok_or(MetalError::LengthOverflow(bytes))?;
        self.sync_lifecycle_gauges(&state);
        Ok(SecretBufferLease {
            raw: buffer.raw,
            bytes,
            allocation_id,
            lease_generation: 1,
        })
    }

    fn sync_lifecycle_gauges(&self, state: &LifecycleState) {
        let live = state.owned_secret_bytes as u64;
        self.activity
            .secret_live_bytes
            .store(live, Ordering::Relaxed);
        self.activity
            .secret_peak_live_bytes
            .fetch_max(live, Ordering::Relaxed);
        self.activity
            .pooled_secret_bytes
            .store(state.pooled_bytes as u64, Ordering::Relaxed);
        self.activity
            .quarantined_secret_bytes
            .store(state.quarantined_bytes as u64, Ordering::Relaxed);
    }

    fn clear_and_recycle_secret_buffers(
        &self,
        buffers: Vec<SecretBufferLease>,
    ) -> Result<(), MetalError> {
        if buffers.is_empty() {
            return Ok(());
        }
        let unavailable = {
            let mut state = self.lifecycle_guard();
            match &state.health {
                SessionHealth::Healthy => {
                    state.health = SessionHealth::Clearing;
                    None
                }
                SessionHealth::Clearing => {
                    Some("nested secret-buffer clearing is forbidden".to_owned())
                }
                SessionHealth::Poisoned(reason) => Some(reason.clone()),
            }
        };
        if let Some(reason) = unavailable {
            let mut state = self.lifecycle_guard();
            state.quarantined_bytes = buffers
                .iter()
                .fold(state.quarantined_bytes, |bytes, buffer| {
                    bytes.saturating_add(buffer.bytes)
                });
            state.quarantine.extend(buffers);
            if !matches!(state.health, SessionHealth::Poisoned(_)) {
                state.health = SessionHealth::Poisoned(reason.clone());
                self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
            }
            self.sync_lifecycle_gauges(&state);
            return Err(MetalError::SessionPoisoned { reason });
        }

        let clear = self.clear_secret_buffers_on_gpu(&buffers);
        let mut state = self.lifecycle_guard();
        match clear {
            Ok(()) => {
                state.health = SessionHealth::Healthy;
                for buffer in buffers {
                    if state
                        .pooled_bytes
                        .checked_add(buffer.bytes)
                        .is_some_and(|bytes| bytes <= SECRET_BUFFER_POOL_MAX_BYTES)
                    {
                        state.pooled_bytes += buffer.bytes;
                        state.pool.entry(buffer.bytes).or_default().push(buffer);
                        self.activity.pool_returns.fetch_add(1, Ordering::Relaxed);
                    } else {
                        state.owned_secret_bytes =
                            state.owned_secret_bytes.saturating_sub(buffer.bytes);
                    }
                }
                self.sync_lifecycle_gauges(&state);
                Ok(())
            }
            Err(error) => {
                let reason = format!("secret-buffer GPU clear failed: {error}");
                state.quarantined_bytes = buffers
                    .iter()
                    .fold(state.quarantined_bytes, |bytes, buffer| {
                        bytes.saturating_add(buffer.bytes)
                    });
                state.quarantine.extend(buffers);
                state.health = SessionHealth::Poisoned(reason.clone());
                self.activity.clear_failures.fetch_add(1, Ordering::Relaxed);
                self.activity.poison_events.fetch_add(1, Ordering::Relaxed);
                self.sync_lifecycle_gauges(&state);
                Err(MetalError::SessionPoisoned { reason })
            }
        }
    }

    fn clear_secret_buffers_on_gpu(&self, buffers: &[SecretBufferLease]) -> Result<(), MetalError> {
        let mut cleared_bytes = 0usize;
        for buffer in buffers {
            if buffer.bytes == 0 || !buffer.bytes.is_multiple_of(size_of::<u64>()) {
                return Err(MetalError::ResidentState(
                    "secret buffer capacity is not a nonzero whole-word length",
                ));
            }
            cleared_bytes = cleared_bytes
                .checked_add(buffer.bytes)
                .ok_or(MetalError::LengthOverflow(buffer.bytes))?;
        }

        let command = self.command_buffer_unchecked("midnight.secret_buffer.clear")?;
        self.activity.clear_commands.fetch_add(1, Ordering::Relaxed);
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.clear_words);
        let mut resources = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            // SAFETY: every lease owns a zero-offset whole-u64 allocation and
            // both the lease and retained resource survive command completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buffer.raw), 0, 0);
            }
            self.dispatch(&encoder, &self.clear_words, buffer.bytes / size_of::<u64>());
            self.activity
                .clear_dispatches
                .fetch_add(1, Ordering::Relaxed);
            resources.push(buffer.raw.clone());
        }
        encoder.endEncoding();
        let injected = self
            .inject_next_clear_failure
            .swap(false, Ordering::SeqCst)
            .then_some("injected secret clear completion failure");
        let submitted = self.submit(PendingCommand {
            command,
            _resources: resources,
        });
        self.activity.clear_waits.fetch_add(1, Ordering::Relaxed);
        self.wait_for_completion_inner(submitted, injected)?;
        self.activity
            .cleared_bytes
            .fetch_add(cleared_bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Snapshots cumulative activity and current device allocation size.
    pub fn activity(&self) -> MetalActivity {
        let msm = msm_worker::activity();
        MetalActivity {
            command_buffers: self.activity.command_buffers.load(Ordering::Relaxed),
            dispatches: self.activity.dispatches.load(Ordering::Relaxed),
            host_waits: self.activity.host_waits.load(Ordering::Relaxed),
            allocations: self.activity.allocations.load(Ordering::Relaxed),
            allocated_bytes: self.activity.allocated_bytes.load(Ordering::Relaxed),
            host_write_bytes: self.activity.host_write_bytes.load(Ordering::Relaxed),
            host_read_bytes: self.activity.host_read_bytes.load(Ordering::Relaxed),
            blit_bytes: self.activity.blit_bytes.load(Ordering::Relaxed),
            current_allocated_bytes: self.device.currentAllocatedSize() as u64,
            execution_failures: self.activity.execution_failures.load(Ordering::Relaxed),
            msm_metal_submissions: msm.metal_submissions,
            msm_metal_outputs: msm.metal_outputs,
            msm_metal_logical_scalars: msm.metal_logical_scalars,
            msm_metal_nonzero_scalars: msm.metal_nonzero_scalars,
            msm_metal_failures: msm.metal_failures,
            msm_metal_service_ns: msm.metal_service_ns,
            msm_metal_coalesce_ns: msm.metal_coalesce_ns,
            msm_metal_canonicalization_ns: msm.metal_canonicalization_ns,
            msm_cpu_outputs: msm.cpu_outputs,
            msm_cpu_logical_scalars: msm.cpu_logical_scalars,
            msm_cpu_nonzero_scalars: msm.cpu_nonzero_scalars,
            msm_cpu_service_ns: msm.cpu_service_ns,
            msm_cpu_lock_wait_ns: msm.cpu_lock_wait_ns,
            pool_hits: self.activity.pool_hits.load(Ordering::Relaxed),
            pool_misses: self.activity.pool_misses.load(Ordering::Relaxed),
            pool_returns: self.activity.pool_returns.load(Ordering::Relaxed),
            clear_commands: self.activity.clear_commands.load(Ordering::Relaxed),
            clear_dispatches: self.activity.clear_dispatches.load(Ordering::Relaxed),
            clear_waits: self.activity.clear_waits.load(Ordering::Relaxed),
            cleared_bytes: self.activity.cleared_bytes.load(Ordering::Relaxed),
            clear_failures: self.activity.clear_failures.load(Ordering::Relaxed),
            poison_events: self.activity.poison_events.load(Ordering::Relaxed),
            resident_generations: self.activity.resident_generations.load(Ordering::Relaxed),
            stale_handle_rejections: self
                .activity
                .stale_handle_rejections
                .load(Ordering::Relaxed),
            materializations: self.activity.materializations.load(Ordering::Relaxed),
            whole_polynomial_materializations: self
                .activity
                .whole_polynomial_materializations
                .load(Ordering::Relaxed),
            secret_live_bytes: self.activity.secret_live_bytes.load(Ordering::Relaxed),
            secret_peak_live_bytes: self.activity.secret_peak_live_bytes.load(Ordering::Relaxed),
            pooled_secret_bytes: self.activity.pooled_secret_bytes.load(Ordering::Relaxed),
            quarantined_secret_bytes: self
                .activity
                .quarantined_secret_bytes
                .load(Ordering::Relaxed),
            key_cache_hits: self.activity.key_cache_hits.load(Ordering::Relaxed),
            key_cache_misses: self.activity.key_cache_misses.load(Ordering::Relaxed),
            key_context_builds: self.activity.key_context_builds.load(Ordering::Relaxed),
            key_context_build_failures: self
                .activity
                .key_context_build_failures
                .load(Ordering::Relaxed),
            quotient_epochs: self.activity.quotient_epochs.load(Ordering::Relaxed),
            quotient_epoch_elapsed_ns: self
                .activity
                .quotient_epoch_elapsed_ns
                .load(Ordering::Relaxed),
            quotient_epoch_failures: self
                .activity
                .quotient_epoch_failures
                .load(Ordering::Relaxed),
            quotient_epoch_failure_elapsed_ns: self
                .activity
                .quotient_epoch_failure_elapsed_ns
                .load(Ordering::Relaxed),
            quotient_materializations: self
                .activity
                .quotient_materializations
                .load(Ordering::Relaxed),
            quotient_whole_polynomial_materializations: self
                .activity
                .quotient_whole_polynomial_materializations
                .load(Ordering::Relaxed),
            cpu_msm_leases: self.activity.cpu_msm_leases.load(Ordering::Relaxed),
            cpu_msm_lease_bytes: self.activity.cpu_msm_lease_bytes.load(Ordering::Relaxed),
            resident_msm_leases: self.activity.resident_msm_leases.load(Ordering::Relaxed),
            resident_msm_lease_bytes: self
                .activity
                .resident_msm_lease_bytes
                .load(Ordering::Relaxed),
            immutable_key_live_bytes: self
                .immutable_key_accounting
                .live_bytes
                .load(Ordering::Relaxed),
            immutable_key_peak_bytes: self
                .immutable_key_accounting
                .peak_bytes
                .load(Ordering::Relaxed),
            live_key_contexts: self
                .immutable_key_accounting
                .live_contexts
                .load(Ordering::Relaxed),
        }
    }

    /// Resets cumulative counters without changing current device allocation size.
    pub fn reset_activity(&self) {
        msm_worker::reset_activity();
        self.activity.command_buffers.store(0, Ordering::Relaxed);
        self.activity.dispatches.store(0, Ordering::Relaxed);
        self.activity.host_waits.store(0, Ordering::Relaxed);
        self.activity.allocations.store(0, Ordering::Relaxed);
        self.activity.allocated_bytes.store(0, Ordering::Relaxed);
        self.activity.host_write_bytes.store(0, Ordering::Relaxed);
        self.activity.host_read_bytes.store(0, Ordering::Relaxed);
        self.activity.blit_bytes.store(0, Ordering::Relaxed);
        self.activity.execution_failures.store(0, Ordering::Relaxed);
        self.activity.pool_hits.store(0, Ordering::Relaxed);
        self.activity.pool_misses.store(0, Ordering::Relaxed);
        self.activity.pool_returns.store(0, Ordering::Relaxed);
        self.activity.clear_commands.store(0, Ordering::Relaxed);
        self.activity.clear_dispatches.store(0, Ordering::Relaxed);
        self.activity.clear_waits.store(0, Ordering::Relaxed);
        self.activity.cleared_bytes.store(0, Ordering::Relaxed);
        self.activity.clear_failures.store(0, Ordering::Relaxed);
        self.activity.poison_events.store(0, Ordering::Relaxed);
        self.activity
            .resident_generations
            .store(0, Ordering::Relaxed);
        self.activity
            .stale_handle_rejections
            .store(0, Ordering::Relaxed);
        self.activity.materializations.store(0, Ordering::Relaxed);
        self.activity
            .whole_polynomial_materializations
            .store(0, Ordering::Relaxed);
        self.activity.key_cache_hits.store(0, Ordering::Relaxed);
        self.activity.key_cache_misses.store(0, Ordering::Relaxed);
        self.activity.key_context_builds.store(0, Ordering::Relaxed);
        self.activity
            .key_context_build_failures
            .store(0, Ordering::Relaxed);
        self.activity.quotient_epochs.store(0, Ordering::Relaxed);
        self.activity
            .quotient_epoch_elapsed_ns
            .store(0, Ordering::Relaxed);
        self.activity
            .quotient_epoch_failures
            .store(0, Ordering::Relaxed);
        self.activity
            .quotient_epoch_failure_elapsed_ns
            .store(0, Ordering::Relaxed);
        self.activity
            .quotient_materializations
            .store(0, Ordering::Relaxed);
        self.activity
            .quotient_whole_polynomial_materializations
            .store(0, Ordering::Relaxed);
        self.activity.cpu_msm_leases.store(0, Ordering::Relaxed);
        self.activity
            .cpu_msm_lease_bytes
            .store(0, Ordering::Relaxed);
        self.activity
            .resident_msm_leases
            .store(0, Ordering::Relaxed);
        self.activity
            .resident_msm_lease_bytes
            .store(0, Ordering::Relaxed);
        let immutable_live = self
            .immutable_key_accounting
            .live_bytes
            .load(Ordering::Relaxed);
        self.immutable_key_accounting
            .peak_bytes
            .store(immutable_live, Ordering::Relaxed);
        let state = self.lifecycle_guard();
        let live = state.owned_secret_bytes as u64;
        self.activity
            .secret_live_bytes
            .store(live, Ordering::Relaxed);
        self.activity
            .secret_peak_live_bytes
            .store(live, Ordering::Relaxed);
    }

    fn buffer(&self, bytes: usize, label: &str) -> Result<TrackedBuffer, MetalError> {
        self.ensure_session_healthy()?;
        let raw = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Buffer { bytes })?;
        raw.setLabel(Some(&NSString::from_str(label)));
        self.activity.allocations.fetch_add(1, Ordering::Relaxed);
        self.activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(TrackedBuffer { raw, bytes })
    }

    fn buffer_from_slice<T: MetalPod>(
        &self,
        values: &[T],
        label: &str,
    ) -> Result<TrackedBuffer, MetalError> {
        let bytes = size_of_val(values);
        let buffer = self.buffer(bytes, label)?;
        // SAFETY: `MetalPod` is sealed to no-padding integer/raw-field types, the
        // destination allocation contains at least `bytes`, and Metal shared
        // buffers expose a valid CPU mapping for their full allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.raw.contents().as_ptr().cast::<u8>(),
                bytes,
            );
        }
        self.activity
            .host_write_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }

    fn read_values<T: MetalPod + Default>(
        &self,
        buffer: &TrackedBuffer,
        len: usize,
    ) -> Result<Vec<T>, MetalError> {
        self.read_values_from_raw(&buffer.raw, buffer.bytes, len)
    }

    fn read_secret_values<T: MetalPod + Default>(
        &self,
        buffer: &SecretBufferLease,
        len: usize,
    ) -> Result<Vec<T>, MetalError> {
        self.read_values_from_raw(&buffer.raw, buffer.bytes, len)
    }

    fn read_values_from_raw<T: MetalPod + Default>(
        &self,
        raw: &ProtocolObject<dyn MTLBuffer>,
        available: usize,
        len: usize,
    ) -> Result<Vec<T>, MetalError> {
        let requested = len
            .checked_mul(size_of::<T>())
            .ok_or(MetalError::LengthOverflow(len))?;
        if requested > available {
            return Err(MetalError::BufferBounds {
                requested,
                available,
            });
        }
        let mut values = vec![T::default(); len];
        // SAFETY: `MetalPod` permits every bit pattern and has no padding, the
        // source mapping has at least `requested` initialized device-written
        // bytes, and Metal buffer mappings are sufficiently aligned for `T`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                raw.contents().as_ptr().cast::<T>(),
                values.as_mut_ptr(),
                len,
            );
        }
        self.activity
            .host_read_bytes
            .fetch_add(requested as u64, Ordering::Relaxed);
        Ok(values)
    }

    fn read_words<const N: usize>(&self, buffer: &TrackedBuffer) -> Result<[u64; N], MetalError> {
        self.read_values(buffer, N)?
            .try_into()
            .map_err(|_| MetalError::BufferBounds {
                requested: N * size_of::<u64>(),
                available: buffer.bytes,
            })
    }

    fn command_buffer(&self, label: &str) -> Result<CommandBuffer, MetalError> {
        self.ensure_session_healthy()?;
        self.command_buffer_unchecked(label)
    }

    fn command_buffer_unchecked(&self, label: &str) -> Result<CommandBuffer, MetalError> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBuffer)?;
        command.setLabel(Some(&NSString::from_str(label)));
        self.activity
            .command_buffers
            .fetch_add(1, Ordering::Relaxed);
        Ok(command)
    }

    fn dispatch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        elements: usize,
    ) {
        let width = elements
            .min(pipeline.maxTotalThreadsPerThreadgroup())
            .max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn dispatch_threadgroups(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        threadgroups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), MetalError> {
        if threadgroups.width == 0
            || threadgroups.height == 0
            || threadgroups.depth == 0
            || threads.width == 0
            || threads.height == 0
            || threads.depth == 0
        {
            return Err(MetalError::Dispatch(
                "threadgroup grid dimensions must be nonzero",
            ));
        }
        let total_threads = threads
            .width
            .checked_mul(threads.height)
            .and_then(|value| value.checked_mul(threads.depth))
            .ok_or(MetalError::Dispatch("threads-per-group size overflow"))?;
        if total_threads > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(MetalError::Dispatch(
                "threads per group exceed the pipeline limit",
            ));
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
        self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn submit(&self, pending: PendingCommand) -> SubmittedCommand {
        pending.command.commit();
        SubmittedCommand {
            command: pending.command,
            _resources: pending._resources,
        }
    }

    fn wait_for_completion(&self, submitted: SubmittedCommand) -> Result<(), MetalError> {
        self.wait_for_completion_inner(submitted, None)
    }

    fn wait_for_completion_inner(
        &self,
        submitted: SubmittedCommand,
        injected_failure: Option<&'static str>,
    ) -> Result<(), MetalError> {
        self.wait_for_completion_observed(submitted, injected_failure, |_| {})
    }

    fn wait_for_completion_observed<F>(
        &self,
        submitted: SubmittedCommand,
        injected_failure: Option<&'static str>,
        completion_observer: F,
    ) -> Result<(), MetalError>
    where
        F: FnOnce(&SubmittedCommand),
    {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        submitted.command.waitUntilCompleted();
        completion_observer(&submitted);
        let failure = submitted
            .command
            .error()
            .map(|error| format!("{error:?}"))
            .or_else(|| {
                let status = submitted.command.status();
                (status != MTLCommandBufferStatus::Completed)
                    .then(|| format!("unexpected completion status {status:?}"))
            })
            .or_else(|| injected_failure.map(str::to_owned));
        if let Some(failure) = failure {
            self.activity
                .execution_failures
                .fetch_add(1, Ordering::Relaxed);
            // `submitted` still owns all buffers at this point. The error is
            // classified before the state is dropped, and callers never get a
            // completed token from which they could materialize or recycle it.
            return Err(MetalError::Execution(failure));
        }
        Ok(())
    }

    fn submit_and_wait(&self, pending: PendingCommand) -> Result<(), MetalError> {
        let submitted = self.submit(pending);
        self.wait_for_completion(submitted)
    }

    #[cfg(test)]
    fn injected_command_failure_probe(&self) -> Result<Vec<u64>, MetalError> {
        let _execution = self.execution_guard();
        let buffer =
            self.buffer_from_slice(&[0xfeed_face_cafe_beefu64], "midnight.failure_probe")?;
        let command = self.command_buffer("midnight.failure_probe")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.clear_words);
        // SAFETY: the shared u64 buffer is retained by `SubmittedCommand` until
        // the real device completion and injected validation failure both run.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buffer.raw), 0, 0);
        }
        self.dispatch(&encoder, &self.clear_words, 1);
        encoder.endEncoding();
        let submitted = self.submit(PendingCommand {
            command,
            _resources: vec![buffer.raw.clone()],
        });
        self.wait_for_completion_inner(submitted, Some("injected completion failure"))?;

        // The injected failure above must make this readback unreachable.
        self.read_values(&buffer, 1)
    }

    #[cfg(test)]
    fn injected_ntt_completion_failure_probe(
        &self,
        values: &[Bls12_381Fr],
        domain: NttDomainSpec,
        retained_at_completion: &std::sync::atomic::AtomicUsize,
    ) -> Result<(), MetalError> {
        let domain_len = validate_ntt_domain(domain)?;
        let batch_count = complete_ntt_batches(values.len(), domain_len)?;
        if batch_count == 0 {
            return Err(MetalError::ResidentState(
                "failure probe requires a nonempty resident NTT",
            ));
        }

        let plan = self.ntt_plan(domain)?;
        if plan.schedule.len() < 2 {
            return Err(MetalError::ResidentState(
                "failure probe requires a multi-pass NTT plan",
            ));
        }
        self.with_resident_fr_arena(
            values,
            domain_len,
            domain_len,
            batch_count,
            FrDomainTag::coefficient(domain.log_size),
            |arena, input| {
                let command = self.command_buffer("midnight.failure_probe.ntt")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let mut resources = arena.retained_resources();
                let output = self.encode_ntt(
                    &encoder,
                    &mut resources,
                    arena,
                    input,
                    &plan,
                    NttDirection::Forward,
                    false,
                )?;
                arena.validate(&output)?;
                encoder.endEncoding();
                let submitted = self.submit(PendingCommand {
                    command,
                    _resources: resources,
                });
                self.wait_for_completion_observed(
                    submitted,
                    Some("injected NTT completion failure"),
                    |submitted| {
                        assert_eq!(
                            submitted.command.status(),
                            MTLCommandBufferStatus::Completed
                        );
                        retained_at_completion.store(submitted._resources.len(), Ordering::SeqCst);
                    },
                )
            },
        )
    }

    fn execution_guard(&self) -> MutexGuard<'_, ()> {
        match self.execution.lock() {
            Ok(execution) => execution,
            Err(poisoned) => {
                self.poison_session("host panic poisoned the Metal execution mutex".to_owned());
                self.execution.clear_poison();
                poisoned.into_inner()
            }
        }
    }
}

impl NttPlan {
    fn create(session: &MetalSession, domain: NttDomainSpec) -> Result<Self, MetalError> {
        let domain_len = validate_ntt_domain(domain)?;
        let max_degree = domain.log_size.min(MAX_NTT_RADIX_LOG);
        let schedule = ntt_radix_schedule(domain.log_size);
        let forward =
            NttTwiddlePlan::create(session, domain.root, domain_len, max_degree, "forward")?;
        let inverse = NttTwiddlePlan::create(
            session,
            domain.inverse_root,
            domain_len,
            max_degree,
            "inverse",
        )?;
        Ok(Self {
            domain,
            domain_len,
            max_degree,
            schedule,
            forward,
            inverse,
        })
    }
}

impl NttTwiddlePlan {
    fn create(
        session: &MetalSession,
        root: Bls12_381Fr,
        domain_len: usize,
        max_degree: u32,
        direction: &str,
    ) -> Result<Self, MetalError> {
        let one = Bls12_381Fr {
            limbs: generated::FR_R,
        };
        let pq_len = 1usize << max_degree.saturating_sub(1);
        let twiddle_exponent = if max_degree == 0 {
            0
        } else {
            (domain_len >> max_degree) as u64
        };
        let twiddle = fr_pow_host(root, twiddle_exponent);
        let mut pq = vec![one; pq_len];
        for index in 1..pq.len() {
            pq[index] = fr_montgomery_mul_host(pq[index - 1], twiddle);
        }

        let mut omega_squares = [Bls12_381Fr::default(); 32];
        omega_squares[0] = root;
        for index in 1..omega_squares.len() {
            omega_squares[index] =
                fr_montgomery_mul_host(omega_squares[index - 1], omega_squares[index - 1]);
        }
        Ok(Self {
            pq: session.buffer_from_slice(&pq, &format!("midnight.ntt.{direction}.pq"))?,
            omega_squares: session.buffer_from_slice(
                &omega_squares,
                &format!("midnight.ntt.{direction}.omega_squares"),
            )?,
        })
    }
}

fn complete_ntt_batches(elements: usize, domain_len: usize) -> Result<usize, MetalError> {
    if !elements.is_multiple_of(domain_len) {
        return Err(MetalError::NttBatch {
            elements,
            batch_len: domain_len,
        });
    }
    Ok(elements / domain_len)
}

fn validate_ntt_domain(domain: NttDomainSpec) -> Result<usize, MetalError> {
    if domain.log_size > MAX_NTT_LOG_SIZE {
        return Err(MetalError::NttDomain(
            "log size exceeds the 32-bit Stockham index ABI",
        ));
    }
    let domain_len = 1usize << domain.log_size;
    for value in [
        domain.root,
        domain.inverse_root,
        domain.inverse_size,
        domain.coset_generator,
        domain.inverse_coset_generator,
    ] {
        if fr_greater_or_equal_host(value.limbs, generated::FR_MODULUS) {
            return Err(MetalError::NttDomain(
                "a Montgomery field element is not reduced",
            ));
        }
    }

    let one = Bls12_381Fr {
        limbs: generated::FR_R,
    };
    if fr_montgomery_mul_host(domain.root, domain.inverse_root) != one {
        return Err(MetalError::NttDomain(
            "root and inverse root are not multiplicative inverses",
        ));
    }
    if fr_pow_host(domain.root, domain_len as u64) != one {
        return Err(MetalError::NttDomain(
            "root does not have the requested power-of-two order",
        ));
    }
    if domain.log_size != 0 && fr_pow_host(domain.root, (domain_len >> 1) as u64) == one {
        return Err(MetalError::NttDomain(
            "root is not primitive for the requested domain",
        ));
    }

    let mut size = one;
    for _ in 0..domain.log_size {
        size = fr_add_host(size, size);
    }
    if fr_montgomery_mul_host(size, domain.inverse_size) != one {
        return Err(MetalError::NttDomain(
            "inverse size does not equal the transform-size inverse",
        ));
    }
    if fr_montgomery_mul_host(domain.coset_generator, domain.inverse_coset_generator) != one {
        return Err(MetalError::NttDomain(
            "coset generator and inverse are not multiplicative inverses",
        ));
    }
    if domain.coset_generator == one || fr_pow_host(domain.coset_generator, 3) != one {
        return Err(MetalError::NttDomain(
            "coset generator is not a nontrivial cube root of unity",
        ));
    }
    Ok(domain_len)
}

fn validate_ntt_domain_pair(
    base: NttDomainSpec,
    extended: NttDomainSpec,
) -> Result<(), MetalError> {
    validate_ntt_domain(base)?;
    validate_ntt_domain(extended)?;
    let extension_log =
        extended
            .log_size
            .checked_sub(base.log_size)
            .ok_or(MetalError::NttDomain(
                "extended NTT domain is smaller than the base domain",
            ))?;
    let root_exponent = 1u64
        .checked_shl(extension_log)
        .ok_or(MetalError::NttDomain(
            "NTT domain-pair extension exponent overflows",
        ))?;
    if fr_pow_host(extended.root, root_exponent) != base.root
        || fr_pow_host(extended.inverse_root, root_exponent) != base.inverse_root
    {
        return Err(MetalError::NttDomain(
            "base and extended NTT roots do not form the same authority chain",
        ));
    }
    if extended.coset_generator != base.coset_generator
        || extended.inverse_coset_generator != base.inverse_coset_generator
    {
        return Err(MetalError::NttDomain(
            "base and extended NTT domains use different cosets",
        ));
    }
    Ok(())
}

fn ntt_radix_schedule(log_size: u32) -> Box<[u32]> {
    if log_size == 0 {
        return Box::new([]);
    }
    // Balance pass degrees instead of leaving a small radix tail. On this
    // device k=17 fits the measured two-pass 9+8 schedule while keeping the
    // radix-512 tile within both thread and threadgroup-memory limits.
    let passes = log_size.div_ceil(MAX_NTT_RADIX_LOG);
    let base = log_size / passes;
    let larger = log_size % passes;
    (0..passes)
        .map(|pass| base + u32::from(pass < larger))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn fr_pow_host(mut base: Bls12_381Fr, mut exponent: u64) -> Bls12_381Fr {
    let mut result = Bls12_381Fr {
        limbs: generated::FR_R,
    };
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = fr_montgomery_mul_host(result, base);
        }
        base = fr_montgomery_mul_host(base, base);
        exponent >>= 1;
    }
    result
}

fn build_extended_coset_points(
    domain: NttDomainSpec,
    len: usize,
) -> Result<Vec<Bls12_381Fr>, MetalError> {
    let expected_len = 1usize
        .checked_shl(domain.log_size)
        .ok_or_else(|| MetalError::KeyContext("extended domain length overflows usize".into()))?;
    if len != expected_len {
        return Err(MetalError::KeyContext(
            "extended-coset point table length differs from its exact NTT domain".into(),
        ));
    }

    let mut points = Vec::new();
    points.try_reserve_exact(len).map_err(|_| {
        MetalError::KeyContext("cannot allocate the extended-coset point table".into())
    })?;
    let mut point = domain.coset_generator;
    for _ in 0..len {
        points.push(point);
        point = fr_montgomery_mul_host(point, domain.root);
    }
    if point != domain.coset_generator {
        return Err(MetalError::KeyContext(
            "extended-coset point table does not close over the validated NTT domain".into(),
        ));
    }
    Ok(points)
}

fn fr_add_host(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    let mut limbs = [0u64; 4];
    let mut carry = false;
    for (index, limb) in limbs.iter_mut().enumerate() {
        let (sum, first_carry) = lhs.limbs[index].overflowing_add(rhs.limbs[index]);
        let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
        *limb = sum;
        carry = first_carry || second_carry;
    }
    debug_assert!(!carry, "two reduced Fr values fit in four limbs");
    if fr_greater_or_equal_host(limbs, generated::FR_MODULUS) {
        limbs = fr_subtract_host(limbs, generated::FR_MODULUS);
    }
    Bls12_381Fr { limbs }
}

fn fr_montgomery_mul_host(lhs: Bls12_381Fr, rhs: Bls12_381Fr) -> Bls12_381Fr {
    // Host-side CIOS mirrors field.metal only for immutable plan construction
    // and validation; online polynomial values never take a CPU arithmetic path.
    let mut temporary = [0u64; 6];
    for rhs_index in 0..4 {
        let mut carry = 0;
        for (lhs_index, temporary_word) in temporary[..4].iter_mut().enumerate() {
            *temporary_word = multiply_accumulate_host(
                lhs.limbs[lhs_index],
                rhs.limbs[rhs_index],
                *temporary_word,
                &mut carry,
            );
        }
        temporary[4] = add_carry_word_host(temporary[4], &mut carry);
        temporary[5] = carry;

        carry = 0;
        let reduction = temporary[0].wrapping_mul(generated::FR_INV);
        let _ = multiply_accumulate_host(
            reduction,
            generated::FR_MODULUS[0],
            temporary[0],
            &mut carry,
        );
        for index in 1..4 {
            temporary[index - 1] = multiply_accumulate_host(
                reduction,
                generated::FR_MODULUS[index],
                temporary[index],
                &mut carry,
            );
        }
        temporary[3] = add_carry_word_host(temporary[4], &mut carry);
        temporary[4] = temporary[5].wrapping_add(carry);
    }

    let mut limbs: [u64; 4] = temporary[..4]
        .try_into()
        .expect("four-limb Montgomery output");
    if fr_greater_or_equal_host(limbs, generated::FR_MODULUS) {
        limbs = fr_subtract_host(limbs, generated::FR_MODULUS);
    }
    Bls12_381Fr { limbs }
}

fn multiply_accumulate_host(lhs: u64, rhs: u64, addend: u64, carry: &mut u64) -> u64 {
    let product = u128::from(lhs) * u128::from(rhs);
    let mut low = product as u64;
    let mut high = (product >> 64) as u64;
    let (next, overflow) = low.overflowing_add(addend);
    low = next;
    high = high.wrapping_add(u64::from(overflow));
    let (next, overflow) = low.overflowing_add(*carry);
    low = next;
    high = high.wrapping_add(u64::from(overflow));
    *carry = high;
    low
}

fn add_carry_word_host(value: u64, carry: &mut u64) -> u64 {
    let (result, overflow) = value.overflowing_add(*carry);
    *carry = u64::from(overflow);
    result
}

fn fr_greater_or_equal_host(lhs: [u64; 4], rhs: [u64; 4]) -> bool {
    for index in (0..4).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn fr_subtract_host(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    let mut output = [0u64; 4];
    let mut borrow = false;
    for index in 0..4 {
        let (difference, first_borrow) = lhs[index].overflowing_sub(rhs[index]);
        let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
        output[index] = difference;
        borrow = first_borrow || second_borrow;
    }
    debug_assert!(!borrow, "greater-or-equal subtraction cannot underflow");
    output
}

fn bls12_381_ntt_domain(log_size: u32) -> Result<NttDomainSpec, MetalError> {
    const TWO_ADIC_ROOT: Bls12_381Fr = Bls12_381Fr {
        limbs: [
            0xb9b5_8d8c_5f0e_466a,
            0x5b1b_4c80_1819_d7ec,
            0x0af5_3ae3_52a3_1e64,
            0x5bf3_adda_19e9_b27b,
        ],
    };
    const TWO_ADIC_ROOT_INV: Bls12_381Fr = Bls12_381Fr {
        limbs: [
            0x4256_481a_dcf3_219a,
            0x45f3_7b7f_96b6_cad3,
            0xf9c3_f1d7_5f7a_3b27,
            0x2d2f_c049_658a_fd43,
        ],
    };
    const TWO_INV: Bls12_381Fr = Bls12_381Fr {
        limbs: [
            0x0000_0000_ffff_ffff,
            0xac42_5bfd_0001_a401,
            0xccc6_27f7_f65e_27fa,
            0x0c12_58ac_d662_82b7,
        ],
    };
    const ZETA: Bls12_381Fr = Bls12_381Fr {
        limbs: [
            0x92d9_090b_0930_11d2,
            0xfc9c_bd71_9d6a_a073,
            0xc1f1_4ef0_cd65_a1a6,
            0x017f_6d35_e72f_cdeb,
        ],
    };

    if log_size > 32 {
        return Err(MetalError::NttDomain(
            "canonical BLS12-381 domain exceeds two-adicity",
        ));
    }
    let mut root = TWO_ADIC_ROOT;
    let mut inverse_root = TWO_ADIC_ROOT_INV;
    for _ in log_size..32 {
        root = fr_montgomery_mul_host(root, root);
        inverse_root = fr_montgomery_mul_host(inverse_root, inverse_root);
    }
    let mut inverse_size = Bls12_381Fr {
        limbs: generated::FR_R,
    };
    for _ in 0..log_size {
        inverse_size = fr_montgomery_mul_host(inverse_size, TWO_INV);
    }
    Ok(NttDomainSpec {
        log_size,
        root,
        inverse_root,
        inverse_size,
        coset_generator: ZETA,
        inverse_coset_generator: fr_montgomery_mul_host(ZETA, ZETA),
    })
}

fn graph_abi_from_words(words: [u64; GRAPH_ABI_WORDS]) -> GraphAbiProbe {
    GraphAbiProbe {
        abi_version: words[0],
        value_ref_layout: words[1..6]
            .try_into()
            .expect("graph value-reference layout has a fixed width"),
        operation_layout: words[6..14]
            .try_into()
            .expect("graph operation layout has a fixed width"),
        parameters_layout: words[14..45]
            .try_into()
            .expect("graph parameter layout has a fixed width"),
        value_tags: words[45..57]
            .try_into()
            .expect("graph value-tag table has a fixed width"),
        operation_tags: words[57..65]
            .try_into()
            .expect("graph operation-tag table has a fixed width"),
        post_kinds: words[65..68]
            .try_into()
            .expect("graph post-kind table has a fixed width"),
        scratch_classes: words[68..74]
            .try_into()
            .expect("graph scratch-class table has a fixed width"),
        layout_fingerprint: words[74],
    }
}

fn graph_abi_words(probe: GraphAbiProbe) -> [u64; GRAPH_ABI_WORDS] {
    let mut words = [0; GRAPH_ABI_WORDS];
    words[0] = probe.abi_version;
    words[1..6].copy_from_slice(&probe.value_ref_layout);
    words[6..14].copy_from_slice(&probe.operation_layout);
    words[14..45].copy_from_slice(&probe.parameters_layout);
    words[45..57].copy_from_slice(&probe.value_tags);
    words[57..65].copy_from_slice(&probe.operation_tags);
    words[65..68].copy_from_slice(&probe.post_kinds);
    words[68..74].copy_from_slice(&probe.scratch_classes);
    words[74] = probe.layout_fingerprint;
    words
}

fn expected_graph_abi() -> GraphAbiProbe {
    GraphAbiProbe {
        abi_version: GRAPH_ABI_VERSION,
        value_ref_layout: [
            size_of::<GraphValueRefAbi>() as u64,
            align_of::<GraphValueRefAbi>() as u64,
            offset_of!(GraphValueRefAbi, kind) as u64,
            offset_of!(GraphValueRefAbi, a) as u64,
            offset_of!(GraphValueRefAbi, b) as u64,
        ],
        operation_layout: [
            size_of::<GraphOperationAbi>() as u64,
            align_of::<GraphOperationAbi>() as u64,
            offset_of!(GraphOperationAbi, opcode) as u64,
            offset_of!(GraphOperationAbi, target) as u64,
            offset_of!(GraphOperationAbi, parts_offset) as u64,
            offset_of!(GraphOperationAbi, parts_len) as u64,
            offset_of!(GraphOperationAbi, lhs) as u64,
            offset_of!(GraphOperationAbi, rhs) as u64,
        ],
        parameters_layout: [
            size_of::<GraphEvalParametersAbi>() as u64,
            align_of::<GraphEvalParametersAbi>() as u64,
            offset_of!(GraphEvalParametersAbi, len) as u64,
            offset_of!(GraphEvalParametersAbi, stride) as u64,
            offset_of!(GraphEvalParametersAbi, rotation_scale) as u64,
            offset_of!(GraphEvalParametersAbi, post_kind) as u64,
            offset_of!(GraphEvalParametersAbi, constants_offset) as u64,
            offset_of!(GraphEvalParametersAbi, constants_count) as u64,
            offset_of!(GraphEvalParametersAbi, rotations_offset) as u64,
            offset_of!(GraphEvalParametersAbi, rotations_count) as u64,
            offset_of!(GraphEvalParametersAbi, operations_offset) as u64,
            offset_of!(GraphEvalParametersAbi, operations_count) as u64,
            offset_of!(GraphEvalParametersAbi, parts_offset) as u64,
            offset_of!(GraphEvalParametersAbi, parts_count) as u64,
            offset_of!(GraphEvalParametersAbi, fixed_columns) as u64,
            offset_of!(GraphEvalParametersAbi, advice_offset) as u64,
            offset_of!(GraphEvalParametersAbi, advice_columns) as u64,
            offset_of!(GraphEvalParametersAbi, instance_offset) as u64,
            offset_of!(GraphEvalParametersAbi, instance_columns) as u64,
            offset_of!(GraphEvalParametersAbi, challenge_count) as u64,
            offset_of!(GraphEvalParametersAbi, previous_mode) as u64,
            offset_of!(GraphEvalParametersAbi, lookup_product_column) as u64,
            offset_of!(GraphEvalParametersAbi, lookup_input_column) as u64,
            offset_of!(GraphEvalParametersAbi, lookup_table_column) as u64,
            offset_of!(GraphEvalParametersAbi, trash_column) as u64,
            offset_of!(GraphEvalParametersAbi, trash_selector_fixed_column) as u64,
            offset_of!(GraphEvalParametersAbi, beta) as u64,
            offset_of!(GraphEvalParametersAbi, gamma) as u64,
            offset_of!(GraphEvalParametersAbi, theta) as u64,
            offset_of!(GraphEvalParametersAbi, trash_challenge) as u64,
            offset_of!(GraphEvalParametersAbi, y) as u64,
        ],
        value_tags: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        operation_tags: [0, 1, 2, 3, 4, 5, 6, 7],
        post_kinds: [0, 1, 2],
        scratch_classes: [8, 16, 32, 64, 128, 256],
        layout_fingerprint: GRAPH_ABI_FINGERPRINT,
    }
}

fn graph_abi_fingerprint(words: &[u64]) -> u64 {
    words
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |fingerprint, word| {
            (fingerprint ^ word).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn validate_graph_abi(actual: GraphAbiProbe) -> Result<(), MetalError> {
    let expected = expected_graph_abi();
    let expected_words = graph_abi_words(expected);
    let expected_fingerprint = graph_abi_fingerprint(&expected_words[..GRAPH_ABI_WORDS - 1]);
    if expected_fingerprint != GRAPH_ABI_FINGERPRINT {
        return Err(MetalError::GraphAbi(format!(
            "host layout changed without an ABI fingerprint update: expected pinned \
             {GRAPH_ABI_FINGERPRINT:#018x}, computed {expected_fingerprint:#018x}"
        )));
    }

    let actual_words = graph_abi_words(actual);
    if actual_words == expected_words {
        return Ok(());
    }
    let mismatch = actual_words
        .iter()
        .zip(expected_words)
        .position(|(actual, expected)| *actual != expected)
        .expect("different graph ABI arrays contain a mismatch");
    Err(MetalError::GraphAbi(format!(
        "word {mismatch} expected {}, device reported {}; expected fingerprint \
         {GRAPH_ABI_FINGERPRINT:#018x}, device reported {:#018x}",
        expected_words[mismatch], actual_words[mismatch], actual.layout_fingerprint
    )))
}

fn validate_bls12_381_abi(actual: Bls12_381AbiProbe) -> Result<(), MetalError> {
    let expected = Bls12_381AbiProbe {
        abi_version: generated::BLS12_381_ABI_VERSION,
        fr_size: size_of::<Bls12_381Fr>() as u64,
        fr_alignment: align_of::<Bls12_381Fr>() as u64,
        fr_limbs_offset: offset_of!(Bls12_381Fr, limbs) as u64,
        fp_size: size_of::<Bls12_381Fp>() as u64,
        fp_alignment: align_of::<Bls12_381Fp>() as u64,
        fp_limbs_offset: offset_of!(Bls12_381Fp, limbs) as u64,
        g1_affine_size: size_of::<Bls12_381G1Affine>() as u64,
        g1_affine_alignment: align_of::<Bls12_381G1Affine>() as u64,
        g1_affine_x_offset: offset_of!(Bls12_381G1Affine, x) as u64,
        g1_affine_y_offset: offset_of!(Bls12_381G1Affine, y) as u64,
        g1_projective_size: size_of::<Bls12_381G1Projective>() as u64,
        g1_projective_alignment: align_of::<Bls12_381G1Projective>() as u64,
        g1_projective_x_offset: offset_of!(Bls12_381G1Projective, x) as u64,
        g1_projective_y_offset: offset_of!(Bls12_381G1Projective, y) as u64,
        g1_projective_z_offset: offset_of!(Bls12_381G1Projective, z) as u64,
        g1_homogeneous_size: size_of::<Bls12_381G1Projective>() as u64,
        g1_homogeneous_alignment: align_of::<Bls12_381G1Projective>() as u64,
        g1_homogeneous_x_offset: offset_of!(Bls12_381G1Projective, x) as u64,
        g1_homogeneous_y_offset: offset_of!(Bls12_381G1Projective, y) as u64,
        g1_homogeneous_z_offset: offset_of!(Bls12_381G1Projective, z) as u64,
        layout_fingerprint: BLS12_381_LAYOUT_ABI_FINGERPRINT,
        constants_fingerprint: generated::BLS12_381_CONSTANTS_FINGERPRINT,
    };
    if actual.layout_fingerprint != BLS12_381_LAYOUT_ABI_FINGERPRINT {
        Err(MetalError::Bls12_381Abi(format!(
            "layout fingerprint mismatch: expected {BLS12_381_LAYOUT_ABI_FINGERPRINT:#018x}, \
             device reported {:#018x}; full device layout: {actual:?}",
            actual.layout_fingerprint
        )))
    } else if actual == expected {
        Ok(())
    } else {
        Err(MetalError::Bls12_381Abi(format!(
            "expected {expected:?}, device reported {actual:?}"
        )))
    }
}

#[cfg(test)]
fn expected_batch_affine_abi() -> [u64; BATCH_AFFINE_ABI_WORDS] {
    [
        BATCH_AFFINE_ABI_VERSION,
        size_of::<BatchAffineFp32>() as u64,
        align_of::<BatchAffineFp32>() as u64,
        offset_of!(BatchAffineFp32, limbs) as u64,
        size_of::<BatchAffinePoint32>() as u64,
        align_of::<BatchAffinePoint32>() as u64,
        offset_of!(BatchAffinePoint32, x) as u64,
        offset_of!(BatchAffinePoint32, y) as u64,
        size_of::<BatchAffineGatherParameters>() as u64,
        align_of::<BatchAffineGatherParameters>() as u64,
        offset_of!(BatchAffineGatherParameters, record_offset) as u64,
        offset_of!(BatchAffineGatherParameters, record_count) as u64,
        offset_of!(BatchAffineGatherParameters, srs_count) as u64,
        offset_of!(BatchAffineGatherParameters, reserved) as u64,
        size_of::<BatchAffineLevelParameters>() as u64,
        align_of::<BatchAffineLevelParameters>() as u64,
        offset_of!(BatchAffineLevelParameters, destination_map_offset) as u64,
        offset_of!(BatchAffineLevelParameters, pair_count) as u64,
        offset_of!(BatchAffineLevelParameters, carry_count) as u64,
        offset_of!(BatchAffineLevelParameters, destination_len) as u64,
        size_of::<BatchAffineProductParameters>() as u64,
        align_of::<BatchAffineProductParameters>() as u64,
        offset_of!(BatchAffineProductParameters, input_offset) as u64,
        offset_of!(BatchAffineProductParameters, input_count) as u64,
        offset_of!(BatchAffineProductParameters, output_offset) as u64,
        offset_of!(BatchAffineProductParameters, output_count) as u64,
        size_of::<BatchAffineStatus>() as u64,
        align_of::<BatchAffineStatus>() as u64,
        offset_of!(BatchAffineStatus, flags) as u64,
        offset_of!(BatchAffineStatus, record_errors) as u64,
        offset_of!(BatchAffineStatus, destination_errors) as u64,
        offset_of!(BatchAffineStatus, reserved) as u64,
    ]
}

#[cfg(test)]
fn expected_batch_affine_ceiling_abi() -> [u64; BATCH_AFFINE_CEILING_ABI_WORDS] {
    [
        BATCH_AFFINE_CEILING_ABI_VERSION,
        size_of::<BatchAffineCeilingParameters>() as u64,
        align_of::<BatchAffineCeilingParameters>() as u64,
        offset_of!(BatchAffineCeilingParameters, element_count) as u64,
        offset_of!(BatchAffineCeilingParameters, iterations) as u64,
        offset_of!(BatchAffineCeilingParameters, reserved_0) as u64,
        offset_of!(BatchAffineCeilingParameters, reserved_1) as u64,
    ]
}

#[cfg(test)]
fn expected_flat_segmented_abi() -> [u64; FLAT_SEGMENTED_ABI_WORDS] {
    use self::flat_segmented_plan::FlatSegmentedTile;

    [
        FLAT_SEGMENTED_ABI_VERSION,
        size_of::<FlatSegmentedTile>() as u64,
        align_of::<FlatSegmentedTile>() as u64,
        offset_of!(FlatSegmentedTile, input_offset) as u64,
        offset_of!(FlatSegmentedTile, input_len) as u64,
        offset_of!(FlatSegmentedTile, output_offset) as u64,
        offset_of!(FlatSegmentedTile, reserved) as u64,
        size_of::<FlatSegmentedParameters>() as u64,
        align_of::<FlatSegmentedParameters>() as u64,
        offset_of!(FlatSegmentedParameters, tile_count) as u64,
        offset_of!(FlatSegmentedParameters, input_count) as u64,
        offset_of!(FlatSegmentedParameters, output_count) as u64,
        offset_of!(FlatSegmentedParameters, srs_count) as u64,
        offset_of!(FlatSegmentedParameters, group_count) as u64,
        offset_of!(FlatSegmentedParameters, input_kind) as u64,
        offset_of!(FlatSegmentedParameters, group_width) as u64,
        offset_of!(FlatSegmentedParameters, reserved) as u64,
    ]
}

fn expected_ntt_parameter_abi() -> [u64; NTT_PARAMETER_ABI_WORDS] {
    let mut words = [0u64; NTT_PARAMETER_ABI_WORDS];
    words[0] = NTT_PARAMETER_ABI_VERSION;
    words[1] = size_of::<NttPassParameters>() as u64;
    words[2] = align_of::<NttPassParameters>() as u64;
    words[3] = offset_of!(NttPassParameters, n) as u64;
    words[4] = offset_of!(NttPassParameters, source_stride) as u64;
    words[5] = offset_of!(NttPassParameters, destination_stride) as u64;
    words[6] = offset_of!(NttPassParameters, log_p) as u64;
    words[7] = offset_of!(NttPassParameters, degree) as u64;
    words[8] = offset_of!(NttPassParameters, max_degree) as u64;
    words[9] = offset_of!(NttPassParameters, batch_count) as u64;
    words[10] = offset_of!(NttPassParameters, flags) as u64;
    words[11] = offset_of!(NttPassParameters, reserved) as u64;
    words[12] = offset_of!(NttPassParameters, one) as u64;
    words[13] = offset_of!(NttPassParameters, inverse_size) as u64;
    words[14] = offset_of!(NttPassParameters, coset_generator) as u64;
    words[15] = offset_of!(NttPassParameters, inverse_coset_generator) as u64;
    words[16] = size_of::<NttCosetParameters>() as u64;
    words[17] = align_of::<NttCosetParameters>() as u64;
    words[18] = offset_of!(NttCosetParameters, input_length) as u64;
    words[19] = offset_of!(NttCosetParameters, output_length) as u64;
    words[20] = offset_of!(NttCosetParameters, source_stride) as u64;
    words[21] = offset_of!(NttCosetParameters, destination_stride) as u64;
    words[22] = offset_of!(NttCosetParameters, batch_count) as u64;
    words[23] = offset_of!(NttCosetParameters, reserved) as u64;
    words[24] = offset_of!(NttCosetParameters, coset_generator) as u64;
    words[25] = offset_of!(NttCosetParameters, inverse_coset_generator) as u64;
    // Updating either repr(C) block requires an intentional ABI version and
    // fingerprint update instead of silently accepting coordinated drift.
    words[26] = NTT_PARAMETER_ABI_FINGERPRINT;
    words
}

#[cfg(test)]
fn ntt_parameter_abi_fingerprint(words: &[u64]) -> u64 {
    words
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |fingerprint, word| {
            (fingerprint ^ word).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn validate_ntt_parameter_abi(actual: [u64; NTT_PARAMETER_ABI_WORDS]) -> Result<(), MetalError> {
    const WORD_LABELS: [&str; NTT_PARAMETER_ABI_WORDS] = [
        "version",
        "pass.size",
        "pass.alignment",
        "pass.n",
        "pass.source_stride",
        "pass.destination_stride",
        "pass.log_p",
        "pass.degree",
        "pass.max_degree",
        "pass.batch_count",
        "pass.flags",
        "pass.reserved",
        "pass.one",
        "pass.inverse_size",
        "pass.coset_generator",
        "pass.inverse_coset_generator",
        "coset.size",
        "coset.alignment",
        "coset.input_length",
        "coset.output_length",
        "coset.source_stride",
        "coset.destination_stride",
        "coset.batch_count",
        "coset.reserved",
        "coset.coset_generator",
        "coset.inverse_coset_generator",
        "layout_fingerprint",
    ];

    let expected = expected_ntt_parameter_abi();
    if actual == expected {
        return Ok(());
    }
    let mismatch = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| *actual != expected)
        .expect("different arrays contain a mismatch");
    Err(MetalError::NttAbi(format!(
        "{} expected {}, device reported {}; expected fingerprint {:#018x}, device reported {:#018x}",
        WORD_LABELS[mismatch], expected[mismatch], actual[mismatch], expected[26], actual[26]
    )))
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
) -> Result<Pipeline, MetalError> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or(MetalError::Function(name))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::Pipeline {
            name,
            reason: format!("{error:?}"),
        })
}

pub(crate) fn probe_status() -> MetalStatus {
    match MetalSession::new() {
        Ok(session) => match session.device_info() {
            Ok(device) => MetalStatus::runtime_ready(device),
            Err(error) => MetalStatus::unavailable(true, &error),
        },
        Err(error) => MetalStatus::unavailable(true, &error),
    }
}

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn session_can_be_shared_by_the_process_level_backend_owner() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetalSession>();
    }

    #[test]
    fn batch_affine_parameter_blocks_match_the_device_layout_contract() {
        let expected = expected_batch_affine_abi();

        assert_eq!(
            expected,
            [
                1, 48, 4, 0, 96, 4, 0, 48, 16, 4, 0, 4, 8, 12, 16, 4, 0, 4, 8, 12, 16, 4, 0, 4, 8,
                12, 16, 4, 0, 4, 8, 12,
            ]
        );

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .batch_affine_abi_probe()
            .expect("execute batch-affine parameter ABI probe");
        assert_eq!(actual, expected);
    }

    #[test]
    fn batch_affine_ceiling_parameter_block_matches_the_device_layout_contract() {
        let expected = expected_batch_affine_ceiling_abi();
        assert_eq!(expected, [1, 16, 4, 0, 4, 8, 12]);

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .batch_affine_ceiling_abi_probe()
            .expect("execute batch-affine ceiling ABI probe");
        assert_eq!(actual, expected);
    }

    #[test]
    fn flat_segmented_parameter_blocks_match_the_device_layout_contract() {
        let expected = expected_flat_segmented_abi();
        assert_eq!(
            expected,
            [1, 16, 4, 0, 4, 8, 12, 32, 4, 0, 4, 8, 12, 16, 20, 24, 28]
        );

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .flat_segmented_abi_probe()
            .expect("execute flat-segmented parameter ABI probe");
        assert_eq!(actual, expected);
    }

    #[test]
    fn ntt_parameter_blocks_match_the_device_layout_contract() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .ntt_parameter_abi_probe()
            .expect("execute NTT parameter ABI probe");
        let expected = expected_ntt_parameter_abi();

        assert_eq!(actual, expected);
        assert_eq!(actual[1], 168);
        assert_eq!(actual[2], 8);
        assert_eq!((actual[4], actual[5]), (4, 8));
        assert_eq!(actual[16], 88);
        assert_eq!(actual[17], 8);
        assert_eq!((actual[20], actual[21]), (8, 12));
        assert_eq!(
            ntt_parameter_abi_fingerprint(&expected[..26]),
            NTT_PARAMETER_ABI_FINGERPRINT
        );

        let mut shifted = actual;
        shifted[24] += 8;
        shifted[26] = ntt_parameter_abi_fingerprint(&shifted[..26]);
        let error = validate_ntt_parameter_abi(shifted)
            .expect_err("a shifted Metal parameter member must be rejected");
        assert!(
            matches!(error, MetalError::NttAbi(ref reason) if reason.contains("coset.coset_generator"))
        );
    }

    #[test]
    fn coset_prepare_honors_independent_batch_strides() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let first = Bls12_381Fr {
            limbs: generated::FR_R,
        };
        let second = Bls12_381Fr {
            limbs: generated::FR_R2,
        };
        let zero = Bls12_381Fr::default();
        let source = session
            .buffer_from_slice(
                &[first, zero, second, zero],
                "midnight.test.ntt.stride.source",
            )
            .expect("allocate strided source");
        let destination = session
            .buffer_from_slice(&[zero; 6], "midnight.test.ntt.stride.destination")
            .expect("allocate strided destination");
        let parameters = NttCosetParameters {
            input_length: 1,
            output_length: 1,
            source_stride: 2,
            destination_stride: 3,
            batch_count: 2,
            reserved: 0,
            coset_generator: first,
            inverse_coset_generator: first,
        };

        let command = session
            .command_buffer("midnight.test.ntt.independent_strides")
            .expect("create command buffer");
        let encoder = command
            .computeCommandEncoder()
            .expect("create compute encoder");
        encoder.setComputePipelineState(&session.ntt.coset_prepare);
        // SAFETY: both buffers are retained through completion, their zero
        // offsets satisfy the Fr ABI, and the parameter block has no padding.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&destination.raw), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast::<c_void>(),
                size_of::<NttCosetParameters>(),
                2,
            );
        }
        session
            .dispatch_threadgroups(
                &encoder,
                &session.ntt.coset_prepare,
                MTLSize {
                    width: 1,
                    height: 2,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
            .expect("dispatch strided coset preparation");
        encoder.endEncoding();
        session
            .submit_and_wait(PendingCommand {
                command,
                _resources: vec![source.raw.clone(), destination.raw.clone()],
            })
            .expect("complete strided coset preparation");

        let actual = session
            .read_values::<Bls12_381Fr>(&destination, 6)
            .expect("read strided destination");
        assert_eq!(actual, [first, zero, zero, second, zero, zero]);
    }

    #[test]
    fn graph_expression_abi_is_complete_and_pinned() {
        let expected = expected_graph_abi();
        assert_eq!(expected.value_ref_layout, [12, 4, 0, 4, 8]);
        assert_eq!(expected.operation_layout, [40, 4, 0, 4, 8, 12, 16, 28]);
        assert_eq!(
            expected.parameters_layout,
            [
                256, 8, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72,
                76, 80, 84, 88, 92, 96, 128, 160, 192, 224,
            ]
        );
        assert_eq!(expected.value_tags, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(expected.operation_tags, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(expected.post_kinds, [0, 1, 2]);
        assert_eq!(expected.scratch_classes, [8, 16, 32, 64, 128, 256]);

        let expected_words = graph_abi_words(expected);
        assert_eq!(
            graph_abi_fingerprint(&expected_words[..GRAPH_ABI_WORDS - 1]),
            GRAPH_ABI_FINGERPRINT
        );

        let session = MetalSession::new().expect("initialize Metal runtime");
        let actual = session
            .graph_abi_probe()
            .expect("execute graph-expression ABI probe");
        assert_eq!(actual, expected);

        for index in 0..GRAPH_ABI_WORDS {
            let mut shifted = expected_words;
            shifted[index] ^= 1;
            assert!(
                matches!(
                    validate_graph_abi(graph_abi_from_words(shifted)),
                    Err(MetalError::GraphAbi(_))
                ),
                "graph ABI word {index} was not validated"
            );
        }
    }

    #[test]
    fn host_plan_montgomery_multiplication_matches_blst() {
        let mut state = 0x4e54_545f_504c_414e;
        for _ in 0..2_048 {
            let lhs = fr_from_u64(splitmix64(&mut state));
            let rhs = fr_from_u64(splitmix64(&mut state));
            let lhs_blst = blst::blst_fr { l: lhs.limbs };
            let rhs_blst = blst::blst_fr { l: rhs.limbs };
            let mut expected = blst::blst_fr::default();
            // SAFETY: all three BLST field objects are initialized and live for
            // the full non-overlapping call.
            unsafe {
                blst::blst_fr_mul(&mut expected, &lhs_blst, &rhs_blst);
            }
            assert_eq!(fr_montgomery_mul_host(lhs, rhs).limbs, expected.l);
        }
    }

    #[test]
    fn fp32_internal_chains_match_repeated_blst_multiplication() {
        let mut state = 0x4650_3332_4348_4149;
        let lhs = (0..128)
            .map(|_| fp_from_u64(splitmix64(&mut state)))
            .collect::<Vec<_>>();
        let rhs = (0..128)
            .map(|_| fp_from_u64(splitmix64(&mut state) | 1))
            .collect::<Vec<_>>();
        let iterations = 7;

        let mut expected = Vec::with_capacity(lhs.len());
        let mut expected_pair = Vec::with_capacity(lhs.len());
        let mut expected_triple = Vec::with_capacity(lhs.len());
        for index in 0..lhs.len() {
            let multiplier = blst::blst_fp {
                l: rhs[index].limbs,
            };
            let mut accumulator = blst::blst_fp {
                l: lhs[index].limbs,
            };
            for _ in 0..iterations {
                let input = accumulator;
                // SAFETY: the initialized input/multiplier remain valid and
                // BLST permits the destination to alias neither immutable input.
                unsafe {
                    blst::blst_fp_mul(&mut accumulator, &input, &multiplier);
                }
            }
            expected.push(accumulator.l);

            let reverse_multiplier = blst::blst_fp {
                l: lhs[index].limbs,
            };
            let mut reverse_accumulator = blst::blst_fp {
                l: rhs[index].limbs,
            };
            for _ in 0..iterations {
                let input = reverse_accumulator;
                // SAFETY: all BLST values are initialized and the output does
                // not alias either immutable multiplication input.
                unsafe {
                    blst::blst_fp_mul(&mut reverse_accumulator, &input, &reverse_multiplier);
                }
            }
            let mut combined = blst::blst_fp::default();
            // SAFETY: the initialized accumulators remain live and the output
            // is a distinct BLST field object.
            unsafe {
                blst::blst_fp_add(&mut combined, &accumulator, &reverse_accumulator);
            }
            expected_pair.push(combined.l);

            let lhs_value = blst::blst_fp {
                l: lhs[index].limbs,
            };
            let rhs_value = blst::blst_fp {
                l: rhs[index].limbs,
            };
            let mut third_accumulator = blst::blst_fp::default();
            let mut third_multiplier = blst::blst_fp::default();
            // SAFETY: all inputs are initialized and both outputs are distinct
            // from their immutable BLST inputs.
            unsafe {
                blst::blst_fp_add(&mut third_accumulator, &lhs_value, &rhs_value);
                blst::blst_fp_sub(&mut third_multiplier, &lhs_value, &rhs_value);
            }
            for _ in 0..iterations {
                let input = third_accumulator;
                // SAFETY: the initialized input/multiplier remain valid and the
                // output is a separate BLST field object.
                unsafe {
                    blst::blst_fp_mul(&mut third_accumulator, &input, &third_multiplier);
                }
            }
            let mut triple = blst::blst_fp::default();
            // SAFETY: all inputs are initialized and the output is distinct.
            unsafe {
                blst::blst_fp_add(&mut triple, &combined, &third_accumulator);
            }
            expected_triple.push(triple.l);
        }

        let session = MetalSession::new().expect("initialize Metal runtime");
        for (variant, pipeline, expected) in [
            ("fp32_mulhi", &session.field.fp32.mul_chain, &expected),
            ("fp32_wide", &session.field.fp32.wide_mul_chain, &expected),
            (
                "fp32_wide_2",
                &session.field.fp32.wide_mul_chain_2,
                &expected_pair,
            ),
            (
                "fp32_wide_3",
                &session.field.fp32.wide_mul_chain_3,
                &expected_triple,
            ),
        ] {
            session.reset_activity();
            let (actual, elapsed) = session
                .benchmark_fp32_mul_chain(
                    &lhs,
                    &rhs,
                    Fp32ChainBenchmark {
                        pipeline,
                        variant,
                        iterations,
                        timed_dispatches: 2,
                        threadgroup_width: 32,
                    },
                )
                .expect("execute raw internal Fp32 multiplication chains");
            assert_ne!(elapsed, std::time::Duration::ZERO);
            for index in 0..lhs.len() {
                assert_eq!(
                    actual[index].limbs, expected[index],
                    "{variant} chain[{index}]"
                );
            }

            let activity = session.activity();
            assert_eq!(activity.command_buffers, 3);
            assert_eq!(activity.dispatches, 5);
            assert_eq!(activity.host_waits, 3);
        }
    }

    #[test]
    fn balanced_radix_schedule_covers_product_domain_sizes() {
        assert_eq!(&*ntt_radix_schedule(0), &[] as &[u32]);
        assert_eq!(&*ntt_radix_schedule(1), &[1]);
        assert_eq!(&*ntt_radix_schedule(8), &[8]);
        assert_eq!(&*ntt_radix_schedule(9), &[9]);
        assert_eq!(&*ntt_radix_schedule(13), &[7, 6]);
        assert_eq!(&*ntt_radix_schedule(14), &[7, 7]);
        assert_eq!(&*ntt_radix_schedule(15), &[8, 7]);
        assert_eq!(&*ntt_radix_schedule(16), &[8, 8]);
        assert_eq!(&*ntt_radix_schedule(17), &[9, 8]);
        for log_size in 1..=MAX_NTT_LOG_SIZE {
            let schedule = ntt_radix_schedule(log_size);
            assert_eq!(schedule.iter().sum::<u32>(), log_size);
            assert!(
                schedule
                    .iter()
                    .all(|degree| (1..=MAX_NTT_RADIX_LOG).contains(degree))
            );
        }
    }

    #[test]
    fn standard_product_domains_pass_algebraic_validation() {
        for log_size in 0..=17 {
            assert_eq!(
                validate_ntt_domain(
                    bls12_381_ntt_domain(log_size).expect("construct canonical NTT domain"),
                )
                .expect("standard domain must validate"),
                1usize << log_size
            );
        }
    }

    #[test]
    fn resident_domain_tags_reject_invalid_roots_before_allocation() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let mut invalid = bls12_381_ntt_domain(2).expect("construct canonical NTT domain");
        invalid.root = Bls12_381Fr::default();
        session.reset_activity();

        let values = vec![Bls12_381Fr::default(); 4];
        let error = match session.upload_resident_fr(
            &values,
            4,
            4,
            1,
            FrDomainTag::for_ntt(invalid, FrBasis::Lagrange),
        ) {
            Ok(_) => panic!("an invalid exact NTT domain must not enter resident storage"),
            Err(error) => error,
        };
        assert!(matches!(error, MetalError::NttDomain(_)));
        let activity = session.activity();
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.host_write_bytes, 0);
    }

    #[test]
    fn resident_views_reject_zero_reused_and_cross_arena_generations() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let values = vec![Bls12_381Fr::default(); 4];
        let domain = FrDomainTag::coefficient(2);
        let (mut arena, original) = session
            .upload_resident_fr(&values, 4, 4, 1, domain)
            .expect("allocate resident test arena");
        let never_initialized = ResidentFrView {
            session_id: session.session_id,
            arena_id: arena.arena_id,
            slot: 1,
            allocation_id: arena.slots[1].allocation_id,
            lease_generation: arena.slots[1].lease_generation,
            write_generation: 0,
            byte_offset: 0,
            logical_len: 4,
            stride: 4,
            batch_count: 1,
            domain,
        };
        assert!(matches!(
            arena.validate(&never_initialized),
            Err(MetalError::ResidentState(reason)) if reason.contains("generation zero")
        ));

        let first = arena
            .transition(&original, 4, 4, domain)
            .expect("write slot one");
        let second = arena
            .transition(&first, 4, 4, domain)
            .expect("reuse slot zero");
        let third = arena
            .transition(&second, 4, 4, domain)
            .expect("reuse slot one");
        assert!(arena.validate(&third).is_ok());
        assert!(matches!(
            arena.validate(&original),
            Err(MetalError::ResidentState("view generation is stale"))
        ));
        assert!(matches!(
            arena.validate(&first),
            Err(MetalError::ResidentState("view generation is stale"))
        ));

        let generations_before_rejection = arena.write_generations;
        assert!(matches!(
            arena.transition(&original, 4, 4, domain),
            Err(MetalError::ResidentState("view generation is stale"))
        ));
        assert_eq!(arena.write_generations, generations_before_rejection);

        arena.write_generations[1 - third.slot] = u64::MAX;
        let generations_before_exhaustion = arena.write_generations;
        assert!(matches!(
            arena.transition(&third, 4, 4, domain),
            Err(MetalError::ResidentState(
                "resident generation counter is exhausted"
            ))
        ));
        assert_eq!(arena.write_generations, generations_before_exhaustion);

        arena.finish(Ok(())).expect("clear and pool first arena");
        let (second_arena, _) = session
            .upload_resident_fr(&values, 4, 4, 1, domain)
            .expect("reuse buffers in a new arena");
        assert!(matches!(
            second_arena.validate(&third),
            Err(MetalError::ResidentState(
                "view belongs to a different resident arena"
            ))
        ));
    }

    #[test]
    fn resident_slot_state_rejects_basis_domain_and_stockham_relabeling() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let values = vec![Bls12_381Fr::default(); 4];
        let (mut arena, coefficients) = session
            .upload_resident_fr(&values, 4, 4, 1, FrDomainTag::coefficient(2))
            .expect("allocate authoritative-state test arena");
        let domain = bls12_381_ntt_domain(2).expect("construct canonical NTT domain");
        let working_domain = FrDomainTag::for_ntt(
            domain,
            FrBasis::StockhamWorking {
                direction: NttDirection::Forward,
                completed_log_p: 1,
            },
        );
        let working = arena
            .transition(&coefficients, 4, 4, working_domain)
            .expect("publish partial Stockham state");

        let mut relabeled = ResidentFrView {
            session_id: working.session_id,
            arena_id: working.arena_id,
            slot: working.slot,
            allocation_id: working.allocation_id,
            lease_generation: working.lease_generation,
            write_generation: working.write_generation,
            byte_offset: working.byte_offset,
            logical_len: working.logical_len,
            stride: working.stride,
            batch_count: working.batch_count,
            domain: FrDomainTag::for_ntt(domain, FrBasis::Lagrange),
        };
        assert!(matches!(
            arena.validate(&relabeled),
            Err(MetalError::ResidentState(
                "resident view does not match the authoritative slot state"
            ))
        ));

        let mut alternate = domain;
        std::mem::swap(&mut alternate.root, &mut alternate.inverse_root);
        relabeled.domain = FrDomainTag::for_ntt(
            alternate,
            FrBasis::StockhamWorking {
                direction: NttDirection::Forward,
                completed_log_p: 1,
            },
        );
        assert!(matches!(
            arena.validate(&relabeled),
            Err(MetalError::ResidentState(
                "resident view does not match the authoritative slot state"
            ))
        ));

        arena
            .finish(Ok::<(), MetalError>(()))
            .expect("clear authoritative-state test arena");
    }

    #[test]
    fn resident_views_reject_cross_session_identity_collisions() {
        let first_session = MetalSession::new().expect("initialize first Metal runtime");
        let second_session = MetalSession::new().expect("initialize second Metal runtime");
        let _first_execution = first_session.execution_guard();
        let _second_execution = second_session.execution_guard();
        let values = vec![Bls12_381Fr::default(); 4];
        let domain = FrDomainTag::coefficient(2);
        let (first_arena, first_view) = first_session
            .upload_resident_fr(&values, 4, 4, 1, domain)
            .expect("allocate first-session resident arena");
        let (second_arena, second_view) = second_session
            .upload_resident_fr(&values, 4, 4, 1, domain)
            .expect("allocate second-session resident arena");

        assert_eq!(first_view.arena_id, second_view.arena_id);
        assert_eq!(first_view.allocation_id, second_view.allocation_id);
        assert_eq!(first_view.lease_generation, second_view.lease_generation);
        assert_eq!(first_view.write_generation, second_view.write_generation);
        assert_ne!(first_view.session_id, second_view.session_id);
        assert!(matches!(
            second_arena.validate(&first_view),
            Err(MetalError::ResidentState(
                "view belongs to a different Metal session"
            ))
        ));

        first_arena
            .finish(Ok::<(), MetalError>(()))
            .expect("clear first-session arena");
        second_arena
            .finish(Ok::<(), MetalError>(()))
            .expect("clear second-session arena");
    }

    #[test]
    fn resident_view_constructor_rejects_misaligned_bounds_and_overflow() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let values = vec![Bls12_381Fr::default(); 8];
        let domain = FrDomainTag::coefficient(2);
        let (arena, full) = session
            .upload_resident_fr(&values, 4, 8, 2, domain)
            .expect("allocate strided resident test arena");
        assert!(arena.validate(&full).is_ok());

        assert!(matches!(
            arena.view_for_slot(0, 1, 1, 1, 1, 1, FrDomainTag::coefficient(0)),
            Err(MetalError::ResidentState(
                "resident view offset is not on an Fr element boundary"
            ))
        ));
        assert!(matches!(
            arena.view_for_slot(0, 1, 4 * size_of::<Bls12_381Fr>(), 5, 8, 2, domain),
            Err(MetalError::BufferBounds {
                requested,
                available
            }) if requested == 17 * size_of::<Bls12_381Fr>()
                && available == 16 * size_of::<Bls12_381Fr>()
        ));
        assert!(matches!(
            arena.view_for_slot(0, 1, size_of::<Bls12_381Fr>(), 4, 4, 1, domain),
            Err(MetalError::ResidentState(
                "resident Fr view is not aligned to a complete column"
            ))
        ));
        assert!(matches!(
            arena.view_for_slot(0, 1, 0, 9, 8, 1, domain),
            Err(MetalError::ResidentState(
                "resident view logical length exceeds its stride"
            ))
        ));
        assert!(matches!(
            arena.view_for_slot(0, 1, 0, 0, 8, 1, domain),
            Err(MetalError::ResidentState(
                "resident view dimensions must be nonzero"
            ))
        ));

        assert!(matches!(
            validate_resident_fr_view_shape(usize::MAX, 0, 1, usize::MAX, 2),
            Err(MetalError::ResidentState(
                "resident view element span overflows"
            ))
        ));
        let overflowing_elements = usize::MAX / size_of::<Bls12_381Fr>() + 1;
        assert!(matches!(
            validate_resident_fr_view_shape(
                usize::MAX,
                0,
                overflowing_elements,
                overflowing_elements,
                1
            ),
            Err(MetalError::ResidentState(
                "resident view byte span overflows"
            ))
        ));
        let aligned_end = usize::MAX - usize::MAX % size_of::<Bls12_381Fr>();
        assert!(matches!(
            validate_resident_fr_view_shape(usize::MAX, aligned_end, 1, 1, 1),
            Err(MetalError::ResidentState(
                "resident view end offset overflows"
            ))
        ));
    }

    #[test]
    fn resident_column_views_preserve_domain_and_cannot_escape_parent() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let values = vec![Bls12_381Fr::default(); 12];
        let domain = FrDomainTag::coefficient(2);
        let (arena, full) = session
            .upload_resident_fr(&values, 4, 8, 3, domain)
            .expect("allocate multi-batch resident test arena");

        let selected = arena
            .columns(&full, 1, 2)
            .expect("derive checked resident column view");
        assert_eq!(selected.arena_id, full.arena_id);
        assert_eq!(selected.allocation_id, full.allocation_id);
        assert_eq!(selected.write_generation, full.write_generation);
        assert_eq!(selected.byte_offset, 4 * size_of::<Bls12_381Fr>());
        assert_eq!(selected.logical_len, 4);
        assert_eq!(selected.stride, 4);
        assert_eq!(selected.batch_count, 2);
        assert_eq!(selected.domain, domain);
        assert!(arena.validate(&selected).is_ok());

        assert!(matches!(
            arena.columns(&full, 2, 2),
            Err(MetalError::ResidentState(
                "resident column range escapes its parent"
            ))
        ));
        assert!(matches!(
            arena.columns(&full, usize::MAX, 2),
            Err(MetalError::ResidentState("resident column range overflows"))
        ));
    }

    #[test]
    fn nonzero_column_subview_preserves_identity_across_stride_extension() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let one = Bls12_381Fr {
            limbs: generated::FR_R,
        };
        let zero = Bls12_381Fr::default();
        let values = [
            zero, zero, zero, zero, one, zero, zero, zero, zero, zero, zero, zero,
        ];
        let (mut arena, full) = session
            .upload_resident_fr(&values, 4, 8, 3, FrDomainTag::coefficient(2))
            .expect("allocate packed multi-column arena");
        let selected = arena
            .columns(&full, 1, 1)
            .expect("select the nonzero packed column");
        let domain = bls12_381_ntt_domain(3).expect("construct extended NTT domain");
        let command = session
            .command_buffer("midnight.ntt.nonzero_column_stride_extension")
            .expect("create stride-extension command");
        let encoder = command
            .computeCommandEncoder()
            .expect("create stride-extension encoder");
        let output = session
            .encode_coset_prepare(&encoder, &mut arena, selected, 4, 8, domain)
            .expect("encode nonzero-column stride extension");
        assert_eq!(output.byte_offset, 8 * size_of::<Bls12_381Fr>());
        assert_eq!(output.stride, 8);
        assert_eq!(output.batch_count, 1);
        encoder.endEncoding();
        session
            .submit_and_wait(PendingCommand {
                command,
                _resources: arena.retained_resources(),
            })
            .expect("complete nonzero-column stride extension");
        let actual = session
            .materialize_resident_fr(&arena, output)
            .expect("read nonzero-column stride extension");
        assert_eq!(actual, [one, zero, zero, zero, zero, zero, zero, zero]);
        arena
            .finish(Ok::<(), MetalError>(()))
            .expect("clear stride-extension test arena");
    }

    #[test]
    fn ntt_rejects_same_size_view_from_a_different_exact_domain() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let _execution = session.execution_guard();
        let canonical = bls12_381_ntt_domain(2).expect("construct canonical NTT domain");
        let mut alternate = canonical;
        std::mem::swap(&mut alternate.root, &mut alternate.inverse_root);
        validate_ntt_domain(alternate).expect("inverse root is also primitive at this size");
        let plan = session
            .ntt_plan(alternate)
            .expect("prepare alternate NTT plan");
        let values = vec![Bls12_381Fr::default(); 4];
        let (mut arena, input) = session
            .upload_resident_fr(
                &values,
                4,
                4,
                1,
                FrDomainTag::for_ntt(canonical, FrBasis::Lagrange),
            )
            .expect("allocate exact-domain test arena");
        let command = session
            .command_buffer("midnight.ntt.domain_mismatch")
            .expect("create exact-domain test command");
        let encoder = command
            .computeCommandEncoder()
            .expect("create exact-domain test encoder");
        let mut resources = arena.retained_resources();
        let error = match session.encode_ntt(
            &encoder,
            &mut resources,
            &mut arena,
            input,
            &plan,
            NttDirection::Inverse,
            false,
        ) {
            Ok(_) => panic!("a different exact NTT domain must be rejected"),
            Err(error) => error,
        };
        encoder.endEncoding();
        assert!(matches!(
            error,
            MetalError::ResidentState(
                "NTT direction/coset mode does not accept the resident Fr basis"
            )
        ));
        arena
            .finish(Ok::<(), MetalError>(()))
            .expect("clear exact-domain test arena");
    }

    #[test]
    fn injected_completion_failure_is_counted_and_never_read_back() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        session.reset_activity();
        let error = session
            .injected_command_failure_probe()
            .expect_err("injected completion failure must propagate");
        assert!(
            matches!(error, MetalError::Execution(ref reason) if reason == "injected completion failure")
        );

        let activity = session.activity();
        assert_eq!(activity.command_buffers, 1);
        assert_eq!(activity.dispatches, 1);
        assert_eq!(activity.host_waits, 1);
        assert_eq!(activity.execution_failures, 1);
        assert_eq!(activity.host_write_bytes, size_of::<u64>() as u64);
        assert_eq!(activity.host_read_bytes, 0);
    }

    #[test]
    fn injected_multi_pass_ntt_failure_clears_recycles_and_allows_fresh_work() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(10).expect("construct canonical NTT domain");
        let values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            1 << domain.log_size
        ];
        session
            .probe_bls12_381_ntt(&values, domain, NttDirection::Forward)
            .expect("warm immutable NTT plan");
        session.reset_activity();

        let retained_at_completion = std::sync::atomic::AtomicUsize::new(0);
        let error = session
            .injected_ntt_completion_failure_probe(&values, domain, &retained_at_completion)
            .expect_err("injected NTT completion failure must propagate");
        assert!(
            matches!(error, MetalError::Execution(ref reason) if reason == "injected NTT completion failure")
        );
        assert_eq!(retained_at_completion.load(Ordering::SeqCst), 4);

        let activity = session.activity();
        let value_bytes = size_of_val(values.as_slice()) as u64;
        assert_eq!(activity.command_buffers, 2);
        assert_eq!(activity.dispatches, 4); // Two NTT passes plus two full clears.
        assert_eq!(activity.host_waits, 2);
        assert_eq!(activity.allocations, 0);
        assert_eq!(activity.allocated_bytes, 0);
        assert_eq!(activity.host_write_bytes, value_bytes);
        assert_eq!(activity.host_read_bytes, 0);
        assert_eq!(activity.execution_failures, 1);
        assert_eq!(activity.pool_hits, 2);
        assert_eq!(activity.pool_misses, 0);
        assert_eq!(activity.pool_returns, 2);
        assert_eq!(activity.clear_commands, 1);
        assert_eq!(activity.clear_dispatches, 2);
        assert_eq!(activity.cleared_bytes, 2 * value_bytes);
        assert_eq!(activity.clear_failures, 0);
        assert_eq!(activity.quarantined_secret_bytes, 0);

        let restored = session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect("a successfully cleared compute failure must not poison the session");
        assert_eq!(restored, values);
    }

    #[test]
    fn resident_operation_panic_clears_recycles_and_resumes_unwind() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(9).expect("construct canonical NTT domain");
        let domain_len = 1usize << domain.log_size;
        let values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            domain_len
        ];
        session.reset_activity();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), MetalError> = session.with_resident_fr_arena(
                &values,
                domain_len,
                domain_len,
                1,
                FrDomainTag::coefficient(domain.log_size),
                |_arena, _input| panic!("injected resident operation panic"),
            );
        }));
        assert!(panic.is_err(), "the original operation panic must resume");

        let activity = session.activity();
        let value_bytes = size_of_val(values.as_slice()) as u64;
        assert_eq!(activity.pool_misses, 2);
        assert_eq!(activity.pool_returns, 2);
        assert_eq!(activity.clear_commands, 1);
        assert_eq!(activity.clear_dispatches, 2);
        assert_eq!(activity.cleared_bytes, 2 * value_bytes);
        assert_eq!(activity.clear_failures, 0);
        assert_eq!(activity.poison_events, 0);
        assert_eq!(activity.pooled_secret_bytes, 2 * value_bytes);
        assert_eq!(activity.quarantined_secret_bytes, 0);

        let restored = session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect("successfully cleared unwind capacity must be reusable");
        assert_eq!(restored, values);
    }

    #[test]
    fn resident_operation_panic_with_clear_failure_quarantines_and_poison_session() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(9).expect("construct canonical NTT domain");
        let domain_len = 1usize << domain.log_size;
        let values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            domain_len
        ];
        session.reset_activity();
        session
            .inject_next_clear_failure
            .store(true, Ordering::SeqCst);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), MetalError> = session.with_resident_fr_arena(
                &values,
                domain_len,
                domain_len,
                1,
                FrDomainTag::coefficient(domain.log_size),
                |_arena, _input| panic!("injected resident operation panic"),
            );
        }));
        assert!(panic.is_err(), "the original operation panic must resume");

        let failed = session.activity();
        let quarantined_bytes = 2 * size_of_val(values.as_slice()) as u64;
        assert_eq!(failed.clear_failures, 1);
        assert_eq!(failed.poison_events, 1);
        assert_eq!(failed.pooled_secret_bytes, 0);
        assert_eq!(failed.quarantined_secret_bytes, quarantined_bytes);

        let commands_before_rejection = failed.command_buffers;
        let error = session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect_err("a session poisoned during unwind must fail closed");
        assert!(matches!(error, MetalError::SessionPoisoned { .. }));
        assert_eq!(
            session.activity().command_buffers,
            commands_before_rejection
        );
    }

    #[test]
    fn benchmark_ntt_roundtrips_reports_cleanup_failure() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(9).expect("construct canonical NTT domain");
        let values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            1usize << domain.log_size
        ];
        session
            .inject_next_clear_failure
            .store(true, Ordering::SeqCst);

        let error = session
            .benchmark_ntt_roundtrips(&values, domain, 1)
            .expect_err("the benchmark must not hide resident cleanup failure");
        assert!(matches!(error, MetalError::SessionPoisoned { .. }));
    }

    #[test]
    fn activity_reset_rebases_peak_secret_bytes_to_current_ownership() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        session
            .activity
            .secret_peak_live_bytes
            .store(123, Ordering::Relaxed);

        session.reset_activity();
        let activity = session.activity();
        assert_eq!(activity.secret_live_bytes, 0);
        assert_eq!(activity.secret_peak_live_bytes, 0);
    }

    #[test]
    fn resident_pool_reuses_exact_sizes_and_contains_only_cleared_capacities() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(9).expect("construct canonical NTT domain");
        let values = (0..1usize << domain.log_size)
            .map(|index| fr_from_u64(index as u64 + 1))
            .collect::<Vec<_>>();
        session.ntt_plan(domain).expect("warm immutable NTT plan");
        session.reset_activity();

        for _ in 0..2 {
            let restored = session
                .probe_bls12_381_ntt_roundtrip(&values, domain)
                .expect("resident roundtrip");
            assert_eq!(restored, values);

            let state = session.lifecycle_guard();
            let pooled = state.pool.get(&size_of_val(values.as_slice())).unwrap();
            assert_eq!(pooled.len(), 2);
            for buffer in pooled {
                let words = buffer.bytes / size_of::<u64>();
                // SAFETY: cleanup completed before the pool was locked, the
                // shared mapping spans the full capacity, and u64 has no invalid bits.
                let contents = unsafe {
                    std::slice::from_raw_parts(buffer.raw.contents().as_ptr().cast::<u64>(), words)
                };
                assert!(contents.iter().all(|word| *word == 0));
            }
        }

        let activity = session.activity();
        let value_bytes = size_of_val(values.as_slice()) as u64;
        assert_eq!(activity.allocations, 2);
        assert_eq!(activity.pool_misses, 2);
        assert_eq!(activity.pool_hits, 2);
        assert_eq!(activity.pool_returns, 4);
        assert_eq!(activity.clear_commands, 2);
        assert_eq!(activity.clear_dispatches, 4);
        assert_eq!(activity.cleared_bytes, 4 * value_bytes);
        assert_eq!(activity.pooled_secret_bytes, 2 * value_bytes);
        assert_eq!(activity.secret_live_bytes, 2 * value_bytes);
        assert_eq!(activity.secret_peak_live_bytes, 2 * value_bytes);
        assert_eq!(activity.materializations, 2);
        assert_eq!(activity.whole_polynomial_materializations, 2);
    }

    #[test]
    fn injected_clear_failure_quarantines_poisoned_capacity_and_fails_closed() {
        let session = MetalSession::new().expect("initialize Metal runtime");
        let domain = bls12_381_ntt_domain(9).expect("construct canonical NTT domain");
        let values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            1 << domain.log_size
        ];
        session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect("warm plan and resident pool");
        session.reset_activity();
        session
            .inject_next_clear_failure
            .store(true, Ordering::SeqCst);

        let error = session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect_err("injected clear failure must poison the session");
        assert!(matches!(error, MetalError::SessionPoisoned { .. }));
        let failed = session.activity();
        let quarantined_bytes = 2 * size_of_val(values.as_slice()) as u64;
        assert_eq!(failed.clear_failures, 1);
        assert_eq!(failed.poison_events, 1);
        assert_eq!(failed.quarantined_secret_bytes, quarantined_bytes);
        assert_eq!(failed.pooled_secret_bytes, 0);

        let commands_before_rejection = failed.command_buffers;
        let empty_results = [
            session
                .probe_bls12_381_ntt(&[], domain, NttDirection::Forward)
                .map(|_| ()),
            session
                .probe_bls12_381_ntt_roundtrip(&[], domain)
                .map(|_| ()),
            session
                .probe_bls12_381_coeff_to_extended(&[], values.len(), domain)
                .map(|_| ()),
            session
                .probe_bls12_381_extended_to_coeff(&[], domain)
                .map(|_| ()),
        ];
        for result in empty_results {
            assert!(matches!(result, Err(MetalError::SessionPoisoned { .. })));
        }
        assert_eq!(
            session.activity().command_buffers,
            commands_before_rejection
        );

        let error = session
            .probe_bls12_381_ntt_roundtrip(&values, domain)
            .expect_err("a poisoned session must reject future work");
        assert!(matches!(error, MetalError::SessionPoisoned { .. }));
        assert_eq!(
            session.activity().command_buffers,
            commands_before_rejection
        );
    }

    fn fr_from_u64(value: u64) -> Bls12_381Fr {
        let canonical = [value, 0, 0, 0];
        let mut raw = blst::blst_fr::default();
        // SAFETY: the output is valid and the canonical input contains the four
        // limbs expected by BLST for the duration of the call.
        unsafe {
            blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr());
        }
        Bls12_381Fr { limbs: raw.l }
    }

    fn fp_from_u64(value: u64) -> Bls12_381Fp {
        let canonical = [value, 0, 0, 0, 0, 0];
        let mut raw = blst::blst_fp::default();
        // SAFETY: the output is valid and the canonical input contains the six
        // limbs expected by BLST for the duration of the call.
        unsafe {
            blst::blst_fp_from_uint64(&mut raw, canonical.as_ptr());
        }
        Bls12_381Fp { limbs: raw.l }
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod throughput_tests {
    use super::*;

    const DEFAULT_ELEMENTS: usize = 1 << 18;
    const DEFAULT_ROUNDS: usize = 32;
    const DEFAULT_NTT_LOG_SIZE: usize = 17;
    const DEFAULT_NTT_BATCHES: usize = 4;
    const DEFAULT_NTT_ROUNDS: usize = 16;

    #[test]
    #[ignore = "release-mode Metal throughput diagnostic"]
    fn report_field64_throughput() {
        if cfg!(debug_assertions) {
            panic!(
                "run this diagnostic with `cargo test -p midnight-metal-prover --release \
                 report_field64_throughput -- --ignored --nocapture`"
            );
        }

        let elements = environment_usize("MIDNIGHT_METAL_BENCH_ELEMENTS", DEFAULT_ELEMENTS);
        let rounds = environment_usize("MIDNIGHT_METAL_BENCH_ROUNDS", DEFAULT_ROUNDS);
        assert!(
            elements != 0,
            "MIDNIGHT_METAL_BENCH_ELEMENTS must be nonzero"
        );
        assert!(rounds != 0, "MIDNIGHT_METAL_BENCH_ROUNDS must be nonzero");

        let session = MetalSession::new().expect("initialize Metal runtime");
        let device = session.device_info().expect("query Metal device");
        println!(
            "metal_field64_throughput device={:?} elements={elements} rounds={rounds}",
            device.name
        );

        let fr_lhs = vec![
            Bls12_381Fr {
                limbs: generated::FR_R
            };
            elements
        ];
        let fr_rhs = vec![
            Bls12_381Fr {
                limbs: generated::FR_R2
            };
            elements
        ];
        for (operation, pipeline) in [
            ("add", &session.field.fr64.add),
            ("sub", &session.field.fr64.sub),
            ("mul", &session.field.fr64.mul),
            ("square", &session.field.fr64.square),
        ] {
            report_pipeline(
                &session, "Fr64", operation, &fr_lhs, &fr_rhs, pipeline, rounds,
            );
        }

        let fp_lhs = vec![
            Bls12_381Fp {
                limbs: generated::FP_R
            };
            elements
        ];
        let fp_rhs = vec![
            Bls12_381Fp {
                limbs: generated::FP_R2
            };
            elements
        ];
        for (operation, pipeline) in [
            ("add", &session.field.fp64.add),
            ("sub", &session.field.fp64.sub),
            ("mul", &session.field.fp64.mul),
            ("square", &session.field.fp64.square),
        ] {
            report_pipeline(
                &session, "Fp64", operation, &fp_lhs, &fp_rhs, pipeline, rounds,
            );
        }
    }

    #[test]
    #[ignore = "release-mode raw internal Fp32 multiplication diagnostic"]
    fn report_fp32_raw_mul_throughput() {
        report_fp32_raw_mul_variant("mulhi", "report_fp32_raw_mul_throughput", false, 1);
    }

    #[test]
    #[ignore = "release-mode raw wide-accumulator Fp32 multiplication diagnostic"]
    fn report_fp32_wide_raw_mul_throughput() {
        report_fp32_raw_mul_variant("wide", "report_fp32_wide_raw_mul_throughput", true, 1);
    }

    #[test]
    #[ignore = "release-mode two-state wide-accumulator Fp32 ILP diagnostic"]
    fn report_fp32_wide_interleaved_2_raw_mul_throughput() {
        report_fp32_raw_mul_variant(
            "wide_interleaved_2",
            "report_fp32_wide_interleaved_2_raw_mul_throughput",
            true,
            2,
        );
    }

    #[test]
    #[ignore = "release-mode three-state wide-accumulator Fp32 ILP diagnostic"]
    fn report_fp32_wide_interleaved_3_raw_mul_throughput() {
        report_fp32_raw_mul_variant(
            "wide_interleaved_3",
            "report_fp32_wide_interleaved_3_raw_mul_throughput",
            true,
            3,
        );
    }

    fn report_fp32_raw_mul_variant(
        variant: &str,
        test_name: &str,
        wide: bool,
        interleaving: usize,
    ) {
        if cfg!(debug_assertions) {
            panic!(
                "run this diagnostic with `cargo test -p midnight-metal-prover --release \
                 {test_name} -- --ignored --nocapture`"
            );
        }

        const REQUIRED_OPS_PER_SECOND: f64 = 2_000_000_000.0;
        let elements = environment_usize("MIDNIGHT_METAL_FP32_ELEMENTS", DEFAULT_ELEMENTS);
        let iterations = environment_usize("MIDNIGHT_METAL_FP32_ITERATIONS", 64);
        let timed_dispatches = environment_usize("MIDNIGHT_METAL_FP32_DISPATCHES", 4);
        assert!(elements != 0, "Fp32 benchmark elements must be nonzero");
        let iterations = u32::try_from(iterations).expect("Fp32 iterations fit u32");
        let timed_dispatches =
            u32::try_from(timed_dispatches).expect("Fp32 timed dispatches fit u32");
        assert!(iterations != 0, "Fp32 iterations must be nonzero");
        assert!(
            timed_dispatches != 0,
            "Fp32 timed dispatches must be nonzero"
        );

        let lhs = vec![
            Bls12_381Fp {
                limbs: generated::FP_R,
            };
            elements
        ];
        let rhs = vec![
            Bls12_381Fp {
                limbs: generated::FP_R2,
            };
            elements
        ];
        let session = MetalSession::new().expect("initialize Metal runtime");
        let device = session.device_info().expect("query Metal device");
        let pipeline = match (wide, interleaving) {
            (false, 1) => &session.field.fp32.mul_chain,
            (true, 1) => &session.field.fp32.wide_mul_chain,
            (true, 2) => &session.field.fp32.wide_mul_chain_2,
            (true, 3) => &session.field.fp32.wide_mul_chain_3,
            _ => panic!("unsupported Fp32 benchmark variant"),
        };
        let execution_width = pipeline.threadExecutionWidth();
        let maximum_width = pipeline.maxTotalThreadsPerThreadgroup();
        let mut widths = vec![execution_width, 64, 128, 256, maximum_width];
        widths.retain(|width| *width <= maximum_width && width.is_multiple_of(execution_width));
        widths.sort_unstable();
        widths.dedup();

        let operations = elements as f64
            * f64::from(iterations)
            * f64::from(timed_dispatches)
            * interleaving as f64;
        let mut best = (0.0, 0usize);
        for width in widths {
            session
                .benchmark_fp32_mul_chain(
                    &lhs,
                    &rhs,
                    Fp32ChainBenchmark {
                        pipeline,
                        variant,
                        iterations: 8,
                        timed_dispatches: 1,
                        threadgroup_width: width,
                    },
                )
                .expect("warm up raw Fp32 multiplication chain");
            session.reset_activity();
            let (output, elapsed) = session
                .benchmark_fp32_mul_chain(
                    &lhs,
                    &rhs,
                    Fp32ChainBenchmark {
                        pipeline,
                        variant,
                        iterations,
                        timed_dispatches,
                        threadgroup_width: width,
                    },
                )
                .expect("execute raw Fp32 multiplication throughput chain");
            assert_eq!(output.len(), elements);
            let rate = operations / elapsed.as_secs_f64();
            if rate > best.0 {
                best = (rate, width);
            }
            let activity = session.activity();
            println!(
                "metal_fp32_{variant}_raw_mul device={:?} elements={elements} iterations={} \
                 interleaving={interleaving} \
                 threadgroup_width={width} execution_width={execution_width} \
                 maximum_width={maximum_width} timed_commands=1 timed_dispatches={} \
                 timed_waits=1 total_commands={} total_dispatches={} total_waits={} \
                 ops_per_second={rate:.0} elapsed_ms={:.3}",
                device.name,
                iterations,
                timed_dispatches,
                activity.command_buffers,
                activity.dispatches,
                activity.host_waits,
                elapsed.as_secs_f64() * 1_000.0,
            );
        }

        let decision = if best.0 > REQUIRED_OPS_PER_SECOND {
            "GO"
        } else {
            "NO_GO"
        };
        println!(
            "metal_fp32_{variant}_raw_mul_gate threshold_ops_per_second={REQUIRED_OPS_PER_SECOND:.0} \
             best_ops_per_second={:.0} best_threadgroup_width={} ratio={:.4} decision={decision}",
            best.0,
            best.1,
            best.0 / REQUIRED_OPS_PER_SECOND,
        );
    }

    #[test]
    #[ignore = "release-mode resident NTT phase diagnostic"]
    fn report_resident_ntt_throughput() {
        if cfg!(debug_assertions) {
            panic!(
                "run this diagnostic with `cargo test -p midnight-metal-prover --release \
                 report_resident_ntt_throughput -- --ignored --nocapture`"
            );
        }

        let log_size = environment_usize("MIDNIGHT_METAL_NTT_BENCH_LOG_SIZE", DEFAULT_NTT_LOG_SIZE);
        let batches = environment_usize("MIDNIGHT_METAL_NTT_BENCH_BATCHES", DEFAULT_NTT_BATCHES);
        let rounds = environment_usize("MIDNIGHT_METAL_NTT_BENCH_ROUNDS", DEFAULT_NTT_ROUNDS);
        let log_size = u32::try_from(log_size).expect("NTT benchmark log size fits u32");
        assert!((1..=MAX_NTT_LOG_SIZE).contains(&log_size));
        assert!(batches != 0, "NTT benchmark batches must be nonzero");
        assert!(rounds != 0, "NTT benchmark rounds must be nonzero");

        let domain = bls12_381_ntt_domain(log_size).expect("construct canonical NTT domain");
        let domain_len = 1usize << log_size;
        let mut values = vec![
            Bls12_381Fr {
                limbs: generated::FR_R,
            };
            domain_len * batches
        ];
        for (index, value) in values.iter_mut().enumerate().skip(1) {
            if index & 1 != 0 {
                *value = Bls12_381Fr {
                    limbs: generated::FR_R2,
                };
            }
        }

        let session = MetalSession::new().expect("initialize Metal runtime");
        let device = session.device_info().expect("query Metal device");
        session
            .benchmark_ntt_roundtrips(&values, domain, 1)
            .expect("warm up resident NTT");
        session.reset_activity();
        let elapsed = session
            .benchmark_ntt_roundtrips(&values, domain, rounds)
            .expect("execute resident NTT phase benchmark");
        let activity = session.activity();
        let transforms = 2 * rounds;
        let element_transforms = (domain_len * batches * transforms) as f64;
        println!(
            "metal_resident_ntt_throughput device={:?} log_size={log_size} batches={batches} \
             roundtrips={rounds} transforms={transforms} elements_per_second={:.0} \
             elapsed_ms={:.3} command_buffers={} dispatches={} host_waits={} \
             host_write_bytes={} host_read_bytes={}",
            device.name,
            element_transforms / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0,
            activity.command_buffers,
            activity.dispatches,
            activity.host_waits,
            activity.host_write_bytes,
            activity.host_read_bytes,
        );
    }

    fn report_pipeline<T: MetalPod + Default>(
        session: &MetalSession,
        field: &str,
        operation: &str,
        lhs: &[T],
        rhs: &[T],
        pipeline: &Pipeline,
        rounds: usize,
    ) {
        let label = format!("{}.{}", field.to_ascii_lowercase(), operation);
        session
            .benchmark_field_pipeline(lhs, rhs, pipeline, &label, 1)
            .expect("warm up field pipeline");
        let elapsed = session
            .benchmark_field_pipeline(lhs, rhs, pipeline, &label, rounds)
            .expect("execute field throughput benchmark");
        let operations = lhs.len() as f64 * rounds as f64;
        println!(
            "metal_field64_throughput field={field} op={operation} ops_per_second={:.0} \
             elapsed_ms={:.3}",
            operations / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0
        );
    }

    fn environment_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .unwrap_or_else(|error| panic!("invalid {name}={value:?}: {error}"))
            })
            .unwrap_or(default)
    }
}
