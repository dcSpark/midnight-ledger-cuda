// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Audited Objective-C and shared-memory boundary for the manual Metal gate.

use std::ffi::c_void;
use std::fmt;
use std::format;
use std::mem::{size_of, size_of_val};
use std::ptr::NonNull;
use std::string::{String, ToString};
use std::time::{Duration, Instant};
use std::vec;
use std::vec::Vec;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
};

use super::{
    AMNS8_ALGORITHM_VERSION, AMNS8_ISSUED_LEAF_SLOTS_PER_CONVOLUTION,
    AMNS8_ISSUED_PRODUCTS_PER_FIELD, AMNS8_ISSUED_WIDENED_PRODUCTS_PER_FIELD,
    AMNS8_ISSUED_WRAPPING_PRODUCTS_PER_FIELD, AMNS8_LOGICAL_LEAVES_PER_CONVOLUTION,
    AMNS8_LOGICAL_PRODUCTS_PER_FIELD, AMNS8_LOGICAL_WIDENED_PRODUCTS_PER_FIELD,
    AMNS8_LOGICAL_WRAPPING_PRODUCTS_PER_FIELD, AMNS8_OUTPUT_CANARY, AMNS8_SCHEDULE_FINGERPRINT,
    AMNS8_TOKENIZED_ALLOCATION_PEAK_LIVE_WORDS, AMNS8_TOKENIZED_PEAK_LIVE_WORDS,
    AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS, AMNS8_TYPED_PEAK_LIVE_WORDS, Amns8FieldElement, FP_INV,
    FP_MODULUS, GateInput, GateOutput, GateParameters, KARATSUBA_L4_OUTPUT_CANARY,
    U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT, U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS,
    U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION,
    U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4,
    U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC, U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS,
    U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE, U32_BLOCKED128_KARATSUBA_L4_ROUNDS,
    U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
    U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT,
    U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS,
    U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS,
    U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS, U32_FIELD_ABI_FINGERPRINT,
    U32_FIELD_ALGORITHM_VERSION, U32_KARATSUBA_L4_ABI_FINGERPRINT,
    U32_KARATSUBA_L4_ALGORITHM_VERSION, U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12,
    U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC, U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS,
    U32_KARATSUBA_L4_LAZY_MULTIPLE, U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
    U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT, U32_NAMED_FIELD_ABI_FINGERPRINT,
    U32_NAMED_FIELD_ALGORITHM_VERSION, U32_NAMED_FIELD_VARIANT_COUNT, U32FieldElement,
    U32FieldParameters, U32ProductGateInput, U32ProductGateOutput, u32_field_is_l4,
    u32_field_is_reduced,
};

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bls12-381-metal-msm.metallib"));

const ABI_FUNCTION: &str = "bls12_381_metal_msm_product_issue_abi";
const PRODUCT_FUNCTION: &str = "bls12_381_metal_msm_precise_fp32_product_issue";
const U32_PRODUCT_ABI_FUNCTION: &str = "bls12_381_metal_msm_u32_low_limb_product_issue_abi";
const U32_PRODUCT_FUNCTION: &str = "bls12_381_metal_msm_u32_low_limb_product_issue";
const U32_FIELD_ABI_FUNCTION: &str = "bls12_381_metal_msm_u32_field_abi";
const U32_FIELD_FUNCTION: &str = "bls12_381_metal_msm_u32_field_chain";
const U32_NAMED_FIELD_ABI_FUNCTION: &str = "bls12_381_metal_msm_u32_named_field_abi";
const U32_NAMED_FIELD_FUNCTION: &str = "bls12_381_metal_msm_u32_named_fios_chain";
const U32_NAMED_FIELD_REVERSED_FUNCTION: &str = "bls12_381_metal_msm_u32_named_fios_reversed_chain";
const U32_KARATSUBA_L4_ABI_FUNCTION: &str = "bls12_381_metal_msm_u32_named_karatsuba_l4_abi";
const U32_KARATSUBA_L4_FUNCTION: &str = "bls12_381_metal_msm_u32_named_karatsuba_l4_chain";
const U32_BLOCKED128_KARATSUBA_L4_ABI_FUNCTION: &str =
    "bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_abi";
const U32_BLOCKED128_KARATSUBA_L4_FUNCTION: &str =
    "bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_chain";
const U32_BLOCKED128_KARATSUBA_L4_ROTATING_FUNCTION: &str =
    "bls12_381_metal_msm_u32_named_blocked128_karatsuba_l4_rotating_chain";
const AMNS8_FIELD_ABI_FUNCTION: &str = "bls12_381_metal_msm_amns8_field_abi";
const AMNS8_FIELD_MULTIPLY_FUNCTION: &str = "bls12_381_metal_msm_amns8_field_multiply";
const AMNS8_ROTATING_CHAIN_FUNCTION: &str = "bls12_381_metal_msm_amns8_rotating_chain";
const ABI_FINGERPRINT: u64 = 0x56b2_7f5b_4ad9_e13d;
const U32_PRODUCT_ABI_FINGERPRINT: u64 = 0xa513_30b1_9d22_e047;
const EXPECTED_ABI: [u64; 8] = [
    1,
    size_of::<GateInput>() as u64,
    16,
    16,
    size_of::<GateOutput>() as u64,
    size_of::<GateParameters>() as u64,
    32,
    ABI_FINGERPRINT,
];
const EXPECTED_U32_PRODUCT_ABI: [u64; 8] = [
    1,
    size_of::<U32ProductGateInput>() as u64,
    16,
    16,
    size_of::<U32ProductGateOutput>() as u64,
    size_of::<GateParameters>() as u64,
    32,
    U32_PRODUCT_ABI_FINGERPRINT,
];
const EXPECTED_U32_FIELD_ABI: [u64; 16] = [
    1,
    size_of::<U32FieldElement>() as u64,
    4,
    size_of::<U32FieldParameters>() as u64,
    4,
    4,
    0,
    2,
    12,
    384,
    FP_INV as u64,
    FP_MODULUS[0] as u64,
    FP_MODULUS[11] as u64,
    U32_FIELD_ALGORITHM_VERSION as u64,
    U32_FIELD_ABI_FINGERPRINT,
    4,
];
const EXPECTED_U32_NAMED_FIELD_ABI: [u64; 16] = [
    1,
    size_of::<U32FieldElement>() as u64,
    4,
    size_of::<U32FieldParameters>() as u64,
    4,
    4,
    0,
    1,
    12,
    384,
    FP_INV as u64,
    FP_MODULUS[0] as u64,
    FP_MODULUS[11] as u64,
    U32_NAMED_FIELD_ALGORITHM_VERSION as u64,
    U32_NAMED_FIELD_ABI_FINGERPRINT,
    U32_NAMED_FIELD_VARIANT_COUNT as u64,
];
const EXPECTED_U32_KARATSUBA_L4_ABI: [u64; 18] = [
    1,
    size_of::<U32FieldElement>() as u64,
    4,
    size_of::<U32FieldParameters>() as u64,
    4,
    4,
    0,
    1,
    12,
    384,
    U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12 as u64,
    U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC as u64,
    U32_KARATSUBA_L4_LAZY_MULTIPLE as u64,
    U32_KARATSUBA_L4_ALGORITHM_VERSION as u64,
    U32_KARATSUBA_L4_ABI_FINGERPRINT,
    U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
    U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS as u64,
    U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT as u64,
];
const EXPECTED_U32_BLOCKED128_KARATSUBA_L4_ABI: [u64; 23] = [
    1,
    size_of::<U32FieldElement>() as u64,
    4,
    size_of::<U32FieldParameters>() as u64,
    4,
    4,
    0,
    1,
    12,
    384,
    U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4 as u64,
    U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC as u64,
    U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS as u64,
    U32_BLOCKED128_KARATSUBA_L4_ROUNDS as u64,
    U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS as u64,
    U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE as u64,
    U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION as u64,
    U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT,
    U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
    U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS as u64,
    U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS as u64,
    U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS as u64,
    U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT as u64,
];
const EXPECTED_AMNS8_FIELD_ABI: [u64; 28] = [
    0x414d_4e53_384d_534d,
    1,
    28,
    size_of::<Amns8FieldElement>() as u64,
    8,
    size_of::<U32FieldParameters>() as u64,
    4,
    0,
    4,
    8,
    12,
    8,
    8,
    64,
    AMNS8_ALGORITHM_VERSION as u64,
    AMNS8_LOGICAL_LEAVES_PER_CONVOLUTION as u64,
    AMNS8_LOGICAL_WIDENED_PRODUCTS_PER_FIELD as u64,
    AMNS8_LOGICAL_WRAPPING_PRODUCTS_PER_FIELD as u64,
    AMNS8_LOGICAL_PRODUCTS_PER_FIELD as u64,
    AMNS8_ISSUED_LEAF_SLOTS_PER_CONVOLUTION as u64,
    AMNS8_ISSUED_WIDENED_PRODUCTS_PER_FIELD as u64,
    AMNS8_ISSUED_WRAPPING_PRODUCTS_PER_FIELD as u64,
    AMNS8_ISSUED_PRODUCTS_PER_FIELD as u64,
    AMNS8_TYPED_PEAK_LIVE_WORDS as u64,
    AMNS8_TOKENIZED_PEAK_LIVE_WORDS as u64,
    AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS as u64,
    AMNS8_TOKENIZED_ALLOCATION_PEAK_LIVE_WORDS as u64,
    AMNS8_SCHEDULE_FINGERPRINT,
];

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Debug)]
pub(super) struct DiagnosticError(String);

impl DiagnosticError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DiagnosticError {}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProductIssueSample {
    pub(super) gpu: Duration,
    pub(super) wall: Duration,
}

pub(super) struct ProductIssueHarness {
    device: Device,
    queue: Queue,
    pipeline: Pipeline,
    input: Buffer,
    output: Buffer,
    length: usize,
}

pub(super) struct U32FieldHarness {
    device: Device,
    queue: Queue,
    pipeline: Pipeline,
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    logical_fields: usize,
    physical_fields: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NamedFiosOrientation {
    AccumulatorFixed,
    AccumulatorScanned,
}

impl NamedFiosOrientation {
    pub(super) const ALL: [Self; 2] = [Self::AccumulatorFixed, Self::AccumulatorScanned];

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::AccumulatorFixed => "accumulator_fixed",
            Self::AccumulatorScanned => "accumulator_scanned",
        }
    }
}

pub(super) struct NamedFiosFieldHarness {
    device: Device,
    queue: Queue,
    accumulator_fixed_pipeline: Pipeline,
    accumulator_scanned_pipeline: Pipeline,
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    logical_fields: usize,
    physical_fields: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KaratsubaL4FieldVariant {
    GeneratedV2,
    BlockedRadix128V3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KaratsubaL4ChainMode {
    FixedRhs,
    RotatingOperands,
}

impl KaratsubaL4ChainMode {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::FixedRhs => "fixed_rhs",
            Self::RotatingOperands => "rotating_operands",
        }
    }
}

impl KaratsubaL4FieldVariant {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::GeneratedV2 => "Karatsuba-L4 v2",
            Self::BlockedRadix128V3 => "blocked-radix-128 Karatsuba-L4 v3",
        }
    }

    const fn fixed_rhs_pipeline_function(self) -> &'static str {
        match self {
            Self::GeneratedV2 => U32_KARATSUBA_L4_FUNCTION,
            Self::BlockedRadix128V3 => U32_BLOCKED128_KARATSUBA_L4_FUNCTION,
        }
    }

    const fn rotating_operands_pipeline_function(self) -> Option<&'static str> {
        match self {
            Self::GeneratedV2 => None,
            Self::BlockedRadix128V3 => Some(U32_BLOCKED128_KARATSUBA_L4_ROTATING_FUNCTION),
        }
    }

    const fn abi_function(self) -> &'static str {
        match self {
            Self::GeneratedV2 => U32_KARATSUBA_L4_ABI_FUNCTION,
            Self::BlockedRadix128V3 => U32_BLOCKED128_KARATSUBA_L4_ABI_FUNCTION,
        }
    }
}

pub(super) struct KaratsubaL4FieldHarness {
    device: Device,
    queue: Queue,
    fixed_rhs_pipeline: Pipeline,
    rotating_operands_pipeline: Option<Pipeline>,
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    logical_fields: usize,
    physical_fields: usize,
    variant: KaratsubaL4FieldVariant,
}

pub(super) struct Amns8FieldHarness {
    device: Device,
    queue: Queue,
    multiply_pipeline: Pipeline,
    rotating_pipeline: Pipeline,
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    logical_fields: usize,
    physical_fields: usize,
}

impl ProductIssueHarness {
    pub(super) fn new(inputs: &[GateInput]) -> Result<Self, DiagnosticError> {
        if inputs.is_empty() {
            return Err(DiagnosticError::new("the product gate requires input"));
        }
        u32::try_from(inputs.len())
            .map_err(|_| DiagnosticError::new("the input length exceeds the u32 shader ABI"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.diagnostic")));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let pipeline = load_pipeline(&device, &library, PRODUCT_FUNCTION)?;
        let abi_pipeline = load_pipeline(&device, &library, ABI_FUNCTION)?;

        let input = new_shared_buffer(&device, size_of_val(inputs), "diagnostic.input")?;
        let output_bytes = inputs
            .len()
            .checked_mul(size_of::<GateOutput>())
            .ok_or_else(|| DiagnosticError::new("output buffer size overflow"))?;
        let output = new_shared_buffer(&device, output_bytes, "diagnostic.output")?;
        copy_inputs(&input, inputs);
        zero_shared_buffer(&output, output_bytes);
        validate_abi(&device, &queue, &abi_pipeline)?;

        Ok(Self { device, queue, pipeline, input, output, length: inputs.len() })
    }

    pub(super) fn new_u32(inputs: &[U32ProductGateInput]) -> Result<Self, DiagnosticError> {
        if inputs.is_empty() {
            return Err(DiagnosticError::new("the u32 product gate requires input"));
        }
        u32::try_from(inputs.len())
            .map_err(|_| DiagnosticError::new("the u32 input length exceeds the shader ABI"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.u32-product")));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let pipeline = load_pipeline(&device, &library, U32_PRODUCT_FUNCTION)?;
        let abi_pipeline = load_pipeline(&device, &library, U32_PRODUCT_ABI_FUNCTION)?;

        let input = new_shared_buffer(&device, size_of_val(inputs), "u32-product.input")?;
        let output_bytes = inputs
            .len()
            .checked_mul(size_of::<U32ProductGateOutput>())
            .ok_or_else(|| DiagnosticError::new("u32 product output size overflow"))?;
        let output = new_shared_buffer(&device, output_bytes, "u32-product.output")?;
        copy_u32_product_inputs(&input, inputs);
        zero_shared_buffer(&output, output_bytes);
        validate_u32_product_abi(&device, &queue, &abi_pipeline)?;

        Ok(Self { device, queue, pipeline, input, output, length: inputs.len() })
    }

    pub(super) fn candidate_threadgroup_widths(&self) -> Vec<usize> {
        let execution = self.thread_execution_width();
        let maximum = self.max_threadgroup_width();
        let mut widths = vec![execution, 64, 128, 256, 512, maximum];
        widths.retain(|width| *width > 0 && *width <= maximum && width.is_multiple_of(execution));
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    pub(super) fn run(
        &self,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        if iterations == 0 || dispatches == 0 {
            return Err(DiagnosticError::new("iterations and dispatches must both be nonzero"));
        }
        if threadgroup_width == 0
            || threadgroup_width > self.max_threadgroup_width()
            || !threadgroup_width.is_multiple_of(self.thread_execution_width())
        {
            return Err(DiagnosticError::new(format!(
                "invalid threadgroup width {threadgroup_width}"
            )));
        }
        let length = u32::try_from(self.length)
            .map_err(|_| DiagnosticError::new("the input length exceeds the u32 shader ABI"))?;
        let parameters = GateParameters { length, iterations, reserved0: 0, reserved1: 0 };

        let wall_start = Instant::now();
        let command = retained_command(&self.queue)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| DiagnosticError::new("create compute command encoder"))?;
        encoder.setComputePipelineState(&self.pipeline);
        bind_product_arguments(&encoder, &self.input, &self.output, &parameters);
        let grid = MTLSize { width: self.length, height: 1, depth: 1 };
        let threads = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        for _ in 0..dispatches {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        check_completed(&command)?;
        let wall = wall_start.elapsed();
        let gpu = completed_gpu_duration(&command).ok_or_else(|| {
            DiagnosticError::new("Metal returned invalid completed-command GPU timestamps")
        })?;
        if gpu.is_zero() {
            return Err(DiagnosticError::new("Metal reported zero GPU duration"));
        }
        Ok(ProductIssueSample { gpu, wall })
    }

    pub(super) fn read_output(&self, index: usize) -> Result<GateOutput, DiagnosticError> {
        if index >= self.length {
            return Err(DiagnosticError::new(format!(
                "output index {index} exceeds length {}",
                self.length
            )));
        }
        Ok(read_gate_output(&self.output, index))
    }

    pub(super) fn read_u32_output(
        &self,
        index: usize,
    ) -> Result<U32ProductGateOutput, DiagnosticError> {
        if index >= self.length {
            return Err(DiagnosticError::new(format!(
                "u32 output index {index} exceeds length {}",
                self.length
            )));
        }
        Ok(read_u32_product_output(&self.output, index))
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn thread_execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    pub(super) fn max_threadgroup_width(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }
}

impl U32FieldHarness {
    pub(super) fn new(
        lhs: &[U32FieldElement],
        rhs: &[U32FieldElement],
    ) -> Result<Self, DiagnosticError> {
        if lhs.len() != rhs.len() {
            return Err(DiagnosticError::new(format!(
                "u32 field input length mismatch: {} != {}",
                lhs.len(),
                rhs.len()
            )));
        }
        let logical_fields = lhs.len();
        let logical_u32 = u32::try_from(logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;
        let physical_fields = logical_fields
            .max(1)
            .div_ceil(16)
            .checked_mul(16)
            .ok_or_else(|| DiagnosticError::new("padded field count overflow"))?;
        let _physical_threads = physical_fields
            .checked_mul(2)
            .ok_or_else(|| DiagnosticError::new("padded thread count overflow"))?;

        let mut padded_lhs = vec![U32FieldElement::default(); physical_fields];
        let mut padded_rhs = vec![U32FieldElement::default(); physical_fields];
        padded_lhs[..logical_fields].copy_from_slice(lhs);
        padded_rhs[..logical_fields].copy_from_slice(rhs);

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.u32-field")));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let pipeline = load_pipeline(&device, &library, U32_FIELD_FUNCTION)?;
        let abi_pipeline = load_pipeline(&device, &library, U32_FIELD_ABI_FUNCTION)?;
        if pipeline.threadExecutionWidth() != 32 {
            return Err(DiagnosticError::new(format!(
                "two-lane candidate requires SIMD32, device pipeline reports {}",
                pipeline.threadExecutionWidth()
            )));
        }

        let input_bytes = physical_fields
            .checked_mul(size_of::<U32FieldElement>())
            .ok_or_else(|| DiagnosticError::new("u32 field input byte count overflow"))?;
        let lhs_buffer = new_shared_buffer(&device, input_bytes, "u32-field.lhs")?;
        let rhs_buffer = new_shared_buffer(&device, input_bytes, "u32-field.rhs")?;
        let output = new_shared_buffer(&device, input_bytes, "u32-field.output")?;
        copy_u32_field_inputs(&lhs_buffer, &padded_lhs);
        copy_u32_field_inputs(&rhs_buffer, &padded_rhs);
        zero_shared_buffer(&output, input_bytes);
        validate_u32_field_abi(&device, &queue, &abi_pipeline)?;

        debug_assert_eq!(logical_u32 as usize, logical_fields);
        Ok(Self {
            device,
            queue,
            pipeline,
            lhs: lhs_buffer,
            rhs: rhs_buffer,
            output,
            logical_fields,
            physical_fields,
        })
    }

    pub(super) fn candidate_threadgroup_widths(&self) -> Vec<usize> {
        let execution = self.thread_execution_width();
        let maximum = self.max_threadgroup_width();
        let mut widths = vec![execution, 64, 128, 256, 512, maximum];
        widths.retain(|width| *width > 0 && *width <= maximum && width.is_multiple_of(execution));
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    pub(super) fn run(
        &self,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        if iterations == 0 || dispatches == 0 {
            return Err(DiagnosticError::new("iterations and dispatches must both be nonzero"));
        }
        if threadgroup_width == 0
            || threadgroup_width > self.max_threadgroup_width()
            || !threadgroup_width.is_multiple_of(self.thread_execution_width())
        {
            return Err(DiagnosticError::new(format!(
                "invalid threadgroup width {threadgroup_width}"
            )));
        }
        let logical_fields = u32::try_from(self.logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;
        let parameters =
            U32FieldParameters { logical_fields, iterations, reserved0: 0, reserved1: 0 };

        let wall_start = Instant::now();
        let command = retained_command(&self.queue)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| DiagnosticError::new("create compute command encoder"))?;
        encoder.setComputePipelineState(&self.pipeline);
        bind_u32_field_arguments(&encoder, &self.lhs, &self.rhs, &self.output, &parameters);
        let grid = MTLSize { width: self.physical_threads(), height: 1, depth: 1 };
        let threads = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        for _ in 0..dispatches {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        check_completed(&command)?;
        let wall = wall_start.elapsed();
        let gpu = completed_gpu_duration(&command).ok_or_else(|| {
            DiagnosticError::new("Metal returned invalid completed-command GPU timestamps")
        })?;
        if gpu.is_zero() {
            return Err(DiagnosticError::new("Metal reported zero GPU duration"));
        }
        Ok(ProductIssueSample { gpu, wall })
    }

    pub(super) fn read_outputs(&self) -> Vec<U32FieldElement> {
        (0..self.logical_fields).map(|index| read_u32_field_output(&self.output, index)).collect()
    }

    pub(super) fn padding_is_zero(&self) -> bool {
        (self.logical_fields..self.physical_fields)
            .all(|index| read_u32_field_output(&self.output, index) == U32FieldElement::default())
    }

    pub(super) const fn logical_fields(&self) -> usize {
        self.logical_fields
    }

    pub(super) const fn physical_fields(&self) -> usize {
        self.physical_fields
    }

    pub(super) const fn physical_threads(&self) -> usize {
        self.physical_fields * 2
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn thread_execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    pub(super) fn max_threadgroup_width(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn static_threadgroup_memory_bytes(&self) -> usize {
        self.pipeline.staticThreadgroupMemoryLength()
    }
}

impl NamedFiosFieldHarness {
    /// Creates a harness for reduced Montgomery residues.
    ///
    /// The generated FIOS schedule performs one final conditional subtraction,
    /// so every raw input must be strictly less than the field modulus.
    pub(super) fn new(
        lhs: &[U32FieldElement],
        rhs: &[U32FieldElement],
    ) -> Result<Self, DiagnosticError> {
        if lhs.len() != rhs.len() {
            return Err(DiagnosticError::new(format!(
                "named FIOS input length mismatch: {} != {}",
                lhs.len(),
                rhs.len()
            )));
        }
        if let Some(index) = lhs.iter().position(|&value| !u32_field_is_reduced(value)) {
            return Err(DiagnosticError::new(format!(
                "named FIOS lhs input {index} is not a reduced Montgomery residue"
            )));
        }
        if let Some(index) = rhs.iter().position(|&value| !u32_field_is_reduced(value)) {
            return Err(DiagnosticError::new(format!(
                "named FIOS rhs input {index} is not a reduced Montgomery residue"
            )));
        }
        let logical_fields = lhs.len();
        let logical_u32 = u32::try_from(logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.named-fios-field")));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let accumulator_scanned_pipeline =
            load_pipeline(&device, &library, U32_NAMED_FIELD_FUNCTION)?;
        let accumulator_fixed_pipeline =
            load_pipeline(&device, &library, U32_NAMED_FIELD_REVERSED_FUNCTION)?;
        let abi_pipeline = load_pipeline(&device, &library, U32_NAMED_FIELD_ABI_FUNCTION)?;

        let execution_width = accumulator_fixed_pipeline.threadExecutionWidth();
        if execution_width == 0
            || accumulator_scanned_pipeline.threadExecutionWidth() != execution_width
        {
            return Err(DiagnosticError::new(format!(
                "named FIOS variants report incompatible execution widths: {} and {}",
                execution_width,
                accumulator_scanned_pipeline.threadExecutionWidth()
            )));
        }
        let physical_fields = logical_fields
            .max(1)
            .div_ceil(execution_width)
            .checked_mul(execution_width)
            .ok_or_else(|| DiagnosticError::new("named FIOS padded field count overflow"))?;

        let mut padded_lhs = vec![U32FieldElement::default(); physical_fields];
        let mut padded_rhs = vec![U32FieldElement::default(); physical_fields];
        padded_lhs[..logical_fields].copy_from_slice(lhs);
        padded_rhs[..logical_fields].copy_from_slice(rhs);
        let input_bytes = physical_fields
            .checked_mul(size_of::<U32FieldElement>())
            .ok_or_else(|| DiagnosticError::new("named FIOS input byte count overflow"))?;
        let lhs_buffer = new_shared_buffer(&device, input_bytes, "named-fios-field.lhs")?;
        let rhs_buffer = new_shared_buffer(&device, input_bytes, "named-fios-field.rhs")?;
        let output = new_shared_buffer(&device, input_bytes, "named-fios-field.output")?;
        copy_u32_field_inputs(&lhs_buffer, &padded_lhs);
        copy_u32_field_inputs(&rhs_buffer, &padded_rhs);
        zero_shared_buffer(&output, input_bytes);
        validate_u32_named_field_abi(&device, &queue, &abi_pipeline)?;

        debug_assert_eq!(logical_u32 as usize, logical_fields);
        Ok(Self {
            device,
            queue,
            accumulator_fixed_pipeline,
            accumulator_scanned_pipeline,
            lhs: lhs_buffer,
            rhs: rhs_buffer,
            output,
            logical_fields,
            physical_fields,
        })
    }

    fn pipeline(
        &self,
        orientation: NamedFiosOrientation,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        match orientation {
            NamedFiosOrientation::AccumulatorFixed => &self.accumulator_fixed_pipeline,
            NamedFiosOrientation::AccumulatorScanned => &self.accumulator_scanned_pipeline,
        }
    }

    pub(super) fn candidate_threadgroup_widths(
        &self,
        orientation: NamedFiosOrientation,
    ) -> Vec<usize> {
        let execution = self.thread_execution_width(orientation);
        let maximum = self.max_threadgroup_width(orientation);
        let mut widths = vec![execution, 64, 128, 256, 512, maximum];
        widths.retain(|width| *width > 0 && *width <= maximum && width.is_multiple_of(execution));
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    pub(super) fn run(
        &self,
        orientation: NamedFiosOrientation,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        if iterations == 0 || dispatches == 0 {
            return Err(DiagnosticError::new("iterations and dispatches must both be nonzero"));
        }
        if threadgroup_width == 0
            || threadgroup_width > self.max_threadgroup_width(orientation)
            || !threadgroup_width.is_multiple_of(self.thread_execution_width(orientation))
        {
            return Err(DiagnosticError::new(format!(
                "invalid named FIOS threadgroup width {threadgroup_width}"
            )));
        }
        let logical_fields = u32::try_from(self.logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;
        let parameters =
            U32FieldParameters { logical_fields, iterations, reserved0: 0, reserved1: 0 };

        let wall_start = Instant::now();
        let command = retained_command(&self.queue)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| DiagnosticError::new("create compute command encoder"))?;
        encoder.setComputePipelineState(self.pipeline(orientation));
        bind_u32_field_arguments(&encoder, &self.lhs, &self.rhs, &self.output, &parameters);
        let grid = MTLSize { width: self.physical_fields, height: 1, depth: 1 };
        let threads = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        for _ in 0..dispatches {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        check_completed(&command)?;
        let wall = wall_start.elapsed();
        let gpu = completed_gpu_duration(&command).ok_or_else(|| {
            DiagnosticError::new("Metal returned invalid completed-command GPU timestamps")
        })?;
        if gpu.is_zero() {
            return Err(DiagnosticError::new("Metal reported zero GPU duration"));
        }
        Ok(ProductIssueSample { gpu, wall })
    }

    pub(super) fn read_outputs(&self) -> Vec<U32FieldElement> {
        (0..self.logical_fields).map(|index| read_u32_field_output(&self.output, index)).collect()
    }

    pub(super) fn padding_is_zero(&self) -> bool {
        (self.logical_fields..self.physical_fields)
            .all(|index| read_u32_field_output(&self.output, index) == U32FieldElement::default())
    }

    pub(super) const fn logical_fields(&self) -> usize {
        self.logical_fields
    }

    pub(super) const fn physical_fields(&self) -> usize {
        self.physical_fields
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn thread_execution_width(&self, orientation: NamedFiosOrientation) -> usize {
        self.pipeline(orientation).threadExecutionWidth()
    }

    pub(super) fn max_threadgroup_width(&self, orientation: NamedFiosOrientation) -> usize {
        self.pipeline(orientation).maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn static_threadgroup_memory_bytes(
        &self,
        orientation: NamedFiosOrientation,
    ) -> usize {
        self.pipeline(orientation).staticThreadgroupMemoryLength()
    }
}

impl KaratsubaL4FieldHarness {
    /// Creates a harness for lazy Montgomery residues strictly below `4p`.
    pub(super) fn new(
        lhs: &[U32FieldElement],
        rhs: &[U32FieldElement],
    ) -> Result<Self, DiagnosticError> {
        Self::new_for_variant(lhs, rhs, KaratsubaL4FieldVariant::GeneratedV2)
    }

    /// Creates a v3 blocked-radix-128 harness for lazy Montgomery residues
    /// strictly below `4p`.
    pub(super) fn new_blocked128(
        lhs: &[U32FieldElement],
        rhs: &[U32FieldElement],
    ) -> Result<Self, DiagnosticError> {
        Self::new_for_variant(lhs, rhs, KaratsubaL4FieldVariant::BlockedRadix128V3)
    }

    fn new_for_variant(
        lhs: &[U32FieldElement],
        rhs: &[U32FieldElement],
        variant: KaratsubaL4FieldVariant,
    ) -> Result<Self, DiagnosticError> {
        let label = variant.label();
        if lhs.len() != rhs.len() {
            return Err(DiagnosticError::new(format!(
                "{label} input length mismatch: {} != {}",
                lhs.len(),
                rhs.len()
            )));
        }
        if let Some(index) = lhs.iter().position(|&value| !u32_field_is_l4(value)) {
            return Err(DiagnosticError::new(format!("{label} lhs input {index} is not below 4p")));
        }
        if let Some(index) = rhs.iter().position(|&value| !u32_field_is_l4(value)) {
            return Err(DiagnosticError::new(format!("{label} rhs input {index} is not below 4p")));
        }
        let logical_fields = lhs.len();
        let logical_u32 = u32::try_from(logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str(&format!(
            "bls12-381-metal-msm.{}",
            match variant {
                KaratsubaL4FieldVariant::GeneratedV2 => "named-karatsuba-l4-field",
                KaratsubaL4FieldVariant::BlockedRadix128V3 => {
                    "named-blocked128-karatsuba-l4-field"
                }
            }
        ))));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let fixed_rhs_pipeline =
            load_pipeline(&device, &library, variant.fixed_rhs_pipeline_function())?;
        let rotating_operands_pipeline = variant
            .rotating_operands_pipeline_function()
            .map(|function| load_pipeline(&device, &library, function))
            .transpose()?;
        let abi_pipeline = load_pipeline(&device, &library, variant.abi_function())?;
        let execution_width = fixed_rhs_pipeline.threadExecutionWidth();
        if execution_width == 0 {
            return Err(DiagnosticError::new(format!(
                "{label} pipeline reported zero execution width"
            )));
        }
        if let Some(rotating_pipeline) = &rotating_operands_pipeline
            && rotating_pipeline.threadExecutionWidth() != execution_width
        {
            return Err(DiagnosticError::new(format!(
                "{label} fixed-RHS and rotating-operands pipelines report incompatible execution widths: {} and {}",
                execution_width,
                rotating_pipeline.threadExecutionWidth()
            )));
        }
        let physical_fields = logical_fields
            .max(1)
            .div_ceil(execution_width)
            .checked_mul(execution_width)
            .ok_or_else(|| DiagnosticError::new(format!("{label} padded field count overflow")))?;

        let mut padded_lhs = vec![U32FieldElement::default(); physical_fields];
        let mut padded_rhs = vec![U32FieldElement::default(); physical_fields];
        padded_lhs[..logical_fields].copy_from_slice(lhs);
        padded_rhs[..logical_fields].copy_from_slice(rhs);
        let input_bytes = physical_fields
            .checked_mul(size_of::<U32FieldElement>())
            .ok_or_else(|| DiagnosticError::new(format!("{label} input byte count overflow")))?;
        let lhs_buffer = new_shared_buffer(&device, input_bytes, "karatsuba-l4-field.lhs")?;
        let rhs_buffer = new_shared_buffer(&device, input_bytes, "karatsuba-l4-field.rhs")?;
        let output = new_shared_buffer(&device, input_bytes, "karatsuba-l4-field.output")?;
        copy_u32_field_inputs(&lhs_buffer, &padded_lhs);
        copy_u32_field_inputs(&rhs_buffer, &padded_rhs);
        let padded_output = vec![KARATSUBA_L4_OUTPUT_CANARY; physical_fields];
        copy_u32_field_inputs(&output, &padded_output);
        validate_u32_karatsuba_l4_abi(&device, &queue, &abi_pipeline, variant)?;

        debug_assert_eq!(logical_u32 as usize, logical_fields);
        Ok(Self {
            device,
            queue,
            fixed_rhs_pipeline,
            rotating_operands_pipeline,
            lhs: lhs_buffer,
            rhs: rhs_buffer,
            output,
            logical_fields,
            physical_fields,
            variant,
        })
    }

    fn pipeline(
        &self,
        chain_mode: KaratsubaL4ChainMode,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        match chain_mode {
            KaratsubaL4ChainMode::FixedRhs => &self.fixed_rhs_pipeline,
            KaratsubaL4ChainMode::RotatingOperands => {
                self.rotating_operands_pipeline.as_deref().expect(
                    "rotating operands are available only for the blocked-radix-128 v3 harness",
                )
            }
        }
    }

    pub(super) fn candidate_threadgroup_widths(&self) -> Vec<usize> {
        self.candidate_threadgroup_widths_for(KaratsubaL4ChainMode::FixedRhs)
    }

    pub(super) fn candidate_threadgroup_widths_for(
        &self,
        chain_mode: KaratsubaL4ChainMode,
    ) -> Vec<usize> {
        let execution = self.thread_execution_width_for(chain_mode);
        let maximum = self.max_threadgroup_width_for(chain_mode);
        let mut widths = vec![execution, 64, 128, 256, 512, maximum];
        widths.retain(|width| *width > 0 && *width <= maximum && width.is_multiple_of(execution));
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    pub(super) fn run(
        &self,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        self.run_chain(KaratsubaL4ChainMode::FixedRhs, iterations, dispatches, threadgroup_width)
    }

    pub(super) fn run_chain(
        &self,
        chain_mode: KaratsubaL4ChainMode,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        if iterations == 0 || dispatches == 0 {
            return Err(DiagnosticError::new("iterations and dispatches must both be nonzero"));
        }
        if threadgroup_width == 0
            || threadgroup_width > self.max_threadgroup_width_for(chain_mode)
            || !threadgroup_width.is_multiple_of(self.thread_execution_width_for(chain_mode))
        {
            return Err(DiagnosticError::new(format!(
                "invalid {} {} threadgroup width {threadgroup_width}",
                self.variant.label(),
                chain_mode.label(),
            )));
        }
        let logical_fields = u32::try_from(self.logical_fields)
            .map_err(|_| DiagnosticError::new("the field count exceeds the u32 shader ABI"))?;
        let parameters =
            U32FieldParameters { logical_fields, iterations, reserved0: 0, reserved1: 0 };

        // One command is committed and waited exactly once. All requested
        // dispatches are encoded before that single commit.
        let wall_start = Instant::now();
        let command = retained_command(&self.queue)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| DiagnosticError::new("create compute command encoder"))?;
        encoder.setComputePipelineState(self.pipeline(chain_mode));
        bind_u32_field_arguments(&encoder, &self.lhs, &self.rhs, &self.output, &parameters);
        let grid = MTLSize { width: self.physical_fields, height: 1, depth: 1 };
        let threads = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        for _ in 0..dispatches {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        check_completed(&command)?;
        let wall = wall_start.elapsed();
        let gpu = completed_gpu_duration(&command).ok_or_else(|| {
            DiagnosticError::new("Metal returned invalid completed-command GPU timestamps")
        })?;
        if gpu.is_zero() {
            return Err(DiagnosticError::new("Metal reported zero GPU duration"));
        }
        Ok(ProductIssueSample { gpu, wall })
    }

    pub(super) fn read_outputs(&self) -> Vec<U32FieldElement> {
        (0..self.logical_fields).map(|index| read_u32_field_output(&self.output, index)).collect()
    }

    pub(super) fn padding_is_untouched(&self) -> bool {
        (self.logical_fields..self.physical_fields)
            .all(|index| read_u32_field_output(&self.output, index) == KARATSUBA_L4_OUTPUT_CANARY)
    }

    pub(super) const fn logical_fields(&self) -> usize {
        self.logical_fields
    }

    pub(super) const fn physical_fields(&self) -> usize {
        self.physical_fields
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn thread_execution_width(&self) -> usize {
        self.thread_execution_width_for(KaratsubaL4ChainMode::FixedRhs)
    }

    pub(super) fn thread_execution_width_for(&self, chain_mode: KaratsubaL4ChainMode) -> usize {
        self.pipeline(chain_mode).threadExecutionWidth()
    }

    pub(super) fn max_threadgroup_width(&self) -> usize {
        self.max_threadgroup_width_for(KaratsubaL4ChainMode::FixedRhs)
    }

    pub(super) fn max_threadgroup_width_for(&self, chain_mode: KaratsubaL4ChainMode) -> usize {
        self.pipeline(chain_mode).maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn static_threadgroup_memory_bytes(&self) -> usize {
        self.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::FixedRhs)
    }

    pub(super) fn static_threadgroup_memory_bytes_for(
        &self,
        chain_mode: KaratsubaL4ChainMode,
    ) -> usize {
        self.pipeline(chain_mode).staticThreadgroupMemoryLength()
    }
}

impl Amns8FieldHarness {
    pub(super) fn new(
        lhs: &[Amns8FieldElement],
        rhs: &[Amns8FieldElement],
    ) -> Result<Self, DiagnosticError> {
        if lhs.len() != rhs.len() {
            return Err(DiagnosticError::new(format!(
                "AMNS8 input length mismatch: {} != {}",
                lhs.len(),
                rhs.len()
            )));
        }
        if let Some(index) =
            lhs.iter().position(|value| !crate::amns8_oracle::is_lazy(&value.coefficients))
        {
            return Err(DiagnosticError::new(format!(
                "AMNS8 lhs input {index} exceeds the lazy coefficient bound"
            )));
        }
        if let Some(index) =
            rhs.iter().position(|value| !crate::amns8_oracle::is_lazy(&value.coefficients))
        {
            return Err(DiagnosticError::new(format!(
                "AMNS8 rhs input {index} exceeds the lazy coefficient bound"
            )));
        }
        let logical_fields = lhs.len();
        u32::try_from(logical_fields)
            .map_err(|_| DiagnosticError::new("the AMNS8 field count exceeds the u32 ABI"))?;

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| DiagnosticError::new("no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| DiagnosticError::new("create retained-reference command queue"))?;
        queue.setLabel(Some(&NSString::from_str("bls12-381-metal-msm.amns8-field")));
        let library =
            device.newLibraryWithData_error(&DispatchData::from_static_bytes(METALLIB)).map_err(
                |error| DiagnosticError::new(format!("load embedded metallib: {error:?}")),
            )?;
        let multiply_pipeline = load_pipeline(&device, &library, AMNS8_FIELD_MULTIPLY_FUNCTION)?;
        let rotating_pipeline = load_pipeline(&device, &library, AMNS8_ROTATING_CHAIN_FUNCTION)?;
        let abi_pipeline = load_pipeline(&device, &library, AMNS8_FIELD_ABI_FUNCTION)?;
        let execution_width = rotating_pipeline.threadExecutionWidth();
        if execution_width == 0 || multiply_pipeline.threadExecutionWidth() != execution_width {
            return Err(DiagnosticError::new(format!(
                "AMNS8 pipelines report incompatible execution widths: {} and {}",
                multiply_pipeline.threadExecutionWidth(),
                execution_width
            )));
        }
        if execution_width != 32 {
            return Err(DiagnosticError::new(format!(
                "AMNS8 requires SIMD32, device pipeline reported width {execution_width}"
            )));
        }
        let physical_fields = logical_fields
            .max(1)
            .div_ceil(4)
            .checked_mul(4)
            .ok_or_else(|| DiagnosticError::new("AMNS8 padded field count overflow"))?;
        let mut padded_lhs = vec![Amns8FieldElement::default(); physical_fields];
        let mut padded_rhs = vec![Amns8FieldElement::default(); physical_fields];
        padded_lhs[..logical_fields].copy_from_slice(lhs);
        padded_rhs[..logical_fields].copy_from_slice(rhs);
        let input_bytes = physical_fields
            .checked_mul(size_of::<Amns8FieldElement>())
            .ok_or_else(|| DiagnosticError::new("AMNS8 input byte count overflow"))?;
        let lhs_buffer = new_shared_buffer(&device, input_bytes, "amns8-field.lhs")?;
        let rhs_buffer = new_shared_buffer(&device, input_bytes, "amns8-field.rhs")?;
        let output = new_shared_buffer(&device, input_bytes, "amns8-field.output")?;
        copy_amns8_field_inputs(&lhs_buffer, &padded_lhs);
        copy_amns8_field_inputs(&rhs_buffer, &padded_rhs);
        copy_amns8_field_inputs(&output, &vec![AMNS8_OUTPUT_CANARY; physical_fields]);
        validate_amns8_field_abi(&device, &queue, &abi_pipeline)?;

        Ok(Self {
            device,
            queue,
            multiply_pipeline,
            rotating_pipeline,
            lhs: lhs_buffer,
            rhs: rhs_buffer,
            output,
            logical_fields,
            physical_fields,
        })
    }

    pub(super) fn candidate_threadgroup_widths(&self) -> Vec<usize> {
        let execution = self.thread_execution_width();
        let maximum = self.max_threadgroup_width();
        let mut widths = vec![execution, 64, 128, 256, 512, maximum];
        widths.retain(|width| *width > 0 && *width <= maximum && width.is_multiple_of(execution));
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    pub(super) fn run_multiply(
        &self,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        self.run_pipeline(&self.multiply_pipeline, 1, dispatches, threadgroup_width)
    }

    pub(super) fn run_rotating(
        &self,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        self.run_pipeline(&self.rotating_pipeline, iterations, dispatches, threadgroup_width)
    }

    fn run_pipeline(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        iterations: u32,
        dispatches: u32,
        threadgroup_width: usize,
    ) -> Result<ProductIssueSample, DiagnosticError> {
        if iterations == 0 || dispatches == 0 {
            return Err(DiagnosticError::new("iterations and dispatches must both be nonzero"));
        }
        if threadgroup_width == 0
            || threadgroup_width > pipeline.maxTotalThreadsPerThreadgroup()
            || !threadgroup_width.is_multiple_of(pipeline.threadExecutionWidth())
        {
            return Err(DiagnosticError::new(format!(
                "invalid AMNS8 threadgroup width {threadgroup_width}"
            )));
        }
        let parameters = U32FieldParameters {
            logical_fields: u32::try_from(self.logical_fields)
                .map_err(|_| DiagnosticError::new("the AMNS8 field count exceeds the u32 ABI"))?,
            iterations,
            reserved0: 0,
            reserved1: 0,
        };

        let wall_start = Instant::now();
        let command = retained_command(&self.queue)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| DiagnosticError::new("create AMNS8 compute command encoder"))?;
        encoder.setComputePipelineState(pipeline);
        bind_u32_field_arguments(&encoder, &self.lhs, &self.rhs, &self.output, &parameters);
        let grid = MTLSize {
            width: self
                .physical_fields
                .checked_mul(8)
                .ok_or_else(|| DiagnosticError::new("AMNS8 physical thread count overflow"))?,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        for _ in 0..dispatches {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        check_completed(&command)?;
        let wall = wall_start.elapsed();
        let gpu = completed_gpu_duration(&command).ok_or_else(|| {
            DiagnosticError::new("Metal returned invalid completed-command GPU timestamps")
        })?;
        if gpu.is_zero() {
            return Err(DiagnosticError::new("Metal reported zero GPU duration"));
        }
        Ok(ProductIssueSample { gpu, wall })
    }

    pub(super) fn read_outputs(&self) -> Vec<Amns8FieldElement> {
        (0..self.logical_fields).map(|index| read_amns8_field_output(&self.output, index)).collect()
    }

    pub(super) fn padding_is_untouched(&self) -> bool {
        (self.logical_fields..self.physical_fields)
            .all(|index| read_amns8_field_output(&self.output, index) == AMNS8_OUTPUT_CANARY)
    }

    pub(super) const fn logical_fields(&self) -> usize {
        self.logical_fields
    }

    pub(super) const fn physical_fields(&self) -> usize {
        self.physical_fields
    }

    pub(super) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub(super) fn thread_execution_width(&self) -> usize {
        self.rotating_pipeline.threadExecutionWidth()
    }

    pub(super) fn max_threadgroup_width(&self) -> usize {
        self.rotating_pipeline.maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn static_threadgroup_memory_bytes(&self) -> usize {
        self.rotating_pipeline.staticThreadgroupMemoryLength()
    }
}

fn load_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
) -> Result<Pipeline, DiagnosticError> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| DiagnosticError::new(format!("missing Metal function {name}")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| DiagnosticError::new(format!("create pipeline {name}: {error:?}")))
}

fn new_shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    label: &str,
) -> Result<Buffer, DiagnosticError> {
    let buffer =
        device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| DiagnosticError::new(format!("allocate {bytes}-byte Metal buffer")))?;
    buffer.setLabel(Some(&NSString::from_str(label)));
    Ok(buffer)
}

fn retained_command(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, DiagnosticError> {
    let command =
        queue.commandBuffer().ok_or_else(|| DiagnosticError::new("create Metal command buffer"))?;
    if !command.retainedReferences() {
        return Err(DiagnosticError::new(
            "diagnostic requires a retained-reference command buffer",
        ));
    }
    Ok(command)
}

fn check_completed(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), DiagnosticError> {
    if let Some(error) = command.error() {
        return Err(DiagnosticError::new(format!("Metal command failed: {error:?}")));
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(DiagnosticError::new(format!("unexpected Metal completion status {status:?}")));
    }
    Ok(())
}

fn validate_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(device, size_of::<[u64; 8]>(), "diagnostic.abi")?;
    zero_shared_buffer(&output, size_of::<[u64; 8]>());
    let command = retained_command(queue)?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| DiagnosticError::new("create ABI command encoder"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_abi(&output);
    if actual != EXPECTED_ABI {
        return Err(DiagnosticError::new(format!(
            "Metal ABI mismatch: expected {EXPECTED_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_u32_product_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(device, size_of::<[u64; 8]>(), "u32-product.abi")?;
    zero_shared_buffer(&output, size_of::<[u64; 8]>());
    let command = retained_command(queue)?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| DiagnosticError::new("create u32 product ABI command encoder"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_abi(&output);
    if actual != EXPECTED_U32_PRODUCT_ABI {
        return Err(DiagnosticError::new(format!(
            "u32 product Metal ABI mismatch: expected {EXPECTED_U32_PRODUCT_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_u32_field_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(device, size_of::<[u64; 16]>(), "u32-field.abi")?;
    zero_shared_buffer(&output, size_of::<[u64; 16]>());
    let command = retained_command(queue)?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| DiagnosticError::new("create u32 field ABI command encoder"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_u32_field_abi(&output);
    if actual != EXPECTED_U32_FIELD_ABI {
        return Err(DiagnosticError::new(format!(
            "u32 field Metal ABI mismatch: expected {EXPECTED_U32_FIELD_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_u32_named_field_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(device, size_of::<[u64; 16]>(), "named-fios-field.abi")?;
    zero_shared_buffer(&output, size_of::<[u64; 16]>());
    let command = retained_command(queue)?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| DiagnosticError::new("create named FIOS field ABI command encoder"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_u32_field_abi(&output);
    if actual != EXPECTED_U32_NAMED_FIELD_ABI {
        return Err(DiagnosticError::new(format!(
            "named FIOS field Metal ABI mismatch: expected {EXPECTED_U32_NAMED_FIELD_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_u32_karatsuba_l4_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    variant: KaratsubaL4FieldVariant,
) -> Result<(), DiagnosticError> {
    match variant {
        KaratsubaL4FieldVariant::GeneratedV2 => validate_u32_karatsuba_l4_abi_words(
            device,
            queue,
            pipeline,
            &EXPECTED_U32_KARATSUBA_L4_ABI,
            "karatsuba-l4-field.abi",
            variant.label(),
        ),
        KaratsubaL4FieldVariant::BlockedRadix128V3 => validate_u32_karatsuba_l4_abi_words(
            device,
            queue,
            pipeline,
            &EXPECTED_U32_BLOCKED128_KARATSUBA_L4_ABI,
            "blocked128-karatsuba-l4-field.abi",
            variant.label(),
        ),
    }
}

fn validate_u32_karatsuba_l4_abi_words<const WORDS: usize>(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    expected: &[u64; WORDS],
    buffer_label: &str,
    variant_label: &str,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(device, size_of::<[u64; WORDS]>(), buffer_label)?;
    zero_shared_buffer(&output, size_of::<[u64; WORDS]>());
    let command = retained_command(queue)?;
    let encoder = command.computeCommandEncoder().ok_or_else(|| {
        DiagnosticError::new(format!("create {variant_label} field ABI command encoder"))
    })?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_u64_words(&output);
    if &actual != expected {
        return Err(DiagnosticError::new(format!(
            "{variant_label} field Metal ABI mismatch: expected {expected:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn validate_amns8_field_abi(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), DiagnosticError> {
    let output = new_shared_buffer(
        device,
        size_of::<[u64; EXPECTED_AMNS8_FIELD_ABI.len()]>(),
        "amns8-field.abi",
    )?;
    zero_shared_buffer(&output, size_of::<[u64; EXPECTED_AMNS8_FIELD_ABI.len()]>());
    let command = retained_command(queue)?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| DiagnosticError::new("create AMNS8 ABI command encoder"))?;
    encoder.setComputePipelineState(pipeline);
    bind_buffer(&encoder, &output, 0);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: EXPECTED_AMNS8_FIELD_ABI.len(), height: 1, depth: 1 },
        MTLSize { width: EXPECTED_AMNS8_FIELD_ABI.len(), height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check_completed(&command)?;
    let actual = read_u64_words::<{ EXPECTED_AMNS8_FIELD_ABI.len() }>(&output);
    if actual != EXPECTED_AMNS8_FIELD_ABI {
        return Err(DiagnosticError::new(format!(
            "AMNS8 Metal ABI mismatch: expected {EXPECTED_AMNS8_FIELD_ABI:#x?}, device reported {actual:#x?}"
        )));
    }
    Ok(())
}

fn bind_product_arguments(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    parameters: &GateParameters,
) {
    // SAFETY: the command buffer retains the two zero-offset resources through
    // synchronous completion. `GateParameters` is a fully initialized, probed
    // 16-byte repr(C) value copied by Metal during encoding.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(parameters).cast::<c_void>(),
            size_of::<GateParameters>(),
            2,
        );
    }
}

fn bind_u32_field_arguments(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    parameters: &U32FieldParameters,
) {
    // SAFETY: the retained command owns all zero-offset resources through
    // synchronous completion. The parameters are a fully initialized, probed
    // repr(C) value copied by Metal while encoding.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(lhs), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(rhs), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::from(parameters).cast::<c_void>(),
            size_of::<U32FieldParameters>(),
            3,
        );
    }
}

fn bind_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    index: usize,
) {
    // SAFETY: the retained-reference command owns this zero-offset resource
    // until synchronous completion.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
    }
}

fn copy_inputs(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[GateInput]) {
    // SAFETY: GateInput has a compile-time checked, padding-free 32-byte layout;
    // the shared allocation was sized to `values`, and is CPU-addressable.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            size_of_val(values),
        );
    }
}

fn copy_u32_product_inputs(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[U32ProductGateInput]) {
    // SAFETY: the input has a compile-time checked, padding-free 32-byte layout;
    // the shared allocation was sized to the complete slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            size_of_val(values),
        );
    }
}

fn copy_u32_field_inputs(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[U32FieldElement]) {
    // SAFETY: U32FieldElement is a compile-time checked, padding-free 48-byte
    // value; the shared allocation was sized to the full padded slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            size_of_val(values),
        );
    }
}

fn copy_amns8_field_inputs(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[Amns8FieldElement]) {
    // SAFETY: Amns8FieldElement is a compile-time checked, padding-free 64-byte
    // value; the shared allocation was sized to the full padded slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            size_of_val(values),
        );
    }
}

fn zero_shared_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, bytes: usize) {
    // SAFETY: the shared allocation is CPU-addressable for its full requested
    // extent and is not yet visible to an encoded command.
    unsafe {
        std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, bytes);
    }
}

fn read_abi(buffer: &ProtocolObject<dyn MTLBuffer>) -> [u64; 8] {
    let mut value = [0_u64; 8];
    // SAFETY: successful command completion initialized the 64-byte ABI output,
    // and the shared allocation remains alive and CPU-addressable.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<u64>(),
            value.as_mut_ptr(),
            value.len(),
        );
    }
    value
}

fn read_u32_field_abi(buffer: &ProtocolObject<dyn MTLBuffer>) -> [u64; 16] {
    let mut value = [0_u64; 16];
    // SAFETY: successful command completion initialized the 128-byte ABI
    // output, which remains alive in shared CPU-addressable storage.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<u64>(),
            value.as_mut_ptr(),
            value.len(),
        );
    }
    value
}

fn read_u64_words<const WORDS: usize>(buffer: &ProtocolObject<dyn MTLBuffer>) -> [u64; WORDS] {
    let mut value = [0_u64; WORDS];
    // SAFETY: successful command completion initialized the complete ABI output,
    // which remains alive in shared CPU-addressable storage.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<u64>(),
            value.as_mut_ptr(),
            value.len(),
        );
    }
    value
}

fn read_gate_output(buffer: &ProtocolObject<dyn MTLBuffer>, index: usize) -> GateOutput {
    let mut value = GateOutput::default();
    // SAFETY: the caller bounds-checks `index`; a completed product command
    // initialized all 128 bytes, and GateOutput is padding-free float storage.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<GateOutput>().add(index),
            &mut value,
            1,
        );
    }
    value
}

fn read_u32_product_output(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    index: usize,
) -> U32ProductGateOutput {
    let mut value = U32ProductGateOutput::default();
    // SAFETY: the harness bounds-checks the index and the completed command
    // initialized one complete padding-free output value.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<U32ProductGateOutput>().add(index),
            &mut value,
            1,
        );
    }
    value
}

fn read_u32_field_output(buffer: &ProtocolObject<dyn MTLBuffer>, index: usize) -> U32FieldElement {
    let mut value = U32FieldElement::default();
    // SAFETY: the harness allocates `physical_fields` complete elements and
    // callers restrict `index` to that range after successful completion.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<U32FieldElement>().add(index),
            &mut value,
            1,
        );
    }
    value
}

fn read_amns8_field_output(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    index: usize,
) -> Amns8FieldElement {
    let mut value = Amns8FieldElement::default();
    // SAFETY: the harness allocates `physical_fields` complete elements and
    // callers restrict `index` to that range after successful completion.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<Amns8FieldElement>().add(index),
            &mut value,
            1,
        );
    }
    value
}

fn completed_gpu_duration(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Option<Duration> {
    // SAFETY: both selectors return the ABI-equivalent f64 CFTimeInterval and
    // are queried only after successful command completion.
    let start: f64 = unsafe { objc2::msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { objc2::msg_send![command, GPUEndTime] };
    let seconds = end - start;
    if start <= 0.0 || !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let nanoseconds = seconds * 1_000_000_000.0;
    if nanoseconds > u64::MAX as f64 {
        return None;
    }
    Some(Duration::from_nanos(nanoseconds.round() as u64))
}
