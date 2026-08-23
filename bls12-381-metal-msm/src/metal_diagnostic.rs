// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::format;
use std::vec;
use std::vec::Vec;

#[allow(unsafe_code)]
mod sys;

use sys::{
    Amns8FieldHarness, KaratsubaL4ChainMode, KaratsubaL4FieldHarness, KaratsubaL4FieldVariant,
    NamedFiosFieldHarness, NamedFiosOrientation, ProductIssueHarness, ProductIssueSample,
    U32FieldHarness,
};

include!(concat!(env!("OUT_DIR"), "/bls12_381_u32_constants.rs"));

const PRODUCTS_PER_ITERATION: u64 = 32;
const PRODUCTS_PER_COMPLETE_EQUIVALENT: f64 = 1_941.0;
const U32_LOW_LIMB_PRODUCTS_PER_COMPLETE: f64 = 1_800.0;
const MEDIAN_GATE: f64 = 2_400_000_000.0;
const P5_GATE: f64 = 2_250_000_000.0;
const U32_FIELD_MEDIAN_GATE: f64 = 2_000_000_000.0;
const U32_FIELD_P5_GATE: f64 = 1_800_000_000.0;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct GateInput {
    lhs: [f32; 4],
    rhs: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct GateOutput {
    accumulators: [[f32; 4]; 8],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct U32ProductGateInput {
    lhs: [u32; 4],
    rhs: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct U32ProductGateOutput {
    accumulators: [[u32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GateParameters {
    length: u32,
    iterations: u32,
    reserved0: u32,
    reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct U32FieldElement {
    limbs: [u32; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Amns8FieldElement {
    coefficients: [i64; crate::amns8_oracle::COEFFICIENTS],
}

const KARATSUBA_L4_OUTPUT_CANARY: U32FieldElement = U32FieldElement {
    limbs: [
        0xa5a5_0000,
        0xa5a5_0001,
        0xa5a5_0002,
        0xa5a5_0003,
        0xa5a5_0004,
        0xa5a5_0005,
        0xa5a5_0006,
        0xa5a5_0007,
        0xa5a5_0008,
        0xa5a5_0009,
        0xa5a5_000a,
        0xa5a5_000b,
    ],
};

const AMNS8_OUTPUT_CANARY: Amns8FieldElement = Amns8FieldElement {
    coefficients: [
        -0x5a5a_0000_0000_0001,
        -0x5a5a_0000_0000_0002,
        -0x5a5a_0000_0000_0003,
        -0x5a5a_0000_0000_0004,
        -0x5a5a_0000_0000_0005,
        -0x5a5a_0000_0000_0006,
        -0x5a5a_0000_0000_0007,
        -0x5a5a_0000_0000_0008,
    ],
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct U32FieldParameters {
    logical_fields: u32,
    iterations: u32,
    reserved0: u32,
    reserved1: u32,
}

const _: [(); 32] = [(); core::mem::size_of::<GateInput>()];
const _: [(); 128] = [(); core::mem::size_of::<GateOutput>()];
const _: [(); 32] = [(); core::mem::size_of::<U32ProductGateInput>()];
const _: [(); 128] = [(); core::mem::size_of::<U32ProductGateOutput>()];
const _: [(); 16] = [(); core::mem::size_of::<GateParameters>()];
const _: [(); 48] = [(); core::mem::size_of::<U32FieldElement>()];
const _: [(); 64] = [(); core::mem::size_of::<Amns8FieldElement>()];
const _: [(); 16] = [(); core::mem::size_of::<U32FieldParameters>()];

fn input_for(index: usize) -> GateInput {
    let offset = (index % 17) as f32;
    let scale = 1.0 / 4096.0;
    GateInput {
        lhs: [
            (17.0 + offset) * scale,
            (29.0 + offset) * scale,
            (43.0 + offset) * scale,
            (61.0 + offset) * scale,
        ],
        rhs: [
            (7.0 + offset) * scale,
            (19.0 + offset) * scale,
            (31.0 + offset) * scale,
            (47.0 + offset) * scale,
        ],
    }
}

fn permute<T: Copy>(value: [T; 4], indices: [usize; 4]) -> [T; 4] {
    [value[indices[0]], value[indices[1]], value[indices[2]], value[indices[3]]]
}

fn add(lhs: [f32; 4], rhs: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|index| lhs[index] + rhs[index])
}

fn sub(lhs: [f32; 4], rhs: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|index| lhs[index] - rhs[index])
}

fn fma(lhs: [f32; 4], rhs: [f32; 4], addend: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|index| lhs[index].mul_add(rhs[index], addend[index]))
}

fn cpu_oracle(input: GateInput, iterations: u32) -> GateOutput {
    let lhs = input.lhs;
    let rhs = input.rhs;
    let yzwx = [1, 2, 3, 0];
    let zwxy = [2, 3, 0, 1];
    let wxyz = [3, 0, 1, 2];
    let yxwz = [1, 0, 3, 2];
    let xwzy = [0, 3, 2, 1];
    let zyxw = [2, 1, 0, 3];
    let wzyx = [3, 2, 1, 0];
    let mut accumulators = [
        add(lhs, rhs),
        sub(lhs, rhs),
        add(permute(lhs, yzwx), permute(rhs, wxyz)),
        sub(permute(lhs, zwxy), permute(rhs, yzwx)),
        add(permute(lhs, wxyz), permute(rhs, zwxy)),
        sub(permute(lhs, yxwz), permute(rhs, zwxy)),
        add(permute(lhs, xwzy), permute(rhs, yxwz)),
        sub(permute(lhs, zyxw), permute(rhs, xwzy)),
    ];

    for _ in 0..iterations {
        accumulators[0] = fma(lhs, rhs, accumulators[0]);
        accumulators[1] = fma(permute(lhs, yzwx), permute(rhs, zwxy), accumulators[1]);
        accumulators[2] = fma(permute(lhs, zwxy), permute(rhs, wxyz), accumulators[2]);
        accumulators[3] = fma(permute(lhs, wxyz), permute(rhs, yzwx), accumulators[3]);
        accumulators[4] = fma(permute(lhs, yxwz), permute(rhs, xwzy), accumulators[4]);
        accumulators[5] = fma(permute(lhs, xwzy), permute(rhs, zyxw), accumulators[5]);
        accumulators[6] = fma(permute(lhs, zyxw), permute(rhs, yxwz), accumulators[6]);
        accumulators[7] = fma(permute(lhs, wzyx), permute(rhs, wzyx), accumulators[7]);
    }
    GateOutput { accumulators }
}

fn assert_output_matches(actual: GateOutput, expected: GateOutput) {
    for accumulator in 0..8 {
        for lane in 0..4 {
            assert_eq!(
                actual.accumulators[accumulator][lane].to_bits(),
                expected.accumulators[accumulator][lane].to_bits(),
                "accumulator {accumulator} lane {lane}"
            );
        }
    }
}

fn u32_product_input_for(index: usize) -> U32ProductGateInput {
    let offset = (index % 257) as u32;
    U32ProductGateInput {
        lhs: [17 + offset, 509 + offset, 4_093 + offset, 8_191 - (offset & 255)],
        rhs: [7 + offset, 251 + offset, 2_047 + offset, 8_173 - (offset & 255)],
    }
}

fn u32_add(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
    core::array::from_fn(|index| lhs[index].wrapping_add(rhs[index]))
}

fn u32_sub(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
    core::array::from_fn(|index| lhs[index].wrapping_sub(rhs[index]))
}

fn u32_mul_add(lhs: [u32; 4], rhs: [u32; 4], addend: [u32; 4]) -> [u32; 4] {
    core::array::from_fn(|index| lhs[index].wrapping_mul(rhs[index]).wrapping_add(addend[index]))
}

fn u32_product_cpu_oracle(input: U32ProductGateInput, iterations: u32) -> U32ProductGateOutput {
    let lhs = input.lhs;
    let rhs = input.rhs;
    let yzwx = [1, 2, 3, 0];
    let zwxy = [2, 3, 0, 1];
    let wxyz = [3, 0, 1, 2];
    let yxwz = [1, 0, 3, 2];
    let xwzy = [0, 3, 2, 1];
    let zyxw = [2, 1, 0, 3];
    let wzyx = [3, 2, 1, 0];
    let mut accumulators = [
        u32_add(lhs, rhs),
        u32_sub(lhs, rhs),
        u32_add(permute(lhs, yzwx), permute(rhs, wxyz)),
        u32_sub(permute(lhs, zwxy), permute(rhs, yzwx)),
        u32_add(permute(lhs, wxyz), permute(rhs, zwxy)),
        u32_sub(permute(lhs, yxwz), permute(rhs, zwxy)),
        u32_add(permute(lhs, xwzy), permute(rhs, yxwz)),
        u32_sub(permute(lhs, zyxw), permute(rhs, xwzy)),
    ];

    for _ in 0..iterations {
        accumulators[0] = u32_mul_add(lhs, rhs, accumulators[0]);
        accumulators[1] = u32_mul_add(permute(lhs, yzwx), permute(rhs, zwxy), accumulators[1]);
        accumulators[2] = u32_mul_add(permute(lhs, zwxy), permute(rhs, wxyz), accumulators[2]);
        accumulators[3] = u32_mul_add(permute(lhs, wxyz), permute(rhs, yzwx), accumulators[3]);
        accumulators[4] = u32_mul_add(permute(lhs, yxwz), permute(rhs, xwzy), accumulators[4]);
        accumulators[5] = u32_mul_add(permute(lhs, xwzy), permute(rhs, zyxw), accumulators[5]);
        accumulators[6] = u32_mul_add(permute(lhs, zyxw), permute(rhs, yxwz), accumulators[6]);
        accumulators[7] = u32_mul_add(permute(lhs, wzyx), rhs, accumulators[7]);
    }
    U32ProductGateOutput { accumulators }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn product_rate(
    sample: ProductIssueSample,
    elements: usize,
    iterations: u32,
    dispatches: u32,
) -> f64 {
    let products = elements as f64
        * f64::from(iterations)
        * f64::from(dispatches)
        * PRODUCTS_PER_ITERATION as f64;
    products / sample.gpu.as_secs_f64()
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn nearest_rank_percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    assert!(!sorted.is_empty() && numerator > 0 && numerator <= denominator);
    let rank = (sorted.len() * numerator).div_ceil(denominator).max(1);
    sorted[rank - 1]
}

fn u32_field_is_reduced(value: U32FieldElement) -> bool {
    u32_field_less_than(value, &FP_MODULUS)
}

fn u32_field_is_l4(value: U32FieldElement) -> bool {
    u32_field_less_than(value, &U32_KARATSUBA_L4_MODULUS_X4)
}

fn u32_field_less_than(value: U32FieldElement, bound: &[u32; 12]) -> bool {
    for index in (0..12).rev() {
        if value.limbs[index] != bound[index] {
            return value.limbs[index] < bound[index];
        }
    }
    false
}

fn u32_field_subtract_modulus(value: U32FieldElement) -> (U32FieldElement, u32) {
    let mut difference = U32FieldElement::default();
    let mut borrow = 0_u32;
    for (index, &modulus) in FP_MODULUS.iter().enumerate() {
        let first = value.limbs[index].wrapping_sub(modulus);
        difference.limbs[index] = first.wrapping_sub(borrow);
        let first_borrow = u32::from(value.limbs[index] < modulus);
        let second_borrow = u32::from(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    (difference, borrow)
}

fn u32_field_reduce_l4(mut value: U32FieldElement) -> U32FieldElement {
    assert!(u32_field_is_l4(value));
    for _ in 0..3 {
        let (difference, borrow) = u32_field_subtract_modulus(value);
        let select = 0_u32.wrapping_sub(borrow ^ 1);
        value = U32FieldElement {
            limbs: core::array::from_fn(|index| {
                (value.limbs[index] & !select) | (difference.limbs[index] & select)
            }),
        };
    }
    assert!(u32_field_is_reduced(value));
    value
}

fn u32_field_add_modulus_multiple(mut value: U32FieldElement, multiple: u32) -> U32FieldElement {
    assert!(u32_field_is_reduced(value));
    assert!(multiple < U32_KARATSUBA_L4_LAZY_MULTIPLE);
    for _ in 0..multiple {
        let mut carry = 0_u64;
        for (limb, &modulus) in value.limbs.iter_mut().zip(&FP_MODULUS) {
            let sum = u64::from(*limb) + u64::from(modulus) + carry;
            *limb = sum as u32;
            carry = sum >> 32;
        }
        assert_eq!(carry, 0);
    }
    assert!(u32_field_is_l4(value));
    value
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_amns8_coefficient(state: &mut u64, bound: i64) -> i64 {
    let span = u64::try_from(2_i128 * i128::from(bound) + 1).expect("AMNS8 range fits u64");
    i64::try_from(splitmix64(state) % span).expect("AMNS8 sample fits i64") - bound
}

fn amns8_inputs(length: usize, seed: u64) -> (Vec<Amns8FieldElement>, Vec<Amns8FieldElement>) {
    let mut state = seed;
    let bound = crate::amns8_oracle::RHO - 1;
    let mut make = || Amns8FieldElement {
        coefficients: core::array::from_fn(|_| random_amns8_coefficient(&mut state, bound)),
    };
    ((0..length).map(|_| make()).collect(), (0..length).map(|_| make()).collect())
}

fn expected_amns8_multiply_outputs(
    lhs: &[Amns8FieldElement],
    rhs: &[Amns8FieldElement],
) -> Vec<Amns8FieldElement> {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| Amns8FieldElement {
            coefficients: crate::amns8_oracle::multiply(&lhs.coefficients, &rhs.coefficients),
        })
        .collect()
}

fn expected_amns8_rotating_outputs(
    lhs: &[Amns8FieldElement],
    rhs: &[Amns8FieldElement],
    iterations: u32,
) -> Vec<Amns8FieldElement> {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| Amns8FieldElement {
            coefficients: crate::amns8_oracle::rotating_chain(
                lhs.coefficients,
                rhs.coefficients,
                iterations,
            ),
        })
        .collect()
}

fn assert_amns8_outputs(
    actual: &[Amns8FieldElement],
    expected: &[Amns8FieldElement],
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: output length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{context}: coefficient vector {index}");
        assert!(
            actual
                .coefficients
                .iter()
                .all(|coefficient| coefficient.unsigned_abs() < crate::amns8_oracle::RHO as u64),
            "{context}: output {index} escaped rho"
        );
    }
}

fn fp_modulus_u64() -> [u64; 6] {
    core::array::from_fn(|index| {
        u64::from(FP_MODULUS[2 * index]) | (u64::from(FP_MODULUS[2 * index + 1]) << 32)
    })
}

fn canonical_less_than_modulus(value: &[u64; 6]) -> bool {
    let modulus = fp_modulus_u64();
    for index in (0..6).rev() {
        if value[index] != modulus[index] {
            return value[index] < modulus[index];
        }
    }
    false
}

fn random_canonical(state: &mut u64) -> [u64; 6] {
    loop {
        let value = [
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state),
            splitmix64(state) & 0x1fff_ffff_ffff_ffff,
        ];
        if canonical_less_than_modulus(&value) {
            return value;
        }
    }
}

#[allow(unsafe_code)]
mod blst_oracle {
    use super::U32FieldElement;

    fn from_blst(value: blst::blst_fp) -> U32FieldElement {
        let mut limbs = [0_u32; 12];
        for (index, word) in value.l.into_iter().enumerate() {
            limbs[2 * index] = word as u32;
            limbs[2 * index + 1] = (word >> 32) as u32;
        }
        U32FieldElement { limbs }
    }

    fn to_blst(value: U32FieldElement) -> blst::blst_fp {
        let limbs = core::array::from_fn(|index| {
            u64::from(value.limbs[2 * index]) | (u64::from(value.limbs[2 * index + 1]) << 32)
        });
        blst::blst_fp { l: limbs }
    }

    pub(super) fn from_canonical(value: [u64; 6]) -> U32FieldElement {
        let mut output = blst::blst_fp::default();
        // SAFETY: `output` is valid writable storage and `value` supplies six
        // initialized little-endian canonical limbs for the duration of BLST.
        unsafe {
            blst::blst_fp_from_uint64(&mut output, value.as_ptr());
        }
        from_blst(output)
    }

    pub(super) fn multiply(lhs: U32FieldElement, rhs: U32FieldElement) -> U32FieldElement {
        let lhs = to_blst(lhs);
        let rhs = to_blst(rhs);
        let mut output = blst::blst_fp::default();
        // SAFETY: the initialized BLST inputs are immutable and do not overlap
        // the writable output for the duration of the field multiplication.
        unsafe {
            blst::blst_fp_mul(&mut output, &lhs, &rhs);
        }
        from_blst(output)
    }

    pub(super) fn chain(
        mut value: U32FieldElement,
        multiplier: U32FieldElement,
        iterations: u32,
    ) -> U32FieldElement {
        for _ in 0..iterations {
            value = multiply(value, multiplier);
        }
        value
    }

    pub(super) fn rotating_chain(
        mut left: U32FieldElement,
        mut right: U32FieldElement,
        iterations: u32,
    ) -> U32FieldElement {
        for _ in 0..iterations {
            let product = multiply(left, right);
            left = right;
            right = product;
        }
        right
    }
}

fn u32_field_inputs(count: usize, seed: u64) -> (Vec<U32FieldElement>, Vec<U32FieldElement>) {
    let mut state = seed;
    let mut lhs = Vec::with_capacity(count);
    let mut rhs = Vec::with_capacity(count);
    let mut boundary = Vec::new();
    let modulus = fp_modulus_u64();
    let mut modulus_minus_one = modulus;
    modulus_minus_one[0] -= 1;
    let mut modulus_minus_two = modulus;
    modulus_minus_two[0] -= 2;
    boundary.extend([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
        modulus_minus_one,
        modulus_minus_two,
        [u64::MAX, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [u64::MAX, u64::MAX, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [u64::MAX, u64::MAX, u64::MAX, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555, 0, 0, 0, 0],
    ]);

    for index in 0..count {
        let canonical_lhs =
            boundary.get(index).copied().unwrap_or_else(|| random_canonical(&mut state));
        let canonical_rhs = if index < boundary.len() {
            boundary[(index.wrapping_mul(5).wrapping_add(1)) % boundary.len()]
        } else {
            random_canonical(&mut state)
        };
        lhs.push(blst_oracle::from_canonical(canonical_lhs));
        rhs.push(blst_oracle::from_canonical(canonical_rhs));
    }
    (lhs, rhs)
}

fn expected_u32_field_outputs(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
) -> Vec<U32FieldElement> {
    lhs.iter().zip(rhs).map(|(&lhs, &rhs)| blst_oracle::chain(lhs, rhs, iterations)).collect()
}

fn assert_u32_field_outputs(
    actual: &[U32FieldElement],
    expected: &[U32FieldElement],
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: output length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{context}: raw Montgomery output {index}");
        assert!(u32_field_is_reduced(actual), "{context}: output {index} is not reduced");
    }
}

fn run_u32_field_case(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
    width: usize,
    context: &str,
) {
    let expected = expected_u32_field_outputs(lhs, rhs, iterations);
    let harness = U32FieldHarness::new(lhs, rhs).expect("create standalone u32 field harness");
    assert_eq!(harness.thread_execution_width(), 32);
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    assert_eq!(harness.physical_threads() % 32, 0);
    harness.run(iterations, 1, width).unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_u32_field_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_zero(), "{context}: padding store detected");
}

const SUBTRACT_LHS: U32FieldElement = U32FieldElement {
    limbs: [
        0x29bd_4ff9,
        0xec17_8afc,
        0x0e68_a95f,
        0x8bf1_3f66,
        0x99f8_b61a,
        0x5d2f_650e,
        0xb0bd_f626,
        0xb520_b7d1,
        0xb4ef_6779,
        0xb173_a2a7,
        0xa1ac_40ac,
        0x181c_ea5c,
    ],
};
const SUBTRACT_RHS: U32FieldElement = U32FieldElement {
    limbs: [
        0xae04_26f9,
        0x6bd3_8547,
        0x3cf8_0d6b,
        0x1e02_dc62,
        0xe314_c6a1,
        0xc5b0_85b9,
        0x2c9c_458d,
        0xb303_49ef,
        0xa4ef_a85b,
        0x0033_a1f2,
        0x4c0c_573a,
        0x12ec_81db,
    ],
};
const SUBTRACT_EXPECTED: U32FieldElement = U32FieldElement {
    limbs: [
        0x1461_a559,
        0x4143_dfb0,
        0x1064_7329,
        0x905c_2918,
        0xdf48_d915,
        0x24db_9cb7,
        0xfd94_cec2,
        0x9450_6203,
        0x7c26_ed42,
        0xdb1b_25e5,
        0x3e8e_8171,
        0x0076_c216,
    ],
};

fn run_u32_field_correctness_suite() {
    assert_eq!(blst_oracle::multiply(SUBTRACT_LHS, SUBTRACT_RHS), SUBTRACT_EXPECTED);
    run_u32_field_case(&[SUBTRACT_LHS], &[SUBTRACT_RHS], 1, 32, "forced final subtraction");

    let (probe_lhs, probe_rhs) = u32_field_inputs(16, 0xa32f_0715_0000_0001);
    let probe = U32FieldHarness::new(&probe_lhs, &probe_rhs).expect("create width probe");
    for width in probe.candidate_threadgroup_widths() {
        let fields_per_group = width / 2;
        let mut lengths = vec![0, 1, 15, 16, 17];
        lengths.extend([
            fields_per_group.saturating_sub(1),
            fields_per_group,
            fields_per_group + 1,
        ]);
        lengths.sort_unstable();
        lengths.dedup();
        for length in lengths {
            let (lhs, rhs) = u32_field_inputs(length, 0x9a77_0000_0000_0000 ^ length as u64);
            run_u32_field_case(&lhs, &rhs, 1, width, &format!("width {width} length {length}"));
        }
    }

    let (lhs, rhs) = u32_field_inputs(4_113, 0xd1ff_e7a1_3810_0007);
    run_u32_field_case(&lhs, &rhs, 7, 64, "4096+ random repeated chain");
}

fn run_named_fios_case(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
    width: usize,
    orientation: NamedFiosOrientation,
    context: &str,
) {
    let expected = expected_u32_field_outputs(lhs, rhs, iterations);
    let harness =
        NamedFiosFieldHarness::new(lhs, rhs).expect("create standalone named FIOS field harness");
    let execution_width = harness.thread_execution_width(orientation);
    assert!(execution_width > 0);
    assert_eq!(harness.static_threadgroup_memory_bytes(orientation), 0);
    assert_eq!(harness.physical_fields() % execution_width, 0);
    harness
        .run(orientation, iterations, 1, width)
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_u32_field_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_zero(), "{context}: padding store detected");
}

fn run_named_fios_correctness_suite() {
    assert_eq!(blst_oracle::multiply(SUBTRACT_LHS, SUBTRACT_RHS), SUBTRACT_EXPECTED);

    let (probe_lhs, probe_rhs) = u32_field_inputs(32, 0xa32f_f105_0000_0001);
    let probe =
        NamedFiosFieldHarness::new(&probe_lhs, &probe_rhs).expect("create named FIOS width probe");
    for orientation in NamedFiosOrientation::ALL {
        let widths = probe.candidate_threadgroup_widths(orientation);
        assert!(!widths.is_empty());
        run_named_fios_case(
            &[SUBTRACT_LHS],
            &[SUBTRACT_RHS],
            1,
            widths[0],
            orientation,
            &format!("{} forced final subtraction", orientation.label()),
        );

        for width in widths {
            let mut lengths = vec![0, 1, 15, 16, 17];
            lengths.extend([width.saturating_sub(1), width, width + 1]);
            lengths.sort_unstable();
            lengths.dedup();
            for length in lengths {
                let (lhs, rhs) =
                    u32_field_inputs(length, 0xf105_0000_0000_0000 ^ length as u64 ^ width as u64);
                run_named_fios_case(
                    &lhs,
                    &rhs,
                    1,
                    width,
                    orientation,
                    &format!("{} width {width} length {length}", orientation.label()),
                );
            }
        }

        let repeated_width = probe
            .candidate_threadgroup_widths(orientation)
            .into_iter()
            .find(|&width| width == 64)
            .unwrap_or_else(|| probe.thread_execution_width(orientation));
        let (lhs, rhs) =
            u32_field_inputs(4_113, 0xd1ff_f105_3810_0007 ^ orientation.label().len() as u64);
        run_named_fios_case(
            &lhs,
            &rhs,
            7,
            repeated_width,
            orientation,
            &format!("{} 4096+ random repeated chain", orientation.label()),
        );
    }
}

fn u32_karatsuba_l4_inputs(
    count: usize,
    seed: u64,
) -> (Vec<U32FieldElement>, Vec<U32FieldElement>) {
    let (mut lhs, mut rhs) = u32_field_inputs(count, seed);
    for (index, value) in lhs.iter_mut().enumerate() {
        *value = u32_field_add_modulus_multiple(*value, (index % 4) as u32);
    }
    for (index, value) in rhs.iter_mut().enumerate() {
        *value = u32_field_add_modulus_multiple(*value, ((3 * index + 1) % 4) as u32);
    }
    (lhs, rhs)
}

fn blocked128_forced_quotient_inputs() -> (Vec<U32FieldElement>, Vec<U32FieldElement>) {
    let zero = U32FieldElement::default();
    let one = U32FieldElement { limbs: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] };
    let mut lhs = vec![zero];
    for round in 0..3 {
        let mut limbs = [0_u32; 12];
        let offset = 4 * round;
        limbs[offset..offset + 4].copy_from_slice(&FP_MODULUS[..4]);
        let value = U32FieldElement { limbs };
        assert!(u32_field_is_l4(value));
        lhs.push(value);
    }
    let rhs = vec![one; lhs.len()];
    (lhs, rhs)
}

fn expected_u32_karatsuba_l4_outputs(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
) -> Vec<U32FieldElement> {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| {
            blst_oracle::chain(u32_field_reduce_l4(lhs), u32_field_reduce_l4(rhs), iterations)
        })
        .collect()
}

fn expected_u32_blocked128_rotating_outputs(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
) -> Vec<U32FieldElement> {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| {
            blst_oracle::rotating_chain(
                u32_field_reduce_l4(lhs),
                u32_field_reduce_l4(rhs),
                iterations,
            )
        })
        .collect()
}

fn assert_u32_karatsuba_l4_outputs(
    actual: &[U32FieldElement],
    expected: &[U32FieldElement],
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: output length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(u32_field_is_l4(actual), "{context}: output {index} is not below 4p");
        assert_eq!(
            u32_field_reduce_l4(actual),
            expected,
            "{context}: reduced Montgomery output {index}"
        );
    }
}

fn run_u32_karatsuba_l4_case(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
    width: usize,
    variant: KaratsubaL4FieldVariant,
    context: &str,
) {
    let expected = expected_u32_karatsuba_l4_outputs(lhs, rhs, iterations);
    let harness = match variant {
        KaratsubaL4FieldVariant::GeneratedV2 => KaratsubaL4FieldHarness::new(lhs, rhs),
        KaratsubaL4FieldVariant::BlockedRadix128V3 => {
            KaratsubaL4FieldHarness::new_blocked128(lhs, rhs)
        }
    }
    .unwrap_or_else(|error| panic!("create standalone {} field harness: {error}", variant.label()));
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    assert_eq!(harness.physical_fields() % harness.thread_execution_width(), 0);
    harness.run(iterations, 1, width).unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_u32_karatsuba_l4_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_untouched(), "{context}: padding store detected");
}

fn run_u32_blocked128_rotating_case(
    lhs: &[U32FieldElement],
    rhs: &[U32FieldElement],
    iterations: u32,
    width: usize,
    context: &str,
) {
    let expected = expected_u32_blocked128_rotating_outputs(lhs, rhs, iterations);
    let harness = KaratsubaL4FieldHarness::new_blocked128(lhs, rhs)
        .unwrap_or_else(|error| panic!("create blocked-radix-128 rotating harness: {error}"));
    assert_eq!(
        harness.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::RotatingOperands),
        0
    );
    assert_eq!(
        harness.physical_fields()
            % harness.thread_execution_width_for(KaratsubaL4ChainMode::RotatingOperands),
        0
    );
    harness
        .run_chain(KaratsubaL4ChainMode::RotatingOperands, iterations, 1, width)
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_u32_karatsuba_l4_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_untouched(), "{context}: padding store detected");
}

fn run_u32_karatsuba_l4_correctness_suite_for_variant(variant: KaratsubaL4FieldVariant) {
    let variant_label = variant.label();
    let probe_lhs: Vec<_> = (0..16)
        .map(|index| u32_field_add_modulus_multiple(SUBTRACT_LHS, (index / 4) as u32))
        .collect();
    let probe_rhs: Vec<_> = (0..16)
        .map(|index| u32_field_add_modulus_multiple(SUBTRACT_RHS, (index % 4) as u32))
        .collect();
    let probe = match variant {
        KaratsubaL4FieldVariant::GeneratedV2 => {
            KaratsubaL4FieldHarness::new(&probe_lhs, &probe_rhs)
        }
        KaratsubaL4FieldVariant::BlockedRadix128V3 => {
            KaratsubaL4FieldHarness::new_blocked128(&probe_lhs, &probe_rhs)
        }
    }
    .unwrap_or_else(|error| panic!("create {variant_label} width probe: {error}"));
    let widths = probe.candidate_threadgroup_widths();
    assert!(!widths.is_empty());
    run_u32_karatsuba_l4_case(
        &probe_lhs,
        &probe_rhs,
        1,
        widths[0],
        variant,
        &format!("{variant_label}: all canonical-equivalent L4 multiples"),
    );

    if variant == KaratsubaL4FieldVariant::BlockedRadix128V3 {
        let (forced_lhs, forced_rhs) = blocked128_forced_quotient_inputs();
        run_u32_karatsuba_l4_case(
            &forced_lhs,
            &forced_rhs,
            1,
            widths[0],
            variant,
            "blocked-radix-128 v3: q=0 and q=2^128-1 in rounds 0..2",
        );
    }

    let mut modulus_minus_one = fp_modulus_u64();
    modulus_minus_one[0] -= 1;
    let canonical_boundary = [
        blst_oracle::from_canonical([0, 0, 0, 0, 0, 0]),
        blst_oracle::from_canonical([1, 0, 0, 0, 0, 0]),
        blst_oracle::from_canonical(modulus_minus_one),
    ];
    let mut lazy_boundary = Vec::with_capacity(12);
    for canonical in canonical_boundary {
        for multiple in 0..4 {
            lazy_boundary.push(u32_field_add_modulus_multiple(canonical, multiple));
        }
    }
    let mut boundary_lhs = Vec::with_capacity(lazy_boundary.len() * lazy_boundary.len());
    let mut boundary_rhs = Vec::with_capacity(lazy_boundary.len() * lazy_boundary.len());
    for &lhs in &lazy_boundary {
        for &rhs in &lazy_boundary {
            boundary_lhs.push(lhs);
            boundary_rhs.push(rhs);
        }
    }
    run_u32_karatsuba_l4_case(
        &boundary_lhs,
        &boundary_rhs,
        1,
        widths[0],
        variant,
        &format!("{variant_label}: Cartesian x={{0,1,p-1}} plus k*p for k=0..3"),
    );

    for width in widths {
        let mut lengths = vec![0, 1, 15, 16, 17];
        lengths.extend([width.saturating_sub(1), width, width + 1]);
        lengths.sort_unstable();
        lengths.dedup();
        for length in lengths {
            let (lhs, rhs) = u32_karatsuba_l4_inputs(
                length,
                0x4b31_324c_3400_0000 ^ length as u64 ^ width as u64,
            );
            run_u32_karatsuba_l4_case(
                &lhs,
                &rhs,
                1,
                width,
                variant,
                &format!("{variant_label}: width {width} length {length}"),
            );
        }
    }

    let repeated_width = probe
        .candidate_threadgroup_widths()
        .into_iter()
        .find(|&width| width == 64)
        .unwrap_or_else(|| probe.thread_execution_width());
    let (lhs, rhs) = u32_karatsuba_l4_inputs(4_113, 0xd1ff_4b31_3810_0007);
    run_u32_karatsuba_l4_case(
        &lhs,
        &rhs,
        7,
        repeated_width,
        variant,
        &format!("{variant_label}: 4096+ random repeated chain"),
    );
}

fn run_u32_karatsuba_l4_correctness_suite() {
    run_u32_karatsuba_l4_correctness_suite_for_variant(KaratsubaL4FieldVariant::GeneratedV2);
}

fn run_u32_blocked128_karatsuba_l4_correctness_suite() {
    run_u32_karatsuba_l4_correctness_suite_for_variant(KaratsubaL4FieldVariant::BlockedRadix128V3);
    run_u32_blocked128_rotating_correctness_suite();
}

fn run_u32_blocked128_rotating_correctness_suite() {
    let (forced_lhs, forced_rhs) = blocked128_forced_quotient_inputs();
    let probe = KaratsubaL4FieldHarness::new_blocked128(&forced_lhs, &forced_rhs)
        .expect("create blocked-radix-128 rotating width probe");
    let widths = probe.candidate_threadgroup_widths_for(KaratsubaL4ChainMode::RotatingOperands);
    assert!(!widths.is_empty());
    run_u32_blocked128_rotating_case(
        &forced_lhs,
        &forced_rhs,
        7,
        widths[0],
        "blocked-radix-128 rotating recurrence: forced quotient vectors",
    );

    let width = widths.into_iter().find(|&width| width == 64).unwrap_or_else(|| {
        probe.thread_execution_width_for(KaratsubaL4ChainMode::RotatingOperands)
    });
    let (lhs, rhs) = u32_karatsuba_l4_inputs(4_113, 0xb128_b128_3810_0007);
    run_u32_blocked128_rotating_case(
        &lhs,
        &rhs,
        11,
        width,
        "blocked-radix-128 rotating recurrence: 4096+ random inputs",
    );
}

fn u32_field_rate(
    sample: ProductIssueSample,
    logical_fields: usize,
    iterations: u32,
    dispatches: u32,
) -> (f64, f64) {
    let operations = logical_fields as f64 * f64::from(iterations) * f64::from(dispatches);
    (operations / sample.gpu.as_secs_f64(), operations / sample.wall.as_secs_f64())
}

fn run_amns8_multiply_case(
    lhs: &[Amns8FieldElement],
    rhs: &[Amns8FieldElement],
    width: usize,
    context: &str,
) {
    let expected = expected_amns8_multiply_outputs(lhs, rhs);
    let harness = Amns8FieldHarness::new(lhs, rhs)
        .unwrap_or_else(|error| panic!("create AMNS8 harness for {context}: {error}"));
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    harness
        .run_multiply(1, width)
        .unwrap_or_else(|error| panic!("AMNS8 multiply {context}: {error}"));
    assert_amns8_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_untouched(), "{context}: padding store detected");
}

fn run_amns8_rotating_case(
    lhs: &[Amns8FieldElement],
    rhs: &[Amns8FieldElement],
    iterations: u32,
    width: usize,
    context: &str,
) {
    let expected = expected_amns8_rotating_outputs(lhs, rhs, iterations);
    let harness = Amns8FieldHarness::new(lhs, rhs)
        .unwrap_or_else(|error| panic!("create AMNS8 rotating harness for {context}: {error}"));
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    harness
        .run_rotating(iterations, 1, width)
        .unwrap_or_else(|error| panic!("AMNS8 rotating {context}: {error}"));
    assert_amns8_outputs(&harness.read_outputs(), &expected, context);
    assert!(harness.padding_is_untouched(), "{context}: padding store detected");
}

fn run_amns8_correctness_suite() {
    let (probe_lhs, probe_rhs) = amns8_inputs(8, 0x414d_4e53_3857_4944);
    let probe = Amns8FieldHarness::new(&probe_lhs, &probe_rhs).expect("create AMNS8 width probe");
    assert_eq!(probe.thread_execution_width(), 32);
    assert_eq!(probe.physical_fields() % 4, 0);
    assert_eq!(probe.static_threadgroup_memory_bytes(), 0);
    let widths = probe.candidate_threadgroup_widths();
    assert!(!widths.is_empty());

    let rho = crate::amns8_oracle::RHO;
    let lazy = crate::amns8_oracle::LAZY_LIMIT;
    let boundary_values = [0, 1, -1, rho - 1, 1 - rho, lazy, -lazy];
    let mut boundary = Vec::new();
    for value in boundary_values {
        let mut single = Amns8FieldElement::default();
        single.coefficients[0] = value;
        boundary.push(single);
        boundary.push(Amns8FieldElement { coefficients: [value; 8] });
        boundary.push(Amns8FieldElement {
            coefficients: core::array::from_fn(|index| {
                if index.is_multiple_of(2) { value } else { -value }
            }),
        });
    }
    let mut boundary_lhs = Vec::new();
    let mut boundary_rhs = Vec::new();
    for lhs in &boundary {
        for rhs in &boundary {
            boundary_lhs.push(*lhs);
            boundary_rhs.push(*rhs);
        }
    }
    run_amns8_multiply_case(
        &boundary_lhs,
        &boundary_rhs,
        widths[0],
        "Cartesian coefficient boundaries",
    );

    for width in widths.iter().copied() {
        let fields_per_threadgroup = width / 8;
        let mut lengths = vec![0, 1, 3, 4, 5];
        lengths.extend([
            fields_per_threadgroup.saturating_sub(1),
            fields_per_threadgroup,
            fields_per_threadgroup + 1,
        ]);
        lengths.sort_unstable();
        lengths.dedup();
        for length in lengths {
            let (lhs, rhs) =
                amns8_inputs(length, 0x414d_4e53_384c_454e ^ length as u64 ^ width as u64);
            run_amns8_multiply_case(
                &lhs,
                &rhs,
                width,
                &format!("width {width} ragged length {length}"),
            );
        }
    }

    let width = widths.iter().copied().find(|width| *width == 64).unwrap_or(widths[0]);
    let (lhs, rhs) = amns8_inputs(4_099, 0x414d_4e53_3843_484e);
    run_amns8_rotating_case(&lhs, &rhs, 11, width, "4096+ rotating chain");
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and is a manual performance decision gate"]
fn report_precise_fp32_product_issue_gate() {
    let elements = env_usize("BLS12_381_METAL_MSM_GATE_ELEMENTS", 262_144);
    let iterations = env_usize("BLS12_381_METAL_MSM_GATE_ITERATIONS", 256) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_GATE_DISPATCHES", 4) as u32;
    let scout_samples = env_usize("BLS12_381_METAL_MSM_GATE_SCOUT_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_GATE_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_GATE_SAMPLES", 20);
    assert!(elements > 0 && iterations > 0 && dispatches > 0 && scout_samples > 0 && samples >= 5);

    let inputs: Vec<_> = (0..elements).map(input_for).collect();
    let harness = ProductIssueHarness::new(&inputs).expect("create standalone Metal harness");
    let mut best_width = 0;
    let mut best_rate = 0.0;
    for width in harness.candidate_threadgroup_widths() {
        let mut scout_rates = Vec::with_capacity(scout_samples);
        for scout_index in 0..scout_samples {
            let sample = harness
                .run(64, 1, width)
                .unwrap_or_else(|error| panic!("scout width {width}: {error}"));
            let rate = product_rate(sample, elements, 64, 1);
            std::println!(
                "metal_msm_precise_fp32_scout width={width} index={scout_index} gpu_ms={:.6} products_per_second={rate:.3}",
                sample.gpu.as_secs_f64() * 1_000.0,
            );
            scout_rates.push(rate);
        }
        scout_rates.sort_by(f64::total_cmp);
        let rate = median(&scout_rates);
        if rate > best_rate {
            best_rate = rate;
            best_width = width;
        }
    }
    assert!(best_width > 0);

    for _ in 0..warmups {
        harness.run(iterations, dispatches, best_width).expect("warm product-issue gate");
    }

    let mut rates = Vec::with_capacity(samples);
    let mut wall_seconds = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("sample {sample_index}: {error}"));
        let rate = product_rate(sample, elements, iterations, dispatches);
        std::println!(
            "metal_msm_precise_fp32_sample index={sample_index} gpu_ms={:.6} wall_ms={:.6} products_per_second={rate:.3}",
            sample.gpu.as_secs_f64() * 1_000.0,
            sample.wall.as_secs_f64() * 1_000.0,
        );
        rates.push(rate);
        wall_seconds.push(sample.wall.as_secs_f64());
    }

    let mut oracle_indices = vec![0, 1, 16, 17, elements / 2, elements - 1];
    oracle_indices.retain(|index| *index < elements);
    oracle_indices.sort_unstable();
    oracle_indices.dedup();
    for index in oracle_indices {
        let actual = harness.read_output(index).expect("read diagnostic output");
        assert_output_matches(actual, cpu_oracle(inputs[index], iterations));
    }

    rates.sort_by(f64::total_cmp);
    wall_seconds.sort_by(f64::total_cmp);
    let median_products = median(&rates);
    let p5_products = nearest_rank_percentile(&rates, 5, 100);
    let median_equivalent = median_products / PRODUCTS_PER_COMPLETE_EQUIVALENT;
    let p5_equivalent = p5_products / PRODUCTS_PER_COMPLETE_EQUIVALENT;
    let decision = if median_equivalent >= MEDIAN_GATE && p5_equivalent >= P5_GATE {
        "GO_FOR_TESTED_SCHEDULE"
    } else {
        "NO_GO_FOR_TESTED_SCHEDULE"
    };
    std::println!(
        "metal_msm_precise_fp32_gate device={:?} registry_id={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} elements={} iterations={} dispatches={} samples={} median_products_per_second={median_products:.3} p5_products_per_second={p5_products:.3} median_complete_equivalent_per_second={median_equivalent:.3} p5_complete_equivalent_per_second={p5_equivalent:.3} median_wall_ms={:.6} decision={decision}",
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        best_width,
        harness.max_threadgroup_width(),
        elements,
        iterations,
        dispatches,
        samples,
        median(&wall_seconds) * 1_000.0,
    );
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and is a manual performance decision gate"]
fn report_u32_low_limb_product_issue_gate() {
    let elements = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_ELEMENTS", 262_144);
    let iterations = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_ITERATIONS", 256) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_DISPATCHES", 4) as u32;
    let scout_samples = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_SCOUT_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_PRODUCT_SAMPLES", 20);
    assert!(elements > 0 && iterations > 0 && dispatches > 0);
    assert!(scout_samples >= 3 && warmups >= 2 && samples >= 20);

    let inputs: Vec<_> = (0..elements).map(u32_product_input_for).collect();
    let harness = ProductIssueHarness::new_u32(&inputs).expect("create u32 product harness");
    let mut best_width = 0;
    let mut best_rate = 0.0;
    for width in harness.candidate_threadgroup_widths() {
        let mut rates = Vec::with_capacity(scout_samples);
        for scout_index in 0..scout_samples {
            let sample = harness
                .run(64, 1, width)
                .unwrap_or_else(|error| panic!("u32 scout width {width}: {error}"));
            let rate = product_rate(sample, elements, 64, 1);
            std::println!(
                "metal_msm_u32_product_scout width={width} index={scout_index} gpu_ns={} raw_products_per_second={rate:.3}",
                sample.gpu.as_nanos(),
            );
            rates.push(rate);
        }
        rates.sort_by(f64::total_cmp);
        let rate = median(&rates);
        if rate > best_rate {
            best_rate = rate;
            best_width = width;
        }
    }
    assert!(best_width > 0);

    for warmup_index in 0..warmups {
        harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("u32 warmup {warmup_index}: {error}"));
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    let mut oracle_indices = vec![0, 1, 16, 17, elements / 2, elements - 1];
    oracle_indices.retain(|index| *index < elements);
    oracle_indices.sort_unstable();
    oracle_indices.dedup();
    for sample_index in 0..samples {
        let sample = harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("u32 sample {sample_index}: {error}"));
        let gpu_rate = product_rate(sample, elements, iterations, dispatches);
        let products = elements as f64
            * f64::from(iterations)
            * f64::from(dispatches)
            * PRODUCTS_PER_ITERATION as f64;
        let wall_rate = products / sample.wall.as_secs_f64();
        for &index in &oracle_indices {
            let actual = harness.read_u32_output(index).expect("read u32 diagnostic output");
            assert_eq!(
                actual,
                u32_product_cpu_oracle(inputs[index], iterations),
                "u32 product output {index} after sample {sample_index}"
            );
        }
        std::println!(
            "metal_msm_u32_product_sample index={sample_index} gpu_ns={} wall_ns={} raw_products_per_second={gpu_rate:.3} wall_products_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }

    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let median_complete = median_gpu / U32_LOW_LIMB_PRODUCTS_PER_COMPLETE;
    let p5_complete = p5_gpu / U32_LOW_LIMB_PRODUCTS_PER_COMPLETE;
    let decision = if median_complete >= U32_FIELD_MEDIAN_GATE && p5_complete >= U32_FIELD_P5_GATE {
        "GO_FOR_COMPLETE_30X13_FIELD"
    } else {
        "NO_GO_FOR_30X13_FIELD"
    };
    std::println!(
        "metal_msm_u32_product_gate candidate=radix_2^13_limbs_30 products_per_complete={} device={:?} registry_id={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} elements={} iterations={} dispatches={} samples={} median_gpu_raw_products_per_second={median_gpu:.3} p5_gpu_raw_products_per_second={p5_gpu:.3} median_wall_raw_products_per_second={median_wall:.3} p5_wall_raw_products_per_second={p5_wall:.3} median_product_only_complete_per_second={median_complete:.3} p5_product_only_complete_per_second={p5_complete:.3} required_median_raw_products_per_second={:.3} required_p5_raw_products_per_second={:.3} decision={decision}",
        U32_LOW_LIMB_PRODUCTS_PER_COMPLETE,
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        best_width,
        harness.max_threadgroup_width(),
        elements,
        iterations,
        dispatches,
        samples,
        U32_FIELD_MEDIAN_GATE * U32_LOW_LIMB_PRODUCTS_PER_COMPLETE,
        U32_FIELD_P5_GATE * U32_LOW_LIMB_PRODUCTS_PER_COMPLETE,
    );
}

#[test]
fn generated_u32_field_constants_are_consistent() {
    assert_eq!(FP_MODULUS[0].wrapping_mul(FP_INV), u32::MAX);
    assert_eq!(U32_FIELD_ALGORITHM_VERSION, 2);
    assert_eq!(U32_FIPS_SLOT_COUNT, 144);
    assert_eq!(U32_NAMED_FIELD_ALGORITHM_VERSION, 1);
    assert_eq!(U32_NAMED_FIELD_MATRIX_TERM_COUNT, 288);
    assert_eq!(U32_NAMED_FIELD_VARIANT_COUNT, 2);
    assert_eq!(U32_KARATSUBA_L4_ALGORITHM_VERSION, 2);
    assert_eq!(U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12, 63);
    assert_eq!(U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC, 189);
    assert_eq!(U32_KARATSUBA_L4_LAZY_MULTIPLE, 4);
    assert_eq!(U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT, 56);
    assert_eq!(U32_KARATSUBA_L4_DECLARED_PEAK_LIVE_LIMBS, 48);
    assert_eq!(U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS, 70);
    let flattened_source_peak =
        core::hint::black_box(U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS);
    assert!(flattened_source_peak > U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT);
    assert_eq!(U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT, 0x757d_b367_3700_e544);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION, 3);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4, 9);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC, 189);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS, 4);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_ROUNDS, 3);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS, 16);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE, 4);
    assert_eq!(
        U32_BLOCKED128_KARATSUBA_L4_ROUNDS * 7 * U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC
    );
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS * U32_BLOCKED128_KARATSUBA_L4_ROUNDS, 12);
    assert_eq!(
        U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS
    );
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS, 54);
    assert_eq!(U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS, 55);
    let blocked128_peak = core::hint::black_box(U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS);
    let blocked128_limit =
        core::hint::black_box(U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT);
    assert!(blocked128_peak <= blocked128_limit);
    assert_ne!(U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT, 0);
    assert_ne!(U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT, 0);
    let mut carry = 0_u128;
    for column in 0..12 {
        let mut sum = carry;
        for modulus_index in 0..=column {
            sum += u128::from(FP_MODULUS[modulus_index])
                * u128::from(U32_KARATSUBA_L4_NEGATIVE_INVERSE[column - modulus_index]);
        }
        assert_eq!(sum as u32, u32::MAX, "negative inverse column {column}");
        carry = sum >> 32;
    }
    let mut carry = 0_u128;
    for column in 0..4 {
        let mut sum = carry;
        for modulus_index in 0..=column {
            sum += u128::from(FP_MODULUS[modulus_index])
                * u128::from(U32_BLOCKED128_KARATSUBA_L4_NEGATIVE_INVERSE[column - modulus_index]);
        }
        assert_eq!(sum as u32, u32::MAX, "blocked inverse column {column}");
        carry = sum >> 32;
    }
    let l4_bound = U32FieldElement { limbs: U32_KARATSUBA_L4_MODULUS_X4 };
    assert!(!u32_field_is_l4(l4_bound));
    let mut l4_maximum = l4_bound;
    l4_maximum.limbs[0] -= 1;
    assert!(u32_field_is_l4(l4_maximum));
    let one = blst_oracle::from_canonical([1, 0, 0, 0, 0, 0]);
    assert!(u32_field_is_reduced(one));
    assert_eq!(blst_oracle::multiply(one, one), one);
}

#[test]
fn karatsuba_l4_harness_rejects_invalid_inputs_before_device_setup() {
    let zero = U32FieldElement::default();
    let l4_bound = U32FieldElement { limbs: U32_KARATSUBA_L4_MODULUS_X4 };
    assert!(KaratsubaL4FieldHarness::new(&[zero], &[]).is_err());
    assert!(KaratsubaL4FieldHarness::new(&[l4_bound], &[zero]).is_err());
    assert!(KaratsubaL4FieldHarness::new(&[zero], &[l4_bound]).is_err());
}

#[test]
fn blocked128_karatsuba_l4_harness_rejects_invalid_inputs_before_device_setup() {
    let zero = U32FieldElement::default();
    let l4_bound = U32FieldElement { limbs: U32_KARATSUBA_L4_MODULUS_X4 };
    assert!(KaratsubaL4FieldHarness::new_blocked128(&[zero], &[]).is_err());
    assert!(KaratsubaL4FieldHarness::new_blocked128(&[l4_bound], &[zero]).is_err());
    assert!(KaratsubaL4FieldHarness::new_blocked128(&[zero], &[l4_bound]).is_err());
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and runs deterministic two-lane field differentials"]
fn validate_two_lane_u32_field_correctness() {
    run_u32_field_correctness_suite();
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is a manual performance decision gate"]
fn report_two_lane_u32_field_gate() {
    run_u32_field_correctness_suite();

    let fields = env_usize("BLS12_381_METAL_MSM_U32_FIELD_ELEMENTS", 131_072);
    let iterations = env_usize("BLS12_381_METAL_MSM_U32_FIELD_ITERATIONS", 32) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_U32_FIELD_DISPATCHES", 4) as u32;
    let scout_iterations = env_usize("BLS12_381_METAL_MSM_U32_FIELD_SCOUT_ITERATIONS", 16) as u32;
    let scout_samples = env_usize("BLS12_381_METAL_MSM_U32_FIELD_SCOUT_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_FIELD_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_FIELD_SAMPLES", 20);
    assert!(fields > 0 && iterations > 0 && dispatches > 0 && scout_iterations > 0);
    assert!(scout_samples >= 3, "width selection requires at least three samples");
    assert!(warmups >= 2, "the decision gate requires at least two warmups");
    assert!(samples >= 20, "the decision gate requires at least twenty samples");

    let (lhs, rhs) = u32_field_inputs(fields, 0x3810_f1f5_2026_0715);
    let expected_scout = expected_u32_field_outputs(&lhs, &rhs, scout_iterations);
    let expected = expected_u32_field_outputs(&lhs, &rhs, iterations);
    let harness = U32FieldHarness::new(&lhs, &rhs).expect("create standalone u32 field harness");
    assert_eq!(harness.logical_fields(), fields);
    assert_eq!(harness.thread_execution_width(), 32);
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    assert_eq!(harness.physical_threads() % 32, 0);

    let mut best_width = 0;
    let mut best_rate = 0.0;
    for width in harness.candidate_threadgroup_widths() {
        let mut rates = Vec::with_capacity(scout_samples);
        for scout_index in 0..scout_samples {
            let sample = harness
                .run(scout_iterations, 1, width)
                .unwrap_or_else(|error| panic!("u32 field scout width {width}: {error}"));
            let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, scout_iterations, 1);
            assert_u32_field_outputs(
                &harness.read_outputs(),
                &expected_scout,
                &format!("scout width {width} sample {scout_index}"),
            );
            assert!(harness.padding_is_zero());
            std::println!(
                "metal_msm_u32_field_scout width={width} index={scout_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
                sample.gpu.as_nanos(),
                sample.wall.as_nanos(),
            );
            rates.push(gpu_rate);
        }
        rates.sort_by(f64::total_cmp);
        let rate = median(&rates);
        if rate > best_rate {
            best_rate = rate;
            best_width = width;
        }
    }
    assert!(best_width > 0);

    for warmup_index in 0..warmups {
        harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("u32 field warmup {warmup_index}: {error}"));
        assert_u32_field_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("warmup {warmup_index}"),
        );
        assert!(harness.padding_is_zero());
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("u32 field sample {sample_index}: {error}"));
        assert!(sample.gpu <= sample.wall + std::time::Duration::from_millis(1));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_field_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("decision sample {sample_index}"),
        );
        assert!(harness.padding_is_zero());
        std::println!(
            "metal_msm_u32_field_sample index={sample_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }

    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let throughput_pass = median_gpu >= U32_FIELD_MEDIAN_GATE && p5_gpu >= U32_FIELD_P5_GATE;
    let decision = if throughput_pass { "THROUGHPUT_GO_SPILL_EVIDENCE_REQUIRED" } else { "NO_GO" };
    std::println!(
        "metal_msm_u32_field_gate algorithm_version={} generated_product_slots={} device={:?} registry_id={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} physical_threads={} iterations={} dispatches={} samples={} static_threadgroup_memory_bytes={} median_gpu_complete_fp_per_second={median_gpu:.3} p5_gpu_complete_fp_per_second={p5_gpu:.3} median_wall_complete_fp_per_second={median_wall:.3} p5_wall_complete_fp_per_second={p5_wall:.3} compiler_spill_evidence=UNKNOWN decision={decision}",
        U32_FIELD_ALGORITHM_VERSION,
        U32_FIPS_SLOT_COUNT,
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        best_width,
        harness.max_threadgroup_width(),
        harness.logical_fields(),
        harness.physical_fields(),
        harness.physical_threads(),
        iterations,
        dispatches,
        samples,
        harness.static_threadgroup_memory_bytes(),
    );
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and runs deterministic named FIOS field differentials"]
fn validate_named_fios_u32_field_correctness() {
    run_named_fios_correctness_suite();
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is a manual performance decision gate"]
fn report_named_fios_u32_field_gate() {
    run_named_fios_correctness_suite();

    let fields = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_ELEMENTS", 131_072);
    let iterations = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_ITERATIONS", 32) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_DISPATCHES", 4) as u32;
    let scout_iterations =
        env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_SCOUT_ITERATIONS", 16) as u32;
    let scout_warmups = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_SCOUT_WARMUPS", 1);
    let scout_samples = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_SCOUT_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_NAMED_FIELD_SAMPLES", 20);
    assert!(fields > 0 && iterations > 0 && dispatches > 0 && scout_iterations > 0);
    assert!(scout_warmups >= 1, "every width scout must be warmed");
    assert!(scout_samples >= 3, "variant and width selection requires three samples");
    assert!(warmups >= 2, "the decision gate requires at least two warmups");
    assert!(samples >= 20, "the decision gate requires at least twenty samples");

    let (lhs, rhs) = u32_field_inputs(fields, 0x3810_f105_2026_0715);
    let expected_scout = expected_u32_field_outputs(&lhs, &rhs, scout_iterations);
    let expected = expected_u32_field_outputs(&lhs, &rhs, iterations);
    let harness =
        NamedFiosFieldHarness::new(&lhs, &rhs).expect("create named FIOS decision harness");
    assert_eq!(harness.logical_fields(), fields);

    let mut best_orientation = None;
    let mut best_width = 0;
    let mut best_rate = 0.0;
    for orientation in NamedFiosOrientation::ALL {
        let execution_width = harness.thread_execution_width(orientation);
        assert!(execution_width > 0);
        assert_eq!(harness.physical_fields() % execution_width, 0);
        assert_eq!(harness.static_threadgroup_memory_bytes(orientation), 0);
        for width in harness.candidate_threadgroup_widths(orientation) {
            for warmup_index in 0..scout_warmups {
                let sample = harness
                    .run(orientation, scout_iterations, 1, width)
                    .unwrap_or_else(|error| {
                        panic!(
                            "named FIOS scout warmup {} width {width} index {warmup_index}: {error}",
                            orientation.label()
                        )
                    });
                let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, scout_iterations, 1);
                assert_u32_field_outputs(
                    &harness.read_outputs(),
                    &expected_scout,
                    &format!(
                        "{} scout warmup width {width} index {warmup_index}",
                        orientation.label()
                    ),
                );
                assert!(harness.padding_is_zero());
                std::println!(
                    "metal_msm_u32_named_field_scout_warmup orientation={} width={width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
                    orientation.label(),
                    sample.gpu.as_nanos(),
                    sample.wall.as_nanos(),
                );
            }

            let mut rates = Vec::with_capacity(scout_samples);
            for scout_index in 0..scout_samples {
                let sample =
                    harness.run(orientation, scout_iterations, 1, width).unwrap_or_else(|error| {
                        panic!(
                            "named FIOS scout {} width {width} index {scout_index}: {error}",
                            orientation.label()
                        )
                    });
                let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, scout_iterations, 1);
                assert_u32_field_outputs(
                    &harness.read_outputs(),
                    &expected_scout,
                    &format!("{} scout width {width} sample {scout_index}", orientation.label()),
                );
                assert!(harness.padding_is_zero());
                std::println!(
                    "metal_msm_u32_named_field_scout orientation={} width={width} index={scout_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
                    orientation.label(),
                    sample.gpu.as_nanos(),
                    sample.wall.as_nanos(),
                );
                rates.push(gpu_rate);
            }
            rates.sort_by(f64::total_cmp);
            let rate = median(&rates);
            if rate > best_rate {
                best_rate = rate;
                best_width = width;
                best_orientation = Some(orientation);
            }
        }
    }
    let best_orientation = best_orientation.expect("at least one named FIOS candidate");
    assert!(best_width > 0);

    for warmup_index in 0..warmups {
        let sample = harness
            .run(best_orientation, iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("named FIOS decision warmup {warmup_index}: {error}"));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_field_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("named FIOS decision warmup {warmup_index}"),
        );
        assert!(harness.padding_is_zero());
        std::println!(
            "metal_msm_u32_named_field_warmup orientation={} width={best_width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            best_orientation.label(),
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run(best_orientation, iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("named FIOS decision sample {sample_index}: {error}"));
        assert!(sample.gpu <= sample.wall + std::time::Duration::from_millis(1));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_field_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("named FIOS decision sample {sample_index}"),
        );
        assert!(harness.padding_is_zero());
        std::println!(
            "metal_msm_u32_named_field_sample orientation={} index={sample_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            best_orientation.label(),
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }

    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let throughput_pass = median_gpu >= U32_FIELD_MEDIAN_GATE && p5_gpu >= U32_FIELD_P5_GATE;
    let decision = if throughput_pass { "THROUGHPUT_GO_SPILL_EVIDENCE_REQUIRED" } else { "NO_GO" };
    let complete_operations_per_sample = u64::try_from(fields)
        .expect("field count fits u64")
        .checked_mul(u64::from(iterations))
        .and_then(|operations| operations.checked_mul(u64::from(dispatches)))
        .expect("complete-operation numerator fits u64");
    std::println!(
        "metal_msm_u32_named_field_gate algorithm_version={} matrix_terms={} variant_count={} selected_orientation={} device={:?} registry_id={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} threads_per_field=1 iterations={} dispatches={} complete_operations_per_sample={complete_operations_per_sample} scout_iterations={} scout_warmups_per_orientation_width={} scout_samples_per_orientation_width={} decision_warmups={} decision_samples={} static_threadgroup_memory_bytes={} median_gpu_complete_fp_per_second={median_gpu:.3} p5_gpu_complete_fp_per_second={p5_gpu:.3} median_wall_complete_fp_per_second={median_wall:.3} p5_wall_complete_fp_per_second={p5_wall:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3} compiler_spill_evidence=UNKNOWN decision={decision}",
        U32_NAMED_FIELD_ALGORITHM_VERSION,
        U32_NAMED_FIELD_MATRIX_TERM_COUNT,
        U32_NAMED_FIELD_VARIANT_COUNT,
        best_orientation.label(),
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(best_orientation),
        best_width,
        harness.max_threadgroup_width(best_orientation),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        dispatches,
        scout_iterations,
        scout_warmups,
        scout_samples,
        warmups,
        samples,
        harness.static_threadgroup_memory_bytes(best_orientation),
    );
}

struct KaratsubaL4ScoutSettings<'a> {
    iterations: u32,
    warmups_per_width: usize,
    samples_per_width: usize,
    label: &'a str,
}

fn require_u32_karatsuba_l4_source_liveness_gate() {
    // Keep this as a runtime guard so the ignored scout and decision tests
    // remain executable diagnostics while failing closed for this schedule.
    let flattened_source_peak =
        core::hint::black_box(U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS);
    assert!(
        flattened_source_peak <= U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
        "Karatsuba-L4 pre-scout liveness gate failed: flattened source peak {} exceeds limit {}",
        flattened_source_peak,
        U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
    );
}

fn require_u32_blocked128_karatsuba_l4_source_liveness_gate() {
    let source_peak = core::hint::black_box(U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS);
    let source_limit = core::hint::black_box(U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT);
    assert_eq!(source_peak, 55, "the reviewed v3 schedule must retain its 55-word peak");
    assert_eq!(source_limit, 56, "the reviewed v3 source-live limit must remain 56 words");
    assert!(
        source_peak <= source_limit,
        "blocked-radix-128 v3 source liveness gate failed: {source_peak} > {source_limit}"
    );
}

fn select_u32_karatsuba_l4_width(
    harness: &KaratsubaL4FieldHarness,
    expected: &[U32FieldElement],
    chain_mode: KaratsubaL4ChainMode,
    settings: KaratsubaL4ScoutSettings<'_>,
) -> (usize, f64) {
    assert!(settings.iterations > 0);
    assert!(settings.warmups_per_width >= 1);
    assert!(settings.samples_per_width >= 3);
    let fields = harness.logical_fields();
    let mut best_width = 0_usize;
    let mut best_rate = 0.0_f64;
    for width in harness.candidate_threadgroup_widths_for(chain_mode) {
        for warmup_index in 0..settings.warmups_per_width {
            let sample = harness
                .run_chain(chain_mode, settings.iterations, 1, width)
                .unwrap_or_else(|error| {
                    panic!(
                        "Karatsuba-L4 {} {} width {width} warmup {warmup_index}: {error}",
                        chain_mode.label(),
                        settings.label
                    )
                });
            let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, settings.iterations, 1);
            assert_u32_karatsuba_l4_outputs(
                &harness.read_outputs(),
                expected,
                &format!("{} width {width} warmup {warmup_index}", settings.label),
            );
            assert!(harness.padding_is_untouched());
            std::println!(
                "metal_msm_u32_karatsuba_l4_scout_warmup chain_mode={} label={} width={width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_redc_per_second={gpu_rate:.3} wall_complete_redc_per_second={wall_rate:.3}",
                chain_mode.label(),
                settings.label,
                sample.gpu.as_nanos(),
                sample.wall.as_nanos(),
            );
        }

        let mut rates = Vec::with_capacity(settings.samples_per_width);
        for sample_index in 0..settings.samples_per_width {
            let sample = harness
                .run_chain(chain_mode, settings.iterations, 1, width)
                .unwrap_or_else(|error| {
                    panic!(
                        "Karatsuba-L4 {} {} width {width} sample {sample_index}: {error}",
                        chain_mode.label(),
                        settings.label
                    )
                });
            let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, settings.iterations, 1);
            assert_u32_karatsuba_l4_outputs(
                &harness.read_outputs(),
                expected,
                &format!("{} width {width} sample {sample_index}", settings.label),
            );
            assert!(harness.padding_is_untouched());
            std::println!(
                "metal_msm_u32_karatsuba_l4_scout chain_mode={} label={} width={width} index={sample_index} gpu_ns={} wall_ns={} gpu_complete_redc_per_second={gpu_rate:.3} wall_complete_redc_per_second={wall_rate:.3}",
                chain_mode.label(),
                settings.label,
                sample.gpu.as_nanos(),
                sample.wall.as_nanos(),
            );
            rates.push(gpu_rate);
        }
        rates.sort_by(f64::total_cmp);
        let rate = median(&rates);
        if rate > best_rate {
            best_rate = rate;
            best_width = width;
        }
    }
    assert!(best_width > 0);
    (best_width, best_rate)
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and runs deterministic Karatsuba-L4 differentials"]
fn validate_named_karatsuba_l4_u32_field_correctness() {
    run_u32_karatsuba_l4_correctness_suite();
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and runs deterministic blocked-radix-128 Karatsuba-L4 v3 differentials"]
fn validate_named_blocked128_karatsuba_l4_v3_u32_field_correctness() {
    run_u32_blocked128_karatsuba_l4_correctness_suite();
}

#[test]
#[ignore = "requires an Apple-silicon Metal device and checks the v3 rotating-operands BLST recurrence"]
fn validate_named_blocked128_karatsuba_l4_v3_rotating_recurrence() {
    run_u32_blocked128_rotating_correctness_suite();
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is a short non-decision scout"]
fn report_named_karatsuba_l4_u32_field_scout() {
    require_u32_karatsuba_l4_source_liveness_gate();
    let fields = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_SCOUT_ELEMENTS", 131_072);
    let iterations = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_SCOUT_ITERATIONS", 8) as u32;
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_SCOUT_WARMUPS", 1);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_SCOUT_SAMPLES", 3);
    assert!(fields > 0 && iterations > 0 && warmups >= 1 && samples >= 3);

    let (lhs, rhs) = u32_field_inputs(fields, 0x4b31_324c_3453_434f);
    let expected = expected_u32_karatsuba_l4_outputs(&lhs, &rhs, iterations);
    let harness =
        KaratsubaL4FieldHarness::new(&lhs, &rhs).expect("create Karatsuba-L4 scout harness");
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    assert_eq!(harness.physical_fields() % harness.thread_execution_width(), 0);
    let (best_width, best_rate) = select_u32_karatsuba_l4_width(
        &harness,
        &expected,
        KaratsubaL4ChainMode::FixedRhs,
        KaratsubaL4ScoutSettings {
            iterations,
            warmups_per_width: warmups,
            samples_per_width: samples,
            label: "short_scout",
        },
    );
    std::println!(
        "metal_msm_u32_karatsuba_l4_scout_summary algorithm_version={} modeled_base_products_per_k12={} modeled_base_products_per_redc={} top_level_peak_live_limbs={} flattened_source_peak_live_limbs={} source_live_limb_limit={} fixed_rhs_hoisting_makes_result_optimistic=true device={:?} registry_id={} thread_execution_width={} selected_threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} iterations={} warmups_per_width={} samples_per_width={} best_median_gpu_complete_fp_per_second={best_rate:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3} optimized_air_required_before_interpretation=true decision=SCOUT_ONLY_NOT_A_GATE",
        U32_KARATSUBA_L4_ALGORITHM_VERSION,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_KARATSUBA_L4_DECLARED_PEAK_LIVE_LIMBS,
        U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS,
        U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        best_width,
        harness.max_threadgroup_width(),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        warmups,
        samples,
    );
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is a manual performance decision gate"]
fn report_named_karatsuba_l4_u32_field_gate() {
    require_u32_karatsuba_l4_source_liveness_gate();
    run_u32_karatsuba_l4_correctness_suite();

    let fields = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_ELEMENTS", 131_072);
    let iterations = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_ITERATIONS", 32) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_DISPATCHES", 4) as u32;
    let scout_iterations =
        env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_WIDTH_ITERATIONS", 16) as u32;
    let scout_warmups = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_WIDTH_WARMUPS", 1);
    let scout_samples = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_WIDTH_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_KARATSUBA_L4_SAMPLES", 20);
    assert!(fields > 0 && iterations > 0 && dispatches > 0 && scout_iterations > 0);
    assert!(scout_warmups >= 1 && scout_samples >= 3);
    assert!(warmups >= 2, "the decision gate requires at least two warmups");
    assert!(samples >= 20, "the decision gate requires at least twenty samples");

    let (lhs, rhs) = u32_field_inputs(fields, 0x4b31_324c_3420_2607);
    let expected_scout = expected_u32_karatsuba_l4_outputs(&lhs, &rhs, scout_iterations);
    let expected = expected_u32_karatsuba_l4_outputs(&lhs, &rhs, iterations);
    let harness =
        KaratsubaL4FieldHarness::new(&lhs, &rhs).expect("create Karatsuba-L4 decision harness");
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);
    let (best_width, _) = select_u32_karatsuba_l4_width(
        &harness,
        &expected_scout,
        KaratsubaL4ChainMode::FixedRhs,
        KaratsubaL4ScoutSettings {
            iterations: scout_iterations,
            warmups_per_width: scout_warmups,
            samples_per_width: scout_samples,
            label: "decision_width",
        },
    );

    for warmup_index in 0..warmups {
        let sample = harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("Karatsuba-L4 decision warmup {warmup_index}: {error}"));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_karatsuba_l4_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("Karatsuba-L4 decision warmup {warmup_index}"),
        );
        assert!(harness.padding_is_untouched());
        std::println!(
            "metal_msm_u32_karatsuba_l4_warmup width={best_width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("Karatsuba-L4 sample {sample_index}: {error}"));
        assert!(sample.gpu <= sample.wall + std::time::Duration::from_millis(1));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_karatsuba_l4_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("Karatsuba-L4 sample {sample_index}"),
        );
        assert!(harness.padding_is_untouched());
        std::println!(
            "metal_msm_u32_karatsuba_l4_sample index={sample_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }

    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let throughput_pass = median_gpu >= U32_FIELD_MEDIAN_GATE && p5_gpu >= U32_FIELD_P5_GATE;
    let decision = if throughput_pass {
        "THROUGHPUT_GO_ONLY_IF_OPTIMIZED_AIR_IS_REGISTER_RESIDENT"
    } else {
        "NO_GO"
    };
    let complete_operations_per_sample = u64::try_from(fields)
        .expect("field count fits u64")
        .checked_mul(u64::from(iterations))
        .and_then(|operations| operations.checked_mul(u64::from(dispatches)))
        .expect("complete-operation numerator fits u64");
    std::println!(
        "metal_msm_u32_karatsuba_l4_gate algorithm_version={} modeled_base_products_per_k12={} modeled_base_products_per_redc={} lazy_input_multiple={} final_canonical_reductions=0 top_level_peak_live_limbs={} flattened_source_peak_live_limbs={} source_live_limb_limit={} fixed_rhs_hoisting_makes_result_optimistic=true varying_operand_stream_gate_required_after_any_pass=true device={:?} registry_id={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} threads_per_field=1 iterations={} dispatches={} complete_operations_per_sample={complete_operations_per_sample} scout_iterations={} scout_warmups_per_width={} scout_samples_per_width={} decision_warmups={} decision_samples={} static_threadgroup_memory_bytes={} median_gpu_complete_fp_per_second={median_gpu:.3} p5_gpu_complete_fp_per_second={p5_gpu:.3} median_wall_complete_fp_per_second={median_wall:.3} p5_wall_complete_fp_per_second={p5_wall:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3} optimized_air_required=true decision={decision}",
        U32_KARATSUBA_L4_ALGORITHM_VERSION,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_KARATSUBA_L4_LAZY_MULTIPLE,
        U32_KARATSUBA_L4_DECLARED_PEAK_LIVE_LIMBS,
        U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS,
        U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        best_width,
        harness.max_threadgroup_width(),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        dispatches,
        scout_iterations,
        scout_warmups,
        scout_samples,
        warmups,
        samples,
        harness.static_threadgroup_memory_bytes(),
    );
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is an optimistic fixed-RHS v3 scout only"]
fn report_named_blocked128_karatsuba_l4_v3_fixed_rhs_scout() {
    require_u32_blocked128_karatsuba_l4_source_liveness_gate();
    let fields =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_FIXED_SCOUT_ELEMENTS", 65_536);
    let iterations =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_FIXED_SCOUT_ITERATIONS", 4)
            as u32;
    let warmups =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_FIXED_SCOUT_WARMUPS", 1);
    let samples =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_FIXED_SCOUT_SAMPLES", 3);
    assert!(fields > 0 && iterations > 0 && warmups >= 1 && samples >= 3);

    let (lhs, rhs) = u32_karatsuba_l4_inputs(fields, 0xb128_b128_5343_4f55);
    let expected = expected_u32_karatsuba_l4_outputs(&lhs, &rhs, iterations);
    let harness = KaratsubaL4FieldHarness::new_blocked128(&lhs, &rhs)
        .expect("create blocked-radix-128 v3 fixed-RHS scout harness");
    assert_eq!(harness.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::FixedRhs), 0);
    assert_eq!(
        harness.physical_fields()
            % harness.thread_execution_width_for(KaratsubaL4ChainMode::FixedRhs),
        0
    );
    let (best_width, best_rate) = select_u32_karatsuba_l4_width(
        &harness,
        &expected,
        KaratsubaL4ChainMode::FixedRhs,
        KaratsubaL4ScoutSettings {
            iterations,
            warmups_per_width: warmups,
            samples_per_width: samples,
            label: "blocked128_v3_fixed_rhs_short",
        },
    );
    std::println!(
        "metal_msm_u32_blocked128_karatsuba_l4_v3_fixed_rhs_scout algorithm_version={} abi_fingerprint={:#018x} schedule_fingerprint={:#018x} base_products_per_k4={} base_products_per_redc={} block_limbs={} rounds={} accumulator_limbs={} lazy_input_multiple={} typed_peak_live_words={} tokenized_peak_live_words={} source_peak_live_words={} source_live_word_limit={} chain_mode={} fixed_rhs_hoisting_makes_result_optimistic=true decision=SCOUT_ONLY_NOT_A_GATE device={:?} registry_id={} thread_execution_width={} selected_threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} iterations={} dispatches_per_sample=1 warmups_per_width={} samples_per_width={} static_threadgroup_memory_bytes={} best_median_gpu_complete_redc_per_second={best_rate:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3}",
        U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION,
        U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT,
        U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_ROUNDS,
        U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE,
        U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT,
        KaratsubaL4ChainMode::FixedRhs.label(),
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width_for(KaratsubaL4ChainMode::FixedRhs),
        best_width,
        harness.max_threadgroup_width_for(KaratsubaL4ChainMode::FixedRhs),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        warmups,
        samples,
        harness.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::FixedRhs),
    );
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is the 20-sample rotating-operands v3 decision gate"]
fn report_named_blocked128_karatsuba_l4_v3_rotating_field_gate() {
    require_u32_blocked128_karatsuba_l4_source_liveness_gate();
    // The complete fixed-RHS plus rotating BLST differential suite runs before
    // any performance sample is collected.
    run_u32_blocked128_karatsuba_l4_correctness_suite();

    let fields =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_ELEMENTS", 131_072);
    let iterations =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_ITERATIONS", 32) as u32;
    let dispatches =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_DISPATCHES", 4) as u32;
    let scout_iterations =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_WIDTH_ITERATIONS", 8)
            as u32;
    let scout_warmups =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_WIDTH_WARMUPS", 1);
    let scout_samples =
        env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_WIDTH_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_U32_BLOCKED128_KARATSUBA_L4_ROTATING_SAMPLES", 20);
    assert!(fields > 0 && iterations > 0 && dispatches > 0 && scout_iterations > 0);
    assert!(scout_warmups >= 1 && scout_samples >= 3);
    assert!(warmups >= 2, "the v3 decision gate requires at least two warmups");
    assert_eq!(samples, 20, "the v3 decision gate is frozen at twenty samples");

    let (lhs, rhs) = u32_karatsuba_l4_inputs(fields, 0xb128_b128_4741_5445);
    let expected_scout = expected_u32_blocked128_rotating_outputs(&lhs, &rhs, scout_iterations);
    let expected = expected_u32_blocked128_rotating_outputs(&lhs, &rhs, iterations);
    let harness = KaratsubaL4FieldHarness::new_blocked128(&lhs, &rhs)
        .expect("create blocked-radix-128 v3 rotating decision harness");
    assert_eq!(
        harness.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::RotatingOperands),
        0
    );
    assert_eq!(
        harness.physical_fields()
            % harness.thread_execution_width_for(KaratsubaL4ChainMode::RotatingOperands),
        0
    );
    std::println!(
        "metal_msm_u32_blocked128_karatsuba_l4_v3_rotating_gate_settings algorithm_version={} abi_fingerprint={:#018x} schedule_fingerprint={:#018x} base_products_per_k4={} base_products_per_redc={} block_limbs={} rounds={} accumulator_limbs={} lazy_input_multiple={} typed_peak_live_words={} tokenized_peak_live_words={} source_peak_live_words={} source_live_word_limit={} chain_mode={} fixed_rhs_hoisting=false recurrence=left_next_right_and_right_next_product device={:?} registry_id={} logical_fields={} physical_fields={} iterations={} dispatches={} rotating_width_iterations={} rotating_width_warmups={} rotating_width_samples={} decision_warmups={} decision_samples={} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3}",
        U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION,
        U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT,
        U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_ROUNDS,
        U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE,
        U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT,
        KaratsubaL4ChainMode::RotatingOperands.label(),
        harness.device_name(),
        harness.registry_id(),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        dispatches,
        scout_iterations,
        scout_warmups,
        scout_samples,
        warmups,
        samples,
    );

    let (best_width, _) = select_u32_karatsuba_l4_width(
        &harness,
        &expected_scout,
        KaratsubaL4ChainMode::RotatingOperands,
        KaratsubaL4ScoutSettings {
            iterations: scout_iterations,
            warmups_per_width: scout_warmups,
            samples_per_width: scout_samples,
            label: "blocked128_v3_rotating_decision_width",
        },
    );

    for warmup_index in 0..warmups {
        let sample = harness
            .run_chain(KaratsubaL4ChainMode::RotatingOperands, iterations, dispatches, best_width)
            .unwrap_or_else(|error| {
                panic!("blocked-radix-128 v3 rotating warmup {warmup_index}: {error}")
            });
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_karatsuba_l4_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("blocked-radix-128 v3 rotating warmup {warmup_index}"),
        );
        assert!(harness.padding_is_untouched());
        std::println!(
            "metal_msm_u32_blocked128_karatsuba_l4_v3_rotating_warmup width={best_width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_redc_per_second={gpu_rate:.3} wall_complete_redc_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run_chain(KaratsubaL4ChainMode::RotatingOperands, iterations, dispatches, best_width)
            .unwrap_or_else(|error| {
                panic!("blocked-radix-128 v3 rotating sample {sample_index}: {error}")
            });
        assert!(sample.gpu <= sample.wall + std::time::Duration::from_millis(1));
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        assert_u32_karatsuba_l4_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("blocked-radix-128 v3 rotating sample {sample_index}"),
        );
        assert!(harness.padding_is_untouched());
        std::println!(
            "metal_msm_u32_blocked128_karatsuba_l4_v3_rotating_sample width={best_width} index={sample_index} gpu_ns={} wall_ns={} gpu_complete_redc_per_second={gpu_rate:.3} wall_complete_redc_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }

    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let throughput_pass = median_gpu >= U32_FIELD_MEDIAN_GATE && p5_gpu >= U32_FIELD_P5_GATE;
    let decision = if throughput_pass { "GO" } else { "NO_GO" };
    let complete_operations_per_sample = u64::try_from(fields)
        .expect("field count fits u64")
        .checked_mul(u64::from(iterations))
        .and_then(|operations| operations.checked_mul(u64::from(dispatches)))
        .expect("complete REDC numerator fits u64");
    std::println!(
        "metal_msm_u32_blocked128_karatsuba_l4_v3_rotating_gate_result algorithm_version={} abi_fingerprint={:#018x} schedule_fingerprint={:#018x} chain_mode={} thread_execution_width={} threadgroup_width={} max_threadgroup_width={} logical_fields={} physical_fields={} threads_per_field=1 iterations={} dispatches={} complete_redcs_per_sample={complete_operations_per_sample} decision_warmups={} decision_samples={} one_command_and_one_wait_per_sample=true timed_allocation=false timed_upload=false timed_readback=false output_checked_after_every_sample=true padding_untouched=true static_threadgroup_memory_bytes={} median_gpu_complete_redc_per_second={median_gpu:.3} p5_gpu_complete_redc_per_second={p5_gpu:.3} median_wall_complete_redc_per_second={median_wall:.3} p5_wall_complete_redc_per_second={p5_wall:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3} decision={decision}",
        U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION,
        U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT,
        U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT,
        KaratsubaL4ChainMode::RotatingOperands.label(),
        harness.thread_execution_width_for(KaratsubaL4ChainMode::RotatingOperands),
        best_width,
        harness.max_threadgroup_width_for(KaratsubaL4ChainMode::RotatingOperands),
        harness.logical_fields(),
        harness.physical_fields(),
        iterations,
        dispatches,
        warmups,
        samples,
        harness.static_threadgroup_memory_bytes_for(KaratsubaL4ChainMode::RotatingOperands),
    );
    assert!(
        throughput_pass,
        "blocked-radix-128 v3 rotating field gate failed: median {median_gpu:.3} < {U32_FIELD_MEDIAN_GATE:.3} or nearest-rank p5 {p5_gpu:.3} < {U32_FIELD_P5_GATE:.3} complete REDCs/s"
    );
}

#[test]
fn amns8_harness_rejects_invalid_inputs_before_device_setup() {
    let zero = Amns8FieldElement::default();
    let outside = Amns8FieldElement {
        coefficients: [crate::amns8_oracle::LAZY_LIMIT + 1, 0, 0, 0, 0, 0, 0, 0],
    };
    assert!(Amns8FieldHarness::new(&[zero], &[]).is_err());
    assert!(Amns8FieldHarness::new(&[outside], &[zero]).is_err());
    assert!(Amns8FieldHarness::new(&[zero], &[outside]).is_err());
}

#[test]
#[ignore = "requires Apple-silicon Metal and runs the generated AMNS8 differential corpus"]
fn validate_amns8_field_correctness() {
    run_amns8_correctness_suite();
}

#[test]
#[ignore = "requires an otherwise idle Apple M1 Max and is the manual AMNS8 field decision gate"]
fn report_amns8_field_gate() {
    run_amns8_correctness_suite();

    let fields = env_usize("BLS12_381_METAL_MSM_AMNS8_FIELDS", 131_072);
    let iterations = env_usize("BLS12_381_METAL_MSM_AMNS8_ITERATIONS", 32) as u32;
    let dispatches = env_usize("BLS12_381_METAL_MSM_AMNS8_DISPATCHES", 4) as u32;
    let scout_iterations = env_usize("BLS12_381_METAL_MSM_AMNS8_SCOUT_ITERATIONS", 8) as u32;
    let scout_warmups = env_usize("BLS12_381_METAL_MSM_AMNS8_SCOUT_WARMUPS", 1);
    let scout_samples = env_usize("BLS12_381_METAL_MSM_AMNS8_SCOUT_SAMPLES", 3);
    let warmups = env_usize("BLS12_381_METAL_MSM_AMNS8_WARMUPS", 2);
    let samples = env_usize("BLS12_381_METAL_MSM_AMNS8_SAMPLES", 20);
    assert!(fields > 0 && iterations > 0 && dispatches > 0 && scout_iterations > 0);
    assert!(scout_warmups >= 1);
    assert!(scout_samples >= 3);
    assert!(warmups >= 2);
    assert!(samples >= 20);
    assert_eq!(AMNS8_TYPED_PEAK_LIVE_WORDS, AMNS8_TOKENIZED_PEAK_LIVE_WORDS);
    assert_eq!(AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS, AMNS8_TOKENIZED_ALLOCATION_PEAK_LIVE_WORDS);
    const {
        assert!(AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS <= AMNS8_SOURCE_LIVE_WORD_LIMIT);
    }

    let (lhs, rhs) = amns8_inputs(fields, 0x414d_4e53_3847_4154);
    let expected_scout = expected_amns8_rotating_outputs(&lhs, &rhs, scout_iterations);
    let expected = expected_amns8_rotating_outputs(&lhs, &rhs, iterations);
    let harness = Amns8FieldHarness::new(&lhs, &rhs).expect("create AMNS8 decision harness");
    assert_eq!(harness.thread_execution_width(), 32);
    assert_eq!(harness.physical_fields() % 4, 0);
    assert_eq!(harness.static_threadgroup_memory_bytes(), 0);

    let mut best_width = 0;
    let mut best_rate = 0.0;
    for width in harness.candidate_threadgroup_widths() {
        for warmup_index in 0..scout_warmups {
            let sample = harness.run_rotating(scout_iterations, 1, width).unwrap_or_else(|error| {
                panic!("AMNS8 scout warmup width {width} index {warmup_index}: {error}")
            });
            assert_amns8_outputs(
                &harness.read_outputs(),
                &expected_scout,
                &format!("AMNS8 scout warmup width {width} index {warmup_index}"),
            );
            assert!(harness.padding_is_untouched());
            let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, scout_iterations, 1);
            std::println!(
                "metal_msm_amns8_scout_warmup width={width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
                sample.gpu.as_nanos(),
                sample.wall.as_nanos(),
            );
        }
        let mut rates = Vec::with_capacity(scout_samples);
        for sample_index in 0..scout_samples {
            let sample = harness.run_rotating(scout_iterations, 1, width).unwrap_or_else(|error| {
                panic!("AMNS8 scout width {width} index {sample_index}: {error}")
            });
            assert_amns8_outputs(
                &harness.read_outputs(),
                &expected_scout,
                &format!("AMNS8 scout width {width} index {sample_index}"),
            );
            assert!(harness.padding_is_untouched());
            let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, scout_iterations, 1);
            std::println!(
                "metal_msm_amns8_scout width={width} index={sample_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
                sample.gpu.as_nanos(),
                sample.wall.as_nanos(),
            );
            rates.push(gpu_rate);
        }
        rates.sort_by(f64::total_cmp);
        let rate = median(&rates);
        if rate > best_rate {
            best_rate = rate;
            best_width = width;
        }
    }
    assert!(best_width > 0);

    for warmup_index in 0..warmups {
        let sample = harness
            .run_rotating(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("AMNS8 warmup {warmup_index}: {error}"));
        assert_amns8_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("AMNS8 decision warmup {warmup_index}"),
        );
        assert!(harness.padding_is_untouched());
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        std::println!(
            "metal_msm_amns8_warmup width={best_width} index={warmup_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
    }

    let mut gpu_rates = Vec::with_capacity(samples);
    let mut wall_rates = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = harness
            .run_rotating(iterations, dispatches, best_width)
            .unwrap_or_else(|error| panic!("AMNS8 sample {sample_index}: {error}"));
        assert_amns8_outputs(
            &harness.read_outputs(),
            &expected,
            &format!("AMNS8 decision sample {sample_index}"),
        );
        assert!(harness.padding_is_untouched());
        let (gpu_rate, wall_rate) = u32_field_rate(sample, fields, iterations, dispatches);
        std::println!(
            "metal_msm_amns8_sample width={best_width} index={sample_index} gpu_ns={} wall_ns={} gpu_complete_fp_per_second={gpu_rate:.3} wall_complete_fp_per_second={wall_rate:.3}",
            sample.gpu.as_nanos(),
            sample.wall.as_nanos(),
        );
        gpu_rates.push(gpu_rate);
        wall_rates.push(wall_rate);
    }
    gpu_rates.sort_by(f64::total_cmp);
    wall_rates.sort_by(f64::total_cmp);
    let median_gpu = median(&gpu_rates);
    let p5_gpu = nearest_rank_percentile(&gpu_rates, 5, 100);
    let median_wall = median(&wall_rates);
    let p5_wall = nearest_rank_percentile(&wall_rates, 5, 100);
    let throughput_pass = median_gpu >= U32_FIELD_MEDIAN_GATE && p5_gpu >= U32_FIELD_P5_GATE;
    let decision = if throughput_pass { "GO" } else { "NO_GO" };
    let complete_operations = u64::try_from(fields)
        .expect("field count fits u64")
        .checked_mul(u64::from(iterations))
        .and_then(|value| value.checked_mul(u64::from(dispatches)))
        .expect("AMNS8 operation count fits u64");
    std::println!(
        "metal_msm_amns8_gate_result algorithm_version={AMNS8_ALGORITHM_VERSION} schedule_fingerprint={AMNS8_SCHEDULE_FINGERPRINT:#018x} logical_products_per_field={AMNS8_LOGICAL_PRODUCTS_PER_FIELD} logical_widened={AMNS8_LOGICAL_WIDENED_PRODUCTS_PER_FIELD} logical_wrapping={AMNS8_LOGICAL_WRAPPING_PRODUCTS_PER_FIELD} issued_products_per_field={AMNS8_ISSUED_PRODUCTS_PER_FIELD} issued_widened={AMNS8_ISSUED_WIDENED_PRODUCTS_PER_FIELD} issued_wrapping={AMNS8_ISSUED_WRAPPING_PRODUCTS_PER_FIELD} typed_peak_live_words={AMNS8_TYPED_PEAK_LIVE_WORDS} tokenized_peak_live_words={AMNS8_TOKENIZED_PEAK_LIVE_WORDS} allocation_peak_live_words={AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS} source_live_word_limit={AMNS8_SOURCE_LIVE_WORD_LIMIT} device={:?} registry_id={} thread_execution_width={} threadgroup_width={best_width} max_threadgroup_width={} threads_per_field=8 logical_fields={} physical_fields={} iterations={iterations} dispatches={dispatches} complete_operations_per_sample={complete_operations} warmups={warmups} samples={samples} one_command_and_one_wait_per_sample=true timed_allocation=false timed_upload=false timed_readback=false output_checked_after_every_sample=true padding_untouched=true static_threadgroup_memory_bytes={} median_gpu_complete_fp_per_second={median_gpu:.3} p5_gpu_complete_fp_per_second={p5_gpu:.3} median_wall_complete_fp_per_second={median_wall:.3} p5_wall_complete_fp_per_second={p5_wall:.3} median_gate={U32_FIELD_MEDIAN_GATE:.3} p5_gate={U32_FIELD_P5_GATE:.3} decision={decision}",
        harness.device_name(),
        harness.registry_id(),
        harness.thread_execution_width(),
        harness.max_threadgroup_width(),
        harness.logical_fields(),
        harness.physical_fields(),
        harness.static_threadgroup_memory_bytes(),
    );
    assert!(
        throughput_pass,
        "AMNS8 field gate failed: median {median_gpu:.3} < {U32_FIELD_MEDIAN_GATE:.3} or nearest-rank p5 {p5_gpu:.3} < {U32_FIELD_P5_GATE:.3} complete operations/s"
    );
}
