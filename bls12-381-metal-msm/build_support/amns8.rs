// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Standalone generator for an eight-lane AMNS Montgomery multiplication.
//!
//! The generated Metal function assigns one coefficient to each lane of an
//! eight-lane subwarp.  Four independent AMNS elements therefore occupy one
//! SIMD32.  It uses a balanced K8 graph (27 logical leaves) for
//! all three polynomial convolutions and reduces the leaf contribution
//! polynomials directly modulo `E = X^8 + 2`.
//!
//! This module deliberately depends only on `std`: `build.rs`, a future
//! standalone crate, and the focused model test can all include it directly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub(crate) const COEFFICIENTS: usize = 8;
pub(crate) const LEAVES_PER_CONVOLUTION: u32 = 27;
pub(crate) const BASE_PRODUCTS_PER_LEAF: u32 = 3;
pub(crate) const LOGICAL_BASE_PRODUCTS: u32 = 3 * LEAVES_PER_CONVOLUTION * BASE_PRODUCTS_PER_LEAF;
pub(crate) const LOGICAL_WIDENED_PRODUCTS: u32 = 189;
pub(crate) const LOGICAL_WRAPPING_PRODUCTS: u32 = 54;
pub(crate) const ISSUED_LEAF_SLOTS_PER_CONVOLUTION: u32 = 4 * COEFFICIENTS as u32;
pub(crate) const ISSUED_BASE_PRODUCTS_PER_LANE: u32 = 3 * 4 * BASE_PRODUCTS_PER_LEAF;
pub(crate) const ISSUED_WIDENED_PRODUCTS_PER_FIELD: u32 = 224;
pub(crate) const ISSUED_WRAPPING_PRODUCTS_PER_FIELD: u32 = 64;

/// Source-level implementation of each conceptual widened `u32 * u32` leaf.
///
/// V1 is frozen for reproducibility. V2 spells out the low and high halves so
/// the Metal front end has no variable-width `ulong * ulong` expression to
/// retain in optimized AIR.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScheduleVariant {
    UlongMultiplyV1,
    ExplicitMulhiV2,
}

impl ScheduleVariant {
    fn callable_name(self) -> &'static str {
        match self {
            Self::UlongMultiplyV1 => "amns8_multiply",
            Self::ExplicitMulhiV2 => "amns8_mulhi_v2_multiply",
        }
    }

    fn params_name(self) -> &'static str {
        match self {
            Self::UlongMultiplyV1 => "Bls12381MetalMsmAmns8Params",
            Self::ExplicitMulhiV2 => "Bls12381MetalMsmAmns8MulhiV2Params",
        }
    }

    fn abi_kernel_name(self) -> &'static str {
        match self {
            Self::UlongMultiplyV1 => "bls12_381_metal_msm_amns8_field_abi",
            Self::ExplicitMulhiV2 => "bls12_381_metal_msm_amns8_mulhi_v2_field_abi",
        }
    }

    fn multiply_kernel_name(self) -> &'static str {
        match self {
            Self::UlongMultiplyV1 => "bls12_381_metal_msm_amns8_field_multiply",
            Self::ExplicitMulhiV2 => "bls12_381_metal_msm_amns8_mulhi_v2_field_multiply",
        }
    }

    fn rotating_kernel_name(self) -> &'static str {
        match self {
            Self::UlongMultiplyV1 => "bls12_381_metal_msm_amns8_rotating_chain",
            Self::ExplicitMulhiV2 => "bls12_381_metal_msm_amns8_mulhi_v2_rotating_chain",
        }
    }

    fn algorithm_version(self) -> u64 {
        match self {
            Self::UlongMultiplyV1 => 1,
            Self::ExplicitMulhiV2 => 2,
        }
    }
}

/// `M` for `E = X^8 + 2`, sign-matched to [`M_PRIME`].
pub(crate) const M: [i64; COEFFICIENTS] = [
    80_042_099_451_959,
    -438_404_363_789,
    -104_528_439_513_579,
    91_613_899_373_051,
    -7_504_120_839_914,
    125_847_515_758_073,
    -143_766_863_997_100,
    -95_095_973_594_275,
];

/// `-M^-1` in `(Z / 2^64 Z)[X] / (X^8 + 2)`.
pub(crate) const M_PRIME: [u64; COEFFICIENTS] = [
    0x0010_c6fb_6f1f_4907,
    0x7a70_8f07_2a6f_c213,
    0xe6af_80b1_5b4d_0bc4,
    0xa5ee_da87_343c_dd86,
    0x516f_cfe4_2de2_b795,
    0x3382_6e37_0315_ed0a,
    0xc7af_1c65_3aff_f478,
    0x1db4_899b_375f_66c0,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KaratsubaLeaf {
    /// Linear transform of the left polynomial, coefficients in `{-1,0,1}`.
    pub(crate) left: [i8; COEFFICIENTS],
    /// Linear transform of the right polynomial, coefficients in `{-1,0,1}`.
    pub(crate) right: [i8; COEFFICIENTS],
    /// Direct contribution to each coefficient after `X^8 = -2`.
    pub(crate) output_weights: [i8; COEFFICIENTS],
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct GeneratedSchedule {
    pub(crate) source: String,
    /// Stable FNV-1a fingerprint of the callable body and frozen arithmetic graph.
    pub(crate) fingerprint: String,
    pub(crate) leaves: Vec<KaratsubaLeaf>,
    pub(crate) logical_leaves_per_convolution: u32,
    pub(crate) logical_base_products: u32,
    pub(crate) logical_widened_products: u32,
    pub(crate) logical_wrapping_products: u32,
    /// Four rounds issue 32 lane slots; five are algebraic zero in round four.
    pub(crate) issued_leaf_slots_per_convolution: u32,
    /// Base-product instructions executed by one physical lane over all rounds.
    pub(crate) issued_base_products_per_lane: u32,
    pub(crate) issued_widened_products_per_field: u32,
    pub(crate) issued_wrapping_products_per_field: u32,
    pub(crate) emitted_base_product_instructions: u32,
    pub(crate) emitted_widened_product_instructions: u32,
    pub(crate) emitted_wrapping_product_instructions: u32,
    pub(crate) typed_peak_live_words: u32,
    pub(crate) tokenized_peak_live_words: u32,
    pub(crate) typed_allocation_peak_live_words: u32,
    pub(crate) tokenized_allocation_peak_live_words: u32,
    pub(crate) typed_peak_label: String,
    pub(crate) tokenized_peak_label: String,
    pub(crate) typed_peak_values: Vec<String>,
    pub(crate) tokenized_peak_values: Vec<String>,
}

type Mask = [i8; COEFFICIENTS];
type Contribution = [i16; 2 * COEFFICIENTS - 1];

#[derive(Clone, Copy)]
enum OperandSide {
    Left,
    Right,
}

#[derive(Clone)]
struct RecursiveLeaf {
    left: Mask,
    right: Mask,
    contribution: Contribution,
}

fn add_mask(left: &Mask, right: &Mask, right_sign: i8) -> Mask {
    std::array::from_fn(|index| {
        let value = left[index] + right_sign * right[index];
        assert!((-1..=1).contains(&value));
        value
    })
}

fn shifted_contribution(contribution: &Contribution, shifts: &[(usize, i16)]) -> Contribution {
    let mut result = [0_i16; 2 * COEFFICIENTS - 1];
    for (degree, coefficient) in contribution.iter().copied().enumerate() {
        if coefficient == 0 {
            continue;
        }
        for &(shift, weight) in shifts {
            result[degree + shift] += coefficient * weight;
        }
    }
    result
}

fn recursive_k8(left: &[Mask], right: &[Mask]) -> Vec<RecursiveLeaf> {
    assert_eq!(left.len(), right.len());
    assert!(left.len().is_power_of_two());
    if left.len() == 1 {
        let mut contribution = [0_i16; 2 * COEFFICIENTS - 1];
        contribution[0] = 1;
        return vec![RecursiveLeaf { left: left[0], right: right[0], contribution }];
    }

    let half = left.len() / 2;
    let low = recursive_k8(&left[..half], &right[..half]);
    let high = recursive_k8(&left[half..], &right[half..]);
    let left_sum: Vec<_> =
        (0..half).map(|index| add_mask(&left[index], &left[index + half], 1)).collect();
    let right_sum: Vec<_> =
        (0..half).map(|index| add_mask(&right[index], &right[index + half], 1)).collect();
    let middle = recursive_k8(&left_sum, &right_sum);

    let mut result = Vec::with_capacity(low.len() + high.len() + middle.len());
    for mut leaf in low {
        leaf.contribution = shifted_contribution(&leaf.contribution, &[(0, 1), (half, -1)]);
        result.push(leaf);
    }
    for mut leaf in high {
        leaf.contribution = shifted_contribution(&leaf.contribution, &[(half, -1), (2 * half, 1)]);
        result.push(leaf);
    }
    for mut leaf in middle {
        leaf.contribution = shifted_contribution(&leaf.contribution, &[(half, 1)]);
        result.push(leaf);
    }
    result
}

/// Construct the canonical balanced subtractive K8 graph.
pub(crate) fn karatsuba_leaves() -> Vec<KaratsubaLeaf> {
    let basis: Vec<Mask> = (0..COEFFICIENTS)
        .map(|coefficient| {
            let mut mask = [0_i8; COEFFICIENTS];
            mask[coefficient] = 1;
            mask
        })
        .collect();
    let recursive = recursive_k8(&basis, &basis);
    assert_eq!(recursive.len(), LEAVES_PER_CONVOLUTION as usize);

    recursive
        .into_iter()
        .map(|leaf| {
            let mut weights = [0_i16; COEFFICIENTS];
            for (degree, coefficient) in leaf.contribution.into_iter().enumerate() {
                if degree < COEFFICIENTS {
                    weights[degree] += coefficient;
                } else {
                    // E = X^8 + 2, hence X^8 = -2.
                    weights[degree - COEFFICIENTS] -= 2 * coefficient;
                }
            }
            assert!(weights.iter().all(|weight| (-2..=2).contains(weight)));
            KaratsubaLeaf {
                left: leaf.left,
                right: leaf.right,
                output_weights: weights.map(|weight| weight as i8),
            }
        })
        .collect()
}

fn multiply_mod_e_wrapping(left: &[u64; 8], right: &[u64; 8]) -> [u64; 8] {
    let mut result = [0_u64; 8];
    for (left_index, left_value) in left.iter().copied().enumerate() {
        for (right_index, right_value) in right.iter().copied().enumerate() {
            let degree = left_index + right_index;
            let product = left_value.wrapping_mul(right_value);
            if degree < 8 {
                result[degree] = result[degree].wrapping_add(product);
            } else {
                result[degree - 8] = result[degree - 8].wrapping_sub(product.wrapping_mul(2));
            }
        }
    }
    result
}

fn round_coefficient_masks(
    leaves: &[KaratsubaLeaf],
    round: usize,
    side: OperandSide,
    coefficient: usize,
) -> (u8, u8) {
    let mut active = 0_u8;
    let mut negative = 0_u8;
    for owner in 0..COEFFICIENTS {
        let Some(leaf) = leaves.get(round * COEFFICIENTS + owner) else {
            continue;
        };
        let mask = match side {
            OperandSide::Left => leaf.left[coefficient],
            OperandSide::Right => leaf.right[coefficient],
        };
        assert!((-1..=1).contains(&mask));
        if mask != 0 {
            active |= 1 << owner;
        }
        if mask < 0 {
            negative |= 1 << owner;
        }
    }
    assert_eq!(negative & !active, 0);
    (active, negative)
}

fn signed_leaf_transform(mask: &Mask, polynomial: &[i64; 8]) -> i64 {
    let value: i128 = mask
        .iter()
        .zip(polynomial)
        .map(|(&coefficient, &value)| i128::from(coefficient) * i128::from(value))
        .sum();
    i64::try_from(value).expect("frozen M transform fits signed i64")
}

fn wrapping_leaf_transform(mask: &Mask, polynomial: &[u64; 8]) -> u64 {
    mask.iter().zip(polynomial).fold(0_u64, |accumulator, (&coefficient, &value)| match coefficient
    {
        -1 => accumulator.wrapping_sub(value),
        0 => accumulator,
        1 => accumulator.wrapping_add(value),
        _ => panic!("K8 transform coefficient outside -1,0,1"),
    })
}

fn signed_literal(value: i64) -> String {
    if value < 0 { format!("(-{}L)", value.unsigned_abs()) } else { format!("{value}L") }
}

fn packed_weights(weights: &[i8; 8]) -> u32 {
    weights.iter().copied().enumerate().fold(0_u32, |packed, (index, weight)| {
        assert!((-2..=2).contains(&weight));
        packed | (u32::from((weight + 2) as u8) << (3 * index))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MslType {
    Bool,
    Ushort,
    Uint,
    Int,
    Ulong,
    Long,
    Uint4,
}

impl MslType {
    fn source(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Ushort => "ushort",
            Self::Uint => "uint",
            Self::Int => "int",
            Self::Ulong => "ulong",
            Self::Long => "long",
            Self::Uint4 => "uint4",
        }
    }

    fn words(self) -> u32 {
        match self {
            Self::Ulong | Self::Long => 2,
            Self::Uint4 => 4,
            Self::Bool | Self::Ushort | Self::Uint | Self::Int => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Value(usize);

#[derive(Debug)]
struct ValueInfo {
    name: String,
    ty: MslType,
    definition: Option<usize>,
}

#[derive(Debug)]
struct Instruction {
    definition: Option<Value>,
    uses: Vec<Value>,
    source: String,
    label: String,
}

#[derive(Debug)]
struct LivenessResult {
    peak_words: u32,
    allocation_peak_words: u32,
    peak_label: String,
    peak_values: Vec<String>,
}

struct Emitter {
    values: Vec<ValueInfo>,
    instructions: Vec<Instruction>,
    base_products: u32,
    widened_products: u32,
    wrapping_products: u32,
    variant: ScheduleVariant,
}

impl Emitter {
    fn new(variant: ScheduleVariant) -> Self {
        Self {
            values: Vec::new(),
            instructions: Vec::new(),
            base_products: 0,
            widened_products: 0,
            wrapping_products: 0,
            variant,
        }
    }

    fn input(&mut self, name: &str, ty: MslType, _label: &str) -> Value {
        let value = Value(self.values.len());
        self.values.push(ValueInfo { name: name.to_owned(), ty, definition: None });
        value
    }

    fn name(&self, value: Value) -> &str {
        &self.values[value.0].name
    }

    fn define(&mut self, ty: MslType, uses: Vec<Value>, expression: String, label: &str) -> Value {
        let value = Value(self.values.len());
        let name = format!("v{}", value.0);
        let instruction_index = self.instructions.len();
        self.values.push(ValueInfo { name: name.clone(), ty, definition: Some(instruction_index) });
        self.instructions.push(Instruction {
            definition: Some(value),
            uses,
            source: format!("    const {} {name} = {expression};", ty.source()),
            label: label.to_owned(),
        });
        value
    }

    fn define_base_product(&mut self, uses: Vec<Value>, expression: String, label: &str) -> Value {
        let result = match self.variant {
            ScheduleVariant::UlongMultiplyV1 => {
                self.define(MslType::Ulong, uses, expression, label)
            }
            ScheduleVariant::ExplicitMulhiV2 => {
                assert_eq!(uses.len(), 2, "widened base product arity");
                let left = uses[0];
                let right = uses[1];
                assert_eq!(self.values[left.0].ty, MslType::Uint);
                assert_eq!(self.values[right.0].ty, MslType::Uint);
                let low = self.define(
                    MslType::Uint,
                    uses.clone(),
                    format!("{} * {}", self.name(left), self.name(right)),
                    &format!("{label} explicit low"),
                );
                self.instructions
                    .last_mut()
                    .expect("explicit widened low instruction")
                    .source
                    .push_str(" // @widened-u32-product-low");
                let high = self.define(
                    MslType::Uint,
                    uses,
                    format!("metal::mulhi({}, {})", self.name(left), self.name(right)),
                    &format!("{label} explicit high"),
                );
                self.instructions
                    .last_mut()
                    .expect("explicit widened high instruction")
                    .source
                    .push_str(" // @widened-u32-product-high");
                self.define(
                    MslType::Ulong,
                    vec![low, high],
                    format!(
                        "static_cast<ulong>({}) | (static_cast<ulong>({}) << 32)",
                        self.name(low),
                        self.name(high)
                    ),
                    label,
                )
            }
        };
        self.instructions
            .last_mut()
            .expect("base product instruction")
            .source
            .push_str(" // @base-u32-product");
        self.base_products += 1;
        self.widened_products += 1;
        result
    }

    fn define_wrapping_product(
        &mut self,
        uses: Vec<Value>,
        expression: String,
        label: &str,
    ) -> Value {
        let result = self.define(MslType::Uint, uses, expression, label);
        self.instructions
            .last_mut()
            .expect("wrapping product instruction")
            .source
            .push_str(" // @wrapping-u32-product");
        self.base_products += 1;
        self.wrapping_products += 1;
        result
    }

    fn marker(&mut self, uses: Vec<Value>, source: String, label: &str) {
        self.instructions.push(Instruction {
            definition: None,
            uses,
            source,
            label: label.to_owned(),
        });
    }

    fn body_source(&self) -> String {
        let mut result = String::new();
        for instruction in &self.instructions {
            writeln!(result, "{}", instruction.source).expect("write generated instruction");
        }
        result
    }

    fn add128(&mut self, left: Value, right: Value, label: &str) -> Value {
        assert_eq!(self.values[left.0].ty, MslType::Uint4);
        assert_eq!(self.values[right.0].ty, MslType::Uint4);
        let components = ["x", "y", "z", "w"];
        let mut words = Vec::with_capacity(4);
        let mut carry: Option<Value> = None;
        for (index, component) in components.into_iter().enumerate() {
            let mut uses = vec![left, right];
            if let Some(carry) = carry {
                uses.push(carry);
            }
            let expression = match carry {
                Some(carry) => format!(
                    "static_cast<ulong>({left_name}.{component}) + static_cast<ulong>({right_name}.{component}) + static_cast<ulong>({carry_name})",
                    left_name = self.name(left),
                    right_name = self.name(right),
                    carry_name = self.name(carry),
                ),
                None => format!(
                    "static_cast<ulong>({left_name}.{component}) + static_cast<ulong>({right_name}.{component})",
                    left_name = self.name(left),
                    right_name = self.name(right),
                ),
            };
            let sum =
                self.define(MslType::Ulong, uses, expression, &format!("{label} sum word {index}"));
            words.push(self.define(
                MslType::Uint,
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} word {index}"),
            ));
            if index != 3 {
                carry = Some(self.define(
                    MslType::Uint,
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} carry {index}"),
                ));
            }
        }
        self.define(
            MslType::Uint4,
            words.clone(),
            format!(
                "uint4({}, {}, {}, {})",
                self.name(words[0]),
                self.name(words[1]),
                self.name(words[2]),
                self.name(words[3])
            ),
            label,
        )
    }

    fn full_signed_i64_product(&mut self, left: Value, right: Value, label: &str) -> Value {
        assert_eq!(self.values[left.0].ty, MslType::Long);
        assert_eq!(self.values[right.0].ty, MslType::Long);

        let left_bits = self.define(
            MslType::Ulong,
            vec![left],
            format!("as_type<ulong>({})", self.name(left)),
            &format!("{label} left bits"),
        );
        let right_bits = self.define(
            MslType::Ulong,
            vec![right],
            format!("as_type<ulong>({})", self.name(right)),
            &format!("{label} right bits"),
        );
        let result_negative = self.define(
            MslType::Uint,
            vec![left_bits, right_bits],
            format!(
                "static_cast<uint>(({} ^ {}) >> 63)",
                self.name(left_bits),
                self.name(right_bits)
            ),
            &format!("{label} result negative"),
        );

        let left_sign_mask = self.define(
            MslType::Ulong,
            vec![left_bits],
            format!("0UL - ({} >> 63)", self.name(left_bits)),
            &format!("{label} left sign mask"),
        );
        let left_magnitude = self.define(
            MslType::Ulong,
            vec![left_bits, left_sign_mask],
            format!(
                "({} ^ {}) + ({} >> 63)",
                self.name(left_bits),
                self.name(left_sign_mask),
                self.name(left_bits)
            ),
            &format!("{label} left magnitude"),
        );
        let right_sign_mask = self.define(
            MslType::Ulong,
            vec![right_bits],
            format!("0UL - ({} >> 63)", self.name(right_bits)),
            &format!("{label} right sign mask"),
        );
        let right_magnitude = self.define(
            MslType::Ulong,
            vec![right_bits, right_sign_mask],
            format!(
                "({} ^ {}) + ({} >> 63)",
                self.name(right_bits),
                self.name(right_sign_mask),
                self.name(right_bits)
            ),
            &format!("{label} right magnitude"),
        );

        let left_low = self.define(
            MslType::Uint,
            vec![left_magnitude],
            format!("static_cast<uint>({})", self.name(left_magnitude)),
            &format!("{label} left low"),
        );
        let left_high = self.define(
            MslType::Uint,
            vec![left_magnitude],
            format!("static_cast<uint>({} >> 32)", self.name(left_magnitude)),
            &format!("{label} left high"),
        );
        let right_low = self.define(
            MslType::Uint,
            vec![right_magnitude],
            format!("static_cast<uint>({})", self.name(right_magnitude)),
            &format!("{label} right low"),
        );
        let right_high = self.define(
            MslType::Uint,
            vec![right_magnitude],
            format!("static_cast<uint>({} >> 32)", self.name(right_magnitude)),
            &format!("{label} right high"),
        );

        let z0 = self.define_base_product(
            vec![left_low, right_low],
            format!(
                "static_cast<ulong>({}) * static_cast<ulong>({})",
                self.name(left_low),
                self.name(right_low)
            ),
            &format!("{label} z0"),
        );
        let z2 = self.define_base_product(
            vec![left_high, right_high],
            format!(
                "static_cast<ulong>({}) * static_cast<ulong>({})",
                self.name(left_high),
                self.name(right_high)
            ),
            &format!("{label} z2"),
        );

        let left_difference_raw = self.define(
            MslType::Uint,
            vec![left_low, left_high],
            format!("{} - {}", self.name(left_low), self.name(left_high)),
            &format!("{label} left difference raw"),
        );
        let left_difference_negative = self.define(
            MslType::Uint,
            vec![left_low, left_high],
            format!("static_cast<uint>({} < {})", self.name(left_low), self.name(left_high)),
            &format!("{label} left difference negative"),
        );
        let left_difference_mask = self.define(
            MslType::Uint,
            vec![left_difference_negative],
            format!("0U - {}", self.name(left_difference_negative)),
            &format!("{label} left difference mask"),
        );
        let left_difference = self.define(
            MslType::Uint,
            vec![left_difference_raw, left_difference_mask, left_difference_negative],
            format!(
                "({} ^ {}) + {}",
                self.name(left_difference_raw),
                self.name(left_difference_mask),
                self.name(left_difference_negative)
            ),
            &format!("{label} left difference"),
        );
        let right_difference_raw = self.define(
            MslType::Uint,
            vec![right_low, right_high],
            format!("{} - {}", self.name(right_low), self.name(right_high)),
            &format!("{label} right difference raw"),
        );
        let right_difference_negative = self.define(
            MslType::Uint,
            vec![right_low, right_high],
            format!("static_cast<uint>({} < {})", self.name(right_low), self.name(right_high)),
            &format!("{label} right difference negative"),
        );
        let right_difference_mask = self.define(
            MslType::Uint,
            vec![right_difference_negative],
            format!("0U - {}", self.name(right_difference_negative)),
            &format!("{label} right difference mask"),
        );
        let right_difference = self.define(
            MslType::Uint,
            vec![right_difference_raw, right_difference_mask, right_difference_negative],
            format!(
                "({} ^ {}) + {}",
                self.name(right_difference_raw),
                self.name(right_difference_mask),
                self.name(right_difference_negative)
            ),
            &format!("{label} right difference"),
        );
        let difference_negative = self.define(
            MslType::Bool,
            vec![left_difference_negative, right_difference_negative],
            format!(
                "({} ^ {}) != 0U",
                self.name(left_difference_negative),
                self.name(right_difference_negative)
            ),
            &format!("{label} middle sign"),
        );
        let middle = self.define_base_product(
            vec![left_difference, right_difference],
            format!(
                "static_cast<ulong>({}) * static_cast<ulong>({})",
                self.name(left_difference),
                self.name(right_difference)
            ),
            &format!("{label} middle"),
        );

        let cross_base = self.define(
            MslType::Ulong,
            vec![z0, z2],
            format!("{} + {}", self.name(z0), self.name(z2)),
            &format!("{label} cross base"),
        );
        let cross_base_carry = self.define(
            MslType::Uint,
            vec![cross_base, z0],
            format!("static_cast<uint>({} < {})", self.name(cross_base), self.name(z0)),
            &format!("{label} cross base carry"),
        );
        let cross_add = self.define(
            MslType::Ulong,
            vec![cross_base, middle],
            format!("{} + {}", self.name(cross_base), self.name(middle)),
            &format!("{label} cross add"),
        );
        let cross_add_high = self.define(
            MslType::Uint,
            vec![cross_base_carry, cross_add, cross_base],
            format!(
                "{} + static_cast<uint>({} < {})",
                self.name(cross_base_carry),
                self.name(cross_add),
                self.name(cross_base)
            ),
            &format!("{label} cross add high"),
        );
        let cross_subtract = self.define(
            MslType::Ulong,
            vec![cross_base, middle],
            format!("{} - {}", self.name(cross_base), self.name(middle)),
            &format!("{label} cross subtract"),
        );
        let cross_subtract_high = self.define(
            MslType::Uint,
            vec![cross_base_carry, cross_base, middle],
            format!(
                "{} - static_cast<uint>({} < {})",
                self.name(cross_base_carry),
                self.name(cross_base),
                self.name(middle)
            ),
            &format!("{label} cross subtract high"),
        );
        let cross = self.define(
            MslType::Ulong,
            vec![cross_subtract, cross_add, difference_negative],
            format!(
                "select({}, {}, {})",
                self.name(cross_subtract),
                self.name(cross_add),
                self.name(difference_negative)
            ),
            &format!("{label} cross"),
        );
        let cross_high = self.define(
            MslType::Uint,
            vec![cross_subtract_high, cross_add_high, difference_negative],
            format!(
                "select({}, {}, {})",
                self.name(cross_subtract_high),
                self.name(cross_add_high),
                self.name(difference_negative)
            ),
            &format!("{label} cross high"),
        );

        let magnitude_word0 = self.define(
            MslType::Uint,
            vec![z0],
            format!("static_cast<uint>({})", self.name(z0)),
            &format!("{label} magnitude word 0"),
        );
        let magnitude_sum1 = self.define(
            MslType::Ulong,
            vec![z0, cross],
            format!(
                "static_cast<ulong>(static_cast<uint>({} >> 32)) + static_cast<ulong>(static_cast<uint>({}))",
                self.name(z0),
                self.name(cross)
            ),
            &format!("{label} magnitude sum 1"),
        );
        let magnitude_word1 = self.define(
            MslType::Uint,
            vec![magnitude_sum1],
            format!("static_cast<uint>({})", self.name(magnitude_sum1)),
            &format!("{label} magnitude word 1"),
        );
        let magnitude_carry1 = self.define(
            MslType::Uint,
            vec![magnitude_sum1],
            format!("static_cast<uint>({} >> 32)", self.name(magnitude_sum1)),
            &format!("{label} magnitude carry 1"),
        );
        let magnitude_sum2 = self.define(
            MslType::Ulong,
            vec![cross, z2, magnitude_carry1],
            format!(
                "static_cast<ulong>(static_cast<uint>({} >> 32)) + static_cast<ulong>(static_cast<uint>({})) + static_cast<ulong>({})",
                self.name(cross),
                self.name(z2),
                self.name(magnitude_carry1)
            ),
            &format!("{label} magnitude sum 2"),
        );
        let magnitude_word2 = self.define(
            MslType::Uint,
            vec![magnitude_sum2],
            format!("static_cast<uint>({})", self.name(magnitude_sum2)),
            &format!("{label} magnitude word 2"),
        );
        let magnitude_carry2 = self.define(
            MslType::Uint,
            vec![magnitude_sum2],
            format!("static_cast<uint>({} >> 32)", self.name(magnitude_sum2)),
            &format!("{label} magnitude carry 2"),
        );
        let magnitude_sum3 = self.define(
            MslType::Ulong,
            vec![cross_high, z2, magnitude_carry2],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>(static_cast<uint>({} >> 32)) + static_cast<ulong>({})",
                self.name(cross_high),
                self.name(z2),
                self.name(magnitude_carry2)
            ),
            &format!("{label} magnitude sum 3"),
        );
        let magnitude_word3 = self.define(
            MslType::Uint,
            vec![magnitude_sum3],
            format!("static_cast<uint>({})", self.name(magnitude_sum3)),
            &format!("{label} magnitude word 3"),
        );

        let result_mask = self.define(
            MslType::Uint,
            vec![result_negative],
            format!("0U - {}", self.name(result_negative)),
            &format!("{label} result mask"),
        );
        let magnitudes = [magnitude_word0, magnitude_word1, magnitude_word2, magnitude_word3];
        let mut result_words = Vec::with_capacity(4);
        let mut carry = result_negative;
        for (index, magnitude) in magnitudes.into_iter().enumerate() {
            let sum = self.define(
                MslType::Ulong,
                vec![magnitude, result_mask, carry],
                format!(
                    "static_cast<ulong>({} ^ {}) + static_cast<ulong>({})",
                    self.name(magnitude),
                    self.name(result_mask),
                    self.name(carry)
                ),
                &format!("{label} signed sum {index}"),
            );
            result_words.push(self.define(
                MslType::Uint,
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} signed word {index}"),
            ));
            if index != 3 {
                carry = self.define(
                    MslType::Uint,
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} signed carry {index}"),
                );
            }
        }
        self.define(
            MslType::Uint4,
            result_words.clone(),
            format!(
                "uint4({}, {}, {}, {})",
                self.name(result_words[0]),
                self.name(result_words[1]),
                self.name(result_words[2]),
                self.name(result_words[3])
            ),
            label,
        )
    }

    fn low_u64_product(&mut self, left: Value, right: Value, label: &str) -> Value {
        assert_eq!(self.values[left.0].ty, MslType::Ulong);
        assert_eq!(self.values[right.0].ty, MslType::Ulong);
        let left_low = self.define(
            MslType::Uint,
            vec![left],
            format!("static_cast<uint>({})", self.name(left)),
            &format!("{label} left low"),
        );
        let left_high = self.define(
            MslType::Uint,
            vec![left],
            format!("static_cast<uint>({} >> 32)", self.name(left)),
            &format!("{label} left high"),
        );
        let right_low = self.define(
            MslType::Uint,
            vec![right],
            format!("static_cast<uint>({})", self.name(right)),
            &format!("{label} right low"),
        );
        let right_high = self.define(
            MslType::Uint,
            vec![right],
            format!("static_cast<uint>({} >> 32)", self.name(right)),
            &format!("{label} right high"),
        );
        let low = self.define_base_product(
            vec![left_low, right_low],
            format!(
                "static_cast<ulong>({}) * static_cast<ulong>({})",
                self.name(left_low),
                self.name(right_low)
            ),
            &format!("{label} low"),
        );
        let cross_left = self.define_wrapping_product(
            vec![left_low, right_high],
            format!("{} * {}", self.name(left_low), self.name(right_high)),
            &format!("{label} cross left"),
        );
        let cross_right = self.define_wrapping_product(
            vec![left_high, right_low],
            format!("{} * {}", self.name(left_high), self.name(right_low)),
            &format!("{label} cross right"),
        );
        self.define(
            MslType::Ulong,
            vec![low, cross_left, cross_right],
            format!(
                "{} + (static_cast<ulong>({} + {}) << 32)",
                self.name(low),
                self.name(cross_left),
                self.name(cross_right)
            ),
            label,
        )
    }

    fn shuffle_u64(
        &mut self,
        lane_bits: Value,
        subgroup_base: Value,
        coefficient: usize,
        label: &str,
    ) -> Value {
        assert_eq!(self.values[lane_bits.0].ty, MslType::Ulong);
        assert!(coefficient < COEFFICIENTS);
        self.define(
            MslType::Ulong,
            vec![lane_bits, subgroup_base],
            format!(
                "static_cast<ulong>(simd_shuffle(static_cast<uint>({bits}), static_cast<ushort>({base} + {coefficient}U))) | (static_cast<ulong>(simd_shuffle(static_cast<uint>({bits} >> 32), static_cast<ushort>({base} + {coefficient}U))) << 32)",
                bits = self.name(lane_bits),
                base = self.name(subgroup_base),
            ),
            label,
        )
    }

    // Keeping every graph coordinate explicit makes source/audit mismatches
    // harder to hide than packing them into an opaque configuration object.
    #[allow(clippy::too_many_arguments)]
    fn exact_transform(
        &mut self,
        lane_bits: Value,
        local_lane: Value,
        subgroup_base: Value,
        leaves: &[KaratsubaLeaf],
        round: usize,
        side: OperandSide,
        label: &str,
    ) -> Value {
        let mut accumulator =
            self.define(MslType::Long, Vec::new(), "0L".to_owned(), &format!("{label} zero"));
        for coefficient in 0..COEFFICIENTS {
            let (active_mask, negative_mask) =
                round_coefficient_masks(leaves, round, side, coefficient);
            if active_mask == 0 {
                continue;
            }
            let shuffled = self.shuffle_u64(
                lane_bits,
                subgroup_base,
                coefficient,
                &format!("{label} shuffled coefficient {coefficient}"),
            );
            let signed_expression = if negative_mask == 0 {
                format!("as_type<long>({})", self.name(shuffled))
            } else {
                format!(
                    "select(as_type<long>({value}), -as_type<long>({value}), ((0x{negative_mask:02x}U >> {lane}) & 1U) != 0U)",
                    value = self.name(shuffled),
                    lane = self.name(local_lane),
                )
            };
            let term = self.define(
                MslType::Long,
                vec![shuffled, local_lane],
                format!(
                    "select(0L, {signed_expression}, ((0x{active_mask:02x}U >> {lane}) & 1U) != 0U)",
                    lane = self.name(local_lane),
                ),
                &format!("{label} term {coefficient}"),
            );
            accumulator = self.define(
                MslType::Long,
                vec![accumulator, term],
                format!("{} + {}", self.name(accumulator), self.name(term)),
                &format!("{label} partial {coefficient}"),
            );
        }
        accumulator
    }

    #[allow(clippy::too_many_arguments)]
    fn wrapping_transform(
        &mut self,
        lane_bits: Value,
        local_lane: Value,
        subgroup_base: Value,
        leaves: &[KaratsubaLeaf],
        round: usize,
        side: OperandSide,
        label: &str,
    ) -> Value {
        let mut accumulator =
            self.define(MslType::Ulong, Vec::new(), "0UL".to_owned(), &format!("{label} zero"));
        for coefficient in 0..COEFFICIENTS {
            let (active_mask, negative_mask) =
                round_coefficient_masks(leaves, round, side, coefficient);
            if active_mask == 0 {
                continue;
            }
            let shuffled = self.shuffle_u64(
                lane_bits,
                subgroup_base,
                coefficient,
                &format!("{label} shuffled coefficient {coefficient}"),
            );
            let signed_expression = if negative_mask == 0 {
                self.name(shuffled).to_owned()
            } else {
                format!(
                    "select({value}, 0UL - {value}, ((0x{negative_mask:02x}U >> {lane}) & 1U) != 0U)",
                    value = self.name(shuffled),
                    lane = self.name(local_lane),
                )
            };
            let term = self.define(
                MslType::Ulong,
                vec![shuffled, local_lane],
                format!(
                    "select(0UL, {signed_expression}, ((0x{active_mask:02x}U >> {lane}) & 1U) != 0U)",
                    lane = self.name(local_lane),
                ),
                &format!("{label} term {coefficient}"),
            );
            accumulator = self.define(
                MslType::Ulong,
                vec![accumulator, term],
                format!("{} + {}", self.name(accumulator), self.name(term)),
                &format!("{label} partial {coefficient}"),
            );
        }
        accumulator
    }

    /// Return `low_signed, carry` such that the exact transform is
    /// `low_signed + carry * 2^64` and `|carry| <= 4`.
    fn quotient_transform(
        &mut self,
        quotient_lane_bits: Value,
        local_lane: Value,
        subgroup_base: Value,
        leaves: &[KaratsubaLeaf],
        round: usize,
        label: &str,
    ) -> (Value, Value) {
        let mut low =
            self.define(MslType::Ulong, Vec::new(), "0UL".to_owned(), &format!("{label} low zero"));
        let mut high =
            self.define(MslType::Long, Vec::new(), "0L".to_owned(), &format!("{label} high zero"));
        for coefficient in 0..COEFFICIENTS {
            let (active_mask, negative_mask) =
                round_coefficient_masks(leaves, round, OperandSide::Left, coefficient);
            if active_mask == 0 {
                continue;
            }
            let shuffled = self.shuffle_u64(
                quotient_lane_bits,
                subgroup_base,
                coefficient,
                &format!("{label} shuffled coefficient {coefficient}"),
            );
            let negative_condition = format!(
                "((0x{negative_mask:02x}U >> {lane}) & 1U) != 0U",
                lane = self.name(local_lane)
            );
            let active_condition = format!(
                "((0x{active_mask:02x}U >> {lane}) & 1U) != 0U",
                lane = self.name(local_lane)
            );
            let term_low = self.define(
                MslType::Ulong,
                vec![shuffled, local_lane],
                format!(
                    "select(0UL, select({value}, 0UL - {value}, {negative_condition}), {active_condition})",
                    value = self.name(shuffled),
                ),
                &format!("{label} term low {coefficient}"),
            );
            let term_high = self.define(
                MslType::Long,
                vec![shuffled, local_lane],
                format!(
                    "select(0L, select(-static_cast<long>({value} >> 63), -static_cast<long>(as_type<long>({value}) > 0L), {negative_condition}), {active_condition})",
                    value = self.name(shuffled),
                ),
                &format!("{label} term high {coefficient}"),
            );
            let next_low = self.define(
                MslType::Ulong,
                vec![low, term_low],
                format!("{} + {}", self.name(low), self.name(term_low)),
                &format!("{label} low {coefficient}"),
            );
            let carry_out = self.define(
                MslType::Long,
                vec![next_low, low],
                format!("static_cast<long>({} < {})", self.name(next_low), self.name(low)),
                &format!("{label} carry out {coefficient}"),
            );
            let next_high = self.define(
                MslType::Long,
                vec![high, term_high, carry_out],
                format!(
                    "{} + {} + {}",
                    self.name(high),
                    self.name(term_high),
                    self.name(carry_out)
                ),
                &format!("{label} high {coefficient}"),
            );
            low = next_low;
            high = next_high;
        }
        let low_signed = self.define(
            MslType::Long,
            vec![low],
            format!("as_type<long>({})", self.name(low)),
            &format!("{label} centered low"),
        );
        let carry = self.define(
            MslType::Long,
            vec![high, low],
            format!("{} + static_cast<long>({} >> 63)", self.name(high), self.name(low)),
            &format!("{label} centered carry"),
        );
        (low_signed, carry)
    }

    fn select_signed_constant(
        &mut self,
        local_lane: Value,
        values: &[i64; 8],
        label: &str,
    ) -> Value {
        let mut expression = signed_literal(values[0]);
        for (lane, value) in values.iter().copied().enumerate().skip(1) {
            expression = format!(
                "select({expression}, {value}, {local} == {lane}U)",
                value = signed_literal(value),
                local = self.name(local_lane),
            );
        }
        self.define(MslType::Long, vec![local_lane], expression, label)
    }

    fn select_wrapping_constant(
        &mut self,
        local_lane: Value,
        values: &[u64; 8],
        label: &str,
    ) -> Value {
        let mut expression = format!("0x{:016x}UL", values[0]);
        for (lane, value) in values.iter().copied().enumerate().skip(1) {
            expression = format!(
                "select({expression}, 0x{value:016x}UL, {local} == {lane}U)",
                local = self.name(local_lane),
            );
        }
        self.define(MslType::Ulong, vec![local_lane], expression, label)
    }

    fn shuffle_uint4(
        &mut self,
        value: Value,
        subgroup_base: Value,
        owner: usize,
        label: &str,
    ) -> Value {
        assert_eq!(self.values[value.0].ty, MslType::Uint4);
        self.define(
            MslType::Uint4,
            vec![value, subgroup_base],
            format!(
                "uint4(simd_shuffle({value}.x, static_cast<ushort>({base} + {owner}U)), simd_shuffle({value}.y, static_cast<ushort>({base} + {owner}U)), simd_shuffle({value}.z, static_cast<ushort>({base} + {owner}U)), simd_shuffle({value}.w, static_cast<ushort>({base} + {owner}U)))",
                value = self.name(value),
                base = self.name(subgroup_base),
            ),
            label,
        )
    }

    fn shuffle_ulong(
        &mut self,
        value: Value,
        subgroup_base: Value,
        owner: usize,
        label: &str,
    ) -> Value {
        assert_eq!(self.values[value.0].ty, MslType::Ulong);
        self.define(
            MslType::Ulong,
            vec![value, subgroup_base],
            format!(
                "static_cast<ulong>(simd_shuffle(static_cast<uint>({value}), static_cast<ushort>({base} + {owner}U))) | (static_cast<ulong>(simd_shuffle(static_cast<uint>({value} >> 32), static_cast<ushort>({base} + {owner}U))) << 32)",
                value = self.name(value),
                base = self.name(subgroup_base),
            ),
            label,
        )
    }

    fn weighted_add128(
        &mut self,
        accumulator: Value,
        product: Value,
        packed: u32,
        local_lane: Value,
        label: &str,
    ) -> Value {
        let weight = self.define(
            MslType::Int,
            vec![local_lane],
            format!(
                "static_cast<int>((0x{packed:08x}U >> (3U * {})) & 7U) - 2",
                self.name(local_lane)
            ),
            &format!("{label} weight"),
        );
        let absolute_weight = self.define(
            MslType::Uint,
            vec![weight],
            format!("static_cast<uint>(abs({}))", self.name(weight)),
            &format!("{label} absolute weight"),
        );
        let doubled = self.add128(product, product, &format!("{label} doubled"));
        let sign_mask = self.define(
            MslType::Uint,
            vec![weight],
            format!("0U - static_cast<uint>({} < 0)", self.name(weight)),
            &format!("{label} sign mask"),
        );
        let components = ["x", "y", "z", "w"];
        let mut signed_words = Vec::with_capacity(4);
        let mut carry = self.define(
            MslType::Uint,
            vec![weight],
            format!("static_cast<uint>({} < 0)", self.name(weight)),
            &format!("{label} sign carry zero"),
        );
        for (index, component) in components.into_iter().enumerate() {
            let scaled = self.define(
                MslType::Uint,
                vec![product, doubled, absolute_weight],
                format!(
                    "select(0U, select({product}.{component}, {doubled}.{component}, {absolute} == 2U), {absolute} != 0U)",
                    product = self.name(product),
                    doubled = self.name(doubled),
                    absolute = self.name(absolute_weight),
                ),
                &format!("{label} scaled word {index}"),
            );
            let signed_sum = self.define(
                MslType::Ulong,
                vec![scaled, sign_mask, carry],
                format!(
                    "static_cast<ulong>({} ^ {}) + static_cast<ulong>({})",
                    self.name(scaled),
                    self.name(sign_mask),
                    self.name(carry)
                ),
                &format!("{label} signed sum {index}"),
            );
            signed_words.push(self.define(
                MslType::Uint,
                vec![signed_sum],
                format!("static_cast<uint>({})", self.name(signed_sum)),
                &format!("{label} signed word {index}"),
            ));
            if index != 3 {
                carry = self.define(
                    MslType::Uint,
                    vec![signed_sum],
                    format!("static_cast<uint>({} >> 32)", self.name(signed_sum)),
                    &format!("{label} signed carry {index}"),
                );
            }
        }
        let term = self.define(
            MslType::Uint4,
            signed_words.clone(),
            format!(
                "uint4({}, {}, {}, {})",
                self.name(signed_words[0]),
                self.name(signed_words[1]),
                self.name(signed_words[2]),
                self.name(signed_words[3])
            ),
            &format!("{label} signed term"),
        );
        self.add128(accumulator, term, label)
    }

    fn weighted_add_u64(
        &mut self,
        accumulator: Value,
        product: Value,
        packed: u32,
        local_lane: Value,
        label: &str,
    ) -> Value {
        let weight = self.define(
            MslType::Int,
            vec![local_lane],
            format!(
                "static_cast<int>((0x{packed:08x}U >> (3U * {})) & 7U) - 2",
                self.name(local_lane)
            ),
            &format!("{label} weight"),
        );
        let absolute_weight = self.define(
            MslType::Uint,
            vec![weight],
            format!("static_cast<uint>(abs({}))", self.name(weight)),
            &format!("{label} absolute weight"),
        );
        let magnitude = self.define(
            MslType::Ulong,
            vec![product, absolute_weight],
            format!(
                "select(0UL, select({product}, {product} << 1, {absolute} == 2U), {absolute} != 0U)",
                product = self.name(product),
                absolute = self.name(absolute_weight),
            ),
            &format!("{label} magnitude"),
        );
        let signed = self.define(
            MslType::Ulong,
            vec![magnitude, weight],
            format!(
                "select({}, 0UL - {}, {} < 0)",
                self.name(magnitude),
                self.name(magnitude),
                self.name(weight)
            ),
            &format!("{label} signed term"),
        );
        self.define(
            MslType::Ulong,
            vec![accumulator, signed],
            format!("{} + {}", self.name(accumulator), self.name(signed)),
            label,
        )
    }

    fn add_q_carry_correction(
        &mut self,
        low_product: Value,
        carry: Value,
        m_transform: Value,
        label: &str,
    ) -> Value {
        let absolute_carry = self.define(
            MslType::Uint,
            vec![carry],
            format!("static_cast<uint>(abs({}))", self.name(carry)),
            &format!("{label} absolute carry"),
        );
        let twice_m = self.define(
            MslType::Long,
            vec![m_transform],
            format!("{} + {}", self.name(m_transform), self.name(m_transform)),
            &format!("{label} twice M"),
        );
        let three_m = self.define(
            MslType::Long,
            vec![twice_m, m_transform],
            format!("{} + {}", self.name(twice_m), self.name(m_transform)),
            &format!("{label} three M"),
        );
        let four_m = self.define(
            MslType::Long,
            vec![twice_m],
            format!("{} + {}", self.name(twice_m), self.name(twice_m)),
            &format!("{label} four M"),
        );
        let correction_magnitude = self.define(
            MslType::Long,
            vec![absolute_carry, m_transform, twice_m, three_m, four_m],
            format!(
                "select(select(select(select(0L, {m}, {absolute} == 1U), {m2}, {absolute} == 2U), {m3}, {absolute} == 3U), {m4}, {absolute} == 4U)",
                m = self.name(m_transform),
                m2 = self.name(twice_m),
                m3 = self.name(three_m),
                m4 = self.name(four_m),
                absolute = self.name(absolute_carry),
            ),
            &format!("{label} correction magnitude"),
        );
        let correction = self.define(
            MslType::Long,
            vec![correction_magnitude, carry],
            format!(
                "select({}, -{}, {} < 0L)",
                self.name(correction_magnitude),
                self.name(correction_magnitude),
                self.name(carry)
            ),
            &format!("{label} correction"),
        );
        let correction_bits = self.define(
            MslType::Ulong,
            vec![correction],
            format!("as_type<ulong>({})", self.name(correction)),
            &format!("{label} correction bits"),
        );
        let high_sum_low = self.define(
            MslType::Ulong,
            vec![low_product, correction_bits],
            format!(
                "static_cast<ulong>({}.z) + static_cast<ulong>(static_cast<uint>({}))",
                self.name(low_product),
                self.name(correction_bits)
            ),
            &format!("{label} high sum low"),
        );
        let high_word_low = self.define(
            MslType::Uint,
            vec![high_sum_low],
            format!("static_cast<uint>({})", self.name(high_sum_low)),
            &format!("{label} high word low"),
        );
        let high_carry = self.define(
            MslType::Uint,
            vec![high_sum_low],
            format!("static_cast<uint>({} >> 32)", self.name(high_sum_low)),
            &format!("{label} high carry"),
        );
        let high_sum_high = self.define(
            MslType::Ulong,
            vec![low_product, correction_bits, high_carry],
            format!(
                "static_cast<ulong>({}.w) + static_cast<ulong>(static_cast<uint>({} >> 32)) + static_cast<ulong>({})",
                self.name(low_product),
                self.name(correction_bits),
                self.name(high_carry)
            ),
            &format!("{label} high sum high"),
        );
        let high_word_high = self.define(
            MslType::Uint,
            vec![high_sum_high],
            format!("static_cast<uint>({})", self.name(high_sum_high)),
            &format!("{label} high word high"),
        );
        self.define(
            MslType::Uint4,
            vec![low_product, high_word_low, high_word_high],
            format!(
                "uint4({product}.x, {product}.y, {low}, {high})",
                product = self.name(low_product),
                low = self.name(high_word_low),
                high = self.name(high_word_high)
            ),
            label,
        )
    }

    fn assert_source_edges_match_typed_graph(&self) {
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let tokens = source_tokens(&instruction.source);
            let definition = instruction.definition.map(|value| self.name(value).to_owned());
            let source_definition = if instruction.definition.is_some() {
                Some(tokens.first().expect("source definition token").clone())
            } else {
                None
            };
            assert_eq!(
                source_definition, definition,
                "typed/source definition mismatch at instruction {instruction_index}: {}",
                instruction.label
            );
            let source_uses: BTreeSet<_> = tokens
                .into_iter()
                .enumerate()
                .filter_map(|(index, token)| {
                    (source_definition.as_ref() != Some(&token) || index != 0).then_some(token)
                })
                .collect();
            let typed_uses: BTreeSet<_> =
                instruction.uses.iter().map(|&value| self.name(value).to_owned()).collect();
            assert_eq!(
                source_uses, typed_uses,
                "typed/source use mismatch at instruction {instruction_index}: {}",
                instruction.label
            );
        }
    }

    fn liveness(&self) -> LivenessResult {
        let mut last_use = BTreeMap::<Value, usize>::new();
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for value in &instruction.uses {
                last_use.insert(*value, instruction_index);
            }
        }
        for (index, info) in self.values.iter().enumerate() {
            assert!(last_use.contains_key(&Value(index)), "{} is never used", info.name);
        }

        let input_values: Vec<_> = self
            .values
            .iter()
            .enumerate()
            .filter_map(|(index, info)| info.definition.is_none().then_some(Value(index)))
            .collect();
        let mut peak_words: u32 =
            input_values.iter().map(|value| self.values[value.0].ty.words()).sum();
        let mut allocation_peak_words = peak_words;
        let mut peak_label = "typed function entry".to_owned();
        let mut peak_values =
            input_values.iter().map(|value| self.name(*value).to_owned()).collect();
        let mut live: BTreeSet<_> = input_values.into_iter().collect();
        let mut live_words = peak_words;
        let mut retire_at = vec![Vec::<Value>::new(); self.instructions.len()];
        for (&value, &last) in &last_use {
            retire_at[last].push(value);
        }

        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let definition_words =
                instruction.definition.map_or(0, |definition| self.values[definition.0].ty.words());
            allocation_peak_words = allocation_peak_words.max(live_words + definition_words);
            if live_words > peak_words {
                peak_words = live_words;
                peak_label =
                    format!("before instruction {instruction_index}: {}", instruction.label);
                peak_values = live.iter().map(|value| self.name(*value).to_owned()).collect();
            }
            if let Some(definition) = instruction.definition {
                assert!(live.insert(definition));
                live_words += definition_words;
            }
            for value in &retire_at[instruction_index] {
                assert!(live.remove(value));
                live_words -= self.values[value.0].ty.words();
            }
            if live_words > peak_words {
                peak_words = live_words;
                peak_label =
                    format!("after instruction {instruction_index}: {}", instruction.label);
                peak_values = live.iter().map(|value| self.name(*value).to_owned()).collect();
            }
        }
        LivenessResult { peak_words, allocation_peak_words, peak_label, peak_values }
    }
}

#[derive(Debug)]
struct TokenValueInfo {
    words: u32,
    definition: Option<usize>,
}

#[derive(Debug)]
struct TokenInstruction {
    definition: Option<String>,
    uses: Vec<String>,
    label: String,
}

fn recognized_token(token: &str) -> bool {
    matches!(token, "left_coeff" | "right_coeff" | "local_lane" | "subgroup_base")
        || token.strip_prefix('v').is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn source_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in line.chars().chain([' ']) {
        if character.is_ascii_alphanumeric() || character == '_' {
            current.push(character);
        } else if !current.is_empty() {
            if recognized_token(&current) {
                tokens.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    tokens
}

fn source_type(line: &str) -> Option<MslType> {
    let trimmed = line.trim_start();
    [
        MslType::Bool,
        MslType::Ushort,
        MslType::Uint,
        MslType::Int,
        MslType::Ulong,
        MslType::Long,
        MslType::Uint4,
    ]
    .into_iter()
    .find(|ty| trimmed.starts_with(&format!("const {} v", ty.source())))
}

fn tokenized_liveness(source: &str) -> LivenessResult {
    let mut values = BTreeMap::<String, TokenValueInfo>::from([
        ("left_coeff".to_owned(), TokenValueInfo { words: 2, definition: None }),
        ("right_coeff".to_owned(), TokenValueInfo { words: 2, definition: None }),
        ("local_lane".to_owned(), TokenValueInfo { words: 1, definition: None }),
        ("subgroup_base".to_owned(), TokenValueInfo { words: 1, definition: None }),
    ]);
    let mut instructions = Vec::<TokenInstruction>::new();
    for (line_number, line) in source.lines().enumerate() {
        let tokens = source_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        let ty = source_type(line);
        let definition = ty.map(|_| tokens[0].clone());
        if let (Some(name), Some(ty)) = (&definition, ty) {
            assert!(
                values
                    .insert(
                        name.clone(),
                        TokenValueInfo { words: ty.words(), definition: Some(instructions.len()) }
                    )
                    .is_none(),
                "duplicate emitted definition {name}"
            );
        }
        let uses: Vec<_> = tokens
            .into_iter()
            .enumerate()
            .filter_map(|(index, token)| {
                (definition.as_ref() != Some(&token) || index != 0).then_some(token)
            })
            .collect();
        for name in &uses {
            assert!(
                values.contains_key(name),
                "emitted source use before definition: {name} on line {}",
                line_number + 1
            );
        }
        instructions.push(TokenInstruction {
            definition,
            uses,
            label: format!("source line {}: {}", line_number + 1, line.trim()),
        });
    }

    let mut last_use = BTreeMap::<String, usize>::new();
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        for name in &instruction.uses {
            last_use.insert(name.clone(), instruction_index);
        }
    }
    for name in values.keys() {
        assert!(last_use.contains_key(name), "emitted value {name} is never used");
    }

    let mut peak_values: Vec<_> = values
        .iter()
        .filter_map(|(name, info)| info.definition.is_none().then_some(name.clone()))
        .collect();
    let mut peak_words: u32 = peak_values.iter().map(|name| values[name].words).sum();
    let mut allocation_peak_words = peak_words;
    let mut peak_label = "tokenized function entry".to_owned();
    let mut live: BTreeSet<_> = peak_values.iter().cloned().collect();
    let mut live_words = peak_words;
    let mut retire_at = vec![Vec::<String>::new(); instructions.len()];
    for (name, &last) in &last_use {
        retire_at[last].push(name.clone());
    }
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        let definition_words =
            instruction.definition.as_ref().map_or(0, |definition| values[definition].words);
        allocation_peak_words = allocation_peak_words.max(live_words + definition_words);
        if live_words > peak_words {
            peak_words = live_words;
            peak_label = format!("before {}", instruction.label);
            peak_values = live.iter().cloned().collect();
        }
        if let Some(definition) = &instruction.definition {
            assert!(live.insert(definition.clone()));
            live_words += definition_words;
        }
        for name in &retire_at[instruction_index] {
            assert!(live.remove(name));
            live_words -= values[name].words;
        }
        if live_words > peak_words {
            peak_words = live_words;
            peak_label = format!("after {}", instruction.label);
            peak_values = live.iter().cloned().collect();
        }
    }
    LivenessResult { peak_words, allocation_peak_words, peak_label, peak_values }
}

fn stable_fingerprint(source: &str, leaves: &[KaratsubaLeaf]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut consume = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in source.bytes() {
        consume(byte);
    }
    consume(0xff);
    for leaf in leaves {
        for coefficient in leaf.left {
            consume(coefficient as u8);
        }
        for coefficient in leaf.right {
            consume(coefficient as u8);
        }
        for weight in leaf.output_weights {
            consume(weight as u8);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

/// Emit the complete eight-lane callable plus stable flat-buffer ABI kernels.
pub(crate) fn generate() -> GeneratedSchedule {
    generate_variant(ScheduleVariant::UlongMultiplyV1)
}

/// Emit one frozen source-level widened-product variant.
pub(crate) fn generate_variant(variant: ScheduleVariant) -> GeneratedSchedule {
    let inverse_certificate = multiply_mod_e_wrapping(&M.map(|value| value as u64), &M_PRIME);
    assert_eq!(
        inverse_certificate,
        [u64::MAX, 0, 0, 0, 0, 0, 0, 0],
        "Mprime must equal -M^-1 modulo (2^64, X^8+2)"
    );
    assert_eq!(LOGICAL_WIDENED_PRODUCTS + LOGICAL_WRAPPING_PRODUCTS, LOGICAL_BASE_PRODUCTS);
    assert_eq!(
        ISSUED_WIDENED_PRODUCTS_PER_FIELD + ISSUED_WRAPPING_PRODUCTS_PER_FIELD,
        COEFFICIENTS as u32 * ISSUED_BASE_PRODUCTS_PER_LANE
    );

    let leaves = karatsuba_leaves();
    let mut emitter = Emitter::new(variant);
    let left_coeff = emitter.input("left_coeff", MslType::Long, "left coefficient in this lane");
    let right_coeff = emitter.input("right_coeff", MslType::Long, "right coefficient in this lane");
    let local_lane = emitter.input("local_lane", MslType::Ushort, "lane within AMNS field");
    let subgroup_base = emitter.input("subgroup_base", MslType::Ushort, "first lane of AMNS field");
    let left_bits = emitter.define(
        MslType::Ulong,
        vec![left_coeff],
        format!("as_type<ulong>({})", emitter.name(left_coeff)),
        "left coefficient bits",
    );
    let right_bits = emitter.define(
        MslType::Ulong,
        vec![right_coeff],
        format!("as_type<ulong>({})", emitter.name(right_coeff)),
        "right coefficient bits",
    );

    // C = A*B mod (X^8+2), exact signed 128-bit coefficients.
    let mut product = emitter.define(MslType::Uint4, Vec::new(), "uint4(0U)".to_owned(), "C zero");
    for round in 0..4 {
        let left_transform = emitter.exact_transform(
            left_bits,
            local_lane,
            subgroup_base,
            &leaves,
            round,
            OperandSide::Left,
            &format!("C round {round} left transform"),
        );
        let right_transform = emitter.exact_transform(
            right_bits,
            local_lane,
            subgroup_base,
            &leaves,
            round,
            OperandSide::Right,
            &format!("C round {round} right transform"),
        );
        let leaf_product = emitter.full_signed_i64_product(
            left_transform,
            right_transform,
            &format!("C round {round} leaf"),
        );
        for owner in 0..COEFFICIENTS {
            let Some(leaf) = leaves.get(round * COEFFICIENTS + owner) else {
                continue;
            };
            let gathered = emitter.shuffle_uint4(
                leaf_product,
                subgroup_base,
                owner,
                &format!("C round {round} owner {owner} gather"),
            );
            product = emitter.weighted_add128(
                product,
                gathered,
                packed_weights(&leaf.output_weights),
                local_lane,
                &format!("C round {round} owner {owner} sink"),
            );
        }
    }

    // Q = C*Mprime mod (2^64, X^8+2).
    let product_low = emitter.define(
        MslType::Ulong,
        vec![product],
        format!(
            "static_cast<ulong>({}.x) | (static_cast<ulong>({}.y) << 32)",
            emitter.name(product),
            emitter.name(product)
        ),
        "C low 64 bits",
    );
    let mut quotient = emitter.define(MslType::Ulong, Vec::new(), "0UL".to_owned(), "Q zero");
    for round in 0..4 {
        let c_transform = emitter.wrapping_transform(
            product_low,
            local_lane,
            subgroup_base,
            &leaves,
            round,
            OperandSide::Left,
            &format!("Q round {round} C transform"),
        );
        let mprime_transforms: [u64; 8] = std::array::from_fn(|owner| {
            leaves
                .get(round * COEFFICIENTS + owner)
                .map_or(0, |leaf| wrapping_leaf_transform(&leaf.right, &M_PRIME))
        });
        let mprime_transform = emitter.select_wrapping_constant(
            local_lane,
            &mprime_transforms,
            &format!("Q round {round} Mprime transform"),
        );
        let leaf_product = emitter.low_u64_product(
            c_transform,
            mprime_transform,
            &format!("Q round {round} leaf"),
        );
        for owner in 0..COEFFICIENTS {
            let Some(leaf) = leaves.get(round * COEFFICIENTS + owner) else {
                continue;
            };
            let gathered = emitter.shuffle_ulong(
                leaf_product,
                subgroup_base,
                owner,
                &format!("Q round {round} owner {owner} gather"),
            );
            quotient = emitter.weighted_add_u64(
                quotient,
                gathered,
                packed_weights(&leaf.output_weights),
                local_lane,
                &format!("Q round {round} owner {owner} sink"),
            );
        }
    }

    // D = Q*M mod (X^8+2).  A K8 quotient transform can exceed i64, so
    // split it exactly as low_signed + carry*2^64 with |carry| <= 4.
    let mut correction =
        emitter.define(MslType::Uint4, Vec::new(), "uint4(0U)".to_owned(), "D zero");
    for round in 0..4 {
        let (quotient_transform_low, quotient_transform_carry) = emitter.quotient_transform(
            quotient,
            local_lane,
            subgroup_base,
            &leaves,
            round,
            &format!("D round {round} Q transform"),
        );
        let m_transforms: [i64; 8] = std::array::from_fn(|owner| {
            leaves
                .get(round * COEFFICIENTS + owner)
                .map_or(0, |leaf| signed_leaf_transform(&leaf.right, &M))
        });
        assert!(m_transforms.iter().all(|value| value.unsigned_abs() <= 1_444_210_000_000_000));
        let m_transform = emitter.select_signed_constant(
            local_lane,
            &m_transforms,
            &format!("D round {round} M transform"),
        );
        let low_product = emitter.full_signed_i64_product(
            quotient_transform_low,
            m_transform,
            &format!("D round {round} low leaf"),
        );
        let leaf_product = emitter.add_q_carry_correction(
            low_product,
            quotient_transform_carry,
            m_transform,
            &format!("D round {round} carry correction"),
        );
        for owner in 0..COEFFICIENTS {
            let Some(leaf) = leaves.get(round * COEFFICIENTS + owner) else {
                continue;
            };
            let gathered = emitter.shuffle_uint4(
                leaf_product,
                subgroup_base,
                owner,
                &format!("D round {round} owner {owner} gather"),
            );
            correction = emitter.weighted_add128(
                correction,
                gathered,
                packed_weights(&leaf.output_weights),
                local_lane,
                &format!("D round {round} owner {owner} sink"),
            );
        }
    }

    let numerator = emitter.add128(product, correction, "C plus D numerator");
    let output = emitter.define(
        MslType::Long,
        vec![numerator],
        format!(
            "as_type<long>(static_cast<ulong>({}.z) | (static_cast<ulong>({}.w) << 32))",
            emitter.name(numerator),
            emitter.name(numerator)
        ),
        "exact division by 2^64",
    );
    emitter.marker(
        vec![output],
        format!("    return {};", emitter.name(output)),
        "returned AMNS coefficient",
    );

    assert_eq!(emitter.base_products, ISSUED_BASE_PRODUCTS_PER_LANE);
    assert_eq!(emitter.widened_products, 28);
    assert_eq!(emitter.wrapping_products, 8);
    emitter.assert_source_edges_match_typed_graph();
    let typed = emitter.liveness();
    let body = emitter.body_source();
    let tokenized = tokenized_liveness(&body);
    assert_eq!(typed.peak_words, tokenized.peak_words);
    assert_eq!(typed.allocation_peak_words, tokenized.allocation_peak_words);
    // Fingerprint the complete callable arithmetic body and the frozen leaf
    // graph.  ABI wrappers intentionally do not participate, allowing this
    // value to be embedded in the ABI report without a self-hash recursion.
    let fingerprint = stable_fingerprint(&body, &leaves);
    let fingerprint_word =
        u64::from_str_radix(fingerprint.strip_prefix("fnv1a64:").expect("fingerprint prefix"), 16)
            .expect("hexadecimal fingerprint");

    let mut source = String::new();
    writeln!(source, "// @generated by the standalone AMNS8 schedule generator; do not edit.")
        .expect("write generated header");
    writeln!(source, "#include <metal_stdlib>").expect("write Metal include");
    writeln!(source, "using namespace metal;").expect("write Metal namespace");
    writeln!(source).expect("write source separator");
    writeln!(
        source,
        "// E=X^8+2, phi=2^64. One coefficient occupies each lane of an 8-lane subwarp."
    )
    .expect("write arithmetic contract");
    writeln!(
        source,
        "// Logical graph: 27 leaves/conv, 189 widened + 54 wrapping u32 products/multiply."
    )
    .expect("write logical count");
    writeln!(
        source,
        "// SIMD-slot issue: 32 leaves/conv, 224 widened + 64 wrapping u32 products/field."
    )
    .expect("write issued count");
    writeln!(
        source,
        "// Input contract: every K8 transform fits long; Q-transform carry is in [-4,4]."
    )
    .expect("write range contract");
    writeln!(source, "inline __attribute__((always_inline))").expect("write attribute");
    writeln!(
        source,
        "long {}(long left_coeff, long right_coeff, ushort local_lane, ushort subgroup_base)",
        variant.callable_name()
    )
    .expect("write callable signature");
    writeln!(source, "{{").expect("write callable open");
    writeln!(source, "    // C+D is exactly divisible by 2^64; its low two uint words are zero.")
        .expect("write division contract");
    source.push_str(&body);
    writeln!(source, "}}").expect("write callable close");
    writeln!(source).expect("write source separator");
    writeln!(source, "struct {} {{", variant.params_name()).expect("write params open");
    writeln!(source, "    uint logical_fields;").expect("write params field");
    writeln!(source, "    uint iterations;").expect("write params field");
    writeln!(source, "    uint reserved0;").expect("write params field");
    writeln!(source, "    uint reserved1;").expect("write params field");
    writeln!(source, "}};").expect("write params close");
    writeln!(source).expect("write source separator");
    let abi_report = [
        0x414d_4e53_384d_534d_u64,   // "AMNS8MSM"
        1,                           // ABI report version
        28,                          // report words
        64,                          // flat field bytes
        8,                           // flat field alignment
        16,                          // params bytes
        4,                           // params alignment
        0,                           // params.logical_fields offset
        4,                           // params.iterations offset
        8,                           // params.reserved0 offset
        12,                          // params.reserved1 offset
        8,                           // threads per field
        8,                           // coefficients per field
        64,                          // coefficient bits
        variant.algorithm_version(), // algorithm version
        u64::from(LEAVES_PER_CONVOLUTION),
        u64::from(LOGICAL_WIDENED_PRODUCTS),
        u64::from(LOGICAL_WRAPPING_PRODUCTS),
        u64::from(LOGICAL_BASE_PRODUCTS),
        u64::from(ISSUED_LEAF_SLOTS_PER_CONVOLUTION),
        u64::from(ISSUED_WIDENED_PRODUCTS_PER_FIELD),
        u64::from(ISSUED_WRAPPING_PRODUCTS_PER_FIELD),
        u64::from(ISSUED_WIDENED_PRODUCTS_PER_FIELD + ISSUED_WRAPPING_PRODUCTS_PER_FIELD),
        u64::from(typed.peak_words),
        u64::from(tokenized.peak_words),
        u64::from(typed.allocation_peak_words),
        u64::from(tokenized.allocation_peak_words),
        fingerprint_word,
    ];
    let mut abi_value_expression = format!("0x{:016x}UL", abi_report[0]);
    for (index, value) in abi_report.iter().copied().enumerate().skip(1) {
        abi_value_expression =
            format!("select({abi_value_expression}, 0x{value:016x}UL, report_index == {index}U)");
    }
    writeln!(source, "// ABI report words: magic, ABI/version/size/alignment/offsets, topology,")
        .expect("write ABI report legend");
    writeln!(source, "// logical and issued counts, typed/tokenized liveness peaks, fingerprint.")
        .expect("write ABI report legend");
    writeln!(source, "kernel void {}(", variant.abi_kernel_name()).expect("write ABI report name");
    writeln!(source, "    device ulong* output [[buffer(0)]],").expect("write ABI report output");
    writeln!(source, "    uint report_index [[thread_position_in_grid]])")
        .expect("write ABI report index");
    writeln!(source, "{{").expect("write ABI report open");
    writeln!(source, "    const ulong value = {abi_value_expression};")
        .expect("write ABI report selection");
    writeln!(source, "    output[report_index] = value;").expect("write ABI report store");
    writeln!(source, "}}").expect("write ABI report close");
    writeln!(source).expect("write source separator");
    writeln!(source, "kernel void {}(", variant.multiply_kernel_name())
        .expect("write multiply name");
    writeln!(source, "    device const long* left [[buffer(0)]],").expect("write ABI left");
    writeln!(source, "    device const long* right [[buffer(1)]],").expect("write ABI right");
    writeln!(source, "    device long* output [[buffer(2)]],").expect("write ABI output");
    writeln!(source, "    constant {}& params [[buffer(3)]],", variant.params_name())
        .expect("write ABI params");
    writeln!(source, "    uint coefficient_index [[thread_position_in_grid]],")
        .expect("write ABI grid index");
    writeln!(source, "    ushort simd_lane [[thread_index_in_simdgroup]])")
        .expect("write ABI SIMD lane");
    writeln!(source, "{{").expect("write ABI open");
    writeln!(source, "    const uint field_index = coefficient_index >> 3;")
        .expect("write ABI field index");
    writeln!(source, "    const ushort local_lane = simd_lane & 7U;")
        .expect("write ABI local lane");
    writeln!(source, "    const ushort subgroup_base = simd_lane & static_cast<ushort>(~7U);")
        .expect("write ABI subgroup base");
    writeln!(source, "    const long result = {}(left[coefficient_index], right[coefficient_index], local_lane, subgroup_base);", variant.callable_name())
        .expect("write ABI call");
    writeln!(source, "    // Dispatch a padded physical field count divisible by four; guard only the final store.")
        .expect("write ABI padding contract");
    writeln!(source, "    if (field_index < params.logical_fields) {{")
        .expect("write ABI store guard");
    writeln!(source, "        output[coefficient_index] = result;").expect("write ABI store");
    writeln!(source, "    }}").expect("write ABI store close");
    writeln!(source, "}}").expect("write ABI close");
    writeln!(source).expect("write source separator");
    writeln!(source, "kernel void {}(", variant.rotating_kernel_name()).expect("write chain name");
    writeln!(source, "    device const long* left [[buffer(0)]],").expect("write chain left");
    writeln!(source, "    device const long* right [[buffer(1)]],").expect("write chain right");
    writeln!(source, "    device long* output [[buffer(2)]],").expect("write chain output");
    writeln!(source, "    constant {}& params [[buffer(3)]],", variant.params_name())
        .expect("write chain params");
    writeln!(source, "    uint coefficient_index [[thread_position_in_grid]],")
        .expect("write chain grid index");
    writeln!(source, "    ushort simd_lane [[thread_index_in_simdgroup]])")
        .expect("write chain SIMD lane");
    writeln!(source, "{{").expect("write chain open");
    writeln!(source, "    const uint field_index = coefficient_index >> 3;")
        .expect("write chain field index");
    writeln!(source, "    const ushort local_lane = simd_lane & 7U;")
        .expect("write chain local lane");
    writeln!(source, "    const ushort subgroup_base = simd_lane & static_cast<ushort>(~7U);")
        .expect("write chain subgroup base");
    writeln!(source, "    long chain_left = left[coefficient_index];")
        .expect("write chain left state");
    writeln!(source, "    long chain_right = right[coefficient_index];")
        .expect("write chain right state");
    writeln!(
        source,
        "    for (uint iteration = 0U; iteration < params.iterations; ++iteration) {{"
    )
    .expect("write public-count chain loop");
    writeln!(
        source,
        "        const long product = {}(chain_left, chain_right, local_lane, subgroup_base);",
        variant.callable_name()
    )
    .expect("write chain multiply");
    writeln!(source, "        chain_left = chain_right;").expect("write chain left update");
    writeln!(source, "        chain_right = product;").expect("write chain right update");
    writeln!(source, "    }}").expect("write chain loop close");
    writeln!(source, "    if (field_index < params.logical_fields) {{")
        .expect("write chain store guard");
    writeln!(source, "        output[coefficient_index] = chain_right;")
        .expect("write chain store");
    writeln!(source, "    }}").expect("write chain store close");
    writeln!(source, "}}").expect("write chain close");

    assert_eq!(source.matches("@base-u32-product").count() as u32, emitter.widened_products);
    assert_eq!(source.matches("@wrapping-u32-product").count() as u32, emitter.wrapping_products);
    GeneratedSchedule {
        source,
        fingerprint,
        leaves,
        logical_leaves_per_convolution: LEAVES_PER_CONVOLUTION,
        logical_base_products: LOGICAL_BASE_PRODUCTS,
        logical_widened_products: LOGICAL_WIDENED_PRODUCTS,
        logical_wrapping_products: LOGICAL_WRAPPING_PRODUCTS,
        issued_leaf_slots_per_convolution: ISSUED_LEAF_SLOTS_PER_CONVOLUTION,
        issued_base_products_per_lane: ISSUED_BASE_PRODUCTS_PER_LANE,
        issued_widened_products_per_field: ISSUED_WIDENED_PRODUCTS_PER_FIELD,
        issued_wrapping_products_per_field: ISSUED_WRAPPING_PRODUCTS_PER_FIELD,
        emitted_base_product_instructions: emitter.base_products,
        emitted_widened_product_instructions: emitter.widened_products,
        emitted_wrapping_product_instructions: emitter.wrapping_products,
        typed_peak_live_words: typed.peak_words,
        tokenized_peak_live_words: tokenized.peak_words,
        typed_allocation_peak_live_words: typed.allocation_peak_words,
        tokenized_allocation_peak_live_words: tokenized.allocation_peak_words,
        typed_peak_label: typed.peak_label,
        tokenized_peak_label: tokenized.peak_label,
        typed_peak_values: typed.peak_values,
        tokenized_peak_values: tokenized.peak_values,
    }
}
