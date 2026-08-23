// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Generated, flattened radix-2^128 Montgomery schedule support.
//!
//! This module is included by `build.rs` and by a focused integration test. It
//! deliberately has no dependency on the application or on a Metal runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

const FIELD_LIMBS: usize = 12;
const BLOCK_LIMBS: usize = 4;
const ACCUMULATOR_LIMBS: usize = FIELD_LIMBS + BLOCK_LIMBS;
const ROUNDS: usize = FIELD_LIMBS / BLOCK_LIMBS;

#[derive(Debug)]
pub(crate) struct GeneratedSchedule {
    pub(crate) source: String,
    pub(crate) base_products: u32,
    pub(crate) typed_peak_live_words: u32,
    pub(crate) tokenized_peak_live_words: u32,
    pub(crate) typed_allocation_peak_live_words: u32,
    pub(crate) tokenized_allocation_peak_live_words: u32,
    pub(crate) typed_peak_label: String,
    pub(crate) typed_peak_values: Vec<String>,
    pub(crate) typed_peak_descriptions: Vec<String>,
    pub(crate) tokenized_peak_label: String,
    pub(crate) tokenized_peak_values: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Value(usize);

#[derive(Clone, Copy, Debug)]
enum Word {
    Value(Value),
    Literal(u32),
}

impl From<Value> for Word {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

#[derive(Debug)]
struct ValueInfo {
    name: String,
    words: u32,
    definition: Option<usize>,
    label: String,
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
    peak_label: String,
    peak_values: Vec<String>,
    allocation_peak_words: u32,
}

struct SsaEmitter {
    values: Vec<ValueInfo>,
    instructions: Vec<Instruction>,
    base_products: u32,
}

impl SsaEmitter {
    fn new() -> Self {
        Self { values: Vec::new(), instructions: Vec::new(), base_products: 0 }
    }

    fn input(&mut self, name: String, label: String) -> Value {
        let value = Value(self.values.len());
        self.values.push(ValueInfo { name, words: 1, definition: None, label });
        value
    }

    fn name(&self, value: Value) -> &str {
        &self.values[value.0].name
    }

    fn word_expression(&self, word: Word) -> String {
        match word {
            Word::Value(value) => self.name(value).to_owned(),
            Word::Literal(value) => format!("{value:#010x}U"),
        }
    }

    fn word_uses(words: &[Word]) -> Vec<Value> {
        words
            .iter()
            .filter_map(|word| match *word {
                Word::Value(value) => Some(value),
                Word::Literal(_) => None,
            })
            .collect()
    }

    fn define(
        &mut self,
        words: u32,
        ty: &str,
        uses: Vec<Value>,
        expression: String,
        label: &str,
    ) -> Value {
        assert!(words == 1 || words == 2);
        assert_eq!(words, if ty == "ulong" { 2 } else { 1 });
        let value = Value(self.values.len());
        let name = format!("s{}", value.0);
        let instruction_index = self.instructions.len();
        let source = format!("    const {ty} {name} = {expression};");
        self.values.push(ValueInfo {
            name,
            words,
            definition: Some(instruction_index),
            label: label.to_owned(),
        });
        self.instructions.push(Instruction {
            definition: Some(value),
            uses,
            source,
            label: label.to_owned(),
        });
        value
    }

    fn define_uint(&mut self, uses: Vec<Value>, expression: String, label: &str) -> Value {
        self.define(1, "uint", uses, expression, label)
    }

    fn define_ulong(&mut self, uses: Vec<Value>, expression: String, label: &str) -> Value {
        self.define(2, "ulong", uses, expression, label)
    }

    fn marker(&mut self, uses: Vec<Value>, source: String, label: &str) {
        self.instructions.push(Instruction {
            definition: None,
            uses,
            source,
            label: label.to_owned(),
        });
    }

    fn output(&mut self, index: usize, source_value: Value) -> Value {
        let value = Value(self.values.len());
        let name = format!("result.x{index}");
        let instruction_index = self.instructions.len();
        self.values.push(ValueInfo {
            name: name.clone(),
            words: 1,
            definition: Some(instruction_index),
            label: format!("result word {index}"),
        });
        self.instructions.push(Instruction {
            definition: Some(value),
            uses: vec![source_value],
            source: format!("    {name} = {};", self.name(source_value)),
            label: format!("result word {index}"),
        });
        value
    }

    fn output_word(&mut self, index: usize, source_word: Word) -> Value {
        match source_word {
            Word::Value(value) => self.output(index, value),
            Word::Literal(literal) => {
                let value = Value(self.values.len());
                let name = format!("result.x{index}");
                let instruction_index = self.instructions.len();
                self.values.push(ValueInfo {
                    name: name.clone(),
                    words: 1,
                    definition: Some(instruction_index),
                    label: format!("result word {index}"),
                });
                self.instructions.push(Instruction {
                    definition: Some(value),
                    uses: Vec::new(),
                    source: format!("    {name} = {literal:#010x}U;"),
                    label: format!("result word {index}"),
                });
                value
            }
        }
    }

    fn multiply_pair(&mut self, lhs: Word, rhs: Word, label: &str) -> [Value; 2] {
        let lhs_expression = self.word_expression(lhs);
        let rhs_expression = self.word_expression(rhs);
        let uses = Self::word_uses(&[lhs, rhs]);
        let product = self.define_ulong(
            uses,
            format!("static_cast<ulong>({lhs_expression}) * static_cast<ulong>({rhs_expression})"),
            &format!("{label} product"),
        );
        self.base_products += 1;
        let low = self.define_uint(
            vec![product],
            format!("static_cast<uint>({})", self.name(product)),
            &format!("{label} low"),
        );
        let high = self.define_uint(
            vec![product],
            format!("static_cast<uint>({} >> 32)", self.name(product)),
            &format!("{label} high"),
        );
        [low, high]
    }

    fn multiply_low_word(&mut self, lhs: Word, rhs: Word, label: &str) -> Value {
        let lhs_expression = self.word_expression(lhs);
        let rhs_expression = self.word_expression(rhs);
        let uses = Self::word_uses(&[lhs, rhs]);
        let product = self.define_ulong(
            uses,
            format!("static_cast<ulong>({lhs_expression}) * static_cast<ulong>({rhs_expression})"),
            &format!("{label} product"),
        );
        self.base_products += 1;
        self.define_uint(
            vec![product],
            format!("static_cast<uint>({product_name})", product_name = self.name(product)),
            &format!("{label} low"),
        )
    }

    fn absolute_difference_word(&mut self, lhs: Word, rhs: Word, label: &str) -> (Value, Value) {
        let lhs_expression = self.word_expression(lhs);
        let rhs_expression = self.word_expression(rhs);
        let uses = Self::word_uses(&[lhs, rhs]);
        let forward = self.define_uint(
            uses.clone(),
            format!("{lhs_expression} - {rhs_expression}"),
            &format!("{label} forward"),
        );
        let borrow = self.define_uint(
            uses,
            format!("static_cast<uint>({lhs_expression} < {rhs_expression})"),
            &format!("{label} borrow"),
        );
        let mask = self.define_uint(
            vec![borrow],
            format!("0U - {}", self.name(borrow)),
            &format!("{label} mask"),
        );
        let magnitude = self.define_uint(
            vec![forward, mask, borrow],
            format!("({} ^ {}) + {}", self.name(forward), self.name(mask), self.name(borrow)),
            &format!("{label} magnitude"),
        );
        let nonnegative = self.define_uint(
            vec![borrow],
            format!("{} ^ 1U", self.name(borrow)),
            &format!("{label} nonnegative"),
        );
        (magnitude, nonnegative)
    }

    fn absolute_difference_word_or_literal(
        &mut self,
        lhs: Word,
        rhs: Word,
        label: &str,
    ) -> (Word, Word) {
        match (lhs, rhs) {
            (Word::Literal(lhs), Word::Literal(rhs)) => {
                if lhs >= rhs {
                    (Word::Literal(lhs - rhs), Word::Literal(1))
                } else {
                    (Word::Literal(rhs - lhs), Word::Literal(0))
                }
            }
            _ => {
                let (magnitude, nonnegative) = self.absolute_difference_word(lhs, rhs, label);
                (Word::Value(magnitude), Word::Value(nonnegative))
            }
        }
    }

    fn absolute_difference_2(
        &mut self,
        lhs: [Word; 2],
        rhs: [Word; 2],
        label: &str,
    ) -> ([Value; 2], Value) {
        let mut forward = Vec::with_capacity(2);
        let mut incoming_borrow: Option<Value> = None;
        for index in 0..2 {
            let lhs_expression = self.word_expression(lhs[index]);
            let rhs_expression = self.word_expression(rhs[index]);
            let first = self.define_uint(
                Self::word_uses(&[lhs[index], rhs[index]]),
                format!("{lhs_expression} - {rhs_expression}"),
                &format!("{label} first {index}"),
            );
            let mut uses = vec![first];
            if let Some(borrow) = incoming_borrow {
                uses.push(borrow);
            }
            let difference = self.define_uint(
                uses,
                match incoming_borrow {
                    Some(borrow) => format!("{} - {}", self.name(first), self.name(borrow)),
                    None => self.name(first).to_owned(),
                },
                &format!("{label} forward {index}"),
            );
            let first_borrow = self.define_uint(
                Self::word_uses(&[lhs[index], rhs[index]]),
                format!("static_cast<uint>({lhs_expression} < {rhs_expression})"),
                &format!("{label} first borrow {index}"),
            );
            let borrow = if let Some(incoming) = incoming_borrow {
                let second_borrow = self.define_uint(
                    vec![first, incoming],
                    format!("static_cast<uint>({} < {})", self.name(first), self.name(incoming)),
                    &format!("{label} second borrow {index}"),
                );
                self.define_uint(
                    vec![first_borrow, second_borrow],
                    format!("{} | {}", self.name(first_borrow), self.name(second_borrow)),
                    &format!("{label} borrow {index}"),
                )
            } else {
                first_borrow
            };
            forward.push(difference);
            incoming_borrow = Some(borrow);
        }
        let final_borrow = incoming_borrow.expect("two-word difference has a final borrow");
        let mask = self.define_uint(
            vec![final_borrow],
            format!("0U - {}", self.name(final_borrow)),
            &format!("{label} mask"),
        );
        let nonnegative = self.define_uint(
            vec![final_borrow],
            format!("{} ^ 1U", self.name(final_borrow)),
            &format!("{label} nonnegative"),
        );

        let mut magnitude = Vec::with_capacity(2);
        let mut carry = final_borrow;
        for (index, forward_word) in forward.into_iter().enumerate() {
            let sum = self.define_ulong(
                vec![forward_word, mask, carry],
                format!(
                    "static_cast<ulong>({} ^ {}) + static_cast<ulong>({})",
                    self.name(forward_word),
                    self.name(mask),
                    self.name(carry)
                ),
                &format!("{label} magnitude sum {index}"),
            );
            let low = self.define_uint(
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} magnitude {index}"),
            );
            magnitude.push(low);
            if index == 0 {
                carry = self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} magnitude carry"),
                );
            }
        }
        (magnitude.try_into().expect("two-word magnitude"), nonnegative)
    }

    fn signed_sum<const N: usize>(
        &mut self,
        lhs: [Value; N],
        rhs: [Value; N],
        middle: [Value; N],
        negative: Value,
        label: &str,
    ) -> Vec<Value> {
        let mask = self.define_uint(
            vec![negative],
            format!("0U - {}", self.name(negative)),
            &format!("{label} sign mask"),
        );
        let mut result = Vec::with_capacity(N + 1);
        let mut carry = negative;
        for index in 0..N {
            let sum = self.define_ulong(
                vec![lhs[index], rhs[index], middle[index], mask, carry],
                format!(
                    "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({} ^ {}) + static_cast<ulong>({})",
                    self.name(lhs[index]),
                    self.name(rhs[index]),
                    self.name(middle[index]),
                    self.name(mask),
                    self.name(carry)
                ),
                &format!("{label} sum {index}"),
            );
            result.push(self.define_uint(
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} word {index}"),
            ));
            carry = self.define_uint(
                vec![sum],
                format!("static_cast<uint>({} >> 32)", self.name(sum)),
                &format!("{label} carry {index}"),
            );
        }
        let top = self.define_ulong(
            vec![mask, carry],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(mask),
                self.name(carry)
            ),
            &format!("{label} top sum"),
        );
        result.push(self.define_uint(
            vec![top],
            format!("static_cast<uint>({})", self.name(top)),
            &format!("{label} top"),
        ));
        result
    }

    fn signed_sum_low<const N: usize>(
        &mut self,
        lhs: [Value; N],
        rhs: [Value; N],
        middle: [Value; N],
        negative: Value,
        label: &str,
    ) -> [Value; N] {
        let mask = self.define_uint(
            vec![negative],
            format!("0U - {}", self.name(negative)),
            &format!("{label} sign mask"),
        );
        let mut result = Vec::with_capacity(N);
        let mut carry = negative;
        for index in 0..N {
            let sum = self.define_ulong(
                vec![lhs[index], rhs[index], middle[index], mask, carry],
                format!(
                    "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({} ^ {}) + static_cast<ulong>({})",
                    self.name(lhs[index]),
                    self.name(rhs[index]),
                    self.name(middle[index]),
                    self.name(mask),
                    self.name(carry)
                ),
                &format!("{label} sum {index}"),
            );
            result.push(self.define_uint(
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} word {index}"),
            ));
            if index + 1 < N {
                carry = self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} carry {index}"),
                );
            }
        }
        result.try_into().expect("fixed-width truncated signed sum")
    }

    fn full_k2(&mut self, lhs: [Word; 2], rhs: [Word; 2], label: &str) -> [Value; 4] {
        let z0 = self.multiply_pair(lhs[0], rhs[0], &format!("{label} z0"));
        let z2 = self.multiply_pair(lhs[1], rhs[1], &format!("{label} z2"));
        let (lhs_difference, lhs_nonnegative) =
            self.absolute_difference_word_or_literal(lhs[0], lhs[1], &format!("{label} lhs diff"));
        let (rhs_difference, rhs_nonnegative) =
            self.absolute_difference_word_or_literal(rhs[1], rhs[0], &format!("{label} rhs diff"));
        let middle = self.multiply_pair(lhs_difference, rhs_difference, &format!("{label} middle"));
        let mut negative_uses = Self::word_uses(&[lhs_nonnegative, rhs_nonnegative]);
        negative_uses.sort_unstable();
        negative_uses.dedup();
        let negative = self.define_uint(
            negative_uses,
            format!(
                "{} ^ {}",
                self.word_expression(lhs_nonnegative),
                self.word_expression(rhs_nonnegative)
            ),
            &format!("{label} middle sign"),
        );
        let cross = self.signed_sum(z0, z2, middle, negative, &format!("{label} cross"));
        assert_eq!(cross.len(), 3);

        let sum1 = self.define_ulong(
            vec![z0[1], cross[0]],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(z0[1]),
                self.name(cross[0])
            ),
            &format!("{label} result sum 1"),
        );
        let result1 = self.define_uint(
            vec![sum1],
            format!("static_cast<uint>({})", self.name(sum1)),
            &format!("{label} result 1"),
        );
        let carry1 = self.define_uint(
            vec![sum1],
            format!("static_cast<uint>({} >> 32)", self.name(sum1)),
            &format!("{label} result carry 1"),
        );
        let sum2 = self.define_ulong(
            vec![cross[1], z2[0], carry1],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(cross[1]),
                self.name(z2[0]),
                self.name(carry1)
            ),
            &format!("{label} result sum 2"),
        );
        let result2 = self.define_uint(
            vec![sum2],
            format!("static_cast<uint>({})", self.name(sum2)),
            &format!("{label} result 2"),
        );
        let carry2 = self.define_uint(
            vec![sum2],
            format!("static_cast<uint>({} >> 32)", self.name(sum2)),
            &format!("{label} result carry 2"),
        );
        let sum3 = self.define_ulong(
            vec![cross[2], z2[1], carry2],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(cross[2]),
                self.name(z2[1]),
                self.name(carry2)
            ),
            &format!("{label} result sum 3"),
        );
        let result3 = self.define_uint(
            vec![sum3],
            format!("static_cast<uint>({})", self.name(sum3)),
            &format!("{label} result 3"),
        );
        [z0[0], result1, result2, result3]
    }

    fn low_k2(&mut self, lhs: [Word; 2], rhs: [Word; 2], label: &str) -> [Value; 2] {
        let z0 = self.multiply_pair(lhs[0], rhs[0], &format!("{label} z0"));
        let cross0 = self.multiply_low_word(lhs[0], rhs[1], &format!("{label} cross 0"));
        let cross1 = self.multiply_low_word(lhs[1], rhs[0], &format!("{label} cross 1"));
        let sum = self.define_ulong(
            vec![z0[1], cross0, cross1],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(z0[1]),
                self.name(cross0),
                self.name(cross1)
            ),
            &format!("{label} high sum"),
        );
        let high = self.define_uint(
            vec![sum],
            format!("static_cast<uint>({})", self.name(sum)),
            &format!("{label} high"),
        );
        [z0[0], high]
    }

    fn low_k4(&mut self, lhs: [Word; 4], rhs: [Word; 4], label: &str) -> [Value; 4] {
        let products_before = self.base_products;
        let z0 = self.full_k2([lhs[0], lhs[1]], [rhs[0], rhs[1]], &format!("{label} z0"));
        let z2 = self.low_k2([lhs[2], lhs[3]], [rhs[2], rhs[3]], &format!("{label} z2 low"));
        let (lhs_difference, lhs_nonnegative) = self.absolute_difference_2(
            [lhs[0], lhs[1]],
            [lhs[2], lhs[3]],
            &format!("{label} lhs diff"),
        );
        let (rhs_difference, rhs_nonnegative) = self.absolute_difference_2(
            [rhs[2], rhs[3]],
            [rhs[0], rhs[1]],
            &format!("{label} rhs diff"),
        );
        let middle = self.low_k2(
            lhs_difference.map(Word::Value),
            rhs_difference.map(Word::Value),
            &format!("{label} middle low"),
        );
        let negative = self.define_uint(
            vec![lhs_nonnegative, rhs_nonnegative],
            format!("{} ^ {}", self.name(lhs_nonnegative), self.name(rhs_nonnegative)),
            &format!("{label} middle sign"),
        );
        let cross = self.signed_sum_low(
            z0[..2].try_into().expect("two z0 low words"),
            z2,
            middle,
            negative,
            &format!("{label} cross low"),
        );

        let sum2 = self.define_ulong(
            vec![z0[2], cross[0]],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(z0[2]),
                self.name(cross[0])
            ),
            &format!("{label} result sum 2"),
        );
        let result2 = self.define_uint(
            vec![sum2],
            format!("static_cast<uint>({})", self.name(sum2)),
            &format!("{label} result 2"),
        );
        let carry2 = self.define_uint(
            vec![sum2],
            format!("static_cast<uint>({} >> 32)", self.name(sum2)),
            &format!("{label} carry 2"),
        );
        let sum3 = self.define_ulong(
            vec![z0[3], cross[1], carry2],
            format!(
                "static_cast<ulong>({}) + static_cast<ulong>({}) + static_cast<ulong>({})",
                self.name(z0[3]),
                self.name(cross[1]),
                self.name(carry2)
            ),
            &format!("{label} result sum 3"),
        );
        let result3 = self.define_uint(
            vec![sum3],
            format!("static_cast<uint>({})", self.name(sum3)),
            &format!("{label} result 3"),
        );
        assert_eq!(self.base_products - products_before, 9);
        [z0[0], z0[1], result2, result3]
    }

    fn full_k4_add_into(
        &mut self,
        accumulator: &mut [Word; ACCUMULATOR_LIMBS],
        offset: usize,
        discarded_low_product_words: usize,
        lhs: [Word; 4],
        rhs: [Word; 4],
        label: &str,
    ) {
        assert!(offset + 8 <= ACCUMULATOR_LIMBS);
        assert!(discarded_low_product_words == 0 || discarded_low_product_words == BLOCK_LIMBS);
        let cancellation_block = discarded_low_product_words == BLOCK_LIMBS;
        assert!(!cancellation_block || offset == 0);
        let products_before = self.base_products;
        let z0 = self.full_k2([lhs[0], lhs[1]], [rhs[0], rhs[1]], &format!("{label} z0"));
        let z2 = self.full_k2([lhs[2], lhs[3]], [rhs[2], rhs[3]], &format!("{label} z2"));
        let (lhs_difference, lhs_nonnegative) = self.absolute_difference_2(
            [lhs[0], lhs[1]],
            [lhs[2], lhs[3]],
            &format!("{label} lhs diff"),
        );
        let rhs_literals = rhs.map(|word| match word {
            Word::Literal(value) => Some(value),
            Word::Value(_) => None,
        });
        let (rhs_difference, rhs_nonnegative): ([Word; 2], Word) =
            if let [Some(r0), Some(r1), Some(r2), Some(r3)] = rhs_literals {
                let low = u64::from(r0) | (u64::from(r1) << 32);
                let high = u64::from(r2) | (u64::from(r3) << 32);
                let (magnitude, nonnegative) =
                    if high >= low { (high - low, 1_u32) } else { (low - high, 0_u32) };
                (
                    [Word::Literal(magnitude as u32), Word::Literal((magnitude >> 32) as u32)],
                    Word::Literal(nonnegative),
                )
            } else {
                let (magnitude, nonnegative) = self.absolute_difference_2(
                    [rhs[2], rhs[3]],
                    [rhs[0], rhs[1]],
                    &format!("{label} rhs diff"),
                );
                (magnitude.map(Word::Value), Word::Value(nonnegative))
            };
        let middle = self.full_k2(
            lhs_difference.map(Word::Value),
            rhs_difference,
            &format!("{label} middle"),
        );
        let mut negative_uses = vec![lhs_nonnegative];
        if let Word::Value(value) = rhs_nonnegative {
            negative_uses.push(value);
        }
        let negative = self.define_uint(
            negative_uses,
            format!("{} ^ {}", self.name(lhs_nonnegative), self.word_expression(rhs_nonnegative)),
            &format!("{label} middle sign"),
        );
        let cross = self.signed_sum(z0, z2, middle, negative, &format!("{label} cross"));
        assert_eq!(cross.len(), 5);

        let terms: [Vec<Value>; 8] = [
            vec![z0[0]],
            vec![z0[1]],
            vec![z0[2], cross[0]],
            vec![z0[3], cross[1]],
            vec![cross[2], z2[0]],
            vec![cross[3], z2[1]],
            vec![cross[4], z2[2]],
            vec![z2[3]],
        ];
        let mut carry: Option<Value> = None;
        let mut low_or: Option<Value> = None;
        let mut cancellation_carry: Option<Value> = None;
        for (word, product_terms) in terms.into_iter().enumerate() {
            let destination = offset + word;
            let omits_cancelled_low_accumulator = cancellation_block && word < BLOCK_LIMBS;
            let mut uses = if omits_cancelled_low_accumulator {
                Vec::new()
            } else {
                Self::word_uses(&[accumulator[destination]])
            };
            uses.extend(product_terms.iter().copied());
            if let Some(value) = carry {
                uses.push(value);
            }
            if word == BLOCK_LIMBS
                && let Some(value) = cancellation_carry
            {
                uses.push(value);
            }
            let mut expression = if omits_cancelled_low_accumulator {
                "0UL".to_owned()
            } else {
                format!("static_cast<ulong>({})", self.word_expression(accumulator[destination]))
            };
            for term in product_terms {
                write!(expression, " + static_cast<ulong>({})", self.name(term))
                    .expect("write K4 sink term");
            }
            if let Some(value) = carry {
                write!(expression, " + static_cast<ulong>({})", self.name(value))
                    .expect("write K4 sink carry");
            }
            if word == BLOCK_LIMBS
                && let Some(value) = cancellation_carry
            {
                write!(expression, " + static_cast<ulong>({})", self.name(value))
                    .expect("write K4 cancellation carry");
            }
            let sum = self.define_ulong(uses, expression, &format!("{label} sink sum {word}"));
            if omits_cancelled_low_accumulator {
                let low = self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({})", self.name(sum)),
                    &format!("{label} cancelled low product word {word}"),
                );
                low_or = Some(if let Some(previous) = low_or {
                    self.define_uint(
                        vec![previous, low],
                        format!("{} | {}", self.name(previous), self.name(low)),
                        &format!("{label} cancelled low product OR {word}"),
                    )
                } else {
                    low
                });
                if word + 1 == BLOCK_LIMBS {
                    let combined = low_or.expect("cancelled low product has an OR value");
                    cancellation_carry = Some(self.define_uint(
                        vec![combined],
                        format!("static_cast<uint>({} != 0U)", self.name(combined)),
                        &format!("{label} cancellation carry"),
                    ));
                    for low_word in accumulator.iter_mut().take(BLOCK_LIMBS) {
                        *low_word = Word::Literal(0);
                    }
                }
            } else {
                accumulator[destination] = Word::Value(self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({})", self.name(sum)),
                    &format!("{label} sink word {word}"),
                ));
            }
            carry = (word + 1 < 8 || offset + 8 < ACCUMULATOR_LIMBS).then(|| {
                self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} sink carry {word}"),
                )
            });
        }
        assert!(!cancellation_block || cancellation_carry.is_some());
        let mut destination = offset + 8;
        while destination < ACCUMULATOR_LIMBS {
            let incoming = carry.expect("K4 sink has a carry");
            if let Word::Literal(literal) = accumulator[destination] {
                assert_eq!(literal, 0, "only known-zero accumulator literals are supported");
                accumulator[destination] = Word::Value(incoming);
                carry = None;
                break;
            }
            let old_accumulator = match accumulator[destination] {
                Word::Value(value) => value,
                Word::Literal(_) => unreachable!("literal handled above"),
            };
            let sum = self.define_ulong(
                vec![old_accumulator, incoming],
                format!(
                    "static_cast<ulong>({}) + static_cast<ulong>({})",
                    self.name(old_accumulator),
                    self.name(incoming)
                ),
                &format!("{label} carry propagation sum {destination}"),
            );
            accumulator[destination] = Word::Value(self.define_uint(
                vec![sum],
                format!("static_cast<uint>({})", self.name(sum)),
                &format!("{label} carry propagation word {destination}"),
            ));
            if destination + 1 < ACCUMULATOR_LIMBS {
                carry = Some(self.define_uint(
                    vec![sum],
                    format!("static_cast<uint>({} >> 32)", self.name(sum)),
                    &format!("{label} carry propagation {destination}"),
                ));
            } else {
                carry = None;
            }
            destination += 1;
        }
        assert!(
            carry.is_none() || offset + 8 == ACCUMULATOR_LIMBS,
            "{label}: every nonfinal carry must be consumed or stored in a known-zero word"
        );
        assert_eq!(self.base_products - products_before, 9);
    }

    fn liveness(&self) -> LivenessResult {
        let mut last_use = vec![None; self.values.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for &value in &instruction.uses {
                last_use[value.0] = Some(instruction_index);
            }
        }
        for (value_index, info) in self.values.iter().enumerate() {
            assert!(
                last_use[value_index].is_some(),
                "generated value {} ({}) is never used",
                info.name,
                info.label
            );
        }

        let live_set = |instruction_index: usize, after: bool| -> BTreeSet<Value> {
            self.values
                .iter()
                .enumerate()
                .filter_map(|(value_index, info)| {
                    let defined = match info.definition {
                        None => true,
                        Some(definition) => {
                            if after {
                                definition <= instruction_index
                            } else {
                                definition < instruction_index
                            }
                        }
                    };
                    let live = if after {
                        last_use[value_index].is_some_and(|last| last > instruction_index)
                    } else {
                        last_use[value_index].is_some_and(|last| last >= instruction_index)
                    };
                    (defined && live).then_some(Value(value_index))
                })
                .collect()
        };

        let mut peak_words: u32 = self
            .values
            .iter()
            .filter(|info| info.definition.is_none())
            .map(|info| info.words)
            .sum();
        let mut peak_label = "function entry".to_owned();
        let mut peak_values: Vec<_> = self
            .values
            .iter()
            .filter(|info| info.definition.is_none())
            .map(|info| info.name.clone())
            .collect();
        let mut allocation_peak_words = peak_words;

        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let before = live_set(instruction_index, false);
            let after = live_set(instruction_index, true);
            let definition_words =
                instruction.definition.map_or(0, |definition| self.values[definition.0].words);
            allocation_peak_words = allocation_peak_words.max(
                before.iter().map(|value| self.values[value.0].words).sum::<u32>()
                    + definition_words,
            );
            for (phase, set) in [("before", before), ("after", after)] {
                let words: u32 = set.iter().map(|value| self.values[value.0].words).sum();
                if words > peak_words {
                    peak_words = words;
                    peak_label =
                        format!("{phase} instruction {instruction_index}: {}", instruction.label);
                    peak_values = set.iter().map(|value| self.name(*value).to_owned()).collect();
                }
            }
            if let Some(definition) = instruction.definition {
                assert_eq!(self.values[definition.0].definition, Some(instruction_index));
            }
        }
        LivenessResult { peak_words, peak_label, peak_values, allocation_peak_words }
    }

    fn body_source(&self) -> String {
        let mut source = String::new();
        for instruction in &self.instructions {
            writeln!(source, "{}", instruction.source).expect("write generated instruction");
        }
        source
    }

    fn assert_source_edges_match_typed_graph(&self) {
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let tokens = source_tokens(&instruction.source);
            let trimmed = instruction.source.trim_start();
            let source_definition = if trimmed.starts_with("const uint s")
                || trimmed.starts_with("const ulong s")
                || trimmed.starts_with("result.x")
            {
                Some(tokens.first().expect("source definition token").clone())
            } else {
                None
            };
            let typed_definition = instruction.definition.map(|value| self.name(value).to_owned());
            assert_eq!(
                source_definition, typed_definition,
                "typed/source definition mismatch at instruction {instruction_index}: {}",
                instruction.label
            );

            if let Some(definition) = instruction.definition {
                let source_words = if trimmed.starts_with("const ulong s") { 2 } else { 1 };
                assert_eq!(
                    source_words,
                    self.values[definition.0].words,
                    "typed/source width mismatch for {}",
                    self.name(definition)
                );
            }

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
                "typed/source use-edge mismatch at instruction {instruction_index}: {}",
                instruction.label
            );
        }
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
    let numeric_suffix = |prefix: &str| {
        token.strip_prefix(prefix).is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
    };
    numeric_suffix("s")
        || numeric_suffix("lhs.x")
        || numeric_suffix("rhs.x")
        || numeric_suffix("result.x")
}

fn source_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in line.chars().chain([' ']) {
        if character.is_ascii_alphanumeric() || character == '_' || character == '.' {
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

fn tokenized_liveness(source: &str) -> LivenessResult {
    let mut values = BTreeMap::<String, TokenValueInfo>::new();
    for side in ["lhs", "rhs"] {
        for index in 0..FIELD_LIMBS {
            values
                .insert(format!("{side}.x{index}"), TokenValueInfo { words: 1, definition: None });
        }
    }
    let mut instructions = Vec::<TokenInstruction>::new();
    for (line_number, line) in source.lines().enumerate() {
        let tokens = source_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        let trimmed = line.trim_start();
        let definition = if trimmed.starts_with("const uint s")
            || trimmed.starts_with("const ulong s")
            || trimmed.starts_with("result.x")
        {
            Some(tokens[0].clone())
        } else {
            None
        };
        let words = if trimmed.starts_with("const ulong s") { 2 } else { 1 };
        if let Some(name) = &definition {
            assert!(
                values
                    .insert(
                        name.clone(),
                        TokenValueInfo { words, definition: Some(instructions.len()) }
                    )
                    .is_none(),
                "duplicate source definition {name}"
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
                "source use before definition: {name} on line {}",
                line_number + 1
            );
        }
        instructions.push(TokenInstruction {
            definition,
            uses,
            label: format!("source line {}: {}", line_number + 1, line.trim()),
        });
    }

    let names: Vec<_> = values.keys().cloned().collect();
    let mut last_use = BTreeMap::<String, usize>::new();
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        for name in &instruction.uses {
            last_use.insert(name.clone(), instruction_index);
        }
    }
    for name in &names {
        assert!(last_use.contains_key(name), "tokenized source value {name} is never used");
    }

    let live = |instruction_index: usize, after: bool| -> BTreeSet<String> {
        values
            .iter()
            .filter_map(|(name, info)| {
                let defined = match info.definition {
                    None => true,
                    Some(definition) => {
                        if after {
                            definition <= instruction_index
                        } else {
                            definition < instruction_index
                        }
                    }
                };
                let last = last_use[name];
                let live = if after { last > instruction_index } else { last >= instruction_index };
                (defined && live).then_some(name.clone())
            })
            .collect()
    };

    let mut peak_values: Vec<_> = values
        .iter()
        .filter_map(|(name, info)| info.definition.is_none().then_some(name.clone()))
        .collect();
    let mut peak_words: u32 = peak_values.iter().map(|name| values[name].words).sum();
    let mut peak_label = "tokenized function entry".to_owned();
    let mut allocation_peak_words = peak_words;
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        let before = live(instruction_index, false);
        let after = live(instruction_index, true);
        let definition_words =
            instruction.definition.as_ref().map_or(0, |definition| values[definition].words);
        allocation_peak_words = allocation_peak_words
            .max(before.iter().map(|name| values[name].words).sum::<u32>() + definition_words);
        for (phase, set) in [("before", before), ("after", after)] {
            let words: u32 = set.iter().map(|name| values[name].words).sum();
            if words > peak_words {
                peak_words = words;
                peak_label = format!("{phase} {}", instruction.label);
                peak_values = set.into_iter().collect();
            }
        }
        if let Some(definition) = &instruction.definition {
            assert_eq!(values[definition].definition, Some(instruction_index));
        }
    }
    LivenessResult { peak_words, peak_label, peak_values, allocation_peak_words }
}

pub(crate) fn generate(
    modulus: &[u32; FIELD_LIMBS],
    block_negative_inverse: &[u32; BLOCK_LIMBS],
) -> GeneratedSchedule {
    // REDC is only valid when the supplied block constant is -p^-1 modulo
    // 2^128. Keep this invariant in the generator boundary so a shape-only
    // test fixture cannot silently emit an arithmetically invalid kernel.
    let mut product_low = [0_u32; BLOCK_LIMBS];
    let mut carry = 0_u128;
    for (column, output) in product_low.iter_mut().enumerate() {
        let mut sum = carry;
        for lhs_index in 0..=column {
            sum += u128::from(modulus[lhs_index])
                * u128::from(block_negative_inverse[column - lhs_index]);
        }
        *output = sum as u32;
        carry = sum >> 32;
    }
    assert_eq!(
        product_low,
        [u32::MAX; BLOCK_LIMBS],
        "blocked-radix-128 constant is not -p^-1 modulo 2^128"
    );

    let mut emitter = SsaEmitter::new();
    let lhs: [Value; FIELD_LIMBS] = std::array::from_fn(|index| {
        emitter.input(format!("lhs.x{index}"), format!("left input {index}"))
    });
    let rhs: [Value; FIELD_LIMBS] = std::array::from_fn(|index| {
        emitter.input(format!("rhs.x{index}"), format!("right input {index}"))
    });
    let mut accumulator = [Word::Literal(0); ACCUMULATOR_LIMBS];

    for round in 0..ROUNDS {
        let lhs_block: [Word; BLOCK_LIMBS] =
            std::array::from_fn(|index| Word::Value(lhs[round * BLOCK_LIMBS + index]));
        for rhs_block in 0..ROUNDS {
            let rhs_words =
                std::array::from_fn(|index| Word::Value(rhs[rhs_block * BLOCK_LIMBS + index]));
            emitter.full_k4_add_into(
                &mut accumulator,
                rhs_block * BLOCK_LIMBS,
                0,
                lhs_block,
                rhs_words,
                &format!("round {round} operand block {rhs_block}"),
            );
        }

        let quotient = emitter.low_k4(
            accumulator[..BLOCK_LIMBS].to_vec().try_into().expect("four accumulator low words"),
            block_negative_inverse.map(Word::Literal),
            &format!("round {round} quotient"),
        );
        let quotient_words = quotient.map(Word::Value);
        for modulus_block in 0..ROUNDS {
            let modulus_words = std::array::from_fn(|index| {
                Word::Literal(modulus[modulus_block * BLOCK_LIMBS + index])
            });
            emitter.full_k4_add_into(
                &mut accumulator,
                modulus_block * BLOCK_LIMBS,
                if modulus_block == 0 { BLOCK_LIMBS } else { 0 },
                quotient_words,
                modulus_words,
                &format!("round {round} correction block {modulus_block}"),
            );
        }

        if round + 1 < ROUNDS {
            accumulator = std::array::from_fn(|index| {
                if index + BLOCK_LIMBS < ACCUMULATOR_LIMBS {
                    accumulator[index + BLOCK_LIMBS]
                } else {
                    Word::Literal(0)
                }
            });
        }
    }

    let raw_result: [Word; FIELD_LIMBS] =
        std::array::from_fn(|index| accumulator[index + BLOCK_LIMBS]);
    let output_values: Vec<_> = raw_result
        .into_iter()
        .enumerate()
        .map(|(index, value)| emitter.output_word(index, value))
        .collect();
    emitter.marker(
        rhs.to_vec(),
        format!(
            "    // @chain-keepalive {}",
            rhs.iter().map(|value| emitter.name(*value)).collect::<Vec<_>>().join(" ")
        ),
        "dependent-chain fixed RHS keepalive",
    );
    emitter.marker(
        output_values.clone(),
        format!(
            "    // @return-uses {}",
            output_values.iter().map(|value| emitter.name(*value)).collect::<Vec<_>>().join(" ")
        ),
        "returned result components",
    );

    assert_eq!(emitter.base_products, 189);
    emitter.assert_source_edges_match_typed_graph();
    let typed = emitter.liveness();
    let mut source = String::new();
    writeln!(source, "// @generated by build.rs; do not edit.").expect("write header");
    writeln!(source, "inline __attribute__((always_inline))").expect("write attribute");
    writeln!(
        source,
        "U32NamedFieldElement u32_named_blocked128_karatsuba_l4_redc(U32NamedFieldElement lhs, U32NamedFieldElement rhs)"
    )
    .expect("write signature");
    writeln!(source, "{{").expect("write function");
    writeln!(
        source,
        "    // Contract: lhs,rhs < 4p; direct radix-2^128 REDC returns a raw result < 3p."
    )
    .expect("write contract");
    source.push_str(&emitter.body_source());
    writeln!(source, "    U32NamedFieldElement result;").expect("write result declaration");
    // Result assignments are already in the SSA body; the declaration must
    // precede them in Metal source, so move it to immediately before the first
    // assignment after generation.
    let declaration = "    U32NamedFieldElement result;\n";
    let first_result = source.find("    result.x0 =").expect("generated result assignment");
    let trailing_declaration = source.rfind(declaration).expect("trailing result declaration");
    source.replace_range(trailing_declaration..trailing_declaration + declaration.len(), "");
    let first_result = source.find("    result.x0 =").unwrap_or(first_result);
    source.insert_str(first_result, declaration);
    writeln!(source, "    return result;").expect("write return");
    writeln!(source, "}}").expect("write function");

    let tokenized = tokenized_liveness(&source);
    assert_eq!(
        typed.peak_words, tokenized.peak_words,
        "typed and emitted-source liveness audits disagree: typed={} at {}; tokenized={} at {}",
        typed.peak_words, typed.peak_label, tokenized.peak_words, tokenized.peak_label
    );
    assert_eq!(
        typed.allocation_peak_words, tokenized.allocation_peak_words,
        "typed and emitted-source allocation-time liveness audits disagree"
    );
    GeneratedSchedule {
        source,
        base_products: emitter.base_products,
        typed_peak_live_words: typed.peak_words,
        tokenized_peak_live_words: tokenized.peak_words,
        typed_allocation_peak_live_words: typed.allocation_peak_words,
        tokenized_allocation_peak_live_words: tokenized.allocation_peak_words,
        typed_peak_label: typed.peak_label,
        typed_peak_values: typed.peak_values.clone(),
        typed_peak_descriptions: typed
            .peak_values
            .iter()
            .map(|name| {
                let info = emitter
                    .values
                    .iter()
                    .find(|info| &info.name == name)
                    .expect("typed peak value exists");
                format!("{}:{}w:{}", info.name, info.words, info.label)
            })
            .collect(),
        tokenized_peak_label: tokenized.peak_label,
        tokenized_peak_values: tokenized.peak_values,
    }
}
