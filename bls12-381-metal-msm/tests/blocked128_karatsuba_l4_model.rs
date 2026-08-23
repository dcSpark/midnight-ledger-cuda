// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Exact model of a direct, three-round radix-2^128 Montgomery reduction.
//!
//! Write `R = Q^3 = 2^384`, where one block is `Q = 2^128`.  At round `i`,
//! the direct recurrence is
//!
//! ```text
//! N_i     = A_i + a_i * b + q_i * p
//! q_i     = (A_i + a_i * b) * (-p^-1) mod Q
//! A_(i+1) = N_i / Q.
//! ```
//!
//! For inputs `a,b < 4p`, the shifted accumulator has the inductive bound
//! `A_i < R`.  Indeed, `a_i,q_i < Q`, so
//!
//! ```text
//! A_(i+1) < A_i/Q + b + p < Q^2 + 5p < R.
//! ```
//!
//! Before the shift,
//!
//! ```text
//! N_i < R + Q(b+p) < R + 5Qp < Q^4,
//! ```
//!
//! so 16 `u32` words are sufficient for every pre-shift accumulator.  The
//! BLS12-381 modulus satisfies both inequalities used above.  Finally, the
//! three quotient blocks form `m < R`, and
//! `A_3 = (a*b + m*p)/R < 16p^2/R + p < 3p`, because `8p < R`.

use num_bigint::BigUint;

include!("../src/field_constants.rs");

const WORD_BITS: usize = 32;
const FIELD_LIMBS: usize = 12;
const BLOCK_LIMBS: usize = 4;
const BLOCKS: usize = FIELD_LIMBS / BLOCK_LIMBS;
const FIELD_BITS: usize = WORD_BITS * FIELD_LIMBS;
const BLOCK_BITS: usize = WORD_BITS * BLOCK_LIMBS;
const ACCUMULATOR_LIMBS: usize = FIELD_LIMBS + BLOCK_LIMBS;
const RANDOM_CASES: usize = 2_048;
const RANDOM_SEED: u64 = 0x424c_4f43_4b31_3238;
const MODULUS_LIMBS: [u32; FIELD_LIMBS] = [
    0xffff_aaab,
    0xb9fe_ffff,
    0xb153_ffff,
    0x1eab_fffe,
    0xf6b0_f624,
    0x6730_d2a0,
    0xf385_12bf,
    0x6477_4b84,
    0x434b_acd7,
    0x4b1b_a7b6,
    0x397f_e69a,
    0x1a01_11ea,
];

#[derive(Debug)]
struct CountedProduct {
    value: BigUint,
    base_products: usize,
}

#[derive(Debug)]
struct BlockedRedcOutput {
    raw: BigUint,
    multiplier: BigUint,
    base_products: usize,
    quotient_blocks: Vec<BigUint>,
    pre_shift_accumulators: Vec<BigUint>,
}

fn power_of_two(bits: usize) -> BigUint {
    BigUint::from(1_u32) << bits
}

fn limb_radix(limbs: usize) -> BigUint {
    power_of_two(WORD_BITS * limbs)
}

fn absolute_difference(lhs: &BigUint, rhs: &BigUint) -> (BigUint, bool) {
    if lhs >= rhs { (lhs - rhs, true) } else { (rhs - lhs, false) }
}

fn subtractive_karatsuba(lhs: &BigUint, rhs: &BigUint, limbs: usize) -> CountedProduct {
    assert!(limbs > 0);
    let width_radix = limb_radix(limbs);
    assert!(lhs < &width_radix);
    assert!(rhs < &width_radix);

    if limbs == 1 {
        return CountedProduct { value: lhs * rhs, base_products: 1 };
    }

    let low_limbs = limbs.div_ceil(2);
    let high_limbs = limbs / 2;
    let split_bits = WORD_BITS * low_limbs;
    let split_radix = power_of_two(split_bits);
    let split_mask = &split_radix - BigUint::from(1_u32);

    let lhs_low = lhs & &split_mask;
    let lhs_high = lhs >> split_bits;
    let rhs_low = rhs & &split_mask;
    let rhs_high = rhs >> split_bits;

    let low = subtractive_karatsuba(&lhs_low, &rhs_low, low_limbs);
    let high = subtractive_karatsuba(&lhs_high, &rhs_high, high_limbs);
    let (lhs_difference, lhs_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    let middle = subtractive_karatsuba(&lhs_difference, &rhs_difference, low_limbs);

    let mut cross = &low.value + &high.value;
    if lhs_nonnegative == rhs_nonnegative {
        cross += &middle.value;
    } else {
        assert!(cross >= middle.value);
        cross -= &middle.value;
    }

    let value = &low.value + (&cross << split_bits) + (&high.value << (2 * split_bits));
    assert_eq!(value, lhs * rhs, "subtractive Karatsuba reconstruction");
    CountedProduct {
        value,
        base_products: low.base_products + middle.base_products + high.base_products,
    }
}

fn karatsuba_product_count(limbs: usize) -> usize {
    if limbs == 1 {
        1
    } else {
        let low_limbs = limbs.div_ceil(2);
        let high_limbs = limbs / 2;
        2 * karatsuba_product_count(low_limbs) + karatsuba_product_count(high_limbs)
    }
}

fn full_k2(lhs: &BigUint, rhs: &BigUint) -> CountedProduct {
    let word_radix = power_of_two(WORD_BITS);
    let word_mask = &word_radix - BigUint::from(1_u32);
    let width_radix = &word_radix * &word_radix;
    assert!(lhs < &width_radix);
    assert!(rhs < &width_radix);

    let lhs_low = lhs & &word_mask;
    let lhs_high = lhs >> WORD_BITS;
    let rhs_low = rhs & &word_mask;
    let rhs_high = rhs >> WORD_BITS;
    let z0 = &lhs_low * &rhs_low;
    let z2 = &lhs_high * &rhs_high;
    let (lhs_difference, lhs_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    let middle = lhs_difference * rhs_difference;

    let mut cross = &z0 + &z2;
    if lhs_nonnegative == rhs_nonnegative {
        cross += middle;
    } else {
        assert!(cross >= middle);
        cross -= middle;
    }

    let value = &z0 + (cross << WORD_BITS) + (&z2 << (2 * WORD_BITS));
    assert_eq!(value, lhs * rhs, "full K2 reconstruction");
    CountedProduct { value, base_products: 3 }
}

fn truncated_low_k2(lhs: &BigUint, rhs: &BigUint) -> CountedProduct {
    let word_radix = power_of_two(WORD_BITS);
    let word_mask = &word_radix - BigUint::from(1_u32);
    let width_radix = &word_radix * &word_radix;
    let result_mask = &width_radix - BigUint::from(1_u32);
    assert!(lhs < &width_radix);
    assert!(rhs < &width_radix);

    let lhs_low = lhs & &word_mask;
    let lhs_high = lhs >> WORD_BITS;
    let rhs_low = rhs & &word_mask;
    let rhs_high = rhs >> WORD_BITS;
    let low = &lhs_low * &rhs_low;
    let cross = &lhs_low * &rhs_high + &lhs_high * &rhs_low;
    let value = (low + (cross << WORD_BITS)) & &result_mask;
    assert_eq!(value, (lhs * rhs) & result_mask, "truncated-low K2 reconstruction");
    CountedProduct { value, base_products: 3 }
}

fn truncated_low_k4(lhs: &BigUint, rhs: &BigUint) -> CountedProduct {
    let split_bits = 2 * WORD_BITS;
    let split_radix = power_of_two(split_bits);
    let split_mask = &split_radix - BigUint::from(1_u32);
    let result_radix = &split_radix * &split_radix;
    let result_mask = &result_radix - BigUint::from(1_u32);
    assert!(lhs < &result_radix);
    assert!(rhs < &result_radix);

    let lhs_low = lhs & &split_mask;
    let lhs_high = lhs >> split_bits;
    let rhs_low = rhs & &split_mask;
    let rhs_high = rhs >> split_bits;

    // z0 contributes all four of its words.  z2 and the signed middle term
    // contribute only their low two words after the outer two-word shift.
    let z0 = full_k2(&lhs_low, &rhs_low);
    let z2_low = truncated_low_k2(&lhs_high, &rhs_high);
    let (lhs_difference, lhs_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    let middle_low = truncated_low_k2(&lhs_difference, &rhs_difference);

    let cross_base = (&z0.value + &z2_low.value) & &split_mask;
    let cross_low = if lhs_nonnegative == rhs_nonnegative {
        (cross_base + &middle_low.value) & &split_mask
    } else {
        (cross_base + &split_radix - &middle_low.value) & &split_mask
    };
    let value = (&z0.value + (cross_low << split_bits)) & &result_mask;
    assert_eq!(value, (lhs * rhs) & result_mask, "truncated-low K4 reconstruction");

    CountedProduct {
        value,
        base_products: z0.base_products + z2_low.base_products + middle_low.base_products,
    }
}

fn inverse_mod_power_of_two(odd: &BigUint, bits: usize) -> BigUint {
    assert_eq!(odd & BigUint::from(1_u32), BigUint::from(1_u32));
    let one = BigUint::from(1_u32);
    let mut inverse = one.clone();
    for bit in 1..bits {
        if ((odd * &inverse) >> bit) & &one == one {
            inverse += &one << bit;
        }
    }
    inverse
}

fn block(value: &BigUint, block_index: usize, block_mask: &BigUint) -> BigUint {
    (value >> (block_index * BLOCK_BITS)) & block_mask
}

fn separated_redc_raw(
    lhs: &BigUint,
    rhs: &BigUint,
    modulus: &BigUint,
    negative_inverse: &BigUint,
) -> BigUint {
    let radix = power_of_two(FIELD_BITS);
    let radix_mask = &radix - BigUint::from(1_u32);
    let product = lhs * rhs;
    let multiplier = ((&product & &radix_mask) * negative_inverse) & &radix_mask;
    let numerator = product + multiplier * modulus;
    assert_eq!(&numerator & &radix_mask, BigUint::from(0_u32));
    numerator >> FIELD_BITS
}

fn direct_blocked_redc_raw(
    lhs: &BigUint,
    rhs: &BigUint,
    modulus: &BigUint,
    block_negative_inverse: &BigUint,
) -> BlockedRedcOutput {
    let radix = power_of_two(FIELD_BITS);
    let block_radix = power_of_two(BLOCK_BITS);
    let block_mask = &block_radix - BigUint::from(1_u32);
    let accumulator_radix = power_of_two(WORD_BITS * ACCUMULATOR_LIMBS);
    assert!(lhs < &radix);
    assert!(rhs < &radix);

    let mut accumulator = BigUint::from(0_u32);
    let mut multiplier = BigUint::from(0_u32);
    let mut base_products = 0_usize;
    let mut quotient_blocks = Vec::with_capacity(BLOCKS);
    let mut pre_shift_accumulators = Vec::with_capacity(BLOCKS);

    for round in 0..BLOCKS {
        let lhs_block = block(lhs, round, &block_mask);
        for rhs_block_index in 0..BLOCKS {
            let product = subtractive_karatsuba(
                &lhs_block,
                &block(rhs, rhs_block_index, &block_mask),
                BLOCK_LIMBS,
            );
            accumulator += product.value << (rhs_block_index * BLOCK_BITS);
            base_products += product.base_products;
        }

        let quotient_product =
            truncated_low_k4(&(&accumulator & &block_mask), block_negative_inverse);
        let quotient = quotient_product.value;
        base_products += quotient_product.base_products;
        multiplier += &quotient << (round * BLOCK_BITS);
        quotient_blocks.push(quotient.clone());

        // The generated sink drops the cancelled low block before computing
        // q*p.  If P=q*p=P_high*Q+P_low, then the cancellation identity gives
        // A_low=(-P_low mod Q), so the carry into the retained words is
        // P_high + (P_low != 0).  Check that optimized form independently of
        // the blockwise correction loop below.
        let accumulator_before_correction = accumulator.clone();
        let correction_exact = &quotient * modulus;
        let correction_low = &correction_exact & &block_mask;
        assert_eq!(
            (&accumulator_before_correction + &correction_low) & &block_mask,
            BigUint::from(0_u32),
            "round {round} optimized low-block cancellation"
        );
        let optimized_shifted = (&accumulator_before_correction >> BLOCK_BITS)
            + (&correction_exact >> BLOCK_BITS)
            + BigUint::from(u32::from(correction_low != BigUint::from(0_u32)));

        for modulus_block_index in 0..BLOCKS {
            let correction = subtractive_karatsuba(
                &quotient,
                &block(modulus, modulus_block_index, &block_mask),
                BLOCK_LIMBS,
            );
            accumulator += correction.value << (modulus_block_index * BLOCK_BITS);
            base_products += correction.base_products;
        }

        assert_eq!(
            &accumulator & &block_mask,
            BigUint::from(0_u32),
            "round {round} numerator must be divisible by Q"
        );
        assert!(
            accumulator < accumulator_radix,
            "round {round} pre-shift accumulator exceeds 16 words"
        );
        pre_shift_accumulators.push(accumulator.clone());
        accumulator >>= BLOCK_BITS;
        assert_eq!(accumulator, optimized_shifted, "round {round} optimized cancellation carry");
        assert!(accumulator < radix, "round {round} shifted accumulator exceeds R");
    }

    let full_numerator = lhs * rhs + &multiplier * modulus;
    assert_eq!(&full_numerator & (&radix - BigUint::from(1_u32)), BigUint::from(0_u32));
    assert_eq!(accumulator, full_numerator >> FIELD_BITS, "direct recurrence invariant");

    BlockedRedcOutput {
        raw: accumulator,
        multiplier,
        base_products,
        quotient_blocks,
        pre_shift_accumulators,
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_below(state: &mut u64, limit: &BigUint) -> BigUint {
    let mut limbs = Vec::with_capacity(FIELD_LIMBS);
    for _ in 0..(FIELD_LIMBS / 2) {
        let word = splitmix64(state);
        limbs.push(word as u32);
        limbs.push((word >> 32) as u32);
    }
    BigUint::new(limbs) % limit
}

fn l4_boundaries(modulus: &BigUint) -> Vec<BigUint> {
    let one = BigUint::from(1_u32);
    let two_p = modulus << 1_usize;
    let three_p = modulus * BigUint::from(3_u32);
    let four_p = modulus << 2_usize;
    vec![
        BigUint::from(0_u32),
        one.clone(),
        modulus - &one,
        modulus.clone(),
        modulus + &one,
        &two_p - &one,
        two_p,
        &three_p - &one,
        three_p,
        &four_p - &one,
    ]
}

fn adversarial_l4_values(modulus: &BigUint) -> Vec<BigUint> {
    let one = BigUint::from(1_u32);
    let four_p = modulus << 2_usize;
    let mut values = Vec::new();

    // Every u32 limb boundary also covers Q=2^128 at limbs 4 and 8.  The
    // all-ones predecessor and one-above successor exercise carry chains.
    for limb in 1..FIELD_LIMBS {
        let boundary = power_of_two(WORD_BITS * limb);
        for candidate in [&boundary - &one, boundary.clone(), &boundary + &one] {
            if candidate < four_p {
                values.push(candidate);
            }
        }
    }

    let alternating_a = BigUint::new(
        (0..FIELD_LIMBS)
            .map(|limb| {
                if limb + 1 == FIELD_LIMBS {
                    0
                } else if limb % 2 == 0 {
                    0xaaaa_aaaa
                } else {
                    0x5555_5555
                }
            })
            .collect(),
    );
    let alternating_5 = BigUint::new(
        (0..FIELD_LIMBS)
            .map(|limb| {
                if limb + 1 == FIELD_LIMBS {
                    0
                } else if limb % 2 == 0 {
                    0x5555_5555
                } else {
                    0xaaaa_aaaa
                }
            })
            .collect(),
    );
    let low_eleven_words_all_ones = BigUint::new(
        (0..FIELD_LIMBS).map(|limb| if limb + 1 == FIELD_LIMBS { 0 } else { u32::MAX }).collect(),
    );
    let alternating_words = BigUint::new(
        (0..FIELD_LIMBS)
            .map(|limb| if limb + 1 == FIELD_LIMBS || limb % 2 == 1 { 0 } else { u32::MAX })
            .collect(),
    );
    let alternating_words_inverse = BigUint::new(
        (0..FIELD_LIMBS)
            .map(|limb| if limb + 1 == FIELD_LIMBS || limb % 2 == 0 { 0 } else { u32::MAX })
            .collect(),
    );

    values.extend([
        alternating_a,
        alternating_5,
        low_eleven_words_all_ones,
        alternating_words,
        alternating_words_inverse,
        &four_p - BigUint::from(2_u32),
        &four_p - &one,
    ]);
    assert!(values.iter().all(|value| value < &four_p));
    values.sort();
    values.dedup();
    values
}

fn low_k4_boundaries() -> Vec<BigUint> {
    let one = BigUint::from(1_u32);
    let word_radix = power_of_two(WORD_BITS);
    let half_radix = power_of_two(2 * WORD_BITS);
    let block_radix = power_of_two(BLOCK_BITS);
    let alternating_a = BigUint::new(vec![0xaaaa_aaaa, 0x5555_5555, 0xaaaa_aaaa, 0x5555_5555]);
    let alternating_5 = BigUint::new(vec![0x5555_5555, 0xaaaa_aaaa, 0x5555_5555, 0xaaaa_aaaa]);
    vec![
        BigUint::from(0_u32),
        one.clone(),
        &word_radix - &one,
        word_radix.clone(),
        &word_radix + &one,
        &half_radix - &one,
        half_radix.clone(),
        &half_radix + &one,
        alternating_a,
        alternating_5,
        &block_radix - &one,
    ]
}

#[test]
fn truncated_low_k4_is_exact_at_nine_base_products() {
    let one = BigUint::from(1_u32);
    let block_radix = power_of_two(BLOCK_BITS);
    let block_mask = &block_radix - &one;
    let assert_case = |lhs: &BigUint, rhs: &BigUint, context: &str| {
        let actual = truncated_low_k4(lhs, rhs);
        assert_eq!(actual.base_products, 9, "{context}: product count");
        assert_eq!(actual.value, (lhs * rhs) & &block_mask, "{context}: low product");
    };

    let boundaries = low_k4_boundaries();
    for (lhs_index, lhs) in boundaries.iter().enumerate() {
        for (rhs_index, rhs) in boundaries.iter().enumerate() {
            assert_case(lhs, rhs, &format!("boundary lhs={lhs_index} rhs={rhs_index}"));
        }
    }

    let mut state = RANDOM_SEED ^ 0x4b34_4c4f_575f_4e49;
    for case_index in 0..RANDOM_CASES {
        let lhs = random_below(&mut state, &block_radix);
        let rhs = random_below(&mut state, &block_radix);
        assert_case(&lhs, &rhs, &format!("random case={case_index}"));
    }
}

#[test]
fn direct_blocked128_redc_is_exact_at_189_base_products() {
    let modulus_from_hex =
        BigUint::parse_bytes(FP_MODULUS_HEX.as_bytes(), 16).expect("valid BLS12-381 modulus");
    let modulus = BigUint::new(MODULUS_LIMBS.to_vec());
    assert_eq!(modulus, modulus_from_hex);

    let one = BigUint::from(1_u32);
    let radix = power_of_two(FIELD_BITS);
    let radix_mask = &radix - &one;
    let block_radix = power_of_two(BLOCK_BITS);
    let block_mask = &block_radix - &one;
    let accumulator_radix = power_of_two(WORD_BITS * ACCUMULATOR_LIMBS);
    let block_radix_squared = &block_radix * &block_radix;
    let three_p = &modulus * BigUint::from(3_u32);
    let four_p = &modulus << 2_usize;
    let five_p = &modulus * BigUint::from(5_u32);
    let eight_p = &modulus << 3_usize;
    assert!(four_p < radix, "L4 representatives must fit R");

    // These exact modulus inequalities discharge the accumulator and final
    // raw-range proofs in the module-level derivation.
    assert!(&block_radix_squared + &five_p < radix, "Q^2 + 5p < R");
    assert!(&radix + &block_radix * &five_p < accumulator_radix, "R + 5Qp < Q^4");
    assert!(eight_p < radix, "8p < R");

    let inverse = inverse_mod_power_of_two(&modulus, FIELD_BITS);
    let negative_inverse = &radix - &inverse;
    let block_negative_inverse = &negative_inverse & &block_mask;
    assert_eq!(((&modulus & &block_mask) * &block_negative_inverse) & &block_mask, block_mask);
    let radix_inverse = (&radix % &modulus).modpow(&(&modulus - BigUint::from(2_u32)), &modulus);
    assert_eq!((&radix * &radix_inverse) % &modulus, one);

    assert_eq!(karatsuba_product_count(BLOCK_LIMBS), 9);
    assert_eq!(karatsuba_product_count(FIELD_LIMBS), 63);
    let direct_count = BLOCKS
        * (BLOCKS * karatsuba_product_count(BLOCK_LIMBS)
            + karatsuba_product_count(BLOCK_LIMBS)
            + BLOCKS * karatsuba_product_count(BLOCK_LIMBS));
    assert_eq!(direct_count, 189);

    // 171 is a valid arithmetic count only for a different topology that
    // first materializes the complete 24-word numerator product:
    // K12 + 3 * (K4 quotient + 3*K4 correction) = 63 + 3*36.
    let precomputed_numerator_count = karatsuba_product_count(FIELD_LIMBS)
        + BLOCKS
            * (karatsuba_product_count(BLOCK_LIMBS)
                + BLOCKS * karatsuba_product_count(BLOCK_LIMBS));
    assert_eq!(precomputed_numerator_count, 171);

    let assert_case = |lhs: &BigUint, rhs: &BigUint, context: &str| {
        assert!(lhs < &four_p, "{context}: left input outside L4");
        assert!(rhs < &four_p, "{context}: right input outside L4");
        let product = lhs * rhs;
        let global_multiplier = ((&product & &radix_mask) * &negative_inverse) & &radix_mask;
        let expected_canonical = ((&product % &modulus) * &radix_inverse) % &modulus;
        let expected = separated_redc_raw(lhs, rhs, &modulus, &negative_inverse);
        let actual = direct_blocked_redc_raw(lhs, rhs, &modulus, &block_negative_inverse);
        assert_eq!(actual.base_products, direct_count, "{context}: product count");
        assert_eq!(actual.raw, expected, "{context}: exact raw REDC");
        assert_eq!(actual.multiplier, global_multiplier, "{context}: global multiplier");
        assert_eq!(&actual.raw % &modulus, expected_canonical, "{context}: canonical oracle");
        assert!(actual.raw < three_p, "{context}: L4 raw range");
        assert!(actual.multiplier < radix, "{context}: multiplier range");
        assert_eq!(actual.quotient_blocks.len(), BLOCKS);
        assert_eq!(actual.pre_shift_accumulators.len(), BLOCKS);
        assert!(
            actual.pre_shift_accumulators.iter().all(|value| value < &accumulator_radix),
            "{context}: pre-shift accumulator range"
        );
        assert_eq!(
            (&actual.raw * &radix) % &modulus,
            (lhs * rhs) % &modulus,
            "{context}: Montgomery congruence"
        );
    };

    let boundaries = l4_boundaries(&modulus);
    for (lhs_index, lhs) in boundaries.iter().enumerate() {
        for (rhs_index, rhs) in boundaries.iter().enumerate() {
            assert_case(lhs, rhs, &format!("boundary lhs={lhs_index} rhs={rhs_index}"));
        }
    }

    let adversarial = adversarial_l4_values(&modulus);
    for (lhs_index, lhs) in adversarial.iter().enumerate() {
        for (rhs_index, rhs) in adversarial.iter().enumerate() {
            assert_case(lhs, rhs, &format!("adversarial lhs={lhs_index} rhs={rhs_index}"));
        }
    }

    // With rhs=1 and all earlier lhs blocks zero, choosing a_i=p mod Q makes
    // q_i=(p mod Q)*(-p^-1)=Q-1.  lhs=0 forces q_i=0 in every round.  All four
    // constructed representatives remain inside the strict L4 domain.
    let zero_case =
        direct_blocked_redc_raw(&BigUint::from(0_u32), &one, &modulus, &block_negative_inverse);
    assert!(zero_case.quotient_blocks.iter().all(|quotient| quotient == &BigUint::from(0_u32)));
    let quotient_inducing_block = &modulus & &block_mask;
    for round in 0..BLOCKS {
        let lhs = &quotient_inducing_block << (round * BLOCK_BITS);
        assert!(lhs < four_p, "forced q=Q-1 round {round}: lhs outside L4");
        assert_case(&lhs, &one, &format!("forced q=Q-1 round={round}"));
        let actual = direct_blocked_redc_raw(&lhs, &one, &modulus, &block_negative_inverse);
        assert_eq!(actual.quotient_blocks[round], block_mask, "forced q=Q-1 round {round}");
    }

    let mut state = RANDOM_SEED;
    for case_index in 0..RANDOM_CASES {
        let lhs = random_below(&mut state, &four_p);
        let rhs = random_below(&mut state, &four_p);
        assert_case(&lhs, &rhs, &format!("random case={case_index}"));
    }

    let maximum = &four_p - &one;
    assert_case(&maximum, &maximum, "maximum L4 representatives");
    assert_eq!((&modulus * &negative_inverse) & &radix_mask, radix_mask);
}
