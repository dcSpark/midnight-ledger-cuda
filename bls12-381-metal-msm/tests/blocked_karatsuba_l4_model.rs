// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;

include!("../src/field_constants.rs");

const WORD_BITS: usize = 32;
const FIELD_LIMBS: usize = 12;
const BLOCK_LIMBS: usize = 6;
const FIELD_BITS: usize = WORD_BITS * FIELD_LIMBS;
const BLOCK_BITS: usize = WORD_BITS * BLOCK_LIMBS;
const RANDOM_CASES: usize = 2_048;
const RANDOM_SEED: u64 = 0x424c_4f43_4b45_4433;
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

fn power_of_two(bits: usize) -> BigUint {
    BigUint::from(1_u32) << bits
}

fn limb_radix(limbs: usize) -> BigUint {
    power_of_two(WORD_BITS * limbs)
}

fn absolute_difference(lhs: &BigUint, rhs: &BigUint) -> (BigUint, bool) {
    if lhs >= rhs { (lhs - rhs, true) } else { (rhs - lhs, false) }
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

fn truncated_k3_low(lhs: &BigUint, rhs: &BigUint) -> CountedProduct {
    let radix = limb_radix(3);
    let split_radix = limb_radix(2);
    let word_radix = limb_radix(1);
    let result_mask = &radix - BigUint::from(1_u32);
    let split_mask = &split_radix - BigUint::from(1_u32);
    let word_mask = &word_radix - BigUint::from(1_u32);
    assert!(lhs < &radix);
    assert!(rhs < &radix);

    let lhs_low = lhs & &split_mask;
    let lhs_high = lhs >> (2 * WORD_BITS);
    let rhs_low = rhs & &split_mask;
    let rhs_high = rhs >> (2 * WORD_BITS);
    let z0 = &lhs_low * &rhs_low;
    let z2_low = (&lhs_high * &rhs_high) & &word_mask;
    let (lhs_difference, lhs_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    let middle_low = ((lhs_difference & &word_mask) * (rhs_difference & &word_mask)) & &word_mask;

    let cross_base = (&z0 + z2_low) & &word_mask;
    let cross = if lhs_nonnegative == rhs_nonnegative {
        (cross_base + middle_low) & &word_mask
    } else {
        (cross_base + &word_radix - middle_low) & &word_mask
    };
    let value = (&z0 + (cross << (2 * WORD_BITS))) & &result_mask;
    assert_eq!(value, (lhs * rhs) & result_mask, "truncated K3 reconstruction");

    // Full K2 z0 uses three products. Only the low word of z2 and of the
    // signed middle product contributes to the requested three limbs.
    CountedProduct { value, base_products: karatsuba_product_count(2) + 1 + 1 }
}

fn truncated_k6_low(lhs: &BigUint, rhs: &BigUint) -> CountedProduct {
    let radix = limb_radix(BLOCK_LIMBS);
    let split_radix = limb_radix(3);
    let result_mask = &radix - BigUint::from(1_u32);
    let split_mask = &split_radix - BigUint::from(1_u32);
    assert!(lhs < &radix);
    assert!(rhs < &radix);

    let lhs_low = lhs & &split_mask;
    let lhs_high = lhs >> (3 * WORD_BITS);
    let rhs_low = rhs & &split_mask;
    let rhs_high = rhs >> (3 * WORD_BITS);
    let z0 = &lhs_low * &rhs_low;
    let z2_low = truncated_k3_low(&lhs_high, &rhs_high);
    let (lhs_difference, lhs_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    let middle_low = truncated_k3_low(&lhs_difference, &rhs_difference);

    let cross_base = (&z0 + &z2_low.value) & &split_mask;
    let cross = if lhs_nonnegative == rhs_nonnegative {
        (cross_base + &middle_low.value) & &split_mask
    } else {
        (cross_base + &split_radix - &middle_low.value) & &split_mask
    };
    let value = (&z0 + (cross << (3 * WORD_BITS))) & &result_mask;
    assert_eq!(value, (lhs * rhs) & result_mask, "truncated K6 reconstruction");

    CountedProduct {
        value,
        base_products: karatsuba_product_count(3) + z2_low.base_products + middle_low.base_products,
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

fn separated_redc_raw(
    lhs: &BigUint,
    rhs: &BigUint,
    modulus: &BigUint,
    negative_inverse: &BigUint,
) -> BigUint {
    let radix = power_of_two(FIELD_BITS);
    let mask = &radix - BigUint::from(1_u32);
    let product = lhs * rhs;
    let multiplier = ((&product & &mask) * negative_inverse) & &mask;
    let numerator = product + multiplier * modulus;
    assert_eq!(&numerator & mask, BigUint::from(0_u32));
    numerator >> FIELD_BITS
}

fn blocked_redc_raw(
    lhs: &BigUint,
    rhs: &BigUint,
    modulus: &BigUint,
    block_negative_inverse: &BigUint,
) -> CountedProduct {
    let block_radix = power_of_two(BLOCK_BITS);
    let block_mask = &block_radix - BigUint::from(1_u32);
    let rhs_low = rhs & &block_mask;
    let rhs_high = rhs >> BLOCK_BITS;
    let modulus_low = modulus & &block_mask;
    let modulus_high = modulus >> BLOCK_BITS;
    let mut accumulator = BigUint::from(0_u32);
    let mut base_products = 0_usize;

    for block_index in 0..2 {
        let lhs_block = (lhs >> (block_index * BLOCK_BITS)) & &block_mask;
        accumulator += &lhs_block * &rhs_low + ((&lhs_block * &rhs_high) << BLOCK_BITS);
        base_products += 2 * karatsuba_product_count(BLOCK_LIMBS);

        let quotient_product =
            truncated_k6_low(&(&accumulator & &block_mask), block_negative_inverse);
        let quotient = &quotient_product.value & &block_mask;
        base_products += quotient_product.base_products;

        accumulator += &quotient * &modulus_low + ((&quotient * &modulus_high) << BLOCK_BITS);
        base_products += 2 * karatsuba_product_count(BLOCK_LIMBS);
        assert_eq!(&accumulator & &block_mask, BigUint::from(0_u32));
        accumulator >>= BLOCK_BITS;
    }

    CountedProduct { value: accumulator, base_products }
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

#[test]
fn blocked_karatsuba_redc_is_exact_at_202_base_products() {
    let modulus_from_hex =
        BigUint::parse_bytes(FP_MODULUS_HEX.as_bytes(), 16).expect("valid BLS12-381 modulus");
    let modulus = BigUint::new(MODULUS_LIMBS.to_vec());
    assert_eq!(modulus, modulus_from_hex);

    let one = BigUint::from(1_u32);
    let radix = power_of_two(FIELD_BITS);
    let radix_mask = &radix - &one;
    let block_radix = power_of_two(BLOCK_BITS);
    let block_mask = &block_radix - &one;
    let four_p = &modulus << 2_usize;
    let three_p = &modulus * BigUint::from(3_u32);
    assert!(four_p < radix);

    let inverse = inverse_mod_power_of_two(&modulus, FIELD_BITS);
    let negative_inverse = &radix - &inverse;
    let block_negative_inverse = &negative_inverse & &block_mask;
    assert_eq!(((&modulus & &block_mask) * &block_negative_inverse) & &block_mask, block_mask);
    assert_eq!(karatsuba_product_count(3), 7);
    assert_eq!(karatsuba_product_count(BLOCK_LIMBS), 21);
    assert_eq!(truncated_k6_low(&block_mask, &block_mask).base_products, 17);

    let assert_case = |lhs: &BigUint, rhs: &BigUint, context: &str| {
        let expected = separated_redc_raw(lhs, rhs, &modulus, &negative_inverse);
        let actual = blocked_redc_raw(lhs, rhs, &modulus, &block_negative_inverse);
        assert_eq!(actual.base_products, 202, "{context}: product count");
        assert_eq!(actual.value, expected, "{context}: exact raw REDC");
        assert!(actual.value < three_p, "{context}: L4 raw range");
    };

    let boundaries = l4_boundaries(&modulus);
    for (lhs_index, lhs) in boundaries.iter().enumerate() {
        for (rhs_index, rhs) in boundaries.iter().enumerate() {
            assert_case(lhs, rhs, &format!("boundary lhs={lhs_index} rhs={rhs_index}"));
        }
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
