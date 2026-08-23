// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;

include!("../src/field_constants.rs");

const WORD_BITS: usize = 32;
const LIMBS: usize = 12;
const RADIX_BITS: usize = WORD_BITS * LIMBS;
const RANDOM_CASES: usize = 2_048;
const RANDOM_SEED: u64 = 0x4b31_324c_3452_4544;
const MODULUS_LIMBS: [u32; LIMBS] = [
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
struct KaratsubaProduct {
    value: BigUint,
    base_products: usize,
}

#[derive(Debug)]
struct SeparatedRedcOutput {
    raw: BigUint,
    canonical: BigUint,
    base_products: usize,
    canonical_subtractions: usize,
}

struct RedcReference<'a> {
    modulus: &'a BigUint,
    negative_inverse: &'a BigUint,
    radix: &'a BigUint,
    radix_inverse: &'a BigUint,
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

fn subtractive_karatsuba(lhs: &BigUint, rhs: &BigUint, limbs: usize) -> KaratsubaProduct {
    assert!(limbs > 0, "Karatsuba width must be nonzero");
    let width_radix = limb_radix(limbs);
    assert!(lhs < &width_radix, "left operand exceeds declared limb width");
    assert!(rhs < &width_radix, "right operand exceeds declared limb width");

    if limbs == 1 {
        return KaratsubaProduct { value: lhs * rhs, base_products: 1 };
    }

    // The low side owns the extra limb at odd widths. This makes the signed
    // differences fit the same width as the low child and gives
    // K(n) = K(ceil(n/2)) + K(floor(n/2)) + K(ceil(n/2)).
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

    // The middle product uses
    //   (lhs_low - lhs_high) * (rhs_high - rhs_low).
    // Adding it to z0 + z2 recovers lhs_low*rhs_high + lhs_high*rhs_low.
    let (lhs_difference, lhs_difference_nonnegative) = absolute_difference(&lhs_low, &lhs_high);
    let (rhs_difference, rhs_difference_nonnegative) = absolute_difference(&rhs_high, &rhs_low);
    assert!(lhs_difference < split_radix, "left signed difference exceeds low child width");
    assert!(rhs_difference < split_radix, "right signed difference exceeds low child width");
    let middle = subtractive_karatsuba(&lhs_difference, &rhs_difference, low_limbs);

    let mut cross = &low.value + &high.value;
    if lhs_difference_nonnegative == rhs_difference_nonnegative {
        cross += &middle.value;
    } else {
        assert!(cross >= middle.value, "subtractive Karatsuba cross term is negative");
        cross -= &middle.value;
    }

    let value = &low.value + (&cross << split_bits) + (&high.value << (2 * split_bits));
    let base_products = low.base_products + middle.base_products + high.base_products;
    assert_eq!(value, lhs * rhs, "subtractive Karatsuba reconstruction");

    KaratsubaProduct { value, base_products }
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

fn inverse_mod_power_of_two(odd: &BigUint, bits: usize) -> BigUint {
    assert_eq!(odd & BigUint::from(1_u32), BigUint::from(1_u32));
    let one = BigUint::from(1_u32);
    let mut inverse = one.clone();

    // Hensel-lift one bit at a time. At step i, inverse is already correct
    // modulo 2^i; setting bit i when needed makes it correct modulo 2^(i+1).
    for bit in 1..bits {
        if ((odd * &inverse) >> bit) & &one == one {
            inverse += &one << bit;
        }
        let modulus = &one << (bit + 1);
        assert_eq!((odd * &inverse) % modulus, one);
    }

    inverse
}

fn separated_redc(
    lhs: &BigUint,
    rhs: &BigUint,
    modulus: &BigUint,
    negative_inverse: &BigUint,
    radix: &BigUint,
) -> SeparatedRedcOutput {
    assert!(lhs < radix, "left REDC operand exceeds R");
    assert!(rhs < radix, "right REDC operand exceeds R");
    let radix_mask = radix - BigUint::from(1_u32);

    let product = subtractive_karatsuba(lhs, rhs, LIMBS);
    let product_low = &product.value & &radix_mask;
    let multiplier_product = subtractive_karatsuba(&product_low, negative_inverse, LIMBS);
    let multiplier = &multiplier_product.value & &radix_mask;
    let correction = subtractive_karatsuba(&multiplier, modulus, LIMBS);

    let numerator = &product.value + &correction.value;
    assert_eq!(
        &numerator & &radix_mask,
        BigUint::from(0_u32),
        "T + m*p must be exactly divisible by R"
    );
    let raw = numerator >> RADIX_BITS;
    let mut canonical = raw.clone();
    let mut canonical_subtractions = 0_usize;
    while &canonical >= modulus {
        canonical -= modulus;
        canonical_subtractions += 1;
    }

    SeparatedRedcOutput {
        raw,
        canonical,
        base_products: product.base_products
            + multiplier_product.base_products
            + correction.base_products,
        canonical_subtractions,
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
    let mut limbs = Vec::with_capacity(LIMBS);
    for _ in 0..(LIMBS / 2) {
        let word = splitmix64(state);
        limbs.push(word as u32);
        limbs.push((word >> 32) as u32);
    }
    BigUint::new(limbs) % limit
}

fn canonical_boundary_values(modulus: &BigUint) -> Vec<BigUint> {
    let one = BigUint::from(1_u32);
    let mut values = vec![
        BigUint::from(0_u32),
        one.clone(),
        BigUint::from(2_u32),
        modulus >> 1_usize,
        modulus - BigUint::from(2_u32),
        modulus - &one,
    ];
    for limb in 1..LIMBS {
        let boundary = power_of_two(WORD_BITS * limb);
        values.push((&boundary - &one) % modulus);
        values.push(boundary % modulus);
    }
    values.sort();
    values.dedup();
    values
}

fn l4_boundary_values(modulus: &BigUint) -> Vec<BigUint> {
    let zero = BigUint::from(0_u32);
    let one = BigUint::from(1_u32);
    let two = BigUint::from(2_u32);
    let two_p = modulus << 1_usize;
    let three_p = modulus * BigUint::from(3_u32);
    let four_p = modulus << 2_usize;
    vec![
        zero,
        one.clone(),
        modulus - &one,
        modulus.clone(),
        modulus + &one,
        &two_p - &one,
        two_p,
        modulus * &two + &one,
        &three_p - &one,
        three_p,
        modulus * BigUint::from(3_u32) + &one,
        &four_p - &two,
        four_p - &one,
    ]
}

impl RedcReference<'_> {
    fn assert_matches(
        &self,
        lhs: &BigUint,
        rhs: &BigUint,
        raw_limit: &BigUint,
        maximum_subtractions: usize,
        context: &str,
    ) -> usize {
        let actual = separated_redc(lhs, rhs, self.modulus, self.negative_inverse, self.radix);
        let expected = ((lhs * rhs) * self.radix_inverse) % self.modulus;

        assert_eq!(actual.base_products, 3 * 63, "{context}: base-product count");
        assert_eq!(actual.canonical, expected, "{context}: canonical REDC value");
        assert_eq!(&actual.raw % self.modulus, expected, "{context}: raw REDC residue");
        assert!(&actual.raw < raw_limit, "{context}: raw REDC range");
        assert!(
            actual.canonical_subtractions <= maximum_subtractions,
            "{context}: canonical subtraction bound"
        );
        actual.canonical_subtractions
    }
}

#[test]
fn subtractive_karatsuba_recurrence_is_exact() {
    assert_eq!(karatsuba_product_count(1), 1);
    assert_eq!(karatsuba_product_count(2), 3);
    assert_eq!(karatsuba_product_count(3), 7);
    assert_eq!(karatsuba_product_count(6), 21);
    assert_eq!(karatsuba_product_count(12), 63);
    assert_eq!(3 * karatsuba_product_count(12), 189);

    let one = BigUint::from(1_u32);
    let mut state = RANDOM_SEED;
    for limbs in [2_usize, 3, 6, 12] {
        let radix = limb_radix(limbs);
        let maximum = &radix - &one;
        for (lhs, rhs) in [
            (BigUint::from(0_u32), maximum.clone()),
            (one.clone(), maximum.clone()),
            (maximum.clone(), maximum),
        ] {
            let product = subtractive_karatsuba(&lhs, &rhs, limbs);
            assert_eq!(product.value, &lhs * &rhs);
            assert_eq!(product.base_products, karatsuba_product_count(limbs));
        }
        for _ in 0..256 {
            let lhs = random_below(&mut state, &radix);
            let rhs = random_below(&mut state, &radix);
            let product = subtractive_karatsuba(&lhs, &rhs, limbs);
            assert_eq!(product.value, &lhs * &rhs);
            assert_eq!(product.base_products, karatsuba_product_count(limbs));
        }
    }
}

#[test]
fn separated_redc_matches_reference_for_canonical_and_l4_inputs() {
    let modulus_from_hex =
        BigUint::parse_bytes(FP_MODULUS_HEX.as_bytes(), 16).expect("valid BLS12-381 modulus");
    let modulus = BigUint::new(MODULUS_LIMBS.to_vec());
    assert_eq!(modulus, modulus_from_hex, "canonical BLS12-381 base modulus");
    assert_eq!(modulus.bits(), 381);
    assert_eq!(&modulus & BigUint::from(1_u32), BigUint::from(1_u32));

    let one = BigUint::from(1_u32);
    let radix = power_of_two(RADIX_BITS);
    let radix_mask = &radix - &one;
    let four_p = &modulus << 2_usize;
    assert!(four_p < radix, "L4 operands must fit the 12-by-u32 ABI");

    let inverse = inverse_mod_power_of_two(&modulus, RADIX_BITS);
    let negative_inverse = &radix - &inverse;
    assert_eq!((&modulus * &inverse) & &radix_mask, one);
    assert_eq!((&modulus * &negative_inverse) & &radix_mask, radix_mask);

    let radix_inverse = (&radix % &modulus).modpow(&(&modulus - BigUint::from(2_u32)), &modulus);
    assert_eq!((&radix * &radix_inverse) % &modulus, BigUint::from(1_u32));
    let reference = RedcReference {
        modulus: &modulus,
        negative_inverse: &negative_inverse,
        radix: &radix,
        radix_inverse: &radix_inverse,
    };

    // These are exact worst-case integer bounds, independent of the sampled
    // operands. Canonical inputs produce raw REDC below 2p. Inputs below 4p
    // produce raw REDC below 3p, hence remain inside the L4 domain without a
    // final subtraction and need at most two subtractions when canonicalized.
    let canonical_numerator_bound = (&modulus - &one).pow(2_u32) + (&radix - &one) * &modulus;
    let canonical_raw_bound = canonical_numerator_bound / &radix;
    let two_p = &modulus << 1_usize;
    assert!(canonical_raw_bound < two_p);

    let l4_maximum = &four_p - &one;
    let l4_numerator_bound = l4_maximum.pow(2_u32) + (&radix - &one) * &modulus;
    let l4_raw_bound = l4_numerator_bound / &radix;
    let three_p = &modulus * BigUint::from(3_u32);
    assert!(l4_raw_bound < three_p);
    assert!(three_p < four_p);

    let canonical_boundary = canonical_boundary_values(&modulus);
    let mut canonical_subtraction_paths = [0_usize; 2];
    for (lhs_index, lhs) in canonical_boundary.iter().enumerate() {
        for (rhs_index, rhs) in canonical_boundary.iter().enumerate() {
            let context = format!("canonical boundary lhs={lhs_index} rhs={rhs_index}");
            let subtractions = reference.assert_matches(lhs, rhs, &two_p, 1, &context);
            canonical_subtraction_paths[subtractions] += 1;
        }
    }

    let l4_boundary = l4_boundary_values(&modulus);
    let mut l4_subtraction_paths = [0_usize; 3];
    for (lhs_index, lhs) in l4_boundary.iter().enumerate() {
        for (rhs_index, rhs) in l4_boundary.iter().enumerate() {
            let context = format!("L4 boundary lhs={lhs_index} rhs={rhs_index}");
            let subtractions = reference.assert_matches(lhs, rhs, &three_p, 2, &context);
            l4_subtraction_paths[subtractions] += 1;
        }
    }

    let mut state = RANDOM_SEED;
    for case_index in 0..RANDOM_CASES {
        let canonical_lhs = random_below(&mut state, &modulus);
        let canonical_rhs = random_below(&mut state, &modulus);
        let context = format!("canonical seed={RANDOM_SEED:#018x} case={case_index}");
        let subtractions =
            reference.assert_matches(&canonical_lhs, &canonical_rhs, &two_p, 1, &context);
        canonical_subtraction_paths[subtractions] += 1;

        let l4_lhs = random_below(&mut state, &four_p);
        let l4_rhs = random_below(&mut state, &four_p);
        let context = format!("L4 seed={RANDOM_SEED:#018x} case={case_index}");
        let subtractions = reference.assert_matches(&l4_lhs, &l4_rhs, &three_p, 2, &context);
        l4_subtraction_paths[subtractions] += 1;
    }

    assert!(canonical_subtraction_paths.into_iter().all(|count| count > 0));
    assert!(l4_subtraction_paths.into_iter().all(|count| count > 0));
}
