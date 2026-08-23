// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;

include!("../src/field_constants.rs");

const LIMBS: usize = 12;
const RANDOM_CASES: usize = 10_000;
const RANDOM_SEED: u64 = 0xf105_3810_2026_0715;
const FORCED_SUBTRACTION_LHS: [u32; LIMBS] = [
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
];
const FORCED_SUBTRACTION_RHS: [u32; LIMBS] = [
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
];

#[derive(Debug)]
struct NamedFiosOutput {
    limbs: [u32; LIMBS],
    final_top: u32,
    maximum_top: u32,
    did_subtract: bool,
}

fn montgomery_inverse_word(modulus: u32) -> u32 {
    assert_eq!(modulus & 1, 1, "Montgomery modulus must be odd");
    let mut inverse = 1_u32;
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2_u32.wrapping_sub(modulus.wrapping_mul(inverse)));
    }
    inverse.wrapping_neg()
}

fn mulhi(lhs: u32, rhs: u32) -> u32 {
    ((u64::from(lhs) * u64::from(rhs)) >> 32) as u32
}

fn checked_add2(lhs: u64, rhs: u64) -> u64 {
    lhs.checked_add(rhs).expect("FIOS two-term word overflow")
}

fn checked_add3(first: u64, second: u64, third: u64) -> u64 {
    checked_add2(first, second).checked_add(third).expect("FIOS three-term word overflow")
}

fn named_fios_montgomery_multiply(
    fixed: [u32; LIMBS],
    scanned: [u32; LIMBS],
    modulus: [u32; LIMBS],
    inverse: u32,
) -> NamedFiosOutput {
    // This mirrors the generated, straight-line Metal recurrence. Both operands
    // must be reduced Montgomery residues; the test checks that precondition at
    // every call site.
    let mut temporary = [0_u32; LIMBS + 1];
    let mut maximum_top = 0_u32;

    for digit in scanned {
        let mut word =
            checked_add2(u64::from(fixed[0]) * u64::from(digit), u64::from(temporary[0]));
        let mut low = word as u32;
        let mut product_carry = (word >> 32) as u32;
        let reduction = low.wrapping_mul(inverse);
        let mut reduction_carry = mulhi(reduction, modulus[0]) + u32::from(low != 0_u32);

        for fixed_index in 1..LIMBS {
            word = checked_add3(
                u64::from(fixed[fixed_index]) * u64::from(digit),
                u64::from(temporary[fixed_index]),
                u64::from(product_carry),
            );
            low = word as u32;
            product_carry = (word >> 32) as u32;
            word = checked_add3(
                u64::from(reduction) * u64::from(modulus[fixed_index]),
                u64::from(low),
                u64::from(reduction_carry),
            );
            temporary[fixed_index - 1] = word as u32;
            reduction_carry = (word >> 32) as u32;
        }

        let top = checked_add2(u64::from(temporary[LIMBS]), u64::from(product_carry));
        word = checked_add2(u64::from(top as u32), u64::from(reduction_carry));
        temporary[LIMBS - 1] = word as u32;
        temporary[LIMBS] = (top >> 32) as u32 + (word >> 32) as u32;
        assert!(temporary[LIMBS] <= 1, "FIOS top carry exceeds one word");
        maximum_top = maximum_top.max(temporary[LIMBS]);
    }

    let top = temporary[LIMBS];
    let mut borrow = 0_u32;
    let mut subtracted = [0_u32; LIMBS];
    for index in 0..LIMBS {
        let first = temporary[index].wrapping_sub(modulus[index]);
        subtracted[index] = first.wrapping_sub(borrow);
        let first_borrow = u32::from(temporary[index] < modulus[index]);
        let second_borrow = u32::from(first < borrow);
        borrow = first_borrow | second_borrow;
    }
    borrow = u32::from(top < borrow);
    let did_subtract = borrow == 0;
    let use_subtracted = 0_u32.wrapping_sub(borrow ^ 1_u32);
    let limbs = core::array::from_fn(|index| {
        (temporary[index] & !use_subtracted) | (subtracted[index] & use_subtracted)
    });

    NamedFiosOutput { limbs, final_top: top, maximum_top, did_subtract }
}

fn limbs_to_biguint(limbs: &[u32; LIMBS]) -> BigUint {
    BigUint::new(limbs.to_vec())
}

fn biguint_to_limbs(value: &BigUint) -> [u32; LIMBS] {
    let digits = value.to_u32_digits();
    assert!(digits.len() <= LIMBS, "BigUint exceeds the field ABI width");
    let mut limbs = [0_u32; LIMBS];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_reduced(state: &mut u64, modulus: &BigUint) -> [u32; LIMBS] {
    loop {
        let mut limbs = [0_u32; LIMBS];
        for pair in limbs.chunks_exact_mut(2) {
            let word = splitmix64(state);
            pair[0] = word as u32;
            pair[1] = (word >> 32) as u32;
        }
        limbs[LIMBS - 1] &= 0x1fff_ffff;
        if limbs_to_biguint(&limbs) < *modulus {
            return limbs;
        }
    }
}

fn assert_orientation(
    fixed: [u32; LIMBS],
    scanned: [u32; LIMBS],
    modulus_limbs: [u32; LIMBS],
    modulus: &BigUint,
    expected: [u32; LIMBS],
    case_index: usize,
    orientation: &str,
) -> bool {
    let inverse = montgomery_inverse_word(modulus_limbs[0]);
    let actual = named_fios_montgomery_multiply(fixed, scanned, modulus_limbs, inverse);

    assert_eq!(
        actual.final_top, 0,
        "seed={RANDOM_SEED:#018x} case={case_index} orientation={orientation}: final top word; fixed={fixed:x?} scanned={scanned:x?}"
    );
    assert!(
        actual.maximum_top <= 1,
        "seed={RANDOM_SEED:#018x} case={case_index} orientation={orientation}: top carry bound; fixed={fixed:x?} scanned={scanned:x?}"
    );
    assert_eq!(
        actual.limbs, expected,
        "seed={RANDOM_SEED:#018x} case={case_index} orientation={orientation}: Montgomery result; fixed={fixed:x?} scanned={scanned:x?}"
    );
    assert!(
        limbs_to_biguint(&actual.limbs) < *modulus,
        "seed={RANDOM_SEED:#018x} case={case_index} orientation={orientation}: result is not reduced"
    );
    actual.did_subtract
}

fn assert_matches_biguint(
    fixed: [u32; LIMBS],
    scanned: [u32; LIMBS],
    modulus_limbs: [u32; LIMBS],
    modulus: &BigUint,
    r_inverse: &BigUint,
    case_index: usize,
) -> [bool; 2] {
    let fixed_value = limbs_to_biguint(&fixed);
    let scanned_value = limbs_to_biguint(&scanned);
    assert!(&fixed_value < modulus, "case {case_index}: fixed operand is not reduced");
    assert!(&scanned_value < modulus, "case {case_index}: scanned operand is not reduced");

    let expected_value = ((&fixed_value * &scanned_value) * r_inverse) % modulus;
    let expected = biguint_to_limbs(&expected_value);
    [
        assert_orientation(
            fixed,
            scanned,
            modulus_limbs,
            modulus,
            expected,
            case_index,
            "fixed_scanned",
        ),
        assert_orientation(
            scanned,
            fixed,
            modulus_limbs,
            modulus,
            expected,
            case_index,
            "scanned_fixed",
        ),
    ]
}

fn boundary_values(modulus: &BigUint, radix: &BigUint) -> Vec<[u32; LIMBS]> {
    let one = BigUint::from(1_u32);
    let alternating_low = BigUint::new(
        (0..LIMBS)
            .map(|index| if index.is_multiple_of(2) { 0xaaaa_aaaa } else { 0x5555_5555 })
            .collect(),
    );
    let alternating_high = BigUint::new(
        (0..LIMBS)
            .map(|index| if index.is_multiple_of(2) { 0x5555_5555 } else { 0xaaaa_aaaa })
            .collect(),
    );
    let around_380 = &one << 380_usize;
    let mut values = vec![
        BigUint::from(0_u32),
        one.clone(),
        BigUint::from(2_u32),
        modulus - &one,
        modulus - BigUint::from(2_u32),
        modulus >> 1_usize,
        radix.clone(),
        radix * radix,
        &around_380 - &one,
        around_380.clone(),
        &around_380 + &one,
        alternating_low,
        alternating_high,
        limbs_to_biguint(&FORCED_SUBTRACTION_LHS),
        limbs_to_biguint(&FORCED_SUBTRACTION_RHS),
    ];
    for word_count in 1..LIMBS {
        let boundary = &one << (32 * word_count);
        values.push(&boundary - &one);
        values.push(boundary);
    }
    values.into_iter().map(|value| biguint_to_limbs(&(value % modulus))).collect()
}

#[test]
fn named_fios_recurrence_matches_exact_biguint_montgomery_arithmetic() {
    let modulus = BigUint::parse_bytes(FP_MODULUS_HEX.as_bytes(), 16).expect("valid modulus");
    let modulus_limbs = biguint_to_limbs(&modulus);
    let inverse = montgomery_inverse_word(modulus_limbs[0]);
    assert_eq!(modulus_limbs[0].wrapping_mul(inverse), u32::MAX);
    assert_eq!(2 * LIMBS * LIMBS, 288, "operand and correction matrix terms");

    let one = BigUint::from(1_u32);
    let radix = &one << (32 * LIMBS);
    assert!((&modulus << 1_usize) < radix, "BLS12-381 expects a zero final top word");
    let inverse_exponent = &modulus - BigUint::from(2_u32);
    let r_inverse = (&radix % &modulus).modpow(&inverse_exponent, &modulus);
    assert_eq!((&radix * &r_inverse) % &modulus, one);

    let boundary = boundary_values(&modulus, &radix);

    let mut case_index = 0_usize;
    let mut subtraction_paths = [0_usize; 2];
    let forced_subtraction = assert_matches_biguint(
        FORCED_SUBTRACTION_LHS,
        FORCED_SUBTRACTION_RHS,
        modulus_limbs,
        &modulus,
        &r_inverse,
        case_index,
    );
    assert_eq!(forced_subtraction, [true, true], "forced T - p path in both orientations");
    for did_subtract in forced_subtraction {
        subtraction_paths[usize::from(did_subtract)] += 1;
    }
    case_index += 1;
    for &fixed in &boundary {
        for &scanned in &boundary {
            for did_subtract in assert_matches_biguint(
                fixed,
                scanned,
                modulus_limbs,
                &modulus,
                &r_inverse,
                case_index,
            ) {
                subtraction_paths[usize::from(did_subtract)] += 1;
            }
            case_index += 1;
        }
    }

    let mut state = RANDOM_SEED;
    for _ in 0..RANDOM_CASES {
        let fixed = random_reduced(&mut state, &modulus);
        let scanned = random_reduced(&mut state, &modulus);
        for did_subtract in
            assert_matches_biguint(fixed, scanned, modulus_limbs, &modulus, &r_inverse, case_index)
        {
            subtraction_paths[usize::from(did_subtract)] += 1;
        }
        case_index += 1;
    }
    assert_eq!(case_index, 1 + boundary.len() * boundary.len() + RANDOM_CASES);
    assert!(subtraction_paths[0] > 0, "conditional subtraction never kept the raw result");
    assert!(subtraction_paths[1] > 0, "conditional subtraction never selected T - p");
}
