// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

pub(super) const COEFFICIENTS: usize = 8;
pub(super) const RHO: i64 = 4_600_539_647_907_200;
pub(super) const LAZY_LIMIT: i64 = 11 * (RHO - 1);

const PHI: i128 = 1_i128 << 64;
const M: [i64; COEFFICIENTS] = [
    80_042_099_451_959,
    -438_404_363_789,
    -104_528_439_513_579,
    91_613_899_373_051,
    -7_504_120_839_914,
    125_847_515_758_073,
    -143_766_863_997_100,
    -95_095_973_594_275,
];
const M_PRIME: [u64; COEFFICIENTS] = [
    0x0010_c6fb_6f1f_4907,
    0x7a70_8f07_2a6f_c213,
    0xe6af_80b1_5b4d_0bc4,
    0xa5ee_da87_343c_dd86,
    0x516f_cfe4_2de2_b795,
    0x3382_6e37_0315_ed0a,
    0xc7af_1c65_3aff_f478,
    0x1db4_899b_375f_66c0,
];

pub(super) fn is_lazy(value: &[i64; COEFFICIENTS]) -> bool {
    value.iter().all(|coefficient| coefficient.unsigned_abs() <= LAZY_LIMIT as u64)
}

fn multiply_i128_mod_e(
    left: &[i128; COEFFICIENTS],
    right: &[i128; COEFFICIENTS],
) -> [i128; COEFFICIENTS] {
    let mut result = [0_i128; COEFFICIENTS];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.checked_mul(*right_value).expect("AMNS leaf fits i128");
            let degree = left_index + right_index;
            let (destination, term) = if degree < COEFFICIENTS {
                (degree, product)
            } else {
                (
                    degree - COEFFICIENTS,
                    product.checked_mul(-2).expect("AMNS external fold fits i128"),
                )
            };
            result[destination] =
                result[destination].checked_add(term).expect("AMNS coefficient fits i128");
        }
    }
    result
}

fn multiply_u64_mod_e(
    left: &[u64; COEFFICIENTS],
    right: &[u64; COEFFICIENTS],
) -> [u64; COEFFICIENTS] {
    let mut result = [0_u64; COEFFICIENTS];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.wrapping_mul(*right_value);
            let degree = left_index + right_index;
            if degree < COEFFICIENTS {
                result[degree] = result[degree].wrapping_add(product);
            } else {
                result[degree - COEFFICIENTS] =
                    result[degree - COEFFICIENTS].wrapping_sub(product.wrapping_mul(2));
            }
        }
    }
    result
}

pub(super) fn multiply(
    left: &[i64; COEFFICIENTS],
    right: &[i64; COEFFICIENTS],
) -> [i64; COEFFICIENTS] {
    assert!(is_lazy(left));
    assert!(is_lazy(right));
    let product = multiply_i128_mod_e(&left.map(i128::from), &right.map(i128::from));
    let quotient_bits = multiply_u64_mod_e(
        &product.map(|coefficient| coefficient.rem_euclid(PHI) as u64),
        &M_PRIME,
    );
    let correction = multiply_i128_mod_e(
        &quotient_bits.map(|coefficient| i128::from(coefficient as i64)),
        &M.map(i128::from),
    );
    let output = core::array::from_fn(|index| {
        let numerator = product[index]
            .checked_add(correction[index])
            .expect("AMNS Montgomery numerator fits i128");
        assert_eq!(numerator.rem_euclid(PHI), 0);
        i64::try_from(numerator / PHI).expect("reduced AMNS coefficient fits i64")
    });
    assert!(output.iter().all(|coefficient| coefficient.unsigned_abs() < RHO as u64));
    output
}

pub(super) fn rotating_chain(
    mut left: [i64; COEFFICIENTS],
    mut right: [i64; COEFFICIENTS],
    iterations: u32,
) -> [i64; COEFFICIENTS] {
    assert!(iterations > 0);
    for _ in 0..iterations {
        let product = multiply(&left, &right);
        left = right;
        right = product;
    }
    right
}
