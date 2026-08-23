// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::{BigInt, BigUint, Sign};

const N: usize = 25;
const LAMBDA: i128 = 2;
const PHI: i128 = 1_i128 << 32;
const DELTA: i128 = 2;
const RHO: i64 = 410_240;

const M: [i64; N] = [
    5_265, -6_042, -550, -3_841, -46_309, -3_691, -13_917, 54, 11_281, -18_175, -73, 6_377, 11_698,
    -9_308, 1_330, 2_745, -40, 8_179, -3_550, -11_129, -22_495, 9_238, 6_654, 3_275, 2_537,
];

// -M^-1 in (Z / 2^32 Z)[X] / (X^25 - 2).
const M_PRIME: [u32; N] = [
    0xfc3f_8f21,
    0xb502_f538,
    0xd6ad_15a0,
    0xb9fd_d013,
    0xd5c0_24e5,
    0xd67a_f7eb,
    0x3908_b0fa,
    0xdc54_86d6,
    0x8410_e10c,
    0x57a3_85a2,
    0xee49_d0dd,
    0x095d_a865,
    0x219d_03da,
    0x1f1e_89b8,
    0xec61_750d,
    0xdc1f_27d7,
    0x0db5_34ed,
    0x1c29_ae95,
    0xd8d4_f6c4,
    0xab92_ad15,
    0x41a9_74f7,
    0xd185_f4d0,
    0xe2a9_65b3,
    0x8981_f32c,
    0xf03c_c984,
];

fn modulus() -> BigUint {
    BigUint::parse_bytes(
        b"4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        10,
    )
    .expect("valid BLS12-381 base-field modulus")
}

fn gamma() -> BigUint {
    BigUint::parse_bytes(
        b"511066730143548652923330221406862432842513716081412796986393940547425281101465886366786490203512647746495335967402",
        10,
    )
    .expect("valid PMNS evaluation point")
}

fn modulus_bigint() -> BigInt {
    BigInt::from_biguint(Sign::Plus, modulus())
}

fn mod_positive(value: BigInt, modulus: &BigInt) -> BigInt {
    let residue = value % modulus;
    if residue.sign() == Sign::Minus { residue + modulus } else { residue }
}

fn evaluate(coefficients: &[i64; N]) -> BigUint {
    let p = modulus_bigint();
    let x = BigInt::from_biguint(Sign::Plus, gamma());
    let mut accumulator = BigInt::from(0_u8);
    for coefficient in coefficients.iter().rev() {
        accumulator = mod_positive(accumulator * &x + BigInt::from(*coefficient), &p);
    }
    accumulator.to_biguint().expect("a positive residue converts to BigUint")
}

fn multiply_mod_e_i128(left: &[i128; N], right: &[i128; N]) -> [i128; N] {
    let mut result = [0_i128; N];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.checked_mul(*right_value).expect("PMNS product fits i128");
            let degree = left_index + right_index;
            if degree < N {
                result[degree] =
                    result[degree].checked_add(product).expect("PMNS coefficient fits i128");
            } else {
                result[degree - N] = result[degree - N]
                    .checked_add(LAMBDA * product)
                    .expect("externally reduced PMNS coefficient fits i128");
            }
        }
    }
    result
}

fn multiply_mod_e_u32(left: &[u32; N], right: &[u32; N]) -> [u32; N] {
    let mut result = [0_u32; N];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.wrapping_mul(*right_value);
            let degree = left_index + right_index;
            if degree < N {
                result[degree] = result[degree].wrapping_add(product);
            } else {
                result[degree - N] = result[degree - N].wrapping_add(product.wrapping_mul(2));
            }
        }
    }
    result
}

fn karatsuba_split(length: usize) -> usize {
    let mut counts = vec![0_usize; length + 1];
    let mut splits = vec![0_usize; length + 1];
    counts[1] = 1;
    for current_length in 2..=length {
        let (count, split) = (1..current_length)
            .map(|low_length| {
                let high_length = current_length - low_length;
                (
                    counts[low_length] + counts[high_length] + counts[low_length.max(high_length)],
                    low_length,
                )
            })
            .min()
            .expect("a nontrivial product has a split");
        counts[current_length] = count;
        splits[current_length] = split;
    }
    splits[length]
}

fn karatsuba_i128(left: &[i128], right: &[i128], products: &mut usize) -> Vec<i128> {
    assert_eq!(left.len(), right.len());
    if left.len() == 1 {
        *products += 1;
        return vec![left[0].checked_mul(right[0]).expect("Karatsuba leaf fits i128")];
    }

    let split = karatsuba_split(left.len());
    let sum_length = split.max(left.len() - split);
    let mut left_sum = vec![0_i128; sum_length];
    let mut right_sum = vec![0_i128; sum_length];
    for index in 0..sum_length {
        if index < split {
            left_sum[index] += left[index];
            right_sum[index] += right[index];
        }
        if split + index < left.len() {
            left_sum[index] += left[split + index];
            right_sum[index] += right[split + index];
        }
    }

    let low = karatsuba_i128(&left[..split], &right[..split], products);
    let high = karatsuba_i128(&left[split..], &right[split..], products);
    let mut cross = karatsuba_i128(&left_sum, &right_sum, products);
    for (index, value) in low.iter().enumerate() {
        cross[index] -= value;
    }
    for (index, value) in high.iter().enumerate() {
        cross[index] -= value;
    }

    let mut result = vec![0_i128; 2 * left.len() - 1];
    for (index, value) in low.iter().enumerate() {
        result[index] += value;
    }
    for (index, value) in cross.iter().enumerate() {
        result[split + index] += value;
    }
    for (index, value) in high.iter().enumerate() {
        result[2 * split + index] += value;
    }
    result
}

fn karatsuba_u32(left: &[u32], right: &[u32], products: &mut usize) -> Vec<u32> {
    assert_eq!(left.len(), right.len());
    if left.len() == 1 {
        *products += 1;
        return vec![left[0].wrapping_mul(right[0])];
    }

    let split = karatsuba_split(left.len());
    let sum_length = split.max(left.len() - split);
    let mut left_sum = vec![0_u32; sum_length];
    let mut right_sum = vec![0_u32; sum_length];
    for index in 0..sum_length {
        if index < split {
            left_sum[index] = left_sum[index].wrapping_add(left[index]);
            right_sum[index] = right_sum[index].wrapping_add(right[index]);
        }
        if split + index < left.len() {
            left_sum[index] = left_sum[index].wrapping_add(left[split + index]);
            right_sum[index] = right_sum[index].wrapping_add(right[split + index]);
        }
    }

    let low = karatsuba_u32(&left[..split], &right[..split], products);
    let high = karatsuba_u32(&left[split..], &right[split..], products);
    let mut cross = karatsuba_u32(&left_sum, &right_sum, products);
    for (index, value) in low.iter().enumerate() {
        cross[index] = cross[index].wrapping_sub(*value);
    }
    for (index, value) in high.iter().enumerate() {
        cross[index] = cross[index].wrapping_sub(*value);
    }

    let mut result = vec![0_u32; 2 * left.len() - 1];
    for (index, value) in low.iter().enumerate() {
        result[index] = result[index].wrapping_add(*value);
    }
    for (index, value) in cross.iter().enumerate() {
        result[split + index] = result[split + index].wrapping_add(*value);
    }
    for (index, value) in high.iter().enumerate() {
        result[2 * split + index] = result[2 * split + index].wrapping_add(*value);
    }
    result
}

fn reduce_i128_mod_e(mut product: Vec<i128>) -> [i128; N] {
    for degree in (N..product.len()).rev() {
        let high = product[degree];
        product[degree - N] += LAMBDA * high;
    }
    core::array::from_fn(|index| product[index])
}

fn reduce_u32_mod_e(mut product: Vec<u32>) -> [u32; N] {
    for degree in (N..product.len()).rev() {
        let high = product[degree];
        product[degree - N] = product[degree - N].wrapping_add(high.wrapping_mul(2));
    }
    core::array::from_fn(|index| product[index])
}

fn multiply_karatsuba(left: &[i64; N], right: &[i64; N]) -> ([i64; N], [usize; 3]) {
    let left_wide = left.map(i128::from);
    let right_wide = right.map(i128::from);
    let mut product_counts = [0_usize; 3];
    let product =
        reduce_i128_mod_e(karatsuba_i128(&left_wide, &right_wide, &mut product_counts[0]));

    let product_low = product.map(|coefficient| coefficient.rem_euclid(PHI) as u32);
    let quotient_bits =
        reduce_u32_mod_e(karatsuba_u32(&product_low, &M_PRIME, &mut product_counts[1]));
    let quotient = quotient_bits.map(|coefficient| i128::from(coefficient as i32));
    let reduction_polynomial = M.map(i128::from);
    let correction =
        reduce_i128_mod_e(karatsuba_i128(&quotient, &reduction_polynomial, &mut product_counts[2]));

    let output = core::array::from_fn(|index| {
        let numerator = product[index] + correction[index];
        assert_eq!(numerator.rem_euclid(PHI), 0);
        i64::try_from(numerator / PHI).expect("reduced PMNS coefficient fits i64")
    });
    (output, product_counts)
}

fn multiply(left: &[i64; N], right: &[i64; N]) -> [i64; N] {
    let left_wide = left.map(i128::from);
    let right_wide = right.map(i128::from);
    let product = multiply_mod_e_i128(&left_wide, &right_wide);

    let product_low = product.map(|coefficient| coefficient.rem_euclid(PHI) as u32);
    let quotient_bits = multiply_mod_e_u32(&product_low, &M_PRIME);
    let quotient = quotient_bits.map(|coefficient| i128::from(coefficient as i32));
    let reduction_polynomial = M.map(i128::from);
    let correction = multiply_mod_e_i128(&quotient, &reduction_polynomial);

    core::array::from_fn(|index| {
        let numerator =
            product[index].checked_add(correction[index]).expect("PMNS correction fits i128");
        assert_eq!(numerator.rem_euclid(PHI), 0, "coefficient {index} does not cancel modulo phi");
        i64::try_from(numerator / PHI).expect("reduced PMNS coefficient fits i64")
    })
}

fn multiplication_matrix() -> [[i64; N]; N] {
    core::array::from_fn(|shift| {
        let mut row = [0_i64; N];
        for (index, coefficient) in M.iter().enumerate() {
            let degree = shift + index;
            if degree < N {
                row[degree] += coefficient;
            } else {
                row[degree - N] += 2 * coefficient;
            }
        }
        row
    })
}

fn determinant_bareiss(mut matrix: Vec<Vec<BigInt>>) -> BigInt {
    let length = matrix.len();
    let mut sign = BigInt::from(1_i8);
    let mut previous_pivot = BigInt::from(1_i8);

    for pivot_index in 0..length - 1 {
        let pivot_row = (pivot_index..length)
            .find(|row| matrix[*row][pivot_index] != BigInt::from(0_u8))
            .expect("multiplication matrix is nonsingular");
        if pivot_row != pivot_index {
            matrix.swap(pivot_row, pivot_index);
            sign = -sign;
        }
        let pivot = matrix[pivot_index][pivot_index].clone();
        for row in pivot_index + 1..length {
            for column in pivot_index + 1..length {
                let numerator = &matrix[row][column] * &pivot
                    - &matrix[row][pivot_index] * &matrix[pivot_index][column];
                matrix[row][column] = numerator / &previous_pivot;
            }
        }
        previous_pivot = pivot;
    }
    sign * &matrix[length - 1][length - 1]
}

fn karatsuba_product_count(length: usize) -> usize {
    let mut counts = vec![0_usize; length + 1];
    counts[1] = 1;
    for current_length in 2..=length {
        counts[current_length] = (1..current_length)
            .map(|low_length| {
                let high_length = current_length - low_length;
                counts[low_length] + counts[high_length] + counts[low_length.max(high_length)]
            })
            .min()
            .expect("a nontrivial product has a split");
    }
    counts[length]
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn pmns_basis_and_reduction_constants_are_self_consistent() {
    let p = modulus();
    let x = gamma();
    assert_eq!((x.modpow(&BigUint::from(25_u8), &p) + &p - 2_u8) % &p, BigUint::from(0_u8));

    assert_eq!(evaluate(&M), BigUint::from(0_u8));
    let inverse_check = multiply_mod_e_u32(&M.map(|value| value as u32), &M_PRIME);
    assert_eq!(inverse_check[0], u32::MAX);
    assert!(inverse_check[1..].iter().all(|coefficient| *coefficient == 0));

    let matrix = multiplication_matrix();
    let norm = (0..N)
        .map(|column| (0..N).map(|row| matrix[row][column].unsigned_abs()).sum::<u64>())
        .max()
        .expect("the matrix has columns");
    assert_eq!(norm - 1, RHO as u64);
    let required_phi = 2 * 49 * i128::from(RHO - 1) * (DELTA + 1).pow(2);
    assert_eq!(required_phi, 361_830_798);
    assert!(required_phi < PHI);

    let determinant = determinant_bareiss(
        matrix.iter().map(|row| row.iter().map(|value| BigInt::from(*value)).collect()).collect(),
    );
    assert_eq!(determinant, BigInt::from(-19_431_i64) * BigInt::from_biguint(Sign::Plus, p));
}

#[test]
fn pmns_multiplication_is_exact_and_closes_lazy_inputs() {
    let p = modulus();
    let phi_inverse = BigUint::from(1_u64 << 32).modpow(&(p.clone() - 2_u8), &p);
    let lazy_limit = (DELTA + 1) as i64 * (RHO - 1);

    let mut cases = Vec::new();
    cases.push([0_i64; N]);
    for value in [1, -1, RHO - 1, 1 - RHO, lazy_limit, -lazy_limit] {
        let mut single = [0_i64; N];
        single[0] = value;
        cases.push(single);
        cases.push([value; N]);
        cases.push(core::array::from_fn(|index| if index % 2 == 0 { value } else { -value }));
    }

    let mut state = 0x504d_4e53_4b32_354d_u64;
    for _ in 0..512 {
        cases.push(core::array::from_fn(|_| {
            let span = u64::try_from(2 * lazy_limit + 1).expect("range fits u64");
            i64::try_from(next_random(&mut state) % span).expect("sample fits i64") - lazy_limit
        }));
    }

    for (index, left) in cases.iter().enumerate() {
        let right = &cases[(index * 73 + 19) % cases.len()];
        let output = multiply(left, right);
        let (karatsuba_output, product_counts) = multiply_karatsuba(left, right);
        assert_eq!(karatsuba_output, output, "Karatsuba case {index}");
        assert_eq!(product_counts, [205, 205, 205]);
        assert!(
            output.iter().all(|coefficient| coefficient.unsigned_abs() < RHO as u64),
            "case {index} escaped the PMNS coefficient bound"
        );
        let expected = evaluate(left) * evaluate(right) % &p * &phi_inverse % &p;
        assert_eq!(evaluate(&output), expected, "case {index}");
    }

    let mut left = cases[7];
    let right = cases[29];
    for iteration in 0..64 {
        let expected = evaluate(&left) * evaluate(&right) % &p * &phi_inverse % &p;
        left = multiply(&left, &right);
        assert_eq!(evaluate(&left), expected, "chain iteration {iteration}");
        assert!(left.iter().all(|coefficient| coefficient.unsigned_abs() < RHO as u64));
    }
}

#[test]
fn frozen_three_karatsuba_graph_has_615_coefficient_products() {
    assert_eq!(karatsuba_product_count(25), 205);
    assert_eq!(3 * karatsuba_product_count(25), 615);

    // This count combines 410 exact signed-widening leaves with 205 wrapping
    // low-word leaves. It is not homogeneous with the historical wrapping-u32
    // issue probe, so no throughput conclusion is encoded here.
    assert_eq!(2 * 205 + 205, 615);
}
