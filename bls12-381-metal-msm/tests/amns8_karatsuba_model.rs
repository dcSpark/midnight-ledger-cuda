// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::{BigInt, BigUint, Sign};

const N: usize = 8;
const LAMBDA: i128 = -2;
const PHI: i128 = 1_i128 << 64;
const DELTA: i128 = 10;
const SIGMA: i64 = 143_766_863_997_100;
const RHO: i64 = 4_600_539_647_907_200;

const M: [i64; N] = [
    80_042_099_451_959,
    -438_404_363_789,
    -104_528_439_513_579,
    91_613_899_373_051,
    -7_504_120_839_914,
    125_847_515_758_073,
    -143_766_863_997_100,
    -95_095_973_594_275,
];

// -M^-1 in (Z / 2^64 Z)[X] / (X^8 + 2).
const M_PRIME: [u64; N] = [
    0x0010_c6fb_6f1f_4907,
    0x7a70_8f07_2a6f_c213,
    0xe6af_80b1_5b4d_0bc4,
    0xa5ee_da87_343c_dd86,
    0x516f_cfe4_2de2_b795,
    0x3382_6e37_0315_ed0a,
    0xc7af_1c65_3aff_f478,
    0x1db4_899b_375f_66c0,
];

const FROZEN_BASIS: [[i64; N]; N] = [
    [
        165_287_713_379_363,
        -5_030_420_591_247,
        137_672_150_042_232,
        119_775_351_316_119,
        -20_239_437_355_587,
        63_135_815_810_068,
        -43_664_550_814_409,
        -11_848_715_286_990,
    ],
    [
        74_829_388_475_034,
        4_902_425_732_877,
        -73_494_590_381_457,
        -149_295_178_729_809,
        -30_065_716_019_627,
        -162_375_608_766_907,
        121_383_494_388_985,
        34_256_165_897_936,
    ],
    [
        -90_148_039_660_281,
        78_086_606_608_915,
        -92_905_410_826_002,
        1_904_264_076_559,
        175_110_925_282_580,
        -58_671_794_440_980,
        -134_358_479_080_627,
        -45_832_564_069_768,
    ],
    [
        -76_912_876_176_448,
        -33_191_872_534_315,
        150_407_466_557_905,
        182_487_051_264_124,
        -120_341_750_538_278,
        -20_111_442_497_217,
        -1_041_743_850_707,
        -14_144_723_400_719,
    ],
    [
        80_042_099_451_959,
        -438_404_363_789,
        -104_528_439_513_579,
        91_613_899_373_051,
        -7_504_120_839_914,
        125_847_515_758_073,
        -143_766_863_997_100,
        -95_095_973_594_275,
    ],
    [
        -566_399_222_159,
        -40_350_879_852_804,
        62_094_071_959_361,
        -57_809_274_215_128,
        26_607_722_801_234,
        -66_047_920_422_524,
        -72_688_522_983_329,
        -160_079_600_653_178,
    ],
    [
        52_008_642_367_261,
        -51_880_647_508_891,
        -12_296_912_151_884,
        41_816_739_565_574,
        8_488_413_809_640,
        90_751_379_147_199,
        -168_470_322_721_775,
        146_062_872_110_829,
    ],
    [
        -46_850_226_917_644,
        -149_969_062_194_116,
        -77_958_611_750_545,
        28_727_851_165_227,
        27_615_563_337_131,
        -124_805_771_907_366,
        157_911_587_397_819,
        56_639_535_506_051,
    ],
];

#[derive(Debug, Default, Eq, PartialEq)]
struct ProductCounts {
    coefficient_products: [usize; 3],
    mixed_u32_products: [usize; 3],
    max_q_transform_carry: i64,
}

impl ProductCounts {
    fn widened_u32_products(&self) -> usize {
        // All A*B and Q*M leaf products are exact widened products.  In the
        // low-word C*M' leaf, low*low is widened while the two cross products
        // contribute only their low words.
        self.mixed_u32_products[0] + self.coefficient_products[1] + self.mixed_u32_products[2]
    }

    fn wrapping_u32_products(&self) -> usize {
        2 * self.coefficient_products[1]
    }
}

fn modulus() -> BigUint {
    BigUint::parse_bytes(
        b"4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        10,
    )
    .expect("valid BLS12-381 base-field modulus")
}

fn gamma() -> BigUint {
    BigUint::parse_bytes(
        b"1320341115756334504903338613189055857314888584589395389559536083285462048544550127191972802724328330560637937311830",
        10,
    )
    .expect("valid AMNS evaluation point")
}

fn mod_positive(value: BigInt, modulus: &BigInt) -> BigInt {
    let residue = value % modulus;
    if residue.sign() == Sign::Minus { residue + modulus } else { residue }
}

fn evaluate(coefficients: &[i64; N]) -> BigUint {
    let p = BigInt::from_biguint(Sign::Plus, modulus());
    let x = BigInt::from_biguint(Sign::Plus, gamma());
    let mut accumulator = BigInt::from(0_u8);
    for coefficient in coefficients.iter().rev() {
        accumulator = mod_positive(accumulator * &x + BigInt::from(*coefficient), &p);
    }
    accumulator.to_biguint().expect("a positive residue converts to BigUint")
}

fn multiply_u32(left: u32, right: u32, count: &mut usize) -> u64 {
    *count += 1;
    u64::from(left) * u64::from(right)
}

/// Exact signed i64 multiplication using three unsigned 32x32 products.
///
/// The third product uses the signed differences of the two 32-bit halves;
/// their magnitudes still fit u32.  Recombination is widened, so this also
/// covers i64::MIN without relying on signed overflow.
fn multiply_signed_i64_three_u32(left: i64, right: i64, count: &mut usize) -> i128 {
    let negative = (left < 0) ^ (right < 0);
    let left_magnitude = left.unsigned_abs();
    let right_magnitude = right.unsigned_abs();
    let left_low = left_magnitude as u32;
    let left_high = (left_magnitude >> 32) as u32;
    let right_low = right_magnitude as u32;
    let right_high = (right_magnitude >> 32) as u32;

    let low = i128::from(multiply_u32(left_low, right_low, count));
    let high = i128::from(multiply_u32(left_high, right_high, count));
    let left_difference = i64::from(left_low) - i64::from(left_high);
    let right_difference = i64::from(right_low) - i64::from(right_high);
    let difference_magnitude = i128::from(multiply_u32(
        u32::try_from(left_difference.unsigned_abs()).expect("difference magnitude fits u32"),
        u32::try_from(right_difference.unsigned_abs()).expect("difference magnitude fits u32"),
        count,
    ));
    let signed_difference = if (left_difference < 0) ^ (right_difference < 0) {
        -difference_magnitude
    } else {
        difference_magnitude
    };
    let cross = low + high - signed_difference;
    let magnitude = low + (cross << 32) + (high << 64);
    let result = if negative { -magnitude } else { magnitude };
    assert_eq!(result, i128::from(left) * i128::from(right));
    result
}

/// Low 64 bits of u64 multiplication using the three relevant 32x32 products.
fn multiply_wrapping_u64_three_u32(left: u64, right: u64, count: &mut usize) -> u64 {
    let left_low = left as u32;
    let left_high = (left >> 32) as u32;
    let right_low = right as u32;
    let right_high = (right >> 32) as u32;
    let low = multiply_u32(left_low, right_low, count);
    let cross_low = multiply_u32(left_low, right_high, count)
        .wrapping_add(multiply_u32(left_high, right_low, count));
    let result = low.wrapping_add(cross_low << 32);
    assert_eq!(result, left.wrapping_mul(right));
    result
}

fn karatsuba_i128_with<F>(left: &[i128], right: &[i128], leaf: &mut F) -> Vec<i128>
where
    F: FnMut(i128, i128) -> i128,
{
    assert_eq!(left.len(), right.len());
    assert!(left.len().is_power_of_two());
    if left.len() == 1 {
        return vec![leaf(left[0], right[0])];
    }

    let split = left.len() / 2;
    let left_sum: Vec<_> = (0..split)
        .map(|index| {
            left[index].checked_add(left[split + index]).expect("K8 input transform fits i128")
        })
        .collect();
    let right_sum: Vec<_> = (0..split)
        .map(|index| {
            right[index].checked_add(right[split + index]).expect("K8 input transform fits i128")
        })
        .collect();

    let low = karatsuba_i128_with(&left[..split], &right[..split], leaf);
    let high = karatsuba_i128_with(&left[split..], &right[split..], leaf);
    let mut cross = karatsuba_i128_with(&left_sum, &right_sum, leaf);
    for (index, value) in low.iter().enumerate() {
        cross[index] = cross[index].checked_sub(*value).expect("K8 cross subtraction fits i128");
    }
    for (index, value) in high.iter().enumerate() {
        cross[index] = cross[index].checked_sub(*value).expect("K8 cross subtraction fits i128");
    }

    let mut result = vec![0_i128; 2 * left.len() - 1];
    for (index, value) in low.iter().enumerate() {
        result[index] = result[index].checked_add(*value).expect("K8 low sum fits i128");
    }
    for (index, value) in cross.iter().enumerate() {
        result[split + index] =
            result[split + index].checked_add(*value).expect("K8 cross sum fits i128");
    }
    for (index, value) in high.iter().enumerate() {
        result[2 * split + index] =
            result[2 * split + index].checked_add(*value).expect("K8 high sum fits i128");
    }
    result
}

fn karatsuba_u64_with<F>(left: &[u64], right: &[u64], leaf: &mut F) -> Vec<u64>
where
    F: FnMut(u64, u64) -> u64,
{
    assert_eq!(left.len(), right.len());
    assert!(left.len().is_power_of_two());
    if left.len() == 1 {
        return vec![leaf(left[0], right[0])];
    }

    let split = left.len() / 2;
    let left_sum: Vec<_> =
        (0..split).map(|index| left[index].wrapping_add(left[split + index])).collect();
    let right_sum: Vec<_> =
        (0..split).map(|index| right[index].wrapping_add(right[split + index])).collect();
    let low = karatsuba_u64_with(&left[..split], &right[..split], leaf);
    let high = karatsuba_u64_with(&left[split..], &right[split..], leaf);
    let mut cross = karatsuba_u64_with(&left_sum, &right_sum, leaf);
    for (index, value) in low.iter().enumerate() {
        cross[index] = cross[index].wrapping_sub(*value);
    }
    for (index, value) in high.iter().enumerate() {
        cross[index] = cross[index].wrapping_sub(*value);
    }

    let mut result = vec![0_u64; 2 * left.len() - 1];
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
    assert_eq!(product.len(), 2 * N - 1);
    for degree in (N..product.len()).rev() {
        let folded = product[degree].checked_mul(LAMBDA).expect("external K8 fold fits i128");
        product[degree - N] = product[degree - N]
            .checked_add(folded)
            .expect("externally reduced K8 coefficient fits i128");
    }
    core::array::from_fn(|index| product[index])
}

fn reduce_u64_mod_e(mut product: Vec<u64>) -> [u64; N] {
    assert_eq!(product.len(), 2 * N - 1);
    for degree in (N..product.len()).rev() {
        product[degree - N] = product[degree - N].wrapping_sub(product[degree].wrapping_mul(2));
    }
    core::array::from_fn(|index| product[index])
}

fn split_centered_phi(value: i128) -> (i64, i64) {
    let low = value.rem_euclid(PHI) as u64 as i64;
    let carry = (value - i128::from(low)) / PHI;
    assert_eq!(value, i128::from(low) + carry * PHI);
    (low, i64::try_from(carry).expect("K8 quotient-transform carry fits i64"))
}

fn multiply_q_transform_by_m_transform(
    quotient_transform: i128,
    m_transform: i128,
    counts: &mut ProductCounts,
) -> i128 {
    counts.coefficient_products[2] += 1;
    let (low, carry) = split_centered_phi(quotient_transform);
    counts.max_q_transform_carry = counts.max_q_transform_carry.max(carry.unsigned_abs() as i64);
    assert!(carry.unsigned_abs() <= 4, "an eight-term centered transform has |carry| <= 4");
    let m_signed = i64::try_from(m_transform).expect("every frozen M transform fits signed i64");
    let low_product =
        multiply_signed_i64_three_u32(low, m_signed, &mut counts.mixed_u32_products[2]);
    let carry_product = i128::from(carry)
        .checked_mul(i128::from(m_signed))
        .and_then(|value| value.checked_mul(PHI))
        .expect("range-managed Q*M leaf fits i128");
    let result = low_product.checked_add(carry_product).expect("Q*M leaf fits i128");
    assert_eq!(result, quotient_transform * m_transform);
    result
}

fn multiply_karatsuba(left: &[i64; N], right: &[i64; N]) -> ([i64; N], ProductCounts) {
    let mut counts = ProductCounts::default();
    let left_wide = left.map(i128::from);
    let right_wide = right.map(i128::from);
    let product = reduce_i128_mod_e(karatsuba_i128_with(
        &left_wide,
        &right_wide,
        &mut |left_transform, right_transform| {
            counts.coefficient_products[0] += 1;
            let left_signed =
                i64::try_from(left_transform).expect("lazy A transform fits signed i64");
            let right_signed =
                i64::try_from(right_transform).expect("lazy B transform fits signed i64");
            multiply_signed_i64_three_u32(
                left_signed,
                right_signed,
                &mut counts.mixed_u32_products[0],
            )
        },
    ));

    let product_low = product.map(|coefficient| coefficient.rem_euclid(PHI) as u64);
    let quotient_bits = reduce_u64_mod_e(karatsuba_u64_with(
        &product_low,
        &M_PRIME,
        &mut |left_transform, right_transform| {
            counts.coefficient_products[1] += 1;
            multiply_wrapping_u64_three_u32(
                left_transform,
                right_transform,
                &mut counts.mixed_u32_products[1],
            )
        },
    ));
    let quotient = quotient_bits.map(|coefficient| i128::from(coefficient as i64));
    let reduction_polynomial = M.map(i128::from);
    let correction = reduce_i128_mod_e(karatsuba_i128_with(
        &quotient,
        &reduction_polynomial,
        &mut |quotient_transform, m_transform| {
            multiply_q_transform_by_m_transform(quotient_transform, m_transform, &mut counts)
        },
    ));

    let output = core::array::from_fn(|index| {
        let numerator =
            product[index].checked_add(correction[index]).expect("AMNS numerator fits i128");
        assert_eq!(numerator.rem_euclid(PHI), 0, "coefficient {index} cancels modulo phi");
        i64::try_from(numerator / PHI).expect("reduced AMNS coefficient fits i64")
    });
    (output, counts)
}

fn multiply_mod_e_i128(left: &[i128; N], right: &[i128; N]) -> [i128; N] {
    let mut result = [0_i128; N];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.checked_mul(*right_value).expect("schoolbook leaf fits i128");
            let degree = left_index + right_index;
            let (destination, term) = if degree < N {
                (degree, product)
            } else {
                (degree - N, product.checked_mul(LAMBDA).expect("schoolbook fold fits i128"))
            };
            result[destination] =
                result[destination].checked_add(term).expect("schoolbook coefficient fits i128");
        }
    }
    result
}

fn multiply_mod_e_u64(left: &[u64; N], right: &[u64; N]) -> [u64; N] {
    let mut result = [0_u64; N];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let product = left_value.wrapping_mul(*right_value);
            let degree = left_index + right_index;
            if degree < N {
                result[degree] = result[degree].wrapping_add(product);
            } else {
                result[degree - N] = result[degree - N].wrapping_sub(product.wrapping_mul(2));
            }
        }
    }
    result
}

fn multiply_reference(left: &[i64; N], right: &[i64; N]) -> [i64; N] {
    let product = multiply_mod_e_i128(&left.map(i128::from), &right.map(i128::from));
    let product_low = product.map(|coefficient| coefficient.rem_euclid(PHI) as u64);
    let quotient_bits = multiply_mod_e_u64(&product_low, &M_PRIME);
    let quotient = quotient_bits.map(|coefficient| i128::from(coefficient as i64));
    let correction = multiply_mod_e_i128(&quotient, &M.map(i128::from));
    core::array::from_fn(|index| {
        let numerator = product[index] + correction[index];
        assert_eq!(numerator.rem_euclid(PHI), 0);
        i64::try_from(numerator / PHI).expect("reference output fits i64")
    })
}

fn multiplication_matrix(polynomial: &[i64; N]) -> [[i64; N]; N] {
    core::array::from_fn(|shift| {
        let mut row = [0_i64; N];
        for (index, original_coefficient) in polynomial.iter().enumerate() {
            let degree = shift + index;
            if degree < N {
                row[degree] += original_coefficient;
            } else {
                row[degree - N] -= 2 * original_coefficient;
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
            .expect("matrix is nonsingular");
        if pivot_row != pivot_index {
            matrix.swap(pivot_row, pivot_index);
            sign = -sign;
        }
        let pivot = matrix[pivot_index][pivot_index].clone();
        for row in pivot_index + 1..length {
            for column in pivot_index + 1..length {
                let numerator = &matrix[row][column] * &pivot
                    - &matrix[row][pivot_index] * &matrix[pivot_index][column];
                let (quotient, remainder) =
                    (numerator.clone() / &previous_pivot, numerator % &previous_pivot);
                assert_eq!(remainder, BigInt::from(0_u8));
                matrix[row][column] = quotient;
            }
        }
        previous_pivot = pivot;
    }
    sign * &matrix[length - 1][length - 1]
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn amns8_basis_inverse_coverage_and_bounds_are_self_consistent() {
    let p = modulus();
    let x = gamma();
    assert_eq!(x.modpow(&BigUint::from(8_u8), &p), &p - 2_u8);

    assert_eq!(evaluate(&M), BigUint::from(0_u8));
    assert!(FROZEN_BASIS.iter().all(|row| evaluate(row) == BigUint::from(0_u8)));
    let inverse_check = multiply_mod_e_u64(&M.map(|coefficient| coefficient as u64), &M_PRIME);
    assert_eq!(inverse_check, [u64::MAX, 0, 0, 0, 0, 0, 0, 0]);

    let basis_determinant = determinant_bareiss(
        FROZEN_BASIS
            .iter()
            .map(|row| row.iter().map(|value| BigInt::from(*value)).collect())
            .collect(),
    );
    assert_eq!(basis_determinant, BigInt::from_biguint(Sign::Plus, p.clone()));
    let max_row_l1 = FROZEN_BASIS
        .iter()
        .map(|row| row.iter().map(|value| value.unsigned_abs()).sum::<u64>())
        .max()
        .expect("basis has rows");
    let max_column_l1 = (0..N)
        .map(|column| FROZEN_BASIS.iter().map(|row| row[column].unsigned_abs()).sum::<u64>())
        .max()
        .expect("basis has columns");
    assert_eq!(max_row_l1, 677_018_084_045_712);
    assert_eq!(max_column_l1, 843_285_565_234_751);
    assert!(2 * RHO as u64 > max_row_l1);
    assert!(2 * RHO as u64 > max_column_l1);

    assert_eq!(M.iter().map(|value| value.unsigned_abs()).max(), Some(SIGMA as u64));
    assert_eq!(RHO, 32 * SIGMA);
    let capacity_floor = (PHI as u128) / (32 * RHO as u128);
    assert_eq!(capacity_floor, 125);
    assert!(capacity_floor >= (DELTA as u128 + 1).pow(2));

    let matrix = multiplication_matrix(&M);
    let determinant = determinant_bareiss(
        matrix.iter().map(|row| row.iter().map(|value| BigInt::from(*value)).collect()).collect(),
    );
    assert_eq!(determinant, BigInt::from(33_u8) * BigInt::from_biguint(Sign::Plus, p));

    let lazy_limit = (DELTA as u128 + 1) * (RHO as u128 - 1);
    let product_bound = 15 * lazy_limit * lazy_limit;
    let correction_bound = 15 * (1_u128 << 63) * SIGMA as u128;
    assert_eq!(product_bound, 38_414_411_569_318_461_597_832_836_586_465_815);
    assert_eq!(correction_bound, 19_890_229_098_257_336_436_485_208_932_352_000);
    assert!(product_bound + correction_bound < RHO as u128 * PHI as u128);

    // T_1=1 and T_2k <= 8*T_k for the held values in this recursive K8
    // implementation, hence T_8<=512.  Folding X^8=-2 costs at most 3.
    let karatsuba_numerator_bound =
        1_536 * (lazy_limit * lazy_limit + (1_u128 << 63) * SIGMA as u128);
    assert_eq!(karatsuba_numerator_bound, 5_970_395_204_359_761_718_714_167_861_126_944_256);
    assert!(karatsuba_numerator_bound <= i128::MAX as u128);
}

#[test]
fn signed_i64_leaf_model_is_exact_at_three_u32_products() {
    let boundaries = [i64::MIN, i64::MIN + 1, -1_i64, 0, 1, i64::MAX - 1, i64::MAX];
    for left in boundaries {
        for right in boundaries {
            let mut count = 0;
            let result = multiply_signed_i64_three_u32(left, right, &mut count);
            assert_eq!(result, i128::from(left) * i128::from(right));
            assert_eq!(count, 3);
        }
    }

    let mut state = 0x414d_4e53_384b_3634_u64;
    for _ in 0..1_024 {
        let left = next_random(&mut state) as i64;
        let right = next_random(&mut state) as i64;
        let mut count = 0;
        assert_eq!(
            multiply_signed_i64_three_u32(left, right, &mut count),
            i128::from(left) * i128::from(right)
        );
        assert_eq!(count, 3);
    }
}

#[test]
fn k8_is_exact_and_range_managed_q_transforms_keep_the_243_product_graph() {
    let mut state = 0x4b38_5f32_3758_3332_u64;
    for _ in 0..128 {
        // Divide by eight so every possible eight-term K8 input transform is
        // representable as i64 for this direct three-product leaf test.
        let left: [i128; N] =
            core::array::from_fn(|_| i128::from((next_random(&mut state) as i64) / 8));
        let right: [i128; N] =
            core::array::from_fn(|_| i128::from((next_random(&mut state) as i64) / 8));
        let mut coefficient_products = 0;
        let mut mixed_products = 0;
        let actual = karatsuba_i128_with(&left, &right, &mut |left_leaf, right_leaf| {
            coefficient_products += 1;
            multiply_signed_i64_three_u32(
                i64::try_from(left_leaf).expect("random K8 transform fits i64"),
                i64::try_from(right_leaf).expect("random K8 transform fits i64"),
                &mut mixed_products,
            )
        });
        let mut expected = vec![0_i128; 2 * N - 1];
        for (left_index, left_value) in left.iter().enumerate() {
            for (right_index, right_value) in right.iter().enumerate() {
                expected[left_index + right_index] += left_value * right_value;
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(coefficient_products, 27);
        assert_eq!(mixed_products, 81);
    }

    for quotient in [[i64::MIN; N], [i64::MAX; N]] {
        let quotient_wide = quotient.map(i128::from);
        let m_wide = M.map(i128::from);
        let mut counts = ProductCounts::default();
        let actual =
            karatsuba_i128_with(&quotient_wide, &m_wide, &mut |q_transform, m_transform| {
                multiply_q_transform_by_m_transform(q_transform, m_transform, &mut counts)
            });
        let mut expected = vec![0_i128; 2 * N - 1];
        for (q_index, q_value) in quotient_wide.iter().enumerate() {
            for (m_index, m_value) in m_wide.iter().enumerate() {
                expected[q_index + m_index] += q_value * m_value;
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(counts.coefficient_products[2], 27);
        assert_eq!(counts.mixed_u32_products[2], 81);
        assert_eq!(counts.max_q_transform_carry, 4);
    }

    // Three K8 products, each with 27 scalar leaves and three mixed u32
    // products per range-managed leaf.
    assert_eq!(3 * 27, 81);
    assert_eq!(3 * 27 * 3, 243);
}

#[test]
fn amns8_multiplication_is_exact_closes_lazy_inputs_and_counts_243_products() {
    let p = modulus();
    let phi_inverse = BigUint::from(1_u128 << 64).modpow(&(p.clone() - 2_u8), &p);
    let lazy_limit = (DELTA + 1) as i64 * (RHO - 1);

    let mut cases = vec![[0_i64; N]];
    for value in [1, -1, RHO - 1, 1 - RHO, lazy_limit, -lazy_limit] {
        let mut single = [0_i64; N];
        single[0] = value;
        cases.push(single);
        cases.push([value; N]);
        cases.push(core::array::from_fn(|index| if index % 2 == 0 { value } else { -value }));
    }

    let mut state = 0x414d_4e53_385f_4b38_u64;
    let span = u64::try_from(2 * lazy_limit + 1).expect("lazy input range fits u64");
    for _ in 0..512 {
        cases.push(core::array::from_fn(|_| {
            i64::try_from(next_random(&mut state) % span).expect("sample fits i64") - lazy_limit
        }));
    }

    for (index, left) in cases.iter().enumerate() {
        let right = &cases[(index * 73 + 19) % cases.len()];
        let (output, counts) = multiply_karatsuba(left, right);
        assert_eq!(output, multiply_reference(left, right), "reference case {index}");
        assert_eq!(counts.coefficient_products, [27, 27, 27]);
        assert_eq!(counts.mixed_u32_products, [81, 81, 81]);
        assert_eq!(counts.widened_u32_products(), 189);
        assert_eq!(counts.wrapping_u32_products(), 54);
        assert!(counts.max_q_transform_carry <= 4);
        assert!(
            output.iter().all(|coefficient| coefficient.unsigned_abs() < RHO as u64),
            "case {index} escaped the AMNS coefficient bound"
        );
        let expected = evaluate(left) * evaluate(right) % &p * &phi_inverse % &p;
        assert_eq!(evaluate(&output), expected, "field identity case {index}");
    }

    let mut left = cases[7];
    let right = cases[29];
    for iteration in 0..64 {
        let expected = evaluate(&left) * evaluate(&right) % &p * &phi_inverse % &p;
        let (output, counts) = multiply_karatsuba(&left, &right);
        assert_eq!(counts.coefficient_products, [27, 27, 27]);
        assert_eq!(counts.mixed_u32_products, [81, 81, 81]);
        assert_eq!(counts.widened_u32_products(), 189);
        assert_eq!(counts.wrapping_u32_products(), 54);
        left = output;
        assert_eq!(evaluate(&left), expected, "chain iteration {iteration}");
        assert!(left.iter().all(|coefficient| coefficient.unsigned_abs() < RHO as u64));
    }
}
