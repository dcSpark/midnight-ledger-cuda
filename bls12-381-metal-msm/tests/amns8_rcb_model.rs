// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Affine {
    x: BigUint,
    y: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Homogeneous {
    x: BigUint,
    y: BigUint,
    z: BigUint,
}

fn modulus() -> BigUint {
    BigUint::parse_bytes(
        b"4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        10,
    )
    .expect("valid BLS12-381 base-field modulus")
}

fn parse_hex(value: &[u8]) -> BigUint {
    BigUint::parse_bytes(value, 16).expect("valid hexadecimal field element")
}

fn generator() -> Affine {
    Affine {
        x: parse_hex(
            b"17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb",
        ),
        y: parse_hex(
            b"08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
        ),
    }
}

fn kappa() -> BigUint {
    parse_hex(
        b"18cccb8f6dbc6d2c6a3f657c444c53a4aba6efa5332d0d60549e1bc9ae14c29be771dd57afac430286a29ef641420cc6",
    )
}

fn add(left: &BigUint, right: &BigUint, p: &BigUint) -> BigUint {
    (left + right) % p
}

fn sub(left: &BigUint, right: &BigUint, p: &BigUint) -> BigUint {
    if left >= right { (left - right) % p } else { (left + p - right) % p }
}

fn mul(left: &BigUint, right: &BigUint, p: &BigUint) -> BigUint {
    left * right % p
}

fn scale(value: &BigUint, factor: u32, p: &BigUint) -> BigUint {
    value * factor % p
}

fn negate(value: &BigUint, p: &BigUint) -> BigUint {
    if value == &BigUint::from(0_u8) { value.clone() } else { p - value }
}

fn inverse(value: &BigUint, p: &BigUint) -> BigUint {
    assert_ne!(value, &BigUint::from(0_u8));
    value.modpow(&(p - 2_u8), p)
}

fn is_on_curve(point: &Affine, b: &BigUint, p: &BigUint) -> bool {
    mul(&point.y, &point.y, p) == add(&mul(&mul(&point.x, &point.x, p), &point.x, p), b, p)
}

fn affine_add(left: Option<&Affine>, right: Option<&Affine>, p: &BigUint) -> Option<Affine> {
    match (left, right) {
        (None, None) => None,
        (Some(point), None) | (None, Some(point)) => Some(point.clone()),
        (Some(left), Some(right)) => {
            if left.x == right.x && add(&left.y, &right.y, p) == BigUint::from(0_u8) {
                return None;
            }
            let slope = if left == right {
                if left.y == BigUint::from(0_u8) {
                    return None;
                }
                mul(&scale(&mul(&left.x, &left.x, p), 3, p), &inverse(&scale(&left.y, 2, p), p), p)
            } else {
                mul(&sub(&right.y, &left.y, p), &inverse(&sub(&right.x, &left.x, p), p), p)
            };
            let x = sub(&sub(&mul(&slope, &slope, p), &left.x, p), &right.x, p);
            let y = sub(&mul(&slope, &sub(&left.x, &x, p), p), &left.y, p);
            Some(Affine { x, y })
        }
    }
}

fn scalar_multiply(point: &Affine, mut scalar: u64, p: &BigUint) -> Option<Affine> {
    let mut accumulator = None;
    let mut addend = Some(point.clone());
    while scalar != 0 {
        if scalar & 1 != 0 {
            accumulator = affine_add(accumulator.as_ref(), addend.as_ref(), p);
        }
        addend = affine_add(addend.as_ref(), addend.as_ref(), p);
        scalar >>= 1;
    }
    accumulator
}

fn to_homogeneous(point: Option<&Affine>, scale_factor: &BigUint, p: &BigUint) -> Homogeneous {
    match point {
        None => {
            Homogeneous { x: BigUint::from(0_u8), y: BigUint::from(1_u8), z: BigUint::from(0_u8) }
        }
        Some(point) => Homogeneous {
            x: mul(&point.x, scale_factor, p),
            y: mul(&point.y, scale_factor, p),
            z: scale_factor % p,
        },
    }
}

fn to_affine(point: &Homogeneous, p: &BigUint) -> Option<Affine> {
    if point.z == BigUint::from(0_u8) {
        None
    } else {
        let z_inverse = inverse(&point.z, p);
        Some(Affine { x: mul(&point.x, &z_inverse, p), y: mul(&point.y, &z_inverse, p) })
    }
}

fn rcb_standard(left: &Homogeneous, right: &Homogeneous, p: &BigUint) -> Homogeneous {
    let mut t0 = mul(&left.x, &right.x, p);
    let mut t1 = mul(&left.y, &right.y, p);
    let mut t2 = mul(&left.z, &right.z, p);
    let mut t3 = mul(&add(&left.x, &left.y, p), &add(&right.x, &right.y, p), p);
    t3 = sub(&t3, &add(&t0, &t1, p), p);
    let mut t4 = mul(&add(&left.y, &left.z, p), &add(&right.y, &right.z, p), p);
    t4 = sub(&t4, &add(&t1, &t2, p), p);
    let mut t5 = mul(&add(&left.x, &left.z, p), &add(&right.x, &right.z, p), p);
    t5 = sub(&t5, &add(&t0, &t2, p), p);
    t0 = scale(&t0, 3, p);
    t2 = negate(&scale(&t2, 9, p), p);
    let mut z = add(&t1, &t2, p);
    t1 = sub(&t1, &t2, p);
    t5 = negate(&scale(&t5, 9, p), p);
    let mut x = sub(&mul(&t3, &t1, p), &mul(&t4, &t5, p), p);
    let y = add(&mul(&t1, &z, p), &mul(&t5, &t0, p), p);
    z = add(&mul(&z, &t4, p), &mul(&t0, &t3, p), p);
    if z == BigUint::from(0_u8) {
        x = BigUint::from(0_u8);
        Homogeneous { x, y: BigUint::from(1_u8), z }
    } else {
        Homogeneous { x, y, z }
    }
}

fn rcb_expanded(left: &Homogeneous, right: &Homogeneous, p: &BigUint) -> Homogeneous {
    let a = mul(&left.x, &right.x, p);
    let b = mul(&left.y, &right.y, p);
    let c = mul(&left.z, &right.z, p);
    let d =
        sub(&mul(&add(&left.x, &left.y, p), &add(&right.x, &right.y, p), p), &add(&a, &b, p), p);
    let e =
        sub(&mul(&add(&left.y, &left.z, p), &add(&right.y, &right.z, p), p), &add(&b, &c, p), p);
    let f =
        sub(&mul(&add(&left.x, &left.z, p), &add(&right.x, &right.z, p), p), &add(&a, &c, p), p);

    let mut x = add(&mul(&d, &b, p), &scale(&mul(&d, &c, p), 9, p), p);
    x = add(&x, &scale(&mul(&e, &f, p), 9, p), p);
    let mut y = sub(&mul(&b, &b, p), &scale(&mul(&c, &c, p), 81, p), p);
    y = sub(&y, &scale(&mul(&f, &a, p), 27, p), p);
    let mut z = sub(&mul(&e, &b, p), &scale(&mul(&e, &c, p), 9, p), p);
    z = add(&z, &scale(&mul(&a, &d, p), 3, p), p);
    if z == BigUint::from(0_u8) {
        Homogeneous { x: BigUint::from(0_u8), y: BigUint::from(1_u8), z: BigUint::from(0_u8) }
    } else {
        Homogeneous { x, y, z }
    }
}

fn map_to_minus_three(point: Option<&Affine>, p: &BigUint) -> Option<Affine> {
    point.map(|point| {
        let kappa = kappa();
        let kappa_squared = mul(&kappa, &kappa, p);
        let kappa_cubed = mul(&kappa_squared, &kappa, p);
        Affine {
            x: mul(&point.x, &inverse(&kappa_squared, p), p),
            y: mul(&point.y, &inverse(&kappa_cubed, p), p),
        }
    })
}

fn map_from_minus_three(point: Option<&Affine>, p: &BigUint) -> Option<Affine> {
    point.map(|point| {
        let kappa = kappa();
        let kappa_squared = mul(&kappa, &kappa, p);
        let kappa_cubed = mul(&kappa_squared, &kappa, p);
        Affine { x: mul(&point.x, &kappa_squared, p), y: mul(&point.y, &kappa_cubed, p) }
    })
}

fn expanded_coordinate_weights() -> [u32; 3] {
    [1 + 9 + 9, 1 + 81 + 27, 1 + 9 + 3]
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn frozen_isomorphism_maps_bls12_381_g1_to_b_minus_three() {
    let p = modulus();
    let three_inverse = inverse(&BigUint::from(3_u8), &p);
    let expected = negate(&scale(&three_inverse, 4, &p), &p);
    assert_eq!(kappa().modpow(&BigUint::from(6_u8), &p), expected);

    let original_b = BigUint::from(4_u8);
    let transformed_b = &p - 3_u8;
    let generator = generator();
    assert!(is_on_curve(&generator, &original_b, &p));
    for scalar in 1..=512 {
        let original = scalar_multiply(&generator, scalar, &p).expect("nonzero small multiple");
        let transformed = map_to_minus_three(Some(&original), &p).expect("finite map");
        assert!(is_on_curve(&transformed, &transformed_b, &p));
        assert_eq!(map_from_minus_three(Some(&transformed), &p), Some(original));
    }
    assert_eq!(map_to_minus_three(None, &p), None);
    assert_eq!(map_from_minus_three(None, &p), None);
}

#[test]
fn expanded_complete_formula_matches_rcb_and_affine_oracle() {
    let p = modulus();
    let transformed_b = &p - 3_u8;
    let transformed_generator =
        map_to_minus_three(Some(&generator()), &p).expect("finite generator");
    assert!(is_on_curve(&transformed_generator, &transformed_b, &p));
    let coordinate_weights = expanded_coordinate_weights();
    assert_eq!(coordinate_weights, [19, 109, 13]);
    assert!(coordinate_weights[1] < 125, "expanded coordinate weight must fit the AMNS capacity");

    let identity = to_homogeneous(None, &BigUint::from(1_u8), &p);
    let point = scalar_multiply(&transformed_generator, 17, &p).expect("nonzero point");
    let inverse_point = Affine { x: point.x.clone(), y: negate(&point.y, &p) };
    let explicit = [
        (None, None),
        (None, Some(point.clone())),
        (Some(point.clone()), None),
        (Some(point.clone()), Some(point.clone())),
        (Some(point.clone()), Some(inverse_point)),
    ];
    for (index, (left, right)) in explicit.into_iter().enumerate() {
        let left_scale = BigUint::from((index + 1) as u32);
        let right_scale = BigUint::from((index + 7) as u32);
        let left_h = to_homogeneous(left.as_ref(), &left_scale, &p);
        let right_h = to_homogeneous(right.as_ref(), &right_scale, &p);
        let standard = rcb_standard(&left_h, &right_h, &p);
        let expanded = rcb_expanded(&left_h, &right_h, &p);
        assert_eq!(expanded, standard, "explicit homogeneous case {index}");
        assert_eq!(to_affine(&expanded, &p), affine_add(left.as_ref(), right.as_ref(), &p));
    }
    assert_eq!(rcb_expanded(&identity, &identity, &p), identity);

    let mut state = 0x414d_4e53_3852_4342_u64;
    for case in 0..2_048 {
        let left_scalar = next_random(&mut state) % 1_000_000 + 1;
        let right_scalar = next_random(&mut state) % 1_000_000 + 1;
        let left = scalar_multiply(&transformed_generator, left_scalar, &p).expect("nonzero point");
        let right =
            scalar_multiply(&transformed_generator, right_scalar, &p).expect("nonzero point");
        let left_scale = BigUint::from(next_random(&mut state) | 1);
        let right_scale = BigUint::from(next_random(&mut state) | 1);
        let left_h = to_homogeneous(Some(&left), &left_scale, &p);
        let right_h = to_homogeneous(Some(&right), &right_scale, &p);
        let standard = rcb_standard(&left_h, &right_h, &p);
        let expanded = rcb_expanded(&left_h, &right_h, &p);
        assert_eq!(expanded, standard, "random homogeneous case {case}");
        assert_eq!(
            to_affine(&expanded, &p),
            affine_add(Some(&left), Some(&right), &p),
            "random affine case {case}"
        );
    }
}
