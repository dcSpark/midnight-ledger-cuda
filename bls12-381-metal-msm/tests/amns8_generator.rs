// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#[path = "../build_support/amns8.rs"]
mod amns8;

const N: usize = 8;
const PHI: i128 = 1_i128 << 64;
const V1_FULL_SOURCE_FNV1A64: u64 = 0xbe11_ba95_230f_ba8e;
const V2_CALLABLE_FINGERPRINT: &str = "fnv1a64:4cddfae73658b39f";
const V2_FULL_SOURCE_FNV1A64: u64 = 0x24bb_cb2a_2d68_29aa;

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn schoolbook_signed(left: &[i64; N], right: &[i64; N]) -> [i128; N] {
    let mut result = [0_i128; N];
    for (left_index, left_value) in left.iter().copied().enumerate() {
        for (right_index, right_value) in right.iter().copied().enumerate() {
            let degree = left_index + right_index;
            let product = i128::from(left_value) * i128::from(right_value);
            if degree < N {
                result[degree] += product;
            } else {
                result[degree - N] -= 2 * product;
            }
        }
    }
    result
}

fn schoolbook_wrapping(left: &[u64; N], right: &[u64; N]) -> [u64; N] {
    let mut result = [0_u64; N];
    for (left_index, left_value) in left.iter().copied().enumerate() {
        for (right_index, right_value) in right.iter().copied().enumerate() {
            let degree = left_index + right_index;
            let product = left_value.wrapping_mul(right_value);
            if degree < N {
                result[degree] = result[degree].wrapping_add(product);
            } else {
                result[degree - N] = result[degree - N].wrapping_sub(product.wrapping_mul(2));
            }
        }
    }
    result
}

fn leaf_signed(left: &[i64; N], right: &[i64; N]) -> [i128; N] {
    let mut result = [0_i128; N];
    for leaf in amns8::karatsuba_leaves() {
        let left_transform: i128 = leaf
            .left
            .iter()
            .zip(left)
            .map(|(&mask, &value)| i128::from(mask) * i128::from(value))
            .sum();
        let right_transform: i128 = leaf
            .right
            .iter()
            .zip(right)
            .map(|(&mask, &value)| i128::from(mask) * i128::from(value))
            .sum();
        let product = left_transform * right_transform;
        for (output, weight) in leaf.output_weights.iter().copied().enumerate() {
            result[output] += i128::from(weight) * product;
        }
    }
    result
}

fn leaf_wrapping(left: &[u64; N], right: &[u64; N]) -> [u64; N] {
    let mut result = [0_u64; N];
    for leaf in amns8::karatsuba_leaves() {
        let transform = |mask: &[i8; N], values: &[u64; N]| {
            mask.iter().zip(values).fold(0_u64, |accumulator, (&mask, &value)| match mask {
                -1 => accumulator.wrapping_sub(value),
                0 => accumulator,
                1 => accumulator.wrapping_add(value),
                _ => panic!("invalid K8 leaf mask"),
            })
        };
        let product = transform(&leaf.left, left).wrapping_mul(transform(&leaf.right, right));
        for (output, weight) in leaf.output_weights.iter().copied().enumerate() {
            result[output] = match weight {
                -2 => result[output].wrapping_sub(product.wrapping_mul(2)),
                -1 => result[output].wrapping_sub(product),
                0 => result[output],
                1 => result[output].wrapping_add(product),
                2 => result[output].wrapping_add(product.wrapping_mul(2)),
                _ => panic!("invalid direct recombination weight"),
            };
        }
    }
    result
}

fn split_centered(value: i128) -> (i64, i64) {
    let low = value as u64 as i64;
    let carry = (value - i128::from(low)) / PHI;
    assert_eq!(value, i128::from(low) + carry * PHI);
    (low, i64::try_from(carry).expect("eight-term transform carry fits i64"))
}

fn explicit_mulhi_u32_product_model(left: u32, right: u32) -> u64 {
    let oracle = u64::from(left) * u64::from(right);
    let low = left.wrapping_mul(right);
    let high = (oracle >> 32) as u32;
    u64::from(low) | (u64::from(high) << 32)
}

fn signed_i64_three_product_model(left: i64, right: i64) -> (i128, usize) {
    let left_magnitude = left.unsigned_abs();
    let right_magnitude = right.unsigned_abs();
    let left_low = left_magnitude as u32;
    let left_high = (left_magnitude >> 32) as u32;
    let right_low = right_magnitude as u32;
    let right_high = (right_magnitude >> 32) as u32;
    let z0 = u128::from(explicit_mulhi_u32_product_model(left_low, right_low));
    let z2 = u128::from(explicit_mulhi_u32_product_model(left_high, right_high));
    let left_difference = i128::from(left_low) - i128::from(left_high);
    let right_difference = i128::from(right_low) - i128::from(right_high);
    let middle_magnitude = u128::from(explicit_mulhi_u32_product_model(
        left_difference.unsigned_abs() as u32,
        right_difference.unsigned_abs() as u32,
    ));
    let middle = if (left_difference < 0) ^ (right_difference < 0) {
        -(middle_magnitude as i128)
    } else {
        middle_magnitude as i128
    };
    let cross = i128::try_from(z0 + z2).expect("cross base fits i128") - middle;
    let magnitude = z0 + (u128::try_from(cross).expect("cross is nonnegative") << 32) + (z2 << 64);
    let signed = if (left < 0) ^ (right < 0) { -(magnitude as i128) } else { magnitude as i128 };
    (signed, 3)
}

fn wrapping_u64_three_product_model(left: u64, right: u64) -> (u64, usize, usize) {
    let left_low = left as u32;
    let left_high = (left >> 32) as u32;
    let right_low = right as u32;
    let right_high = (right >> 32) as u32;
    let low = explicit_mulhi_u32_product_model(left_low, right_low);
    let cross = left_low.wrapping_mul(right_high).wrapping_add(left_high.wrapping_mul(right_low));
    (low.wrapping_add(u64::from(cross) << 32), 1, 2)
}

fn leaf_signed_explicit_mulhi(left: &[i64; N], right: &[i64; N]) -> [i128; N] {
    let mut result = [0_i128; N];
    for leaf in amns8::karatsuba_leaves() {
        let transform = |mask: &[i8; N], values: &[i64; N]| {
            let value: i128 = mask
                .iter()
                .zip(values)
                .map(|(&mask, &value)| i128::from(mask) * i128::from(value))
                .sum();
            i64::try_from(value).expect("test K8 transform fits i64")
        };
        let left_transform = transform(&leaf.left, left);
        let right_transform = transform(&leaf.right, right);
        let (product, widened_products) =
            signed_i64_three_product_model(left_transform, right_transform);
        assert_eq!(widened_products, 3);
        for (output, weight) in leaf.output_weights.iter().copied().enumerate() {
            result[output] += i128::from(weight) * product;
        }
    }
    result
}

fn leaf_wrapping_explicit_mulhi(left: &[u64; N], right: &[u64; N]) -> [u64; N] {
    let mut result = [0_u64; N];
    for leaf in amns8::karatsuba_leaves() {
        let transform = |mask: &[i8; N], values: &[u64; N]| {
            mask.iter().zip(values).fold(0_u64, |accumulator, (&mask, &value)| match mask {
                -1 => accumulator.wrapping_sub(value),
                0 => accumulator,
                1 => accumulator.wrapping_add(value),
                _ => panic!("invalid K8 leaf mask"),
            })
        };
        let (product, widened_products, wrapping_products) = wrapping_u64_three_product_model(
            transform(&leaf.left, left),
            transform(&leaf.right, right),
        );
        assert_eq!((widened_products, wrapping_products), (1, 2));
        for (output, weight) in leaf.output_weights.iter().copied().enumerate() {
            result[output] = match weight {
                -2 => result[output].wrapping_sub(product.wrapping_mul(2)),
                -1 => result[output].wrapping_sub(product),
                0 => result[output],
                1 => result[output].wrapping_add(product),
                2 => result[output].wrapping_add(product.wrapping_mul(2)),
                _ => panic!("invalid direct recombination weight"),
            };
        }
    }
    result
}

fn amns_leaf_schedule(left: &[i64; N], right: &[i64; N]) -> [i64; N] {
    let product = leaf_signed_explicit_mulhi(left, right);
    let product_low = product.map(|value| value as u64);
    let quotient_bits = leaf_wrapping_explicit_mulhi(&product_low, &amns8::M_PRIME);
    let quotient = quotient_bits.map(|value| value as i64);
    let mut correction = [0_i128; N];
    for leaf in amns8::karatsuba_leaves() {
        let q_transform: i128 = leaf
            .left
            .iter()
            .zip(quotient)
            .map(|(&mask, value)| i128::from(mask) * i128::from(value))
            .sum();
        let m_transform: i128 = leaf
            .right
            .iter()
            .zip(amns8::M)
            .map(|(&mask, value)| i128::from(mask) * i128::from(value))
            .sum();
        let (low, carry) = split_centered(q_transform);
        assert!(carry.unsigned_abs() <= 4);
        let m_signed = i64::try_from(m_transform).expect("M transform fits i64");
        let (low_product, count) = signed_i64_three_product_model(low, m_signed);
        assert_eq!(count, 3);
        let leaf_product = low_product + i128::from(carry) * i128::from(m_signed) * PHI;
        assert_eq!(leaf_product, q_transform * m_transform);
        for (output, weight) in leaf.output_weights.iter().copied().enumerate() {
            correction[output] += i128::from(weight) * leaf_product;
        }
    }
    std::array::from_fn(|index| {
        let numerator = product[index] + correction[index];
        assert_eq!(numerator.rem_euclid(PHI), 0);
        i64::try_from(numerator / PHI).expect("AMNS result fits i64")
    })
}

fn amns_schoolbook_reference(left: &[i64; N], right: &[i64; N]) -> [i64; N] {
    let product = schoolbook_signed(left, right);
    let quotient_bits = schoolbook_wrapping(&product.map(|value| value as u64), &amns8::M_PRIME);
    let quotient = quotient_bits.map(|value| value as i64);
    let correction = schoolbook_signed(&quotient, &amns8::M);
    std::array::from_fn(|index| {
        let numerator = product[index] + correction[index];
        assert_eq!(numerator.rem_euclid(PHI), 0);
        i64::try_from(numerator / PHI).expect("reference result fits i64")
    })
}

#[test]
fn balanced_k8_masks_and_direct_recombination_match_schoolbook() {
    let leaves = amns8::karatsuba_leaves();
    assert_eq!(leaves.len(), 27);
    assert!(leaves.iter().all(|leaf| {
        leaf.left.iter().chain(&leaf.right).all(|mask| (-1..=1).contains(mask))
            && leaf.output_weights.iter().all(|weight| (-2..=2).contains(weight))
    }));

    let mut random = XorShift64(0x8d58_ac26_4f91_3be7);
    for _ in 0..512 {
        let left_signed = std::array::from_fn(|_| (random.next() % 2_000_001) as i64 - 1_000_000);
        let right_signed = std::array::from_fn(|_| (random.next() % 2_000_001) as i64 - 1_000_000);
        assert_eq!(
            leaf_signed(&left_signed, &right_signed),
            schoolbook_signed(&left_signed, &right_signed)
        );

        let left_wrapping = std::array::from_fn(|_| random.next());
        let right_wrapping = std::array::from_fn(|_| random.next());
        assert_eq!(
            leaf_wrapping(&left_wrapping, &right_wrapping),
            schoolbook_wrapping(&left_wrapping, &right_wrapping)
        );
    }
}

#[test]
fn three_product_leaf_models_cover_signed_edges_and_wrapping_values() {
    let signed_edges = [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];
    for left in signed_edges {
        for right in signed_edges {
            let (actual, products) = signed_i64_three_product_model(left, right);
            assert_eq!(actual, i128::from(left) * i128::from(right));
            assert_eq!(products, 3);
        }
    }

    let mut random = XorShift64(0xd6e8_feb8_6659_fd93);
    for _ in 0..4_096 {
        let left = random.next() as i64;
        let right = random.next() as i64;
        assert_eq!(
            signed_i64_three_product_model(left, right),
            (i128::from(left) * i128::from(right), 3)
        );
        let left = random.next();
        let right = random.next();
        assert_eq!(wrapping_u64_three_product_model(left, right), (left.wrapping_mul(right), 1, 2));
    }
}

#[test]
fn explicit_mulhi_leaf_schedule_matches_independent_amns_reference() {
    let mut random = XorShift64(0x243f_6a88_85a3_08d3);
    for _ in 0..512 {
        let left = std::array::from_fn(|_| (random.next() % 20_000_001) as i64 - 10_000_000);
        let right = std::array::from_fn(|_| (random.next() % 20_000_001) as i64 - 10_000_000);
        assert_eq!(amns_leaf_schedule(&left, &right), amns_schoolbook_reference(&left, &right));
    }
}

#[test]
fn generated_v1_schedule_is_byte_stable_flat_and_honestly_counted() {
    let first = amns8::generate();
    let second = amns8::generate();
    assert_eq!(first.source, second.source);
    assert_eq!(first.fingerprint, second.fingerprint);
    assert_eq!(first.fingerprint, "fnv1a64:f8a2079cfdcc25d5");
    assert_eq!(fnv1a64(first.source.as_bytes()), V1_FULL_SOURCE_FNV1A64);
    assert_eq!(first.logical_leaves_per_convolution, 27);
    assert_eq!(first.logical_base_products, 243);
    assert_eq!(first.logical_widened_products, 189);
    assert_eq!(first.logical_wrapping_products, 54);
    assert_eq!(first.issued_leaf_slots_per_convolution, 32);
    assert_eq!(first.issued_base_products_per_lane, 36);
    assert_eq!(first.issued_widened_products_per_field, 224);
    assert_eq!(first.issued_wrapping_products_per_field, 64);
    assert_eq!(first.emitted_base_product_instructions, 36);
    assert_eq!(first.emitted_widened_product_instructions, 28);
    assert_eq!(first.emitted_wrapping_product_instructions, 8);
    assert_eq!(first.source.matches("@base-u32-product").count(), 28);
    assert_eq!(first.source.matches("@wrapping-u32-product").count(), 8);
    assert!(!first.source.contains("@widened-u32-product-low"));
    assert!(!first.source.contains("@widened-u32-product-high"));
    assert!(!first.source.contains("metal::mulhi("));

    assert_eq!(first.typed_peak_live_words, first.tokenized_peak_live_words);
    assert_eq!(first.typed_allocation_peak_live_words, first.tokenized_allocation_peak_live_words);
    assert!(first.typed_allocation_peak_live_words <= 56);
    assert_eq!(first.leaves, amns8::karatsuba_leaves());

    assert!(first.source.contains("long amns8_multiply("));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_field_abi"));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_field_multiply"));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_rotating_chain"));
    assert!(!first.source.contains("threadgroup"));
    assert!(!first.source.contains("private "));
    assert!(!first.source.contains("switch ("));
    assert!(!first.source.contains("new "));
    assert!(!first.source.contains("malloc"));
    assert_eq!(first.source.matches("    if (").count(), 2, "only final stores are guarded");
    for line in first.source.lines().filter(|line| line.trim_start().starts_with("const ")) {
        let declaration = line.split('=').next().expect("const declaration prefix");
        assert!(!declaration.contains('['), "no private array declaration: {line}");
    }

    std::println!(
        "AMNS8 fingerprint={} peak={} allocation_peak={} typed_label={} token_label={} typed_values={:?} token_values={:?}",
        first.fingerprint,
        first.typed_peak_live_words,
        first.typed_allocation_peak_live_words,
        first.typed_peak_label,
        first.tokenized_peak_label,
        first.typed_peak_values,
        first.tokenized_peak_values
    );
}

#[test]
fn generated_v2_explicit_mulhi_schedule_is_distinct_flat_and_honestly_counted() {
    let first = amns8::generate_variant(amns8::ScheduleVariant::ExplicitMulhiV2);
    let second = amns8::generate_variant(amns8::ScheduleVariant::ExplicitMulhiV2);
    let v1 = amns8::generate();
    assert_eq!(first.source, second.source);
    assert_eq!(first.fingerprint, second.fingerprint);
    assert_eq!(first.fingerprint, V2_CALLABLE_FINGERPRINT);
    assert_eq!(fnv1a64(first.source.as_bytes()), V2_FULL_SOURCE_FNV1A64);
    assert_ne!(first.fingerprint, v1.fingerprint);
    assert_ne!(first.source, v1.source);

    assert_eq!(first.logical_leaves_per_convolution, 27);
    assert_eq!(first.logical_base_products, 243);
    assert_eq!(first.logical_widened_products, 189);
    assert_eq!(first.logical_wrapping_products, 54);
    assert_eq!(first.issued_leaf_slots_per_convolution, 32);
    assert_eq!(first.issued_base_products_per_lane, 36);
    assert_eq!(first.issued_widened_products_per_field, 224);
    assert_eq!(first.issued_wrapping_products_per_field, 64);
    assert_eq!(first.emitted_base_product_instructions, 36);
    assert_eq!(first.emitted_widened_product_instructions, 28);
    assert_eq!(first.emitted_wrapping_product_instructions, 8);
    assert_eq!(first.source.matches("@base-u32-product").count(), 28);
    assert_eq!(first.source.matches("@wrapping-u32-product").count(), 8);
    assert_eq!(first.source.matches("@widened-u32-product-low").count(), 28);
    assert_eq!(first.source.matches("@widened-u32-product-high").count(), 28);
    assert_eq!(first.source.matches("metal::mulhi(").count(), 28);

    for line in first.source.lines().filter(|line| line.contains("@widened-u32-product-low")) {
        assert!(line.trim_start().starts_with("const uint "), "low half type: {line}");
        assert!(line.contains(" * "), "low half multiply: {line}");
        assert!(!line.contains("static_cast<ulong>"), "low half widened unexpectedly: {line}");
    }
    for line in first.source.lines().filter(|line| line.contains("@widened-u32-product-high")) {
        assert!(line.trim_start().starts_with("const uint "), "high half type: {line}");
        assert!(line.contains("metal::mulhi("), "high half primitive: {line}");
        assert!(!line.contains(" * "), "high half multiply lowered in source: {line}");
    }
    assert!(
        !first.source.lines().any(|line| line.contains("static_cast<ulong>(v")
            && line.contains(" * static_cast<ulong>(v"))
    );

    assert_eq!(first.typed_peak_live_words, first.tokenized_peak_live_words);
    assert_eq!(first.typed_allocation_peak_live_words, first.tokenized_allocation_peak_live_words);
    assert!(first.typed_allocation_peak_live_words <= 56);
    assert_eq!(first.leaves, amns8::karatsuba_leaves());

    assert!(first.source.contains("long amns8_mulhi_v2_multiply("));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_mulhi_v2_field_abi"));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_mulhi_v2_field_multiply"));
    assert!(first.source.contains("bls12_381_metal_msm_amns8_mulhi_v2_rotating_chain"));
    assert!(!first.source.contains("threadgroup"));
    assert!(!first.source.contains("private "));
    assert!(!first.source.contains("switch ("));
    assert!(!first.source.contains("new "));
    assert!(!first.source.contains("malloc"));
    assert_eq!(first.source.matches("    if (").count(), 2, "only final stores are guarded");
    for line in first.source.lines().filter(|line| line.trim_start().starts_with("const ")) {
        let declaration = line.split('=').next().expect("const declaration prefix");
        assert!(!declaration.contains('['), "no private array declaration: {line}");
    }

    std::println!(
        "AMNS8 mulhi-v2 fingerprint={} source_fnv={:016x} peak={} allocation_peak={} typed_label={} token_label={}",
        first.fingerprint,
        fnv1a64(first.source.as_bytes()),
        first.typed_peak_live_words,
        first.typed_allocation_peak_live_words,
        first.typed_peak_label,
        first.tokenized_peak_label,
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires the Xcode Metal compiler"]
fn generated_v2_source_compiles_to_optimized_air_without_variable_i64_multiply() {
    let generated = amns8::generate_variant(amns8::ScheduleVariant::ExplicitMulhiV2);
    let directory = std::env::temp_dir()
        .join(format!("bls12-381-metal-msm-amns8-mulhi-v2-{}", std::process::id()));
    let source = directory.join("amns8_mulhi_v2.metal");
    let air = directory.join("amns8_mulhi_v2.air");
    std::fs::create_dir_all(&directory).expect("create isolated Metal test directory");
    std::fs::write(&source, generated.source).expect("write generated v2 Metal source");
    let output = std::process::Command::new("xcrun")
        .arg("--sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-std=metal3.0")
        .arg("-mmacosx-version-min=13.0")
        .arg("-O3")
        .arg("-fno-fast-math")
        .arg("-fmetal-math-mode=safe")
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&air)
        .output()
        .expect("launch Xcode Metal compiler");
    assert!(
        output.status.success(),
        "v2 Metal compilation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let disassembly = std::process::Command::new("xcrun")
        .arg("--sdk")
        .arg("macosx")
        .arg("air-objdump")
        .arg("-d")
        .arg(&air)
        .output()
        .expect("launch AIR disassembler");
    assert!(
        disassembly.status.success(),
        "v2 AIR disassembly failed: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    let disassembly = String::from_utf8(disassembly.stdout).expect("AIR disassembly is UTF-8");
    let function_marker = "define void @bls12_381_metal_msm_amns8_mulhi_v2_field_multiply(";
    let function_start = disassembly.find(function_marker).expect("v2 one-shot AIR function");
    let function_tail = &disassembly[function_start..];
    let function_end = function_tail.find("\n}").expect("v2 one-shot AIR function end") + 2;
    let function = &function_tail[..function_end];
    let variable_multiply = |line: &str, ty: &str| {
        let Some((_, operands)) = line.split_once(ty) else {
            return false;
        };
        let Some((left, right)) = operands.split_once(", ") else {
            return false;
        };
        line.contains(" = mul ")
            && left.trim_start().starts_with('%')
            && right.trim_start().starts_with('%')
    };
    let variable_i64_multiply: Vec<_> =
        function.lines().filter(|line| variable_multiply(line, " i64 ")).collect();
    assert!(
        variable_i64_multiply.is_empty(),
        "explicit-mulhi AIR retained variable i64 multiplication: {variable_i64_multiply:?}"
    );
    assert_eq!(function.matches("@air.mul_hi.u.i32").count(), 28);
    assert_eq!(function.lines().filter(|line| variable_multiply(line, " i32 ")).count(), 36);
    assert_eq!(
        function.lines().filter(|line| line.contains(" = mul ") && line.contains(" i64 ")).count(),
        4,
        "only the four constant-times-three carry terms may remain as i64 multiplies"
    );
    assert_eq!(function.matches("@air.simd_shuffle.u.i32").count(), 494);
    assert!(!function.contains("alloca"));
    assert!(!function.contains("addrspace(3)"));
    std::fs::remove_dir_all(&directory).expect("remove isolated Metal test directory");
}
