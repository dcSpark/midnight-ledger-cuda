// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#[path = "../build_support/blocked_radix128.rs"]
mod blocked_radix128;

const MODULUS: [u32; 12] = [
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

// -p^-1 mod 2^128, independently checked by the BigUint model.
const BLOCK_NEGATIVE_INVERSE: [u32; 4] = [0xfffc_fffd, 0x89f3_fffc, 0xd9d1_13e8, 0x286a_db92];

#[test]
fn generator_rejects_an_invalid_block_negative_inverse() {
    let mut invalid = BLOCK_NEGATIVE_INVERSE;
    invalid[3] ^= 1;
    let result = std::panic::catch_unwind(|| blocked_radix128::generate(&MODULUS, &invalid));
    assert!(result.is_err());
}

#[test]
fn generated_blocked128_schedule_is_flat_and_audited() {
    let schedule = blocked_radix128::generate(&MODULUS, &BLOCK_NEGATIVE_INVERSE);
    assert_eq!(schedule.base_products, 189);
    assert_eq!(schedule.typed_peak_live_words, schedule.tokenized_peak_live_words);
    assert_eq!(schedule.typed_allocation_peak_live_words, 55);
    assert_eq!(
        schedule.typed_allocation_peak_live_words,
        schedule.tokenized_allocation_peak_live_words
    );
    assert!(schedule.typed_allocation_peak_live_words <= 56);
    assert!(!schedule.source.contains('['));
    assert!(!schedule.source.contains("for ("));
    assert!(!schedule.source.contains("threadgroup"));
    assert!(!schedule.source.contains("simd_"));
    assert!(schedule.source.contains("u32_named_blocked128_karatsuba_l4_redc"));
    std::println!(
        "blocked128 typed_peak={} token_peak={} allocation_peak={} typed_label={} token_label={} typed_values={:?} token_values={:?}",
        schedule.typed_peak_live_words,
        schedule.tokenized_peak_live_words,
        schedule.typed_allocation_peak_live_words,
        schedule.typed_peak_label,
        schedule.tokenized_peak_label,
        schedule.typed_peak_values,
        schedule.tokenized_peak_values
    );
    std::println!("blocked128 typed_peak_descriptions={:?}", schedule.typed_peak_descriptions);
}
