// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "build_support/amns8.rs"]
mod amns8;
#[path = "build_support/blocked_radix128.rs"]
mod blocked_radix128;

include!("src/field_constants.rs");

const MACOS_DEPLOYMENT_TARGET: &str = "13.0";
const DIAGNOSTIC_ROOT_SHADER: &str = "root.metal";
const PRODUCTION_ROOT_SHADER: &str = "msm_root.metal";
const MSM_SHADER_ABI_VERSION: u64 = 1;
// FNV-1a over the ABI version, Fp modulus, Montgomery inverse, and R. Any
// representation change is an ABI change and must update both constants.
const MSM_EXPECTED_CONSTANTS_FINGERPRINT: u64 = 0xd599_f628_1415_d2c5;
const MSM_EXPECTED_LAYOUT_FINGERPRINT: u64 = 0xa393_9e5a_6148_f9b7;
const MSM_FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];
const MSM_FR_MODULUS: [u64; 4] =
    [0xffff_ffff_0000_0001, 0x53bd_a402_fffe_5bfe, 0x3339_d808_09a1_d805, 0x73ed_a753_299d_7d48];
const U32_FIELD_ALGORITHM_VERSION: u32 = 2;
const U32_FIPS_SLOT_COUNT: u32 = 144;
const U32_NAMED_FIELD_ALGORITHM_VERSION: u32 = 1;
const U32_NAMED_FIELD_MATRIX_TERM_COUNT: u32 = 288;
const U32_NAMED_FIELD_VARIANT_COUNT: u32 = 2;
const U32_KARATSUBA_L4_ALGORITHM_VERSION: u32 = 2;
const U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12: u32 = 63;
const U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC: u32 = 189;
const U32_KARATSUBA_L4_LAZY_MULTIPLE: u32 = 4;
const U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT: u32 = 56;
const U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS: u32 = 70;
const _: () = assert!(
    U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS > U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
    "the current expanded schedule is intentionally fail-closed before scouting"
);
// Frozen independently of generated output. Intentional schedule changes must
// bump the algorithm version and explicitly update this reviewed snapshot.
const U32_KARATSUBA_L4_EXPECTED_SCHEDULE_FINGERPRINT: u64 = 0x757d_b367_3700_e544;
const U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION: u32 = 3;
const U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4: u32 = 9;
const U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC: u32 = 189;
const U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS: u32 = 4;
const U32_BLOCKED128_KARATSUBA_L4_ROUNDS: u32 = 3;
const U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS: u32 = 16;
const U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE: u32 = 4;
const U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT: u32 = 56;
// Frozen independently of generated output. Intentional schedule changes must
// bump the algorithm version and explicitly update this reviewed snapshot.
const U32_BLOCKED128_KARATSUBA_L4_EXPECTED_SCHEDULE_FINGERPRINT: u64 = 0x0ead_48b6_6cf3_f532;
const AMNS8_ALGORITHM_VERSION: u32 = 1;
const AMNS8_SOURCE_LIVE_WORD_LIMIT: u32 = 56;
const AMNS8_EXPECTED_SCHEDULE_FINGERPRINT: u64 = 0xf8a2_079c_fdcc_25d5;
const AMNS8_EXPECTED_FULL_SOURCE_FINGERPRINT: u64 = 0xbe11_ba95_230f_ba8e;
const AMNS8_MULHI_V2_EXPECTED_SCHEDULE_FINGERPRINT: u64 = 0x4cdd_fae7_3658_b39f;
const AMNS8_MULHI_V2_EXPECTED_FULL_SOURCE_FINGERPRINT: u64 = 0x24bb_cb2a_2d68_29aa;
const _: () = assert!(
    U32_BLOCKED128_KARATSUBA_L4_ROUNDS * 7 * U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4
        == U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
    "blocked-radix-128 REDC base-product accounting"
);

fn main() {
    println!("cargo:rustc-check-cfg=cfg(bls12_381_metal_diagnostics)");
    println!("cargo:rustc-check-cfg=cfg(bls12_381_metal_shaders)");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support/amns8.rs");
    println!("cargo:rerun-if-changed=build_support/blocked_radix128.rs");
    println!("cargo:rerun-if-changed=src/field_constants.rs");
    println!("cargo:rerun-if-changed=shaders/{DIAGNOSTIC_ROOT_SHADER}");
    println!("cargo:rerun-if-changed=shaders/{PRODUCTION_ROOT_SHADER}");
    println!("cargo:rerun-if-changed=shaders/product_issue.metal");
    println!("cargo:rerun-if-changed=shaders/u32_field.metal");
    println!("cargo:rerun-if-changed=shaders/msm_field.metal");
    println!("cargo:rerun-if-changed=shaders/msm_g1.metal");
    println!("cargo:rerun-if-changed=shaders/msm_pipeline.metal");
    println!("cargo:rerun-if-changed=shaders/msm_parallel_bucket_v1.metal");
    println!("cargo:rerun-if-changed=shaders/msm_parallel_jacobian_v2.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_dense_v3.metal");
    println!("cargo:rerun-if-changed=shaders/msm_xyzz_v4.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_xyzz_v4.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_xyzz_v5.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_xyzz_v6.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_xyzz_v7.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_xyzz_v8.metal");
    println!("cargo:rerun-if-changed=shaders/msm_batch_affine.metal");
    println!("cargo:rerun-if-changed=shaders/msm_gpu_route_batch_affine_v9.metal");
    println!("cargo:rerun-if-changed=shaders/msm_fixed_base_v10.metal");

    if env::var_os("CARGO_FEATURE_METAL").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let diagnostics = env::var_os("CARGO_FEATURE_METAL_DIAGNOSTICS").is_some();
    write_u32_constants(&out);
    write_msm_constants(&out);
    let root_shader = if diagnostics { DIAGNOSTIC_ROOT_SHADER } else { PRODUCTION_ROOT_SHADER };
    let source = manifest.join("shaders").join(root_shader);
    let air = out.join("bls12-381-metal-msm.air");
    let module_cache = out.join("metal-module-cache");
    fs::create_dir_all(&module_cache).expect("create private Metal module cache");
    let developer_dir = developer_dir();
    let args = vec![
        OsString::from("-std=metal3.0"),
        OsString::from(format!("-mmacosx-version-min={MACOS_DEPLOYMENT_TARGET}")),
        OsString::from(format!("-fmodules-cache-path={}", module_cache.display())),
        OsString::from("-fno-fast-math"),
        OsString::from("-fmetal-math-mode=safe"),
        OsString::from("-fmetal-math-fp32-functions=precise"),
        OsString::from("-ffp-contract=on"),
        OsString::from("-I"),
        manifest.join("shaders").into_os_string(),
        OsString::from("-I"),
        out.as_os_str().to_owned(),
        OsString::from("-c"),
        source.into_os_string(),
        OsString::from("-o"),
        air.as_os_str().to_owned(),
    ];
    let output = run_xcrun(developer_dir.as_deref(), "metal", &args)
        .unwrap_or_else(|error| panic!("run Metal compiler: {error}"));
    require_success("compile", root_shader, &output);
    require_air_deployment_target(&air);

    let library = out.join("bls12-381-metal-msm.metallib");
    let link_args = vec![air.into_os_string(), OsString::from("-o"), library.into_os_string()];
    let output = run_xcrun(developer_dir.as_deref(), "metallib", &link_args)
        .unwrap_or_else(|error| panic!("run Metal library linker: {error}"));
    require_success("link", "bls12-381-metal-msm.metallib", &output);

    println!("cargo:rustc-cfg=bls12_381_metal_shaders");
    if diagnostics {
        println!("cargo:rustc-cfg=bls12_381_metal_diagnostics");
    }
}

fn write_u32_constants(out: &Path) {
    let modulus = parse_le_u32_limbs::<12>(FP_MODULUS_HEX);
    let inv = montgomery_inverse_word(modulus[0]);
    let negative_inverse = negative_inverse_mod_radix(&modulus, inv);
    let modulus_x2 = multiply_limbs_by_small(&modulus, 2);
    let modulus_x4 = multiply_limbs_by_small(&modulus, U32_KARATSUBA_L4_LAZY_MULTIPLE);
    let karatsuba_l4_schedule =
        generate_u32_named_karatsuba_l4_schedule(&modulus, &negative_inverse);
    let karatsuba_l4_schedule_fingerprint =
        stable_source_fingerprint(karatsuba_l4_schedule.source.as_bytes());
    assert_eq!(
        karatsuba_l4_schedule_fingerprint, U32_KARATSUBA_L4_EXPECTED_SCHEDULE_FINGERPRINT,
        "generated Karatsuba-L4 schedule changed; review it, bump the algorithm version, and update the frozen fingerprint"
    );
    let block_negative_inverse =
        [negative_inverse[0], negative_inverse[1], negative_inverse[2], negative_inverse[3]];
    let blocked128_schedule = blocked_radix128::generate(&modulus, &block_negative_inverse);
    let amns8_schedule = amns8::generate();
    let amns8_mulhi_v2_schedule = amns8::generate_variant(amns8::ScheduleVariant::ExplicitMulhiV2);
    let amns8_schedule_fingerprint = u64::from_str_radix(
        amns8_schedule.fingerprint.strip_prefix("fnv1a64:").expect("AMNS8 fingerprint prefix"),
        16,
    )
    .expect("AMNS8 hexadecimal fingerprint");
    assert_eq!(
        amns8_schedule_fingerprint, AMNS8_EXPECTED_SCHEDULE_FINGERPRINT,
        "generated AMNS8 schedule changed; review it, bump the algorithm version, and update the frozen fingerprint"
    );
    assert_eq!(
        stable_source_fingerprint(amns8_schedule.source.as_bytes()),
        AMNS8_EXPECTED_FULL_SOURCE_FINGERPRINT,
        "generated AMNS8 v1 wrappers changed; preserve the frozen result or version the source"
    );
    let amns8_mulhi_v2_schedule_fingerprint = u64::from_str_radix(
        amns8_mulhi_v2_schedule
            .fingerprint
            .strip_prefix("fnv1a64:")
            .expect("AMNS8 mulhi-v2 fingerprint prefix"),
        16,
    )
    .expect("AMNS8 mulhi-v2 hexadecimal fingerprint");
    assert_eq!(
        amns8_mulhi_v2_schedule_fingerprint, AMNS8_MULHI_V2_EXPECTED_SCHEDULE_FINGERPRINT,
        "generated AMNS8 mulhi-v2 schedule changed; review and version the source"
    );
    assert_eq!(
        stable_source_fingerprint(amns8_mulhi_v2_schedule.source.as_bytes()),
        AMNS8_MULHI_V2_EXPECTED_FULL_SOURCE_FINGERPRINT,
        "generated AMNS8 mulhi-v2 wrappers changed; review and version the source"
    );
    assert_eq!(amns8_schedule.logical_leaves_per_convolution, 27);
    assert_eq!(amns8_schedule.leaves.len(), 27);
    assert_eq!(amns8_schedule.logical_base_products, 243);
    assert_eq!(amns8_schedule.logical_widened_products, 189);
    assert_eq!(amns8_schedule.logical_wrapping_products, 54);
    assert_eq!(amns8_schedule.issued_leaf_slots_per_convolution, 32);
    assert_eq!(amns8_schedule.issued_widened_products_per_field, 224);
    assert_eq!(amns8_schedule.issued_wrapping_products_per_field, 64);
    assert_eq!(amns8_schedule.issued_base_products_per_lane, 36);
    assert_eq!(amns8_schedule.emitted_base_product_instructions, 36);
    assert_eq!(amns8_schedule.emitted_widened_product_instructions, 28);
    assert_eq!(amns8_schedule.emitted_wrapping_product_instructions, 8);
    assert_eq!(
        amns8_schedule.typed_peak_live_words, amns8_schedule.tokenized_peak_live_words,
        "AMNS8 typed and tokenized last-use liveness disagree"
    );
    assert_eq!(
        amns8_schedule.typed_allocation_peak_live_words,
        amns8_schedule.tokenized_allocation_peak_live_words,
        "AMNS8 typed and tokenized allocation-time liveness disagree"
    );
    assert!(
        amns8_schedule.typed_allocation_peak_live_words <= AMNS8_SOURCE_LIVE_WORD_LIMIT,
        "AMNS8 schedule exceeds its per-lane source-liveness gate: {} > {}",
        amns8_schedule.typed_allocation_peak_live_words,
        AMNS8_SOURCE_LIVE_WORD_LIMIT
    );
    assert!(!amns8_schedule.typed_peak_label.is_empty());
    assert!(!amns8_schedule.tokenized_peak_label.is_empty());
    assert!(!amns8_schedule.typed_peak_values.is_empty());
    assert!(!amns8_schedule.tokenized_peak_values.is_empty());
    assert_eq!(amns8_mulhi_v2_schedule.leaves, amns8_schedule.leaves);
    assert_eq!(amns8_mulhi_v2_schedule.logical_base_products, amns8_schedule.logical_base_products);
    assert_eq!(
        amns8_mulhi_v2_schedule.issued_widened_products_per_field,
        amns8_schedule.issued_widened_products_per_field
    );
    assert_eq!(
        amns8_mulhi_v2_schedule.issued_wrapping_products_per_field,
        amns8_schedule.issued_wrapping_products_per_field
    );
    assert_eq!(amns8_mulhi_v2_schedule.typed_peak_live_words, 32);
    assert_eq!(amns8_mulhi_v2_schedule.tokenized_peak_live_words, 32);
    assert_eq!(amns8_mulhi_v2_schedule.typed_allocation_peak_live_words, 33);
    assert_eq!(amns8_mulhi_v2_schedule.tokenized_allocation_peak_live_words, 33);
    assert_eq!(amns8_mulhi_v2_schedule.source.matches("metal::mulhi(").count(), 28);
    assert_eq!(amns8_mulhi_v2_schedule.source.matches("@widened-u32-product-low").count(), 28);
    assert_eq!(amns8_mulhi_v2_schedule.source.matches("@widened-u32-product-high").count(), 28);
    assert_eq!(
        blocked128_schedule.base_products, U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        "blocked-radix-128 generated base-product count"
    );
    assert_eq!(
        blocked128_schedule.typed_peak_live_words, blocked128_schedule.tokenized_peak_live_words,
        "blocked-radix-128 typed and emitted-source liveness audits disagree"
    );
    assert_eq!(
        blocked128_schedule.typed_allocation_peak_live_words,
        blocked128_schedule.tokenized_allocation_peak_live_words,
        "blocked-radix-128 typed and emitted-source allocation-time liveness audits disagree"
    );
    let blocked128_source_peak_live_words = blocked128_schedule.typed_allocation_peak_live_words;
    assert!(!blocked128_schedule.typed_peak_label.is_empty());
    assert!(!blocked128_schedule.typed_peak_values.is_empty());
    assert_eq!(
        blocked128_schedule.typed_peak_values.len(),
        blocked128_schedule.typed_peak_descriptions.len()
    );
    assert!(!blocked128_schedule.tokenized_peak_label.is_empty());
    assert!(!blocked128_schedule.tokenized_peak_values.is_empty());
    assert!(
        blocked128_source_peak_live_words <= U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT,
        "blocked-radix-128 schedule exceeds its source liveness gate: {} > {}",
        blocked128_source_peak_live_words,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT
    );
    let blocked128_schedule_fingerprint =
        stable_source_fingerprint(blocked128_schedule.source.as_bytes());
    assert!(
        blocked128_schedule_fingerprint
            == U32_BLOCKED128_KARATSUBA_L4_EXPECTED_SCHEDULE_FINGERPRINT,
        "generated blocked-radix-128 Karatsuba-L4 schedule changed; review it, bump the algorithm version, and update the frozen fingerprint (actual {blocked128_schedule_fingerprint:#018x})"
    );
    let fingerprint = u32_field_abi_fingerprint(&modulus, inv);
    let named_fingerprint = u32_named_field_abi_fingerprint(&modulus, inv);
    let karatsuba_l4_fingerprint = u32_karatsuba_l4_abi_fingerprint(
        &modulus,
        &negative_inverse,
        karatsuba_l4_schedule_fingerprint,
        karatsuba_l4_schedule.peak_live_limbs,
    );
    let blocked128_fingerprint = u32_blocked128_karatsuba_l4_abi_fingerprint(
        &modulus,
        &block_negative_inverse,
        blocked128_schedule_fingerprint,
        blocked128_schedule.typed_peak_live_words,
        blocked128_schedule.tokenized_peak_live_words,
        blocked128_source_peak_live_words,
    );

    let mut metal = String::new();
    writeln!(metal, "// @generated by build.rs; do not edit.").expect("write Metal constants");
    writeln!(metal, "#ifndef BLS12_381_U32_CONSTANTS_GENERATED").expect("write Metal guard");
    writeln!(metal, "#define BLS12_381_U32_CONSTANTS_GENERATED").expect("write Metal guard");
    writeln!(metal, "constant uint BLS12_381_U32_FP_MODULUS[12] = {{")
        .expect("write Metal constants");
    for limb in modulus {
        writeln!(metal, "    {limb:#010x}U,").expect("write Metal modulus limb");
    }
    writeln!(metal, "}};").expect("write Metal constants");
    writeln!(metal, "constant uint BLS12_381_U32_FP_MODULUS_X2[12] = {{")
        .expect("write doubled Metal modulus");
    for limb in modulus_x2 {
        writeln!(metal, "    {limb:#010x}U,").expect("write doubled modulus limb");
    }
    writeln!(metal, "}};").expect("write doubled Metal modulus");
    writeln!(metal, "constant uint BLS12_381_U32_FP_INV = {inv:#010x}U;")
        .expect("write Metal inverse");
    writeln!(metal, "constant ulong BLS12_381_U32_FIELD_ABI_FINGERPRINT = {fingerprint:#018x}UL;")
        .expect("write Metal fingerprint");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_FIELD_ALGORITHM_VERSION = {U32_FIELD_ALGORITHM_VERSION}U;"
    )
    .expect("write Metal algorithm version");
    writeln!(
        metal,
        "constant ulong BLS12_381_U32_NAMED_FIELD_ABI_FINGERPRINT = {named_fingerprint:#018x}UL;"
    )
    .expect("write named-field fingerprint");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_NAMED_FIELD_ALGORITHM_VERSION = {U32_NAMED_FIELD_ALGORITHM_VERSION}U;"
    )
    .expect("write named-field algorithm version");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_NAMED_FIELD_VARIANT_COUNT = {U32_NAMED_FIELD_VARIANT_COUNT}U;"
    )
    .expect("write named-field variant count");
    writeln!(
        metal,
        "constant ulong BLS12_381_U32_KARATSUBA_L4_ABI_FINGERPRINT = {karatsuba_l4_fingerprint:#018x}UL;"
    )
    .expect("write Karatsuba-L4 fingerprint");
    writeln!(
        metal,
        "constant ulong BLS12_381_U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT = {karatsuba_l4_schedule_fingerprint:#018x}UL;"
    )
    .expect("write Karatsuba-L4 schedule fingerprint");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_ALGORITHM_VERSION = {U32_KARATSUBA_L4_ALGORITHM_VERSION}U;"
    )
    .expect("write Karatsuba-L4 algorithm version");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12 = {U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12}U;"
    )
    .expect("write Karatsuba-L4 K12 count");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC = {U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC}U;"
    )
    .expect("write Karatsuba-L4 REDC count");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_LAZY_MULTIPLE = {U32_KARATSUBA_L4_LAZY_MULTIPLE}U;"
    )
    .expect("write Karatsuba-L4 lazy multiple");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT = {U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT}U;"
    )
    .expect("write Karatsuba-L4 liveness limit");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS = {U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS}U;"
    )
    .expect("write Karatsuba-L4 flattened liveness result");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_KARATSUBA_L4_DECLARED_PEAK_LIVE_LIMBS = {}U;",
        karatsuba_l4_schedule.peak_live_limbs
    )
    .expect("write Karatsuba-L4 declared liveness");
    writeln!(
        metal,
        "constant ulong BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT = {blocked128_fingerprint:#018x}UL;"
    )
    .expect("write blocked-radix-128 ABI fingerprint");
    writeln!(
        metal,
        "constant ulong BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT = {blocked128_schedule_fingerprint:#018x}UL;"
    )
    .expect("write blocked-radix-128 schedule fingerprint");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION = {U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION}U;"
    )
    .expect("write blocked-radix-128 algorithm version");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4 = {U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4}U;"
    )
    .expect("write blocked-radix-128 K4 product count");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC = {U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC}U;"
    )
    .expect("write blocked-radix-128 REDC product count");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS = {U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS}U;"
    )
    .expect("write blocked-radix-128 block width");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ROUNDS = {U32_BLOCKED128_KARATSUBA_L4_ROUNDS}U;"
    )
    .expect("write blocked-radix-128 round count");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS = {U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS}U;"
    )
    .expect("write blocked-radix-128 accumulator width");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE = {U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE}U;"
    )
    .expect("write blocked-radix-128 lazy multiple");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS = {}U;",
        blocked128_schedule.typed_peak_live_words
    )
    .expect("write blocked-radix-128 typed liveness");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS = {}U;",
        blocked128_schedule.tokenized_peak_live_words
    )
    .expect("write blocked-radix-128 tokenized liveness");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS = {blocked128_source_peak_live_words}U;"
    )
    .expect("write blocked-radix-128 conservative source liveness");
    writeln!(
        metal,
        "constant uint BLS12_381_U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT = {U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT}U;"
    )
    .expect("write blocked-radix-128 liveness limit");
    writeln!(metal, "#endif").expect("write Metal guard");
    fs::write(out.join("bls12_381_u32_constants.metal"), metal)
        .expect("write generated Metal constants");

    let rust = format!(
        "// @generated by build.rs; do not edit.\n\
         const FP_MODULUS: [u32; 12] = {modulus:?};\n\
         const FP_INV: u32 = {inv:#010x};\n\
         const U32_FIELD_ABI_FINGERPRINT: u64 = {fingerprint:#018x};\n\
         const U32_FIELD_ALGORITHM_VERSION: u32 = {U32_FIELD_ALGORITHM_VERSION};\n\
         const U32_FIPS_SLOT_COUNT: u32 = {U32_FIPS_SLOT_COUNT};\n\
         const U32_NAMED_FIELD_ABI_FINGERPRINT: u64 = {named_fingerprint:#018x};\n\
         const U32_NAMED_FIELD_ALGORITHM_VERSION: u32 = {U32_NAMED_FIELD_ALGORITHM_VERSION};\n\
         const U32_NAMED_FIELD_MATRIX_TERM_COUNT: u32 = {U32_NAMED_FIELD_MATRIX_TERM_COUNT};\n\
         const U32_NAMED_FIELD_VARIANT_COUNT: u32 = {U32_NAMED_FIELD_VARIANT_COUNT};\n\
         const U32_KARATSUBA_L4_NEGATIVE_INVERSE: [u32; 12] = {negative_inverse:?};\n\
         const U32_KARATSUBA_L4_MODULUS_X4: [u32; 12] = {modulus_x4:?};\n\
         const U32_KARATSUBA_L4_ABI_FINGERPRINT: u64 = {karatsuba_l4_fingerprint:#018x};\n\
         const U32_KARATSUBA_L4_SCHEDULE_FINGERPRINT: u64 = {karatsuba_l4_schedule_fingerprint:#018x};\n\
         const U32_KARATSUBA_L4_ALGORITHM_VERSION: u32 = {U32_KARATSUBA_L4_ALGORITHM_VERSION};\n\
         const U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12: u32 = {U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12};\n\
         const U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC: u32 = {U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC};\n\
         const U32_KARATSUBA_L4_LAZY_MULTIPLE: u32 = {U32_KARATSUBA_L4_LAZY_MULTIPLE};\n\
         const U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT: u32 = {U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT};\n\
         const U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS: u32 = {U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS};\n\
         const U32_KARATSUBA_L4_DECLARED_PEAK_LIVE_LIMBS: u32 = {};\n\
         const U32_BLOCKED128_KARATSUBA_L4_NEGATIVE_INVERSE: [u32; 4] = {block_negative_inverse:?};\n\
         const U32_BLOCKED128_KARATSUBA_L4_ABI_FINGERPRINT: u64 = {blocked128_fingerprint:#018x};\n\
         const U32_BLOCKED128_KARATSUBA_L4_SCHEDULE_FINGERPRINT: u64 = {blocked128_schedule_fingerprint:#018x};\n\
         const U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION: u32 = {U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION};\n\
         const U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4: u32 = {U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4};\n\
         const U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC: u32 = {U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC};\n\
         const U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS: u32 = {U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS};\n\
         const U32_BLOCKED128_KARATSUBA_L4_ROUNDS: u32 = {U32_BLOCKED128_KARATSUBA_L4_ROUNDS};\n\
         const U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS: u32 = {U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS};\n\
         const U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE: u32 = {U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE};\n\
         const U32_BLOCKED128_KARATSUBA_L4_TYPED_PEAK_LIVE_WORDS: u32 = {};\n\
         const U32_BLOCKED128_KARATSUBA_L4_TOKENIZED_PEAK_LIVE_WORDS: u32 = {};\n\
         const U32_BLOCKED128_KARATSUBA_L4_SOURCE_PEAK_LIVE_WORDS: u32 = {blocked128_source_peak_live_words};\n\
         const U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT: u32 = {U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT};\n\
         const AMNS8_ALGORITHM_VERSION: u32 = {AMNS8_ALGORITHM_VERSION};\n\
         const AMNS8_SCHEDULE_FINGERPRINT: u64 = {amns8_schedule_fingerprint:#018x};\n\
         const AMNS8_LOGICAL_LEAVES_PER_CONVOLUTION: u32 = {};\n\
         const AMNS8_LOGICAL_WIDENED_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_LOGICAL_WRAPPING_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_LOGICAL_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_ISSUED_LEAF_SLOTS_PER_CONVOLUTION: u32 = {};\n\
         const AMNS8_ISSUED_WIDENED_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_ISSUED_WRAPPING_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_ISSUED_PRODUCTS_PER_FIELD: u32 = {};\n\
         const AMNS8_TYPED_PEAK_LIVE_WORDS: u32 = {};\n\
         const AMNS8_TOKENIZED_PEAK_LIVE_WORDS: u32 = {};\n\
         const AMNS8_TYPED_ALLOCATION_PEAK_LIVE_WORDS: u32 = {};\n\
         const AMNS8_TOKENIZED_ALLOCATION_PEAK_LIVE_WORDS: u32 = {};\n\
         const AMNS8_SOURCE_LIVE_WORD_LIMIT: u32 = {AMNS8_SOURCE_LIVE_WORD_LIMIT};\n",
        karatsuba_l4_schedule.peak_live_limbs,
        blocked128_schedule.typed_peak_live_words,
        blocked128_schedule.tokenized_peak_live_words,
        amns8_schedule.logical_leaves_per_convolution,
        amns8_schedule.logical_widened_products,
        amns8_schedule.logical_wrapping_products,
        amns8_schedule.logical_base_products,
        amns8_schedule.issued_leaf_slots_per_convolution,
        amns8_schedule.issued_widened_products_per_field,
        amns8_schedule.issued_wrapping_products_per_field,
        amns8_schedule.issued_widened_products_per_field
            + amns8_schedule.issued_wrapping_products_per_field,
        amns8_schedule.typed_peak_live_words,
        amns8_schedule.tokenized_peak_live_words,
        amns8_schedule.typed_allocation_peak_live_words,
        amns8_schedule.tokenized_allocation_peak_live_words,
    );
    fs::write(out.join("bls12_381_u32_constants.rs"), rust)
        .expect("write generated Rust constants");
    write_u32_fips_schedule(out);
    write_u32_named_fios_schedule(out, &modulus, inv);
    fs::write(out.join("bls12_381_u32_named_karatsuba_l4.metal"), karatsuba_l4_schedule.source)
        .expect("write generated Karatsuba-L4 schedule");
    fs::write(
        out.join("bls12_381_u32_named_blocked128_karatsuba_l4.metal"),
        blocked128_schedule.source,
    )
    .expect("write generated blocked-radix-128 Karatsuba-L4 schedule");
    fs::write(out.join("bls12_381_amns8.metal"), amns8_schedule.source)
        .expect("write generated AMNS8 schedule");
    fs::write(out.join("bls12_381_amns8_mulhi_v2.metal"), amns8_mulhi_v2_schedule.source)
        .expect("write generated AMNS8 mulhi-v2 schedule");
}

#[derive(Clone)]
struct FipsTerm {
    lhs: String,
    rhs: String,
}

fn write_u32_fips_schedule(out: &Path) {
    let mut product_seen = [[false; 12]; 12];
    let mut correction_seen = [[false; 12]; 12];
    let mut slots = 0_usize;
    let mut columns = String::new();

    for column in 0_usize..24 {
        let mut lane_terms: [Vec<FipsTerm>; 2] = [Vec::new(), Vec::new()];
        let begin = column.saturating_sub(11);
        let end = column.min(11);

        for (lhs_index, seen_row) in product_seen.iter_mut().enumerate().take(end + 1).skip(begin) {
            let rhs_index = column - lhs_index;
            let lane = lhs_index & 1;
            assert!(!seen_row[rhs_index]);
            seen_row[rhs_index] = true;
            let rhs_prefix = if (rhs_index & 1) == lane { "b" } else { "bp" };
            lane_terms[lane].push(FipsTerm {
                lhs: format!("a{}", lhs_index / 2),
                rhs: format!("{rhs_prefix}{}", rhs_index / 2),
            });
        }

        if column > 0 {
            let correction_end = (column - 1).min(11);
            for (correction_index, seen_row) in
                correction_seen.iter_mut().enumerate().take(correction_end + 1).skip(begin)
            {
                let modulus_index = column - correction_index;
                assert!((1..12).contains(&modulus_index));
                let lane = 1 - (correction_index & 1);
                assert!(!seen_row[modulus_index]);
                seen_row[modulus_index] = true;
                lane_terms[lane].push(FipsTerm {
                    lhs: format!("m{}", correction_index / 2),
                    rhs: format!("BLS12_381_U32_FP_MODULUS[{modulus_index}]"),
                });
            }
        }

        let column_slots = lane_terms[0].len().max(lane_terms[1].len());
        slots += column_slots;
        writeln!(columns, "    // Integrated product/reduction column {column}.")
            .expect("write FIPS column");
        writeln!(
            columns,
            "    U32FieldAccumulator accumulator_{column} = u32_field_accumulator_from_carry(carry_low, carry_high, side_mask);"
        )
        .expect("write FIPS accumulator");
        for slot in 0..column_slots {
            let lane0 = lane_terms[0].get(slot);
            let lane1 = lane_terms[1].get(slot);
            let lhs0 = lane0.map_or("0U", |term| term.lhs.as_str());
            let rhs0 = lane0.map_or("0U", |term| term.rhs.as_str());
            let lhs1 = lane1.map_or("0U", |term| term.lhs.as_str());
            let rhs1 = lane1.map_or("0U", |term| term.rhs.as_str());
            writeln!(
                columns,
                "    u32_field_accumulate_product(accumulator_{column}, u32_field_role_select({lhs0}, {lhs1}, side_mask), u32_field_role_select({rhs0}, {rhs1}, side_mask));"
            )
            .expect("write FIPS product slot");
        }
        writeln!(
            columns,
            "    const uint3 sum_{column} = u32_field_combine_accumulator(accumulator_{column});"
        )
        .expect("write FIPS combine");

        if column < 12 {
            let slot = column / 2;
            let owner = 1 - (column & 1);
            writeln!(
                columns,
                "    const uint correction_{column} = sum_{column}.x * BLS12_381_U32_FP_INV;"
            )
            .expect("write FIPS correction");
            writeln!(
                columns,
                "    m{slot} = u32_field_assign_owner(m{slot}, correction_{column}, side, {owner}U);"
            )
            .expect("write FIPS correction owner");
            writeln!(
                columns,
                "    const uint delta_{column} = mulhi(correction_{column}, BLS12_381_U32_FP_MODULUS[0]) + static_cast<uint>(sum_{column}.x != 0U);"
            )
            .expect("write FIPS cancellation high");
            writeln!(columns, "    carry_low = sum_{column}.y + delta_{column};")
                .expect("write FIPS carry low");
            writeln!(
                columns,
                "    carry_high = sum_{column}.z + static_cast<uint>(carry_low < sum_{column}.y);"
            )
            .expect("write FIPS carry high");
        } else {
            let output_index = column - 12;
            let slot = output_index / 2;
            let owner = output_index & 1;
            writeln!(
                columns,
                "    a{slot} = u32_field_assign_owner(a{slot}, sum_{column}.x, side, {owner}U);"
            )
            .expect("write FIPS output owner");
            writeln!(columns, "    carry_low = sum_{column}.y;")
                .expect("write FIPS output carry low");
            writeln!(columns, "    carry_high = sum_{column}.z;")
                .expect("write FIPS output carry high");
        }
    }

    assert_eq!(slots, U32_FIPS_SLOT_COUNT as usize);
    assert!(product_seen.into_iter().flatten().all(|seen| seen));
    for (correction_index, row) in correction_seen.into_iter().enumerate() {
        assert!(!row[0], "p0 is handled by the cancellation recurrence");
        for (modulus_index, seen) in row.into_iter().enumerate().skip(1) {
            assert!(seen, "missing m_{correction_index} * p_{modulus_index}");
        }
    }

    let mut shader = String::new();
    writeln!(shader, "// @generated by build.rs; do not edit.").expect("write FIPS header");
    writeln!(
        shader,
        "inline U32FieldHalf u32_field_fips_montgomery_multiply(U32FieldHalf lhs, U32FieldHalf rhs, uint side)"
    )
    .expect("write FIPS function");
    writeln!(shader, "{{").expect("write FIPS function");
    writeln!(shader, "    const uint side_mask = u32_field_side_mask(side);")
        .expect("write FIPS mask");
    for index in 0..6 {
        writeln!(shader, "    uint a{index} = lhs.limbs[{index}];").expect("write FIPS lhs");
        writeln!(shader, "    const uint b{index} = rhs.limbs[{index}];").expect("write FIPS rhs");
        writeln!(shader, "    uint m{index} = 0U;").expect("write FIPS correction slot");
    }
    writeln!(shader, "    const uint4 peer_b_03 = simd_shuffle_xor(uint4(b0, b1, b2, b3), 1);")
        .expect("write FIPS peer rhs");
    writeln!(shader, "    const uint2 peer_b_45 = simd_shuffle_xor(uint2(b4, b5), 1);")
        .expect("write FIPS peer rhs");
    for index in 0..4 {
        writeln!(shader, "    const uint bp{index} = peer_b_03[{index}];")
            .expect("write FIPS peer rhs scalar");
    }
    for index in 4..6 {
        writeln!(shader, "    const uint bp{index} = peer_b_45[{}];", index - 4)
            .expect("write FIPS peer rhs scalar");
    }
    writeln!(shader, "    uint carry_low = 0U;").expect("write FIPS carry");
    writeln!(shader, "    uint carry_high = 0U;").expect("write FIPS carry");
    shader.push_str(&columns);

    writeln!(shader, "    const uint4 local_flags_03 = uint4(").expect("write reduce flags");
    for index in 0..4 {
        let suffix = if index == 3 { "" } else { "," };
        writeln!(shader, "        u32_field_subtract_flags(a{index}, side, {index}U){suffix}")
            .expect("write reduce flags");
    }
    writeln!(shader, "    );").expect("write reduce flags");
    writeln!(shader, "    const uint2 local_flags_45 = uint2(").expect("write reduce flags");
    writeln!(shader, "        u32_field_subtract_flags(a4, side, 4U),")
        .expect("write reduce flags");
    writeln!(shader, "        u32_field_subtract_flags(a5, side, 5U)").expect("write reduce flags");
    writeln!(shader, "    );").expect("write reduce flags");
    writeln!(shader, "    const uint4 peer_flags_03 = simd_shuffle_xor(local_flags_03, 1);")
        .expect("write reduce peer flags");
    writeln!(shader, "    const uint2 peer_flags_45 = simd_shuffle_xor(local_flags_45, 1);")
        .expect("write reduce peer flags");
    writeln!(shader, "    uint borrow = 0U;").expect("write reduce borrow");
    for index in 0..6 {
        let peer = if index < 4 {
            format!("peer_flags_03[{index}]")
        } else {
            format!("peer_flags_45[{}]", index - 4)
        };
        let local = if index < 4 {
            format!("local_flags_03[{index}]")
        } else {
            format!("local_flags_45[{}]", index - 4)
        };
        writeln!(
            shader,
            "    m{index} = u32_field_subtract_pair(a{index}, {local}, {peer}, side, {index}U, borrow);"
        )
        .expect("write reduce round");
    }
    writeln!(shader, "    const uint use_subtracted = 0U - (borrow ^ 1U);")
        .expect("write reduce select");
    writeln!(shader, "    U32FieldHalf result;").expect("write FIPS result");
    for index in 0..6 {
        writeln!(
            shader,
            "    result.limbs[{index}] = (a{index} & ~use_subtracted) | (m{index} & use_subtracted);"
        )
        .expect("write FIPS result limb");
    }
    writeln!(shader, "    return result;").expect("write FIPS return");
    writeln!(shader, "}}").expect("write FIPS function");

    fs::write(out.join("bls12_381_u32_fips.metal"), shader).expect("write generated FIPS schedule");
}

fn write_u32_named_fios_schedule(out: &Path, modulus: &[u32; 12], inv: u32) {
    assert_eq!(modulus[0].wrapping_mul(inv), u32::MAX);
    let mut operand_seen = [[false; 12]; 12];
    let mut correction_seen = [[false; 12]; 12];
    let mut matrix_terms = 0_u32;
    let mut shader = String::new();

    writeln!(shader, "// @generated by build.rs; do not edit.").expect("write named FIOS header");
    writeln!(shader, "#ifndef BLS12_381_U32_NAMED_FIOS_GENERATED").expect("write named FIOS guard");
    writeln!(shader, "#define BLS12_381_U32_NAMED_FIOS_GENERATED").expect("write named FIOS guard");
    writeln!(shader, "struct U32NamedFieldElement {{").expect("write named field type");
    for index in 0..12 {
        writeln!(shader, "    uint x{index};").expect("write named field limb");
    }
    writeln!(shader, "}};").expect("write named field type");

    writeln!(shader, "inline __attribute__((always_inline))").expect("write named conversion");
    writeln!(shader, "U32NamedFieldElement u32_named_field_from_abi(U32FieldElement value)")
        .expect("write named conversion");
    writeln!(shader, "{{").expect("write named conversion");
    writeln!(shader, "    U32NamedFieldElement result;").expect("write named conversion");
    for index in 0..12 {
        writeln!(shader, "    result.x{index} = value.limbs[{index}];")
            .expect("write named conversion limb");
    }
    writeln!(shader, "    return result;").expect("write named conversion");
    writeln!(shader, "}}").expect("write named conversion");

    writeln!(shader, "inline __attribute__((always_inline))").expect("write ABI conversion");
    writeln!(shader, "U32FieldElement u32_named_field_to_abi(U32NamedFieldElement value)")
        .expect("write ABI conversion");
    writeln!(shader, "{{").expect("write ABI conversion");
    writeln!(shader, "    U32FieldElement result;").expect("write ABI conversion");
    for index in 0..12 {
        writeln!(shader, "    result.limbs[{index}] = value.x{index};")
            .expect("write ABI conversion limb");
    }
    writeln!(shader, "    return result;").expect("write ABI conversion");
    writeln!(shader, "}}").expect("write ABI conversion");

    writeln!(shader, "inline __attribute__((always_inline))").expect("write named FIOS function");
    writeln!(
        shader,
        "// Contract: both operands are reduced Montgomery residues strictly less than p."
    )
    .expect("write named FIOS input contract");
    writeln!(shader, "// The single final conditional subtraction relies on this invariant.")
        .expect("write named FIOS input contract");
    writeln!(
        shader,
        "U32NamedFieldElement u32_named_fios_montgomery_multiply(U32NamedFieldElement fixed, U32NamedFieldElement scanned)"
    )
    .expect("write named FIOS function");
    writeln!(shader, "{{").expect("write named FIOS function");
    for index in 0..13 {
        writeln!(shader, "    uint t{index} = 0U;").expect("write named FIOS state");
    }
    writeln!(
        shader,
        "    // Each straight-line round realizes T' = (T + fixed * digit + q * p) / 2^32."
    )
    .expect("write named FIOS invariant");

    for scanned_index in 0..12 {
        writeln!(shader, "    {{").expect("write named FIOS round");
        writeln!(shader, "        const uint digit = scanned.x{scanned_index};")
            .expect("write named FIOS digit");
        writeln!(
            shader,
            "        ulong word = static_cast<ulong>(fixed.x0) * static_cast<ulong>(digit) + static_cast<ulong>(t0);"
        )
        .expect("write named FIOS first product");
        writeln!(shader, "        uint low = static_cast<uint>(word);")
            .expect("write named FIOS low word");
        writeln!(shader, "        uint product_carry = static_cast<uint>(word >> 32);")
            .expect("write named FIOS product carry");
        writeln!(shader, "        const uint reduction = low * {inv:#010x}U;")
            .expect("write named FIOS correction digit");
        writeln!(
            shader,
            "        uint reduction_carry = mulhi(reduction, {:#010x}U) + static_cast<uint>(low != 0U);",
            modulus[0]
        )
        .expect("write named FIOS cancellation");
        assert!(!operand_seen[0][scanned_index]);
        operand_seen[0][scanned_index] = true;
        assert!(!correction_seen[scanned_index][0]);
        correction_seen[scanned_index][0] = true;
        matrix_terms += 2;

        for fixed_index in 1..12 {
            writeln!(
                shader,
                "        word = static_cast<ulong>(fixed.x{fixed_index}) * static_cast<ulong>(digit) + static_cast<ulong>(t{fixed_index}) + static_cast<ulong>(product_carry);"
            )
            .expect("write named FIOS product");
            writeln!(shader, "        low = static_cast<uint>(word);")
                .expect("write named FIOS product low");
            writeln!(shader, "        product_carry = static_cast<uint>(word >> 32);")
                .expect("write named FIOS product carry");
            writeln!(
                shader,
                "        word = static_cast<ulong>(reduction) * static_cast<ulong>({:#010x}U) + static_cast<ulong>(low) + static_cast<ulong>(reduction_carry);",
                modulus[fixed_index]
            )
            .expect("write named FIOS reduction");
            writeln!(shader, "        t{} = static_cast<uint>(word);", fixed_index - 1)
                .expect("write named FIOS reduced limb");
            writeln!(shader, "        reduction_carry = static_cast<uint>(word >> 32);")
                .expect("write named FIOS reduction carry");
            assert!(!operand_seen[fixed_index][scanned_index]);
            operand_seen[fixed_index][scanned_index] = true;
            assert!(!correction_seen[scanned_index][fixed_index]);
            correction_seen[scanned_index][fixed_index] = true;
            matrix_terms += 2;
        }

        writeln!(
            shader,
            "        const ulong top = static_cast<ulong>(t12) + static_cast<ulong>(product_carry);"
        )
        .expect("write named FIOS top");
        writeln!(
            shader,
            "        word = static_cast<ulong>(static_cast<uint>(top)) + static_cast<ulong>(reduction_carry);"
        )
        .expect("write named FIOS top reduction");
        writeln!(shader, "        t11 = static_cast<uint>(word);")
            .expect("write named FIOS final limb");
        writeln!(
            shader,
            "        t12 = static_cast<uint>(top >> 32) + static_cast<uint>(word >> 32);"
        )
        .expect("write named FIOS top carry");
        writeln!(shader, "    }}").expect("write named FIOS round");
    }

    assert_eq!(matrix_terms, U32_NAMED_FIELD_MATRIX_TERM_COUNT);
    assert!(operand_seen.into_iter().flatten().all(|seen| seen));
    assert!(correction_seen.into_iter().flatten().all(|seen| seen));

    writeln!(shader, "    uint borrow = 0U;").expect("write named final reduction");
    writeln!(shader, "    U32NamedFieldElement subtracted;").expect("write named final reduction");
    for (index, limb) in modulus.iter().enumerate() {
        writeln!(shader, "    {{").expect("write named subtract limb");
        writeln!(shader, "        const uint first = t{index} - {limb:#010x}U;")
            .expect("write named subtract limb");
        writeln!(shader, "        const uint difference = first - borrow;")
            .expect("write named subtract limb");
        writeln!(
            shader,
            "        const uint first_borrow = static_cast<uint>(t{index} < {limb:#010x}U);"
        )
        .expect("write named subtract borrow");
        writeln!(shader, "        const uint second_borrow = static_cast<uint>(first < borrow);")
            .expect("write named subtract borrow");
        writeln!(shader, "        subtracted.x{index} = difference;")
            .expect("write named subtract result");
        writeln!(shader, "        borrow = first_borrow | second_borrow;")
            .expect("write named subtract borrow");
        writeln!(shader, "    }}").expect("write named subtract limb");
    }
    writeln!(shader, "    borrow = static_cast<uint>(t12 < borrow);")
        .expect("write named top-word subtraction");
    writeln!(shader, "    const uint use_subtracted = 0U - (borrow ^ 1U);")
        .expect("write named subtract selection");
    writeln!(shader, "    U32NamedFieldElement result;").expect("write named result");
    for index in 0..12 {
        writeln!(
            shader,
            "    result.x{index} = (t{index} & ~use_subtracted) | (subtracted.x{index} & use_subtracted);"
        )
        .expect("write named result limb");
    }
    writeln!(shader, "    return result;").expect("write named FIOS return");
    writeln!(shader, "}}").expect("write named FIOS function");
    write_u32_named_square_schedule(&mut shader, modulus, inv);
    writeln!(shader, "#endif").expect("write named FIOS guard");

    fs::write(out.join("bls12_381_u32_named_fios.metal"), shader)
        .expect("write generated named FIOS schedule");
}

fn write_u32_named_square_schedule(shader: &mut String, modulus: &[u32; 12], inv: u32) {
    writeln!(shader, "__attribute__((noinline))").expect("write named square attribute");
    writeln!(shader, "// Contract: value is a reduced Montgomery residue strictly less than p.")
        .expect("write named square contract");
    writeln!(
        shader,
        "U32NamedFieldElement u32_named_montgomery_square(U32NamedFieldElement value)"
    )
    .expect("write named square signature");
    writeln!(shader, "{{").expect("write named square body");
    for index in 0..25 {
        writeln!(shader, "    uint t{index} = 0U;").expect("write named square product limb");
    }
    writeln!(shader, "    ulong column_carry = 0UL;").expect("write square column carry");

    let mut square_products = 0_u32;
    for column in 0_usize..=22 {
        writeln!(shader, "    {{").expect("write square column scope");
        writeln!(shader, "        ulong accumulator = column_carry;")
            .expect("write square column accumulator");
        writeln!(shader, "        uint accumulator_high = 0U;")
            .expect("write square column high word");
        let first = column.saturating_sub(11);
        let last = column.min(11);
        for lhs in first..=last {
            let rhs = column - lhs;
            if lhs > rhs {
                continue;
            }
            square_products += 1;
            writeln!(shader, "        {{").expect("write square product scope");
            writeln!(
                shader,
                "            const ulong product = static_cast<ulong>(value.x{lhs}) * static_cast<ulong>(value.x{rhs});"
            )
            .expect("write square product");
            writeln!(shader, "            ulong next = accumulator + product;")
                .expect("write square product sum");
            writeln!(
                shader,
                "            accumulator_high += static_cast<uint>(next < accumulator);"
            )
            .expect("write square product carry");
            writeln!(shader, "            accumulator = next;")
                .expect("write square product accumulation");
            if lhs != rhs {
                writeln!(shader, "            next = accumulator + product;")
                    .expect("write doubled square product sum");
                writeln!(
                    shader,
                    "            accumulator_high += static_cast<uint>(next < accumulator);"
                )
                .expect("write doubled square product carry");
                writeln!(shader, "            accumulator = next;")
                    .expect("write doubled square product accumulation");
            }
            writeln!(shader, "        }}").expect("write square product scope");
        }
        writeln!(shader, "        t{column} = static_cast<uint>(accumulator);")
            .expect("write square product limb");
        writeln!(
            shader,
            "        column_carry = (accumulator >> 32) | (static_cast<ulong>(accumulator_high) << 32);"
        )
        .expect("write square next-column carry");
        writeln!(shader, "    }}").expect("write square column scope");
    }
    assert_eq!(square_products, 78, "12-limb symmetric square product count");
    writeln!(shader, "    t23 = static_cast<uint>(column_carry);")
        .expect("write square top product limb");

    let mut reduction_products = 0_u32;
    for round in 0_usize..12 {
        writeln!(shader, "    {{").expect("write square REDC round");
        writeln!(shader, "        const uint reduction = t{round} * {inv:#010x}U;")
            .expect("write square REDC digit");
        writeln!(
            shader,
            "        uint carry = mulhi(reduction, {:#010x}U) + static_cast<uint>(t{round} != 0U);",
            modulus[0]
        )
        .expect("write square REDC cancellation");
        writeln!(shader, "        ulong word = 0UL;").expect("write square REDC word");
        reduction_products += 1;
        for (modulus_index, limb) in modulus.iter().enumerate().skip(1) {
            let product_index = round + modulus_index;
            writeln!(
                shader,
                "        word = static_cast<ulong>(reduction) * static_cast<ulong>({limb:#010x}U) + static_cast<ulong>(t{product_index}) + static_cast<ulong>(carry);"
            )
            .expect("write square REDC product");
            writeln!(shader, "        t{product_index} = static_cast<uint>(word);")
                .expect("write square REDC limb");
            writeln!(shader, "        carry = static_cast<uint>(word >> 32);")
                .expect("write square REDC carry");
            reduction_products += 1;
        }
        for product_index in (round + 12)..=24 {
            writeln!(
                shader,
                "        word = static_cast<ulong>(t{product_index}) + static_cast<ulong>(carry);"
            )
            .expect("write square REDC carry propagation");
            writeln!(shader, "        t{product_index} = static_cast<uint>(word);")
                .expect("write square REDC propagated limb");
            writeln!(shader, "        carry = static_cast<uint>(word >> 32);")
                .expect("write square REDC propagated carry");
        }
        writeln!(shader, "    }}").expect("write square REDC round");
    }
    assert_eq!(reduction_products, 144, "12-round Montgomery correction count");

    writeln!(shader, "    uint borrow = 0U;").expect("write square final subtraction");
    writeln!(shader, "    U32NamedFieldElement subtracted;")
        .expect("write square final subtraction");
    for (index, limb) in modulus.iter().enumerate() {
        let product_index = index + 12;
        writeln!(shader, "    {{").expect("write square subtract limb");
        writeln!(shader, "        const uint first = t{product_index} - {limb:#010x}U;")
            .expect("write square subtract limb");
        writeln!(shader, "        const uint difference = first - borrow;")
            .expect("write square subtract limb");
        writeln!(
            shader,
            "        const uint first_borrow = static_cast<uint>(t{product_index} < {limb:#010x}U);"
        )
        .expect("write square subtract borrow");
        writeln!(shader, "        const uint second_borrow = static_cast<uint>(first < borrow);")
            .expect("write square subtract borrow");
        writeln!(shader, "        subtracted.x{index} = difference;")
            .expect("write square subtract result");
        writeln!(shader, "        borrow = first_borrow | second_borrow;")
            .expect("write square subtract borrow");
        writeln!(shader, "    }}").expect("write square subtract limb");
    }
    writeln!(shader, "    borrow = static_cast<uint>(t24 < borrow);")
        .expect("write square top-word subtraction");
    writeln!(shader, "    const uint use_subtracted = 0U - (borrow ^ 1U);")
        .expect("write square subtraction mask");
    writeln!(shader, "    U32NamedFieldElement result;").expect("write named square result");
    for index in 0..12 {
        let product_index = index + 12;
        writeln!(
            shader,
            "    result.x{index} = (t{product_index} & ~use_subtracted) | (subtracted.x{index} & use_subtracted);"
        )
        .expect("write named square result limb");
    }
    writeln!(shader, "    return result;").expect("write named square return");
    writeln!(shader, "}}").expect("write named square body");
}

#[allow(dead_code)]
struct KaratsubaDifference {
    limbs: Vec<String>,
    nonnegative: String,
}

#[allow(dead_code)]
struct KaratsubaScheduleEmitter {
    source: String,
    next_identifier: usize,
    base_products: u32,
}

impl KaratsubaScheduleEmitter {
    fn new() -> Self {
        Self { source: String::new(), next_identifier: 0, base_products: 0 }
    }

    fn fresh(&mut self) -> String {
        let identifier = format!("v{}", self.next_identifier);
        self.next_identifier += 1;
        identifier
    }

    fn emit_difference(&mut self, lhs: &[String], rhs: &[String]) -> KaratsubaDifference {
        assert_eq!(lhs.len(), rhs.len());
        let mut forward = Vec::with_capacity(lhs.len());
        let mut incoming_borrow = "0U".to_owned();
        for (lhs_limb, rhs_limb) in lhs.iter().zip(rhs) {
            let first = self.fresh();
            let difference = self.fresh();
            let first_borrow = self.fresh();
            let second_borrow = self.fresh();
            let borrow = self.fresh();
            writeln!(self.source, "    const uint {first} = {lhs_limb} - {rhs_limb};")
                .expect("write Karatsuba difference");
            writeln!(self.source, "    const uint {difference} = {first} - {incoming_borrow};")
                .expect("write Karatsuba difference");
            writeln!(
                self.source,
                "    const uint {first_borrow} = static_cast<uint>({lhs_limb} < {rhs_limb});"
            )
            .expect("write Karatsuba difference");
            writeln!(
                self.source,
                "    const uint {second_borrow} = static_cast<uint>({first} < {incoming_borrow});"
            )
            .expect("write Karatsuba difference");
            writeln!(self.source, "    const uint {borrow} = {first_borrow} | {second_borrow};")
                .expect("write Karatsuba difference");
            forward.push(difference);
            incoming_borrow = borrow;
        }

        let selection_mask = self.fresh();
        let nonnegative = self.fresh();
        writeln!(self.source, "    const uint {selection_mask} = 0U - {incoming_borrow};")
            .expect("write Karatsuba difference mask");
        writeln!(self.source, "    const uint {nonnegative} = {incoming_borrow} ^ 1U;")
            .expect("write Karatsuba difference sign");

        let mut magnitude = Vec::with_capacity(lhs.len());
        let mut complement_carry = "1U".to_owned();
        for forward_limb in forward {
            let complement_word = self.fresh();
            let reverse = self.fresh();
            let selected = self.fresh();
            writeln!(
                self.source,
                "    const ulong {complement_word} = static_cast<ulong>(~{forward_limb}) + static_cast<ulong>({complement_carry});"
            )
            .expect("write Karatsuba difference complement");
            writeln!(
                self.source,
                "    const uint {reverse} = static_cast<uint>({complement_word});"
            )
            .expect("write Karatsuba difference complement");
            writeln!(
                self.source,
                "    const uint {selected} = ({forward_limb} & ~{selection_mask}) | ({reverse} & {selection_mask});"
            )
            .expect("write Karatsuba difference selection");
            magnitude.push(selected);
            complement_carry = format!("static_cast<uint>({complement_word} >> 32)");
        }

        KaratsubaDifference { limbs: magnitude, nonnegative }
    }

    fn emit_add_into(&mut self, result: &[String], addend: &[String], shift: usize) {
        assert!(shift + addend.len() <= result.len());
        let mut carry = "0U".to_owned();
        for (destination, result_limb) in result.iter().enumerate().skip(shift) {
            let addend_limb = addend.get(destination - shift).map_or("0U", String::as_str);
            let sum = self.fresh();
            writeln!(
                self.source,
                "    const ulong {sum} = static_cast<ulong>({}) + static_cast<ulong>({addend_limb}) + static_cast<ulong>({carry});",
                result_limb
            )
            .expect("write Karatsuba accumulation");
            writeln!(self.source, "    {result_limb} = static_cast<uint>({sum});")
                .expect("write Karatsuba accumulation");
            carry = format!("static_cast<uint>({sum} >> 32)");
        }
    }

    fn emit_signed_add_into(
        &mut self,
        result: &[String],
        magnitude: &[String],
        shift: usize,
        negative: &str,
    ) {
        assert!(shift + magnitude.len() <= result.len());
        let sign_mask = self.fresh();
        writeln!(self.source, "    const uint {sign_mask} = 0U - {negative};")
            .expect("write signed Karatsuba mask");
        let mut carry = negative.to_owned();
        for (destination, result_limb) in result.iter().enumerate().skip(shift) {
            let signed_limb = magnitude
                .get(destination - shift)
                .map_or_else(|| sign_mask.clone(), |limb| format!("({limb} ^ {sign_mask})"));
            let sum = self.fresh();
            writeln!(
                self.source,
                "    const ulong {sum} = static_cast<ulong>({}) + static_cast<ulong>({signed_limb}) + static_cast<ulong>({carry});",
                result_limb
            )
            .expect("write signed Karatsuba accumulation");
            writeln!(self.source, "    {result_limb} = static_cast<uint>({sum});")
                .expect("write signed Karatsuba accumulation");
            carry = format!("static_cast<uint>({sum} >> 32)");
        }
    }

    fn emit_multiply(&mut self, lhs: &[String], rhs: &[String]) -> Vec<String> {
        assert_eq!(lhs.len(), rhs.len());
        assert!(!lhs.is_empty());
        let limbs = lhs.len();
        if limbs == 1 {
            let product = self.fresh();
            let low = self.fresh();
            let high = self.fresh();
            writeln!(
                self.source,
                "    const ulong {product} = static_cast<ulong>({}) * static_cast<ulong>({});",
                lhs[0], rhs[0]
            )
            .expect("write Karatsuba base product");
            writeln!(self.source, "    const uint {low} = static_cast<uint>({product});")
                .expect("write Karatsuba base product");
            writeln!(self.source, "    const uint {high} = static_cast<uint>({product} >> 32);")
                .expect("write Karatsuba base product");
            self.base_products += 1;
            return vec![low, high];
        }

        let low_limbs = limbs.div_ceil(2);
        let high_limbs = limbs / 2;
        writeln!(
            self.source,
            "    // Depth-first subtractive Karatsuba K{limbs}: K{low_limbs} + K{high_limbs} + K{low_limbs}."
        )
        .expect("write Karatsuba node");

        let low = self.emit_multiply(&lhs[..low_limbs], &rhs[..low_limbs]);
        let mut result = Vec::with_capacity(2 * limbs);
        for index in 0..(2 * limbs) {
            let limb = self.fresh();
            let initial = low.get(index).map_or("0U", String::as_str);
            writeln!(self.source, "    uint {limb} = {initial};")
                .expect("write Karatsuba in-place result");
            result.push(limb);
        }
        self.emit_add_into(&result, &low, low_limbs);

        let high = self.emit_multiply(&lhs[low_limbs..], &rhs[low_limbs..]);
        self.emit_add_into(&result, &high, low_limbs);
        self.emit_add_into(&result, &high, 2 * low_limbs);

        let mut lhs_high = lhs[low_limbs..].to_vec();
        lhs_high.resize(low_limbs, "0U".to_owned());
        let mut rhs_high = rhs[low_limbs..].to_vec();
        rhs_high.resize(low_limbs, "0U".to_owned());
        let lhs_difference = self.emit_difference(&lhs[..low_limbs], &lhs_high);
        let rhs_difference = self.emit_difference(&rhs_high, &rhs[..low_limbs]);
        let middle = self.emit_multiply(&lhs_difference.limbs, &rhs_difference.limbs);
        let negative = self.fresh();
        writeln!(
            self.source,
            "    const uint {negative} = {} ^ {};",
            lhs_difference.nonnegative, rhs_difference.nonnegative
        )
        .expect("write Karatsuba middle sign");
        self.emit_signed_add_into(&result, &middle, low_limbs, &negative);
        result
    }
}

struct GeneratedKaratsubaL4Schedule {
    source: String,
    peak_live_limbs: u32,
}

struct NamedLimbLivenessAudit {
    live: Vec<(&'static str, u32)>,
    current: u32,
    peak: u32,
}

impl NamedLimbLivenessAudit {
    fn new() -> Self {
        Self { live: Vec::new(), current: 0, peak: 0 }
    }

    fn declare(&mut self, name: &'static str, limbs: u32) {
        assert!(!self.live.iter().any(|(live_name, _)| *live_name == name));
        self.live.push((name, limbs));
        self.current += limbs;
        self.peak = self.peak.max(self.current);
    }

    fn replace(&mut self, consumed: &'static str, produced: &'static str, limbs: u32) {
        let index = self
            .live
            .iter()
            .position(|(name, _)| *name == consumed)
            .unwrap_or_else(|| panic!("replace unknown live aggregate {consumed}"));
        let consumed_limbs = self.live[index].1;
        let base = self.current - consumed_limbs;
        // The consumed aggregate's last use is the produced aggregate's
        // initializer, so a liveness transition needs max(input, output), not
        // their sum. This matches the source-level kill/definition boundary.
        self.peak = self.peak.max(base + consumed_limbs.max(limbs));
        self.live[index] = (produced, limbs);
        self.current = base + limbs;
    }

    fn resize(&mut self, name: &'static str, limbs: u32) {
        let entry = self
            .live
            .iter_mut()
            .find(|(live_name, _)| *live_name == name)
            .unwrap_or_else(|| panic!("resize unknown live aggregate {name}"));
        self.current = self.current - entry.1 + limbs;
        entry.1 = limbs;
        self.peak = self.peak.max(self.current);
    }

    fn drop(&mut self, name: &'static str) {
        let index = self
            .live
            .iter()
            .position(|(live_name, _)| *live_name == name)
            .unwrap_or_else(|| panic!("drop unknown live aggregate {name}"));
        self.current -= self.live[index].1;
        self.live.swap_remove(index);
    }
}

fn generate_u32_named_karatsuba_l4_schedule(
    modulus: &[u32; 12],
    negative_inverse: &[u32; 12],
) -> GeneratedKaratsubaL4Schedule {
    assert_eq!(karatsuba_product_count(2), 3);
    assert_eq!(karatsuba_product_count(3), 7);
    assert_eq!(karatsuba_product_count(6), 21);
    assert_eq!(karatsuba_product_count(12), U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12);
    assert_eq!(3 * karatsuba_product_count(12), U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC);

    let product_lhs: Vec<_> = (0..12).map(|index| format!("lhs.x{index}")).collect();
    let product_rhs: Vec<_> = (0..12).map(|index| format!("rhs.x{index}")).collect();
    let mut product_emitter = KaratsubaScheduleEmitter::new();
    let product = product_emitter.emit_multiply(&product_lhs, &product_rhs);

    let multiplier_lhs: Vec<_> = (0..12).map(|index| format!("product.x{index}")).collect();
    let multiplier_rhs: Vec<_> =
        negative_inverse.iter().map(|value| format!("{value:#010x}U")).collect();
    let mut multiplier_emitter = KaratsubaScheduleEmitter::new();
    let multiplier_product = multiplier_emitter.emit_multiply(&multiplier_lhs, &multiplier_rhs);

    let correction_lhs: Vec<_> = (0..12).map(|index| format!("multiplier.x{index}")).collect();
    let correction_rhs: Vec<_> = modulus.iter().map(|value| format!("{value:#010x}U")).collect();
    let mut correction_emitter = KaratsubaScheduleEmitter::new();
    let correction = correction_emitter.emit_multiply(&correction_lhs, &correction_rhs);

    for emitter in [&product_emitter, &multiplier_emitter, &correction_emitter] {
        assert_eq!(emitter.base_products, U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12);
        assert!(!emitter.source.contains("for ("));
        assert!(!emitter.source.contains('['));
    }
    assert_eq!(product.len(), 24);
    assert_eq!(multiplier_product.len(), 24);
    assert_eq!(correction.len(), 24);

    let mut shader = String::new();
    writeln!(shader, "// @generated by build.rs; do not edit.").expect("write Karatsuba-L4 header");
    writeln!(shader, "struct U32NamedWideElement {{").expect("write named wide type");
    for index in 0..24 {
        writeln!(shader, "    uint x{index};").expect("write named wide limb");
    }
    writeln!(shader, "}};").expect("write named wide type");

    writeln!(shader, "inline __attribute__((always_inline))").expect("write Karatsuba function");
    writeln!(
        shader,
        "U32NamedWideElement u32_named_subtractive_karatsuba_k12(U32NamedFieldElement lhs, U32NamedFieldElement rhs)"
    )
    .expect("write Karatsuba function");
    writeln!(shader, "{{").expect("write Karatsuba function");
    shader.push_str(&product_emitter.source);
    writeln!(shader, "    U32NamedWideElement product;").expect("write Karatsuba result");
    for (index, value) in product.iter().enumerate() {
        writeln!(shader, "    product.x{index} = {value};").expect("write Karatsuba result limb");
    }
    writeln!(shader, "    return product;").expect("write Karatsuba result");
    writeln!(shader, "}}\n").expect("write Karatsuba function");

    writeln!(shader, "inline __attribute__((always_inline))")
        .expect("write Karatsuba multiplier function");
    writeln!(
        shader,
        "U32NamedFieldElement u32_named_karatsuba_l4_multiplier(U32NamedWideElement product)"
    )
    .expect("write Karatsuba multiplier function");
    writeln!(shader, "{{").expect("write Karatsuba multiplier function");
    shader.push_str(&multiplier_emitter.source);
    writeln!(shader, "    U32NamedFieldElement multiplier;")
        .expect("write Karatsuba multiplier result");
    for (index, value) in multiplier_product.iter().take(12).enumerate() {
        writeln!(shader, "    multiplier.x{index} = {value};")
            .expect("write Karatsuba multiplier limb");
    }
    writeln!(shader, "    return multiplier;").expect("write Karatsuba multiplier result");
    writeln!(shader, "}}\n").expect("write Karatsuba multiplier function");

    writeln!(shader, "inline __attribute__((always_inline))")
        .expect("write Karatsuba correction function");
    writeln!(
        shader,
        "U32NamedWideElement u32_named_karatsuba_l4_correction(U32NamedFieldElement multiplier)"
    )
    .expect("write Karatsuba correction function");
    writeln!(shader, "{{").expect("write Karatsuba correction function");
    shader.push_str(&correction_emitter.source);
    writeln!(shader, "    U32NamedWideElement correction;")
        .expect("write Karatsuba correction result");
    for (index, value) in correction.iter().enumerate() {
        writeln!(shader, "    correction.x{index} = {value};")
            .expect("write Karatsuba correction limb");
    }
    writeln!(shader, "    return correction;").expect("write Karatsuba correction result");
    writeln!(shader, "}}\n").expect("write Karatsuba correction function");

    writeln!(shader, "inline __attribute__((always_inline))")
        .expect("write Karatsuba-L4 REDC function");
    writeln!(
        shader,
        "U32NamedFieldElement u32_named_karatsuba_l4_redc(U32NamedFieldElement lhs, U32NamedFieldElement rhs)"
    )
    .expect("write Karatsuba-L4 REDC function");
    writeln!(shader, "{{").expect("write Karatsuba-L4 REDC function");
    let mut liveness = NamedLimbLivenessAudit::new();
    writeln!(
        shader,
        "    // Contract: lhs,rhs < 4p. The raw result is < 3p; no canonical subtraction is performed."
    )
    .expect("write Karatsuba-L4 contract");
    writeln!(shader, "    // Top-level named-limb liveness is generated and bounded by build.rs.")
        .expect("write Karatsuba-L4 liveness contract");
    writeln!(
        shader,
        "    U32NamedWideElement product = u32_named_subtractive_karatsuba_k12(lhs, rhs);"
    )
    .expect("write Karatsuba-L4 product");
    liveness.declare("product", 24);
    writeln!(
        shader,
        "    const U32NamedFieldElement multiplier = u32_named_karatsuba_l4_multiplier(product);"
    )
    .expect("write Karatsuba-L4 multiplier");
    liveness.declare("multiplier", 12);
    writeln!(shader, "    {{").expect("write scoped Karatsuba-L4 correction");
    writeln!(
        shader,
        "        const U32NamedWideElement correction = u32_named_karatsuba_l4_correction(multiplier);"
    )
    .expect("write Karatsuba-L4 correction product");
    liveness.replace("multiplier", "correction", 24);
    let mut carry = "0U".to_owned();
    for index in 0..24 {
        writeln!(
            shader,
            "        const ulong sum_{index} = static_cast<ulong>(product.x{index}) + static_cast<ulong>(correction.x{index}) + static_cast<ulong>({carry});"
        )
        .expect("write Karatsuba-L4 REDC addition");
        if index >= 12 {
            writeln!(shader, "        product.x{} = static_cast<uint>(sum_{index});", index - 12)
                .expect("write in-place Karatsuba-L4 REDC result");
        }
        carry = format!("static_cast<uint>(sum_{index} >> 32)");
    }
    writeln!(shader, "    }}").expect("write scoped Karatsuba-L4 correction");
    liveness.drop("correction");
    liveness.resize("product", 12);
    writeln!(shader, "    U32NamedFieldElement result;").expect("write Karatsuba-L4 result");
    liveness.replace("product", "result", 12);
    for index in 0..12 {
        writeln!(shader, "    result.x{index} = product.x{index};")
            .expect("write Karatsuba-L4 result limb");
    }
    writeln!(shader, "    return result;").expect("write Karatsuba-L4 result");
    liveness.drop("result");
    writeln!(shader, "}}\n").expect("write Karatsuba-L4 REDC function");

    assert!(!shader.contains("uint limbs["));
    assert_eq!(liveness.current, 0);
    assert!(liveness.live.is_empty());
    assert_eq!(liveness.peak, 48);
    assert!(liveness.peak <= U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT);
    GeneratedKaratsubaL4Schedule { source: shader, peak_live_limbs: liveness.peak }
}

const fn karatsuba_product_count(limbs: usize) -> u32 {
    if limbs == 1 {
        1
    } else {
        let low_limbs = limbs.div_ceil(2);
        let high_limbs = limbs / 2;
        2 * karatsuba_product_count(low_limbs) + karatsuba_product_count(high_limbs)
    }
}

fn parse_le_u32_limbs<const N: usize>(hex: &str) -> [u32; N] {
    assert_eq!(hex.len(), N * 8, "canonical modulus width");
    let mut limbs = [0_u32; N];
    for (index, limb) in limbs.iter_mut().enumerate() {
        let end = hex.len() - index * 8;
        let start = end - 8;
        *limb = u32::from_str_radix(&hex[start..end], 16).expect("valid modulus hex");
    }
    limbs
}

fn montgomery_inverse_word(modulus: u32) -> u32 {
    assert_eq!(modulus & 1, 1, "Montgomery modulus must be odd");
    let mut inverse = 1_u32;
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2_u32.wrapping_sub(modulus.wrapping_mul(inverse)));
    }
    inverse.wrapping_neg()
}

fn negative_inverse_mod_radix(modulus: &[u32; 12], low_negative_inverse: u32) -> [u32; 12] {
    let positive_low_inverse = low_negative_inverse.wrapping_neg();
    assert_eq!(modulus[0].wrapping_mul(positive_low_inverse), 1);
    let mut result = [0_u32; 12];
    let mut carry = 0_u128;
    for column in 0..12 {
        let mut sum = carry;
        for modulus_index in 1..=column {
            sum += u128::from(modulus[modulus_index]) * u128::from(result[column - modulus_index]);
        }
        let digit = u32::MAX.wrapping_sub(sum as u32).wrapping_mul(positive_low_inverse);
        result[column] = digit;
        sum += u128::from(modulus[0]) * u128::from(digit);
        assert_eq!(sum as u32, u32::MAX, "negative inverse column {column}");
        carry = sum >> 32;
    }
    assert_eq!(result[0], low_negative_inverse);
    result
}

fn multiply_limbs_by_small(limbs: &[u32; 12], multiplier: u32) -> [u32; 12] {
    let mut result = [0_u32; 12];
    let mut carry = 0_u64;
    for (index, limb) in limbs.iter().copied().enumerate() {
        let value = u64::from(limb) * u64::from(multiplier) + carry;
        result[index] = value as u32;
        carry = value >> 32;
    }
    assert_eq!(carry, 0, "small multiple must fit the 384-bit field ABI");
    result
}

fn u32_field_abi_fingerprint(modulus: &[u32; 12], inv: u32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in modulus.iter().copied().chain([
        inv,
        U32_FIELD_ALGORITHM_VERSION,
        12,
        2,
        384,
        U32_FIPS_SLOT_COUNT,
    ]) {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn u32_named_field_abi_fingerprint(modulus: &[u32; 12], inv: u32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in modulus.iter().copied().chain([
        inv,
        U32_NAMED_FIELD_ALGORITHM_VERSION,
        12,
        1,
        384,
        U32_NAMED_FIELD_MATRIX_TERM_COUNT,
        U32_NAMED_FIELD_VARIANT_COUNT,
    ]) {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn u32_karatsuba_l4_abi_fingerprint(
    modulus: &[u32; 12],
    negative_inverse: &[u32; 12],
    schedule_fingerprint: u64,
    peak_live_limbs: u32,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in modulus.iter().copied().chain(negative_inverse.iter().copied()).chain([
        U32_KARATSUBA_L4_ALGORITHM_VERSION,
        12,
        1,
        384,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_K12,
        U32_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_KARATSUBA_L4_LAZY_MULTIPLE,
        U32_KARATSUBA_L4_SOURCE_LIVE_LIMB_LIMIT,
        U32_KARATSUBA_L4_FLATTENED_SOURCE_PEAK_LIVE_LIMBS,
        peak_live_limbs,
        schedule_fingerprint as u32,
        (schedule_fingerprint >> 32) as u32,
        0,
    ]) {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn u32_blocked128_karatsuba_l4_abi_fingerprint(
    modulus: &[u32; 12],
    block_negative_inverse: &[u32; 4],
    schedule_fingerprint: u64,
    typed_peak_live_words: u32,
    tokenized_peak_live_words: u32,
    source_peak_live_words: u32,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in modulus.iter().copied().chain(block_negative_inverse.iter().copied()).chain([
        U32_BLOCKED128_KARATSUBA_L4_ALGORITHM_VERSION,
        12,
        1,
        384,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_K4,
        U32_BLOCKED128_KARATSUBA_L4_BASE_PRODUCTS_PER_REDC,
        U32_BLOCKED128_KARATSUBA_L4_BLOCK_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_ROUNDS,
        U32_BLOCKED128_KARATSUBA_L4_ACCUMULATOR_LIMBS,
        U32_BLOCKED128_KARATSUBA_L4_LAZY_MULTIPLE,
        typed_peak_live_words,
        tokenized_peak_live_words,
        source_peak_live_words,
        U32_BLOCKED128_KARATSUBA_L4_SOURCE_LIVE_WORD_LIMIT,
        schedule_fingerprint as u32,
        (schedule_fingerprint >> 32) as u32,
        23,
    ]) {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn stable_source_fingerprint(source: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in source {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn write_msm_constants(out: &Path) {
    let mut inverse = 1_u64;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2_u64.wrapping_sub(MSM_FP_MODULUS[0].wrapping_mul(inverse)));
    }
    let inv = inverse.wrapping_neg();
    assert_eq!(MSM_FP_MODULUS[0].wrapping_mul(inv), u64::MAX);
    let mut fr_inverse = 1_u64;
    for _ in 0..6 {
        fr_inverse =
            fr_inverse.wrapping_mul(2_u64.wrapping_sub(MSM_FR_MODULUS[0].wrapping_mul(fr_inverse)));
    }
    let fr_inv = fr_inverse.wrapping_neg();
    assert_eq!(MSM_FR_MODULUS[0].wrapping_mul(fr_inv), u64::MAX);

    let mut montgomery_r = [0_u64; 6];
    montgomery_r[0] = 1;
    for _ in 0..384 {
        montgomery_r = msm_double_mod(montgomery_r, MSM_FP_MODULUS);
    }
    let modulus_minus_two = msm_subtract(MSM_FP_MODULUS, [2, 0, 0, 0, 0, 0]);

    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for word in std::iter::once(MSM_SHADER_ABI_VERSION)
        .chain(MSM_FP_MODULUS)
        .chain([inv])
        .chain(montgomery_r)
    {
        for byte in word.to_le_bytes() {
            fingerprint ^= u64::from(byte);
            fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    assert_eq!(
        fingerprint, MSM_EXPECTED_CONSTANTS_FINGERPRINT,
        "standalone MSM constants changed; bump the shader ABI and review the frozen fingerprint"
    );

    let mut metal = String::from(
        "// @generated by build.rs; do not edit.\n\
         #ifndef BLS12_381_METAL_MSM_CONSTANTS\n\
         #define BLS12_381_METAL_MSM_CONSTANTS\n\n",
    );
    writeln!(metal, "constant ulong BLS12_381_METAL_MSM_ABI_VERSION = {MSM_SHADER_ABI_VERSION}UL;")
        .expect("write MSM ABI version");
    writeln!(
        metal,
        "constant ulong BLS12_381_METAL_MSM_CONSTANTS_FINGERPRINT = {fingerprint:#018x}UL;"
    )
    .expect("write MSM constants fingerprint");
    writeln!(
        metal,
        "constant ulong BLS12_381_METAL_MSM_EXPECTED_LAYOUT_FINGERPRINT = {MSM_EXPECTED_LAYOUT_FINGERPRINT:#018x}UL;"
    )
    .expect("write MSM layout fingerprint");
    write_msm_metal_array(&mut metal, "BLS12_381_METAL_MSM_FP_MODULUS", &MSM_FP_MODULUS);
    write_msm_metal_array(
        &mut metal,
        "BLS12_381_METAL_MSM_FP_MODULUS_MINUS_TWO",
        &modulus_minus_two,
    );
    writeln!(metal, "constant ulong BLS12_381_METAL_MSM_FP_INV = {inv:#018x}UL;")
        .expect("write MSM Montgomery inverse");
    write_msm_metal_array(&mut metal, "BLS12_381_METAL_MSM_FP_R", &montgomery_r);
    write_msm_metal_array(&mut metal, "BLS12_381_METAL_MSM_FR_MODULUS", &MSM_FR_MODULUS);
    writeln!(metal, "constant ulong BLS12_381_METAL_MSM_FR_INV = {fr_inv:#018x}UL;")
        .expect("write MSM scalar Montgomery inverse");
    metal.push_str("\n#endif\n");
    fs::write(out.join("bls12_381_msm_constants.metal"), metal)
        .expect("write standalone MSM Metal constants");
}

fn msm_double_mod(mut value: [u64; 6], modulus: [u64; 6]) -> [u64; 6] {
    let mut carry = 0_u64;
    for limb in &mut value {
        let next_carry = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next_carry;
    }
    assert_eq!(carry, 0, "BLS12-381 Fp doubling overflow");
    if msm_greater_or_equal(&value, &modulus) { msm_subtract(value, modulus) } else { value }
}

fn msm_greater_or_equal(lhs: &[u64; 6], rhs: &[u64; 6]) -> bool {
    for index in (0..6).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn msm_subtract(lhs: [u64; 6], rhs: [u64; 6]) -> [u64; 6] {
    let mut output = [0_u64; 6];
    let mut borrow = false;
    for index in 0..6 {
        let (difference, first_borrow) = lhs[index].overflowing_sub(rhs[index]);
        let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
        output[index] = difference;
        borrow = first_borrow || second_borrow;
    }
    assert!(!borrow, "BLS12-381 Fp modular subtraction underflow");
    output
}

fn write_msm_metal_array<const N: usize>(output: &mut String, name: &str, values: &[u64; N]) {
    write!(output, "constant ulong {name}[{N}] = {{").expect("write MSM Metal array prefix");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{value:#018x}UL").expect("write MSM Metal array limb");
    }
    output.push_str("};\n");
}

fn developer_dir() -> Option<PathBuf> {
    if let Some(path) = env::var_os("DEVELOPER_DIR").map(PathBuf::from) {
        return Some(path);
    }
    if let Ok(selected) = Command::new("xcode-select").arg("-p").output()
        && selected.status.success()
    {
        let path = PathBuf::from(String::from_utf8_lossy(&selected.stdout).trim());
        if path.join("Platforms").is_dir() {
            return Some(path);
        }
    }
    let standard = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
    standard.is_dir().then_some(standard)
}

fn run_xcrun(
    developer_dir: Option<&Path>,
    tool: &str,
    args: &[OsString],
) -> std::io::Result<Output> {
    let mut command = Command::new("xcrun");
    if let Some(developer_dir) = developer_dir {
        command.env("DEVELOPER_DIR", developer_dir);
    }
    command.arg("--sdk").arg("macosx").arg(tool).args(args).output()
}

fn require_success(action: &str, artifact: &str, output: &Output) {
    if output.status.success() {
        return;
    }
    panic!(
        "Metal shader {action} failed for {artifact}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
}

fn require_air_deployment_target(air: &Path) {
    let bytes = fs::read(air).unwrap_or_else(|error| panic!("read {}: {error}", air.display()));
    let expected = format!("apple-macosx{MACOS_DEPLOYMENT_TARGET}.0");
    assert!(
        bytes.windows(expected.len()).any(|window| window == expected.as_bytes()),
        "{} is not stamped for {expected}",
        air.display()
    );
}
