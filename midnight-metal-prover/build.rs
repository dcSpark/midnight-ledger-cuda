// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const METAL_ROOT: &str = "midnight.metal";
const MACOS_DEPLOYMENT_TARGET: &str = "13.0";
const METAL_SOURCES: &[&str] = &[
    METAL_ROOT,
    "proof_digest.metal",
    "memory.metal",
    "field.metal",
    "expression.metal",
    "fp32.metal",
    "batch_affine.metal",
    "flat_segmented.metal",
    "ntt.metal",
    "quotient.metal",
    "multiopen.metal",
    "g1.metal",
];

const BLS12_381_ABI_VERSION: u64 = 1;
const FR_MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];
const FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

struct FieldConstants<const N: usize> {
    modulus: [u64; N],
    modulus_minus_two: [u64; N],
    inv: u64,
    r: [u64; N],
    r2: [u64; N],
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(midnight_metal_shaders)");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    for source in METAL_SOURCES {
        println!("cargo:rerun-if-changed=shaders/{source}");
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    generate_bls12_381_constants(&out);

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo sets target OS");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo sets target architecture");
    if target_os != "macos" || target_arch != "aarch64" {
        return;
    }

    let sdk = "macosx";
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let developer_dir = developer_dir();
    let source = manifest.join("shaders").join(METAL_ROOT);
    let air = out.join(Path::new(METAL_ROOT).with_extension("air"));
    let args = vec![
        OsString::from("-std=metal3.0"),
        // Keep the embedded AIR loadable on the documented runtime floor.
        // Without this flag, recent Xcodes silently stamp their own host SDK
        // version into the AIR (for example macOS 26 with Xcode 26).
        OsString::from(format!("-mmacosx-version-min={MACOS_DEPLOYMENT_TARGET}")),
        OsString::from("-I"),
        out.as_os_str().to_owned(),
        OsString::from("-I"),
        manifest.join("shaders").as_os_str().to_owned(),
        OsString::from("-c"),
        source.as_os_str().to_owned(),
        OsString::from("-o"),
        air.as_os_str().to_owned(),
    ];
    let output = run_xcrun(developer_dir.as_deref(), sdk, "metal", &args)
        .unwrap_or_else(|error| panic!("run Metal compiler for {METAL_ROOT}: {error}"));
    require_success("compile", METAL_ROOT, &output);
    require_air_deployment_target(&air);

    let library = out.join("midnight-metal.metallib");
    let mut link_args = vec![air.as_os_str().to_owned()];
    link_args.push(OsString::from("-o"));
    link_args.push(library.as_os_str().to_owned());
    let output = run_xcrun(developer_dir.as_deref(), sdk, "metallib", &link_args)
        .unwrap_or_else(|error| panic!("run Metal library linker: {error}"));
    require_success("link", "midnight-metal.metallib", &output);

    println!("cargo:rustc-cfg=midnight_metal_shaders");
}

fn require_air_deployment_target(air: &Path) {
    let bytes = fs::read(air).unwrap_or_else(|error| panic!("read {}: {error}", air.display()));
    let expected = format!("apple-macosx{MACOS_DEPLOYMENT_TARGET}.0");
    assert!(
        bytes
            .windows(expected.len())
            .any(|window| window == expected.as_bytes()),
        "{} is not stamped for the required {expected} deployment target",
        air.display()
    );
}

fn generate_bls12_381_constants(out: &Path) {
    let fr = field_constants(FR_MODULUS);
    let fp = field_constants(FP_MODULUS);
    let fingerprint = constants_fingerprint(&fr, &fp);

    let mut rust = String::new();
    writeln!(
        rust,
        "pub(crate) const BLS12_381_ABI_VERSION: u64 = {BLS12_381_ABI_VERSION};"
    )
    .expect("write generated Rust ABI version");
    writeln!(
        rust,
        "pub(crate) const BLS12_381_CONSTANTS_FINGERPRINT: u64 = {fingerprint:#018x};"
    )
    .expect("write generated Rust fingerprint");
    write_rust_field(&mut rust, "FR", &fr);
    write_rust_field(&mut rust, "FP", &fp);
    write_rust_fp32(&mut rust, &fp);
    fs::write(out.join("bls12_381_constants.rs"), rust)
        .expect("write generated BLS12-381 Rust constants");

    let mut metal = String::from(
        "#ifndef MIDNIGHT_BLS12_381_CONSTANTS_METAL\n#define MIDNIGHT_BLS12_381_CONSTANTS_METAL\n\n",
    );
    writeln!(
        metal,
        "constant ulong MIDNIGHT_BLS12_381_ABI_VERSION = {BLS12_381_ABI_VERSION}UL;"
    )
    .expect("write generated Metal ABI version");
    writeln!(
        metal,
        "constant ulong MIDNIGHT_BLS12_381_CONSTANTS_FINGERPRINT = {fingerprint:#018x}UL;"
    )
    .expect("write generated Metal fingerprint");
    write_metal_field(&mut metal, "FR", &fr);
    write_metal_field(&mut metal, "FP", &fp);
    write_metal_fp32(&mut metal, &fp);
    metal.push_str("\n#endif\n");
    fs::write(out.join("bls12_381_constants.metal"), metal)
        .expect("write generated BLS12-381 Metal constants");
}

fn field_constants<const N: usize>(modulus: [u64; N]) -> FieldConstants<N> {
    assert!(N > 0, "field modulus must contain at least one limb");
    assert_eq!(modulus[0] & 1, 1, "Montgomery modulus must be odd");
    assert_eq!(
        modulus[N - 1] >> 63,
        0,
        "double-and-reduce must not overflow"
    );

    let mut two = [0u64; N];
    two[0] = 2;
    let modulus_minus_two = subtract(modulus, two);

    let mut r = [0u64; N];
    r[0] = 1;
    for _ in 0..(N * u64::BITS as usize) {
        r = double_mod(r, modulus);
    }
    let mut r2 = r;
    for _ in 0..(N * u64::BITS as usize) {
        r2 = double_mod(r2, modulus);
    }

    let mut inverse = 1u64;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(modulus[0].wrapping_mul(inverse)));
    }
    let inv = inverse.wrapping_neg();
    assert_eq!(modulus[0].wrapping_mul(inv), u64::MAX);

    FieldConstants {
        modulus,
        modulus_minus_two,
        inv,
        r,
        r2,
    }
}

fn double_mod<const N: usize>(value: [u64; N], modulus: [u64; N]) -> [u64; N] {
    let mut doubled = [0u64; N];
    let mut carry = 0u64;
    for index in 0..N {
        doubled[index] = (value[index] << 1) | carry;
        carry = value[index] >> 63;
    }
    assert_eq!(
        carry, 0,
        "field modulus is too wide for no-overflow doubling"
    );
    if greater_or_equal(&doubled, &modulus) {
        subtract(doubled, modulus)
    } else {
        doubled
    }
}

fn greater_or_equal<const N: usize>(lhs: &[u64; N], rhs: &[u64; N]) -> bool {
    for index in (0..N).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] > rhs[index];
        }
    }
    true
}

fn subtract<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> [u64; N] {
    let mut output = [0u64; N];
    let mut borrow = false;
    for index in 0..N {
        let (difference, first_borrow) = lhs[index].overflowing_sub(rhs[index]);
        let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
        output[index] = difference;
        borrow = first_borrow || second_borrow;
    }
    assert!(!borrow, "modular subtraction underflowed");
    output
}

fn constants_fingerprint(fr: &FieldConstants<4>, fp: &FieldConstants<6>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for word in std::iter::once(BLS12_381_ABI_VERSION)
        .chain(fr.modulus)
        .chain(fr.modulus_minus_two)
        .chain([fr.inv])
        .chain(fr.r)
        .chain(fr.r2)
        .chain(fp.modulus)
        .chain(fp.modulus_minus_two)
        .chain([fp.inv])
        .chain(fp.r)
        .chain(fp.r2)
    {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn write_rust_field<const N: usize>(
    output: &mut String,
    name: &str,
    constants: &FieldConstants<N>,
) {
    writeln!(
        output,
        "pub(crate) const {name}_MODULUS: [u64; {N}] = {:?};",
        constants.modulus
    )
    .expect("write generated Rust modulus");
    writeln!(
        output,
        "pub(crate) const {name}_MODULUS_MINUS_TWO: [u64; {N}] = {:?};",
        constants.modulus_minus_two
    )
    .expect("write generated Rust inversion exponent");
    writeln!(
        output,
        "pub(crate) const {name}_INV: u64 = {:#018x};",
        constants.inv
    )
    .expect("write generated Rust Montgomery inverse");
    writeln!(
        output,
        "pub(crate) const {name}_R: [u64; {N}] = {:?};",
        constants.r
    )
    .expect("write generated Rust Montgomery R");
    writeln!(
        output,
        "pub(crate) const {name}_R2: [u64; {N}] = {:?};",
        constants.r2
    )
    .expect("write generated Rust Montgomery R2");
}

fn write_metal_field<const N: usize>(
    output: &mut String,
    name: &str,
    constants: &FieldConstants<N>,
) {
    write_metal_array(
        output,
        &format!("MIDNIGHT_BLS12_381_{name}_MODULUS"),
        &constants.modulus,
    );
    write_metal_array(
        output,
        &format!("MIDNIGHT_BLS12_381_{name}_MODULUS_MINUS_TWO"),
        &constants.modulus_minus_two,
    );
    writeln!(
        output,
        "constant ulong MIDNIGHT_BLS12_381_{name}_INV = {:#018x}UL;",
        constants.inv
    )
    .expect("write generated Metal Montgomery inverse");
    write_metal_array(
        output,
        &format!("MIDNIGHT_BLS12_381_{name}_R"),
        &constants.r,
    );
    write_metal_array(
        output,
        &format!("MIDNIGHT_BLS12_381_{name}_R2"),
        &constants.r2,
    );
}

fn write_metal_array<const N: usize>(output: &mut String, name: &str, values: &[u64; N]) {
    write!(output, "constant ulong {name}[{N}] = {{").expect("write Metal array prefix");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{value:#018x}UL").expect("write Metal array limb");
    }
    output.push_str("};\n");
}

fn write_rust_fp32(output: &mut String, fp: &FieldConstants<6>) {
    let modulus = split_u64_limbs(&fp.modulus);
    let r = split_u64_limbs(&fp.r);
    let r2 = split_u64_limbs(&fp.r2);
    let inv = fp.inv as u32;
    assert_eq!(
        (fp.modulus[0] as u32).wrapping_mul(inv),
        u32::MAX,
        "32-bit Montgomery inverse must satisfy -p^-1 mod 2^32"
    );
    writeln!(
        output,
        "pub(crate) const FP32_MODULUS: [u32; 12] = {modulus:?};"
    )
    .expect("write generated Rust Fp32 modulus");
    writeln!(output, "pub(crate) const FP32_INV: u32 = {inv:#010x};")
        .expect("write generated Rust Fp32 Montgomery inverse");
    writeln!(output, "pub(crate) const FP32_R: [u32; 12] = {r:?};")
        .expect("write generated Rust Fp32 Montgomery R");
    writeln!(output, "pub(crate) const FP32_R2: [u32; 12] = {r2:?};")
        .expect("write generated Rust Fp32 Montgomery R2");
}

fn write_metal_fp32(output: &mut String, fp: &FieldConstants<6>) {
    let modulus = split_u64_limbs(&fp.modulus);
    let r = split_u64_limbs(&fp.r);
    let r2 = split_u64_limbs(&fp.r2);
    let inv = fp.inv as u32;
    write_metal_u32_array(output, "MIDNIGHT_BLS12_381_FP32_MODULUS", &modulus);
    writeln!(
        output,
        "constant uint MIDNIGHT_BLS12_381_FP32_INV = {inv:#010x}U;"
    )
    .expect("write generated Metal Fp32 Montgomery inverse");
    write_metal_u32_array(output, "MIDNIGHT_BLS12_381_FP32_R", &r);
    write_metal_u32_array(output, "MIDNIGHT_BLS12_381_FP32_R2", &r2);
}

fn split_u64_limbs(values: &[u64]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|value| [*value as u32, (*value >> 32) as u32])
        .collect()
}

fn write_metal_u32_array(output: &mut String, name: &str, values: &[u32]) {
    write!(output, "constant uint {name}[{}] = {{", values.len())
        .expect("write Metal u32 array prefix");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{value:#010x}U").expect("write Metal u32 array limb");
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
    sdk: &str,
    tool: &str,
    args: &[OsString],
) -> std::io::Result<Output> {
    let mut command = Command::new("xcrun");
    if let Some(developer_dir) = developer_dir {
        command.env("DEVELOPER_DIR", developer_dir);
    }
    command.arg("--sdk").arg(sdk).arg(tool).args(args).output()
}

fn require_success(action: &str, artifact: &str, output: &Output) {
    if output.status.success() {
        return;
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    panic!(
        "Metal shader {action} failed for {artifact}. Install the Metal toolchain with \
         `xcodebuild -downloadComponent metalToolchain`. If Xcode is not selected, run \
         `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`. Compiler output: {}",
        stderr.trim()
    );
}
