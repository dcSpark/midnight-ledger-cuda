// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Representation-safe, backend-neutral values used by the Metal MSM API.
//!
//! Field coordinates are raw Montgomery residues because conversion belongs at
//! the application boundary, not in a GPU submission. Scalars are different:
//! routing consumes canonical integers, so construction rejects values outside
//! the BLS12-381 scalar field.

use core::fmt;

use zeroize::Zeroize;

/// BLS12-381 scalar modulus in little-endian 64-bit limbs.
pub const BLS12_381_SCALAR_MODULUS: [u64; 4] =
    [0xffff_ffff_0000_0001, 0x53bd_a402_fffe_5bfe, 0x3339_d808_09a1_d805, 0x73ed_a753_299d_7d48];

/// A canonical BLS12-381 scalar represented by little-endian limbs.
///
/// The private field preserves the invariant `self < r`. Use
/// [`CanonicalScalar::from_le_limbs`] at trust boundaries.
#[repr(C)]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct CanonicalScalar {
    limbs: [u64; 4],
}

impl CanonicalScalar {
    /// The additive identity.
    pub const ZERO: Self = Self { limbs: [0; 4] };

    /// Validates and constructs a scalar from little-endian limbs.
    pub const fn from_le_limbs(limbs: [u64; 4]) -> Result<Self, ScalarEncodingError> {
        if limbs_are_less_than(&limbs, &BLS12_381_SCALAR_MODULUS) {
            Ok(Self { limbs })
        } else {
            Err(ScalarEncodingError::NotCanonical)
        }
    }

    /// Validates and constructs a scalar from 32 little-endian bytes.
    pub const fn from_le_bytes(bytes: [u8; 32]) -> Result<Self, ScalarEncodingError> {
        let mut limbs = [0_u64; 4];
        let mut limb_index = 0;
        while limb_index < limbs.len() {
            let offset = limb_index * 8;
            limbs[limb_index] = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            limb_index += 1;
        }
        Self::from_le_limbs(limbs)
    }

    /// Returns the canonical little-endian limbs.
    #[must_use]
    pub const fn to_le_limbs(self) -> [u64; 4] {
        self.limbs
    }

    /// Returns the canonical little-endian bytes.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        let mut limb_index = 0;
        while limb_index < self.limbs.len() {
            let limb = self.limbs[limb_index].to_le_bytes();
            let offset = limb_index * 8;
            let mut byte_index = 0;
            while byte_index < limb.len() {
                bytes[offset + byte_index] = limb[byte_index];
                byte_index += 1;
            }
            limb_index += 1;
        }
        bytes
    }

    /// Returns whether this scalar is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }
}

impl fmt::Debug for CanonicalScalar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Scalars can be witness-derived. Avoid making accidental log leakage
        // convenient while retaining a useful marker in structural diagnostics.
        formatter.write_str("CanonicalScalar(REDACTED)")
    }
}

impl Zeroize for CanonicalScalar {
    fn zeroize(&mut self) {
        self.limbs.zeroize();
    }
}

/// A raw BLS12-381 scalar-field value in Montgomery form.
///
/// This POD deliberately carries no reduction invariant. Fixed-base
/// Montgomery-input APIs validate every value on Metal before recoding and
/// fail the entire submission if any integer is greater than or equal to the
/// scalar modulus. It exists so field-native producers need not convert large
/// witness vectors to canonical integers on the CPU.
#[repr(C)]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct RawFrMontgomery {
    /// Four little-endian 64-bit Montgomery limbs.
    pub limbs: [u64; 4],
}

impl fmt::Debug for RawFrMontgomery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RawFrMontgomery(REDACTED)")
    }
}

impl Zeroize for RawFrMontgomery {
    fn zeroize(&mut self) {
        self.limbs.zeroize();
    }
}

/// Failure to decode a canonical BLS12-381 scalar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarEncodingError {
    /// The integer is greater than or equal to the scalar modulus.
    NotCanonical,
}

impl fmt::Display for ScalarEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCanonical => formatter.write_str("scalar is not canonical BLS12-381 Fr"),
        }
    }
}

/// A raw BLS12-381 base-field Montgomery residue.
///
/// This type deliberately does not claim that the residue is reduced.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawFpMontgomery {
    /// Six little-endian 64-bit Montgomery limbs.
    pub limbs: [u64; 6],
}

/// Raw affine G1 coordinates in the Metal ABI's Montgomery representation.
///
/// Prepared bases must be finite, on-curve, and in the prime-order subgroup;
/// those semantic checks are performed by `PreparedBases`, not by this POD.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawG1Affine {
    /// Affine x-coordinate.
    pub x: RawFpMontgomery,
    /// Affine y-coordinate.
    pub y: RawFpMontgomery,
}

/// Raw Jacobian G1 coordinates in the Metal ABI's Montgomery representation.
///
/// A zero z-coordinate is the point at infinity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawG1Jacobian {
    /// Jacobian x-coordinate.
    pub x: RawFpMontgomery,
    /// Jacobian y-coordinate.
    pub y: RawFpMontgomery,
    /// Jacobian z-coordinate; zero denotes infinity.
    pub z: RawFpMontgomery,
}

/// Supported fixed Booth window widths.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WindowBits {
    /// Six-bit windows: 43 windows and 32 magnitude buckets per MSM.
    W6 = 6,
    /// Seven-bit windows: 37 windows and 64 magnitude buckets per MSM.
    W7 = 7,
    /// Eight-bit windows: 32 windows and 128 magnitude buckets per MSM.
    W8 = 8,
    /// Nine-bit windows: 29 windows and 256 magnitude buckets per MSM.
    W9 = 9,
    /// Ten-bit windows: 26 windows and 512 magnitude buckets per MSM.
    W10 = 10,
}

/// Window widths supported by the pre-shifted fixed-base MSM path.
///
/// Unlike [`WindowBits`], these widths are part of prepared-base identity: a
/// prepared allocation can only be used with the width that produced it.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FixedWindowBits {
    /// Eleven-bit windows: 24 shifted layers and 1,024 magnitude buckets.
    W11 = 11,
    /// Twelve-bit windows: 22 shifted layers and 2,048 magnitude buckets.
    W12 = 12,
    /// Thirteen-bit windows: 20 shifted layers and 4,096 magnitude buckets.
    W13 = 13,
    /// Fourteen-bit windows: 19 shifted layers and 8,192 magnitude buckets.
    W14 = 14,
    /// Fifteen-bit windows: 18 shifted layers and 16,384 magnitude buckets.
    W15 = 15,
    /// Sixteen-bit windows: 16 shifted layers and 32,768 magnitude buckets.
    W16 = 16,
}

impl FixedWindowBits {
    /// Returns the numeric window width.
    #[must_use]
    pub const fn get(self) -> u32 {
        self as u32
    }

    /// Returns the exact number of shifted layers needed for a 255-bit scalar.
    #[must_use]
    pub const fn windows_per_scalar(self) -> u32 {
        255 / self.get() + 1
    }

    /// Returns the number of shared magnitude buckets per MSM output.
    #[must_use]
    pub const fn buckets_per_output(self) -> u32 {
        1_u32 << (self.get() - 1)
    }
}

impl WindowBits {
    /// Returns the numeric window width.
    #[must_use]
    pub const fn get(self) -> u32 {
        self as u32
    }

    /// Returns the exact number of windows needed for a 255-bit Fr integer.
    #[must_use]
    pub const fn windows_per_scalar(self) -> u32 {
        match self {
            Self::W6 => 43,
            Self::W7 => 37,
            Self::W8 => 32,
            Self::W9 => 29,
            Self::W10 => 26,
        }
    }

    /// Returns the number of non-zero magnitude buckets in each window.
    #[must_use]
    pub const fn buckets_per_window(self) -> u32 {
        1_u32 << (self.get() - 1)
    }
}

/// One ragged MSM job over shared prepared bases and a shared scalar array.
///
/// Term `i` pairs `bases[base_offset + i]` with
/// `scalars[scalar_offset + i]`. Output indices must form a permutation of
/// `0..jobs.len()`, which keeps the GPU collection buffer dense.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MsmJobV1 {
    /// First prepared base used by this job.
    pub base_offset: u32,
    /// First canonical scalar used by this job.
    pub scalar_offset: u32,
    /// Number of base/scalar pairs; zero represents the identity MSM.
    pub term_count: u32,
    /// Dense destination in the returned output array.
    pub output_index: u32,
}

const fn limbs_are_less_than(lhs: &[u64; 4], rhs: &[u64; 4]) -> bool {
    let mut index = lhs.len();
    while index != 0 {
        index -= 1;
        if lhs[index] < rhs[index] {
            return true;
        }
        if lhs[index] > rhs[index] {
            return false;
        }
    }
    false
}

const _: [(); 32] = [(); core::mem::size_of::<CanonicalScalar>()];
const _: [(); 32] = [(); core::mem::size_of::<RawFrMontgomery>()];
const _: [(); 48] = [(); core::mem::size_of::<RawFpMontgomery>()];
const _: [(); 96] = [(); core::mem::size_of::<RawG1Affine>()];
const _: [(); 144] = [(); core::mem::size_of::<RawG1Jacobian>()];
const _: [(); 16] = [(); core::mem::size_of::<MsmJobV1>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_boundary_is_fail_closed() {
        let mut modulus_minus_one = BLS12_381_SCALAR_MODULUS;
        modulus_minus_one[0] -= 1;

        assert_eq!(CanonicalScalar::from_le_limbs([0; 4]), Ok(CanonicalScalar::ZERO));
        assert!(CanonicalScalar::from_le_limbs(modulus_minus_one).is_ok());
        assert_eq!(
            CanonicalScalar::from_le_limbs(BLS12_381_SCALAR_MODULUS),
            Err(ScalarEncodingError::NotCanonical)
        );
        assert_eq!(
            CanonicalScalar::from_le_limbs([u64::MAX; 4]),
            Err(ScalarEncodingError::NotCanonical)
        );
    }

    #[test]
    fn scalar_byte_round_trip_is_little_endian() {
        let scalar = CanonicalScalar::from_le_limbs([1, 2, 3, 4]).unwrap();
        let bytes = scalar.to_le_bytes();
        assert_eq!(&bytes[0..8], &1_u64.to_le_bytes());
        assert_eq!(&bytes[24..32], &4_u64.to_le_bytes());
        assert_eq!(CanonicalScalar::from_le_bytes(bytes), Ok(scalar));
    }

    #[test]
    fn window_shapes_are_frozen() {
        assert_eq!(
            (WindowBits::W6.windows_per_scalar(), WindowBits::W6.buckets_per_window()),
            (43, 32)
        );
        assert_eq!(
            (WindowBits::W7.windows_per_scalar(), WindowBits::W7.buckets_per_window()),
            (37, 64)
        );
        assert_eq!(
            (WindowBits::W8.windows_per_scalar(), WindowBits::W8.buckets_per_window()),
            (32, 128)
        );
        assert_eq!(
            (WindowBits::W9.windows_per_scalar(), WindowBits::W9.buckets_per_window()),
            (29, 256)
        );
        assert_eq!(
            (WindowBits::W10.windows_per_scalar(), WindowBits::W10.buckets_per_window()),
            (26, 512)
        );
        assert_eq!(
            (FixedWindowBits::W11.windows_per_scalar(), FixedWindowBits::W11.buckets_per_output()),
            (24, 1_024)
        );
        assert_eq!(
            (FixedWindowBits::W12.windows_per_scalar(), FixedWindowBits::W12.buckets_per_output()),
            (22, 2_048)
        );
        assert_eq!(
            (FixedWindowBits::W13.windows_per_scalar(), FixedWindowBits::W13.buckets_per_output()),
            (20, 4_096)
        );
        assert_eq!(
            (FixedWindowBits::W14.windows_per_scalar(), FixedWindowBits::W14.buckets_per_output()),
            (19, 8_192)
        );
        assert_eq!(
            (FixedWindowBits::W15.windows_per_scalar(), FixedWindowBits::W15.buckets_per_output()),
            (18, 16_384)
        );
        assert_eq!(
            (FixedWindowBits::W16.windows_per_scalar(), FixedWindowBits::W16.buckets_per_output()),
            (16, 32_768)
        );
    }
}
