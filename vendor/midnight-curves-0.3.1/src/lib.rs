//! # `blstrs`
//!
//! An implementation of the BLS12-381 pairing-friendly elliptic curve
//! construction.

#![deny(clippy::perf, clippy::correctness)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::wrong_self_convention)]

#[cfg(not(target_endian = "little"))]
compile_error!("blstrs is only supported on little endian architectures");

#[macro_use]
mod arithmetic;

#[macro_use]
mod derive;

mod curve;
pub mod ff_ext;
pub mod fft;
#[cfg(any(test, feature = "dev-curves"))]
pub mod hash_to_curve;
pub mod msm;
mod perf_timer;
pub mod serde;
pub mod serde_traits;

#[cfg(feature = "gpu-sppark")]
pub mod gpu_sppark {
    use std::any::TypeId;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};

    use ff::Field;

    static ENABLED: AtomicBool = AtomicBool::new(false);

    thread_local! {
        static DISABLE_DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    /// Suppresses sppark dispatch on this thread without changing the backend
    /// selected for other concurrent proofs.
    #[derive(Debug)]
    pub struct DisableGuard;

    impl DisableGuard {
        pub fn new() -> Self {
            DISABLE_DEPTH.set(DISABLE_DEPTH.get() + 1);
            Self
        }
    }

    impl Drop for DisableGuard {
        fn drop(&mut self) {
            DISABLE_DEPTH.set(DISABLE_DEPTH.get().saturating_sub(1));
        }
    }

    /// Sets whether sppark-backed curve operations are used by this process.
    pub fn set_enabled(enabled: bool) -> bool {
        ENABLED.swap(enabled, Ordering::SeqCst)
    }

    /// Returns whether sppark-backed curve operations are currently enabled.
    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed) && DISABLE_DEPTH.get() == 0
    }

    pub fn try_bls12_381_fr_mul_assign<F: Field + 'static>(values: &mut [F], scalar: F) -> bool {
        if !vector_ops_enabled(values.len())
            || TypeId::of::<F>() != TypeId::of::<crate::Fq>()
            || scalar == F::ONE
        {
            return false;
        }

        let scalar = raw_from_scalar(scalar);
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_mul_assign(values, scalar)
            .expect("sppark BLS12-381 scalar-field vector multiply failed");
        true
    }

    pub fn try_bls12_381_fr_mul_assign_zeta<F: Field + 'static>(
        values: &mut [F],
        first: F,
        second: F,
    ) -> bool {
        if !vector_ops_enabled(values.len()) || TypeId::of::<F>() != TypeId::of::<crate::Fq>() {
            return false;
        }

        let first = raw_from_scalar(first);
        let second = raw_from_scalar(second);
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_mul_assign_zeta(values, first, second)
            .expect("sppark BLS12-381 scalar-field zeta multiply failed");
        true
    }

    pub fn try_bls12_381_fr_mul_assign_repeating<F: Field + 'static>(
        values: &mut [F],
        factors: &[F],
    ) -> bool {
        if !vector_ops_enabled(values.len())
            || factors.is_empty()
            || TypeId::of::<F>() != TypeId::of::<crate::Fq>()
        {
            return false;
        }

        let raw_factors = factors
            .iter()
            .copied()
            .map(raw_from_scalar)
            .collect::<Vec<_>>();
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_mul_assign_repeating(values, &raw_factors)
            .expect("sppark BLS12-381 scalar-field repeating multiply failed");
        true
    }

    pub fn try_bls12_381_fr_shifted_prefix_product<F: Field + 'static>(
        values: &mut [F],
        initial: F,
    ) -> bool {
        if !prefix_ops_enabled(values.len()) || TypeId::of::<F>() != TypeId::of::<crate::Fq>() {
            return false;
        }

        let initial = raw_from_scalar(initial);
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_shifted_prefix_product(values, initial).is_ok()
    }

    pub fn try_bls12_381_fr_invert_assign<F: Field + 'static>(values: &mut [F]) -> bool {
        if !invert_ops_enabled(values.len()) || TypeId::of::<F>() != TypeId::of::<crate::Fq>() {
            return false;
        }

        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_invert_assign(values).is_ok()
    }

    pub fn try_bls12_381_fr_sort_canonical<F: Field + 'static>(values: &mut [F]) -> bool {
        let sort_enabled = matches!(
            std::env::var("MIDNIGHT_GPU_LOOKUP_SORT_BACKEND")
                .unwrap_or_else(|_| "off".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "cub"
        );
        if !sort_enabled
            || !enabled()
            || values.len() < 2
            || TypeId::of::<F>() != TypeId::of::<crate::Fq>()
        {
            return false;
        }

        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_sort_canonical(values)
            .expect("sppark BLS12-381 scalar-field canonical sort failed");
        true
    }

    pub fn try_bls12_381_fr_coeff_to_extended_batch<F: Field + 'static>(
        inputs: &[&[F]],
        output_len: usize,
        extended_log_n: u32,
        first: F,
        second: F,
    ) -> Option<Vec<Vec<F>>> {
        if inputs.is_empty() || TypeId::of::<F>() != TypeId::of::<crate::Fq>() {
            return None;
        }
        let input_len = inputs[0].len();
        if !coeff_to_extended_enabled(input_len, inputs.len())
            || input_len == 0
            || output_len < input_len
            || !input_len.is_power_of_two()
            || !output_len.is_power_of_two()
            || inputs.iter().any(|input| input.len() != input_len)
        {
            return None;
        }

        let flat_inputs = inputs
            .iter()
            .flat_map(|input| input.iter().copied().map(raw_from_scalar))
            .collect::<Vec<_>>();
        let mut flat_output = vec![F::ZERO; output_len * inputs.len()];
        let raw_output = raw_values(&mut flat_output);
        midnight_sppark_msm::bls12_381_fr_coeff_to_extended_batch(
            raw_output,
            &flat_inputs,
            input_len,
            inputs.len(),
            extended_log_n,
            raw_from_scalar(first),
            raw_from_scalar(second),
        )
        .ok()?;

        Some(
            flat_output
                .chunks(output_len)
                .map(<[F]>::to_vec)
                .collect::<Vec<_>>(),
        )
    }

    pub fn try_bls12_381_fr_ntt_zeta<F: Field + 'static>(
        values: &mut [F],
        inverse: bool,
        scale: F,
        first: F,
        second: F,
    ) -> bool {
        if !fused_ntt_enabled(values.len()) || TypeId::of::<F>() != TypeId::of::<crate::Fq>() {
            return false;
        }

        let scale = raw_from_scalar(scale);
        let first = raw_from_scalar(first);
        let second = raw_from_scalar(second);
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_ntt_zeta(values, inverse, scale, first, second)
            .expect("sppark BLS12-381 scalar-field fused NTT/zeta failed");
        true
    }

    pub fn try_bls12_381_fr_divide_ntt_zeta<F: Field + 'static>(
        values: &mut [F],
        factors: &[F],
        scale: F,
        first: F,
        second: F,
    ) -> bool {
        if !quotient_backend_enabled(values.len())
            || factors.is_empty()
            || TypeId::of::<F>() != TypeId::of::<crate::Fq>()
        {
            return false;
        }

        let raw_factors = factors
            .iter()
            .copied()
            .map(raw_from_scalar)
            .collect::<Vec<_>>();
        let scale = raw_from_scalar(scale);
        let first = raw_from_scalar(first);
        let second = raw_from_scalar(second);
        let values = raw_values(values);
        midnight_sppark_msm::bls12_381_fr_divide_ntt_zeta(
            values,
            &raw_factors,
            scale,
            first,
            second,
        )
        .expect("sppark BLS12-381 scalar-field quotient fused NTT failed");
        true
    }

    pub fn try_bls12_381_poseidon_hash_chain(
        initial: crate::Fq,
        tail0: crate::Fq,
        tail1: crate::Fq,
        input_len: crate::Fq,
        mds: &[crate::Fq],
        round_constants: &[crate::Fq],
        len: usize,
    ) -> Option<Result<Vec<crate::Fq>, String>> {
        if !enabled() {
            return None;
        }

        let raw_mds = mds.iter().copied().map(raw_from_scalar).collect::<Vec<_>>();
        let raw_round_constants = round_constants
            .iter()
            .copied()
            .map(raw_from_scalar)
            .collect::<Vec<_>>();
        Some(
            midnight_sppark_msm::bls12_381_fr_poseidon_hash_chain(
                raw_from_scalar(initial),
                raw_from_scalar(tail0),
                raw_from_scalar(tail1),
                raw_from_scalar(input_len),
                &raw_mds,
                &raw_round_constants,
                len,
            )
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| crate::Fq(blst::blst_fr { l: value.limbs }))
                    .collect()
            })
            .map_err(|error| error.to_string()),
        )
    }

    pub fn try_bls12_381_poseidon_hash_chain_witness(
        initial: crate::Fq,
        tail0: crate::Fq,
        tail1: crate::Fq,
        input_len: crate::Fq,
        mds: &[crate::Fq],
        round_constants: &[crate::Fq],
        len: usize,
    ) -> Option<Result<Vec<crate::Fq>, String>> {
        if !enabled() {
            return None;
        }

        let raw_mds = mds.iter().copied().map(raw_from_scalar).collect::<Vec<_>>();
        let raw_round_constants = round_constants
            .iter()
            .copied()
            .map(raw_from_scalar)
            .collect::<Vec<_>>();
        Some(
            midnight_sppark_msm::bls12_381_fr_poseidon_hash_chain_witness(
                raw_from_scalar(initial),
                raw_from_scalar(tail0),
                raw_from_scalar(tail1),
                raw_from_scalar(input_len),
                &raw_mds,
                &raw_round_constants,
                len,
            )
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| crate::Fq(blst::blst_fr { l: value.limbs }))
                    .collect()
            })
            .map_err(|error| error.to_string()),
        )
    }

    fn vector_ops_enabled(len: usize) -> bool {
        if !enabled() {
            return false;
        }

        // Keep vector ops default-off for now: current benchmarks show the
        // standalone transfer cost can outweigh the kernel win.
        if !field_backend_enabled("MIDNIGHT_GPU_VECTOR_BACKEND", "off") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_VECTOR_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 18);
        len >= min_len
    }

    fn prefix_ops_enabled(len: usize) -> bool {
        if !enabled() {
            return false;
        }

        if !field_backend_enabled("MIDNIGHT_GPU_PREFIX_BACKEND", "off") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_PREFIX_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 15);
        let max_len = std::env::var("MIDNIGHT_GPU_SPPARK_PREFIX_MAX_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 18);
        len >= min_len && len <= max_len
    }

    fn invert_ops_enabled(len: usize) -> bool {
        if !enabled() {
            return false;
        }

        if !field_backend_enabled("MIDNIGHT_GPU_INVERT_BACKEND", "off") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_INVERT_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 15);
        len >= min_len
    }

    fn coeff_to_extended_enabled(len: usize, count: usize) -> bool {
        if !enabled() || count < 2 {
            return false;
        }

        if !field_backend_enabled("MIDNIGHT_GPU_COSET_BACKEND", "off") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_COSET_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 12);
        len >= min_len
    }

    fn fused_ntt_enabled(len: usize) -> bool {
        if !enabled() {
            return false;
        }

        if !field_backend_enabled("MIDNIGHT_GPU_FUSED_NTT_BACKEND", "off") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_FUSED_NTT_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 15);
        len >= min_len
    }

    fn quotient_backend_enabled(len: usize) -> bool {
        if !enabled() {
            return false;
        }

        if !field_backend_enabled("MIDNIGHT_GPU_QUOTIENT_BACKEND", "auto") {
            return false;
        }

        let min_len = std::env::var("MIDNIGHT_GPU_SPPARK_QUOTIENT_MIN_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1 << 15);
        len >= min_len
    }

    fn field_backend_enabled(env_name: &str, default: &str) -> bool {
        matches!(
            std::env::var(env_name)
                .unwrap_or_else(|_| default.to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "auto"
        )
    }

    #[allow(unsafe_code)]
    fn raw_values<F: Field + 'static>(values: &mut [F]) -> &mut [midnight_sppark_msm::Bls12_381Fr] {
        unsafe { &mut *(values as *mut [F] as *mut [midnight_sppark_msm::Bls12_381Fr]) }
    }

    #[allow(unsafe_code)]
    fn raw_from_scalar<F: Field + 'static>(value: F) -> midnight_sppark_msm::Bls12_381Fr {
        let value = unsafe { *(&value as *const F as *const crate::Fq) };
        midnight_sppark_msm::Bls12_381Fr { limbs: value.0.l }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn shifted_prefix_product_matches_cpu() {
            let old_enabled = set_enabled(true);
            let old_backend = std::env::var("MIDNIGHT_GPU_PREFIX_BACKEND").ok();
            let old_min = std::env::var("MIDNIGHT_GPU_SPPARK_PREFIX_MIN_SIZE").ok();
            std::env::set_var("MIDNIGHT_GPU_PREFIX_BACKEND", "on");
            std::env::set_var("MIDNIGHT_GPU_SPPARK_PREFIX_MIN_SIZE", "0");

            let initial = crate::Fq::from(7);
            let mut values = (0..4099)
                .map(|idx| crate::Fq::from(idx as u64 + 2))
                .collect::<Vec<_>>();

            let mut expected = Vec::with_capacity(values.len());
            expected.push(initial);
            for idx in 1..values.len() {
                expected.push(expected[idx - 1] * values[idx - 1]);
            }

            assert!(try_bls12_381_fr_shifted_prefix_product(
                &mut values,
                initial
            ));
            assert_eq!(values, expected);

            if let Some(old_min) = old_min {
                std::env::set_var("MIDNIGHT_GPU_SPPARK_PREFIX_MIN_SIZE", old_min);
            } else {
                std::env::remove_var("MIDNIGHT_GPU_SPPARK_PREFIX_MIN_SIZE");
            }
            if let Some(old_backend) = old_backend {
                std::env::set_var("MIDNIGHT_GPU_PREFIX_BACKEND", old_backend);
            } else {
                std::env::remove_var("MIDNIGHT_GPU_PREFIX_BACKEND");
            }
            set_enabled(old_enabled);
        }
    }
}

// Production curves (always available)
pub mod bls12_381;
mod jubjub;
pub mod k256;
pub mod p256;

pub mod curve25519;

// Development/testing curves (feature-gated)
#[cfg(any(test, feature = "dev-curves"))]
pub mod bn256;

// Re-exports for production curves
pub use bls12_381::{
    A, B, Bls12, Fp, Fq, G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt,
    MillerLoopResult, PairingG1G2, PairingG2G1, unique_messages,
};
pub use curve::{Coordinates, CurveAffine, CurveExt};
pub use jubjub::*;
// // Re-export pairing library for compatibility with halo2 ecosystem
pub use pairing;

#[cfg(feature = "serde")]
mod serde_impl;

#[cfg(test)]
pub mod tests;

#[cfg(feature = "__private_bench")]
pub use bls12_381::{Fp2, Fp12};
