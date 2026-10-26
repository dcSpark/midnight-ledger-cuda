//! We implement the multi-open technique developed in Halo 2. It is designed to
//! efficiently open multiple polynomials at multiple points while minimizing
//! proof size and verification time. In a nutshell, multiple opening queries
//! are batched into a single query by combining the target
//! polynomials/commitments and evaluation points using verifier-chosen
//! random scalars.
//!
//! For a more detailed explanation, see the [Halo 2 Book](https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html) on Multipoint Openings.

#[cfg(any(
    feature = "gpu-blitzar",
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"),
    feature = "metal-resident-quotient-test-support"
))]
use std::any::TypeId;
use std::marker::PhantomData;

use midnight_curves::pairing::Engine;

#[cfg(feature = "gpu-blitzar")]
pub mod gpu_blitzar;
/// Multiscalar multiplication engines
pub mod msm;
/// KZG commitment scheme
pub mod params;
#[cfg(feature = "cuda-resident-multiopen-test-support")]
mod resident_multiopen_test_support;
mod utils;

#[cfg(feature = "cuda-resident-multiopen-test-support")]
#[doc(hidden)]
pub use resident_multiopen_test_support::assert_resident_multiopen_ownership_error_isolation;

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
use std::sync::{Mutex, OnceLock};
use std::{fmt::Debug, hash::Hash};

use ff::Field;
use group::Group;
use midnight_curves::pairing::MultiMillerLoop;
use rand_core::OsRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
use zeroize::{Zeroize, Zeroizing};

#[cfg(feature = "truncated-challenges")]
use crate::utils::arithmetic::{truncate, truncated_powers};
use crate::{
    poly::{
        Coeff, Error, LagrangeCoeff, Polynomial, ProverQuery,
        commitment::{Params, PolynomialCommitmentScheme},
        kzg::{
            msm::{DualMSM, MSMKZG, msm_specific},
            params::{ParamsKZG, ParamsVerifierKZG},
            utils::construct_intermediate_sets,
        },
        query::{CommitmentLabel, VerifierQuery},
    },
    transcript::{Hashable, Sampleable, Transcript},
    utils::{
        arithmetic::{
            CurveAffine, CurveExt, MSM, eval_polynomial, evals_inner_product, inner_product,
            kate_division_owned, lagrange_interpolate, msm_inner_product, parallelize, powers,
        },
        helpers::ProcessedSerdeObject,
    },
};

#[cfg(feature = "gpu-blitzar")]
use crate::poly::query::Query;

#[cfg(any(
    feature = "gpu-blitzar",
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64")
))]
use self::utils::construct_intermediate_set_metadata;

#[derive(Clone, Debug)]
/// KZG verifier
pub struct KZGCommitmentScheme<E: Engine> {
    _marker: PhantomData<E>,
}

#[cfg(all(
    any(
        feature = "metal-resident-quotient-test-support",
        feature = "metal-resident-quotient-diagnostic"
    ),
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) fn commit_resident_quotient_lease(
    session: &midnight_gpu_prover::metal::MetalSession,
    params: &ParamsKZG<midnight_curves::Bls12>,
    prepared_srs: &midnight_gpu_prover::PreparedResidentSrs,
    lease: &midnight_gpu_prover::metal::ResidentQuotientMsmLease<'_>,
) -> Result<Vec<midnight_curves::G1Projective>, String> {
    let n = lease.piece_len();
    if n != prepared_srs.n() as usize {
        return Err("diagnostic quotient lease and prepared SRS lengths differ".to_owned());
    }
    if lease.prepared_srs_fingerprint() != prepared_srs.fingerprint() {
        return Err("diagnostic quotient lease and prepared SRS fingerprints differ".to_owned());
    }
    let params_srs = params.prepared_resident_srs(prepared_srs.n())?;
    if params_srs.fingerprint() != prepared_srs.fingerprint() {
        return Err("CPU KZG parameters differ from the Metal-bound prepared SRS".to_owned());
    }
    if params.g.len() < n {
        return Err(format!(
            "CPU KZG coefficient SRS has length {}, but quotient MSM requires {n}",
            params.g.len()
        ));
    }

    midnight_gpu_prover::metal::msm_prepared_resident_quotient_batch(
        session,
        prepared_srs,
        midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient,
        lease,
        metal_msm_window_for_nonzero(n, n),
    )
    .map(|points| {
        points
            .into_iter()
            .map(metal_g1_from_raw::<midnight_curves::G1Projective>)
            .collect()
    })
    .map_err(|error| error.to_string())
}

#[cfg(all(
    test,
    any(
        feature = "metal-resident-quotient-test-support",
        feature = "metal-resident-quotient-diagnostic"
    ),
    target_os = "macos",
    target_arch = "aarch64"
))]
#[allow(unsafe_code)]
pub(crate) fn diagnostic_scalar_slice(
    values: &[midnight_gpu_prover::RawFr],
) -> &[midnight_curves::Fq] {
    assert_eq!(
        std::mem::size_of::<midnight_gpu_prover::RawFr>(),
        std::mem::size_of::<midnight_curves::Fq>()
    );
    assert_eq!(
        std::mem::align_of::<midnight_gpu_prover::RawFr>(),
        std::mem::align_of::<midnight_curves::Fq>()
    );
    // SAFETY: `midnight_curves::Fq` is this codebase's BLS12-381 scalar
    // field and is repr(transparent) over the same four little-endian reduced
    // Montgomery limbs as `RawFr`. The callback pins the backing allocation.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<midnight_curves::Fq>(), values.len())
    }
}

#[cfg(any(
    feature = "gpu-blitzar",
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64")
))]
fn is_midnight_bls12_381<E: Engine>() -> bool
where
    E::Fr: Sync + 'static,
    E::G1: 'static,
    E::G1Affine: 'static,
{
    TypeId::of::<E::Fr>() == TypeId::of::<midnight_curves::Fq>()
        && TypeId::of::<E::G1>() == TypeId::of::<midnight_curves::G1Projective>()
        && TypeId::of::<E::G1Affine>() == TypeId::of::<midnight_curves::G1Affine>()
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_msm_selected() -> bool {
    if !midnight_curves::gpu_metal::enabled() {
        return false;
    }
    match std::env::var("MIDNIGHT_METAL_MSM") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "metal" | "auto" => true,
            "0" | "false" | "no" | "off" | "cpu" => false,
            _ => panic!("MIDNIGHT_METAL_MSM must be on, off, metal, or cpu, got {value:?}"),
        },
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MIDNIGHT_METAL_MSM must be valid Unicode")
        }
    }
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
static METAL_HYBRID_CPU_MSM_LOCK: Mutex<()> = Mutex::new(());
#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
static METAL_MSM_CPU_MAX_NONZERO: OnceLock<usize> = OnceLock::new();
#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
// A bounded host partition removes tiny, launch-dominated commitments while
// the Metal worker processes the rest of the same batch. The authentic
// workload's measured crossover is 512 nonzero scalars; the cap below keeps
// larger BLST work from contending with witness synthesis.
const METAL_MSM_DEFAULT_CPU_MAX_NONZERO: usize = 512;
#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
const METAL_MSM_MAX_SAFE_CPU_NONZERO: usize = 8_192;

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_msm_cpu_max_nonzero() -> usize {
    *METAL_MSM_CPU_MAX_NONZERO.get_or_init(|| {
        let value = match std::env::var("MIDNIGHT_METAL_MSM_CPU_MAX_NONZERO") {
            Ok(value) => value.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "MIDNIGHT_METAL_MSM_CPU_MAX_NONZERO must be a non-negative integer, got {value:?}"
                )
            }),
            Err(std::env::VarError::NotPresent) => METAL_MSM_DEFAULT_CPU_MAX_NONZERO,
            Err(error) => panic!("could not read MIDNIGHT_METAL_MSM_CPU_MAX_NONZERO: {error}"),
        };
        assert!(
            value <= METAL_MSM_MAX_SAFE_CPU_NONZERO,
            "MIDNIGHT_METAL_MSM_CPU_MAX_NONZERO must not exceed \
             {METAL_MSM_MAX_SAFE_CPU_NONZERO}; larger concurrent BLST MSMs can stall its shared \
             worker pool"
        );
        value
    })
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_msm_nonzero_count<F: Field>(scalars: &[F]) -> usize {
    scalars
        .iter()
        .filter(|scalar| !scalar.is_zero_vartime())
        .count()
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn metal_cpu_prepared_msm(
    prepared: &midnight_gpu_prover::PreparedResidentSrs,
    basis: midnight_gpu_prover::metal::Bls12_381MsmBasis,
    scalars: &[midnight_gpu_prover::RawFr],
) -> Result<midnight_gpu_prover::metal::Bls12_381G1Projective, String> {
    use blst::MultiPoint;

    let prepared_bases = match basis {
        midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient => prepared.coefficient_bases(),
        midnight_gpu_prover::metal::Bls12_381MsmBasis::Lagrange => prepared.lagrange_bases(),
    };
    if scalars.len() > prepared_bases.len() {
        return Err(format!(
            "CPU affine MSM has {} scalars but only {} prepared bases",
            scalars.len(),
            prepared_bases.len()
        ));
    }

    let nonzero = scalars
        .iter()
        .filter(|scalar| scalar.limbs != [0; 4])
        .count();
    let mut bases = Vec::<blst::blst_p1_affine>::with_capacity(nonzero);
    let mut scalar_bytes = Zeroizing::new(Vec::<u8>::with_capacity(nonzero.saturating_mul(32)));
    for (scalar, base) in scalars.iter().zip(prepared_bases) {
        if scalar.limbs == [0; 4] {
            continue;
        }
        bases.push(blst::blst_p1_affine {
            x: blst::blst_fp { l: base.x },
            y: blst::blst_fp { l: base.y },
        });
        let mut raw = blst::blst_fr { l: scalar.limbs };
        let mut canonical = [0_u64; 4];
        // SAFETY: `raw` is this codebase's reduced Montgomery Fr ABI and the
        // destination has exactly the four limbs required by BLST.
        unsafe { blst::blst_uint64_from_fr(canonical.as_mut_ptr(), &raw) };
        for limb in canonical {
            scalar_bytes.extend_from_slice(&limb.to_le_bytes());
        }
        raw.l.zeroize();
        canonical.zeroize();
    }

    let point = if bases.is_empty() {
        blst::blst_p1::default()
    } else {
        bases.as_slice().mult(scalar_bytes.as_slice(), 255)
    };
    Ok(midnight_gpu_prover::metal::Bls12_381G1Projective {
        x: point.x.l,
        y: point.y.l,
        z: point.z.l,
    })
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn try_metal_commit<E: Engine>(
    params: &ParamsKZG<E>,
    scalars: &[E::Fr],
    basis: midnight_gpu_prover::metal::Bls12_381MsmBasis,
) -> Option<Result<E::G1, String>>
where
    E::Fr: 'static,
    E::G1: Copy + 'static,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + 'static,
{
    if !metal_msm_selected() || !is_midnight_bls12_381::<E>() || scalars.is_empty() {
        return None;
    }
    Some((|| {
        let nonzero_scalars = metal_msm_nonzero_count(scalars);
        // SAFETY: the exact associated scalar, projective, and affine TypeIds
        // above identify this repository's sole BLS12-381 specialization.
        let params = unsafe {
            &*(params as *const ParamsKZG<E> as *const ParamsKZG<midnight_curves::Bls12>)
        };
        let n = u32::try_from(params.g.len())
            .map_err(|_| "Metal MSM SRS prefix exceeds u32".to_owned())?;
        let prepared = params.prepared_resident_srs(n)?;
        if nonzero_scalars <= metal_msm_cpu_max_nonzero() {
            let lock_wait_started = std::time::Instant::now();
            let _cpu_msm = METAL_HYBRID_CPU_MSM_LOCK
                .lock()
                .map_err(|_| "hybrid CPU MSM lock was poisoned".to_owned())?;
            let lock_wait_ns = metal_duration_ns(lock_wait_started.elapsed());
            let service_started = std::time::Instant::now();
            let output = metal_cpu_prepared_msm(&prepared, basis, metal_raw_scalars(scalars))?;
            let service_ns = metal_duration_ns(service_started.elapsed());
            midnight_gpu_prover::metal::record_cpu_msm_partition(scalars.len(), nonzero_scalars);
            midnight_gpu_prover::metal::record_cpu_msm_timing(lock_wait_ns, service_ns);
            return Ok(metal_g1_from_raw::<E::G1>(output));
        }
        let session = midnight_gpu_prover::metal::diagnostic_shared_session()
            .map_err(|error| error.to_string())?;
        let raw = metal_raw_scalars(scalars);
        let point = midnight_gpu_prover::metal::msm_prepared(
            session,
            &prepared,
            basis,
            raw,
            metal_msm_window_for_nonzero(scalars.len(), nonzero_scalars),
        )
        .map_err(|error| error.to_string())?;
        Ok(metal_g1_from_raw(point))
    })())
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn try_metal_commit_batch<E: Engine, B>(
    params: &ParamsKZG<E>,
    polynomials: &[Polynomial<E::Fr, B>],
    basis: midnight_gpu_prover::metal::Bls12_381MsmBasis,
) -> Option<Result<Vec<E::G1>, String>>
where
    E::Fr: Sync + 'static,
    E::G1: Copy + 'static,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + 'static,
{
    if !metal_msm_selected()
        || !is_midnight_bls12_381::<E>()
        || polynomials.is_empty()
        || polynomials
            .iter()
            .any(|polynomial| polynomial.values.is_empty())
    {
        return None;
    }
    Some((|| {
        let nonzero_counts = polynomials
            .iter()
            .map(|polynomial| metal_msm_nonzero_count(&polynomial.values))
            .collect::<Vec<_>>();
        let cpu_max_nonzero = metal_msm_cpu_max_nonzero();
        let device_eligible = nonzero_counts
            .iter()
            .map(|&nonzero| nonzero > cpu_max_nonzero)
            .collect::<Vec<_>>();
        let mut outputs = vec![None; polynomials.len()];
        // SAFETY: see `try_metal_commit`; the same exact TypeId gate applies.
        let metal_params = unsafe {
            &*(params as *const ParamsKZG<E> as *const ParamsKZG<midnight_curves::Bls12>)
        };
        let n = u32::try_from(metal_params.g.len())
            .map_err(|_| "Metal MSM SRS prefix exceeds u32".to_owned())?;
        let prepared = metal_params.prepared_resident_srs(n)?;

        let mut device_groups = Vec::<(
            midnight_gpu_prover::metal::Bls12_381MsmWindow,
            Vec<(usize, &[midnight_gpu_prover::RawFr])>,
        )>::new();
        for (index, &eligible) in device_eligible.iter().enumerate() {
            if !eligible {
                continue;
            }
            let window = metal_msm_window_for_nonzero(
                polynomials[index].values.len(),
                nonzero_counts[index],
            );
            let raw = metal_raw_scalars(polynomials[index].values.as_slice());
            if let Some((_, batches)) = device_groups
                .iter_mut()
                .find(|(candidate, _)| *candidate == window)
            {
                batches.push((index, raw));
            } else {
                device_groups.push((window, vec![(index, raw)]));
            }
        }

        let run_cpu = || -> Result<Vec<(usize, E::G1)>, String> {
            if device_eligible.iter().all(|&eligible| eligible) {
                return Ok(Vec::new());
            }
            let mut cpu_outputs = Vec::new();
            let lock_wait_started = std::time::Instant::now();
            let _cpu_msm = METAL_HYBRID_CPU_MSM_LOCK
                .lock()
                .map_err(|_| "hybrid CPU MSM lock was poisoned".to_owned())?;
            let lock_wait_ns = metal_duration_ns(lock_wait_started.elapsed());
            let service_started = std::time::Instant::now();
            for (index, (polynomial, &eligible)) in
                polynomials.iter().zip(&device_eligible).enumerate()
            {
                if !eligible {
                    let point = metal_g1_from_raw::<E::G1>(metal_cpu_prepared_msm(
                        &prepared,
                        basis,
                        metal_raw_scalars(&polynomial.values),
                    )?);
                    cpu_outputs.push((index, point));
                    midnight_gpu_prover::metal::record_cpu_msm_partition(
                        polynomial.values.len(),
                        nonzero_counts[index],
                    );
                }
            }
            let service_ns = metal_duration_ns(service_started.elapsed());
            midnight_gpu_prover::metal::record_cpu_msm_timing(lock_wait_ns, service_ns);
            Ok(cpu_outputs)
        };

        let session = midnight_gpu_prover::metal::diagnostic_shared_session()
            .map_err(|error| error.to_string())?;
        let run_metal = || -> Result<_, String> {
            let mut completed = Vec::new();
            for (window, batches) in &device_groups {
                let raw = batches
                    .iter()
                    .map(|(_, scalars)| *scalars)
                    .collect::<Vec<_>>();
                let metal_outputs = midnight_gpu_prover::metal::msm_prepared_batch(
                    session, &prepared, basis, &raw, *window,
                )
                .map_err(|error| error.to_string())?;
                if metal_outputs.len() != batches.len() {
                    return Err("hybrid Metal MSM returned the wrong output count".to_owned());
                }
                completed.extend(batches.iter().map(|(index, _)| *index).zip(metal_outputs));
            }
            Ok(completed)
        };

        let has_cpu = device_eligible.iter().any(|&eligible| !eligible);
        let has_metal = !device_groups.is_empty();
        let (cpu_outputs, metal_outputs) = match (has_cpu, has_metal) {
            (true, true) => std::thread::scope(|scope| {
                let metal = scope.spawn(run_metal);
                let cpu = run_cpu();
                let metal = metal
                    .join()
                    .map_err(|_| "hybrid Metal MSM dispatch thread panicked".to_owned())?;
                Ok::<_, String>((cpu?, metal?))
            })?,
            (true, false) => (run_cpu()?, Vec::new()),
            (false, true) => (Vec::new(), run_metal()?),
            (false, false) => return Err("hybrid MSM batch has no outputs".to_owned()),
        };
        for (index, point) in cpu_outputs {
            outputs[index] = Some(point);
        }
        for (index, point) in metal_outputs {
            outputs[index] = Some(metal_g1_from_raw(point));
        }
        if outputs.iter().any(Option::is_none) {
            return Err("hybrid MSM output is missing".to_owned());
        }
        Ok(outputs
            .into_iter()
            .map(|output| {
                output.expect("the complete hybrid output vector was checked immediately above")
            })
            .collect())
    })())
}

#[cfg(any(
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"),
    feature = "metal-resident-quotient-test-support"
))]
fn metal_msm_window_for_nonzero(
    term_count: usize,
    nonzero_count: usize,
) -> midnight_gpu_prover::metal::Bls12_381MsmWindow {
    match std::env::var("MIDNIGHT_METAL_MSM_WINDOW").as_deref() {
        Ok("6") => midnight_gpu_prover::metal::Bls12_381MsmWindow::W6,
        Ok("7") => midnight_gpu_prover::metal::Bls12_381MsmWindow::W7,
        Ok("8") => midnight_gpu_prover::metal::Bls12_381MsmWindow::W8,
        Ok("9") => midnight_gpu_prover::metal::Bls12_381MsmWindow::W9,
        Ok("10") => midnight_gpu_prover::metal::Bls12_381MsmWindow::W10,
        Ok("fixed11") | Ok("fixed-w11") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW11,
        Ok("fixed12") | Ok("fixed-w12") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW12,
        Ok("fixed13") | Ok("fixed-w13") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW13,
        Ok("fixed14") | Ok("fixed-w14") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW14,
        Ok("fixed15") | Ok("fixed-w15") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW15,
        Ok("fixed16") | Ok("fixed-w16") => midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW16,
        Err(std::env::VarError::NotPresent) => {
            default_metal_msm_window_for_nonzero(term_count, nonzero_count)
        }
        Ok(value) => {
            panic!(
                "MIDNIGHT_METAL_MSM_WINDOW must be one of 6, 7, 8, 9, 10, fixed11, fixed12, fixed13, fixed14, fixed15, or fixed16, got {value:?}"
            )
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MIDNIGHT_METAL_MSM_WINDOW must be valid Unicode")
        }
    }
}

#[cfg(any(
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"),
    feature = "metal-resident-quotient-test-support"
))]
fn default_metal_msm_window_for_nonzero(
    term_count: usize,
    nonzero_count: usize,
) -> midnight_gpu_prover::metal::Bls12_381MsmWindow {
    if nonzero_count.saturating_mul(8) <= term_count.saturating_mul(5) {
        midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW11
    } else {
        midnight_gpu_prover::metal::Bls12_381MsmWindow::FixedW13
    }
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn metal_raw_scalars<F: 'static>(scalars: &[F]) -> &[midnight_gpu_prover::RawFr] {
    assert_eq!(TypeId::of::<F>(), TypeId::of::<midnight_curves::Fq>());
    assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<midnight_gpu_prover::RawFr>()
    );
    assert_eq!(
        std::mem::align_of::<F>(),
        std::mem::align_of::<midnight_gpu_prover::RawFr>()
    );
    // SAFETY: the TypeId and layout checks bind `F` to the transparent Fq
    // wrapper whose raw value is the same reduced four-limb Montgomery ABI.
    unsafe {
        std::slice::from_raw_parts(
            scalars.as_ptr().cast::<midnight_gpu_prover::RawFr>(),
            scalars.len(),
        )
    }
}

#[cfg(any(
    all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"),
    feature = "metal-resident-quotient-test-support"
))]
#[allow(unsafe_code)]
fn metal_g1_from_raw<C: Copy + 'static>(
    raw: midnight_gpu_prover::metal::Bls12_381G1Projective,
) -> C {
    assert_eq!(
        TypeId::of::<C>(),
        TypeId::of::<midnight_curves::G1Projective>()
    );
    let mut point = midnight_curves::G1Projective::default();
    *point.as_mut() = blst::blst_p1 {
        x: blst::blst_fp { l: raw.x },
        y: blst::blst_fp { l: raw.y },
        z: blst::blst_fp { l: raw.z },
    };
    // SAFETY: `C` is exactly the checked transparent G1Projective type.
    unsafe { std::mem::transmute_copy(&point) }
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_resident_multiopen_enabled(poly_len: usize) -> bool {
    if !metal_msm_selected() {
        return false;
    }
    let enabled = matches!(
        std::env::var("MIDNIGHT_METAL_RESIDENT_MULTIOPEN")
            .unwrap_or_else(|_| "on".to_owned())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on" | "metal" | "auto"
    );
    let min_len = std::env::var("MIDNIGHT_METAL_RESIDENT_MULTIOPEN_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 14);
    enabled && poly_len >= min_len
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_raw_scalar<F: Field + 'static>(value: F) -> midnight_gpu_prover::RawFr {
    metal_raw_scalars(std::slice::from_ref(&value))[0]
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn metal_field_from_raw<F: Field + 'static>(value: midnight_gpu_prover::RawFr) -> F {
    assert_eq!(TypeId::of::<F>(), TypeId::of::<midnight_curves::Fq>());
    assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<midnight_gpu_prover::RawFr>()
    );
    // SAFETY: the exact TypeId gate binds `F` to the transparent Fq wrapper,
    // and both values are the same reduced four-limb Montgomery representation.
    unsafe { std::mem::transmute_copy(&value) }
}

#[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
fn try_metal_resident_multi_open<E: MultiMillerLoop, T: Transcript>(
    params: &ParamsKZG<E>,
    prover_query: &[ProverQuery<E::Fr>],
    transcript: &mut T,
    x1: E::Fr,
    x2: E::Fr,
) -> Option<Result<(), Error>>
where
    E::Fr: Sync + 'static + Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
    E::G1: Default
        + Copy
        + CurveExt<ScalarExt = E::Fr>
        + ProcessedSerdeObject
        + Hashable<T::Hash>
        + 'static,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + 'static,
    ParamsKZG<E>: Sync,
{
    let poly_len = prover_query.first().map(|query| query.poly.len())?;
    if !is_midnight_bls12_381::<E>() || !metal_resident_multiopen_enabled(poly_len) {
        return None;
    }

    Some((|| {
        let (poly_map, point_sets, _) = construct_intermediate_set_metadata(prover_query)?;
        if poly_map.is_empty()
            || point_sets.is_empty()
            || poly_map
                .iter()
                .any(|data| data.commitment.poly.len() != poly_len)
        {
            return Err(Error::OpeningError);
        }

        let mut order = (0..point_sets.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| (point_sets[index].len(), index));
        let q_polynomials = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.q_polys.cpu_for_resident_metal",
            poly_len.saturating_mul(poly_map.len()),
            || {
                order
                    .iter()
                    .map(|&set_index| {
                        let sources = poly_map
                            .iter()
                            .filter_map(|data| {
                                (data.set_index == set_index).then_some(data.commitment.poly)
                            })
                            .collect::<Vec<_>>();
                        #[cfg(feature = "truncated-challenges")]
                        let coefficients =
                            truncated_powers(x1).take(sources.len()).collect::<Vec<_>>();
                        #[cfg(not(feature = "truncated-challenges"))]
                        let coefficients = powers(x1).take(sources.len()).collect::<Vec<_>>();
                        polynomial_inner_product_refs(&sources, &coefficients)
                    })
                    .collect::<Vec<_>>()
            },
        );
        let polynomials = q_polynomials
            .iter()
            .map(|polynomial| metal_raw_scalars(polynomial.values.as_slice()))
            .collect::<Vec<_>>();
        let mut q_offsets = Vec::with_capacity(point_sets.len() + 1);
        let mut q_poly_indices = Vec::with_capacity(point_sets.len());
        let mut q_coefficients = Vec::with_capacity(point_sets.len());
        let mut point_offsets = Vec::with_capacity(point_sets.len() + 1);
        let mut points = Vec::new();
        q_offsets.push(0);
        point_offsets.push(0);
        for (q_index, &set_index) in order.iter().enumerate() {
            q_poly_indices.push(u32::try_from(q_index).map_err(|_| Error::OpeningError)?);
            q_coefficients.push(metal_raw_scalar(E::Fr::ONE));
            q_offsets.push(u32::try_from(q_poly_indices.len()).map_err(|_| Error::OpeningError)?);
            points.extend(point_sets[set_index].iter().copied().map(metal_raw_scalar));
            point_offsets.push(u32::try_from(points.len()).map_err(|_| Error::OpeningError)?);
        }
        let f_weights = powers(x2)
            .take(point_sets.len())
            .map(metal_raw_scalar)
            .collect::<Vec<_>>();

        // SAFETY: the exact scalar/projective/affine TypeId gate above binds
        // this generic instantiation to this repository's BLS12-381 KZG types.
        let metal_params = unsafe {
            &*(params as *const ParamsKZG<E> as *const ParamsKZG<midnight_curves::Bls12>)
        };
        let prepared = metal_params
            .prepared_resident_srs(u32::try_from(poly_len).map_err(|_| Error::OpeningError)?)
            .map_err(|_| Error::OpeningError)?;
        let session = midnight_gpu_prover::metal::diagnostic_shared_session()
            .map_err(|_| Error::OpeningError)?;
        let request = midnight_gpu_prover::metal::ResidentMultiopenRequest::new(
            &polynomials,
            &q_offsets,
            &q_poly_indices,
            &q_coefficients,
            &point_offsets,
            &points,
            &f_weights,
        );
        let window = metal_msm_window_for_nonzero(poly_len, poly_len);
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.resident_metal",
            poly_len.saturating_mul(polynomials.len()),
            || {
                midnight_gpu_prover::metal::with_resident_multiopen(session, request, |arena| {
                    let f_com = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.f_commit.resident_metal",
                        poly_len,
                        || {
                            let f = arena.f_polynomial()?;
                            midnight_gpu_prover::metal::msm_prepared_resident_multiopen(
                                session,
                                &prepared,
                                midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient,
                                &f,
                                window,
                            )
                        },
                    )?;
                    let f_com = metal_g1_from_raw::<E::G1>(f_com);
                    transcript.write(&f_com).map_err(|_| {
                        midnight_gpu_prover::ResidentError::Backend(
                            "failed to write resident Metal f commitment".to_owned(),
                        )
                    })?;

                    let x3: E::Fr = transcript.squeeze_challenge();
                    #[cfg(feature = "truncated-challenges")]
                    let x3 = truncate(x3);
                    let q_evaluations = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.q_eval.resident_metal",
                        poly_len.saturating_mul(arena.q_count()),
                        || arena.evaluate_q(metal_raw_scalar(x3)),
                    )?;
                    for evaluation in q_evaluations {
                        transcript
                            .write(&metal_field_from_raw::<E::Fr>(evaluation))
                            .map_err(|_| {
                                midnight_gpu_prover::ResidentError::Backend(
                                    "failed to write a resident Metal q evaluation".to_owned(),
                                )
                            })?;
                    }

                    let x4: E::Fr = transcript.squeeze_challenge();
                    #[cfg(feature = "truncated-challenges")]
                    let pi_weights = truncated_powers(x4)
                        .take(arena.q_count() + 1)
                        .map(metal_raw_scalar)
                        .collect::<Vec<_>>();
                    #[cfg(not(feature = "truncated-challenges"))]
                    let pi_weights = powers(x4)
                        .take(arena.q_count() + 1)
                        .map(metal_raw_scalar)
                        .collect::<Vec<_>>();
                    let pi_polynomial = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.pi_poly.resident_metal",
                        poly_len.saturating_mul(arena.q_count() + 1),
                        || arena.compute_pi(metal_raw_scalar(x3), &pi_weights),
                    )?;
                    let pi = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.pi_commit.resident_metal",
                        poly_len,
                        || {
                            midnight_gpu_prover::metal::msm_prepared_resident_multiopen(
                                session,
                                &prepared,
                                midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient,
                                &pi_polynomial,
                                window,
                            )
                        },
                    )?;
                    transcript
                        .write(&metal_g1_from_raw::<E::G1>(pi))
                        .map_err(|_| {
                            midnight_gpu_prover::ResidentError::Backend(
                                "failed to write resident Metal pi commitment".to_owned(),
                            )
                        })
                })
            },
        )
        .map_err(|_| Error::OpeningError)
    })())
}

fn polynomial_inner_product<F: Field + 'static>(
    polys: &[Polynomial<F, Coeff>],
    scalars: impl IntoIterator<Item = F>,
) -> Polynomial<F, Coeff> {
    let scalars = scalars.into_iter().take(polys.len()).collect::<Vec<_>>();

    #[cfg(feature = "gpu-blitzar")]
    if let Some(poly) = try_polynomial_inner_product_sppark(polys, &scalars) {
        return poly;
    }

    let polys = polys.iter().collect::<Vec<_>>();
    polynomial_inner_product_refs(&polys, &scalars)
}

fn polynomial_inner_product_refs<F: Field>(
    polys: &[&Polynomial<F, Coeff>],
    scalars: &[F],
) -> Polynomial<F, Coeff> {
    assert_eq!(polys.len(), scalars.len());
    let first = polys.first().expect("polynomial inner product is empty");
    let first_scalar = scalars[0];
    let mut acc = (*first).clone();
    let terms = polys
        .iter()
        .skip(1)
        .zip(&scalars[1..])
        .filter_map(|(&poly, &scalar)| (scalar != F::ZERO).then_some((poly, scalar)))
        .collect::<Vec<_>>();

    // Keep each output coefficient in a register while applying every source.
    // Besides avoiding one read/write pass over the accumulator per polynomial,
    // this creates one Rayon scope instead of one scope per source. That matters
    // when several proofs build their multi-open polynomials concurrently.
    parallelize(&mut acc.values, |values, start| {
        for (offset, value) in values.iter_mut().enumerate() {
            if first_scalar == F::ZERO {
                *value = F::ZERO;
            } else if first_scalar != F::ONE {
                *value *= first_scalar;
            }
            let coefficient = start + offset;
            for (poly, scalar) in &terms {
                if *scalar == F::ONE {
                    *value += poly.values[coefficient];
                } else {
                    *value += poly.values[coefficient] * *scalar;
                }
            }
        }
    });

    acc
}

#[cfg(feature = "gpu-blitzar")]
fn try_polynomial_inner_product_sppark<F: Field + 'static>(
    polys: &[Polynomial<F, Coeff>],
    scalars: &[F],
) -> Option<Polynomial<F, Coeff>> {
    if polys.is_empty()
        || polys.len() != scalars.len()
        || !multi_open_poly_backend_enabled(polys[0].values.len(), polys.len())
        || TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>()
    {
        return None;
    }

    let poly_len = polys[0].values.len();
    if poly_len == 0 || !polys.iter().all(|poly| poly.values.len() == poly_len) {
        return None;
    }

    let _timer = crate::perf_timer::ScopedTimer::with_count(
        "proof_server.prover.kzg.multi_open.poly_lincomb.gpu_sppark",
        Some(poly_len * polys.len()),
    );

    let mut flat = Vec::with_capacity(poly_len * polys.len());
    for poly in polys {
        flat.extend(poly.values.iter().copied().map(raw_scalar));
    }
    let scalars = scalars.iter().copied().map(raw_scalar).collect::<Vec<_>>();
    let mut out = vec![midnight_sppark_msm::Bls12_381Fr::default(); poly_len];

    if midnight_sppark_msm::bls12_381_fr_poly_lincomb(
        &mut out,
        &flat,
        poly_len,
        polys.len(),
        &scalars,
    )
    .is_err()
    {
        midnight_sppark_msm::record_accelerator_fallback();
        return None;
    }

    Some(Polynomial {
        values: out.into_iter().map(field_from_raw).collect(),
        _marker: PhantomData,
    })
}

#[cfg(feature = "gpu-blitzar")]
fn multi_open_poly_backend_enabled(poly_len: usize, poly_count: usize) -> bool {
    if !midnight_curves::gpu_sppark::enabled() {
        return false;
    }
    let backend_enabled = matches!(
        std::env::var("MIDNIGHT_GPU_MULTI_OPEN_POLY_BACKEND")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "auto"
    );
    if !backend_enabled {
        return false;
    }

    let min_total = std::env::var("MIDNIGHT_GPU_MULTI_OPEN_POLY_MIN_TOTAL")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 20);
    poly_len * poly_count >= min_total
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn raw_scalar<F: Field + 'static>(value: F) -> midnight_sppark_msm::Bls12_381Fr {
    debug_assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<midnight_sppark_msm::Bls12_381Fr>()
    );
    unsafe { std::mem::transmute_copy(&value) }
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn raw_scalar_slice<F: Field + 'static>(values: &[F]) -> &[midnight_sppark_msm::Bls12_381Fr] {
    debug_assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<midnight_sppark_msm::Bls12_381Fr>()
    );
    debug_assert_eq!(
        std::mem::align_of::<F>(),
        std::mem::align_of::<midnight_sppark_msm::Bls12_381Fr>()
    );
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn field_from_raw<F: Field + 'static>(value: midnight_sppark_msm::Bls12_381Fr) -> F {
    debug_assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<midnight_sppark_msm::Bls12_381Fr>()
    );
    unsafe { std::mem::transmute_copy(&value) }
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_multiopen_prepare_enabled(poly_len: usize) -> bool {
    if !gpu_blitzar::enabled() {
        return false;
    }
    let enabled = matches!(
        std::env::var("MIDNIGHT_GPU_MULTIOPEN_PREPARE")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "auto"
    );
    let min_len = std::env::var("MIDNIGHT_GPU_MULTIOPEN_PREPARE_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 15);
    enabled && poly_len >= min_len
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_resident_multiopen_enabled(poly_len: usize) -> bool {
    crate::plonk::resident_multiopen_policy_enabled(poly_len)
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_resident_multiopen_pipeline_enabled() -> bool {
    matches!(
        std::env::var("MIDNIGHT_GPU_RESIDENT_MULTIOPEN_PIPELINE")
            .unwrap_or_else(|_| "on".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "pipeline" | "debug"
    )
}

#[cfg(feature = "gpu-blitzar")]
fn gpu_resident_multiopen_pipeline_debug() -> bool {
    std::env::var("MIDNIGHT_GPU_RESIDENT_MULTIOPEN_PIPELINE")
        .is_ok_and(|value| value.eq_ignore_ascii_case("debug"))
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn midnight_g1_bases<C>(bases: &[C]) -> &[midnight_curves::G1Projective] {
    unsafe { &*(bases as *const [C] as *const [midnight_curves::G1Projective]) }
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn midnight_fq_scalars<F>(scalars: &[F]) -> &[midnight_curves::Fq] {
    unsafe { &*(scalars as *const [F] as *const [midnight_curves::Fq]) }
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn midnight_g1_from_gpu<C: Copy>(point: midnight_curves::G1Projective) -> C {
    unsafe { std::mem::transmute_copy(&point) }
}

#[cfg(feature = "gpu-blitzar")]
fn nonzero_scalar_count<F: Field, B>(polynomials: &[Polynomial<F, B>]) -> usize {
    polynomials
        .iter()
        .map(|polynomial| {
            polynomial
                .values
                .iter()
                .filter(|scalar| !scalar.is_zero_vartime())
                .count()
        })
        .sum()
}

impl<E: MultiMillerLoop> PolynomialCommitmentScheme<E::Fr> for KZGCommitmentScheme<E>
where
    E::Fr: Sync + 'static,
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject + 'static,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + 'static,
    ParamsKZG<E>: Sync,
{
    type Parameters = ParamsKZG<E>;
    type VerifierParameters = ParamsVerifierKZG<E>;
    type Commitment = E::G1;
    type VerificationGuard = DualMSM<E>;

    fn gen_params(k: u32) -> Self::Parameters {
        ParamsKZG::unsafe_setup(k, OsRng)
    }

    fn get_verifier_params(params: &Self::Parameters) -> Self::VerifierParameters {
        params.verifier_params()
    }

    fn commit(
        params: &Self::Parameters,
        polynomial: &Polynomial<E::Fr, Coeff>,
    ) -> Self::Commitment {
        let size = polynomial.values.len();
        assert!(params.g.len() >= size);
        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        if metal_msm_selected() && is_midnight_bls12_381::<E>() && !polynomial.values.is_empty() {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.coeff_msm.metal",
                size,
                || {
                    try_metal_commit(
                        params,
                        &polynomial.values,
                        midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient,
                    )
                    .expect("selected Metal coefficient MSM shape became ineligible")
                    .expect("selected Metal coefficient MSM failed")
                },
            );
        }
        #[cfg(feature = "gpu-blitzar")]
        if gpu_blitzar::should_accelerate(size, 1) {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.coeff_msm.gpu_blitzar",
                size,
                || {
                    gpu_blitzar::commit::<E::Fr, E::G1>(&polynomial.values, &params.g)
                        .expect("Blitzar KZG coefficient MSM failed")
                },
            );
        }
        crate::perf_timer::timed_count("proof_server.prover.kzg.commit.coeff_msm", size, || {
            msm_specific::<E::G1Affine>(&polynomial.values, &params.g[..size])
        })
    }

    fn commit_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<E::Fr, Coeff>],
    ) -> Vec<Self::Commitment> {
        if polynomials.is_empty() {
            return Vec::new();
        }

        let size = polynomials
            .iter()
            .map(|polynomial| polynomial.values.len())
            .max()
            .unwrap_or(0);
        assert!(params.g.len() >= size);

        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        if metal_msm_selected()
            && is_midnight_bls12_381::<E>()
            && polynomials
                .iter()
                .all(|polynomial| !polynomial.values.is_empty())
        {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.coeff_msm_batch.metal",
                polynomials
                    .iter()
                    .map(|polynomial| polynomial.values.len())
                    .sum(),
                || {
                    try_metal_commit_batch(
                        params,
                        polynomials,
                        midnight_gpu_prover::metal::Bls12_381MsmBasis::Coefficient,
                    )
                    .expect("selected Metal coefficient MSM batch shape became ineligible")
                    .expect("selected Metal coefficient MSM batch failed")
                },
            );
        }

        #[cfg(feature = "gpu-blitzar")]
        {
            let total_scalars = polynomials
                .iter()
                .map(|polynomial| polynomial.values.len())
                .sum();
            let same_size = polynomials
                .iter()
                .all(|polynomial| polynomial.values.len() == size);
            let nonzero_scalars = same_size.then(|| nonzero_scalar_count(polynomials));
            if same_size
                && nonzero_scalars.is_some_and(|nonzero_scalars| {
                    gpu_blitzar::should_accelerate_sppark_batch(
                        size,
                        total_scalars,
                        nonzero_scalars,
                        polynomials.len(),
                    )
                })
                && is_midnight_bls12_381::<E>()
            {
                let bases = midnight_g1_bases(&params.g[..size]);
                let scalar_batches = polynomials
                    .iter()
                    .map(|polynomial| midnight_fq_scalars(polynomial.values.as_slice()))
                    .collect::<Vec<_>>();
                let commitments = crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.commit.coeff_msm_batch.gpu_sppark",
                    total_scalars,
                    || {
                        midnight_curves::G1Projective::multi_exp_sppark_batch(
                            bases,
                            &scalar_batches,
                        )
                        .expect("sppark BLS12-381 coefficient MSM batch failed")
                    },
                );
                return commitments.into_iter().map(midnight_g1_from_gpu).collect();
            }
            if same_size
                && polynomials.len() < 4
                && gpu_blitzar::should_accelerate_sppark(size)
                && is_midnight_bls12_381::<E>()
                && gpu_blitzar::use_cpu_for_sppark_small_batch()
            {
                if midnight_sppark_msm::current_execution_lane() != 0 {
                    return polynomials
                        .par_iter()
                        .map(|polynomial| {
                            let _gpu_guard = gpu_blitzar::DisableGuard::new();
                            Self::commit(params, polynomial)
                        })
                        .collect();
                }
                let _gpu_guard = gpu_blitzar::EnableGuard::set(false);
                return polynomials
                    .par_iter()
                    .map(|polynomial| Self::commit(params, polynomial))
                    .collect();
            }
            if gpu_blitzar::should_accelerate(total_scalars, polynomials.len()) {
                let scalar_slices = polynomials
                    .iter()
                    .map(|polynomial| polynomial.values.as_slice())
                    .collect::<Vec<_>>();
                return crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.commit.coeff_msm_batch.gpu_blitzar",
                    total_scalars,
                    || {
                        gpu_blitzar::commit_batch::<E::Fr, E::G1>(&scalar_slices, &params.g)
                            .expect("Blitzar KZG coefficient MSM batch failed")
                    },
                );
            }
        }

        #[cfg(feature = "gpu-blitzar")]
        {
            let lane = midnight_sppark_msm::current_execution_lane();
            if lane != 0 {
                return polynomials
                    .par_iter()
                    .map(|polynomial| {
                        let _lane = midnight_sppark_msm::bind_execution_lane(lane)
                            .expect("failed to propagate sppark proof execution lane");
                        Self::commit(params, polynomial)
                    })
                    .collect();
            }
        }
        polynomials
            .par_iter()
            .map(|polynomial| Self::commit(params, polynomial))
            .collect()
    }

    fn commit_lagrange(
        params: &Self::Parameters,
        poly: &Polynomial<E::Fr, LagrangeCoeff>,
    ) -> E::G1 {
        let size = poly.values.len();

        assert!(params.g_lagrange.len() >= size);

        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        if metal_msm_selected() && is_midnight_bls12_381::<E>() && !poly.values.is_empty() {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.lagrange_msm.metal",
                size,
                || {
                    try_metal_commit(
                        params,
                        &poly.values,
                        midnight_gpu_prover::metal::Bls12_381MsmBasis::Lagrange,
                    )
                    .expect("selected Metal Lagrange MSM shape became ineligible")
                    .expect("selected Metal Lagrange MSM failed")
                },
            );
        }

        #[cfg(feature = "gpu-blitzar")]
        if gpu_blitzar::should_accelerate(size, 1) {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.lagrange_msm.gpu_blitzar",
                size,
                || {
                    gpu_blitzar::commit::<E::Fr, E::G1>(&poly.values, &params.g_lagrange)
                        .expect("Blitzar KZG lagrange MSM failed")
                },
            );
        }

        crate::perf_timer::timed_count("proof_server.prover.kzg.commit.lagrange_msm", size, || {
            msm_specific::<E::G1Affine>(&poly.values, &params.g_lagrange[0..size])
        })
    }

    fn commit_lagrange_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<E::Fr, LagrangeCoeff>],
    ) -> Vec<E::G1> {
        if polynomials.is_empty() {
            return Vec::new();
        }

        let size = polynomials
            .iter()
            .map(|polynomial| polynomial.values.len())
            .max()
            .unwrap_or(0);
        assert!(params.g_lagrange.len() >= size);

        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        if metal_msm_selected()
            && is_midnight_bls12_381::<E>()
            && polynomials
                .iter()
                .all(|polynomial| !polynomial.values.is_empty())
        {
            return crate::perf_timer::timed_count(
                "proof_server.prover.kzg.commit.lagrange_msm_batch.metal",
                polynomials
                    .iter()
                    .map(|polynomial| polynomial.values.len())
                    .sum(),
                || {
                    try_metal_commit_batch(
                        params,
                        polynomials,
                        midnight_gpu_prover::metal::Bls12_381MsmBasis::Lagrange,
                    )
                    .expect("selected Metal Lagrange MSM batch shape became ineligible")
                    .expect("selected Metal Lagrange MSM batch failed")
                },
            );
        }

        #[cfg(feature = "gpu-blitzar")]
        {
            let total_scalars = polynomials
                .iter()
                .map(|polynomial| polynomial.values.len())
                .sum();
            let same_size = polynomials
                .iter()
                .all(|polynomial| polynomial.values.len() == size);
            let nonzero_scalars = same_size.then(|| nonzero_scalar_count(polynomials));
            if same_size
                && nonzero_scalars.is_some_and(|nonzero_scalars| {
                    gpu_blitzar::should_accelerate_sppark_batch(
                        size,
                        total_scalars,
                        nonzero_scalars,
                        polynomials.len(),
                    )
                })
                && is_midnight_bls12_381::<E>()
            {
                let bases = midnight_g1_bases(&params.g_lagrange[..size]);
                let scalar_batches = polynomials
                    .iter()
                    .map(|polynomial| midnight_fq_scalars(polynomial.values.as_slice()))
                    .collect::<Vec<_>>();
                let commitments = crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark",
                    total_scalars,
                    || {
                        midnight_curves::G1Projective::multi_exp_sppark_batch(
                            bases,
                            &scalar_batches,
                        )
                        .expect("sppark BLS12-381 lagrange MSM batch failed")
                    },
                );
                return commitments.into_iter().map(midnight_g1_from_gpu).collect();
            }
            if same_size
                && polynomials.len() < 4
                && gpu_blitzar::should_accelerate_sppark(size)
                && is_midnight_bls12_381::<E>()
                && gpu_blitzar::use_cpu_for_sppark_small_batch()
            {
                if midnight_sppark_msm::current_execution_lane() != 0 {
                    return polynomials
                        .par_iter()
                        .map(|polynomial| {
                            let _gpu_guard = gpu_blitzar::DisableGuard::new();
                            Self::commit_lagrange(params, polynomial)
                        })
                        .collect();
                }
                let _gpu_guard = gpu_blitzar::EnableGuard::set(false);
                return polynomials
                    .par_iter()
                    .map(|polynomial| Self::commit_lagrange(params, polynomial))
                    .collect();
            }
            if gpu_blitzar::should_accelerate(total_scalars, polynomials.len()) {
                let scalar_slices = polynomials
                    .iter()
                    .map(|polynomial| polynomial.values.as_slice())
                    .collect::<Vec<_>>();
                return crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_blitzar",
                    total_scalars,
                    || {
                        gpu_blitzar::commit_batch::<E::Fr, E::G1>(
                            &scalar_slices,
                            &params.g_lagrange,
                        )
                        .expect("Blitzar KZG lagrange MSM batch failed")
                    },
                );
            }
        }

        #[cfg(feature = "gpu-blitzar")]
        {
            let lane = midnight_sppark_msm::current_execution_lane();
            if lane != 0 {
                return polynomials
                    .par_iter()
                    .map(|polynomial| {
                        let _lane = midnight_sppark_msm::bind_execution_lane(lane)
                            .expect("failed to propagate sppark proof execution lane");
                        Self::commit_lagrange(params, polynomial)
                    })
                    .collect();
            }
        }
        polynomials
            .par_iter()
            .map(|polynomial| Self::commit_lagrange(params, polynomial))
            .collect()
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_ec_gpu_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<E::Fr, LagrangeCoeff>],
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if polynomials.is_empty() || !gpu_blitzar::enabled() || !is_midnight_bls12_381::<E>() {
            return Ok(None);
        }
        let size = polynomials[0].values.len();
        if size == 0
            || polynomials
                .iter()
                .any(|polynomial| polynomial.values.len() != size)
        {
            return Ok(None);
        }
        if params.g_lagrange.len() < size {
            return Err("ec-gpu lagrange MSM exceeds the SRS size".to_string());
        }

        let bases = midnight_g1_bases(&params.g_lagrange[..size]);
        let scalar_batches = polynomials
            .iter()
            .map(|polynomial| midnight_fq_scalars(polynomial.values.as_slice()))
            .collect::<Vec<_>>();
        let commitments = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.lagrange_msm_batch.ec_gpu",
            size * polynomials.len(),
            || midnight_curves::G1Projective::multi_exp_ec_gpu_batch(bases, &scalar_batches),
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
        ))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_resident_scrambled_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<E::Fr, LagrangeCoeff>],
    ) -> Result<
        Option<(
            Vec<Self::Commitment>,
            std::sync::Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>,
        )>,
        String,
    > {
        if polynomials.is_empty() || !gpu_blitzar::enabled() || !is_midnight_bls12_381::<E>() {
            return Ok(None);
        }
        let poly_len = polynomials[0].values.len();
        if poly_len == 0
            || polynomials
                .iter()
                .any(|polynomial| polynomial.values.len() != poly_len)
        {
            return Ok(None);
        }
        if params.g_lagrange.len() < poly_len {
            return Err("resident advice MSM exceeds the SRS size".to_string());
        }

        let values = polynomials
            .iter()
            .map(|polynomial| raw_scalar_slice(polynomial.values.as_slice()))
            .collect::<Vec<_>>();
        let value_count = values.iter().map(|values| values.len()).sum();
        let device = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.advice_resident.upload",
            value_count,
            || midnight_sppark_msm::Bls12_381FrDeviceVec::upload_many(&values),
        )
        .map_err(|error| error.to_string())?;
        let bases = midnight_g1_bases(&params.g_lagrange[..poly_len]);
        let scramble = !matches!(
            std::env::var("MIDNIGHT_GPU_ADVICE_MSM_SCRAMBLE")
                .unwrap_or_else(|_| "on".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off" | "none" | "direct"
        );
        let commitments = crate::perf_timer::timed_count(
            if scramble {
                "proof_server.prover.kzg.commit.advice_resident.scrambled_batch"
            } else {
                "proof_server.prover.kzg.commit.advice_resident.direct_batch"
            },
            value_count,
            || {
                if scramble {
                    midnight_curves::G1Projective::multi_exp_sppark_device_scrambled_batch(
                        bases,
                        &device,
                        polynomials.len(),
                    )
                } else {
                    midnight_curves::G1Projective::multi_exp_sppark_device_batch(
                        bases,
                        &device,
                        polynomials.len(),
                    )
                }
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Some((
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
            std::sync::Arc::new(device),
        )))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_suffix_batch(
        params: &Self::Parameters,
        suffixes: &[(usize, Vec<E::Fr>)],
        poly_len: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if suffixes.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if params.g_lagrange.len() < poly_len
            || suffixes
                .iter()
                .any(|(start, values)| *start > poly_len || values.len() > poly_len - *start)
        {
            return Err("sparse Lagrange suffix exceeds the SRS domain".to_string());
        }

        let scalar_count = suffixes.iter().map(|(_, values)| values.len()).sum();
        let lane = midnight_sppark_msm::current_execution_lane();
        let commitments = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.lagrange_msm_suffix_batch.cpu",
            scalar_count,
            || {
                suffixes
                    .par_iter()
                    .map(|(start, values)| {
                        let _lane = (lane != 0).then(|| {
                            midnight_sppark_msm::bind_execution_lane(lane)
                                .expect("failed to propagate sppark proof execution lane")
                        });
                        msm_specific::<E::G1Affine>(
                            values,
                            &params.g_lagrange[*start..*start + values.len()],
                        )
                    })
                    .collect()
            },
        );
        Ok(Some(commitments))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_device_suffix_batch(
        params: &Self::Parameters,
        polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        scalar_offset: usize,
        starts: &[usize],
        nonzero_counts: &[usize],
        poly_len: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if starts.is_empty() || !gpu_blitzar::enabled() || !is_midnight_bls12_381::<E>() {
            return Ok(None);
        }
        let scalar_count = poly_len
            .checked_mul(starts.len())
            .and_then(|len| scalar_offset.checked_add(len))
            .ok_or_else(|| "resident lookup suffix range overflow".to_string())?;
        if poly_len == 0
            || params.g_lagrange.len() < poly_len
            || scalar_count > polynomials.len()
            || starts.iter().any(|&start| start > poly_len)
            || nonzero_counts.len() != starts.len()
            || starts
                .iter()
                .zip(nonzero_counts)
                .any(|(&start, &nonzero_count)| nonzero_count > poly_len - start)
        {
            return Err("resident lookup suffix exceeds its polynomial arena".to_string());
        }

        let bases = midnight_g1_bases(&params.g_lagrange[..poly_len]);
        let slices = starts
            .iter()
            .enumerate()
            .zip(nonzero_counts)
            .map(|((index, &start), &nonzero_count)| {
                (
                    scalar_offset + index * poly_len + start,
                    start,
                    poly_len - start,
                    nonzero_count,
                )
            })
            .collect::<Vec<_>>();
        let total_scalars = slices
            .iter()
            .map(|(_, _, _, nonzero_len)| *nonzero_len)
            .sum();
        let commitments = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.lagrange_msm_suffix_batch.resident_ragged",
            total_scalars,
            || {
                midnight_curves::G1Projective::multi_exp_sppark_device_ragged(
                    bases,
                    polynomials,
                    &slices,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
        ))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_device_batch(
        params: &Self::Parameters,
        polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        poly_len: usize,
        count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if count == 0
            || poly_len == 0
            || polynomials.len() != poly_len.saturating_mul(count)
            || !gpu_blitzar::enabled()
            || !is_midnight_bls12_381::<E>()
        {
            return Ok(None);
        }
        if params.g_lagrange.len() < poly_len {
            return Err("resident lagrange MSM exceeds the SRS size".to_string());
        }

        let bases = midnight_g1_bases(&params.g_lagrange[..poly_len]);
        let scramble = !matches!(
            std::env::var("MIDNIGHT_GPU_LAGRANGE_MSM_SCRAMBLE")
                .unwrap_or_else(|_| "on".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off" | "none" | "direct"
        );
        let commitments = crate::perf_timer::timed_count(
            if scramble {
                "proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark_device_scrambled"
            } else {
                "proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark_device"
            },
            poly_len * count,
            || {
                if scramble {
                    midnight_curves::G1Projective::multi_exp_sppark_device_scrambled_batch(
                        bases,
                        polynomials,
                        count,
                    )
                } else {
                    midnight_curves::G1Projective::multi_exp_sppark_device_batch(
                        bases,
                        polynomials,
                        count,
                    )
                }
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
        ))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_lagrange_device_scrambled_batch(
        params: &Self::Parameters,
        polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        poly_len: usize,
        count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if count == 0
            || poly_len == 0
            || polynomials.len() != poly_len.saturating_mul(count)
            || !gpu_blitzar::enabled()
            || !is_midnight_bls12_381::<E>()
        {
            return Ok(None);
        }
        if params.g_lagrange.len() < poly_len {
            return Err("resident masked lagrange MSM exceeds the SRS size".to_string());
        }

        let bases = midnight_g1_bases(&params.g_lagrange[..poly_len]);
        let commitments = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark_device_scrambled",
            poly_len * count,
            || {
                midnight_curves::G1Projective::multi_exp_sppark_device_scrambled_batch(
                    bases,
                    polynomials,
                    count,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
        ))
    }

    #[cfg(feature = "gpu-blitzar")]
    fn commit_device_scrambled_batch(
        params: &Self::Parameters,
        polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        poly_len: usize,
        count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        if count == 0
            || poly_len == 0
            || polynomials.len() != poly_len.saturating_mul(count)
            || !gpu_blitzar::enabled()
            || !is_midnight_bls12_381::<E>()
        {
            return Ok(None);
        }
        if params.g.len() < poly_len {
            return Err("resident scrambled coefficient MSM exceeds the SRS size".to_string());
        }

        let bases = midnight_g1_bases(&params.g[..poly_len]);
        let commitments = crate::perf_timer::timed_count(
            "proof_server.prover.kzg.commit.coeff_msm_batch.gpu_sppark_device_scrambled",
            poly_len * count,
            || {
                midnight_curves::G1Projective::multi_exp_sppark_device_scrambled_batch(
                    bases,
                    polynomials,
                    count,
                )
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(
            commitments.into_iter().map(midnight_g1_from_gpu).collect(),
        ))
    }

    fn multi_open<T: Transcript>(
        params: &Self::Parameters,
        prover_query: &[ProverQuery<E::Fr>],
        transcript: &mut T,
    ) -> Result<(), Error>
    where
        E::Fr: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        E::G1: Hashable<T::Hash>,
    {
        #[cfg(feature = "gpu-blitzar")]
        {
            Self::multi_open_impl(params, prover_query, transcript, None)
        }
        #[cfg(not(feature = "gpu-blitzar"))]
        {
            Self::multi_open_impl(params, prover_query, transcript)
        }
    }

    #[cfg(feature = "gpu-blitzar")]
    fn multi_open_with_arena<T: Transcript>(
        params: &Self::Parameters,
        prover_query: &[ProverQuery<E::Fr>],
        transcript: &mut T,
        arena: Option<crate::poly::commitment::ResidentMultiopenArena>,
    ) -> Result<(), Error>
    where
        E::Fr: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        E::G1: Hashable<T::Hash>,
    {
        Self::multi_open_impl(params, prover_query, transcript, arena)
    }

    fn multi_prepare<'com, T: Transcript>(
        verifier_query: &[VerifierQuery<'com, E::Fr, KZGCommitmentScheme<E>>],
        transcript: &mut T,
    ) -> Result<DualMSM<E>, Error>
    where
        E::Fr: Sampleable<T::Hash> + Ord + Hash + Hashable<T::Hash>,
        E::G1: 'com + Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr>,
    {
        // Refer to the halo2 book for docs:
        // https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html
        let x1: E::Fr = transcript.squeeze_challenge();
        let x2: E::Fr = transcript.squeeze_challenge();

        let (commitment_map, point_sets) = construct_intermediate_sets(verifier_query)?;

        let mut q_coms: Vec<_> = vec![vec![]; point_sets.len()];
        let mut q_eval_sets = vec![vec![]; point_sets.len()];

        for com_data in commitment_map.into_iter() {
            let mut msm = MSMKZG::init();
            let eval_point_opt = if com_data.commitment.is_chopped() {
                // When the commitment is in chopped form, we require that it be evaluated
                // in a single point.
                debug_assert!(com_data.point_indices.len() == 1);
                Some(point_sets[com_data.set_index][com_data.point_indices[0]])
            } else {
                None
            };
            for (scalar, commitment) in com_data.commitment.as_terms(eval_point_opt) {
                msm.append_term(scalar, commitment, com_data.commitment_label.clone());
            }
            q_coms[com_data.set_index].push(msm);
            q_eval_sets[com_data.set_index].push(com_data.evals);
        }

        let nb_x1_powers = q_coms.iter().map(|v| v.len()).max().unwrap_or(0);
        assert!(nb_x1_powers >= q_eval_sets.iter().map(|v| v.len()).max().unwrap_or(0));

        #[cfg(feature = "truncated-challenges")]
        let powers_x1 = truncated_powers(x1).take(nb_x1_powers).collect::<Vec<_>>();

        #[cfg(not(feature = "truncated-challenges"))]
        let powers_x1 = powers(x1).take(nb_x1_powers).collect::<Vec<_>>();

        let q_coms = q_coms
            .into_iter()
            .map(|msms| msm_inner_product(msms, &powers_x1))
            .collect::<Vec<_>>();

        let q_eval_sets = q_eval_sets
            .iter()
            .map(|evals| evals_inner_product(evals, &powers_x1))
            .collect::<Vec<_>>();

        // Sort point sets by ascending cardinality to ensure the first set is the one
        // that contains fixed commitments (which are evaluated at x only). This
        // property is not necessary for the actual proving system, but it is important
        // for in-circuit verification of proofs. (It enables an optimization based on
        // an internal collapse.)
        //
        // The (len, i) key provides a deterministic total order even when two sets
        // share the same cardinality.
        let (q_coms, q_eval_sets, point_sets) = {
            let mut order: Vec<usize> = (0..point_sets.len()).collect();
            order.sort_by_key(|&i| (point_sets[i].len(), i));
            let q_coms: Vec<_> = order.iter().map(|&i| q_coms[i].clone()).collect();
            let q_eval_sets: Vec<_> = order.iter().map(|&i| q_eval_sets[i].clone()).collect();
            let point_sets: Vec<_> = order.iter().map(|&i| point_sets[i].clone()).collect();
            (q_coms, q_eval_sets, point_sets)
        };

        let f_com: E::G1 = transcript.read().map_err(|_| Error::SamplingError)?;

        // Sample a challenge x_3 for checking that f(X) was committed to
        // correctly.
        let x3: E::Fr = transcript.squeeze_challenge();
        #[cfg(feature = "truncated-challenges")]
        let x3 = truncate(x3);

        let mut q_evals_on_x3 = Vec::<E::Fr>::with_capacity(q_eval_sets.len());
        for _ in 0..q_eval_sets.len() {
            q_evals_on_x3.push(transcript.read().map_err(|_| Error::SamplingError)?);
        }

        // We can compute the expected msm_eval at x_3 using the u provided
        // by the prover and from x_2
        let f_eval = point_sets
            .iter()
            .zip(q_eval_sets.iter())
            .zip(q_evals_on_x3.iter())
            .rev()
            .fold(E::Fr::ZERO, |acc_eval, ((points, evals), proof_eval)| {
                let r_poly = lagrange_interpolate(points, evals);
                let r_eval = eval_polynomial(&r_poly, x3);
                // eval = (proof_eval - r_eval) / prod_i (x3 - point_i)
                let den = points
                    .iter()
                    .fold(E::Fr::ONE, |acc, point| acc * &(x3 - point));
                let eval = (*proof_eval - &r_eval) * den.invert().unwrap();
                acc_eval * &(x2) + &eval
            });

        let x4: E::Fr = transcript.squeeze_challenge();

        let final_com = {
            let size = q_coms.len() + 1;
            let mut coms = q_coms;
            let mut f_com_as_msm = MSMKZG::init();

            f_com_as_msm.append_term(E::Fr::ONE, f_com, CommitmentLabel::NoLabel);

            // Collapse all MSMs before combining with x4 powers, to match the
            // in-circuit verifier. Skip the first one since its x4 power is 1.
            #[cfg(feature = "truncated-challenges")]
            coms.iter_mut().skip(1).for_each(MSMKZG::collapse);
            coms.push(f_com_as_msm);

            #[cfg(feature = "truncated-challenges")]
            let powers = truncated_powers(x4);

            #[cfg(not(feature = "truncated-challenges"))]
            let powers = powers(x4);

            msm_inner_product(coms, &powers.take(size).collect::<Vec<_>>())
        };

        let v = {
            let mut evals = q_evals_on_x3;
            evals.push(f_eval);

            #[cfg(feature = "truncated-challenges")]
            let powers = truncated_powers(x4);

            #[cfg(not(feature = "truncated-challenges"))]
            let powers = powers(x4);

            inner_product(&evals, powers)
        };

        let pi: E::G1 = transcript.read().map_err(|_| Error::SamplingError)?;

        let mut pi_msm = MSMKZG::<E>::init();
        pi_msm.append_term(E::Fr::ONE, pi, CommitmentLabel::Custom("π".into()));

        // Scale zπ - vG
        let scaled_pi = MSMKZG {
            scalars: vec![x3, v],
            bases: vec![pi, -E::G1::generator()],
            labels: vec![
                CommitmentLabel::Custom("π".into()),
                CommitmentLabel::Custom("-G".into()),
            ],
        };

        // (π, C − vG + zπ)
        let mut msm_accumulator = DualMSM {
            left: pi_msm,
            right: final_com,
        };
        msm_accumulator.right.add_msm(&scaled_pi);

        Ok(msm_accumulator)
    }
}

impl<E: MultiMillerLoop> KZGCommitmentScheme<E>
where
    E::Fr: Sync + 'static,
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject + 'static,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + 'static,
    ParamsKZG<E>: Sync,
{
    fn multi_open_impl<T: Transcript>(
        params: &ParamsKZG<E>,
        prover_query: &[ProverQuery<E::Fr>],
        transcript: &mut T,
        #[cfg(feature = "gpu-blitzar")] mut resident_multiopen_arena: Option<
            crate::poly::commitment::ResidentMultiopenArena,
        >,
    ) -> Result<(), Error>
    where
        E::Fr: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        E::G1: Hashable<T::Hash>,
    {
        let _timer = crate::perf_timer::ScopedTimer::with_count(
            "proof_server.prover.kzg.multi_open.body",
            Some(prover_query.len()),
        );

        // Refer to the halo2 book for docs:
        // https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html
        let x1: E::Fr = transcript.squeeze_challenge();
        let x2: E::Fr = transcript.squeeze_challenge();

        #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(result) =
            try_metal_resident_multi_open(params, prover_query, transcript, x1, x2)
        {
            return result;
        }

        #[cfg(feature = "gpu-blitzar")]
        let (mut poly_map, point_sets, eval_targets) =
            construct_intermediate_set_metadata(prover_query)?;
        #[cfg(feature = "gpu-blitzar")]
        let multiopen_poly_len = poly_map
            .first()
            .map(|data| data.commitment.poly.len())
            .ok_or(Error::OpeningError)?;
        #[cfg(not(feature = "gpu-blitzar"))]
        let (poly_map, point_sets) = construct_intermediate_sets(prover_query)?;
        #[cfg(not(feature = "gpu-blitzar"))]
        let multiopen_poly_len = poly_map
            .first()
            .map(|data| data.commitment.poly.len())
            .ok_or(Error::OpeningError)?;

        #[cfg(feature = "gpu-blitzar")]
        let (gpu_q_polys, mut resident_q_polys) = if is_midnight_bls12_381::<E>()
            && poly_map.first().is_some_and(|data| {
                gpu_multiopen_prepare_enabled(data.commitment.poly.len())
                    || (resident_multiopen_arena.is_some()
                        && gpu_resident_multiopen_enabled(data.commitment.poly.len()))
            })
            && poly_map
                .iter()
                .all(|data| data.commitment.poly.len() == poly_map[0].commitment.poly.len())
        {
            let poly_len = poly_map[0].commitment.poly.len();
            let mut q_offsets = Vec::with_capacity(point_sets.len() + 1);
            let mut q_poly_indices = Vec::with_capacity(poly_map.len());
            let mut q_coefficients = Vec::with_capacity(poly_map.len());
            q_offsets.push(0u32);
            for set_index in 0..point_sets.len() {
                let indices = poly_map
                    .iter()
                    .enumerate()
                    .filter_map(|(index, data)| (data.set_index == set_index).then_some(index))
                    .collect::<Vec<_>>();
                #[cfg(feature = "truncated-challenges")]
                let coefficients = truncated_powers(x1).take(indices.len()).collect::<Vec<_>>();
                #[cfg(not(feature = "truncated-challenges"))]
                let coefficients = powers(x1).take(indices.len()).collect::<Vec<_>>();
                for (index, coefficient) in indices.into_iter().zip(coefficients) {
                    q_poly_indices.push(index as u32);
                    q_coefficients.push(raw_scalar(coefficient));
                }
                q_offsets.push(q_poly_indices.len() as u32);
            }

            let mut points = Vec::new();
            let mut query_point_indices = Vec::with_capacity(prover_query.len());
            for query in prover_query {
                let point = query.point;
                let point_index = points
                    .iter()
                    .position(|candidate| *candidate == point)
                    .unwrap_or_else(|| {
                        points.push(point);
                        points.len() - 1
                    });
                query_point_indices.push(point_index as u32);
            }
            let raw_points = points.into_iter().map(raw_scalar).collect::<Vec<_>>();
            let query_poly_indices = eval_targets
                .iter()
                .map(|target| target.commitment_index as u32)
                .collect::<Vec<_>>();

            let (raw_q_polys, query_evals, resident_q_polys) = if let Some(arena) =
                resident_multiopen_arena.take()
            {
                let mut extra_polys = Vec::new();
                let mut poly_refs = Vec::with_capacity(poly_map.len());
                for data in &poly_map {
                    let pointer = data.commitment.poly as *const Polynomial<E::Fr, Coeff> as usize;
                    if let Some(source) = arena.polynomial_sources.get(&pointer) {
                        poly_refs.push(midnight_sppark_msm::RawDevicePolyRef {
                            source: source.source,
                            index: source.index,
                        });
                    } else {
                        let index = extra_polys.len() / poly_len;
                        extra_polys.extend(data.commitment.poly.iter().copied().map(raw_scalar));
                        poly_refs.push(midnight_sppark_msm::RawDevicePolyRef {
                            source: 2,
                            index: index as u32,
                        });
                    }
                }
                if extra_polys.is_empty() {
                    extra_polys.resize(poly_len, midnight_sppark_msm::Bls12_381Fr::default());
                }
                let extra = midnight_sppark_msm::Bls12_381FrDeviceVec::upload(&extra_polys)
                    .map_err(|_| Error::OpeningError)?;
                if gpu_resident_multiopen_pipeline_enabled() {
                    let prepared = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.prepare.resident_pipeline_gpu_sppark",
                        extra_polys.len(),
                        || {
                            midnight_sppark_msm::bls12_381_fr_multiopen_prepare_device_resident(
                                &arena.dynamic,
                                &arena.fixed,
                                &extra,
                                poly_len,
                                &poly_refs,
                                &q_offsets,
                                &q_poly_indices,
                                &q_coefficients,
                                &raw_points,
                                &query_poly_indices,
                                &query_point_indices,
                            )
                        },
                    )
                    .map_err(|_| Error::OpeningError)?;
                    let q_polys = prepared.q_polys;
                    let raw_q_polys = if gpu_resident_multiopen_pipeline_debug() {
                        q_polys.download().map_err(|_| Error::OpeningError)?
                    } else {
                        Vec::new()
                    };
                    (raw_q_polys, prepared.query_evals, Some(q_polys))
                } else {
                    let prepared = crate::perf_timer::timed_count_detailed(
                        "proof_server.prover.kzg.multi_open.prepare.resident_gpu_sppark",
                        extra_polys.len(),
                        || {
                            midnight_sppark_msm::bls12_381_fr_multiopen_prepare_device(
                                &arena.dynamic,
                                &arena.fixed,
                                &extra,
                                poly_len,
                                &poly_refs,
                                &q_offsets,
                                &q_poly_indices,
                                &q_coefficients,
                                &raw_points,
                                &query_poly_indices,
                                &query_point_indices,
                            )
                        },
                    )
                    .map_err(|_| Error::OpeningError)?;
                    (prepared.q_polys, prepared.query_evals, None)
                }
            } else {
                let mut raw_polys = Vec::with_capacity(poly_len * poly_map.len());
                for data in &poly_map {
                    raw_polys.extend(data.commitment.poly.iter().copied().map(raw_scalar));
                }
                let prepared = crate::perf_timer::timed_count_detailed(
                    "proof_server.prover.kzg.multi_open.prepare.gpu_sppark",
                    raw_polys.len(),
                    || {
                        midnight_sppark_msm::bls12_381_fr_multiopen_prepare(
                            &raw_polys,
                            poly_len,
                            &q_offsets,
                            &q_poly_indices,
                            &q_coefficients,
                            &raw_points,
                            &query_poly_indices,
                            &query_point_indices,
                        )
                    },
                )
                .map_err(|_| Error::OpeningError)?;
                (prepared.q_polys, prepared.query_evals, None)
            };
            for (target, eval) in eval_targets.iter().zip(query_evals) {
                poly_map[target.commitment_index].evals[target.eval_index] = field_from_raw(eval);
            }
            let q_count = point_sets.len();
            let host_q_polys =
                if resident_q_polys.is_some() && !gpu_resident_multiopen_pipeline_debug() {
                    (0..q_count)
                        .map(|_| Polynomial {
                            values: Vec::new(),
                            _marker: PhantomData,
                        })
                        .collect()
                } else {
                    raw_q_polys
                        .chunks_exact(poly_len)
                        .map(|values| Polynomial {
                            values: values.iter().copied().map(field_from_raw).collect(),
                            _marker: PhantomData,
                        })
                        .collect()
                };
            (Some(host_q_polys), resident_q_polys)
        } else {
            for (query, target) in prover_query.iter().zip(eval_targets) {
                poly_map[target.commitment_index].evals[target.eval_index] = query.get_eval();
            }
            (None, None)
        };

        #[cfg(feature = "gpu-blitzar")]
        let q_polys = if let Some(q_polys) = gpu_q_polys {
            q_polys
        } else {
            let mut grouped = vec![vec![]; point_sets.len()];
            for com_data in &poly_map {
                grouped[com_data.set_index].push(com_data.commitment.poly.clone());
            }
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.q_polys",
                grouped.iter().map(Vec::len).sum(),
                || {
                    grouped
                        .iter()
                        .map(|polys| {
                            #[cfg(feature = "truncated-challenges")]
                            let x1 = truncated_powers(x1);
                            #[cfg(not(feature = "truncated-challenges"))]
                            let x1 = powers(x1);
                            polynomial_inner_product(polys, x1)
                        })
                        .collect::<Vec<_>>()
                },
            )
        };

        #[cfg(not(feature = "gpu-blitzar"))]
        let q_polys = {
            let mut grouped = vec![vec![]; point_sets.len()];
            for com_data in &poly_map {
                grouped[com_data.set_index].push(com_data.commitment.poly.clone());
            }
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.q_polys",
                grouped.iter().map(Vec::len).sum(),
                || {
                    grouped
                        .iter()
                        .map(|polys| {
                            #[cfg(feature = "truncated-challenges")]
                            let x1 = truncated_powers(x1);
                            #[cfg(not(feature = "truncated-challenges"))]
                            let x1 = powers(x1);
                            polynomial_inner_product(polys, x1)
                        })
                        .collect::<Vec<_>>()
                },
            )
        };

        // Sort point sets by ascending cardinality to ensure the first set is the one
        // that contains fixed commitments (which are evaluated at x only). This
        // property is not necessary for the actual proving system, but it is important
        // for in-circuit verification of proofs. (It enables an optimization based on
        // an internal collapse.)
        //
        // The (len, i) key provides a deterministic total order even when two sets
        // share the same cardinality.
        let (q_polys, point_sets, _q_order) = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.sort_point_sets",
            point_sets.len(),
            || {
                let mut order: Vec<usize> = (0..point_sets.len()).collect();
                order.sort_by_key(|&i| (point_sets[i].len(), i));
                let q_polys: Vec<_> = order.iter().map(|&i| q_polys[i].clone()).collect();
                let point_sets: Vec<_> = order.iter().map(|&i| point_sets[i].clone()).collect();
                (q_polys, point_sets, order)
            },
        );

        #[cfg(feature = "gpu-blitzar")]
        let resident_q_polys = resident_q_polys
            .take()
            .map(|q_polys| {
                let order = _q_order
                    .iter()
                    .copied()
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| Error::OpeningError)?;
                q_polys
                    .gather_polys(q_polys.len() / order.len(), &order)
                    .map_err(|_| Error::OpeningError)
            })
            .transpose()?;

        #[cfg(feature = "gpu-blitzar")]
        let (resident_f_values, resident_f_poly) =
            if let Some(resident_q_polys) = resident_q_polys.as_ref() {
                let poly_len = multiopen_poly_len;
                let mut point_offsets = Vec::with_capacity(point_sets.len() + 1);
                let mut raw_points = Vec::new();
                point_offsets.push(0u32);
                for points in &point_sets {
                    raw_points.extend(points.iter().copied().map(raw_scalar));
                    point_offsets
                        .push(u32::try_from(raw_points.len()).map_err(|_| Error::OpeningError)?);
                }
                let combination_scalars = powers(x2)
                    .take(point_sets.len())
                    .map(raw_scalar)
                    .collect::<Vec<_>>();
                let resident_f = crate::perf_timer::timed_count_detailed(
                    "proof_server.prover.kzg.multi_open.f_poly.resident_gpu_sppark",
                    resident_q_polys.len(),
                    || {
                        midnight_sppark_msm::bls12_381_fr_multiopen_f_poly_device(
                            resident_q_polys,
                            poly_len,
                            &point_offsets,
                            &raw_points,
                            &combination_scalars,
                        )
                    },
                )
                .map_err(|_| Error::OpeningError)?;
                let values = if gpu_resident_multiopen_pipeline_debug() {
                    Some(
                        resident_f
                            .download()
                            .map_err(|_| Error::OpeningError)?
                            .into_iter()
                            .map(field_from_raw)
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                (values, Some(resident_f))
            } else {
                (None, None)
            };
        #[cfg(not(feature = "gpu-blitzar"))]
        let resident_f_values: Option<Vec<E::Fr>> = None;
        #[cfg(not(feature = "gpu-blitzar"))]
        let resident_f_poly: Option<()> = None;

        #[cfg(feature = "gpu-blitzar")]
        if gpu_resident_multiopen_pipeline_debug() {
            if let Some(actual) = resident_f_values.as_ref() {
                let expected_polys = point_sets
                    .iter()
                    .zip(q_polys.clone())
                    .map(|(points, q_poly)| {
                        let mut poly = q_poly.values;
                        for point in points {
                            poly = kate_division_owned(poly, *point);
                        }
                        poly.resize(1 << params.max_k() as usize, E::Fr::ZERO);
                        Polynomial {
                            values: poly,
                            _marker: PhantomData,
                        }
                    })
                    .collect::<Vec<_>>();
                let expected = polynomial_inner_product(&expected_polys, powers(x2));
                if actual != &expected.values {
                    let mismatch = actual
                        .iter()
                        .zip(&expected.values)
                        .position(|(actual, expected)| actual != expected)
                        .unwrap_or(actual.len().min(expected.values.len()));
                    eprintln!("resident multi-open f mismatch at coefficient {mismatch}");
                    return Err(Error::OpeningError);
                }
            }
        }

        let f_poly = if let Some(values) = resident_f_values {
            Polynomial {
                values,
                _marker: PhantomData,
            }
        } else if resident_f_poly.is_some() {
            Polynomial {
                values: Vec::new(),
                _marker: PhantomData,
            }
        } else {
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.f_poly",
                q_polys.iter().map(|poly| poly.values.len()).sum(),
                || {
                    let f_polys = point_sets
                        .iter()
                        .zip(q_polys.clone())
                        .map(|(points, q_poly)| {
                            let mut poly = q_poly.values;
                            for point in points {
                                poly = kate_division_owned(poly, *point);
                            }
                            poly.resize(1 << params.max_k() as usize, E::Fr::ZERO);
                            Polynomial {
                                values: poly,
                                _marker: PhantomData,
                            }
                        })
                        .collect::<Vec<_>>();
                    polynomial_inner_product(&f_polys, powers(x2))
                },
            )
        };
        let f_poly_len = if resident_f_poly.is_some() {
            multiopen_poly_len
        } else {
            f_poly.values.len()
        };

        let f_com = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.f_commit",
            f_poly_len,
            || {
                #[cfg(feature = "gpu-blitzar")]
                if let Some(resident_f_poly) = resident_f_poly.as_ref() {
                    return Self::commit_device_scrambled_batch(
                        params,
                        resident_f_poly,
                        f_poly_len,
                        1,
                    )
                    .expect("resident f commitment failed")
                    .expect("resident f commitment is unsupported")
                    .into_iter()
                    .next()
                    .expect("resident f commitment returned no point");
                }
                Self::commit(params, &f_poly)
            },
        );
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.f_commit_write",
            1,
            || transcript.write(&f_com).map_err(|_| Error::OpeningError),
        )?;

        let x3: E::Fr = transcript.squeeze_challenge();
        #[cfg(feature = "truncated-challenges")]
        let x3 = truncate(x3);

        #[cfg(feature = "gpu-blitzar")]
        let resident_q_evals = resident_q_polys
            .as_ref()
            .map(|resident_q_polys| {
                midnight_sppark_msm::bls12_381_fr_device_eval_polys(
                    resident_q_polys,
                    multiopen_poly_len,
                    q_polys.len(),
                    raw_scalar(x3),
                )
                .map(|evals| {
                    evals
                        .into_iter()
                        .map(field_from_raw::<E::Fr>)
                        .collect::<Vec<_>>()
                })
                .map_err(|_| Error::OpeningError)
            })
            .transpose()?;
        #[cfg(not(feature = "gpu-blitzar"))]
        let resident_q_evals: Option<Vec<E::Fr>> = None;

        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.q_eval_write",
            multiopen_poly_len.saturating_mul(q_polys.len()),
            || {
                if let Some(evals) = resident_q_evals.as_ref() {
                    for eval in evals {
                        transcript.write(eval).map_err(|_| Error::OpeningError)?;
                    }
                } else {
                    for q_poly in q_polys.iter() {
                        transcript
                            .write(&eval_polynomial(&q_poly.values, x3))
                            .map_err(|_| Error::OpeningError)?;
                    }
                }
                Ok::<(), Error>(())
            },
        )?;

        let x4: E::Fr = transcript.squeeze_challenge();

        #[cfg(feature = "gpu-blitzar")]
        let resident_pi = if let (Some(resident_q_polys), Some(resident_f_poly)) =
            (resident_q_polys.as_ref(), resident_f_poly.as_ref())
        {
            #[cfg(feature = "truncated-challenges")]
            let combination_scalars = truncated_powers(x4)
                .take(q_polys.len() + 1)
                .map(raw_scalar)
                .collect::<Vec<_>>();
            #[cfg(not(feature = "truncated-challenges"))]
            let combination_scalars = powers(x4)
                .take(q_polys.len() + 1)
                .map(raw_scalar)
                .collect::<Vec<_>>();
            let pi_poly = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.pi_poly.resident_gpu_sppark",
                resident_q_polys.len() + resident_f_poly.len(),
                || {
                    midnight_sppark_msm::bls12_381_fr_multiopen_pi_device(
                        resident_q_polys,
                        resident_f_poly,
                        f_poly_len,
                        &combination_scalars,
                        raw_scalar(x3),
                    )
                },
            )
            .map_err(|_| Error::OpeningError)?;
            let commitments = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.pi_commit.resident_gpu_sppark",
                pi_poly.len(),
                || Self::commit_device_scrambled_batch(params, &pi_poly, pi_poly.len(), 1),
            )
            .map_err(|_| Error::OpeningError)?
            .ok_or(Error::OpeningError)?;
            Some(commitments.into_iter().next().ok_or(Error::OpeningError)?)
        } else {
            None
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let resident_pi: Option<E::G1> = None;

        let pi = if let Some(pi) = resident_pi {
            pi
        } else {
            let final_poly = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.final_poly",
                q_polys.iter().map(|poly| poly.values.len()).sum::<usize>() + f_poly.values.len(),
                || {
                    let mut polys = q_polys;
                    polys.push(f_poly);
                    #[cfg(feature = "truncated-challenges")]
                    let powers = truncated_powers(x4);

                    #[cfg(not(feature = "truncated-challenges"))]
                    let powers = powers(x4);

                    polynomial_inner_product(&polys, powers)
                },
            );
            let v = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.final_eval",
                final_poly.values.len(),
                || eval_polynomial(&final_poly, x3),
            );

            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.kzg.multi_open.pi_commit",
                final_poly.values.len(),
                || {
                    let pi_poly = Polynomial {
                        values: kate_division_owned((&final_poly - v).values, x3),
                        _marker: PhantomData,
                    };
                    Self::commit(params, &pi_poly)
                },
            )
        };

        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.kzg.multi_open.pi_write",
            1,
            || transcript.write(&pi).map_err(|_| Error::OpeningError),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hash;

    use blake2b_simd::State as Blake2bState;
    use ff::{Field, WithSmallOrderMulGroup};
    use midnight_curves::{
        CurveAffine, CurveExt, Fq, pairing::MultiMillerLoop, serde::SerdeObject,
    };
    use rand_core::OsRng;

    use crate::{
        poly::{
            CommitmentLabel, EvaluationDomain,
            commitment::{Guard, PolynomialCommitmentScheme},
            kzg::{
                KZGCommitmentScheme,
                params::{ParamsKZG, ParamsVerifierKZG},
            },
            query::{ProverQuery, VerifierQuery},
        },
        transcript::{CircuitTranscript, Hashable, Sampleable, Transcript},
        utils::arithmetic::eval_polynomial,
    };

    #[derive(Clone, Copy)]
    enum MultiOpenEntryPoint {
        Standard,
        #[cfg(feature = "gpu-blitzar")]
        ArenaNone,
    }

    #[test]
    fn fused_polynomial_inner_product_matches_coefficient_reference() {
        let domain = EvaluationDomain::<Fq>::new(1, 4);
        let polys = (0u64..4)
            .map(|poly_index| {
                let mut poly = domain.empty_coeff();
                for (coefficient, value) in poly.iter_mut().enumerate() {
                    *value = Fq::from((poly_index + 1) * 100 + coefficient as u64);
                }
                poly
            })
            .collect::<Vec<_>>();
        let refs = polys.iter().collect::<Vec<_>>();

        for scalars in [
            [Fq::ZERO, Fq::ONE, Fq::from(3), Fq::ZERO],
            [Fq::ONE, Fq::ZERO, Fq::ONE, Fq::from(7)],
            [Fq::from(5), Fq::from(2), Fq::ZERO, Fq::from(11)],
        ] {
            let actual = super::polynomial_inner_product_refs(&refs, &scalars);
            let mut expected = domain.empty_coeff();
            for (value, coefficient) in expected.iter_mut().zip(0..) {
                for (poly, scalar) in polys.iter().zip(scalars) {
                    *value += poly[coefficient] * scalar;
                }
            }
            assert_eq!(actual, expected);
        }
    }

    #[cfg(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_msm_default_window_tracks_measured_density_crossover() {
        use midnight_gpu_prover::metal::Bls12_381MsmWindow::{FixedW11, FixedW13};

        assert_eq!(
            super::default_metal_msm_window_for_nonzero(16_384, 10_240),
            FixedW11
        );
        assert_eq!(
            super::default_metal_msm_window_for_nonzero(16_384, 10_241),
            FixedW13
        );
        assert_eq!(
            super::default_metal_msm_window_for_nonzero(32_768, 20_480),
            FixedW11
        );
        assert_eq!(
            super::default_metal_msm_window_for_nonzero(32_768, 20_481),
            FixedW13
        );
        assert_eq!(
            super::default_metal_msm_window_for_nonzero(32_768, 32_768),
            FixedW13
        );
    }

    #[test]
    fn test_roundtrip_gwc() {
        use midnight_curves::Bls12;

        const K: u32 = 4;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);

        let proof = create_proof::<_, CircuitTranscript<Blake2bState>>(&params);

        let verifier_params = params.verifier_params();
        verify::<Bls12, CircuitTranscript<Blake2bState>>(&verifier_params, &proof[..], false);

        verify::<Bls12, CircuitTranscript<Blake2bState>>(&verifier_params, &proof[..], true);
    }

    #[cfg(feature = "gpu-blitzar")]
    #[test]
    fn test_multi_open_with_arena_none_matches_multi_open() {
        use midnight_curves::Bls12;

        const K: u32 = 4;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let standard = create_proof_with_entry_point::<_, CircuitTranscript<Blake2bState>>(
            &params,
            MultiOpenEntryPoint::Standard,
        );
        let arena_none = create_proof_with_entry_point::<_, CircuitTranscript<Blake2bState>>(
            &params,
            MultiOpenEntryPoint::ArenaNone,
        );

        assert_eq!(arena_none, standard);
    }

    #[cfg(feature = "cuda-resident-multiopen-test-support")]
    #[test]
    #[ignore = "requires a CUDA device and the sppark resident multi-open pipeline"]
    fn test_multi_open_with_resident_arena_matches_and_is_error_isolated() {
        super::assert_resident_multiopen_ownership_error_isolation();
    }

    fn verify<E, T>(verifier_params: &ParamsVerifierKZG<E>, proof: &[u8], should_fail: bool)
    where
        E: MultiMillerLoop,
        T: Transcript,
        E::Fr: Hashable<T::Hash> + Sampleable<T::Hash> + Ord + Hash,
        E::G1: Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
        E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + SerdeObject,
    {
        let mut transcript = T::init_from_bytes(proof);

        let a: E::G1 = transcript.read().unwrap();
        let b: E::G1 = transcript.read().unwrap();
        let c: E::G1 = transcript.read().unwrap();

        let x: E::Fr = transcript.squeeze_challenge();
        let y: E::Fr = transcript.squeeze_challenge();

        let avx: E::Fr = transcript.read().unwrap();
        let bvx: E::Fr = transcript.read().unwrap();
        let cvy: E::Fr = transcript.read().unwrap();

        use CommitmentLabel::NoLabel;
        let valid_queries = std::iter::empty()
            .chain(Some(VerifierQuery::new(x, NoLabel, &a, avx)))
            .chain(Some(VerifierQuery::new(x, NoLabel, &b, bvx)))
            .chain(Some(VerifierQuery::new(y, NoLabel, &c, cvy)));

        let invalid_queries = std::iter::empty()
            .chain(Some(VerifierQuery::new(x, NoLabel, &a, avx)))
            .chain(Some(VerifierQuery::new(x, NoLabel, &b, avx)))
            .chain(Some(VerifierQuery::new(y, NoLabel, &c, cvy)));

        let queries = if should_fail {
            invalid_queries
        } else {
            valid_queries
        };

        let result =
            KZGCommitmentScheme::multi_prepare(&queries.collect::<Vec<_>>(), &mut transcript)
                .unwrap();

        if should_fail {
            assert!(result.verify(verifier_params).is_err());
        } else {
            assert!(result.verify(verifier_params).is_ok());
        }
    }

    fn create_proof<E, T>(kzg_params: &ParamsKZG<E>) -> Vec<u8>
    where
        E: MultiMillerLoop,
        T: Transcript,
        E::Fr: WithSmallOrderMulGroup<3> + Hashable<T::Hash> + Hash + Sampleable<T::Hash> + Ord,
        E::G1: Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
        E::G1Affine: SerdeObject + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    {
        create_proof_with_entry_point::<E, T>(kzg_params, MultiOpenEntryPoint::Standard)
    }

    fn create_proof_with_entry_point<E, T>(
        kzg_params: &ParamsKZG<E>,
        entry_point: MultiOpenEntryPoint,
    ) -> Vec<u8>
    where
        E: MultiMillerLoop,
        T: Transcript,
        E::Fr: WithSmallOrderMulGroup<3> + Hashable<T::Hash> + Hash + Sampleable<T::Hash> + Ord,
        E::G1: Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
        E::G1Affine: SerdeObject + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    {
        let k = (kzg_params.g.len() - 1).ilog2() + 1;
        let domain = EvaluationDomain::new(1, k);

        let mut ax = domain.empty_coeff();
        for (i, a) in ax.iter_mut().enumerate() {
            *a = <E::Fr>::from(10 + i as u64);
        }

        let mut bx = domain.empty_coeff();
        for (i, a) in bx.iter_mut().enumerate() {
            *a = <E::Fr>::from(100 + i as u64);
        }

        let mut cx = domain.empty_coeff();
        for (i, a) in cx.iter_mut().enumerate() {
            *a = <E::Fr>::from(100 + i as u64);
        }

        let mut transcript = T::init();

        let a = KZGCommitmentScheme::commit(kzg_params, &ax);
        let b = KZGCommitmentScheme::commit(kzg_params, &bx);
        let c = KZGCommitmentScheme::commit(kzg_params, &cx);

        transcript.write(&a).unwrap();
        transcript.write(&b).unwrap();
        transcript.write(&c).unwrap();

        let x: E::Fr = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenge();

        let avx = eval_polynomial(&ax, x);
        let bvx = eval_polynomial(&bx, x);
        let cvy = eval_polynomial(&cx, y);

        transcript.write(&avx).unwrap();
        transcript.write(&bvx).unwrap();
        transcript.write(&cvy).unwrap();

        let queries = vec![
            ProverQuery {
                point: x,
                poly: &ax,
            },
            ProverQuery {
                point: x,
                poly: &bx,
            },
            ProverQuery {
                point: y,
                poly: &cx,
            },
        ];

        match entry_point {
            MultiOpenEntryPoint::Standard => {
                KZGCommitmentScheme::multi_open(kzg_params, &queries, &mut transcript)
            }
            #[cfg(feature = "gpu-blitzar")]
            MultiOpenEntryPoint::ArenaNone => KZGCommitmentScheme::multi_open_with_arena(
                kzg_params,
                &queries,
                &mut transcript,
                None,
            ),
        }
        .unwrap();

        transcript.finalize()
    }
}
