//! Optional Blitzar-backed BLS12-381 G1 MSM acceleration.

use std::{
    cell::Cell,
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use ark_serialize::CanonicalDeserialize;
use blitzar::{compute::compute_bls12_381_g1_commitments_with_generators, sequence::Sequence};
use ff::PrimeField;
use group::GroupEncoding;

type BlitzarG1Affine = ark_bls12_381::G1Affine;

const DEFAULT_MIN_TOTAL_SCALARS: usize = 1 << 20;
const DEFAULT_SPPARK_MIN_TOTAL_SCALARS: usize = 100_000;
const DEFAULT_SPPARK_BATCH_MIN_DENSITY_PCT: usize = 90;
const MIN_TOTAL_SCALARS_ENV: &str = "MIDNIGHT_GPU_BLITZAR_MIN_SCALARS";
const SPPARK_MIN_TOTAL_SCALARS_ENV: &str = "MIDNIGHT_GPU_SPPARK_MIN_SCALARS";
const SPPARK_BATCH_MIN_DENSITY_PCT_ENV: &str = "MIDNIGHT_GPU_SPPARK_BATCH_MIN_DENSITY_PCT";
const SPPARK_SMALL_BATCH_ENV: &str = "MIDNIGHT_GPU_SPPARK_SMALL_BATCH";
const SPPARK_UNFILTERED_ENV: &str = "MIDNIGHT_GPU_SPPARK_UNFILTERED";
const SPPARK_UNFILTERED_MIN_SCALARS_ENV: &str = "MIDNIGHT_GPU_SPPARK_UNFILTERED_MIN_SCALARS";
const SPPARK_UNFILTERED_MIN_DENSITY_PCT_ENV: &str =
    "MIDNIGHT_GPU_SPPARK_UNFILTERED_MIN_DENSITY_PCT";
const MSM_BACKEND_ENV: &str = "MIDNIGHT_GPU_MSM_BACKEND";

static ENABLED: AtomicBool = AtomicBool::new(false);
thread_local! {
    static DISABLE_DEPTH: Cell<usize> = const { Cell::new(0) };
}
static GENERATOR_CACHE: OnceLock<Mutex<HashMap<GeneratorCacheKey, Arc<Vec<BlitzarG1Affine>>>>> =
    OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GeneratorCacheKey {
    ptr: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MsmBackend {
    Auto,
    Sppark,
    Blitzar,
    Cpu,
}

/// Sets whether GPU-backed KZG MSMs are used by this process.
pub fn set_enabled(enabled: bool) -> bool {
    ENABLED.swap(enabled, Ordering::SeqCst)
}

/// Returns whether GPU-backed KZG MSMs are currently enabled.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed) && DISABLE_DEPTH.get() == 0
}

/// Suppresses all GPU KZG dispatch on the current thread only.
#[derive(Debug)]
pub struct DisableGuard {
    _sppark: midnight_curves::gpu_sppark::DisableGuard,
}

impl DisableGuard {
    /// Disables GPU KZG dispatch until the returned guard is dropped.
    pub fn new() -> Self {
        DISABLE_DEPTH.set(DISABLE_DEPTH.get() + 1);
        Self {
            _sppark: midnight_curves::gpu_sppark::DisableGuard::new(),
        }
    }
}

impl Drop for DisableGuard {
    fn drop(&mut self) {
        DISABLE_DEPTH.set(DISABLE_DEPTH.get().saturating_sub(1));
    }
}

/// Returns whether the current MSM batch is large enough for Blitzar.
///
/// The Blitzar Rust wrapper has a high per-call floor for BLS12-381 G1 MSMs.
/// Use it only when a batch is large enough to plausibly amortize that cost.
/// Set `MIDNIGHT_GPU_BLITZAR_MIN_SCALARS=0` to force Blitzar for profiling.
pub(crate) fn should_accelerate(total_scalars: usize, _sequences: usize) -> bool {
    if !enabled() {
        return false;
    }
    if configured_backend() != MsmBackend::Blitzar {
        return false;
    }

    let min_total_scalars = min_total_scalars(MIN_TOTAL_SCALARS_ENV, DEFAULT_MIN_TOTAL_SCALARS);

    total_scalars >= min_total_scalars
}

/// Returns whether a BLS12-381 MSM should route through the sppark CUDA path.
///
/// `MIDNIGHT_GPU_MSM_BACKEND=sppark` forces sppark selection for eligible MSMs,
/// while `MIDNIGHT_GPU_MSM_BACKEND=blitzar` keeps the legacy Blitzar path.
/// Set `MIDNIGHT_GPU_SPPARK_MIN_SCALARS=0` to force sppark for profiling.
pub(crate) fn should_accelerate_sppark(total_scalars: usize) -> bool {
    if !enabled() {
        return false;
    }
    if !matches!(configured_backend(), MsmBackend::Auto | MsmBackend::Sppark) {
        return false;
    }

    let configured_min_scalars = midnight_sppark_msm::current_msm_config().min_msm_scalars as usize;
    let profile_min_scalars = if configured_min_scalars == 0 {
        DEFAULT_SPPARK_MIN_TOTAL_SCALARS
    } else {
        configured_min_scalars
    };
    let min_total_scalars =
        min_total_scalars(SPPARK_MIN_TOTAL_SCALARS_ENV, profile_min_scalars.max(1));

    total_scalars >= min_total_scalars
}

/// Returns whether a dense BLS12-381 MSM should skip host zero-filtering.
///
/// This preserves the sparse filtered path while avoiding the host allocation
/// and copy for dense scalar vectors that sppark can consume directly.
pub(crate) fn should_accelerate_sppark_dense_unfiltered(
    total_scalars: usize,
    nonzero_scalars: usize,
) -> bool {
    if !should_accelerate_sppark(total_scalars) {
        return false;
    }

    let mode = std::env::var(SPPARK_UNFILTERED_ENV)
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "false" | "no" | "off" | "none") {
        return false;
    }

    let configured_min_scalars =
        midnight_sppark_msm::current_msm_config().unfiltered_min_scalars as usize;
    let profile_min_scalars = if configured_min_scalars == 0 {
        1 << 15
    } else {
        configured_min_scalars
    };
    let min_scalars = min_total_scalars(SPPARK_UNFILTERED_MIN_SCALARS_ENV, profile_min_scalars);
    if total_scalars < min_scalars {
        return false;
    }

    if matches!(
        mode.as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "unfiltered"
    ) {
        return true;
    }

    let min_density_pct = min_total_scalars(SPPARK_UNFILTERED_MIN_DENSITY_PCT_ENV, 30).min(100);
    nonzero_scalars.saturating_mul(100) >= total_scalars.saturating_mul(min_density_pct)
}

/// Returns whether same-base MSMs should be batched through sppark.
///
/// Batched sppark MSM keeps the SRS bases resident across several MSMs, which
/// is profitable for dense polynomial batches. Sparse Lagrange polynomials are
/// better handled by the filtered per-polynomial path, because sending the full
/// domain to the GPU throws away the zero-filtering win.
pub(crate) fn should_accelerate_sppark_batch(
    msm_size: usize,
    total_scalars: usize,
    nonzero_scalars: usize,
    sequences: usize,
) -> bool {
    if sequences < 4 || total_scalars == 0 || !should_accelerate_sppark(msm_size) {
        return false;
    }

    let min_density_pct = min_total_scalars(
        SPPARK_BATCH_MIN_DENSITY_PCT_ENV,
        DEFAULT_SPPARK_BATCH_MIN_DENSITY_PCT,
    )
    .min(100);

    nonzero_scalars.saturating_mul(100) >= total_scalars.saturating_mul(min_density_pct)
}

/// Returns whether same-size batches smaller than the dense-batch threshold
/// should remain on CPU. Set `MIDNIGHT_GPU_SPPARK_SMALL_BATCH=gpu` to profile
/// per-polynomial sppark MSMs without enabling the dense same-base batch path.
pub(crate) fn use_cpu_for_sppark_small_batch() -> bool {
    !matches!(
        std::env::var(SPPARK_SMALL_BATCH_ENV)
            .unwrap_or_else(|_| "cpu".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "gpu" | "cuda" | "sppark" | "per-msm"
    )
}

fn configured_backend() -> MsmBackend {
    match std::env::var(MSM_BACKEND_ENV)
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "auto" => MsmBackend::Auto,
        "sppark" | "cuda" => MsmBackend::Sppark,
        "blitzar" => MsmBackend::Blitzar,
        "cpu" | "off" | "none" => MsmBackend::Cpu,
        _ => MsmBackend::Auto,
    }
}

fn min_total_scalars(env_name: &str, default: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Restores the previous GPU MSM enablement state when dropped.
#[derive(Debug)]
pub struct EnableGuard {
    previous: bool,
    #[cfg(feature = "gpu-blitzar")]
    previous_sppark: bool,
}

impl EnableGuard {
    /// Sets the GPU MSM enablement state for the lifetime of the guard.
    pub fn set(enabled: bool) -> Self {
        Self {
            previous: set_enabled(enabled),
            #[cfg(feature = "gpu-blitzar")]
            previous_sppark: midnight_curves::gpu_sppark::set_enabled(enabled),
        }
    }
}

impl Drop for EnableGuard {
    fn drop(&mut self) {
        set_enabled(self.previous);
        #[cfg(feature = "gpu-blitzar")]
        midnight_curves::gpu_sppark::set_enabled(self.previous_sppark);
    }
}

pub(crate) fn commit<F, G>(scalars: &[F], generators: &[G]) -> Option<G>
where
    F: PrimeField,
    G: GroupEncoding,
{
    commit_batch(&[scalars], generators)?.into_iter().next()
}

pub(crate) fn commit_batch<F, G>(scalar_slices: &[&[F]], generators: &[G]) -> Option<Vec<G>>
where
    F: PrimeField,
    G: GroupEncoding,
{
    if scalar_slices.is_empty() {
        return Some(Vec::new());
    }
    if scalar_slices
        .iter()
        .any(|scalars| scalars.is_empty() || scalars.len() > generators.len())
    {
        return None;
    }

    let repr_len = scalar_slices.iter().find_map(|scalars| {
        scalars
            .first()
            .map(|scalar| scalar.to_repr().as_ref().len())
    })?;
    if repr_len != 32 {
        return None;
    }

    let ark_generators = cached_generators(generators)?;
    let scalar_words = scalar_slices
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|scalar| {
                    let repr = scalar.to_repr();
                    <[u8; 32]>::try_from(repr.as_ref()).ok()
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;

    let sequences = scalar_words
        .iter()
        .map(|words| Sequence::from(words.as_slice()))
        .collect::<Vec<_>>();
    let mut out = vec![[0u8; 48]; scalar_slices.len()];
    compute_bls12_381_g1_commitments_with_generators(
        &mut out,
        &sequences,
        ark_generators.as_slice(),
    );

    out.iter().map(decode_result::<G>).collect()
}

fn cached_generators<G>(generators: &[G]) -> Option<Arc<Vec<BlitzarG1Affine>>>
where
    G: GroupEncoding,
{
    let key = GeneratorCacheKey {
        ptr: generators.as_ptr() as usize,
        len: generators.len(),
    };
    let cache = GENERATOR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(cached) = cache.lock().ok()?.get(&key).cloned() {
        return Some(cached);
    }

    let decoded = Arc::new(
        generators
            .iter()
            .map(decode_generator)
            .collect::<Option<Vec<_>>>()?,
    );
    cache.lock().ok()?.insert(key, decoded.clone());
    Some(decoded)
}

fn decode_generator<G>(generator: &G) -> Option<BlitzarG1Affine>
where
    G: GroupEncoding,
{
    let encoded = generator.to_bytes();
    BlitzarG1Affine::deserialize_compressed(encoded.as_ref()).ok()
}

fn decode_result<G>(bytes: &[u8; 48]) -> Option<G>
where
    G: GroupEncoding,
{
    let mut repr = G::Repr::default();
    if repr.as_ref().len() != bytes.len() {
        return None;
    }
    repr.as_mut().copy_from_slice(bytes);
    Option::<G>::from(G::from_bytes(&repr))
}
