use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    mem::size_of,
    ptr::NonNull,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use ec_gpu_gen::{multiexp::MultiexpKernel, rust_gpu_tools::Device, threadpool::Worker};
use group::Group;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381Fr {
    pub limbs: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381G1Affine {
    pub x: [u64; 6],
    pub y: [u64; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bls12_381G1Projective {
    pub x: [u64; 6],
    pub y: [u64; 6],
    pub z: [u64; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawValueSource {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawCalculation {
    pub op: u32,
    pub target: u32,
    pub parts_offset: u32,
    pub parts_len: u32,
    pub lhs: RawValueSource,
    pub rhs: RawValueSource,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawGraph {
    pub constants: *const Bls12_381Fr,
    pub constants_len: usize,
    pub rotations: *const i32,
    pub rotations_len: usize,
    pub calculations: *const RawCalculation,
    pub calculations_len: usize,
    pub horner_parts: *const RawValueSource,
    pub horner_parts_len: usize,
    pub num_intermediates: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawColumnRef {
    pub kind: u32,
    pub index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawDevicePolyRef {
    pub source: u32,
    pub index: u32,
}

#[derive(Debug)]
pub enum MsmError {
    LengthMismatch { points: usize, scalars: usize },
    NttLengthNotPowerOfTwo { len: usize },
    NttLogSizeOverflow { len: usize },
    Sppark(String),
    EcGpu(String),
    LockPoisoned,
}

impl std::fmt::Display for MsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { points, scalars } => write!(
                f,
                "point/scalar length mismatch: {points} points, {scalars} scalars"
            ),
            Self::NttLengthNotPowerOfTwo { len } => {
                write!(f, "NTT input length must be a power of two, got {len}")
            }
            Self::NttLogSizeOverflow { len } => {
                write!(f, "NTT input length is too large to encode as log2: {len}")
            }
            Self::Sppark(err) => write!(f, "sppark MSM failed: {err}"),
            Self::EcGpu(err) => write!(f, "ec-gpu MSM failed: {err}"),
            Self::LockPoisoned => write!(f, "sppark MSM lock poisoned"),
        }
    }
}

impl std::error::Error for MsmError {}

/// Maximum number of independent CUDA execution lanes owned by concurrent proofs.
pub const MAX_SPPARK_EXECUTION_LANES: usize = 8;

/// Immutable tuning selected when a BLS12-381 MSM context is created.
///
/// Production proofs select an explicit profile from their polynomial-domain k.
/// A zero window size asks sppark to derive it from the point count; diagnostic
/// environment overrides are resolved before the profile enters the lane and
/// context cache keys.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Bls12_381MsmConfig {
    pub domain_k: u32,
    pub min_msm_scalars: u32,
    pub unfiltered_min_scalars: u32,
    pub window_bits: u32,
    pub sort_grid_divisor: u32,
    pub sort_window_lanes: u32,
    pub integration_blocks: u32,
    pub batch_lanes: u32,
    pub reuse_event: u32,
    /// 0 uses upstream host collection, 1 returns GPU-integrated rows for a
    /// narrow host fold, 2 checks all reducers, and 3 folds fully on device.
    pub device_reduce: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bls12_381MsmSlice {
    pub scalar_offset: usize,
    pub point_offset: usize,
    pub len: usize,
    pub nonzero_len: usize,
}

impl Default for Bls12_381MsmConfig {
    fn default() -> Self {
        Self::for_k(0)
    }
}

impl Bls12_381MsmConfig {
    pub const fn for_k(domain_k: u32) -> Self {
        let window_bits = match domain_k {
            1..=15 => 10,
            16..=18 => 12,
            _ => 0,
        };
        let batch_lanes = match domain_k {
            13..=15 => 6,
            _ => 4,
        };
        // Keep the upstream integration topology as the production default.
        // Multi-block integration remains available as a per-k diagnostic
        // override until its scheduling variants pass the transaction stress
        // gate under concurrent proof lanes.
        let integration_blocks = 1;
        let reuse_event = match domain_k {
            13..=18 => 1,
            _ => 0,
        };
        let device_reduce = match domain_k {
            14..=18 => 1,
            _ => 0,
        };
        let sort_window_lanes = 2;
        let min_msm_scalars = match domain_k {
            13..=15 => 1 << 13,
            _ => 100_000,
        };
        let unfiltered_min_scalars = match domain_k {
            13..=15 => 1 << 14,
            _ => 1 << 15,
        };
        Self {
            domain_k,
            min_msm_scalars,
            unfiltered_min_scalars,
            window_bits,
            sort_grid_divisor: 3,
            sort_window_lanes,
            integration_blocks,
            batch_lanes,
            reuse_event,
            device_reduce,
        }
    }

    pub fn for_point_count(point_count: usize) -> Self {
        let domain_k = if point_count <= 1 {
            0
        } else {
            usize::BITS - (point_count - 1).leading_zeros()
        };
        Self::for_k(domain_k)
    }

    pub fn with_diagnostic_overrides(mut self) -> Result<Self, MsmError> {
        fn parse_override(name: &str) -> Result<Option<u32>, MsmError> {
            std::env::var(name)
                .ok()
                .map(|value| {
                    value
                        .parse::<u32>()
                        .map_err(|_| MsmError::Sppark(format!("invalid {name} value {value:?}")))
                })
                .transpose()
        }

        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_WBITS")? {
            self.window_bits = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_GPU_SPPARK_MIN_SCALARS")? {
            self.min_msm_scalars = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_GPU_SPPARK_UNFILTERED_MIN_SCALARS")? {
            self.unfiltered_min_scalars = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_SORT_GRID_DIVISOR")? {
            self.sort_grid_divisor = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_SORT_WINDOW_LANES")? {
            self.sort_window_lanes = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_INTEGRATION_BLOCKS")? {
            self.integration_blocks = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_BATCH_LANES")? {
            self.batch_lanes = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_REUSE_EVENT")? {
            self.reuse_event = value;
        }
        if let Some(value) = parse_override("MIDNIGHT_SPPARK_MSM_DEVICE_REDUCE")? {
            self.device_reduce = value;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(self) -> Result<(), MsmError> {
        if self.domain_k > 31
            || self.min_msm_scalars == 0
            || self.unfiltered_min_scalars == 0
            || (self.window_bits != 0 && !(10..=18).contains(&self.window_bits))
            || !(1..=8).contains(&self.sort_grid_divisor)
            || !(1..=4).contains(&self.sort_window_lanes)
            || !matches!(self.integration_blocks, 1 | 2 | 4 | 8 | 16)
            || !(1..=16).contains(&self.batch_lanes)
            || self.reuse_event > 1
            || self.device_reduce > 3
        {
            return Err(MsmError::Sppark(format!(
                "invalid BLS12-381 MSM config {self:?}"
            )));
        }
        Ok(())
    }
}

thread_local! {
    static CURRENT_EXECUTION_LANE: Cell<u32> = const { Cell::new(0) };
}

#[derive(Default)]
struct ExecutionLanePool {
    in_use: [bool; MAX_SPPARK_EXECUTION_LANES + 1],
    lane_owner: [Option<usize>; MAX_SPPARK_EXECUTION_LANES + 1],
    lane_msm_config: [Bls12_381MsmConfig; MAX_SPPARK_EXECUTION_LANES + 1],
    affinity: HashMap<usize, Vec<usize>>,
}

fn execution_lane_pool() -> &'static (Mutex<ExecutionLanePool>, Condvar) {
    static POOL: OnceLock<(Mutex<ExecutionLanePool>, Condvar)> = OnceLock::new();
    POOL.get_or_init(|| (Mutex::new(ExecutionLanePool::default()), Condvar::new()))
}

fn configured_execution_lanes() -> usize {
    std::env::var("MIDNIGHT_GPU_SPPARK_PROOF_LANES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .clamp(1, MAX_SPPARK_EXECUTION_LANES)
}

/// Returns the CUDA execution lane owned by the current proof thread.
pub fn current_execution_lane() -> u32 {
    CURRENT_EXECUTION_LANE.get()
}

/// Returns the immutable MSM profile owned by the current proof lane.
pub fn current_msm_config() -> Bls12_381MsmConfig {
    let lane = current_execution_lane() as usize;
    if lane == 0 || lane > MAX_SPPARK_EXECUTION_LANES {
        return Bls12_381MsmConfig::default();
    }
    let (pool_lock, _) = execution_lane_pool();
    let pool = pool_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pool.lane_msm_config[lane]
}

/// Owns one private sppark CUDA context for the lifetime of a synchronous proof.
///
/// The guard is intentionally thread-bound: CUDA calls made by the proof must
/// stay on the thread that acquired the lane.
pub struct SpparkExecutionGuard {
    lane: u32,
    owns_lane: bool,
    _not_send: PhantomData<Rc<()>>,
}

/// Temporarily propagates an already-owned proof lane to a scoped worker thread.
pub struct SpparkExecutionBinding {
    previous_lane: u32,
    changed: bool,
    _not_send: PhantomData<Rc<()>>,
}

/// Binds a scoped worker to an execution lane already owned by its parent proof.
/// Calls on the shared lane remain serialized by that lane's lock.
pub fn bind_execution_lane(lane: u32) -> Result<SpparkExecutionBinding, MsmError> {
    if lane == 0 || lane as usize > MAX_SPPARK_EXECUTION_LANES {
        return Err(MsmError::Sppark(format!(
            "invalid sppark proof execution lane {lane}"
        )));
    }
    let previous_lane = current_execution_lane();
    if previous_lane == lane {
        return Ok(SpparkExecutionBinding {
            previous_lane,
            changed: false,
            _not_send: PhantomData,
        });
    }
    let status = unsafe { midnight_sppark_execution_lane_set(lane) };
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    CURRENT_EXECUTION_LANE.set(lane);
    Ok(SpparkExecutionBinding {
        previous_lane,
        changed: true,
        _not_send: PhantomData,
    })
}

impl Drop for SpparkExecutionBinding {
    fn drop(&mut self) {
        if !self.changed {
            return;
        }
        let status = unsafe { midnight_sppark_execution_lane_set(self.previous_lane) };
        CURRENT_EXECUTION_LANE.set(self.previous_lane);
        debug_assert_eq!(status.code, 0, "failed to restore the prior sppark lane");
    }
}

impl SpparkExecutionGuard {
    pub fn acquire() -> Result<Self, MsmError> {
        Self::acquire_with_affinity(0)
    }

    /// Acquires a lane while preferring lanes previously used by `affinity_key`.
    pub fn acquire_with_affinity(affinity_key: usize) -> Result<Self, MsmError> {
        Self::acquire_with_affinity_and_config(affinity_key, Bls12_381MsmConfig::default())
    }

    /// Acquires a proof lane and pins an immutable MSM profile to that lane.
    pub fn acquire_with_affinity_and_config(
        affinity_key: usize,
        config: Bls12_381MsmConfig,
    ) -> Result<Self, MsmError> {
        config.validate()?;
        let current = current_execution_lane();
        if current != 0 {
            let active = current_msm_config();
            if config != Bls12_381MsmConfig::default() && config != active {
                return Err(MsmError::Sppark(format!(
                    "nested proof requested MSM config {config:?}, lane {current} owns {active:?}"
                )));
            }
            return Ok(Self {
                lane: current,
                owns_lane: false,
                _not_send: PhantomData,
            });
        }

        let lane_count = configured_execution_lanes();
        let (pool_lock, wake) = execution_lane_pool();
        let mut pool = pool_lock.lock().map_err(|_| MsmError::LockPoisoned)?;
        let lane = loop {
            let lane = if affinity_key == 0 {
                (1..=lane_count).find(|lane| !pool.in_use[*lane])
            } else {
                let preferred = pool
                    .affinity
                    .get(&affinity_key)
                    .cloned()
                    .unwrap_or_default();
                preferred
                    .into_iter()
                    .find(|lane| *lane <= lane_count && !pool.in_use[*lane])
                    .or_else(|| {
                        (1..=lane_count)
                            .find(|lane| !pool.in_use[*lane] && pool.lane_owner[*lane].is_none())
                    })
                    .or_else(|| (1..=lane_count).find(|lane| !pool.in_use[*lane]))
            };
            if let Some(lane) = lane {
                pool.in_use[lane] = true;
                pool.lane_msm_config[lane] = config;
                if affinity_key != 0 {
                    if pool.lane_owner[lane].is_none() {
                        pool.lane_owner[lane] = Some(affinity_key);
                    }
                    let preferred = pool.affinity.entry(affinity_key).or_default();
                    if !preferred.contains(&lane) {
                        preferred.push(lane);
                    }
                }
                break lane as u32;
            }
            pool = wake.wait(pool).map_err(|_| MsmError::LockPoisoned)?;
        };
        drop(pool);

        let status = unsafe { midnight_sppark_execution_lane_set(lane) };
        if status.code != 0 {
            let mut pool = pool_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pool.in_use[lane as usize] = false;
            wake.notify_one();
            return Err(MsmError::Sppark(String::from(status)));
        }
        CURRENT_EXECUTION_LANE.set(lane);

        Ok(Self {
            lane,
            owns_lane: true,
            _not_send: PhantomData,
        })
    }
}

impl Drop for SpparkExecutionGuard {
    fn drop(&mut self) {
        if !self.owns_lane {
            return;
        }

        let status = unsafe { midnight_sppark_execution_lane_set(0) };
        CURRENT_EXECUTION_LANE.set(0);
        debug_assert_eq!(status.code, 0, "failed to restore the default sppark lane");

        let (pool_lock, wake) = execution_lane_pool();
        let mut pool = pool_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pool.in_use[self.lane as usize] = false;
        wake.notify_one();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransferCounters {
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransferBucketCounters {
    pub name: &'static str,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
enum TransferOp {
    MsmContextCreate,
    MsmContextBatch,
    Msm,
    MsmBatch,
    FrNtt,
    FrMulAssign,
    FrMulAssignZeta,
    FrMulAssignRepeating,
    FrShiftedPrefixProduct,
    FrPolyLincomb,
    FrInvertAssign,
    EvalNumerator,
    FrCoeffToExtendedBatch,
    FrDeviceVecUpload,
    FrDeviceVecDownload,
    FrDeviceVecNtt,
    FrDeviceCoeffToExtended,
    FrDeviceDivideNtt,
    NumeratorContextCreate,
    NumeratorContextEval,
    MsmContextDeviceBatch,
    FrPermutationProducts,
    EvalGraphBatch,
    EcGpuMsm,
    FrLookupProducts,
    FrPoseidonHashChain,
    FrPoseidonHashChainWitness,
    FrMultiopenPrepare,
    FrSortCanonical,
    TraceContextTrash,
    TraceContextPermutation,
    TraceLookupProducts,
    TraceLookupPermute,
}

impl TransferOp {
    fn index(self) -> usize {
        match self {
            Self::MsmContextCreate => 0,
            Self::MsmContextBatch => 1,
            Self::Msm => 2,
            Self::MsmBatch => 3,
            Self::FrNtt => 4,
            Self::FrMulAssign => 5,
            Self::FrMulAssignZeta => 6,
            Self::FrMulAssignRepeating => 7,
            Self::FrShiftedPrefixProduct => 8,
            Self::FrPolyLincomb => 9,
            Self::FrInvertAssign => 10,
            Self::EvalNumerator => 11,
            Self::FrCoeffToExtendedBatch => 12,
            Self::FrDeviceVecUpload => 13,
            Self::FrDeviceVecDownload => 14,
            Self::FrDeviceVecNtt => 15,
            Self::FrDeviceCoeffToExtended => 16,
            Self::FrDeviceDivideNtt => 17,
            Self::NumeratorContextCreate => 18,
            Self::NumeratorContextEval => 19,
            Self::MsmContextDeviceBatch => 20,
            Self::FrPermutationProducts => 21,
            Self::EvalGraphBatch => 22,
            Self::EcGpuMsm => 23,
            Self::FrLookupProducts => 24,
            Self::FrPoseidonHashChain => 25,
            Self::FrPoseidonHashChainWitness => 26,
            Self::FrMultiopenPrepare => 27,
            Self::FrSortCanonical => 28,
            Self::TraceContextTrash => 29,
            Self::TraceContextPermutation => 30,
            Self::TraceLookupProducts => 31,
            Self::TraceLookupPermute => 32,
        }
    }
}

const TRANSFER_OP_NAMES: [&str; 33] = [
    "msm_context_create",
    "msm_context_batch",
    "msm",
    "msm_batch",
    "fr_ntt",
    "fr_mul_assign",
    "fr_mul_assign_zeta",
    "fr_mul_assign_repeating",
    "fr_shifted_prefix_product",
    "fr_poly_lincomb",
    "fr_invert_assign",
    "eval_numerator",
    "fr_coeff_to_extended_batch",
    "fr_device_vec_upload",
    "fr_device_vec_download",
    "fr_device_vec_ntt",
    "fr_device_coeff_to_extended",
    "fr_device_divide_ntt",
    "numerator_context_create",
    "numerator_context_eval",
    "msm_context_device_batch",
    "fr_permutation_products",
    "eval_graph_batch",
    "ec_gpu_msm",
    "fr_lookup_products",
    "fr_poseidon_hash_chain",
    "fr_poseidon_hash_chain_witness",
    "fr_multiopen_prepare",
    "fr_sort_canonical",
    "trace_context_trash",
    "trace_context_permutation",
    "trace_lookup_products",
    "trace_lookup_permute",
];

struct TransferBucketState {
    h2d: AtomicU64,
    d2h: AtomicU64,
}

static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn reset_transfer_counters() {
    H2D_BYTES.store(0, Ordering::Relaxed);
    D2H_BYTES.store(0, Ordering::Relaxed);
    for bucket in transfer_bucket_state() {
        bucket.h2d.store(0, Ordering::Relaxed);
        bucket.d2h.store(0, Ordering::Relaxed);
    }
}

pub fn transfer_counters() -> TransferCounters {
    TransferCounters {
        h2d_bytes: H2D_BYTES.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
    }
}

pub fn transfer_breakdown() -> Vec<TransferBucketCounters> {
    transfer_bucket_state()
        .iter()
        .enumerate()
        .map(|(index, bucket)| TransferBucketCounters {
            name: TRANSFER_OP_NAMES[index],
            h2d_bytes: bucket.h2d.load(Ordering::Relaxed),
            d2h_bytes: bucket.d2h.load(Ordering::Relaxed),
        })
        .collect()
}

fn record_h2d(bytes: usize) {
    H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_d2h(bytes: usize) {
    D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_transfer(op: TransferOp, h2d_bytes: usize, d2h_bytes: usize) {
    record_h2d(h2d_bytes);
    record_d2h(d2h_bytes);
    let bucket = &transfer_bucket_state()[op.index()];
    bucket.h2d.fetch_add(h2d_bytes as u64, Ordering::Relaxed);
    bucket.d2h.fetch_add(d2h_bytes as u64, Ordering::Relaxed);
}

fn slice_bytes<T>(len: usize) -> usize {
    len.saturating_mul(size_of::<T>())
}

fn raw_graph_h2d_bytes(graph: RawGraph) -> usize {
    slice_bytes::<Bls12_381Fr>(graph.constants_len)
        + slice_bytes::<i32>(graph.rotations_len)
        + slice_bytes::<RawCalculation>(graph.calculations_len)
        + slice_bytes::<RawValueSource>(graph.horner_parts_len)
}

fn transfer_bucket_state() -> &'static Vec<TransferBucketState> {
    static BUCKETS: OnceLock<Vec<TransferBucketState>> = OnceLock::new();
    BUCKETS.get_or_init(|| {
        TRANSFER_OP_NAMES
            .iter()
            .map(|_| TransferBucketState {
                h2d: AtomicU64::new(0),
                d2h: AtomicU64::new(0),
            })
            .collect()
    })
}

extern "C" {
    fn midnight_sppark_execution_lane_set(lane: u32) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm(
        out: *mut Bls12_381G1Projective,
        points: *const Bls12_381G1Affine,
        scalars: *const Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_batch(
        out: *mut Bls12_381G1Projective,
        points: *const Bls12_381G1Affine,
        scalars: *const Bls12_381Fr,
        npoints: usize,
        nmsms: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_context_create_with_config(
        out: *mut *mut std::ffi::c_void,
        points: *const Bls12_381G1Affine,
        npoints: usize,
        config: Bls12_381MsmConfig,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_context_free(context: *mut std::ffi::c_void);

    fn midnight_sppark_bls12_381_g1_msm_context_batch(
        out: *mut Bls12_381G1Projective,
        context: *mut std::ffi::c_void,
        scalars: *const Bls12_381Fr,
        npoints: usize,
        nmsms: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_context_device_batch(
        out: *mut Bls12_381G1Projective,
        context: *mut std::ffi::c_void,
        device_vec: *const std::ffi::c_void,
        scalar_offset: usize,
        npoints: usize,
        nmsms: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_context_device_ragged(
        out: *mut Bls12_381G1Projective,
        context: *mut std::ffi::c_void,
        device_vec: *const std::ffi::c_void,
        slices: *const Bls12_381MsmSlice,
        count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_g1_msm_context_device_scrambled_batch(
        out: *mut Bls12_381G1Projective,
        context: *mut std::ffi::c_void,
        device_vec: *const std::ffi::c_void,
        scalar_offset: usize,
        npoints: usize,
        nmsms: usize,
        mask_device_vec: *const std::ffi::c_void,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_ntt(
        values: *mut Bls12_381Fr,
        log_n: u32,
        inverse: bool,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_coeff_to_extended_batch(
        out: *mut Bls12_381Fr,
        inputs: *const Bls12_381Fr,
        input_len: usize,
        count: usize,
        extended_log_n: u32,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_upload(
        out: *mut *mut std::ffi::c_void,
        values: *const Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_upload_many(
        out: *mut *mut std::ffi::c_void,
        values: *const *const Bls12_381Fr,
        lengths: *const usize,
        count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_poseidon_advice_from_witness(
        out: *mut *mut std::ffi::c_void,
        witness: *const Bls12_381Fr,
        hash_count: usize,
        domain_len: usize,
        metadata: *const Bls12_381Fr,
        metadata_len: usize,
        layout: *const u16,
        layout_len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_scatter_host(
        device_vec: *mut std::ffi::c_void,
        indices: *const u32,
        values: *const Bls12_381Fr,
        count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_concat_host(
        out: *mut *mut std::ffi::c_void,
        prefix_device_vec: *const std::ffi::c_void,
        suffix: *const Bls12_381Fr,
        suffix_len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_concat_many_host(
        out: *mut *mut std::ffi::c_void,
        source_device_vecs: *const *const std::ffi::c_void,
        source_count: usize,
        suffix: *const Bls12_381Fr,
        suffix_len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_gather_polys(
        out: *mut *mut std::ffi::c_void,
        source_device_vec: *const std::ffi::c_void,
        poly_len: usize,
        indices: *const u32,
        index_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_coeff_to_extended_batch(
        out: *mut *mut std::ffi::c_void,
        inputs: *const Bls12_381Fr,
        input_len: usize,
        count: usize,
        extended_log_n: u32,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_coeff_to_extended_from_device_batch(
        out: *mut *mut std::ffi::c_void,
        source: *const std::ffi::c_void,
        input_len: usize,
        count: usize,
        extended_log_n: u32,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_download(
        out: *mut Bls12_381Fr,
        device_vec: *const std::ffi::c_void,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_download_nonzero_suffixes(
        out: *mut Bls12_381Fr,
        out_capacity: usize,
        out_len: *mut usize,
        starts: *mut usize,
        nonzero_counts: *mut usize,
        device_vec: *const std::ffi::c_void,
        scalar_offset: usize,
        poly_len: usize,
        count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_quotient_pieces(
        out: *mut *mut std::ffi::c_void,
        quotient_vec: *const std::ffi::c_void,
        poly_len: usize,
        piece_count: usize,
        blinds: *const Bls12_381Fr,
        blind_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_poly_lincomb(
        out: *mut *mut std::ffi::c_void,
        polys_vec: *const std::ffi::c_void,
        poly_len: usize,
        poly_count: usize,
        scalars: *const Bls12_381Fr,
        scalar_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_ntt_zeta(
        device_vec: *mut std::ffi::c_void,
        log_n: u32,
        inverse: bool,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_ntt_zeta_batch(
        device_vec: *mut std::ffi::c_void,
        log_n: u32,
        count: usize,
        inverse: bool,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_divide_ntt_zeta(
        device_vec: *mut std::ffi::c_void,
        log_n: u32,
        factors: *const Bls12_381Fr,
        factor_len: usize,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_vec_free(device_vec: *mut std::ffi::c_void);

    fn midnight_sppark_bls12_381_fr_ntt_zeta(
        values: *mut Bls12_381Fr,
        log_n: u32,
        inverse: bool,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_divide_ntt_zeta(
        values: *mut Bls12_381Fr,
        log_n: u32,
        factors: *const Bls12_381Fr,
        factor_len: usize,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_mul_assign(
        values: *mut Bls12_381Fr,
        scalar: Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_mul_assign_zeta(
        values: *mut Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_mul_assign_repeating(
        values: *mut Bls12_381Fr,
        factors: *const Bls12_381Fr,
        factor_len: usize,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_poly_lincomb(
        out: *mut Bls12_381Fr,
        polys: *const Bls12_381Fr,
        poly_len: usize,
        poly_count: usize,
        scalars: *const Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_poseidon_hash_chain(
        out: *mut Bls12_381Fr,
        len: usize,
        initial: Bls12_381Fr,
        tail0: Bls12_381Fr,
        tail1: Bls12_381Fr,
        input_len: Bls12_381Fr,
        mds: *const Bls12_381Fr,
        mds_len: usize,
        round_constants: *const Bls12_381Fr,
        round_constants_len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_poseidon_hash_chain_witness(
        out: *mut Bls12_381Fr,
        len: usize,
        initial: Bls12_381Fr,
        tail0: Bls12_381Fr,
        tail1: Bls12_381Fr,
        input_len: Bls12_381Fr,
        mds: *const Bls12_381Fr,
        mds_len: usize,
        round_constants: *const Bls12_381Fr,
        round_constants_len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_multiopen_prepare(
        q_out: *mut Bls12_381Fr,
        eval_out: *mut Bls12_381Fr,
        polys: *const Bls12_381Fr,
        poly_len: usize,
        poly_count: usize,
        q_offsets: *const u32,
        q_offsets_len: usize,
        q_poly_indices: *const u32,
        q_coefficients: *const Bls12_381Fr,
        q_items_len: usize,
        points: *const Bls12_381Fr,
        point_count: usize,
        query_poly_indices: *const u32,
        query_point_indices: *const u32,
        query_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_multiopen_prepare_device(
        q_out: *mut Bls12_381Fr,
        q_device_out: *mut *mut std::ffi::c_void,
        eval_out: *mut Bls12_381Fr,
        dynamic_source: *const std::ffi::c_void,
        fixed_source: *const std::ffi::c_void,
        extra_source: *const std::ffi::c_void,
        poly_len: usize,
        poly_refs: *const RawDevicePolyRef,
        poly_count: usize,
        q_offsets: *const u32,
        q_offsets_len: usize,
        q_poly_indices: *const u32,
        q_coefficients: *const Bls12_381Fr,
        q_items_len: usize,
        points: *const Bls12_381Fr,
        point_count: usize,
        query_poly_indices: *const u32,
        query_point_indices: *const u32,
        query_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_multiopen_f_poly_device(
        out: *mut *mut std::ffi::c_void,
        q_device_vec: *const std::ffi::c_void,
        poly_len: usize,
        point_offsets: *const u32,
        point_offsets_len: usize,
        points: *const Bls12_381Fr,
        point_count: usize,
        combination_scalars: *const Bls12_381Fr,
        scalar_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_device_eval_polys(
        out: *mut Bls12_381Fr,
        device_vec: *const std::ffi::c_void,
        poly_len: usize,
        poly_count: usize,
        point: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_multiopen_pi_device(
        out: *mut *mut std::ffi::c_void,
        q_device_vec: *const std::ffi::c_void,
        f_device_vec: *const std::ffi::c_void,
        poly_len: usize,
        combination_scalars: *const Bls12_381Fr,
        scalar_count: usize,
        point: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_invert_assign(
        values: *mut Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_sort_canonical(
        values: *mut Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_shifted_prefix_product(
        values: *mut Bls12_381Fr,
        initial: Bls12_381Fr,
        len: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_permutation_products(
        out: *mut *mut std::ffi::c_void,
        columns: *const Bls12_381Fr,
        permutations: *const Bls12_381Fr,
        len: usize,
        column_count: usize,
        chunk_len: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        omega: Bls12_381Fr,
        delta_powers: *const Bls12_381Fr,
        blind_values: *const Bls12_381Fr,
        blinding_factors: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_lookup_products(
        out: *mut *mut std::ffi::c_void,
        compressed_inputs: *const Bls12_381Fr,
        compressed_tables: *const Bls12_381Fr,
        permuted_inputs: *const Bls12_381Fr,
        permuted_tables: *const Bls12_381Fr,
        len: usize,
        lookup_count: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        blind_values: *const Bls12_381Fr,
        blinding_factors: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_lookup_products_device(
        out: *mut *mut std::ffi::c_void,
        trace_lagrange: *const std::ffi::c_void,
        len: usize,
        lookup_count: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        blind_values: *const Bls12_381Fr,
        blinding_factors: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_fr_lookup_permute_batch(
        out: *mut *mut std::ffi::c_void,
        compressed_inputs: *const Bls12_381Fr,
        compressed_tables: *const Bls12_381Fr,
        len: usize,
        lookup_count: usize,
        usable_rows: usize,
        input_blinds: *const Bls12_381Fr,
        table_blinds: *const Bls12_381Fr,
        blind_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_eval_numerator(
        out: *mut Bls12_381Fr,
        len: usize,
        initial_out: *const Bls12_381Fr,
        run_custom: u32,
        fixed: *const Bls12_381Fr,
        fixed_cols: usize,
        advice: *const Bls12_381Fr,
        advice_cols: usize,
        instance: *const Bls12_381Fr,
        instance_cols: usize,
        challenges: *const Bls12_381Fr,
        challenge_len: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        theta: Bls12_381Fr,
        trash_challenge: Bls12_381Fr,
        y: Bls12_381Fr,
        omega: Bls12_381Fr,
        delta_start: Bls12_381Fr,
        delta_const: Bls12_381Fr,
        rot_scale: i32,
        last_rotation: i32,
        chunk_len: u32,
        l0: *const Bls12_381Fr,
        l_last: *const Bls12_381Fr,
        l_active: *const Bls12_381Fr,
        permutation_products: *const Bls12_381Fr,
        permutation_sets: usize,
        permutation_columns: *const RawColumnRef,
        permutation_columns_len: usize,
        permutation_pk_cosets: *const Bls12_381Fr,
        permutation_pk_cosets_len: usize,
        lookup_product_cosets: *const Bls12_381Fr,
        lookup_input_cosets: *const Bls12_381Fr,
        lookup_table_cosets: *const Bls12_381Fr,
        lookup_count: usize,
        trash_cosets: *const Bls12_381Fr,
        trash_selector_fixed_indices: *const u32,
        trash_count: usize,
        custom_graph: RawGraph,
        lookup_graphs: *const RawGraph,
        trash_graphs: *const RawGraph,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_eval_graph_batch(
        out: *mut Bls12_381Fr,
        len: usize,
        fixed: *const Bls12_381Fr,
        fixed_cols: usize,
        advice: *const Bls12_381Fr,
        advice_cols: usize,
        instance: *const Bls12_381Fr,
        instance_cols: usize,
        challenges: *const Bls12_381Fr,
        challenge_len: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        theta: Bls12_381Fr,
        trash_challenge: Bls12_381Fr,
        y: Bls12_381Fr,
        rot_scale: i32,
        graphs: *const RawGraph,
        graph_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_numerator_context_create(
        out: *mut *mut std::ffi::c_void,
        len: usize,
        input_len: usize,
        fixed: *const Bls12_381Fr,
        fixed_lagrange: *const Bls12_381Fr,
        fixed_cols: usize,
        l0: *const Bls12_381Fr,
        l_last: *const Bls12_381Fr,
        l_active: *const Bls12_381Fr,
        permutation_columns: *const RawColumnRef,
        permutation_columns_len: usize,
        permutation_pk_cosets: *const Bls12_381Fr,
        permutation_pk_lagrange: *const Bls12_381Fr,
        permutation_pk_cosets_len: usize,
        trash_selector_fixed_indices: *const u32,
        trash_count: usize,
        custom_graph: RawGraph,
        lookup_graphs: *const RawGraph,
        lookup_input_graphs: *const RawGraph,
        lookup_table_graphs: *const RawGraph,
        lookup_count: usize,
        trash_graphs: *const RawGraph,
        omega: Bls12_381Fr,
        trace_omega: Bls12_381Fr,
        g_coset: Bls12_381Fr,
        delta_const: Bls12_381Fr,
        rot_scale: i32,
        last_rotation: i32,
        chunk_len: u32,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_numerator_context_free(context: *mut std::ffi::c_void);

    fn midnight_sppark_bls12_381_numerator_context_eval(
        out: *mut *mut std::ffi::c_void,
        context: *const std::ffi::c_void,
        columns: *const std::ffi::c_void,
        total_columns: usize,
        advice_offset: usize,
        advice_cols: usize,
        instance_offset: usize,
        instance_cols: usize,
        challenges: *const Bls12_381Fr,
        challenge_len: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        theta: Bls12_381Fr,
        trash_challenge: Bls12_381Fr,
        y: Bls12_381Fr,
        permutation_offset: usize,
        permutation_sets: usize,
        lookup_product_offset: usize,
        lookup_input_offset: usize,
        lookup_table_offset: usize,
        lookup_count: usize,
        trash_offset: usize,
        trash_count: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_trace_context_trash(
        out: *mut *mut std::ffi::c_void,
        context: *const std::ffi::c_void,
        advice: *const std::ffi::c_void,
        advice_cols: usize,
        instance: *const std::ffi::c_void,
        instance_cols: usize,
        challenges: *const Bls12_381Fr,
        challenge_len: usize,
        trash_challenge: Bls12_381Fr,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_trace_context_permutation(
        out: *mut *mut std::ffi::c_void,
        context: *const std::ffi::c_void,
        advice: *const std::ffi::c_void,
        advice_cols: usize,
        instance: *const std::ffi::c_void,
        instance_cols: usize,
        beta: Bls12_381Fr,
        gamma: Bls12_381Fr,
        delta_powers: *const Bls12_381Fr,
        delta_powers_len: usize,
        blind_values: *const Bls12_381Fr,
        blind_values_len: usize,
        blinding_factors: usize,
    ) -> sppark::Error;

    fn midnight_sppark_bls12_381_trace_context_lookup_permute(
        out: *mut *mut std::ffi::c_void,
        context: *const std::ffi::c_void,
        advice: *const std::ffi::c_void,
        advice_cols: usize,
        instance: *const std::ffi::c_void,
        instance_cols: usize,
        challenges: *const Bls12_381Fr,
        challenge_len: usize,
        theta: Bls12_381Fr,
        usable_rows: usize,
        input_blinds: *const Bls12_381Fr,
        table_blinds: *const Bls12_381Fr,
        blind_count: usize,
    ) -> sppark::Error;
}

struct EcGpuMsmRuntime {
    kernel: MultiexpKernel<'static, blstrs::G1Affine>,
    worker: Worker,
    result_download_bytes: usize,
}

impl EcGpuMsmRuntime {
    fn create() -> Result<Self, String> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err("no CUDA device available to ec-gpu".to_string());
        }
        let programs = devices
            .iter()
            .map(|device| ec_gpu_gen::program!(device))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let result_download_bytes = devices
            .iter()
            .map(|device| {
                let multiplier = match device.compute_capability() {
                    Some((8, _)) => 2,
                    _ => 1,
                };
                128usize
                    .saturating_mul(device.compute_units() as usize)
                    .saturating_mul(multiplier)
                    .saturating_mul(size_of::<blstrs::G1Projective>())
            })
            .sum();
        let kernel =
            MultiexpKernel::create(programs, &devices).map_err(|error| error.to_string())?;
        Ok(Self {
            kernel,
            worker: Worker::new(),
            result_download_bytes,
        })
    }
}

static EC_GPU_MSM_RUNTIME: OnceLock<Result<Mutex<EcGpuMsmRuntime>, String>> = OnceLock::new();

fn ec_gpu_msm_runtime() -> Result<&'static Mutex<EcGpuMsmRuntime>, MsmError> {
    match EC_GPU_MSM_RUNTIME.get_or_init(|| EcGpuMsmRuntime::create().map(Mutex::new)) {
        Ok(runtime) => Ok(runtime),
        Err(error) => Err(MsmError::EcGpu(error.clone())),
    }
}

fn ec_gpu_affine(point: &Bls12_381G1Affine) -> blstrs::G1Affine {
    let mut out = blstrs::G1Affine::default();
    let raw = out.as_mut();
    raw.x.l = point.x;
    raw.y.l = point.y;
    out
}

fn from_ec_gpu_projective(point: blstrs::G1Projective) -> Bls12_381G1Projective {
    let raw: &blst::blst_p1 = point.as_ref();
    Bls12_381G1Projective {
        x: raw.x.l,
        y: raw.y.l,
        z: raw.z.l,
    }
}

fn to_ec_gpu_projective(point: Bls12_381G1Projective) -> blstrs::G1Projective {
    let mut out = blstrs::G1Projective::identity();
    let raw: &mut blst::blst_p1 = out.as_mut();
    raw.x.l = point.x;
    raw.y.l = point.y;
    raw.z.l = point.z;
    out
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn deterministic_msm_scramble_mask(len: usize) -> Vec<Bls12_381Fr> {
    let mut state = 0x4d49_444e_4947_4854_u64 ^ (len as u64).rotate_left(17);
    (0..len)
        .map(|_| {
            loop {
                let limbs = [
                    splitmix64(&mut state),
                    splitmix64(&mut state),
                    splitmix64(&mut state),
                    splitmix64(&mut state) & 0x7fff_ffff_ffff_ffff,
                ];
                if let Some(scalar) =
                    Option::<blstrs::Scalar>::from(blstrs::Scalar::from_u64s_le(&limbs))
                {
                    let raw: blst::blst_fr = scalar.into();
                    break Bls12_381Fr { limbs: raw.l };
                }
            }
        })
        .collect()
}

#[derive(Debug)]
pub struct Bls12_381G1MsmContext {
    raw: NonNull<std::ffi::c_void>,
    npoints: usize,
    config: Bls12_381MsmConfig,
    ec_gpu_bases: Arc<Vec<blstrs::G1Affine>>,
    sppark_scramble_mask: Bls12_381FrDeviceVec,
    sppark_scramble_correction: blstrs::G1Projective,
}

unsafe impl Send for Bls12_381G1MsmContext {}
unsafe impl Sync for Bls12_381G1MsmContext {}

impl Bls12_381G1MsmContext {
    pub fn config(&self) -> Bls12_381MsmConfig {
        self.config
    }
}

#[derive(Debug)]
pub struct Bls12_381FrDeviceVec {
    raw: NonNull<std::ffi::c_void>,
    len: usize,
}

unsafe impl Send for Bls12_381FrDeviceVec {}
unsafe impl Sync for Bls12_381FrDeviceVec {}

#[derive(Debug)]
pub struct Bls12_381NumeratorContext {
    raw: NonNull<std::ffi::c_void>,
    len: usize,
    input_len: usize,
    permutation_column_count: usize,
    permutation_chunk_len: usize,
    lookup_count: usize,
    trash_count: usize,
}

unsafe impl Send for Bls12_381NumeratorContext {}
unsafe impl Sync for Bls12_381NumeratorContext {}

pub struct Bls12_381NumeratorContextInputs<'a> {
    pub len: usize,
    pub input_len: usize,
    pub fixed: &'a [Bls12_381Fr],
    pub fixed_lagrange: &'a [Bls12_381Fr],
    pub fixed_cols: usize,
    pub l0: &'a [Bls12_381Fr],
    pub l_last: &'a [Bls12_381Fr],
    pub l_active: &'a [Bls12_381Fr],
    pub permutation_columns: &'a [RawColumnRef],
    pub permutation_pk_cosets: &'a [Bls12_381Fr],
    pub permutation_pk_lagrange: &'a [Bls12_381Fr],
    pub permutation_pk_cosets_len: usize,
    pub trash_selector_fixed_indices: &'a [u32],
    pub custom_graph: RawGraph,
    pub lookup_graphs: &'a [RawGraph],
    pub lookup_input_graphs: &'a [RawGraph],
    pub lookup_table_graphs: &'a [RawGraph],
    pub trash_graphs: &'a [RawGraph],
    pub omega: Bls12_381Fr,
    pub trace_omega: Bls12_381Fr,
    pub g_coset: Bls12_381Fr,
    pub delta_const: Bls12_381Fr,
    pub rot_scale: i32,
    pub last_rotation: i32,
    pub chunk_len: u32,
}

pub struct Bls12_381TracePermutationInputs<'a> {
    pub advice: &'a Bls12_381FrDeviceVec,
    pub advice_cols: usize,
    pub instance: &'a Bls12_381FrDeviceVec,
    pub instance_cols: usize,
    pub beta: Bls12_381Fr,
    pub gamma: Bls12_381Fr,
    pub delta_powers: &'a [Bls12_381Fr],
    pub blind_values: &'a [Bls12_381Fr],
    pub blinding_factors: usize,
}

pub struct Bls12_381TraceLookupInputs<'a> {
    pub advice: &'a Bls12_381FrDeviceVec,
    pub advice_cols: usize,
    pub instance: &'a Bls12_381FrDeviceVec,
    pub instance_cols: usize,
    pub challenges: &'a [Bls12_381Fr],
    pub theta: Bls12_381Fr,
    pub usable_rows: usize,
    pub input_blinds: &'a [Bls12_381Fr],
    pub table_blinds: &'a [Bls12_381Fr],
}

pub struct Bls12_381TraceTrashInputs<'a> {
    pub advice: &'a Bls12_381FrDeviceVec,
    pub advice_cols: usize,
    pub instance: &'a Bls12_381FrDeviceVec,
    pub instance_cols: usize,
    pub challenges: &'a [Bls12_381Fr],
    pub trash_challenge: Bls12_381Fr,
}

pub struct Bls12_381NumeratorEvalInputs<'a> {
    pub columns: Option<&'a Bls12_381FrDeviceVec>,
    pub total_columns: usize,
    pub advice_offset: usize,
    pub advice_cols: usize,
    pub instance_offset: usize,
    pub instance_cols: usize,
    pub challenges: &'a [Bls12_381Fr],
    pub beta: Bls12_381Fr,
    pub gamma: Bls12_381Fr,
    pub theta: Bls12_381Fr,
    pub trash_challenge: Bls12_381Fr,
    pub y: Bls12_381Fr,
    pub permutation_offset: usize,
    pub permutation_sets: usize,
    pub lookup_product_offset: usize,
    pub lookup_input_offset: usize,
    pub lookup_table_offset: usize,
    pub trash_offset: usize,
}

impl Drop for Bls12_381G1MsmContext {
    fn drop(&mut self) {
        if let Ok(_lock) = maybe_gpu_msm_lock() {
            unsafe { midnight_sppark_bls12_381_g1_msm_context_free(self.raw.as_ptr()) };
        }
    }
}

impl Drop for Bls12_381FrDeviceVec {
    fn drop(&mut self) {
        if let Ok(_lock) = maybe_gpu_msm_lock() {
            unsafe { midnight_sppark_bls12_381_fr_device_vec_free(self.raw.as_ptr()) };
        }
    }
}

impl Drop for Bls12_381NumeratorContext {
    fn drop(&mut self) {
        if let Ok(_lock) = maybe_gpu_msm_lock() {
            unsafe { midnight_sppark_bls12_381_numerator_context_free(self.raw.as_ptr()) };
        }
    }
}

impl Bls12_381FrDeviceVec {
    pub fn upload(values: &[Bls12_381Fr]) -> Result<Self, MsmError> {
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<Bls12_381Fr>(values.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_upload(&mut raw, values.as_ptr(), values.len())
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null scalar device vector".to_string())
        })?;
        Ok(Self {
            raw,
            len: values.len(),
        })
    }

    pub fn upload_many(values: &[&[Bls12_381Fr]]) -> Result<Self, MsmError> {
        if values.is_empty() {
            return Self::upload(&[]);
        }
        let lengths = values.iter().map(|values| values.len()).collect::<Vec<_>>();
        let len = lengths
            .iter()
            .try_fold(0usize, |total, len| total.checked_add(*len))
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: usize::MAX,
            })?;
        let pointers = values
            .iter()
            .map(|values| values.as_ptr())
            .collect::<Vec<_>>();

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<Bls12_381Fr>(len),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_upload_many(
                &mut raw,
                pointers.as_ptr(),
                lengths.as_ptr(),
                values.len(),
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null multi-slice device vector".to_string())
        })?;
        Ok(Self { raw, len })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn poseidon_advice_from_witness(
        witness: &[Bls12_381Fr],
        hash_count: usize,
        domain_len: usize,
        metadata: &[Bls12_381Fr; 11],
        layout: &[u16],
    ) -> Result<Self, MsmError> {
        let expected_witness = hash_count
            .checked_mul(256)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: witness.len(),
            })?;
        let len = domain_len.checked_mul(8).ok_or(MsmError::LengthMismatch {
            points: usize::MAX,
            scalars: witness.len(),
        })?;
        if hash_count == 0
            || witness.len() != expected_witness
            || layout.len() != 128
            || domain_len <= 2 + 41 * hash_count
        {
            return Err(MsmError::LengthMismatch {
                points: expected_witness,
                scalars: witness.len(),
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<Bls12_381Fr>(witness.len() + metadata.len())
                + slice_bytes::<u16>(layout.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_poseidon_advice_from_witness(
                &mut raw,
                witness.as_ptr(),
                hash_count,
                domain_len,
                metadata.as_ptr(),
                metadata.len(),
                layout.as_ptr(),
                layout.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null Poseidon advice vector".to_string())
        })?;
        Ok(Self { raw, len })
    }

    pub fn scatter_host(
        &mut self,
        indices: &[u32],
        values: &[Bls12_381Fr],
    ) -> Result<(), MsmError> {
        if indices.len() != values.len() || indices.iter().any(|&index| index as usize >= self.len)
        {
            return Err(MsmError::LengthMismatch {
                points: self.len,
                scalars: values.len(),
            });
        }
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<u32>(indices.len()) + slice_bytes::<Bls12_381Fr>(values.len()),
            0,
        );
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_scatter_host(
                self.raw.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
                indices.len(),
            )
        };
        drop(lock);
        if status.code == 0 {
            Ok(())
        } else {
            Err(MsmError::Sppark(String::from(status)))
        }
    }

    pub fn concat_host(&self, suffix: &[Bls12_381Fr]) -> Result<Self, MsmError> {
        let len = self
            .len
            .checked_add(suffix.len())
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: suffix.len(),
            })?;
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<Bls12_381Fr>(suffix.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_concat_host(
                &mut raw,
                self.raw.as_ptr(),
                suffix.as_ptr(),
                suffix.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null concatenated device vector".to_string())
        })?;
        Ok(Self { raw, len })
    }

    pub fn concat_many_host(sources: &[&Self], suffix: &[Bls12_381Fr]) -> Result<Self, MsmError> {
        if sources.is_empty() {
            return Self::upload(suffix);
        }
        let len = sources
            .iter()
            .try_fold(suffix.len(), |len, source| len.checked_add(source.len))
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: suffix.len(),
            })?;
        let raw_sources = sources
            .iter()
            .map(|source| source.raw.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecUpload,
            slice_bytes::<Bls12_381Fr>(suffix.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_concat_many_host(
                &mut raw,
                raw_sources.as_ptr(),
                raw_sources.len(),
                suffix.as_ptr(),
                suffix.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null multi-source device vector".to_string())
        })?;
        Ok(Self { raw, len })
    }

    pub fn gather_polys(&self, poly_len: usize, indices: &[u32]) -> Result<Self, MsmError> {
        if poly_len == 0
            || indices.is_empty()
            || self.len % poly_len != 0
            || indices
                .iter()
                .any(|&index| index as usize >= self.len / poly_len)
        {
            return Err(MsmError::LengthMismatch {
                points: self.len,
                scalars: poly_len.saturating_mul(indices.len()),
            });
        }
        let lock = maybe_gpu_msm_lock()?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_gather_polys(
                &mut raw,
                self.raw.as_ptr(),
                poly_len,
                indices.as_ptr(),
                indices.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null gathered polynomial vector".to_string())
        })?;
        Ok(Self {
            raw,
            len: poly_len * indices.len(),
        })
    }

    pub fn coeffs_to_extended(
        values: &[Bls12_381Fr],
        input_len: usize,
        count: usize,
        extended_log_n: u32,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> Result<Self, MsmError> {
        if input_len == 0 || count == 0 || values.len() != input_len.saturating_mul(count) {
            return Err(MsmError::LengthMismatch {
                points: input_len.saturating_mul(count),
                scalars: values.len(),
            });
        }
        let output_len = 1usize
            .checked_shl(extended_log_n)
            .ok_or(MsmError::NttLogSizeOverflow { len: values.len() })?;
        if input_len > output_len {
            return Err(MsmError::LengthMismatch {
                points: output_len,
                scalars: input_len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceCoeffToExtended,
            slice_bytes::<Bls12_381Fr>(values.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_coeff_to_extended_batch(
                &mut raw,
                values.as_ptr(),
                input_len,
                count,
                extended_log_n,
                first,
                second,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null extended device vector".to_string())
        })?;
        Ok(Self {
            raw,
            len: output_len * count,
        })
    }

    pub fn coeffs_to_extended_from_device(
        &self,
        input_len: usize,
        count: usize,
        extended_log_n: u32,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> Result<Self, MsmError> {
        if input_len == 0 || count == 0 || self.len != input_len.saturating_mul(count) {
            return Err(MsmError::LengthMismatch {
                points: input_len.saturating_mul(count),
                scalars: self.len,
            });
        }
        let output_len = 1usize
            .checked_shl(extended_log_n)
            .ok_or(MsmError::NttLogSizeOverflow { len: self.len })?;
        if input_len > output_len {
            return Err(MsmError::LengthMismatch {
                points: output_len,
                scalars: input_len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(TransferOp::FrDeviceCoeffToExtended, 0, 0);
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_coeff_to_extended_from_device_batch(
                &mut raw,
                self.raw.as_ptr(),
                input_len,
                count,
                extended_log_n,
                first,
                second,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null extended device vector".to_string())
        })?;
        Ok(Self {
            raw,
            len: output_len * count,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn download(&self) -> Result<Vec<Bls12_381Fr>, MsmError> {
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceVecDownload,
            0,
            slice_bytes::<Bls12_381Fr>(self.len),
        );
        let mut out = vec![Bls12_381Fr::default(); self.len];
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_download(
                out.as_mut_ptr(),
                self.raw.as_ptr(),
                self.len,
            )
        };
        drop(lock);

        if status.code == 0 {
            Ok(out)
        } else {
            Err(MsmError::Sppark(String::from(status)))
        }
    }

    pub fn download_nonzero_suffixes(
        &self,
        scalar_offset: usize,
        poly_len: usize,
        count: usize,
    ) -> Result<(Vec<usize>, Vec<Bls12_381Fr>), MsmError> {
        let capacity = poly_len
            .checked_mul(count)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            })?;
        let required = scalar_offset
            .checked_add(capacity)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            })?;
        if poly_len == 0 || count == 0 || required > self.len {
            return Err(MsmError::LengthMismatch {
                points: required,
                scalars: self.len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        let mut starts = vec![0usize; count];
        let mut values = Vec::<Bls12_381Fr>::with_capacity(capacity);
        let mut written = 0usize;
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_download_nonzero_suffixes(
                values.as_mut_ptr(),
                capacity,
                &mut written,
                starts.as_mut_ptr(),
                std::ptr::null_mut(),
                self.raw.as_ptr(),
                scalar_offset,
                poly_len,
                count,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        if written > capacity {
            return Err(MsmError::Sppark(
                "sppark returned an oversized nonzero suffix batch".to_string(),
            ));
        }
        unsafe {
            values.set_len(written);
        }
        record_transfer(
            TransferOp::FrDeviceVecDownload,
            0,
            slice_bytes::<usize>(starts.len()) + slice_bytes::<Bls12_381Fr>(values.len()),
        );
        Ok((starts, values))
    }

    pub fn nonzero_suffix_stats(
        &self,
        scalar_offset: usize,
        poly_len: usize,
        count: usize,
    ) -> Result<(Vec<usize>, Vec<usize>), MsmError> {
        let capacity = poly_len
            .checked_mul(count)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            })?;
        let required = scalar_offset
            .checked_add(capacity)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            })?;
        if poly_len == 0 || count == 0 || required > self.len {
            return Err(MsmError::LengthMismatch {
                points: required,
                scalars: self.len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        let mut starts = vec![0usize; count];
        let mut nonzero_counts = vec![0usize; count];
        let mut suffix_len = 0usize;
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_download_nonzero_suffixes(
                std::ptr::null_mut(),
                0,
                &mut suffix_len,
                starts.as_mut_ptr(),
                nonzero_counts.as_mut_ptr(),
                self.raw.as_ptr(),
                scalar_offset,
                poly_len,
                count,
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        if starts.iter().any(|&start| start > poly_len)
            || starts
                .iter()
                .zip(&nonzero_counts)
                .any(|(&start, &nonzero_count)| nonzero_count > poly_len - start)
            || suffix_len != starts.iter().map(|start| poly_len - start).sum::<usize>()
        {
            return Err(MsmError::Sppark(
                "sppark returned invalid nonzero suffix starts".to_string(),
            ));
        }
        record_transfer(
            TransferOp::FrDeviceVecDownload,
            0,
            slice_bytes::<usize>(starts.len() + nonzero_counts.len()),
        );
        Ok((starts, nonzero_counts))
    }

    pub fn nonzero_suffix_starts(
        &self,
        scalar_offset: usize,
        poly_len: usize,
        count: usize,
    ) -> Result<Vec<usize>, MsmError> {
        self.nonzero_suffix_stats(scalar_offset, poly_len, count)
            .map(|(starts, _)| starts)
    }

    pub fn quotient_pieces(
        &self,
        poly_len: usize,
        piece_count: usize,
        blinds: &[Bls12_381Fr],
    ) -> Result<Self, MsmError> {
        let output_len = poly_len
            .checked_mul(piece_count)
            .ok_or(MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            })?;
        let required = poly_len.saturating_sub(1).checked_mul(piece_count).ok_or(
            MsmError::LengthMismatch {
                points: usize::MAX,
                scalars: self.len,
            },
        )?;
        if poly_len < 2 || piece_count < 2 || blinds.len() != piece_count - 1 || self.len < required
        {
            return Err(MsmError::LengthMismatch {
                points: required,
                scalars: self.len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceDivideNtt,
            slice_bytes::<Bls12_381Fr>(blinds.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_quotient_pieces(
                &mut raw,
                self.raw.as_ptr(),
                poly_len,
                piece_count,
                blinds.as_ptr(),
                blinds.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned null resident quotient pieces".to_string())
        })?;
        Ok(Self {
            raw,
            len: output_len,
        })
    }

    pub fn poly_lincomb_device(
        &self,
        poly_len: usize,
        scalars: &[Bls12_381Fr],
    ) -> Result<Self, MsmError> {
        let poly_count = scalars.len();
        if poly_len == 0 || poly_count == 0 || self.len != poly_len.saturating_mul(poly_count) {
            return Err(MsmError::LengthMismatch {
                points: poly_len.saturating_mul(poly_count),
                scalars: self.len,
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrPolyLincomb,
            slice_bytes::<Bls12_381Fr>(scalars.len()),
            0,
        );
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_poly_lincomb(
                &mut raw,
                self.raw.as_ptr(),
                poly_len,
                poly_count,
                scalars.as_ptr(),
                scalars.len(),
            )
        };
        drop(lock);
        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned null resident polynomial combination".to_string())
        })?;
        Ok(Self { raw, len: poly_len })
    }

    pub fn ntt_zeta(
        &mut self,
        inverse: bool,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> Result<(), MsmError> {
        if self.len <= 1 {
            return Ok(());
        }
        if !self.len.is_power_of_two() {
            return Err(MsmError::NttLengthNotPowerOfTwo { len: self.len });
        }
        let log_n = usize::BITS - self.len.leading_zeros() - 1;
        if log_n > u32::MAX {
            return Err(MsmError::NttLogSizeOverflow { len: self.len });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(TransferOp::FrDeviceVecNtt, 0, 0);
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_ntt_zeta(
                self.raw.as_ptr(),
                log_n,
                inverse,
                scale,
                first,
                second,
            )
        };
        drop(lock);

        if status.code == 0 {
            Ok(())
        } else {
            Err(MsmError::Sppark(String::from(status)))
        }
    }

    pub fn ntt_zeta_batch(
        &mut self,
        poly_len: usize,
        count: usize,
        inverse: bool,
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> Result<(), MsmError> {
        if poly_len == 0 || count == 0 || poly_len.checked_mul(count) != Some(self.len) {
            return Err(MsmError::LengthMismatch {
                points: poly_len.saturating_mul(count),
                scalars: self.len,
            });
        }
        if !poly_len.is_power_of_two() {
            return Err(MsmError::NttLengthNotPowerOfTwo { len: poly_len });
        }
        let log_n = usize::BITS - poly_len.leading_zeros() - 1;

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(TransferOp::FrDeviceVecNtt, 0, 0);
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_ntt_zeta_batch(
                self.raw.as_ptr(),
                log_n,
                count,
                inverse,
                scale,
                first,
                second,
            )
        };
        drop(lock);

        if status.code == 0 {
            Ok(())
        } else {
            Err(MsmError::Sppark(String::from(status)))
        }
    }

    pub fn divide_ntt_zeta(
        &mut self,
        factors: &[Bls12_381Fr],
        scale: Bls12_381Fr,
        first: Bls12_381Fr,
        second: Bls12_381Fr,
    ) -> Result<(), MsmError> {
        if self.len <= 1 || factors.is_empty() {
            return Ok(());
        }
        if !self.len.is_power_of_two() {
            return Err(MsmError::NttLengthNotPowerOfTwo { len: self.len });
        }
        let log_n = usize::BITS - self.len.leading_zeros() - 1;

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::FrDeviceDivideNtt,
            slice_bytes::<Bls12_381Fr>(factors.len()),
            0,
        );
        let status = unsafe {
            midnight_sppark_bls12_381_fr_device_vec_divide_ntt_zeta(
                self.raw.as_ptr(),
                log_n,
                factors.as_ptr(),
                factors.len(),
                scale,
                first,
                second,
            )
        };
        drop(lock);

        if status.code == 0 {
            Ok(())
        } else {
            Err(MsmError::Sppark(String::from(status)))
        }
    }
}

pub fn bls12_381_fr_permutation_products(
    columns: &[Bls12_381Fr],
    permutations: &[Bls12_381Fr],
    len: usize,
    chunk_len: usize,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    omega: Bls12_381Fr,
    delta_powers: &[Bls12_381Fr],
    blind_values: &[Bls12_381Fr],
    blinding_factors: usize,
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    if len == 0 || chunk_len == 0 || columns.len() != permutations.len() || columns.len() % len != 0
    {
        return Err(MsmError::LengthMismatch {
            points: columns.len(),
            scalars: permutations.len(),
        });
    }
    let column_count = columns.len() / len;
    let set_count = column_count.div_ceil(chunk_len);
    if delta_powers.len() != column_count
        || blind_values.len() != set_count.saturating_mul(blinding_factors)
    {
        return Err(MsmError::LengthMismatch {
            points: column_count + set_count.saturating_mul(blinding_factors),
            scalars: delta_powers.len() + blind_values.len(),
        });
    }

    let h2d_bytes = slice_bytes::<Bls12_381Fr>(
        columns.len() + permutations.len() + delta_powers.len() + blind_values.len(),
    );
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(TransferOp::FrPermutationProducts, h2d_bytes, 0);
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_permutation_products(
            &mut raw,
            columns.as_ptr(),
            permutations.as_ptr(),
            len,
            column_count,
            chunk_len,
            beta,
            gamma,
            omega,
            delta_powers.as_ptr(),
            blind_values.as_ptr(),
            blinding_factors,
        )
    };
    drop(lock);

    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw).ok_or_else(|| {
        MsmError::Sppark("sppark returned null resident permutation products".to_string())
    })?;
    Ok(Bls12_381FrDeviceVec {
        raw,
        len: len * set_count,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_fr_lookup_products(
    compressed_inputs: &[Bls12_381Fr],
    compressed_tables: &[Bls12_381Fr],
    permuted_inputs: &[Bls12_381Fr],
    permuted_tables: &[Bls12_381Fr],
    len: usize,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    blind_values: &[Bls12_381Fr],
    blinding_factors: usize,
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    if len == 0
        || compressed_inputs.len() != compressed_tables.len()
        || compressed_inputs.len() != permuted_inputs.len()
        || compressed_inputs.len() != permuted_tables.len()
        || compressed_inputs.len() % len != 0
    {
        return Err(MsmError::LengthMismatch {
            points: compressed_inputs.len(),
            scalars: compressed_tables
                .len()
                .saturating_add(permuted_inputs.len())
                .saturating_add(permuted_tables.len()),
        });
    }
    let lookup_count = compressed_inputs.len() / len;
    if lookup_count == 0 || blind_values.len() != lookup_count.saturating_mul(blinding_factors) {
        return Err(MsmError::LengthMismatch {
            points: lookup_count.saturating_mul(blinding_factors),
            scalars: blind_values.len(),
        });
    }

    let h2d_bytes = slice_bytes::<Bls12_381Fr>(
        compressed_inputs.len()
            + compressed_tables.len()
            + permuted_inputs.len()
            + permuted_tables.len()
            + blind_values.len(),
    );
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(TransferOp::FrLookupProducts, h2d_bytes, 0);
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_lookup_products(
            &mut raw,
            compressed_inputs.as_ptr(),
            compressed_tables.as_ptr(),
            permuted_inputs.as_ptr(),
            permuted_tables.as_ptr(),
            len,
            lookup_count,
            beta,
            gamma,
            blind_values.as_ptr(),
            blinding_factors,
        )
    };
    drop(lock);

    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw).ok_or_else(|| {
        MsmError::Sppark("sppark returned null resident lookup products".to_string())
    })?;
    Ok(Bls12_381FrDeviceVec {
        raw,
        len: len * lookup_count,
    })
}

pub fn bls12_381_fr_lookup_products_device(
    trace_lagrange: &Bls12_381FrDeviceVec,
    len: usize,
    lookup_count: usize,
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    blind_values: &[Bls12_381Fr],
    blinding_factors: usize,
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    if len == 0
        || lookup_count == 0
        || trace_lagrange.len != len.saturating_mul(lookup_count).saturating_mul(4)
        || blind_values.len() != lookup_count.saturating_mul(blinding_factors)
    {
        return Err(MsmError::LengthMismatch {
            points: len.saturating_mul(lookup_count).saturating_mul(4),
            scalars: trace_lagrange.len,
        });
    }
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::TraceLookupProducts,
        slice_bytes::<Bls12_381Fr>(blind_values.len()),
        0,
    );
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_lookup_products_device(
            &mut raw,
            trace_lagrange.raw.as_ptr().cast_const(),
            len,
            lookup_count,
            beta,
            gamma,
            blind_values.as_ptr(),
            blinding_factors,
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw).ok_or_else(|| {
        MsmError::Sppark("sppark returned null device-resident lookup products".to_string())
    })?;
    Ok(Bls12_381FrDeviceVec {
        raw,
        len: len * lookup_count,
    })
}

pub fn bls12_381_fr_lookup_permute_batch(
    compressed_inputs: &[Bls12_381Fr],
    compressed_tables: &[Bls12_381Fr],
    len: usize,
    usable_rows: usize,
    input_blinds: &[Bls12_381Fr],
    table_blinds: &[Bls12_381Fr],
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    if len == 0
        || usable_rows == 0
        || usable_rows > len
        || compressed_inputs.len() != compressed_tables.len()
        || compressed_inputs.len() % len != 0
    {
        return Err(MsmError::LengthMismatch {
            points: compressed_inputs.len(),
            scalars: compressed_tables.len(),
        });
    }
    let lookup_count = compressed_inputs.len() / len;
    let blind_count = len - usable_rows;
    if lookup_count == 0
        || input_blinds.len() != lookup_count.saturating_mul(blind_count)
        || table_blinds.len() != input_blinds.len()
    {
        return Err(MsmError::LengthMismatch {
            points: lookup_count.saturating_mul(blind_count),
            scalars: input_blinds.len().saturating_add(table_blinds.len()),
        });
    }
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::TraceLookupPermute,
        slice_bytes::<Bls12_381Fr>(
            compressed_inputs.len()
                + compressed_tables.len()
                + input_blinds.len()
                + table_blinds.len(),
        ),
        0,
    );
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_lookup_permute_batch(
            &mut raw,
            compressed_inputs.as_ptr(),
            compressed_tables.as_ptr(),
            len,
            lookup_count,
            usable_rows,
            input_blinds.as_ptr(),
            table_blinds.as_ptr(),
            blind_count,
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw).ok_or_else(|| {
        MsmError::Sppark("sppark returned null resident lookup permutation".to_string())
    })?;
    Ok(Bls12_381FrDeviceVec {
        raw,
        len: compressed_inputs.len().saturating_mul(4),
    })
}

impl Bls12_381NumeratorContext {
    pub fn new(inputs: Bls12_381NumeratorContextInputs<'_>) -> Result<Self, MsmError> {
        if inputs.len == 0
            || inputs.input_len == 0
            || inputs.fixed.len() != inputs.fixed_cols.saturating_mul(inputs.len)
            || inputs.fixed_lagrange.len() != inputs.fixed_cols.saturating_mul(inputs.input_len)
            || inputs.l0.len() != inputs.len
            || inputs.l_last.len() != inputs.len
            || inputs.l_active.len() != inputs.len
            || inputs.permutation_pk_cosets.len()
                != inputs.permutation_pk_cosets_len.saturating_mul(inputs.len)
            || inputs.permutation_pk_lagrange.len()
                != inputs
                    .permutation_pk_cosets_len
                    .saturating_mul(inputs.input_len)
            || inputs.lookup_input_graphs.len() != inputs.lookup_graphs.len()
            || inputs.lookup_table_graphs.len() != inputs.lookup_graphs.len()
            || inputs.trash_selector_fixed_indices.len() != inputs.trash_graphs.len()
        {
            return Err(MsmError::LengthMismatch {
                points: inputs.len,
                scalars: inputs.fixed.len(),
            });
        }

        let graph_bytes = raw_graph_h2d_bytes(inputs.custom_graph)
            + inputs
                .lookup_graphs
                .iter()
                .copied()
                .map(raw_graph_h2d_bytes)
                .sum::<usize>()
            + inputs
                .lookup_input_graphs
                .iter()
                .copied()
                .map(raw_graph_h2d_bytes)
                .sum::<usize>()
            + inputs
                .lookup_table_graphs
                .iter()
                .copied()
                .map(raw_graph_h2d_bytes)
                .sum::<usize>()
            + inputs
                .trash_graphs
                .iter()
                .copied()
                .map(raw_graph_h2d_bytes)
                .sum::<usize>();
        let h2d_bytes =
            slice_bytes::<Bls12_381Fr>(inputs.fixed.len() + inputs.fixed_lagrange.len())
                + slice_bytes::<Bls12_381Fr>(
                    inputs.l0.len() + inputs.l_last.len() + inputs.l_active.len(),
                )
                + slice_bytes::<RawColumnRef>(inputs.permutation_columns.len())
                + slice_bytes::<Bls12_381Fr>(
                    inputs.permutation_pk_cosets.len() + inputs.permutation_pk_lagrange.len(),
                )
                + slice_bytes::<u32>(inputs.trash_selector_fixed_indices.len())
                + graph_bytes;

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(TransferOp::NumeratorContextCreate, h2d_bytes, 0);
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_numerator_context_create(
                &mut raw,
                inputs.len,
                inputs.input_len,
                inputs.fixed.as_ptr(),
                inputs.fixed_lagrange.as_ptr(),
                inputs.fixed_cols,
                inputs.l0.as_ptr(),
                inputs.l_last.as_ptr(),
                inputs.l_active.as_ptr(),
                inputs.permutation_columns.as_ptr(),
                inputs.permutation_columns.len(),
                inputs.permutation_pk_cosets.as_ptr(),
                inputs.permutation_pk_lagrange.as_ptr(),
                inputs.permutation_pk_cosets_len,
                inputs.trash_selector_fixed_indices.as_ptr(),
                inputs.trash_selector_fixed_indices.len(),
                inputs.custom_graph,
                inputs.lookup_graphs.as_ptr(),
                inputs.lookup_input_graphs.as_ptr(),
                inputs.lookup_table_graphs.as_ptr(),
                inputs.lookup_graphs.len(),
                inputs.trash_graphs.as_ptr(),
                inputs.omega,
                inputs.trace_omega,
                inputs.g_coset,
                inputs.delta_const,
                inputs.rot_scale,
                inputs.last_rotation,
                inputs.chunk_len,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null numerator context".to_string())
        })?;
        Ok(Self {
            raw,
            len: inputs.len,
            input_len: inputs.input_len,
            permutation_column_count: inputs.permutation_columns.len(),
            permutation_chunk_len: inputs.chunk_len as usize,
            lookup_count: inputs.lookup_graphs.len(),
            trash_count: inputs.trash_graphs.len(),
        })
    }

    pub fn evaluate(
        &self,
        inputs: Bls12_381NumeratorEvalInputs<'_>,
    ) -> Result<Bls12_381FrDeviceVec, MsmError> {
        let columns = inputs
            .columns
            .map_or(std::ptr::null(), |value| value.raw.as_ptr().cast_const());
        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::NumeratorContextEval,
            slice_bytes::<Bls12_381Fr>(inputs.challenges.len()),
            0,
        );
        let mut out = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_numerator_context_eval(
                &mut out,
                self.raw.as_ptr(),
                columns,
                inputs.total_columns,
                inputs.advice_offset,
                inputs.advice_cols,
                inputs.instance_offset,
                inputs.instance_cols,
                inputs.challenges.as_ptr(),
                inputs.challenges.len(),
                inputs.beta,
                inputs.gamma,
                inputs.theta,
                inputs.trash_challenge,
                inputs.y,
                inputs.permutation_offset,
                inputs.permutation_sets,
                inputs.lookup_product_offset,
                inputs.lookup_input_offset,
                inputs.lookup_table_offset,
                self.lookup_count,
                inputs.trash_offset,
                self.trash_count,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(out).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null numerator result".to_string())
        })?;
        Ok(Bls12_381FrDeviceVec { raw, len: self.len })
    }

    pub fn evaluate_trace_trash(
        &self,
        inputs: Bls12_381TraceTrashInputs<'_>,
    ) -> Result<Bls12_381FrDeviceVec, MsmError> {
        if inputs.advice.len != inputs.advice_cols.saturating_mul(self.input_len)
            || inputs.instance.len != inputs.instance_cols.saturating_mul(self.input_len)
        {
            return Err(MsmError::LengthMismatch {
                points: inputs
                    .advice_cols
                    .saturating_add(inputs.instance_cols)
                    .saturating_mul(self.input_len),
                scalars: inputs.advice.len.saturating_add(inputs.instance.len),
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::TraceContextTrash,
            slice_bytes::<Bls12_381Fr>(inputs.challenges.len()),
            0,
        );
        let mut out = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_trace_context_trash(
                &mut out,
                self.raw.as_ptr(),
                inputs.advice.raw.as_ptr().cast_const(),
                inputs.advice_cols,
                inputs.instance.raw.as_ptr().cast_const(),
                inputs.instance_cols,
                inputs.challenges.as_ptr(),
                inputs.challenges.len(),
                inputs.trash_challenge,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(out).ok_or_else(|| {
            MsmError::Sppark("sppark returned a null resident trash result".to_string())
        })?;
        Ok(Bls12_381FrDeviceVec {
            raw,
            len: self.input_len.saturating_mul(self.trash_count),
        })
    }

    pub fn trace_permutation_products(
        &self,
        inputs: Bls12_381TracePermutationInputs<'_>,
    ) -> Result<Bls12_381FrDeviceVec, MsmError> {
        if self.permutation_chunk_len == 0 {
            return Err(MsmError::LengthMismatch {
                points: self.permutation_column_count,
                scalars: 0,
            });
        }
        let set_count = self
            .permutation_column_count
            .div_ceil(self.permutation_chunk_len);
        if inputs.advice.len != inputs.advice_cols.saturating_mul(self.input_len)
            || inputs.instance.len != inputs.instance_cols.saturating_mul(self.input_len)
            || inputs.delta_powers.len() != self.permutation_column_count
            || inputs.blind_values.len() != set_count.saturating_mul(inputs.blinding_factors)
        {
            return Err(MsmError::LengthMismatch {
                points: self.permutation_column_count,
                scalars: inputs.delta_powers.len(),
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::TraceContextPermutation,
            slice_bytes::<Bls12_381Fr>(inputs.delta_powers.len() + inputs.blind_values.len()),
            0,
        );
        let mut out = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_trace_context_permutation(
                &mut out,
                self.raw.as_ptr(),
                inputs.advice.raw.as_ptr().cast_const(),
                inputs.advice_cols,
                inputs.instance.raw.as_ptr().cast_const(),
                inputs.instance_cols,
                inputs.beta,
                inputs.gamma,
                inputs.delta_powers.as_ptr(),
                inputs.delta_powers.len(),
                inputs.blind_values.as_ptr(),
                inputs.blind_values.len(),
                inputs.blinding_factors,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(out).ok_or_else(|| {
            MsmError::Sppark("sppark returned null resident permutation products".to_string())
        })?;
        Ok(Bls12_381FrDeviceVec {
            raw,
            len: self.input_len.saturating_mul(set_count),
        })
    }

    pub fn trace_lookup_permutation(
        &self,
        inputs: Bls12_381TraceLookupInputs<'_>,
    ) -> Result<Bls12_381FrDeviceVec, MsmError> {
        let blind_count = self.input_len.saturating_sub(inputs.usable_rows);
        let expected_blinds = self.lookup_count.saturating_mul(blind_count);
        if self.lookup_count == 0
            || inputs.usable_rows == 0
            || inputs.usable_rows > self.input_len
            || inputs.advice.len != inputs.advice_cols.saturating_mul(self.input_len)
            || inputs.instance.len != inputs.instance_cols.saturating_mul(self.input_len)
            || inputs.input_blinds.len() != expected_blinds
            || inputs.table_blinds.len() != expected_blinds
        {
            return Err(MsmError::LengthMismatch {
                points: expected_blinds,
                scalars: inputs
                    .input_blinds
                    .len()
                    .saturating_add(inputs.table_blinds.len()),
            });
        }

        let lock = maybe_gpu_msm_lock()?;
        record_transfer(
            TransferOp::TraceLookupPermute,
            slice_bytes::<Bls12_381Fr>(
                inputs.challenges.len() + inputs.input_blinds.len() + inputs.table_blinds.len(),
            ),
            0,
        );
        let mut out = std::ptr::null_mut();
        let status = unsafe {
            midnight_sppark_bls12_381_trace_context_lookup_permute(
                &mut out,
                self.raw.as_ptr(),
                inputs.advice.raw.as_ptr().cast_const(),
                inputs.advice_cols,
                inputs.instance.raw.as_ptr().cast_const(),
                inputs.instance_cols,
                inputs.challenges.as_ptr(),
                inputs.challenges.len(),
                inputs.theta,
                inputs.usable_rows,
                inputs.input_blinds.as_ptr(),
                inputs.table_blinds.as_ptr(),
                blind_count,
            )
        };
        drop(lock);

        if status.code != 0 {
            return Err(MsmError::Sppark(String::from(status)));
        }
        let raw = NonNull::new(out).ok_or_else(|| {
            MsmError::Sppark("sppark returned null resident lookup permutation".to_string())
        })?;
        Ok(Bls12_381FrDeviceVec {
            raw,
            len: self
                .input_len
                .saturating_mul(self.lookup_count)
                .saturating_mul(4),
        })
    }
}

pub fn bls12_381_g1_msm_context(
    points: &[Bls12_381G1Affine],
) -> Result<Arc<Bls12_381G1MsmContext>, MsmError> {
    let config = Bls12_381MsmConfig::for_point_count(points.len()).with_diagnostic_overrides()?;
    bls12_381_g1_msm_context_with_config(points, config)
}

pub fn bls12_381_g1_msm_context_with_config(
    points: &[Bls12_381G1Affine],
    config: Bls12_381MsmConfig,
) -> Result<Arc<Bls12_381G1MsmContext>, MsmError> {
    if points.is_empty() {
        return Err(MsmError::LengthMismatch {
            points: 0,
            scalars: 0,
        });
    }
    config.validate()?;

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmContextCreate,
        slice_bytes::<Bls12_381G1Affine>(points.len()),
        0,
    );
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_create_with_config(
            &mut raw,
            points.as_ptr(),
            points.len(),
            config,
        )
    };
    if status.code != 0 {
        drop(lock);
        return Err(MsmError::Sppark(String::from(status)));
    }
    let Some(raw) = NonNull::new(raw) else {
        drop(lock);
        return Err(MsmError::Sppark(
            "sppark returned a null MSM context".to_string(),
        ));
    };

    let scramble_mask = deterministic_msm_scramble_mask(points.len());
    let mut scramble_correction = Bls12_381G1Projective::default();
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_batch(
            &mut scramble_correction,
            raw.as_ptr(),
            scramble_mask.as_ptr(),
            points.len(),
            1,
        )
    };
    if status.code != 0 {
        unsafe { midnight_sppark_bls12_381_g1_msm_context_free(raw.as_ptr()) };
        drop(lock);
        return Err(MsmError::Sppark(String::from(status)));
    }

    record_transfer(
        TransferOp::FrDeviceVecUpload,
        slice_bytes::<Bls12_381Fr>(scramble_mask.len()),
        0,
    );
    let mut scramble_mask_raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_device_vec_upload(
            &mut scramble_mask_raw,
            scramble_mask.as_ptr(),
            scramble_mask.len(),
        )
    };
    if status.code != 0 {
        unsafe { midnight_sppark_bls12_381_g1_msm_context_free(raw.as_ptr()) };
        drop(lock);
        return Err(MsmError::Sppark(String::from(status)));
    }
    let Some(scramble_mask_raw) = NonNull::new(scramble_mask_raw) else {
        unsafe { midnight_sppark_bls12_381_g1_msm_context_free(raw.as_ptr()) };
        drop(lock);
        return Err(MsmError::Sppark(
            "sppark returned a null MSM scramble mask".to_string(),
        ));
    };
    drop(lock);

    let ec_gpu_bases = points.iter().map(ec_gpu_affine).collect::<Vec<_>>();
    Ok(Arc::new(Bls12_381G1MsmContext {
        raw,
        npoints: points.len(),
        config,
        ec_gpu_bases: Arc::new(ec_gpu_bases),
        sppark_scramble_mask: Bls12_381FrDeviceVec {
            raw: scramble_mask_raw,
            len: scramble_mask.len(),
        },
        sppark_scramble_correction: to_ec_gpu_projective(scramble_correction),
    }))
}

pub fn bls12_381_g1_msm_context_ec_gpu_batch(
    context: &Bls12_381G1MsmContext,
    scalar_batches: &[&[[u8; 32]]],
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    if scalar_batches
        .iter()
        .any(|scalars| scalars.len() != context.npoints)
    {
        let scalars = scalar_batches
            .iter()
            .map(|scalars| scalars.len())
            .find(|len| *len != context.npoints)
            .unwrap_or(0);
        return Err(MsmError::LengthMismatch {
            points: context.npoints,
            scalars,
        });
    }
    if scalar_batches.is_empty() {
        return Ok(Vec::new());
    }

    let runtime = ec_gpu_msm_runtime()?;
    let mut runtime = runtime.lock().map_err(|_| MsmError::LockPoisoned)?;
    let worker = runtime.worker.clone();
    let result_download_bytes = runtime.result_download_bytes;
    let mut out = Vec::with_capacity(scalar_batches.len());
    for scalars in scalar_batches {
        record_transfer(
            TransferOp::EcGpuMsm,
            slice_bytes::<Bls12_381G1Affine>(context.npoints)
                .saturating_add(slice_bytes::<[u8; 32]>(scalars.len())),
            result_download_bytes,
        );
        let point = runtime
            .kernel
            .multiexp(
                &worker,
                context.ec_gpu_bases.clone(),
                Arc::new(scalars.to_vec()),
                0,
            )
            .map_err(|error| MsmError::EcGpu(error.to_string()))?;
        out.push(from_ec_gpu_projective(point));
    }
    Ok(out)
}

pub fn bls12_381_g1_msm_context_batch(
    context: &Bls12_381G1MsmContext,
    scalar_batches: &[&[Bls12_381Fr]],
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    if scalar_batches
        .iter()
        .any(|scalars| scalars.len() != context.npoints)
    {
        let scalars = scalar_batches
            .iter()
            .map(|scalars| scalars.len())
            .find(|len| *len != context.npoints)
            .unwrap_or(0);
        return Err(MsmError::LengthMismatch {
            points: context.npoints,
            scalars,
        });
    }
    if scalar_batches.is_empty() {
        return Ok(Vec::new());
    }

    let mut flat_scalars = Vec::with_capacity(context.npoints * scalar_batches.len());
    for scalars in scalar_batches {
        flat_scalars.extend_from_slice(scalars);
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmContextBatch,
        slice_bytes::<Bls12_381Fr>(flat_scalars.len()),
        slice_bytes::<Bls12_381G1Projective>(scalar_batches.len()),
    );
    let mut out = vec![Bls12_381G1Projective::default(); scalar_batches.len()];
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_batch(
            out.as_mut_ptr(),
            context.raw.as_ptr(),
            flat_scalars.as_ptr(),
            context.npoints,
            scalar_batches.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_g1_msm_context_device_batch(
    context: &Bls12_381G1MsmContext,
    scalars: &Bls12_381FrDeviceVec,
    scalar_offset: usize,
    nmsms: usize,
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    let required = context
        .npoints
        .checked_mul(nmsms)
        .and_then(|len| scalar_offset.checked_add(len))
        .ok_or(MsmError::LengthMismatch {
            points: context.npoints.saturating_mul(nmsms),
            scalars: scalars.len,
        })?;
    if required > scalars.len {
        return Err(MsmError::LengthMismatch {
            points: required,
            scalars: scalars.len,
        });
    }
    if nmsms == 0 {
        return Ok(Vec::new());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmContextDeviceBatch,
        0,
        slice_bytes::<Bls12_381G1Projective>(nmsms),
    );
    let mut out = vec![Bls12_381G1Projective::default(); nmsms];
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_device_batch(
            out.as_mut_ptr(),
            context.raw.as_ptr(),
            scalars.raw.as_ptr(),
            scalar_offset,
            context.npoints,
            nmsms,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_g1_msm_context_device_ragged(
    context: &Bls12_381G1MsmContext,
    scalars: &Bls12_381FrDeviceVec,
    slices: &[Bls12_381MsmSlice],
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    if slices.iter().any(|slice| {
        slice.point_offset > context.npoints
            || slice.len > context.npoints - slice.point_offset
            || slice.scalar_offset > scalars.len
            || slice.len > scalars.len - slice.scalar_offset
            || slice.nonzero_len > slice.len
    }) {
        return Err(MsmError::LengthMismatch {
            points: context.npoints,
            scalars: scalars.len,
        });
    }
    if slices.is_empty() {
        return Ok(Vec::new());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmContextDeviceBatch,
        0,
        slice_bytes::<Bls12_381G1Projective>(slices.len()),
    );
    let mut out = vec![Bls12_381G1Projective::default(); slices.len()];
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_device_ragged(
            out.as_mut_ptr(),
            context.raw.as_ptr(),
            scalars.raw.as_ptr(),
            slices.as_ptr(),
            slices.len(),
        )
    };
    drop(lock);
    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_g1_msm_context_device_scrambled_batch(
    context: &Bls12_381G1MsmContext,
    scalars: &Bls12_381FrDeviceVec,
    scalar_offset: usize,
    nmsms: usize,
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    let required = context
        .npoints
        .checked_mul(nmsms)
        .and_then(|len| scalar_offset.checked_add(len))
        .ok_or(MsmError::LengthMismatch {
            points: context.npoints.saturating_mul(nmsms),
            scalars: scalars.len,
        })?;
    if required > scalars.len {
        return Err(MsmError::LengthMismatch {
            points: required,
            scalars: scalars.len,
        });
    }
    if nmsms == 0 {
        return Ok(Vec::new());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmContextDeviceBatch,
        0,
        slice_bytes::<Bls12_381G1Projective>(nmsms),
    );
    let mut out = vec![Bls12_381G1Projective::default(); nmsms];
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_context_device_scrambled_batch(
            out.as_mut_ptr(),
            context.raw.as_ptr(),
            scalars.raw.as_ptr(),
            scalar_offset,
            context.npoints,
            nmsms,
            context.sppark_scramble_mask.raw.as_ptr(),
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }

    let correction = context.sppark_scramble_correction;
    Ok(out
        .into_iter()
        .map(|point| from_ec_gpu_projective(to_ec_gpu_projective(point) - correction))
        .collect())
}

pub fn bls12_381_g1_msm_context_msm(
    context: &Bls12_381G1MsmContext,
    scalars: &[Bls12_381Fr],
) -> Result<Bls12_381G1Projective, MsmError> {
    bls12_381_g1_msm_context_batch(context, &[scalars])?
        .into_iter()
        .next()
        .ok_or_else(|| MsmError::Sppark("sppark returned no MSM result".to_string()))
}

pub fn bls12_381_g1_msm(
    points: &[Bls12_381G1Affine],
    scalars: &[Bls12_381Fr],
) -> Result<Bls12_381G1Projective, MsmError> {
    if points.len() != scalars.len() {
        return Err(MsmError::LengthMismatch {
            points: points.len(),
            scalars: scalars.len(),
        });
    }
    if points.is_empty() {
        return Ok(Bls12_381G1Projective::default());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::Msm,
        slice_bytes::<Bls12_381G1Affine>(points.len()) + slice_bytes::<Bls12_381Fr>(scalars.len()),
        size_of::<Bls12_381G1Projective>(),
    );
    let mut out = Bls12_381G1Projective::default();
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm(&mut out, points.as_ptr(), scalars.as_ptr(), points.len())
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_g1_msm_batch(
    points: &[Bls12_381G1Affine],
    scalar_batches: &[&[Bls12_381Fr]],
) -> Result<Vec<Bls12_381G1Projective>, MsmError> {
    if scalar_batches
        .iter()
        .any(|scalars| scalars.len() != points.len())
    {
        let scalars = scalar_batches
            .iter()
            .map(|scalars| scalars.len())
            .find(|len| *len != points.len())
            .unwrap_or(0);
        return Err(MsmError::LengthMismatch {
            points: points.len(),
            scalars,
        });
    }
    if scalar_batches.is_empty() {
        return Ok(Vec::new());
    }
    if points.is_empty() {
        return Ok(vec![Bls12_381G1Projective::default(); scalar_batches.len()]);
    }

    let mut flat_scalars = Vec::with_capacity(points.len() * scalar_batches.len());
    for scalars in scalar_batches {
        flat_scalars.extend_from_slice(scalars);
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::MsmBatch,
        slice_bytes::<Bls12_381G1Affine>(points.len())
            + slice_bytes::<Bls12_381Fr>(flat_scalars.len()),
        slice_bytes::<Bls12_381G1Projective>(scalar_batches.len()),
    );
    let mut out = vec![Bls12_381G1Projective::default(); scalar_batches.len()];
    let status = unsafe {
        midnight_sppark_bls12_381_g1_msm_batch(
            out.as_mut_ptr(),
            points.as_ptr(),
            flat_scalars.as_ptr(),
            points.len(),
            scalar_batches.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_ntt(values: &mut [Bls12_381Fr], inverse: bool) -> Result<(), MsmError> {
    if values.len() <= 1 {
        return Ok(());
    }
    if !values.len().is_power_of_two() {
        return Err(MsmError::NttLengthNotPowerOfTwo { len: values.len() });
    }
    let log_n = usize::BITS - values.len().leading_zeros() - 1;
    if log_n > u32::MAX {
        return Err(MsmError::NttLogSizeOverflow { len: values.len() });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrNtt,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status =
        unsafe { midnight_sppark_bls12_381_fr_ntt(values.as_mut_ptr(), log_n as u32, inverse) };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_coeff_to_extended_batch(
    out: &mut [Bls12_381Fr],
    inputs: &[Bls12_381Fr],
    input_len: usize,
    count: usize,
    extended_log_n: u32,
    first: Bls12_381Fr,
    second: Bls12_381Fr,
) -> Result<(), MsmError> {
    if input_len == 0 || count == 0 {
        return Ok(());
    }
    if input_len.checked_mul(count) != Some(inputs.len()) {
        return Err(MsmError::LengthMismatch {
            points: inputs.len(),
            scalars: input_len.saturating_mul(count),
        });
    }
    let output_len = 1usize
        .checked_shl(extended_log_n)
        .ok_or(MsmError::NttLogSizeOverflow {
            len: extended_log_n as usize,
        })?;
    if output_len.checked_mul(count) != Some(out.len()) {
        return Err(MsmError::LengthMismatch {
            points: out.len(),
            scalars: output_len.saturating_mul(count),
        });
    }
    if !input_len.is_power_of_two() {
        return Err(MsmError::NttLengthNotPowerOfTwo { len: input_len });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrCoeffToExtendedBatch,
        slice_bytes::<Bls12_381Fr>(inputs.len()),
        slice_bytes::<Bls12_381Fr>(out.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_coeff_to_extended_batch(
            out.as_mut_ptr(),
            inputs.as_ptr(),
            input_len,
            count,
            extended_log_n,
            first,
            second,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_ntt_zeta(
    values: &mut [Bls12_381Fr],
    inverse: bool,
    scale: Bls12_381Fr,
    first: Bls12_381Fr,
    second: Bls12_381Fr,
) -> Result<(), MsmError> {
    if values.len() <= 1 {
        return Ok(());
    }
    if !values.len().is_power_of_two() {
        return Err(MsmError::NttLengthNotPowerOfTwo { len: values.len() });
    }
    let log_n = usize::BITS - values.len().leading_zeros() - 1;
    if log_n > u32::MAX {
        return Err(MsmError::NttLogSizeOverflow { len: values.len() });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrNtt,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_ntt_zeta(
            values.as_mut_ptr(),
            log_n as u32,
            inverse,
            scale,
            first,
            second,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_divide_ntt_zeta(
    values: &mut [Bls12_381Fr],
    factors: &[Bls12_381Fr],
    scale: Bls12_381Fr,
    first: Bls12_381Fr,
    second: Bls12_381Fr,
) -> Result<(), MsmError> {
    if values.len() <= 1 || factors.is_empty() {
        return Ok(());
    }
    if !values.len().is_power_of_two() {
        return Err(MsmError::NttLengthNotPowerOfTwo { len: values.len() });
    }
    let log_n = usize::BITS - values.len().leading_zeros() - 1;
    if log_n > u32::MAX {
        return Err(MsmError::NttLogSizeOverflow { len: values.len() });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrNtt,
        slice_bytes::<Bls12_381Fr>(values.len()) + slice_bytes::<Bls12_381Fr>(factors.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_divide_ntt_zeta(
            values.as_mut_ptr(),
            log_n as u32,
            factors.as_ptr(),
            factors.len(),
            scale,
            first,
            second,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_mul_assign(
    values: &mut [Bls12_381Fr],
    scalar: Bls12_381Fr,
) -> Result<(), MsmError> {
    if values.is_empty() {
        return Ok(());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMulAssign,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_mul_assign(values.as_mut_ptr(), scalar, values.len())
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_mul_assign_zeta(
    values: &mut [Bls12_381Fr],
    first: Bls12_381Fr,
    second: Bls12_381Fr,
) -> Result<(), MsmError> {
    if values.is_empty() {
        return Ok(());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMulAssignZeta,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_mul_assign_zeta(
            values.as_mut_ptr(),
            first,
            second,
            values.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_mul_assign_repeating(
    values: &mut [Bls12_381Fr],
    factors: &[Bls12_381Fr],
) -> Result<(), MsmError> {
    if values.is_empty() || factors.is_empty() {
        return Ok(());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMulAssignRepeating,
        slice_bytes::<Bls12_381Fr>(values.len()) + slice_bytes::<Bls12_381Fr>(factors.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_mul_assign_repeating(
            values.as_mut_ptr(),
            factors.as_ptr(),
            factors.len(),
            values.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_shifted_prefix_product(
    values: &mut [Bls12_381Fr],
    initial: Bls12_381Fr,
) -> Result<(), MsmError> {
    if values.is_empty() {
        return Ok(());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrShiftedPrefixProduct,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_shifted_prefix_product(
            values.as_mut_ptr(),
            initial,
            values.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_poly_lincomb(
    out: &mut [Bls12_381Fr],
    polys: &[Bls12_381Fr],
    poly_len: usize,
    poly_count: usize,
    scalars: &[Bls12_381Fr],
) -> Result<(), MsmError> {
    if poly_len == 0 || poly_count == 0 {
        return Ok(());
    }
    if out.len() != poly_len || polys.len() != poly_len * poly_count {
        return Err(MsmError::LengthMismatch {
            points: polys.len(),
            scalars: poly_len * poly_count,
        });
    }
    if scalars.len() != poly_count {
        return Err(MsmError::LengthMismatch {
            points: poly_count,
            scalars: scalars.len(),
        });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrPolyLincomb,
        slice_bytes::<Bls12_381Fr>(polys.len()) + slice_bytes::<Bls12_381Fr>(scalars.len()),
        slice_bytes::<Bls12_381Fr>(out.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_poly_lincomb(
            out.as_mut_ptr(),
            polys.as_ptr(),
            poly_len,
            poly_count,
            scalars.as_ptr(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

#[derive(Debug)]
pub struct Bls12_381MultiopenPreparation {
    pub q_polys: Vec<Bls12_381Fr>,
    pub query_evals: Vec<Bls12_381Fr>,
}

#[derive(Debug)]
pub struct Bls12_381ResidentMultiopenPreparation {
    pub q_polys: Bls12_381FrDeviceVec,
    pub query_evals: Vec<Bls12_381Fr>,
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_fr_multiopen_prepare(
    polys: &[Bls12_381Fr],
    poly_len: usize,
    q_offsets: &[u32],
    q_poly_indices: &[u32],
    q_coefficients: &[Bls12_381Fr],
    points: &[Bls12_381Fr],
    query_poly_indices: &[u32],
    query_point_indices: &[u32],
) -> Result<Bls12_381MultiopenPreparation, MsmError> {
    if poly_len == 0 || polys.is_empty() || polys.len() % poly_len != 0 {
        return Err(MsmError::LengthMismatch {
            points: polys.len(),
            scalars: poly_len,
        });
    }
    if q_offsets.len() < 2
        || q_poly_indices.len() != q_coefficients.len()
        || q_offsets.last().copied().map(|value| value as usize) != Some(q_poly_indices.len())
        || points.is_empty()
        || query_poly_indices.len() != query_point_indices.len()
    {
        return Err(MsmError::LengthMismatch {
            points: q_poly_indices.len() + query_poly_indices.len(),
            scalars: q_coefficients.len() + query_point_indices.len(),
        });
    }

    let q_count = q_offsets.len() - 1;
    let query_count = query_poly_indices.len();
    let mut q_polys = vec![Bls12_381Fr::default(); q_count * poly_len];
    let mut query_evals = vec![Bls12_381Fr::default(); query_count];
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        slice_bytes::<Bls12_381Fr>(polys.len() + q_coefficients.len() + points.len())
            + slice_bytes::<u32>(
                q_offsets.len()
                    + q_poly_indices.len()
                    + query_poly_indices.len()
                    + query_point_indices.len(),
            ),
        slice_bytes::<Bls12_381Fr>(q_polys.len() + query_evals.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_multiopen_prepare(
            q_polys.as_mut_ptr(),
            query_evals.as_mut_ptr(),
            polys.as_ptr(),
            poly_len,
            polys.len() / poly_len,
            q_offsets.as_ptr(),
            q_offsets.len(),
            q_poly_indices.as_ptr(),
            q_coefficients.as_ptr(),
            q_poly_indices.len(),
            points.as_ptr(),
            points.len(),
            query_poly_indices.as_ptr(),
            query_point_indices.as_ptr(),
            query_count,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(Bls12_381MultiopenPreparation {
            q_polys,
            query_evals,
        })
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_fr_multiopen_prepare_device(
    dynamic: &Bls12_381FrDeviceVec,
    fixed: &Bls12_381FrDeviceVec,
    extra: &Bls12_381FrDeviceVec,
    poly_len: usize,
    poly_refs: &[RawDevicePolyRef],
    q_offsets: &[u32],
    q_poly_indices: &[u32],
    q_coefficients: &[Bls12_381Fr],
    points: &[Bls12_381Fr],
    query_poly_indices: &[u32],
    query_point_indices: &[u32],
) -> Result<Bls12_381MultiopenPreparation, MsmError> {
    if poly_len == 0
        || poly_refs.is_empty()
        || q_offsets.len() < 2
        || q_poly_indices.len() != q_coefficients.len()
        || q_offsets.last().copied().map(|value| value as usize) != Some(q_poly_indices.len())
        || points.is_empty()
        || query_poly_indices.len() != query_point_indices.len()
    {
        return Err(MsmError::LengthMismatch {
            points: q_poly_indices.len() + query_poly_indices.len(),
            scalars: q_coefficients.len() + query_point_indices.len(),
        });
    }

    let q_count = q_offsets.len() - 1;
    let query_count = query_poly_indices.len();
    let mut q_polys = vec![Bls12_381Fr::default(); q_count * poly_len];
    let mut query_evals = vec![Bls12_381Fr::default(); query_count];
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        slice_bytes::<Bls12_381Fr>(q_coefficients.len() + points.len())
            + slice_bytes::<RawDevicePolyRef>(poly_refs.len())
            + slice_bytes::<u32>(
                q_offsets.len()
                    + q_poly_indices.len()
                    + query_poly_indices.len()
                    + query_point_indices.len(),
            ),
        slice_bytes::<Bls12_381Fr>(q_polys.len() + query_evals.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_multiopen_prepare_device(
            q_polys.as_mut_ptr(),
            std::ptr::null_mut(),
            query_evals.as_mut_ptr(),
            dynamic.raw.as_ptr(),
            fixed.raw.as_ptr(),
            extra.raw.as_ptr(),
            poly_len,
            poly_refs.as_ptr(),
            poly_refs.len(),
            q_offsets.as_ptr(),
            q_offsets.len(),
            q_poly_indices.as_ptr(),
            q_coefficients.as_ptr(),
            q_poly_indices.len(),
            points.as_ptr(),
            points.len(),
            query_poly_indices.as_ptr(),
            query_point_indices.as_ptr(),
            query_count,
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(Bls12_381MultiopenPreparation {
            q_polys,
            query_evals,
        })
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_fr_multiopen_prepare_device_resident(
    dynamic: &Bls12_381FrDeviceVec,
    fixed: &Bls12_381FrDeviceVec,
    extra: &Bls12_381FrDeviceVec,
    poly_len: usize,
    poly_refs: &[RawDevicePolyRef],
    q_offsets: &[u32],
    q_poly_indices: &[u32],
    q_coefficients: &[Bls12_381Fr],
    points: &[Bls12_381Fr],
    query_poly_indices: &[u32],
    query_point_indices: &[u32],
) -> Result<Bls12_381ResidentMultiopenPreparation, MsmError> {
    if poly_len == 0
        || poly_refs.is_empty()
        || q_offsets.len() < 2
        || q_poly_indices.len() != q_coefficients.len()
        || q_offsets.last().copied().map(|value| value as usize) != Some(q_poly_indices.len())
        || points.is_empty()
        || query_poly_indices.len() != query_point_indices.len()
    {
        return Err(MsmError::LengthMismatch {
            points: q_poly_indices.len() + query_poly_indices.len(),
            scalars: q_coefficients.len() + query_point_indices.len(),
        });
    }

    let q_count = q_offsets.len() - 1;
    let query_count = query_poly_indices.len();
    let mut query_evals = vec![Bls12_381Fr::default(); query_count];
    let mut q_device = std::ptr::null_mut();
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        slice_bytes::<Bls12_381Fr>(q_coefficients.len() + points.len())
            + slice_bytes::<RawDevicePolyRef>(poly_refs.len())
            + slice_bytes::<u32>(
                q_offsets.len()
                    + q_poly_indices.len()
                    + query_poly_indices.len()
                    + query_point_indices.len(),
            ),
        slice_bytes::<Bls12_381Fr>(query_evals.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_fr_multiopen_prepare_device(
            std::ptr::null_mut(),
            &mut q_device,
            query_evals.as_mut_ptr(),
            dynamic.raw.as_ptr(),
            fixed.raw.as_ptr(),
            extra.raw.as_ptr(),
            poly_len,
            poly_refs.as_ptr(),
            poly_refs.len(),
            q_offsets.as_ptr(),
            q_offsets.len(),
            q_poly_indices.as_ptr(),
            q_coefficients.as_ptr(),
            q_poly_indices.len(),
            points.as_ptr(),
            points.len(),
            query_poly_indices.as_ptr(),
            query_point_indices.as_ptr(),
            query_count,
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(q_device).ok_or_else(|| {
        MsmError::Sppark("sppark returned a null resident q-polynomial vector".to_string())
    })?;
    Ok(Bls12_381ResidentMultiopenPreparation {
        q_polys: Bls12_381FrDeviceVec {
            raw,
            len: q_count * poly_len,
        },
        query_evals,
    })
}

pub fn bls12_381_fr_multiopen_f_poly_device(
    q_polys: &Bls12_381FrDeviceVec,
    poly_len: usize,
    point_offsets: &[u32],
    points: &[Bls12_381Fr],
    combination_scalars: &[Bls12_381Fr],
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    let q_count = point_offsets.len().saturating_sub(1);
    if poly_len < 2
        || q_count == 0
        || q_polys.len() != q_count.saturating_mul(poly_len)
        || combination_scalars.len() != q_count
        || point_offsets.first().copied() != Some(0)
        || point_offsets.last().copied().map(|value| value as usize) != Some(points.len())
    {
        return Err(MsmError::LengthMismatch {
            points: q_polys.len(),
            scalars: q_count.saturating_mul(poly_len),
        });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        slice_bytes::<u32>(point_offsets.len())
            + slice_bytes::<Bls12_381Fr>(points.len() + combination_scalars.len()),
        0,
    );
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_multiopen_f_poly_device(
            &mut raw,
            q_polys.raw.as_ptr(),
            poly_len,
            point_offsets.as_ptr(),
            point_offsets.len(),
            points.as_ptr(),
            points.len(),
            combination_scalars.as_ptr(),
            combination_scalars.len(),
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw).ok_or_else(|| {
        MsmError::Sppark("sppark returned a null resident f-polynomial".to_string())
    })?;
    Ok(Bls12_381FrDeviceVec { raw, len: poly_len })
}

pub fn bls12_381_fr_device_eval_polys(
    polynomials: &Bls12_381FrDeviceVec,
    poly_len: usize,
    poly_count: usize,
    point: Bls12_381Fr,
) -> Result<Vec<Bls12_381Fr>, MsmError> {
    if poly_len == 0 || poly_count == 0 || polynomials.len() != poly_len.saturating_mul(poly_count)
    {
        return Err(MsmError::LengthMismatch {
            points: polynomials.len(),
            scalars: poly_len.saturating_mul(poly_count),
        });
    }
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        0,
        slice_bytes::<Bls12_381Fr>(poly_count),
    );
    let mut out = vec![Bls12_381Fr::default(); poly_count];
    let status = unsafe {
        midnight_sppark_bls12_381_fr_device_eval_polys(
            out.as_mut_ptr(),
            polynomials.raw.as_ptr(),
            poly_len,
            poly_count,
            point,
        )
    };
    drop(lock);
    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_multiopen_pi_device(
    q_polys: &Bls12_381FrDeviceVec,
    f_poly: &Bls12_381FrDeviceVec,
    poly_len: usize,
    combination_scalars: &[Bls12_381Fr],
    point: Bls12_381Fr,
) -> Result<Bls12_381FrDeviceVec, MsmError> {
    let q_count = combination_scalars.len().saturating_sub(1);
    if poly_len < 2
        || q_count == 0
        || q_polys.len() != q_count.saturating_mul(poly_len)
        || f_poly.len() != poly_len
    {
        return Err(MsmError::LengthMismatch {
            points: q_polys.len() + f_poly.len(),
            scalars: (q_count + 1).saturating_mul(poly_len),
        });
    }
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrMultiopenPrepare,
        slice_bytes::<Bls12_381Fr>(combination_scalars.len()),
        0,
    );
    let mut raw = std::ptr::null_mut();
    let status = unsafe {
        midnight_sppark_bls12_381_fr_multiopen_pi_device(
            &mut raw,
            q_polys.raw.as_ptr(),
            f_poly.raw.as_ptr(),
            poly_len,
            combination_scalars.as_ptr(),
            combination_scalars.len(),
            point,
        )
    };
    drop(lock);
    if status.code != 0 {
        return Err(MsmError::Sppark(String::from(status)));
    }
    let raw = NonNull::new(raw)
        .ok_or_else(|| MsmError::Sppark("sppark returned a null pi-polynomial".to_string()))?;
    Ok(Bls12_381FrDeviceVec {
        raw,
        len: poly_len - 1,
    })
}

pub fn bls12_381_fr_poseidon_hash_chain(
    initial: Bls12_381Fr,
    tail0: Bls12_381Fr,
    tail1: Bls12_381Fr,
    input_len: Bls12_381Fr,
    mds: &[Bls12_381Fr],
    round_constants: &[Bls12_381Fr],
    len: usize,
) -> Result<Vec<Bls12_381Fr>, MsmError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if mds.len() != 9 || round_constants.len() != 68 * 3 {
        return Err(MsmError::LengthMismatch {
            points: 9 + (68 * 3),
            scalars: mds.len() + round_constants.len(),
        });
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrPoseidonHashChain,
        slice_bytes::<Bls12_381Fr>(mds.len() + round_constants.len() + 4),
        slice_bytes::<Bls12_381Fr>(len),
    );
    let mut out = vec![Bls12_381Fr::default(); len];
    let status = unsafe {
        midnight_sppark_bls12_381_fr_poseidon_hash_chain(
            out.as_mut_ptr(),
            len,
            initial,
            tail0,
            tail1,
            input_len,
            mds.as_ptr(),
            mds.len(),
            round_constants.as_ptr(),
            round_constants.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

/// Number of field elements emitted for the two Poseidon permutations in one
/// three-element hash.
pub const BLS12_381_POSEIDON_HASH_CHAIN_WITNESS_LEN: usize = 256;

pub fn bls12_381_fr_poseidon_hash_chain_witness(
    initial: Bls12_381Fr,
    tail0: Bls12_381Fr,
    tail1: Bls12_381Fr,
    input_len: Bls12_381Fr,
    mds: &[Bls12_381Fr],
    round_constants: &[Bls12_381Fr],
    len: usize,
) -> Result<Vec<Bls12_381Fr>, MsmError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if mds.len() != 9 || round_constants.len() != 68 * 3 {
        return Err(MsmError::LengthMismatch {
            points: 9 + (68 * 3),
            scalars: mds.len() + round_constants.len(),
        });
    }
    let witness_len = len
        .checked_mul(BLS12_381_POSEIDON_HASH_CHAIN_WITNESS_LEN)
        .ok_or_else(|| MsmError::Sppark("Poseidon witness length overflow".to_string()))?;

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrPoseidonHashChainWitness,
        slice_bytes::<Bls12_381Fr>(mds.len() + round_constants.len() + 4),
        slice_bytes::<Bls12_381Fr>(witness_len),
    );
    let mut out = vec![Bls12_381Fr::default(); witness_len];
    let status = unsafe {
        midnight_sppark_bls12_381_fr_poseidon_hash_chain_witness(
            out.as_mut_ptr(),
            len,
            initial,
            tail0,
            tail1,
            input_len,
            mds.as_ptr(),
            mds.len(),
            round_constants.as_ptr(),
            round_constants.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(out)
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_invert_assign(values: &mut [Bls12_381Fr]) -> Result<(), MsmError> {
    if values.is_empty() {
        return Ok(());
    }

    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrInvertAssign,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status =
        unsafe { midnight_sppark_bls12_381_fr_invert_assign(values.as_mut_ptr(), values.len()) };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

pub fn bls12_381_fr_sort_canonical(values: &mut [Bls12_381Fr]) -> Result<(), MsmError> {
    if values.len() < 2 {
        return Ok(());
    }
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::FrSortCanonical,
        slice_bytes::<Bls12_381Fr>(values.len()),
        slice_bytes::<Bls12_381Fr>(values.len()),
    );
    let status =
        unsafe { midnight_sppark_bls12_381_fr_sort_canonical(values.as_mut_ptr(), values.len()) };
    drop(lock);
    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_eval_numerator(
    out: &mut [Bls12_381Fr],
    initial_out: Option<&[Bls12_381Fr]>,
    fixed: &[Bls12_381Fr],
    fixed_cols: usize,
    advice: &[Bls12_381Fr],
    advice_cols: usize,
    instance: &[Bls12_381Fr],
    instance_cols: usize,
    challenges: &[Bls12_381Fr],
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    theta: Bls12_381Fr,
    trash_challenge: Bls12_381Fr,
    y: Bls12_381Fr,
    omega: Bls12_381Fr,
    delta_start: Bls12_381Fr,
    delta_const: Bls12_381Fr,
    rot_scale: i32,
    last_rotation: i32,
    chunk_len: u32,
    l0: &[Bls12_381Fr],
    l_last: &[Bls12_381Fr],
    l_active: &[Bls12_381Fr],
    permutation_products: &[Bls12_381Fr],
    permutation_sets: usize,
    permutation_columns: &[RawColumnRef],
    permutation_pk_cosets: &[Bls12_381Fr],
    permutation_pk_cosets_len: usize,
    lookup_product_cosets: &[Bls12_381Fr],
    lookup_input_cosets: &[Bls12_381Fr],
    lookup_table_cosets: &[Bls12_381Fr],
    lookup_count: usize,
    trash_cosets: &[Bls12_381Fr],
    trash_selector_fixed_indices: &[u32],
    custom_graph: RawGraph,
    lookup_graphs: &[RawGraph],
    trash_graphs: &[RawGraph],
) -> Result<(), MsmError> {
    if out.is_empty() {
        return Ok(());
    }
    if let Some(initial_out) = initial_out {
        if initial_out.len() != out.len() {
            return Err(MsmError::LengthMismatch {
                points: out.len(),
                scalars: initial_out.len(),
            });
        }
    }

    let lock = maybe_gpu_msm_lock()?;
    let custom_graph_bytes = if initial_out.is_none() {
        raw_graph_h2d_bytes(custom_graph)
    } else {
        0
    };
    let graph_bytes = custom_graph_bytes
        + lookup_graphs
            .iter()
            .copied()
            .map(raw_graph_h2d_bytes)
            .sum::<usize>()
        + trash_graphs
            .iter()
            .copied()
            .map(raw_graph_h2d_bytes)
            .sum::<usize>();
    record_transfer(
        TransferOp::EvalNumerator,
        initial_out.map_or(0, |values| slice_bytes::<Bls12_381Fr>(values.len()))
            + slice_bytes::<Bls12_381Fr>(fixed.len())
            + slice_bytes::<Bls12_381Fr>(advice.len())
            + slice_bytes::<Bls12_381Fr>(instance.len())
            + slice_bytes::<Bls12_381Fr>(challenges.len())
            + slice_bytes::<Bls12_381Fr>(l0.len())
            + slice_bytes::<Bls12_381Fr>(l_last.len())
            + slice_bytes::<Bls12_381Fr>(l_active.len())
            + slice_bytes::<Bls12_381Fr>(permutation_products.len())
            + slice_bytes::<RawColumnRef>(permutation_columns.len())
            + slice_bytes::<Bls12_381Fr>(permutation_pk_cosets.len())
            + slice_bytes::<Bls12_381Fr>(lookup_product_cosets.len())
            + slice_bytes::<Bls12_381Fr>(lookup_input_cosets.len())
            + slice_bytes::<Bls12_381Fr>(lookup_table_cosets.len())
            + slice_bytes::<Bls12_381Fr>(trash_cosets.len())
            + slice_bytes::<u32>(trash_selector_fixed_indices.len())
            + graph_bytes,
        slice_bytes::<Bls12_381Fr>(out.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_eval_numerator(
            out.as_mut_ptr(),
            out.len(),
            initial_out
                .map(|values| values.as_ptr())
                .unwrap_or(std::ptr::null()),
            u32::from(initial_out.is_none()),
            fixed.as_ptr(),
            fixed_cols,
            advice.as_ptr(),
            advice_cols,
            instance.as_ptr(),
            instance_cols,
            challenges.as_ptr(),
            challenges.len(),
            beta,
            gamma,
            theta,
            trash_challenge,
            y,
            omega,
            delta_start,
            delta_const,
            rot_scale,
            last_rotation,
            chunk_len,
            l0.as_ptr(),
            l_last.as_ptr(),
            l_active.as_ptr(),
            permutation_products.as_ptr(),
            permutation_sets,
            permutation_columns.as_ptr(),
            permutation_columns.len(),
            permutation_pk_cosets.as_ptr(),
            permutation_pk_cosets_len,
            lookup_product_cosets.as_ptr(),
            lookup_input_cosets.as_ptr(),
            lookup_table_cosets.as_ptr(),
            lookup_count,
            trash_cosets.as_ptr(),
            trash_selector_fixed_indices.as_ptr(),
            trash_selector_fixed_indices.len(),
            custom_graph,
            lookup_graphs.as_ptr(),
            trash_graphs.as_ptr(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bls12_381_eval_graph_batch(
    out: &mut [Bls12_381Fr],
    len: usize,
    fixed: &[Bls12_381Fr],
    fixed_cols: usize,
    advice: &[Bls12_381Fr],
    advice_cols: usize,
    instance: &[Bls12_381Fr],
    instance_cols: usize,
    challenges: &[Bls12_381Fr],
    beta: Bls12_381Fr,
    gamma: Bls12_381Fr,
    theta: Bls12_381Fr,
    trash_challenge: Bls12_381Fr,
    y: Bls12_381Fr,
    rot_scale: i32,
    graphs: &[RawGraph],
) -> Result<(), MsmError> {
    if len == 0 || graphs.is_empty() {
        return Ok(());
    }
    if out.len() != len.saturating_mul(graphs.len())
        || fixed.len() != len.saturating_mul(fixed_cols)
        || advice.len() != len.saturating_mul(advice_cols)
        || instance.len() != len.saturating_mul(instance_cols)
    {
        return Err(MsmError::LengthMismatch {
            points: len.saturating_mul(graphs.len()),
            scalars: out.len(),
        });
    }

    let graph_bytes = graphs
        .iter()
        .copied()
        .map(raw_graph_h2d_bytes)
        .sum::<usize>();
    let h2d_bytes =
        slice_bytes::<Bls12_381Fr>(fixed.len() + advice.len() + instance.len() + challenges.len())
            + graph_bytes;
    let lock = maybe_gpu_msm_lock()?;
    record_transfer(
        TransferOp::EvalGraphBatch,
        h2d_bytes,
        slice_bytes::<Bls12_381Fr>(out.len()),
    );
    let status = unsafe {
        midnight_sppark_bls12_381_eval_graph_batch(
            out.as_mut_ptr(),
            len,
            fixed.as_ptr(),
            fixed_cols,
            advice.as_ptr(),
            advice_cols,
            instance.as_ptr(),
            instance_cols,
            challenges.as_ptr(),
            challenges.len(),
            beta,
            gamma,
            theta,
            trash_challenge,
            y,
            rot_scale,
            graphs.as_ptr(),
            graphs.len(),
        )
    };
    drop(lock);

    if status.code == 0 {
        Ok(())
    } else {
        Err(MsmError::Sppark(String::from(status)))
    }
}

fn maybe_gpu_msm_lock() -> Result<Option<MutexGuard<'static, ()>>, MsmError> {
    let execution_lane = current_execution_lane() as usize;
    if execution_lane != 0 {
        return execution_lane_locks()[execution_lane]
            .lock()
            .map(Some)
            .map_err(|_| MsmError::LockPoisoned);
    }

    let lock_enabled = !matches!(
        std::env::var("MIDNIGHT_GPU_SPPARK_LOCK")
            .unwrap_or_else(|_| "on".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off" | "none" | "unlocked"
    );
    if lock_enabled {
        gpu_msm_lock()
            .lock()
            .map(Some)
            .map_err(|_| MsmError::LockPoisoned)
    } else {
        Ok(None)
    }
}

fn execution_lane_locks() -> &'static [Mutex<()>] {
    static LOCKS: OnceLock<Vec<Mutex<()>>> = OnceLock::new();
    LOCKS.get_or_init(|| {
        (0..=MAX_SPPARK_EXECUTION_LANES)
            .map(|_| Mutex::new(()))
            .collect()
    })
}

fn gpu_msm_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use ff::{Field, PrimeField};
    use group::Group;
    use midnight_curves::{Fq, G1Affine, G1Projective};

    use super::*;

    #[test]
    fn msm_config_profiles_are_domain_scoped() {
        let k13 = Bls12_381MsmConfig::for_k(13);
        let k14 = Bls12_381MsmConfig::for_k(14);
        let k15 = Bls12_381MsmConfig::for_k(15);
        let k16 = Bls12_381MsmConfig::for_k(16);

        assert_eq!(k13.domain_k, 13);
        assert_eq!(k14.domain_k, 14);
        assert_eq!(k15.domain_k, 15);
        assert_eq!(k13.window_bits, 10);
        assert_eq!(k14.window_bits, 10);
        assert_eq!(k15.window_bits, 10);
        assert_eq!(k16.window_bits, 12);
        assert_eq!(k15.min_msm_scalars, 1 << 13);
        assert_eq!(k16.min_msm_scalars, 100_000);
        assert_eq!(k15.unfiltered_min_scalars, 1 << 14);
        assert_eq!(k16.unfiltered_min_scalars, 1 << 15);
        assert_eq!(k15.sort_grid_divisor, 3);
        assert_eq!(k15.sort_window_lanes, 2);
        assert_eq!(k16.sort_window_lanes, 2);
        assert_eq!(k13.integration_blocks, 1);
        assert_eq!(k14.integration_blocks, 1);
        assert_eq!(k15.integration_blocks, 1);
        assert_eq!(k16.integration_blocks, 1);
        assert_eq!(k13.batch_lanes, 6);
        assert_eq!(k14.batch_lanes, 6);
        assert_eq!(k15.batch_lanes, 6);
        assert_eq!(k15.reuse_event, 1);
        assert_eq!(k15.device_reduce, 1);
        assert_eq!(k13.device_reduce, 0);
        assert_eq!(Bls12_381MsmConfig::for_point_count(1 << 15), k15);
        assert!(k15.validate().is_ok());
        assert!(
            Bls12_381MsmConfig {
                window_bits: 19,
                ..k15
            }
            .validate()
            .is_err()
        );
    }

    fn raw_from_fq(value: Fq) -> Bls12_381Fr {
        assert_eq!(
            std::mem::size_of::<Fq>(),
            std::mem::size_of::<Bls12_381Fr>()
        );
        unsafe { std::mem::transmute(value) }
    }

    fn fq_from_raw(value: Bls12_381Fr) -> Fq {
        assert_eq!(
            std::mem::size_of::<Fq>(),
            std::mem::size_of::<Bls12_381Fr>()
        );
        unsafe { std::mem::transmute(value) }
    }

    fn assert_shifted_prefix_product_matches_cpu(len: usize) {
        let initial = Fq::from(7);
        let input = (0..len)
            .map(|idx| Fq::from((idx as u64 % 251) + 2))
            .collect::<Vec<_>>();

        let mut expected = Vec::with_capacity(input.len());
        if !input.is_empty() {
            expected.push(initial);
            for idx in 1..input.len() {
                expected.push(expected[idx - 1] * input[idx - 1]);
            }
        }

        let mut values = input.iter().copied().map(raw_from_fq).collect::<Vec<_>>();
        bls12_381_fr_shifted_prefix_product(&mut values, raw_from_fq(initial)).unwrap();
        let actual = values.into_iter().map(fq_from_raw).collect::<Vec<_>>();
        assert_eq!(actual, expected, "len={len}");
    }

    fn assert_poly_lincomb_matches_cpu(poly_len: usize, poly_count: usize) {
        let polys = (0..poly_count)
            .flat_map(|poly_idx| {
                (0..poly_len)
                    .map(move |idx| Fq::from(((poly_idx * 17 + idx * 31 + 5) % 65521) as u64))
            })
            .collect::<Vec<_>>();
        let scalars = (0..poly_count)
            .map(|idx| Fq::from((idx as u64 * 11) + 3))
            .collect::<Vec<_>>();
        let mut expected = vec![Fq::ZERO; poly_len];
        for poly_idx in 0..poly_count {
            for idx in 0..poly_len {
                expected[idx] += polys[(poly_idx * poly_len) + idx] * scalars[poly_idx];
            }
        }

        let raw_polys = polys.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        let raw_scalars = scalars.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        let mut raw_out = vec![Bls12_381Fr::default(); poly_len];
        bls12_381_fr_poly_lincomb(&mut raw_out, &raw_polys, poly_len, poly_count, &raw_scalars)
            .unwrap();
        let actual = raw_out.into_iter().map(fq_from_raw).collect::<Vec<_>>();

        assert_eq!(
            actual, expected,
            "poly_len={poly_len}, poly_count={poly_count}"
        );
    }

    fn assert_device_vec_ntt_matches_host_bridge(len: usize) {
        let input = (0..len)
            .map(|idx| Fq::from(((idx * 13 + 7) % 65521) as u64))
            .collect::<Vec<_>>();
        let mut expected = input.iter().copied().map(raw_from_fq).collect::<Vec<_>>();
        bls12_381_fr_ntt_zeta(
            &mut expected,
            false,
            raw_from_fq(Fq::ONE),
            raw_from_fq(Fq::ONE),
            raw_from_fq(Fq::ONE),
        )
        .unwrap();

        let raw_input = input.iter().copied().map(raw_from_fq).collect::<Vec<_>>();
        let mut device = Bls12_381FrDeviceVec::upload(&raw_input).unwrap();
        assert_eq!(device.len(), len);
        device
            .ntt_zeta(
                false,
                raw_from_fq(Fq::ONE),
                raw_from_fq(Fq::ONE),
                raw_from_fq(Fq::ONE),
            )
            .unwrap();
        let actual = device.download().unwrap();
        assert_eq!(actual, expected, "len={len}");
    }

    fn raw_sequence(len: usize, offset: usize) -> Vec<Bls12_381Fr> {
        (0..len)
            .map(|idx| raw_from_fq(Fq::from(((idx * 29 + offset * 43 + 11) % 65521) as u64)))
            .collect()
    }

    fn assert_resident_coeff_to_extended_matches_host_bridge(
        input_len: usize,
        count: usize,
        extended_log_n: u32,
    ) {
        let input = raw_sequence(input_len * count, count);
        let output_len = 1usize << extended_log_n;
        let first = raw_from_fq(Fq::from(17));
        let second = raw_from_fq(Fq::from(23));
        let mut expected = vec![Bls12_381Fr::default(); output_len * count];
        bls12_381_fr_coeff_to_extended_batch(
            &mut expected,
            &input,
            input_len,
            count,
            extended_log_n,
            first,
            second,
        )
        .unwrap();

        let actual = Bls12_381FrDeviceVec::coeffs_to_extended(
            &input,
            input_len,
            count,
            extended_log_n,
            first,
            second,
        )
        .unwrap()
        .download()
        .unwrap();

        assert_eq!(actual, expected, "input_len={input_len}, count={count}");
    }

    fn assert_resident_permutation_products_match_cpu(len: usize, column_count: usize) {
        let chunk_len = 2;
        let set_count = column_count.div_ceil(chunk_len);
        let blinding_factors = 4;
        let beta = Fq::from(17);
        let gamma = Fq::from(23);
        let mut omega = Fq::ROOT_OF_UNITY;
        let log_n = len.trailing_zeros();
        for _ in log_n..Fq::S {
            omega = omega.square();
        }
        let columns = (0..column_count)
            .flat_map(|column| {
                (0..len).map(move |row| Fq::from(((column * 71 + row * 29 + 5) % 65521) as u64))
            })
            .collect::<Vec<_>>();
        let permutations = (0..column_count)
            .flat_map(|column| {
                (0..len).map(move |row| Fq::from(((column * 43 + row * 37 + 101) % 65521) as u64))
            })
            .collect::<Vec<_>>();
        let mut delta = Fq::ONE;
        let delta_powers = (0..column_count)
            .map(|_| {
                let current = delta;
                delta *= Fq::DELTA;
                current
            })
            .collect::<Vec<_>>();
        let blind_values = (0..set_count * blinding_factors)
            .map(|idx| Fq::from((idx as u64 * 13) + 211))
            .collect::<Vec<_>>();

        let mut expected = Vec::with_capacity(set_count * len);
        let mut initial = Fq::ONE;
        for set_idx in 0..set_count {
            let column_offset = set_idx * chunk_len;
            let set_columns = chunk_len.min(column_count - column_offset);
            let mut fractions = Vec::with_capacity(len);
            let mut omega_power = Fq::ONE;
            for row in 0..len {
                let mut numerator = Fq::ONE;
                let mut denominator = Fq::ONE;
                for local in 0..set_columns {
                    let column = column_offset + local;
                    let value = columns[column * len + row];
                    numerator *= value + delta_powers[column] * omega_power * beta + gamma;
                    denominator *= value + beta * permutations[column * len + row] + gamma;
                }
                fractions.push(numerator * denominator.invert().unwrap());
                omega_power *= omega;
            }
            let mut z = Vec::with_capacity(len);
            z.push(initial);
            for row in 1..len {
                z.push(z[row - 1] * fractions[row - 1]);
            }
            initial = z[len - (blinding_factors + 1)];
            z[len - blinding_factors..].copy_from_slice(
                &blind_values[set_idx * blinding_factors..(set_idx + 1) * blinding_factors],
            );
            expected.extend(z);
        }

        let raw_columns = columns.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        let raw_permutations = permutations
            .into_iter()
            .map(raw_from_fq)
            .collect::<Vec<_>>();
        let raw_delta = delta_powers
            .into_iter()
            .map(raw_from_fq)
            .collect::<Vec<_>>();
        let raw_blinds = blind_values
            .into_iter()
            .map(raw_from_fq)
            .collect::<Vec<_>>();
        let mut actual = bls12_381_fr_permutation_products(
            &raw_columns,
            &raw_permutations,
            len,
            chunk_len,
            raw_from_fq(beta),
            raw_from_fq(gamma),
            raw_from_fq(omega),
            &raw_delta,
            &raw_blinds,
            blinding_factors,
        )
        .unwrap();
        assert_eq!(
            actual
                .download()
                .unwrap()
                .into_iter()
                .map(fq_from_raw)
                .collect::<Vec<_>>(),
            expected
        );

        let one = raw_from_fq(Fq::ONE);
        let mut expected_coeffs = expected.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        for poly in expected_coeffs.chunks_mut(len) {
            bls12_381_fr_ntt_zeta(poly, true, one, one, one).unwrap();
        }
        actual
            .ntt_zeta_batch(len, set_count, true, one, one, one)
            .unwrap();
        assert_eq!(actual.download().unwrap(), expected_coeffs);
    }

    fn assert_resident_lookup_products_match_cpu(len: usize, lookup_count: usize) {
        let blinding_factors = 4;
        let beta = Fq::from(17);
        let gamma = Fq::from(23);
        let sequence = |offset: usize| {
            (0..lookup_count * len)
                .map(|idx| Fq::from(((idx * 29 + offset * 43 + 101) % 65521) as u64))
                .collect::<Vec<_>>()
        };
        let compressed_inputs = sequence(1);
        let compressed_tables = sequence(2);
        let permuted_inputs = sequence(3);
        let permuted_tables = sequence(4);
        let blind_values = (0..lookup_count * blinding_factors)
            .map(|idx| Fq::from((idx as u64 * 13) + 211))
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(lookup_count * len);
        for lookup_idx in 0..lookup_count {
            let offset = lookup_idx * len;
            let fractions = (0..len)
                .map(|idx| {
                    let idx = offset + idx;
                    (compressed_inputs[idx] + beta)
                        * (compressed_tables[idx] + gamma)
                        * ((permuted_inputs[idx] + beta) * (permuted_tables[idx] + gamma))
                            .invert()
                            .unwrap()
                })
                .collect::<Vec<_>>();
            let mut z = Vec::with_capacity(len);
            z.push(Fq::ONE);
            for row in 1..len {
                z.push(z[row - 1] * fractions[row - 1]);
            }
            z[len - blinding_factors..].copy_from_slice(
                &blind_values[lookup_idx * blinding_factors..(lookup_idx + 1) * blinding_factors],
            );
            expected.extend(z);
        }

        let raw_values = |values: Vec<Fq>| values.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        let actual = bls12_381_fr_lookup_products(
            &raw_values(compressed_inputs),
            &raw_values(compressed_tables),
            &raw_values(permuted_inputs),
            &raw_values(permuted_tables),
            len,
            raw_from_fq(beta),
            raw_from_fq(gamma),
            &raw_values(blind_values),
            blinding_factors,
        )
        .unwrap()
        .download()
        .unwrap()
        .into_iter()
        .map(fq_from_raw)
        .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn shifted_prefix_product_matches_cpu() {
        for len in [0, 1, 2, 17, 1024, 1025, 4099, 32768] {
            assert_shifted_prefix_product_matches_cpu(len);
        }
    }

    #[test]
    fn poly_lincomb_matches_cpu() {
        for (poly_len, poly_count) in [(1, 1), (17, 3), (4096, 5)] {
            assert_poly_lincomb_matches_cpu(poly_len, poly_count);
        }
    }

    #[test]
    fn device_vec_ntt_matches_host_bridge() {
        for len in [2, 1024, 4096] {
            assert_device_vec_ntt_matches_host_bridge(len);
        }
    }

    #[test]
    fn resident_coeff_to_extended_matches_host_bridge() {
        for (input_len, count, extended_log_n) in [(8, 1, 4), (8, 3, 4), (1024, 4, 12)] {
            assert_resident_coeff_to_extended_matches_host_bridge(input_len, count, extended_log_n);
        }
    }

    #[test]
    fn resident_divide_ntt_matches_host_bridge() {
        let input_len = 1024;
        let extended_log_n = 12;
        let first = raw_from_fq(Fq::from(17));
        let second = raw_from_fq(Fq::from(23));
        let scale = raw_from_fq(Fq::from(31));
        let input = raw_sequence(input_len, 7);
        let factors = raw_sequence(8, 13);
        let mut expected = vec![Bls12_381Fr::default(); 1usize << extended_log_n];
        bls12_381_fr_coeff_to_extended_batch(
            &mut expected,
            &input,
            input_len,
            1,
            extended_log_n,
            first,
            second,
        )
        .unwrap();

        let mut actual = Bls12_381FrDeviceVec::coeffs_to_extended(
            &input,
            input_len,
            1,
            extended_log_n,
            first,
            second,
        )
        .unwrap();
        bls12_381_fr_divide_ntt_zeta(&mut expected, &factors, scale, first, second).unwrap();
        actual
            .divide_ntt_zeta(&factors, scale, first, second)
            .unwrap();

        assert_eq!(actual.download().unwrap(), expected);
    }

    #[test]
    fn resident_permutation_products_match_cpu() {
        assert_resident_permutation_products_match_cpu(1024, 5);
    }

    #[test]
    fn resident_lookup_products_match_cpu() {
        assert_resident_lookup_products_match_cpu(1024, 4);
    }

    #[test]
    fn ec_gpu_msm_batch_matches_cpu() {
        let len = 4096;
        let points = (0..len)
            .map(|idx| G1Projective::generator() * Fq::from(idx as u64 + 1))
            .collect::<Vec<_>>();
        let raw_points = points
            .iter()
            .map(|point| {
                let affine = G1Affine::from(point);
                let raw: &blst::blst_p1_affine = affine.as_ref();
                Bls12_381G1Affine {
                    x: raw.x.l,
                    y: raw.y.l,
                }
            })
            .collect::<Vec<_>>();
        let scalar_batches = (0..2)
            .map(|batch| {
                (0..len)
                    .map(|idx| {
                        Fq::from((idx as u64 * 29 + batch as u64 * 43 + 11) % 65521 + 1)
                            .invert()
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_reprs = scalar_batches
            .iter()
            .map(|scalars| scalars.iter().map(PrimeField::to_repr).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let scalar_slices = scalar_reprs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let context = bls12_381_g1_msm_context(&raw_points).unwrap();
        let actual = bls12_381_g1_msm_context_ec_gpu_batch(&context, &scalar_slices).unwrap();

        for (actual, scalars) in actual.iter().zip(&scalar_batches) {
            let expected = G1Projective::multi_exp(&points, scalars);
            let actual = blst::blst_p1 {
                x: blst::blst_fp { l: actual.x },
                y: blst::blst_fp { l: actual.y },
                z: blst::blst_fp { l: actual.z },
            };
            let expected: &blst::blst_p1 = expected.as_ref();
            assert!(unsafe { blst::blst_p1_is_equal(&actual, expected) });
        }
    }

    #[test]
    fn scrambled_device_msm_batch_matches_cpu() {
        for (len, batch_count) in [(4096, 2), (8192, 3), (16384, 3), (32768, 9), (65536, 4)] {
            assert_scrambled_device_msm_batch_matches_cpu(len, batch_count);
        }
    }

    #[test]
    fn fully_device_reducer_resets_persistent_context_state() {
        let len = 4096;
        let batch_count = 2;
        let points = (0..len)
            .map(|idx| G1Projective::generator() * Fq::from(idx as u64 + 1))
            .collect::<Vec<_>>();
        let raw_points = points
            .iter()
            .map(|point| {
                let affine = G1Affine::from(point);
                let raw: &blst::blst_p1_affine = affine.as_ref();
                Bls12_381G1Affine {
                    x: raw.x.l,
                    y: raw.y.l,
                }
            })
            .collect::<Vec<_>>();
        let context = bls12_381_g1_msm_context_with_config(
            &raw_points,
            Bls12_381MsmConfig {
                device_reduce: 3,
                ..Bls12_381MsmConfig::for_k(12)
            },
        )
        .unwrap();

        for round in 0..6 {
            let scalar_batches = (0..batch_count)
                .map(|batch| {
                    (0..len)
                        .map(|idx| {
                            Fq::from(
                                ((idx * 29 + batch * 43 + round * 71 + 11) % 65521 + 1)
                                    as u64,
                            )
                            .invert()
                            .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let raw_scalars = scalar_batches
                .iter()
                .flatten()
                .copied()
                .map(raw_from_fq)
                .collect::<Vec<_>>();
            let device = Bls12_381FrDeviceVec::upload(&raw_scalars).unwrap();
            let actual = bls12_381_g1_msm_context_device_scrambled_batch(
                &context,
                &device,
                0,
                batch_count,
            )
            .unwrap();

            for (batch, (actual, scalars)) in actual.iter().zip(&scalar_batches).enumerate() {
                let expected = G1Projective::multi_exp(&points, scalars);
                let expected: &blst::blst_p1 = expected.as_ref();
                let actual = blst::blst_p1 {
                    x: blst::blst_fp { l: actual.x },
                    y: blst::blst_fp { l: actual.y },
                    z: blst::blst_fp { l: actual.z },
                };
                assert!(
                    unsafe { blst::blst_p1_is_equal(&actual, expected) },
                    "round={round} batch={batch}"
                );
            }
        }
    }

    fn assert_scrambled_device_msm_batch_matches_cpu(len: usize, batch_count: usize) {
        let points = (0..len)
            .map(|idx| G1Projective::generator() * Fq::from(idx as u64 + 1))
            .collect::<Vec<_>>();
        let raw_points = points
            .iter()
            .map(|point| {
                let affine = G1Affine::from(point);
                let raw: &blst::blst_p1_affine = affine.as_ref();
                Bls12_381G1Affine {
                    x: raw.x.l,
                    y: raw.y.l,
                }
            })
            .collect::<Vec<_>>();
        let scalar_batches = (0..batch_count)
            .map(|batch| {
                (0..len)
                    .map(|idx| {
                        if (idx + batch * 3) % 7 == 0 {
                            Fq::from(0u64)
                        } else {
                            Fq::from((idx as u64 * 29 + batch as u64 * 43 + 11) % 65521 + 1)
                                .invert()
                                .unwrap()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let raw_scalars = scalar_batches
            .iter()
            .flatten()
            .copied()
            .map(raw_from_fq)
            .collect::<Vec<_>>();
        let device = Bls12_381FrDeviceVec::upload(&raw_scalars).unwrap();
        let context = bls12_381_g1_msm_context(&raw_points).unwrap();
        let raw_scalar_slices = raw_scalars.chunks_exact(len).collect::<Vec<_>>();
        let host = bls12_381_g1_msm_context_batch(&context, &raw_scalar_slices).unwrap();
        let direct =
            bls12_381_g1_msm_context_device_batch(&context, &device, 0, batch_count).unwrap();
        let scrambled =
            bls12_381_g1_msm_context_device_scrambled_batch(&context, &device, 0, batch_count)
                .unwrap();
        let starts = (0..batch_count)
            .map(|batch| (batch * 977 + 31) % (len / 3).max(1))
            .collect::<Vec<_>>();
        let ragged_slices = starts
            .iter()
            .enumerate()
            .map(|(batch, &start)| Bls12_381MsmSlice {
                scalar_offset: batch * len + start,
                point_offset: start,
                len: len - start,
                nonzero_len: scalar_batches[batch][start..]
                    .iter()
                    .filter(|scalar| !bool::from(scalar.is_zero()))
                    .count(),
            })
            .collect::<Vec<_>>();
        let ragged =
            bls12_381_g1_msm_context_device_ragged(&context, &device, &ragged_slices).unwrap();

        for (((host, direct), scrambled), scalars) in host
            .iter()
            .zip(&direct)
            .zip(&scrambled)
            .zip(&scalar_batches)
        {
            let expected = G1Projective::multi_exp(&points, scalars);
            let expected: &blst::blst_p1 = expected.as_ref();
            for (path, actual) in [("host", host), ("direct", direct), ("scrambled", scrambled)] {
                let actual = blst::blst_p1 {
                    x: blst::blst_fp { l: actual.x },
                    y: blst::blst_fp { l: actual.y },
                    z: blst::blst_fp { l: actual.z },
                };
                assert!(
                    unsafe { blst::blst_p1_is_equal(&actual, expected) },
                    "len={len} path={path}"
                );
            }
        }
        for (((actual, scalars), &start), batch) in ragged
            .iter()
            .zip(&scalar_batches)
            .zip(&starts)
            .zip(0..batch_count)
        {
            let expected = G1Projective::multi_exp(&points[start..], &scalars[start..]);
            let expected: &blst::blst_p1 = expected.as_ref();
            let actual = blst::blst_p1 {
                x: blst::blst_fp { l: actual.x },
                y: blst::blst_fp { l: actual.y },
                z: blst::blst_fp { l: actual.z },
            };
            assert!(
                unsafe { blst::blst_p1_is_equal(&actual, expected) },
                "len={len} path=ragged batch={batch} start={start}"
            );
        }
    }

    #[test]
    fn resident_multiopen_f_poly_matches_cpu() {
        let poly_len = 1024;
        let q_count = 3;
        let q_polys = (0..q_count)
            .flat_map(|poly| {
                (0..poly_len).map(move |idx| Fq::from(((poly * 97 + idx * 31 + 7) % 65521) as u64))
            })
            .collect::<Vec<_>>();
        let point_sets = [
            vec![Fq::from(3)],
            vec![Fq::from(5), Fq::from(7)],
            vec![Fq::from(11)],
        ];
        let mut point_offsets = vec![0u32];
        let mut points = Vec::new();
        for set in &point_sets {
            points.extend(set.iter().copied());
            point_offsets.push(points.len() as u32);
        }
        let combination_scalars = [Fq::ONE, Fq::from(13), Fq::from(17)];
        let divide = |poly: &[Fq], point: Fq| {
            let mut quotient = vec![Fq::ZERO; poly_len];
            quotient[poly_len - 2] = poly[poly_len - 1];
            for idx in (0..poly_len - 2).rev() {
                quotient[idx] = poly[idx + 1] + point * quotient[idx + 1];
            }
            quotient
        };
        let mut expected = vec![Fq::ZERO; poly_len];
        for ((poly, point_set), scalar) in q_polys
            .chunks_exact(poly_len)
            .zip(&point_sets)
            .zip(combination_scalars)
        {
            let divided = point_set
                .iter()
                .copied()
                .fold(poly.to_vec(), |poly, point| divide(&poly, point));
            for (out, value) in expected.iter_mut().zip(divided) {
                *out += value * scalar;
            }
        }

        let raw_q = q_polys.iter().copied().map(raw_from_fq).collect::<Vec<_>>();
        let q_device = Bls12_381FrDeviceVec::upload(&raw_q).unwrap();
        let f_device = bls12_381_fr_multiopen_f_poly_device(
            &q_device,
            poly_len,
            &point_offsets,
            &points.iter().copied().map(raw_from_fq).collect::<Vec<_>>(),
            &combination_scalars
                .iter()
                .copied()
                .map(raw_from_fq)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let actual = f_device
            .download()
            .unwrap()
            .into_iter()
            .map(fq_from_raw)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        let eval = |poly: &[Fq], point: Fq| {
            poly.iter()
                .rev()
                .fold(Fq::ZERO, |acc, coefficient| acc * point + coefficient)
        };
        let x3 = Fq::from(23);
        let actual_evals =
            bls12_381_fr_device_eval_polys(&q_device, poly_len, q_count, raw_from_fq(x3))
                .unwrap()
                .into_iter()
                .map(fq_from_raw)
                .collect::<Vec<_>>();
        let expected_evals = q_polys
            .chunks_exact(poly_len)
            .map(|poly| eval(poly, x3))
            .collect::<Vec<_>>();
        assert_eq!(actual_evals, expected_evals);

        let x4 = Fq::from(19);
        let final_scalars = [Fq::ONE, x4, x4.square(), x4.square() * x4];
        let mut final_poly = vec![Fq::ZERO; poly_len];
        for (poly, scalar) in q_polys
            .chunks_exact(poly_len)
            .zip(final_scalars.iter().copied())
        {
            for (out, value) in final_poly.iter_mut().zip(poly) {
                *out += *value * scalar;
            }
        }
        for (out, value) in final_poly.iter_mut().zip(&expected) {
            *out += *value * final_scalars[q_count];
        }
        let v = eval(&final_poly, x3);
        final_poly[0] -= v;
        let mut expected_pi = divide(&final_poly, x3);
        expected_pi.truncate(poly_len - 1);
        let actual_pi = bls12_381_fr_multiopen_pi_device(
            &q_device,
            &f_device,
            poly_len,
            &final_scalars
                .iter()
                .copied()
                .map(raw_from_fq)
                .collect::<Vec<_>>(),
            raw_from_fq(x3),
        )
        .unwrap()
        .download()
        .unwrap()
        .into_iter()
        .map(fq_from_raw)
        .collect::<Vec<_>>();
        assert_eq!(actual_pi, expected_pi);
    }

    #[test]
    fn invert_assign_matches_cpu() {
        let input = (0..4099)
            .map(|idx| {
                if idx % 257 == 0 {
                    Fq::ZERO
                } else {
                    Fq::from((idx as u64 % 251) + 2)
                }
            })
            .collect::<Vec<_>>();
        let expected = input
            .iter()
            .map(|value| value.invert().unwrap_or(Fq::ZERO))
            .collect::<Vec<_>>();

        let mut values = input.iter().copied().map(raw_from_fq).collect::<Vec<_>>();
        bls12_381_fr_invert_assign(&mut values).unwrap();
        let actual = values.into_iter().map(fq_from_raw).collect::<Vec<_>>();
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(actual, expected, "idx={idx}");
        }
    }

    #[test]
    fn sort_canonical_matches_field_order() {
        let mut current = Fq::from(0x1234_5678_u64);
        let mut input = Vec::with_capacity(16_387);
        for idx in 0..input.capacity() {
            current = current.square() + Fq::from((idx as u64).wrapping_mul(0x9e37_79b9));
            input.push(current);
        }
        let duplicate = input[17];
        input.extend([Fq::ZERO, Fq::ONE, -Fq::ONE, duplicate, duplicate]);

        let mut expected = input.clone();
        expected.sort();
        let mut actual = input.into_iter().map(raw_from_fq).collect::<Vec<_>>();
        bls12_381_fr_sort_canonical(&mut actual).unwrap();
        let actual = actual.into_iter().map(fq_from_raw).collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
