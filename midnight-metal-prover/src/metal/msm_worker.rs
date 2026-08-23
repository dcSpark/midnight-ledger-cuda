// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Thread-affine owner for the independently packaged Metal MSM runtime.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
    mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
};
use std::thread;
use std::time::{Duration, Instant};

use bls12_381_metal_msm::{
    BLS12_381_SCALAR_MODULUS, CanonicalScalar, FixedWindowBits, MetalMsmEngine,
    MetalMsmObservationProfile, MetalMsmReductionStrategy, MsmJobV1, PreparedBases,
    PreparedFixedBases, RawFpMontgomery, RawFrMontgomery, RawG1Affine, ResidentMontgomeryScalars,
    WindowBits,
};
use zeroize::{Zeroize, Zeroizing};

use crate::{
    Bls12_381Fr, Bls12_381G1Affine, Bls12_381G1Projective, Bls12_381MsmBasis, Bls12_381MsmWindow,
    MetalError,
};

type SrsFingerprint = [u8; 32];
type WorkerResult<T> = Result<T, String>;
const WORKER_QUEUE_CAPACITY: usize = 8;
const DEFAULT_WORKER_LANES: usize = 2;
const MAX_WORKER_LANES: usize = 2;
const MAX_COALESCED_REQUESTS: usize = 8;
const DEFAULT_COALESCE_MICROS: u64 = 50;
const MAX_COALESCE_MICROS: u64 = 5_000;

enum ScalarBatch {
    Canonical(Zeroizing<Vec<CanonicalScalar>>),
    Montgomery(Zeroizing<Vec<RawFrMontgomery>>),
}

impl ScalarBatch {
    fn len(&self) -> usize {
        match self {
            Self::Canonical(values) => values.len(),
            Self::Montgomery(values) => values.len(),
        }
    }

    fn nonzero_count(&self) -> usize {
        match self {
            Self::Canonical(values) => values.iter().filter(|scalar| !scalar.is_zero()).count(),
            Self::Montgomery(values) => values
                .iter()
                .filter(|scalar| scalar.limbs != [0; 4])
                .count(),
        }
    }

    fn canonical(&self) -> WorkerResult<&[CanonicalScalar]> {
        match self {
            Self::Canonical(values) => Ok(values),
            Self::Montgomery(_) => {
                Err("variable-base MSM received Montgomery scalar input".to_owned())
            }
        }
    }

    fn montgomery(&self) -> WorkerResult<&[RawFrMontgomery]> {
        match self {
            Self::Montgomery(values) => Ok(values),
            Self::Canonical(_) => Err("fixed-base MSM received canonical scalar input".to_owned()),
        }
    }
}

enum Request {
    Prepare {
        fingerprint: SrsFingerprint,
        coefficient: Box<[Bls12_381G1Affine]>,
        lagrange: Box<[Bls12_381G1Affine]>,
        reply: SyncSender<WorkerResult<()>>,
    },
    Execute {
        fingerprint: SrsFingerprint,
        basis: Bls12_381MsmBasis,
        scalars: ScalarBatch,
        window: Bls12_381MsmWindow,
        reply: SyncSender<WorkerResult<Bls12_381G1Projective>>,
    },
    ExecuteBatch {
        fingerprint: SrsFingerprint,
        basis: Bls12_381MsmBasis,
        scalar_batches: Vec<ScalarBatch>,
        window: Bls12_381MsmWindow,
        reply: SyncSender<WorkerResult<Vec<Bls12_381G1Projective>>>,
    },
    ExecuteResidentBatch {
        fingerprint: SrsFingerprint,
        basis: Bls12_381MsmBasis,
        scalars: ResidentMontgomeryScalars,
        piece_len: usize,
        piece_count: usize,
        window: Bls12_381MsmWindow,
        reply: SyncSender<WorkerResult<Vec<Bls12_381G1Projective>>>,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ExecutionInput {
    Host,
    Resident,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ExecutionKey {
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    window: Bls12_381MsmWindow,
    input: ExecutionInput,
}

enum ExecutionReply {
    Single(SyncSender<WorkerResult<Bls12_381G1Projective>>),
    Batch {
        len: usize,
        sender: SyncSender<WorkerResult<Vec<Bls12_381G1Projective>>>,
    },
}

struct WorkerHandle {
    senders: Vec<SyncSender<Request>>,
    registry_id: u64,
    prepared: Mutex<HashSet<SrsFingerprint>>,
}

impl WorkerHandle {
    fn send(&self, request: Request) -> Result<(), Request> {
        // Keep each basis on one queue so compatible requests still coalesce.
        // Coefficient openings can then overlap the longer Lagrange stream.
        let lane = if self.senders.len() == 1 {
            0
        } else {
            match execution_key(&request).map(|key| key.basis) {
                Some(Bls12_381MsmBasis::Lagrange) | None => 0,
                Some(Bls12_381MsmBasis::Coefficient) => 1,
            }
        };
        self.senders[lane].send(request).map_err(|error| error.0)
    }
}

struct PreparedBasis {
    raw: Box<[RawG1Affine]>,
    variable: Option<PreparedBases>,
    fixed: HashMap<FixedWindowBits, PreparedFixedBases>,
}

static WORKER: OnceLock<WorkerResult<WorkerHandle>> = OnceLock::new();
static ACTIVITY: OnceLock<MsmActivityCounters> = OnceLock::new();

#[derive(Default)]
struct MsmActivityCounters {
    metal_submissions: AtomicU64,
    metal_outputs: AtomicU64,
    metal_logical_scalars: AtomicU64,
    metal_nonzero_scalars: AtomicU64,
    metal_failures: AtomicU64,
    metal_service_ns: AtomicU64,
    metal_coalesce_ns: AtomicU64,
    metal_canonicalization_ns: AtomicU64,
    cpu_outputs: AtomicU64,
    cpu_logical_scalars: AtomicU64,
    cpu_nonzero_scalars: AtomicU64,
    cpu_service_ns: AtomicU64,
    cpu_lock_wait_ns: AtomicU64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct MsmActivity {
    pub(super) metal_submissions: u64,
    pub(super) metal_outputs: u64,
    pub(super) metal_logical_scalars: u64,
    pub(super) metal_nonzero_scalars: u64,
    pub(super) metal_failures: u64,
    pub(super) metal_service_ns: u64,
    pub(super) metal_coalesce_ns: u64,
    pub(super) metal_canonicalization_ns: u64,
    pub(super) cpu_outputs: u64,
    pub(super) cpu_logical_scalars: u64,
    pub(super) cpu_nonzero_scalars: u64,
    pub(super) cpu_service_ns: u64,
    pub(super) cpu_lock_wait_ns: u64,
}

pub(super) fn activity() -> MsmActivity {
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    MsmActivity {
        metal_submissions: activity.metal_submissions.load(Ordering::Relaxed),
        metal_outputs: activity.metal_outputs.load(Ordering::Relaxed),
        metal_logical_scalars: activity.metal_logical_scalars.load(Ordering::Relaxed),
        metal_nonzero_scalars: activity.metal_nonzero_scalars.load(Ordering::Relaxed),
        metal_failures: activity.metal_failures.load(Ordering::Relaxed),
        metal_service_ns: activity.metal_service_ns.load(Ordering::Relaxed),
        metal_coalesce_ns: activity.metal_coalesce_ns.load(Ordering::Relaxed),
        metal_canonicalization_ns: activity.metal_canonicalization_ns.load(Ordering::Relaxed),
        cpu_outputs: activity.cpu_outputs.load(Ordering::Relaxed),
        cpu_logical_scalars: activity.cpu_logical_scalars.load(Ordering::Relaxed),
        cpu_nonzero_scalars: activity.cpu_nonzero_scalars.load(Ordering::Relaxed),
        cpu_service_ns: activity.cpu_service_ns.load(Ordering::Relaxed),
        cpu_lock_wait_ns: activity.cpu_lock_wait_ns.load(Ordering::Relaxed),
    }
}

pub(super) fn reset_activity() {
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    activity.metal_submissions.store(0, Ordering::Relaxed);
    activity.metal_outputs.store(0, Ordering::Relaxed);
    activity.metal_logical_scalars.store(0, Ordering::Relaxed);
    activity.metal_nonzero_scalars.store(0, Ordering::Relaxed);
    activity.metal_failures.store(0, Ordering::Relaxed);
    activity.metal_service_ns.store(0, Ordering::Relaxed);
    activity.metal_coalesce_ns.store(0, Ordering::Relaxed);
    activity
        .metal_canonicalization_ns
        .store(0, Ordering::Relaxed);
    activity.cpu_outputs.store(0, Ordering::Relaxed);
    activity.cpu_logical_scalars.store(0, Ordering::Relaxed);
    activity.cpu_nonzero_scalars.store(0, Ordering::Relaxed);
    activity.cpu_service_ns.store(0, Ordering::Relaxed);
    activity.cpu_lock_wait_ns.store(0, Ordering::Relaxed);
}

pub(super) fn record_cpu_partition(logical_scalars: usize, nonzero_scalars: usize) {
    debug_assert!(nonzero_scalars <= logical_scalars);
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    activity.cpu_outputs.fetch_add(1, Ordering::Relaxed);
    activity
        .cpu_logical_scalars
        .fetch_add(logical_scalars as u64, Ordering::Relaxed);
    activity
        .cpu_nonzero_scalars
        .fetch_add(nonzero_scalars as u64, Ordering::Relaxed);
}

pub(super) fn record_cpu_timing(lock_wait_ns: u64, service_ns: u64) {
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    saturating_add(&activity.cpu_lock_wait_ns, lock_wait_ns);
    saturating_add(&activity.cpu_service_ns, service_ns);
}

pub(super) fn is_prepared(
    expected_registry_id: u64,
    fingerprint: SrsFingerprint,
) -> Result<bool, MetalError> {
    let worker = worker()?;
    require_registry(worker, expected_registry_id)?;
    worker
        .prepared
        .lock()
        .map(|prepared| prepared.contains(&fingerprint))
        .map_err(|_| MetalError::Msm("MSM prepared-SRS cache mutex was poisoned".into()))
}

pub(super) fn prepare_srs(
    expected_registry_id: u64,
    fingerprint: SrsFingerprint,
    coefficient: &[Bls12_381G1Affine],
    lagrange: &[Bls12_381G1Affine],
) -> Result<(), MetalError> {
    if coefficient.is_empty() || coefficient.len() != lagrange.len() {
        return Err(MetalError::Msm(
            "prepared coefficient and Lagrange bases must have one nonempty shape".into(),
        ));
    }
    let worker = worker()?;
    require_registry(worker, expected_registry_id)?;
    let mut prepared = worker
        .prepared
        .lock()
        .map_err(|_| MetalError::Msm("MSM prepared-SRS cache mutex was poisoned".into()))?;
    if prepared.contains(&fingerprint) {
        return Ok(());
    }

    for sender in &worker.senders {
        let (reply, response) = sync_channel(1);
        sender
            .send(Request::Prepare {
                fingerprint,
                coefficient: coefficient.into(),
                lagrange: lagrange.into(),
                reply,
            })
            .map_err(|_| MetalError::Msm("MSM worker terminated during SRS preparation".into()))?;
        response
            .recv()
            .map_err(|_| MetalError::Msm("MSM worker dropped its SRS response".into()))?
            .map_err(MetalError::Msm)?;
    }
    prepared.insert(fingerprint);
    Ok(())
}

pub(super) fn execute(
    expected_registry_id: u64,
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    values: &[Bls12_381Fr],
    window: Bls12_381MsmWindow,
) -> Result<Bls12_381G1Projective, MetalError> {
    let worker = worker()?;
    require_registry(worker, expected_registry_id)?;
    if !worker
        .prepared
        .lock()
        .map_err(|_| MetalError::Msm("MSM prepared-SRS cache mutex was poisoned".into()))?
        .contains(&fingerprint)
    {
        return Err(MetalError::Msm(
            "MSM request references an SRS that was not prepared".into(),
        ));
    }
    let canonicalization_started = Instant::now();
    let scalars = submission_scalars(values, window);
    record_duration(
        &ACTIVITY
            .get_or_init(MsmActivityCounters::default)
            .metal_canonicalization_ns,
        canonicalization_started.elapsed(),
    );
    let scalars = scalars?;
    let (reply, response) = sync_channel(1);
    worker
        .send(Request::Execute {
            fingerprint,
            basis,
            scalars,
            window,
            reply,
        })
        .map_err(|_| MetalError::Msm("MSM worker terminated before submission".into()))?;
    response
        .recv()
        .map_err(|_| MetalError::Msm("MSM worker dropped its execution response".into()))?
        .map_err(MetalError::Msm)
}

pub(super) fn execute_batch(
    expected_registry_id: u64,
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    values: &[&[Bls12_381Fr]],
    window: Bls12_381MsmWindow,
) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
    if values.is_empty() {
        return Err(MetalError::Msm("MSM batch must not be empty".into()));
    }
    let worker = worker()?;
    require_registry(worker, expected_registry_id)?;
    if !worker
        .prepared
        .lock()
        .map_err(|_| MetalError::Msm("MSM prepared-SRS cache mutex was poisoned".into()))?
        .contains(&fingerprint)
    {
        return Err(MetalError::Msm(
            "MSM batch references an SRS that was not prepared".into(),
        ));
    }
    let canonicalization_started = Instant::now();
    let scalar_batches = values
        .iter()
        .map(|values| submission_scalars(values, window))
        .collect::<Result<Vec<_>, _>>();
    record_duration(
        &ACTIVITY
            .get_or_init(MsmActivityCounters::default)
            .metal_canonicalization_ns,
        canonicalization_started.elapsed(),
    );
    let scalar_batches = scalar_batches?;
    let (reply, response) = sync_channel(1);
    worker
        .send(Request::ExecuteBatch {
            fingerprint,
            basis,
            scalar_batches,
            window,
            reply,
        })
        .map_err(|_| MetalError::Msm("MSM worker terminated before batch submission".into()))?;
    response
        .recv()
        .map_err(|_| MetalError::Msm("MSM worker dropped its batch response".into()))?
        .map_err(MetalError::Msm)
}

pub(super) fn execute_resident_batch(
    expected_registry_id: u64,
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    scalars: ResidentMontgomeryScalars,
    piece_len: usize,
    piece_count: usize,
    window: Bls12_381MsmWindow,
) -> Result<Vec<Bls12_381G1Projective>, MetalError> {
    if piece_count == 0
        || piece_len
            .checked_mul(piece_count)
            .is_none_or(|expected| expected != scalars.len())
    {
        return Err(MetalError::Msm(
            "resident MSM piece dimensions do not match the retained scalar view".into(),
        ));
    }
    let worker = worker()?;
    require_registry(worker, expected_registry_id)?;
    if !worker
        .prepared
        .lock()
        .map_err(|_| MetalError::Msm("MSM prepared-SRS cache mutex was poisoned".into()))?
        .contains(&fingerprint)
    {
        return Err(MetalError::Msm(
            "resident MSM batch references an SRS that was not prepared".into(),
        ));
    }
    let (reply, response) = sync_channel(1);
    worker
        .send(Request::ExecuteResidentBatch {
            fingerprint,
            basis,
            scalars,
            piece_len,
            piece_count,
            window,
            reply,
        })
        .map_err(|_| MetalError::Msm("MSM worker terminated before resident submission".into()))?;
    response
        .recv()
        .map_err(|_| MetalError::Msm("MSM worker dropped its resident response".into()))?
        .map_err(MetalError::Msm)
}

fn worker() -> Result<&'static WorkerHandle, MetalError> {
    match WORKER.get_or_init(start_worker) {
        Ok(worker) => Ok(worker),
        Err(reason) => Err(MetalError::Msm(reason.clone())),
    }
}

fn start_worker() -> WorkerResult<WorkerHandle> {
    let lane_count = selected_worker_lanes()?;
    let mut senders = Vec::new();
    senders
        .try_reserve_exact(lane_count)
        .map_err(|_| "could not allocate the Metal MSM worker lane list".to_owned())?;
    let mut registry_id = None;
    for lane in 0..lane_count {
        let (sender, requests) = sync_channel(WORKER_QUEUE_CAPACITY);
        let (initialization, initialized) = sync_channel(1);
        thread::Builder::new()
            .name(format!("midnight-metal-msm-{lane}"))
            .spawn(move || {
                let reduction_strategy = match selected_reduction_strategy() {
                    Ok(strategy) => strategy,
                    Err(error) => {
                        let _ = initialization.send(Err(error));
                        return;
                    }
                };
                let coalesce_latency = match selected_coalesce_latency() {
                    Ok(latency) => latency,
                    Err(error) => {
                        let _ = initialization.send(Err(error));
                        return;
                    }
                };
                let engine = match MetalMsmEngine::new_with_strategy(
                    MetalMsmObservationProfile::TrustedLocalDataDependentV1,
                    reduction_strategy,
                ) {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = initialization.send(Err(error.to_string()));
                        return;
                    }
                };
                let registry_id = engine.device_identity().registry_id();
                if initialization.send(Ok(registry_id)).is_err() {
                    return;
                }
                run_worker(engine, requests, coalesce_latency);
            })
            .map_err(|error| format!("could not spawn Metal MSM worker lane {lane}: {error}"))?;
        let lane_registry_id = initialized
            .recv()
            .map_err(|_| format!("Metal MSM worker lane {lane} exited during initialization"))??;
        if registry_id.is_some_and(|expected| expected != lane_registry_id) {
            return Err("Metal MSM worker lanes selected different devices".to_owned());
        }
        registry_id = Some(lane_registry_id);
        senders.push(sender);
    }
    Ok(WorkerHandle {
        senders,
        registry_id: registry_id.expect("at least one Metal MSM worker lane was requested"),
        prepared: Mutex::new(HashSet::new()),
    })
}

fn selected_worker_lanes() -> WorkerResult<usize> {
    let lanes = match std::env::var("MIDNIGHT_METAL_MSM_WORKERS") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            format!(
                "MIDNIGHT_METAL_MSM_WORKERS must be an integer in 1..={MAX_WORKER_LANES}, got {value:?}"
            )
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_WORKER_LANES,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("MIDNIGHT_METAL_MSM_WORKERS must be valid Unicode".to_owned());
        }
    };
    if !(1..=MAX_WORKER_LANES).contains(&lanes) {
        return Err(format!(
            "MIDNIGHT_METAL_MSM_WORKERS must be in 1..={MAX_WORKER_LANES}, got {lanes}"
        ));
    }
    Ok(lanes)
}

fn selected_reduction_strategy() -> WorkerResult<MetalMsmReductionStrategy> {
    let configured = match std::env::var("MIDNIGHT_METAL_MSM_REDUCER") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "gpu-routed-fine-tiled-xyzz-v8".to_owned(),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("MIDNIGHT_METAL_MSM_REDUCER must be valid Unicode".to_owned());
        }
    };
    match configured.to_ascii_lowercase().as_str() {
        "global-batch-affine-v1" | "global" | "affine" => {
            Ok(MetalMsmReductionStrategy::GlobalBatchAffineV1)
        }
        "parallel-projective-v1" | "parallel" | "projective" => {
            Ok(MetalMsmReductionStrategy::ParallelProjectiveV1)
        }
        "parallel-branch-completed-jacobian-v2" | "jacobian-v2" | "jacobian" => {
            Ok(MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2)
        }
        "gpu-routed-dense-jacobian-v3" | "gpu-routed" | "dense-v3" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3)
        }
        "gpu-routed-dense-xyzz-v4" | "xyzz-v4" | "xyzz" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4)
        }
        "gpu-routed-adaptive-xyzz-v5" | "adaptive-xyzz-v5" | "xyzz-v5" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5)
        }
        "gpu-routed-tiled-xyzz-v6" | "tiled-xyzz-v6" | "xyzz-v6" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6)
        }
        "gpu-routed-packed-tiled-xyzz-v7" | "packed-tiled-xyzz-v7" | "xyzz-v7" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7)
        }
        "gpu-routed-fine-tiled-xyzz-v8" | "fine-tiled-xyzz-v8" | "xyzz-v8" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8)
        }
        "gpu-routed-batch-affine-xyzz-v9" | "batch-affine-xyzz-v9" | "xyzz-v9" => {
            Ok(MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9)
        }
        value => Err(format!(
            "unsupported MIDNIGHT_METAL_MSM_REDUCER value {value:?}"
        )),
    }
}

fn selected_coalesce_latency() -> WorkerResult<Duration> {
    let micros = match std::env::var("MIDNIGHT_METAL_MSM_COALESCE_US") {
        Ok(value) => value.parse::<u64>().map_err(|_| {
            format!(
                "MIDNIGHT_METAL_MSM_COALESCE_US must be an integer in 0..={MAX_COALESCE_MICROS}, got {value:?}"
            )
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_COALESCE_MICROS,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("MIDNIGHT_METAL_MSM_COALESCE_US must be valid Unicode".to_owned());
        }
    };
    if micros > MAX_COALESCE_MICROS {
        return Err(format!(
            "MIDNIGHT_METAL_MSM_COALESCE_US must be in 0..={MAX_COALESCE_MICROS}, got {micros}"
        ));
    }
    Ok(Duration::from_micros(micros))
}

fn run_worker(engine: MetalMsmEngine, requests: Receiver<Request>, coalesce_latency: Duration) {
    let mut prepared = HashMap::<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>::new();
    let mut pending = VecDeque::new();
    loop {
        let request = match pending.pop_front() {
            Some(request) => request,
            None => match requests.recv() {
                Ok(request) => request,
                Err(_) => break,
            },
        };
        match request {
            Request::Prepare {
                fingerprint,
                coefficient,
                lagrange,
                reply,
            } => {
                let result =
                    prepare_on_worker(&engine, &mut prepared, fingerprint, &coefficient, &lagrange);
                let _ = reply.send(result);
            }
            request @ (Request::Execute { .. } | Request::ExecuteBatch { .. }) => {
                execute_coalesced(
                    &engine,
                    &mut prepared,
                    request,
                    &requests,
                    &mut pending,
                    coalesce_latency,
                );
            }
            request @ Request::ExecuteResidentBatch { .. } => {
                execute_resident_coalesced(
                    &engine,
                    &mut prepared,
                    request,
                    &requests,
                    &mut pending,
                    coalesce_latency,
                );
            }
        }
    }
}

fn select_coalesced(
    first: Request,
    requests: &Receiver<Request>,
    pending: &mut VecDeque<Request>,
    coalesce_latency: Duration,
) -> Vec<Request> {
    let coalesce_started = Instant::now();
    let key = execution_key(&first).expect("coalescing receives an execution request");
    let mut selected = vec![first];
    let deadline = Instant::now() + coalesce_latency;
    while selected.len() < MAX_COALESCED_REQUESTS {
        if let Some(index) = pending
            .iter()
            .position(|request| execution_key(request) == Some(key))
        {
            selected.push(
                pending
                    .remove(index)
                    .expect("the selected pending request still exists"),
            );
            continue;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        let received = if remaining.is_zero() {
            // Zero-latency coalescing still drains work queued while the
            // previous Metal command was executing.
            match requests.try_recv() {
                Ok(request) => Ok(request),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => Err(()),
            }
        } else {
            requests.recv_timeout(remaining).map_err(|_| ())
        };
        match received {
            Ok(request) if execution_key(&request) == Some(key) => selected.push(request),
            Ok(request) => pending.push_back(request),
            Err(_) => break,
        }
    }
    record_duration(
        &ACTIVITY
            .get_or_init(MsmActivityCounters::default)
            .metal_coalesce_ns,
        coalesce_started.elapsed(),
    );
    selected
}

fn execute_coalesced(
    engine: &MetalMsmEngine,
    prepared: &mut HashMap<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>,
    first: Request,
    requests: &Receiver<Request>,
    pending: &mut VecDeque<Request>,
    coalesce_latency: Duration,
) {
    let key = execution_key(&first).expect("execute_coalesced receives an execution request");
    let selected = select_coalesced(first, requests, pending, coalesce_latency);

    let mut scalar_batches = Vec::new();
    let mut replies = Vec::with_capacity(selected.len());
    for request in selected {
        match request {
            Request::Execute { scalars, reply, .. } => {
                scalar_batches.push(scalars);
                replies.push(ExecutionReply::Single(reply));
            }
            Request::ExecuteBatch {
                scalar_batches: batches,
                reply,
                ..
            } => {
                let len = batches.len();
                scalar_batches.extend(batches);
                replies.push(ExecutionReply::Batch { len, sender: reply });
            }
            Request::Prepare { .. } => unreachable!("only execution requests are selected"),
            Request::ExecuteResidentBatch { .. } => {
                unreachable!("host and resident execution keys are distinct")
            }
        }
    }

    match execute_batch_on_worker(
        engine,
        prepared,
        key.fingerprint,
        key.basis,
        &scalar_batches,
        key.window,
    ) {
        Ok(outputs) => send_coalesced_outputs(outputs, replies),
        Err(error) => {
            for reply in replies {
                match reply {
                    ExecutionReply::Single(sender) => {
                        let _ = sender.send(Err(error.clone()));
                    }
                    ExecutionReply::Batch { sender, .. } => {
                        let _ = sender.send(Err(error.clone()));
                    }
                }
            }
        }
    }
}

fn send_coalesced_outputs(outputs: Vec<Bls12_381G1Projective>, replies: Vec<ExecutionReply>) {
    let mut outputs = outputs.into_iter();
    for reply in replies {
        match reply {
            ExecutionReply::Single(sender) => {
                let result = outputs
                    .next()
                    .ok_or_else(|| "coalesced MSM returned too few outputs".to_owned());
                let _ = sender.send(result);
            }
            ExecutionReply::Batch { len, sender } => {
                let batch = outputs.by_ref().take(len).collect::<Vec<_>>();
                let result = if batch.len() == len {
                    Ok(batch)
                } else {
                    Err("coalesced MSM returned too few batch outputs".to_owned())
                };
                let _ = sender.send(result);
            }
        }
    }
    debug_assert!(outputs.next().is_none());
}

fn execute_resident_coalesced(
    engine: &MetalMsmEngine,
    prepared: &mut HashMap<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>,
    first: Request,
    requests: &Receiver<Request>,
    pending: &mut VecDeque<Request>,
    coalesce_latency: Duration,
) {
    let key = execution_key(&first)
        .expect("execute_resident_coalesced receives a resident execution request");
    let selected = select_coalesced(first, requests, pending, coalesce_latency);
    let mut sources = Vec::with_capacity(selected.len());
    let mut descriptions = Vec::with_capacity(selected.len());
    for request in selected {
        match request {
            Request::ExecuteResidentBatch {
                scalars,
                piece_len,
                piece_count,
                reply,
                ..
            } => {
                sources.push(scalars);
                descriptions.push((piece_len, piece_count, reply));
            }
            Request::Prepare { .. } | Request::Execute { .. } | Request::ExecuteBatch { .. } => {
                unreachable!("host and resident execution keys are distinct")
            }
        }
    }

    let result = (|| {
        let output_count = descriptions
            .iter()
            .try_fold(0usize, |total, (_, count, _)| {
                total
                    .checked_add(*count)
                    .ok_or_else(|| "coalesced resident MSM output count overflowed".to_owned())
            })?;
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(output_count)
            .map_err(|_| "coalesced resident MSM job allocation failed".to_owned())?;
        let mut source_offset = 0usize;
        let mut output_index = 0usize;
        for (source, (piece_len, piece_count, _)) in sources.iter().zip(&descriptions) {
            let expected_len = piece_len
                .checked_mul(*piece_count)
                .ok_or_else(|| "coalesced resident MSM piece dimensions overflowed".to_owned())?;
            if expected_len != source.len() {
                return Err("coalesced resident MSM piece dimensions changed in flight".to_owned());
            }
            let term_count = u32::try_from(*piece_len)
                .map_err(|_| "resident MSM piece length exceeds the routing ABI".to_owned())?;
            for piece_index in 0..*piece_count {
                let scalar_offset = source_offset
                    .checked_add(piece_index.checked_mul(*piece_len).ok_or_else(|| {
                        "coalesced resident MSM scalar offset overflowed".to_owned()
                    })?)
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        "coalesced resident MSM scalar offset exceeds the routing ABI".to_owned()
                    })?;
                jobs.push(MsmJobV1 {
                    base_offset: 0,
                    scalar_offset,
                    term_count,
                    output_index: u32::try_from(output_index).map_err(|_| {
                        "coalesced resident MSM output count exceeds the routing ABI".to_owned()
                    })?,
                });
                output_index += 1;
            }
            source_offset = source_offset
                .checked_add(source.len())
                .ok_or_else(|| "coalesced resident MSM scalar count overflowed".to_owned())?;
        }
        execute_resident_sources_on_worker(
            engine,
            prepared,
            key.fingerprint,
            key.basis,
            &sources,
            &jobs,
            key.window,
        )
    })();

    match result {
        Ok(outputs) => {
            let mut outputs = outputs.into_iter();
            for (_, piece_count, reply) in descriptions {
                let batch = outputs.by_ref().take(piece_count).collect::<Vec<_>>();
                let result = if batch.len() == piece_count {
                    Ok(batch)
                } else {
                    Err("coalesced resident MSM returned too few outputs".to_owned())
                };
                let _ = reply.send(result);
            }
            debug_assert!(outputs.next().is_none());
        }
        Err(error) => {
            for (_, _, reply) in descriptions {
                let _ = reply.send(Err(error.clone()));
            }
        }
    }
}

fn execution_key(request: &Request) -> Option<ExecutionKey> {
    match request {
        Request::Prepare { .. } => None,
        Request::Execute {
            fingerprint,
            basis,
            window,
            ..
        }
        | Request::ExecuteBatch {
            fingerprint,
            basis,
            window,
            ..
        } => Some(ExecutionKey {
            fingerprint: *fingerprint,
            basis: *basis,
            window: *window,
            input: ExecutionInput::Host,
        }),
        Request::ExecuteResidentBatch {
            fingerprint,
            basis,
            window,
            ..
        } => Some(ExecutionKey {
            fingerprint: *fingerprint,
            basis: *basis,
            window: *window,
            input: ExecutionInput::Resident,
        }),
    }
}

fn prepare_on_worker(
    engine: &MetalMsmEngine,
    prepared: &mut HashMap<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>,
    fingerprint: SrsFingerprint,
    coefficient: &[Bls12_381G1Affine],
    lagrange: &[Bls12_381G1Affine],
) -> WorkerResult<()> {
    let coefficient_key = (fingerprint, Bls12_381MsmBasis::Coefficient);
    let lagrange_key = (fingerprint, Bls12_381MsmBasis::Lagrange);
    if let (Some(cached_coefficient), Some(cached_lagrange)) =
        (prepared.get(&coefficient_key), prepared.get(&lagrange_key))
    {
        if cached_coefficient.raw.len() == coefficient.len()
            && cached_lagrange.raw.len() == lagrange.len()
        {
            return Ok(());
        }
        return Err("prepared SRS fingerprint was reused with a different length".into());
    }
    let coefficient_prepared = prepare_basis(engine, coefficient)?;
    let lagrange_prepared = prepare_basis(engine, lagrange)?;
    prepared.insert(coefficient_key, coefficient_prepared);
    prepared.insert(lagrange_key, lagrange_prepared);
    Ok(())
}

fn prepare_basis(
    engine: &MetalMsmEngine,
    values: &[Bls12_381G1Affine],
) -> WorkerResult<PreparedBasis> {
    let raw = raw_bases(values).into_boxed_slice();
    let default_window = if raw.len() <= 16_384 {
        FixedWindowBits::W11
    } else {
        FixedWindowBits::W13
    };
    // Warm the size-selected layout during public SRS preparation. Explicit
    // override widths and the variable-base path remain lazy below.
    let default_fixed = engine
        .prepare_fixed_bases(&raw, default_window)
        .map_err(|error| error.to_string())?;
    let mut fixed = HashMap::new();
    fixed.insert(default_window, default_fixed);
    Ok(PreparedBasis {
        raw,
        variable: None,
        fixed,
    })
}

fn execute_batch_on_worker(
    engine: &MetalMsmEngine,
    prepared: &mut HashMap<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>,
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    scalar_batches: &[ScalarBatch],
    window: Bls12_381MsmWindow,
) -> WorkerResult<Vec<Bls12_381G1Projective>> {
    let prepared = prepared
        .get_mut(&(fingerprint, basis))
        .ok_or_else(|| "MSM worker does not own the requested prepared basis".to_owned())?;
    let logical_scalars = scalar_batches.iter().map(ScalarBatch::len).sum::<usize>();
    let nonzero_scalars = scalar_batches
        .iter()
        .map(ScalarBatch::nonzero_count)
        .sum::<usize>();
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    activity.metal_submissions.fetch_add(1, Ordering::Relaxed);
    let service_started = Instant::now();
    let result = if let Some(fixed_window) = standalone_fixed_window(window) {
        if !prepared.fixed.contains_key(&fixed_window) {
            let expanded = engine
                .prepare_fixed_bases(&prepared.raw, fixed_window)
                .map_err(|error| error.to_string());
            match expanded {
                Ok(expanded) => {
                    prepared.fixed.insert(fixed_window, expanded);
                }
                Err(error) => {
                    record_duration(&activity.metal_service_ns, service_started.elapsed());
                    activity.metal_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            }
        }
        let slices = scalar_batches
            .iter()
            .map(ScalarBatch::montgomery)
            .collect::<WorkerResult<Vec<_>>>()?;
        engine.msm_fixed_montgomery_batch(
            prepared
                .fixed
                .get(&fixed_window)
                .expect("fixed preparation was inserted"),
            &slices,
        )
    } else {
        if prepared.variable.is_none() {
            let variable = engine
                .prepare_bases(&prepared.raw)
                .map_err(|error| error.to_string());
            match variable {
                Ok(variable) => prepared.variable = Some(variable),
                Err(error) => {
                    record_duration(&activity.metal_service_ns, service_started.elapsed());
                    activity.metal_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            }
        }
        let slices = scalar_batches
            .iter()
            .map(ScalarBatch::canonical)
            .collect::<WorkerResult<Vec<_>>>()?;
        engine.msm_batch(
            prepared
                .variable
                .as_ref()
                .expect("variable preparation was inserted"),
            &slices,
            standalone_variable_window(window)
                .expect("non-fixed window must be a supported variable window"),
        )
    };
    let service_elapsed = service_started.elapsed();
    record_duration(&activity.metal_service_ns, service_elapsed);
    trace_execution_request(
        basis,
        window,
        scalar_batches,
        logical_scalars,
        nonzero_scalars,
        service_elapsed,
        result.is_ok(),
    );
    match result {
        Ok(outputs) => {
            activity
                .metal_outputs
                .fetch_add(outputs.len() as u64, Ordering::Relaxed);
            activity
                .metal_logical_scalars
                .fetch_add(logical_scalars as u64, Ordering::Relaxed);
            activity
                .metal_nonzero_scalars
                .fetch_add(nonzero_scalars as u64, Ordering::Relaxed);
            Ok(outputs
                .into_iter()
                .map(|output| Bls12_381G1Projective {
                    x: output.x.limbs,
                    y: output.y.limbs,
                    z: output.z.limbs,
                })
                .collect())
        }
        Err(error) => {
            activity.metal_failures.fetch_add(1, Ordering::Relaxed);
            Err(error.to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_resident_sources_on_worker(
    engine: &MetalMsmEngine,
    prepared: &mut HashMap<(SrsFingerprint, Bls12_381MsmBasis), PreparedBasis>,
    fingerprint: SrsFingerprint,
    basis: Bls12_381MsmBasis,
    scalar_sources: &[ResidentMontgomeryScalars],
    jobs: &[MsmJobV1],
    window: Bls12_381MsmWindow,
) -> WorkerResult<Vec<Bls12_381G1Projective>> {
    let fixed_window = standalone_fixed_window(window)
        .ok_or_else(|| "resident Montgomery MSM requires a fixed-base window".to_owned())?;
    let prepared = prepared
        .get_mut(&(fingerprint, basis))
        .ok_or_else(|| "MSM worker does not own the requested prepared basis".to_owned())?;
    let max_terms = jobs
        .iter()
        .map(|job| job.term_count as usize)
        .max()
        .unwrap_or(0);
    if max_terms > prepared.raw.len() {
        return Err(format!(
            "resident MSM piece has {max_terms} scalars but the prepared basis has {} points",
            prepared.raw.len()
        ));
    }
    if !prepared.fixed.contains_key(&fixed_window) {
        let expanded = engine
            .prepare_fixed_bases(&prepared.raw, fixed_window)
            .map_err(|error| error.to_string())?;
        prepared.fixed.insert(fixed_window, expanded);
    }

    let logical_scalars = scalar_sources
        .iter()
        .map(ResidentMontgomeryScalars::len)
        .sum::<usize>();
    let activity = ACTIVITY.get_or_init(MsmActivityCounters::default);
    activity.metal_submissions.fetch_add(1, Ordering::Relaxed);
    let service_started = Instant::now();
    let result = engine.msm_fixed_resident_sources_ragged(
        prepared
            .fixed
            .get(&fixed_window)
            .expect("fixed preparation was inserted"),
        scalar_sources,
        jobs,
    );
    let service_elapsed = service_started.elapsed();
    record_duration(&activity.metal_service_ns, service_elapsed);
    trace_resident_execution_request(
        basis,
        window,
        scalar_sources.len(),
        jobs,
        logical_scalars,
        service_elapsed,
        result.is_ok(),
    );
    match result {
        Ok(outputs) => {
            activity
                .metal_outputs
                .fetch_add(outputs.len() as u64, Ordering::Relaxed);
            activity
                .metal_logical_scalars
                .fetch_add(logical_scalars as u64, Ordering::Relaxed);
            Ok(outputs
                .into_iter()
                .map(|output| Bls12_381G1Projective {
                    x: output.x.limbs,
                    y: output.y.limbs,
                    z: output.z.limbs,
                })
                .collect())
        }
        Err(error) => {
            activity.metal_failures.fetch_add(1, Ordering::Relaxed);
            Err(error.to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_execution_request(
    basis: Bls12_381MsmBasis,
    window: Bls12_381MsmWindow,
    scalar_batches: &[ScalarBatch],
    logical_scalars: usize,
    nonzero_scalars: usize,
    elapsed: Duration,
    succeeded: bool,
) {
    if std::env::var_os("MIDNIGHT_METAL_MSM_TRACE_REQUESTS").is_none() {
        return;
    }
    let min_terms = scalar_batches
        .iter()
        .map(ScalarBatch::len)
        .min()
        .unwrap_or(0);
    let max_terms = scalar_batches
        .iter()
        .map(ScalarBatch::len)
        .max()
        .unwrap_or(0);
    std::eprintln!(
        "{{\"schema\":\"midnight-metal-prover/msm-request/v1\",\"basis\":\"{basis:?}\",\"window\":\"{window:?}\",\"outputs\":{},\"min_terms\":{min_terms},\"max_terms\":{max_terms},\"logical_scalars\":{logical_scalars},\"nonzero_scalars\":{nonzero_scalars},\"service_ms\":{:.6},\"succeeded\":{succeeded}}}",
        scalar_batches.len(),
        elapsed.as_secs_f64() * 1_000.0,
    );
}

fn trace_resident_execution_request(
    basis: Bls12_381MsmBasis,
    window: Bls12_381MsmWindow,
    source_count: usize,
    jobs: &[MsmJobV1],
    logical_scalars: usize,
    elapsed: Duration,
    succeeded: bool,
) {
    if std::env::var_os("MIDNIGHT_METAL_MSM_TRACE_REQUESTS").is_none() {
        return;
    }
    let min_terms = jobs.iter().map(|job| job.term_count).min().unwrap_or(0);
    let max_terms = jobs.iter().map(|job| job.term_count).max().unwrap_or(0);
    std::eprintln!(
        "{{\"schema\":\"midnight-metal-prover/msm-request/v1\",\"basis\":\"{basis:?}\",\"window\":\"{window:?}\",\"input\":\"resident-montgomery-v1\",\"sources\":{source_count},\"outputs\":{},\"min_terms\":{min_terms},\"max_terms\":{max_terms},\"logical_scalars\":{logical_scalars},\"nonzero_scalars\":null,\"service_ms\":{:.6},\"succeeded\":{succeeded}}}",
        jobs.len(),
        elapsed.as_secs_f64() * 1_000.0,
    );
}

fn raw_bases(values: &[Bls12_381G1Affine]) -> Vec<RawG1Affine> {
    values
        .iter()
        .map(|point| RawG1Affine {
            x: RawFpMontgomery { limbs: point.x },
            y: RawFpMontgomery { limbs: point.y },
        })
        .collect()
}

fn canonical_scalars(
    values: &[Bls12_381Fr],
) -> Result<Zeroizing<Vec<CanonicalScalar>>, MetalError> {
    let mut scalars = Zeroizing::new(Vec::new());
    scalars
        .try_reserve_exact(values.len())
        .map_err(|_| MetalError::Msm("canonical scalar allocation failed".into()))?;
    for value in values {
        if value.limbs == [0; 4] {
            scalars.push(CanonicalScalar::ZERO);
            continue;
        }
        if !limbs_are_less_than(&value.limbs, &BLS12_381_SCALAR_MODULUS) {
            return Err(MetalError::Msm(
                "MSM scalar is not a reduced Montgomery residue".into(),
            ));
        }
        let mut raw = blst::blst_fr { l: value.limbs };
        let mut canonical = [0u64; 4];
        // SAFETY: both values are initialized BLST scalar-field representations.
        unsafe { blst::blst_uint64_from_fr(canonical.as_mut_ptr(), &raw) };
        let scalar = CanonicalScalar::from_le_limbs(canonical);
        raw.l.zeroize();
        canonical.zeroize();
        scalars.push(scalar.map_err(|error| MetalError::Msm(error.to_string()))?);
    }
    Ok(scalars)
}

fn submission_scalars(
    values: &[Bls12_381Fr],
    window: Bls12_381MsmWindow,
) -> Result<ScalarBatch, MetalError> {
    if standalone_fixed_window(window).is_some() {
        let mut scalars = Zeroizing::new(Vec::new());
        scalars
            .try_reserve_exact(values.len())
            .map_err(|_| MetalError::Msm("Montgomery scalar allocation failed".into()))?;
        scalars.extend(
            values
                .iter()
                .map(|value| RawFrMontgomery { limbs: value.limbs }),
        );
        Ok(ScalarBatch::Montgomery(scalars))
    } else {
        canonical_scalars(values).map(ScalarBatch::Canonical)
    }
}

fn limbs_are_less_than(lhs: &[u64; 4], rhs: &[u64; 4]) -> bool {
    let mut index = lhs.len();
    while index != 0 {
        index -= 1;
        if lhs[index] != rhs[index] {
            return lhs[index] < rhs[index];
        }
    }
    false
}

fn standalone_variable_window(window: Bls12_381MsmWindow) -> Option<WindowBits> {
    match window {
        Bls12_381MsmWindow::W6 => Some(WindowBits::W6),
        Bls12_381MsmWindow::W7 => Some(WindowBits::W7),
        Bls12_381MsmWindow::W8 => Some(WindowBits::W8),
        Bls12_381MsmWindow::W9 => Some(WindowBits::W9),
        Bls12_381MsmWindow::W10 => Some(WindowBits::W10),
        Bls12_381MsmWindow::FixedW11
        | Bls12_381MsmWindow::FixedW12
        | Bls12_381MsmWindow::FixedW13
        | Bls12_381MsmWindow::FixedW14
        | Bls12_381MsmWindow::FixedW15
        | Bls12_381MsmWindow::FixedW16 => None,
    }
}

fn standalone_fixed_window(window: Bls12_381MsmWindow) -> Option<FixedWindowBits> {
    match window {
        Bls12_381MsmWindow::FixedW11 => Some(FixedWindowBits::W11),
        Bls12_381MsmWindow::FixedW12 => Some(FixedWindowBits::W12),
        Bls12_381MsmWindow::FixedW13 => Some(FixedWindowBits::W13),
        Bls12_381MsmWindow::FixedW14 => Some(FixedWindowBits::W14),
        Bls12_381MsmWindow::FixedW15 => Some(FixedWindowBits::W15),
        Bls12_381MsmWindow::FixedW16 => Some(FixedWindowBits::W16),
        Bls12_381MsmWindow::W6
        | Bls12_381MsmWindow::W7
        | Bls12_381MsmWindow::W8
        | Bls12_381MsmWindow::W9
        | Bls12_381MsmWindow::W10 => None,
    }
}

fn record_duration(counter: &AtomicU64, duration: Duration) {
    saturating_add(
        counter,
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
    );
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn require_registry(worker: &WorkerHandle, expected: u64) -> Result<(), MetalError> {
    if worker.registry_id == expected {
        Ok(())
    } else {
        Err(MetalError::Msm(format!(
            "MSM worker device {:#018x} differs from proof session device {expected:#018x}",
            worker.registry_id
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalSession;

    fn g1_multiple(multiplier: u64) -> (Bls12_381G1Affine, blst::blst_p1_affine) {
        let bytes = multiplier.to_le_bytes();
        let mut projective = blst::blst_p1::default();
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: the process-lifetime generator, initialized scalar bytes,
        // and distinct initialized outputs remain valid for both BLST calls.
        unsafe {
            blst::blst_p1_mult(
                &mut projective,
                blst::blst_p1_generator(),
                bytes.as_ptr(),
                u64::BITS as usize,
            );
            blst::blst_p1_to_affine(&mut affine, &projective);
        }
        (
            Bls12_381G1Affine {
                x: affine.x.l,
                y: affine.y.l,
            },
            affine,
        )
    }

    fn fr(value: u64) -> Bls12_381Fr {
        let canonical = [value, 0, 0, 0];
        let mut raw = blst::blst_fr::default();
        // SAFETY: BLST receives four initialized canonical little-endian limbs.
        unsafe { blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr()) };
        Bls12_381Fr { limbs: raw.l }
    }

    fn expected_msm(
        bases: &[blst::blst_p1_affine],
        scalars: &[Bls12_381Fr],
    ) -> blst::blst_p1_affine {
        let mut accumulator = blst::blst_p1::default();
        for (base, scalar) in bases.iter().zip(scalars) {
            let mut canonical = [0u64; 4];
            let raw = blst::blst_fr { l: scalar.limbs };
            // SAFETY: both BLST field representations are initialized.
            unsafe { blst::blst_uint64_from_fr(canonical.as_mut_ptr(), &raw) };
            let bytes = canonical
                .iter()
                .flat_map(|limb| limb.to_le_bytes())
                .collect::<Vec<_>>();
            let mut projective_base = blst::blst_p1::default();
            let mut term = blst::blst_p1::default();
            let mut sum = blst::blst_p1::default();
            // SAFETY: the affine input, scalar bytes, accumulator, and distinct
            // initialized outputs remain valid for the complete calls.
            unsafe {
                blst::blst_p1_from_affine(&mut projective_base, base);
                blst::blst_p1_mult(&mut term, &projective_base, bytes.as_ptr(), 255);
                blst::blst_p1_add_or_double(&mut sum, &accumulator, &term);
            }
            accumulator = sum;
        }
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: the accumulator and distinct affine output are initialized.
        unsafe { blst::blst_p1_to_affine(&mut affine, &accumulator) };
        affine
    }

    fn assert_same_point(actual: Bls12_381G1Projective, expected: blst::blst_p1_affine) {
        let projective = blst::blst_p1 {
            x: blst::blst_fp { l: actual.x },
            y: blst::blst_fp { l: actual.y },
            z: blst::blst_fp { l: actual.z },
        };
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: the worker returned a complete initialized Jacobian point.
        unsafe { blst::blst_p1_to_affine(&mut affine, &projective) };
        assert_eq!(affine.x.l, expected.x.l);
        assert_eq!(affine.y.l, expected.y.l);
    }

    #[test]
    fn canonical_scalar_conversion_preserves_zero_and_rejects_the_modulus() {
        let converted = canonical_scalars(&[fr(0), fr(7)]).expect("convert reduced residues");
        assert_eq!(converted[0], CanonicalScalar::ZERO);
        assert_eq!(converted[1].to_le_limbs(), [7, 0, 0, 0]);

        let error = canonical_scalars(&[Bls12_381Fr {
            limbs: BLS12_381_SCALAR_MODULUS,
        }])
        .expect_err("the modulus is not a reduced residue");
        assert!(
            error
                .to_string()
                .contains("not a reduced Montgomery residue")
        );
    }

    #[test]
    #[ignore = "requires access to an Apple Metal device"]
    fn worker_api_matches_blst_for_single_and_ragged_batch_requests() {
        let session = MetalSession::new().expect("create proof Metal session");
        let (coefficient, coefficient_reference): (Vec<_>, Vec<_>) =
            (1..=8).map(g1_multiple).unzip();
        let (lagrange, lagrange_reference): (Vec<_>, Vec<_>) = (17..=24).map(g1_multiple).unzip();
        let fingerprint = [0xa7; 32];

        assert!(
            !session
                .bls12_381_msm_srs_is_prepared(fingerprint)
                .expect("query worker cache")
        );
        session
            .prepare_bls12_381_msm_srs(fingerprint, &coefficient, &lagrange)
            .expect("prepare worker SRS");
        assert!(
            session
                .bls12_381_msm_srs_is_prepared(fingerprint)
                .expect("query prepared worker cache")
        );

        let single = [fr(1), fr(2), fr(3), fr(5), fr(8)];
        let actual = session
            .bls12_381_msm(
                fingerprint,
                Bls12_381MsmBasis::Coefficient,
                &single,
                Bls12_381MsmWindow::W8,
            )
            .expect("execute single worker MSM");
        assert_same_point(
            actual,
            expected_msm(&coefficient_reference[..single.len()], &single),
        );
        let fixed_actual = session
            .bls12_381_msm(
                fingerprint,
                Bls12_381MsmBasis::Coefficient,
                &single,
                Bls12_381MsmWindow::FixedW13,
            )
            .expect("execute fixed-base worker MSM");
        assert_same_point(
            fixed_actual,
            expected_msm(&coefficient_reference[..single.len()], &single),
        );

        let batch_a = [fr(13)];
        let batch_b = [fr(2), fr(0), fr(5), fr(9), fr(11), fr(13), fr(17), fr(19)];
        let outputs = session
            .bls12_381_msm_batch(
                fingerprint,
                Bls12_381MsmBasis::Lagrange,
                &[&batch_a, &batch_b],
                Bls12_381MsmWindow::W9,
            )
            .expect("execute ragged worker MSM batch");
        assert_eq!(outputs.len(), 2);
        assert_same_point(outputs[0], expected_msm(&lagrange_reference[..1], &batch_a));
        assert_same_point(outputs[1], expected_msm(&lagrange_reference, &batch_b));
        let fixed_outputs = session
            .bls12_381_msm_batch(
                fingerprint,
                Bls12_381MsmBasis::Lagrange,
                &[&batch_a, &batch_b],
                Bls12_381MsmWindow::FixedW13,
            )
            .expect("execute fixed-base ragged worker MSM batch");
        assert_eq!(fixed_outputs.len(), 2);
        assert_same_point(
            fixed_outputs[0],
            expected_msm(&lagrange_reference[..1], &batch_a),
        );
        assert_same_point(
            fixed_outputs[1],
            expected_msm(&lagrange_reference, &batch_b),
        );

        let missing = session
            .bls12_381_msm(
                [0x5c; 32],
                Bls12_381MsmBasis::Coefficient,
                &single,
                Bls12_381MsmWindow::W10,
            )
            .expect_err("unprepared SRS must fail closed");
        assert!(missing.to_string().contains("was not prepared"));
    }
}
