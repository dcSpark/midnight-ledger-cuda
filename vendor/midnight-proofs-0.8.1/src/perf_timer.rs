#![allow(missing_docs)]

use std::{
    collections::BTreeMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const CREATE_PROOF_LABEL: &str = "proof_server.prover.plonk.create_proof";
static ACTIVE_PROOF_TIMERS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Eq, PartialEq)]
enum TimerMode {
    Off,
    Trace,
    Aggregate,
}

#[derive(Default)]
struct Aggregate {
    calls: u64,
    counted_calls: u64,
    count: u128,
    total_ns: u128,
    maximum_ns: u128,
}

pub(crate) struct ScopedTimer {
    label: &'static str,
    count: Option<usize>,
    start: Option<Instant>,
    mode: TimerMode,
    proof_root: bool,
}

impl ScopedTimer {
    pub(crate) fn new(label: &'static str) -> Self {
        Self::with_count(label, None)
    }

    pub(crate) fn with_count(label: &'static str, count: Option<usize>) -> Self {
        let mode = timer_mode();
        let proof_root = mode == TimerMode::Aggregate && label == CREATE_PROOF_LABEL;
        if proof_root {
            ACTIVE_PROOF_TIMERS.fetch_add(1, Ordering::AcqRel);
        }
        Self {
            label,
            count,
            start: (mode != TimerMode::Off
                && (mode != TimerMode::Trace || trace_label_enabled(label)))
            .then(Instant::now),
            mode,
            proof_root,
        }
    }
}

fn trace_label_enabled(label: &str) -> bool {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER
        .get_or_init(|| {
            std::env::var("MIDNIGHT_PROVER_TIMER_FILTER")
                .ok()
                .filter(|filter| !filter.is_empty())
        })
        .as_deref()
        .is_none_or(|filter| label.starts_with(filter))
}

fn timers_enabled() -> bool {
    timer_mode() != TimerMode::Off
}

fn timer_mode() -> TimerMode {
    static MODE: OnceLock<TimerMode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var_os("MIDNIGHT_PROVER_AGGREGATE_TIMERS") {
            Some(_) => TimerMode::Aggregate,
            None => match std::env::var("MIDNIGHT_PROVER_TIMERS") {
                Ok(_) => TimerMode::Trace,
                Err(std::env::VarError::NotPresent) => TimerMode::Off,
                Err(std::env::VarError::NotUnicode(_)) => TimerMode::Trace,
            },
        },
    )
}

fn aggregates() -> &'static Mutex<BTreeMap<&'static str, Aggregate>> {
    static AGGREGATES: OnceLock<Mutex<BTreeMap<&'static str, Aggregate>>> = OnceLock::new();
    AGGREGATES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn record_aggregate(label: &'static str, count: Option<usize>, elapsed_ns: u128) {
    let mut aggregates = aggregates()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = aggregates.entry(label).or_default();
    aggregate.calls = aggregate.calls.saturating_add(1);
    if let Some(count) = count {
        aggregate.counted_calls = aggregate.counted_calls.saturating_add(1);
        aggregate.count = aggregate.count.saturating_add(count as u128);
    }
    aggregate.total_ns = aggregate.total_ns.saturating_add(elapsed_ns);
    aggregate.maximum_ns = aggregate.maximum_ns.max(elapsed_ns);
}

fn emit_aggregates() {
    let snapshot = {
        let mut aggregates = aggregates()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *aggregates)
    };
    let mut snapshot = snapshot.into_iter().collect::<Vec<_>>();
    snapshot.sort_unstable_by(|left, right| right.1.total_ns.cmp(&left.1.total_ns));
    for (label, aggregate) in snapshot {
        let total_ms = aggregate.total_ns as f64 / 1_000_000.0;
        let mean_ms = total_ms / aggregate.calls.max(1) as f64;
        let maximum_ms = aggregate.maximum_ns as f64 / 1_000_000.0;
        eprintln!(
            "[midnight-prover-aggregate] {label} calls={} counted_calls={} count={} total={total_ms:.3}ms mean={mean_ms:.3}ms max={maximum_ms:.3}ms",
            aggregate.calls, aggregate.counted_calls, aggregate.count,
        );
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            match self.mode {
                TimerMode::Off => {}
                TimerMode::Trace => {
                    let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
                    let epoch_ns = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|duration| duration.as_nanos())
                        .unwrap_or_default();
                    if let Some(count) = self.count {
                        eprintln!(
                            "[midnight-prover] {} count={} {:.3}ms @{}",
                            self.label, count, elapsed_ms, epoch_ns
                        );
                    } else {
                        eprintln!(
                            "[midnight-prover] {} {:.3}ms @{}",
                            self.label, elapsed_ms, epoch_ns
                        );
                    }
                }
                TimerMode::Aggregate => {
                    record_aggregate(self.label, self.count, elapsed.as_nanos());
                    if self.proof_root && ACTIVE_PROOF_TIMERS.fetch_sub(1, Ordering::AcqRel) == 1 {
                        emit_aggregates();
                    }
                }
            }
        }
    }
}

pub(crate) fn timed<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let _timer = ScopedTimer::new(label);
    f()
}

pub(crate) fn timed_count<T>(label: &'static str, count: usize, f: impl FnOnce() -> T) -> T {
    let _timer = ScopedTimer::with_count(label, Some(count));
    f()
}

pub(crate) fn timed_count_detailed<T>(
    label: &'static str,
    count: usize,
    f: impl FnOnce() -> T,
) -> T {
    if timers_enabled() {
        timed_count(label, count, f)
    } else {
        f()
    }
}
