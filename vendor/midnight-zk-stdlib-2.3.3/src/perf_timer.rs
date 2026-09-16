use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(crate) struct ScopedTimer {
    label: &'static str,
    count: Option<usize>,
    start: Option<Instant>,
}

impl ScopedTimer {
    pub(crate) fn new(label: &'static str) -> Self {
        Self::with_count(label, None)
    }

    pub(crate) fn with_count(label: &'static str, count: Option<usize>) -> Self {
        Self {
            label,
            count,
            start: std::env::var_os("MIDNIGHT_PROVER_TIMERS").map(|_| Instant::now()),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
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
    }
}

pub(crate) fn timed<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let _timer = ScopedTimer::new(label);
    f()
}
