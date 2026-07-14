use std::{
    future::Future,
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("MIDNIGHT_PROVER_TIMERS").is_some())
}

static ENABLED: OnceLock<bool> = OnceLock::new();

#[doc(hidden)]
pub fn timed<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let result = f();
    emit(label, start.elapsed().as_secs_f64() * 1e3);
    result
}

#[doc(hidden)]
pub async fn timed_async<T, F>(label: &'static str, future: F) -> T
where
    F: Future<Output = T>,
{
    if !enabled() {
        return future.await;
    }
    let start = Instant::now();
    let result = future.await;
    emit(label, start.elapsed().as_secs_f64() * 1e3);
    result
}

fn emit(label: &str, elapsed_ms: f64) {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    eprintln!("[midnight-prover] {label} {elapsed_ms:.3}ms @{nanos}");
}
