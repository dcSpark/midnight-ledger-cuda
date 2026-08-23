//! Opt-in, aggregate-only diagnostics for the Metal MSM workload probe.
//!
//! This module is public only so the prover can delimit `create_proof`. The
//! output intentionally excludes scalar values, signed digits, point indices,
//! and bucket identities. It is not a production telemetry surface.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

use crate::{Fq, msm::get_booth_index};

const OUTPUT_ENV: &str = "MIDNIGHT_MSM_RUN_DIAGNOSTIC";
const SCHEMA: &str = "midnight-msm-runs-v1";
static ACTIVE_PROOF_SCOPES: AtomicUsize = AtomicUsize::new(0);
static NEXT_CALL: AtomicU64 = AtomicU64::new(0);
static OUTPUT_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static OUTPUT: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();

/// Keeps aggregate MSM capture active for the lifetime of one proof.
///
/// The counter is process-global because commitment batches may execute on
/// Rayon workers that do not inherit thread-local state.
#[must_use]
#[derive(Debug)]
pub struct ScopeGuard {
    active: bool,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if self.active {
            let previous = ACTIVE_PROOF_SCOPES.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(previous > 0, "MSM diagnostic scope counter underflow");
        }
    }
}

/// Starts aggregate capture when `MIDNIGHT_MSM_RUN_DIAGNOSTIC` names a file.
pub fn begin_scope() -> ScopeGuard {
    let active = output_path().is_some();
    if active {
        ACTIVE_PROOF_SCOPES.fetch_add(1, Ordering::AcqRel);
    }
    ScopeGuard { active }
}

/// Records one selected-size G1 MSM while a proof scope is active.
pub fn record_g1_msm(scalars: &[Fq]) {
    if ACTIVE_PROOF_SCOPES.load(Ordering::Acquire) == 0 {
        return;
    }
    let Some(profile) = Profile::for_len(scalars.len()) else {
        return;
    };

    let mut occupancies = vec![vec![0u32; profile.bucket_count()]; profile.windows];
    let mut nonzero_scalars = 0usize;
    for scalar in scalars {
        let recoded = RecodedScalar::new(scalar);
        nonzero_scalars += usize::from(recoded.canonical.iter().any(|&byte| byte != 0));
        for (window, buckets) in occupancies.iter_mut().enumerate() {
            let digit = recoded.digit(window, profile.window_bits);
            let magnitude = digit.unsigned_abs() as usize;
            if magnitude != 0 {
                buckets[magnitude - 1] = buckets[magnitude - 1]
                    .checked_add(1)
                    .expect("MSM diagnostic bucket count overflow");
            }
        }
    }

    let call = NEXT_CALL.fetch_add(1, Ordering::Relaxed);
    let line = encode_call(call, profile, nonzero_scalars, &occupancies);
    let mut output = output()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    output
        .write_all(line.as_bytes())
        .and_then(|()| output.write_all(b"\n"))
        .and_then(|()| output.flush())
        .expect("write aggregate MSM run diagnostic");
}

fn output_path() -> Option<&'static PathBuf> {
    OUTPUT_PATH
        .get_or_init(|| {
            std::env::var_os(OUTPUT_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .as_ref()
}

fn output() -> &'static Mutex<BufWriter<File>> {
    OUTPUT.get_or_init(|| {
        let path = output_path().expect("MSM diagnostic output path disappeared");
        let mut options = OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(path).unwrap_or_else(|error| {
            panic!(
                "open aggregate MSM run diagnostic {}: {error}",
                path.display()
            )
        });
        #[cfg(unix)]
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .unwrap_or_else(|error| {
                panic!(
                    "restrict aggregate MSM run diagnostic {}: {error}",
                    path.display()
                )
            });
        Mutex::new(BufWriter::new(file))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Profile {
    logical_len: usize,
    window_bits: u32,
    windows: usize,
}

impl Profile {
    fn for_len(logical_len: usize) -> Option<Self> {
        let window_bits = match logical_len {
            8_191 | 8_192 => 8,
            16_383 | 16_384 => 9,
            32_767 | 32_768 => 10,
            _ => return None,
        };
        let windows = (255usize - 1) / window_bits as usize + 1;
        Some(Self {
            logical_len,
            window_bits,
            windows,
        })
    }

    fn bucket_count(self) -> usize {
        1usize << (self.window_bits - 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecodedScalar {
    canonical: [u8; 32],
}

impl RecodedScalar {
    fn new(scalar: &Fq) -> Self {
        Self {
            canonical: scalar.to_bytes_le(),
        }
    }

    fn digit(self, window: usize, window_bits: u32) -> i32 {
        get_booth_index(window, window_bits as usize, &self.canonical)
    }
}

fn encode_call(
    call: u64,
    profile: Profile,
    nonzero_scalars: usize,
    occupancies: &[Vec<u32>],
) -> String {
    let mut encoded = String::with_capacity(8 * 1024);
    write!(
        encoded,
        "{{\"schema\":\"{SCHEMA}\",\"call\":{call},\"logical_len\":{},\"nonzero_scalars\":{nonzero_scalars},\"window_bits\":{},\"window_count\":{},\"windows\":[",
        profile.logical_len, profile.window_bits, profile.windows
    )
    .expect("write JSON to string");

    let mut total_records = 0u64;
    let mut total_runs = 0u64;
    for (window, buckets) in occupancies.iter().enumerate() {
        if window != 0 {
            encoded.push(',');
        }
        let records = buckets.iter().map(|&count| u64::from(count)).sum::<u64>();
        let runs = buckets.iter().filter(|&&count| count != 0).count();
        let max_run = buckets.iter().copied().max().unwrap_or(0);
        total_records += records;
        total_runs += runs as u64;

        let mut histogram = BTreeMap::<u32, u32>::new();
        for &count in buckets {
            if count != 0 {
                *histogram.entry(count).or_default() += 1;
            }
        }
        write!(
            encoded,
            "{{\"window\":{window},\"records\":{records},\"nonempty_runs\":{runs},\"max_run\":{max_run},\"run_histogram\":["
        )
        .expect("write JSON to string");
        for (index, (run_len, count)) in histogram.into_iter().enumerate() {
            if index != 0 {
                encoded.push(',');
            }
            write!(encoded, "[{run_len},{count}]").expect("write JSON to string");
        }
        encoded.push_str("]}");
    }
    write!(
        encoded,
        "],\"total_records\":{total_records},\"total_nonempty_runs\":{total_runs},\"total_pair_reductions\":{}}}",
        total_records - total_runs
    )
    .expect("write JSON to string");
    encoded
}

#[cfg(test)]
mod tests {
    use ff::Field;

    use super::*;

    #[test]
    fn selected_profiles_match_the_metal_plan() {
        for (logical_len, window_bits, windows, buckets) in [
            (8_191, 8, 32, 128),
            (8_192, 8, 32, 128),
            (16_383, 9, 29, 256),
            (16_384, 9, 29, 256),
            (32_767, 10, 26, 512),
            (32_768, 10, 26, 512),
        ] {
            let profile = Profile::for_len(logical_len).expect("selected profile");
            assert_eq!(profile.window_bits, window_bits);
            assert_eq!(profile.windows, windows);
            assert_eq!(profile.bucket_count(), buckets);
        }
        assert_eq!(Profile::for_len(4_096), None);
    }

    #[test]
    fn booth_digits_recompose_boundary_scalars() {
        let scalars = [
            Fq::ZERO,
            Fq::ONE,
            -Fq::ONE,
            Fq::from(255),
            Fq::from(256),
            Fq::from(257),
            Fq::from(u64::MAX),
        ];
        for scalar in scalars {
            for window_bits in [8, 9, 10] {
                let recoded = RecodedScalar::new(&scalar);
                let windows = (255usize - 1) / window_bits as usize + 1;
                let radix = Fq::from(1u64 << window_bits);
                let mut factor = Fq::ONE;
                let mut recomposed = Fq::ZERO;
                for window in 0..windows {
                    let digit = recoded.digit(window, window_bits);
                    let mut term = Fq::from(u64::from(digit.unsigned_abs()));
                    if digit < 0 {
                        term = -term;
                    }
                    recomposed += term * factor;
                    factor *= radix;
                }
                assert_eq!(recomposed, scalar, "window_bits={window_bits}");
            }
        }
    }

    #[test]
    fn encoded_call_contains_only_aggregate_run_data() {
        let profile = Profile::for_len(8_192).expect("selected profile");
        let mut occupancies = vec![vec![0; profile.bucket_count()]; profile.windows];
        occupancies[0][0] = 3;
        occupancies[0][1] = 3;
        occupancies[0][2] = 1;
        let encoded = encode_call(7, profile, 2, &occupancies);
        assert!(encoded.contains("\"run_histogram\":[[1,1],[3,2]]"));
        assert!(encoded.contains("\"total_pair_reductions\":4"));
        for forbidden in [
            "raw_scalar",
            "scalar_limbs",
            "digits",
            "point_index",
            "bucket_index",
            "signs",
        ] {
            assert!(!encoded.contains(forbidden));
        }
    }
}
