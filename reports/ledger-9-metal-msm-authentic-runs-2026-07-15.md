# Ledger 9 authentic Metal MSM run distribution — 2026-07-15

> Handling: internal sensitive diagnostic. Aggregate occupancy and run-length
> data remain witness-derived and are not approved for public release without
> security-owner review.

## Result

The seed-24302 CPU prover now emits an aggregate-only workload diagnostic at the
actual KZG MSM boundary. It confirms all 147 calls, 3,235,835 logical positions,
and 1,988,928 nonzero scalars from the timer analysis, while replacing the
uniform-Fr estimate with the real signed-digit distribution:

```text
nonzero digit records       47,093,103
nonempty bucket runs         1,054,037
balanced-tree reductions    46,039,066
```

The authentic reduction count is 12.95% below the 52.886M probability-model
projection, but the workload is far more skewed. Maximum individual run lengths
are 8,049 / 9,732 / 16,797 for the 8K / 16K / 32K classes. Rank-by-rank affine
updates are therefore ineligible; only balanced per-bucket trees reach the
public-capacity depths of 13 / 14 / 15.

This improved the pre-measurement wide/global-tree arithmetic projection from
434.1 ms to about 382.5 ms including unrealistically cheap projective
integration. The completed replay showed that projection was not representative
of the full device dataflow: it reached only 30.633M reductions/s and projects
1,502.943 ms for bucket reduction. The global batch-affine branch is rejected;
see `ledger-9-metal-batch-affine-result-2026-07-15.md`.

The bounded recursive flat-segmented follow-up also failed. Its faster
128-lane variant reduced the representative 6,794,123 pairs in a 2,136.823 ms
median, only 3.180M useful reductions/s, and projects 14,479.771 ms for the full
population. That is 48.3x slower than the frozen promotion gate despite a
favorable sorted replay. See
`ledger-9-metal-flat-segmented-result-2026-07-15.md`. No measured M1 Max MSM
candidate remains eligible for proof routing.

## Capture boundary

`midnight_curves::msm_run_diagnostic::begin_scope` brackets only
`midnight_proofs::plonk::create_proof`. The process-global scope counter covers
Rayon commitment workers without admitting generated-key or post-proof verifier
MSMs. At `poly/kzg/msm.rs::msm_specific`, the collector observes the logical
coefficient slice before zero filtering and records only the six selected
8191/8192, 16383/16384, and 32767/32768 shapes.

The process-global counter is safe for this single-proof capture but is not a
concurrent-proof identity. While one scope is active, matching MSMs from another
thread/proof could be captured. Future recapture must serialize the diagnostic
process or propagate a proof-generation token into worker execution.

The collector delegates every digit to the existing
`midnight_curves::msm::get_booth_index` implementation after canonical
little-endian Fr conversion. It deliberately does **not** apply sppark's
additional whole-scalar `abs()` transformation: that is a different recoding
and would not reproduce the Metal plan.

Each JSONL record contains only:

- call and public logical/profile lengths;
- aggregate nonzero scalar count;
- per-window record count, nonempty-run count, maximum run; and
- exact `(run length, number of runs)` histograms.

The schema has no scalar representation, digit, sign, bucket identity, point
index, or point value. The Rust writer and normalized output use mode `0600` on
Unix. Capture is off unless `MIDNIGHT_MSM_RUN_DIAGNOSTIC` names a file.

## Reproduction

Build the deterministic CPU harness:

```bash
CARGO_INCREMENTAL=0 cargo build \
  -p midnight-proof-server \
  --bin proof-backend-parity \
  --release \
  --features deterministic-test-rng
```

Capture the successful proof:

```bash
MIDNIGHT_MSM_RUN_DIAGNOSTIC=/tmp/metal-msm-runs-seed24302.jsonl \
MIDNIGHT_LEDGER_TEST_STATIC_DIR="$PWD/scripts" \
./target/release/proof-backend-parity \
  --backend cpu \
  --fixture zswap-transaction \
  --iterations 1 \
  --warmup 0 \
  --seed 24302 \
  --json /tmp/metal-msm-run-capture-seed24302.json
```

Validate the content-minimization schema, every histogram invariant, and the
exact seed-24302 shape/nonzero anchors:

```bash
python3 scripts/metal_msm_runs.py \
  /tmp/metal-msm-runs-seed24302.jsonl \
  --expect-seed-24302 \
  --output /tmp/metal-msm-runs-seed24302-summary.json
```

The diagnostic adds CPU recoding and synchronized file output inside the proof;
its 6,118.168 ms wall time is not a performance sample. The proof itself retains
the established byte digest:

```text
len:23476 xor:191 sum64:2924713 rolling:105419955334794057
```

## Artifacts

The first successful capture produced:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `/tmp/metal-msm-runs-seed24302.jsonl` | 1.1 MiB | `81672594e3e7ecefe4da9227a4ba39b3b2207b7da96e62891ff57d93a681a4f9` |
| `/tmp/metal-msm-runs-seed24302-summary.json` | 957 KiB | `42f6384747a841352081f78e10e9b6c3964b21ce327c1f7fba05fe4e05b5e47b` |
| `/tmp/metal-msm-run-capture-seed24302.json` | 1.9 KiB | `15d1e430a3059789c2a01703b4710934297ef1b6a93317de81ec3958b6e40282` |

JSONL call completion order is nondeterministic under Rayon. Repeating the
default-parallelism command produced a different source hash but the same proof
digest and normalized workload hash:

```text
47a6510def79f770e1efe57c8efd0630d916e1f60037865a8f9a0a295d4e27f1
```

The artifacts are ephemeral and outside the checkout. The command and normalized
hash, not `/tmp` persistence, are the reproduction contract.

These artifacts are still sensitive witness-derived diagnostics. Omitting
scalar values, signs, bucket identities, and point indices limits content but
does not declassify occupancy or run histograms. Mode `0600` and hashes are
handling controls, not confidentiality proofs.

## Class distribution

| Class | Calls | Logical | Nonzero | Bits/windows | Records | Runs | Pair reductions | Max run/depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 23 | 188,415 | 117,871 | 8 / 32 | 3,380,497 | 81,200 | 3,299,297 | 8,049 / 13 |
| 16K | 62 | 1,015,806 | 625,897 | 9 / 29 | 15,826,318 | 338,553 | 15,487,765 | 9,732 / 14 |
| 32K | 62 | 2,031,614 | 1,245,160 | 10 / 26 | 27,886,288 | 634,284 | 27,252,004 | 16,797 / 15 |
| **Total** | **147** | **3,235,835** | **1,988,928** | | **47,093,103** | **1,054,037** | **46,039,066** | |

The exact-zero/nonzero counts agree with the independent timer parser. The
record/run totals reconcile from the histogram identities for every window:

```text
sum(run_count)                 = nonempty runs
sum(run_length * run_count)    = records
records - nonempty runs        = balanced pair reductions
```

## Updated arithmetic decision

For `N = 46,039,066` balanced reductions, the fixed complete affine schedule
remains `5N` multiplications, `2N` squares, and `9N` add/sub operations, plus a
small number of globally batched inversions. Applying the measured isolated
rates gives:

| Rate assumption | M/S | add/sub | Bucket phase | With optimistic integration |
| --- | ---: | ---: | ---: | ---: |
| Current six-`u64` measured rates | 544.2 ms | 137.9 ms | **682.1 ms** | >717 ms |
| Ideal 1.540404B/s wide M/S plus six-`u64` add/sub proxy | 209.2 ms | 137.9 ms | **347.1 ms** | **382.5 ms** |

The BLST win-4 inversion chain reduces a local inverse from 613 to 460 field
operations, while lower-storage generated chains offer 463 operations with nine
Fp temporaries or 488 operations with five. For the authentic count, raw
M/S-equivalent projections including the existing 33.4M integration allowance
are:

| Local batch `G` | Current inverse | 460-op inverse |
| ---: | ---: | ---: |
| 128 | 576.2M / 374.0 ms | 521.1M / 338.3 ms |
| 256 | 465.9M / 302.5 ms | 438.4M / 284.6 ms |
| 512 | 410.8M / 266.7 ms | 397.0M / 257.7 ms |

Those local numbers still exclude the `9N` add/sub schedule, routing, memory,
and synchronization. Benchmark the 5-/9-/16-temporary chains rather than
selecting the shortest chain by operation count; register pressure may reverse
their order.

The authentic absolute stop/promote gates are now:

- immediate rejection above 364.6 ms bucket reduction, equivalent to below
  about 126.3M weighted reductions/s;
- one bounded layout/inversion review between 126.3M/s and 153.5M/s; and
- promotion to record routing only at or below 300 ms, equivalent to at least
  153.5M/s for this workload.

The absolute 300/400/500 ms phase gates remain unchanged. Reduced authentic
work does not authorize a looser proof target.
