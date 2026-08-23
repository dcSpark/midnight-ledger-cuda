# Ledger 9 resident Metal quotient decision result — 2026-07-15

## Decision

Reject the current resident-field-plus-CPU-BLST quotient route as a 3x-enabling
implementation path. Its five-proof safe-epoch median is **1,184.911 ms**, almost
double the predeclared **600 ms** gate. Do not spend engineering time connecting
this diagnostic prefix to proof finalization. Preserve it as correctness,
lifecycle, and future native-MSM integration evidence.

This result rejects the measured CPU-MSM seam, not the resident field engine and
not every possible Metal quotient. The next material change must remove the CPU
BLST limb commitments through the independently packaged native Metal MSM and
keep the quotient output resident for evaluation and multi-open.

## Measured contract

The release benchmark used the balanced current-prover transaction fixture:

- two zswap spends, two zswap outputs, and one dust spend;
- five proof jobs, 1,397 IR instructions, 56,950 aggregate rows, maximum K=15;
- two warmups followed by five alternating CPU/Metal-hybrid samples;
- seeds 24303 through 24307 for the retained timed samples;
- one process Metal session and retained immutable PK/SRS contexts;
- one diagnostic quotient epoch and one intentional CPU-MSM lease per proof;
- elapsed time measured from request validation through Metal work, CPU BLST
  commitments, transcript-shadow challenge derivation, and mandatory GPU secret
  cleanup; and
- the ordinary CPU quotient still generated the real proof, so this is a
  discarded shadow-stage measurement, not an end-to-end acceleration claim.

Command:

```bash
CARGO_INCREMENTAL=0 ./scripts/gpuprof.sh parity \
  --backend cpu-vs-metal-hybrid \
  --fixture zswap-transaction \
  --repeat 5 \
  --warmup 2 \
  --seed 24301 \
  --metal-quotient-diagnostic \
  --cargo-features deterministic-test-rng,metal-quotient-diagnostic,perf-timers \
  --json benchmark-results/metal-quotient-shadow-smoke.json
```

The ignored local JSON artifact has SHA-256
`6c0109a98b9defd438be26684b5f9deb48ca8389fe069aa10132953f44083d58`.

## Results

| Metric | Result |
| --- | ---: |
| Safe five-proof quotient samples | 1,184.911, 1,157.477, 1,196.440, 1,163.003, 1,226.020 ms |
| Median | **1,184.911 ms** |
| Mean | 1,185.570 ms |
| Observed p95 / maximum | 1,226.020 ms |
| Gate | <=600.000 ms |
| Gate ratio | 1.975x the budget |
| Successful epochs / CPU-MSM leases | 25 / 25 |
| Lease bytes per transaction | 13,631,488 bytes (13.000 MiB) |
| Quotient-specific whole-polynomial materializations | 0 |
| Epoch, clear, poison, or quarantine failures | 0 |
| CPU median | 4,580.182 ms |
| Shadow Metal-hybrid median | 5,711.395 ms |
| Shadow ratio | 0.802x |
| Serialized result parity | byte exact, 23,476 bytes |

The 0.802x shadow ratio is expected because the diagnostic deliberately runs in
addition to the canonical CPU quotient. It must not be cited as the performance
of a substituted quotient route. The meaningful decision value is the isolated
safe-epoch median.

Using the prior 1,513.724 ms CPU quotient interval only as an optimistic
substitution model, replacing it with the measured 1,184.911 ms stage would move
the paired 4,580.182 ms CPU median to about 4,251.369 ms, or only about **1.077x**.
That remains far below the 1.20x per-slice promotion gate and cannot unlock 3x.

## Correctness and accounting

Every retained Metal sample recorded exactly five epochs and five scoped leases,
zero explicit CPU fallback, zero quotient materializations, and zero lifecycle
failure. The harness rejects shape drift, non-RawFr-aligned lease totals,
callback/Metal failure, failed erasure, quarantine, poison, or any quotient-local
whole-polynomial materialization. Proof verification completed normally, and the
CPU and Metal-hybrid serialized transactions were byte-identical for every seed.

The sample also contains the known host-round-trip NTT bridge materializations;
those are accounted separately from the quotient-local zero-copy assertion.

## Consequence

1. Keep pure `metal` fail closed.
2. Do not implement the CPU continuation polynomial solely to route this failed
   diagnostic into complete proofs.
3. Retain the quotient epoch as the resident field producer and integration
   oracle for a future native MSM.
4. Put all new MSM arithmetic, shaders, prepared-base ownership, tests, and
   benchmarks in `bls12-381-metal-msm`; the Midnight proof adapter may depend on
   that library, never the reverse.
5. Resume quotient integration only after a materially new standalone MSM design
   clears its independent arithmetic roofline and 400/500 ms aggregate MSM
   median/p95 gates.
