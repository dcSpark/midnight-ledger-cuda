# Ledger 9 Metal production acceptance — 2026-07-17

## 4x optimization addendum

After the acceptance run below, the standalone MSM integration was changed
from one serialized queue to two basis-sharded queues, and proof-runtime lock
scope was narrowed so callback-scoped standalone MSM work does not leave the
independent field queue idle. The final settled seven-sample run measured
1,128.954 ms median and 1,181.488 ms p95, with 966 Metal MSM outputs, 240
coalesced submissions, zero failed submissions, and correct output for all
warm-up and measured transactions. Relative to the frozen 4,591.330 ms clean
CPU median this is 4.067x.

The one-queue control from the same rebuilt binary measured 1,160.362 ms
median. Naive round-robin queues reached 1,130.301 ms median but increased
submissions from 239 to 302 across seven runs and produced a 1,227.126 ms p95;
that router was rejected. Basis sharding retained coalescing (240 submissions)
and reduced the control median by 23.350 ms with a lower 1,181.294 ms p95.

This addendum records an optimized default, not a new production acceptance:
the 20-pair and 100-transaction gates were not repeated. The prior paired CPU
median sets a 1,133.657 ms 4x proxy line, which the final short run clears by
4.703 ms. The authoritative accepted paired result below therefore remains
3.241x, while 4.067x is the latest settled optimization measurement.

Two final overlap candidates were rejected and removed. A two-session runtime
measured 1,144.959 ms median and 1,307.473 ms p95 and duplicated uploads and key
contexts. Concurrent proof-facing quotient audits were byte-exact but measured
1,185.233 ms in the authentic falsifier. The retained implementation keeps one
Metal session and the exact serialized quotient counter audit while releasing
the session execution lock only around independently owned standalone MSM
callbacks.

## Decision

The production `metal-hybrid` route passes the end-to-end performance gate on
the 32-GPU-core Apple M1 Max development host. The authoritative paired run is
byte-exact, has no hidden fallback, and measures a **3.241x median speedup** over
the CPU route from the same binary and fixture.

This is a CPU+Metal backend, not a CUDA+Metal process and not a pure-Metal
claim. Metal owns the expensive NTT, fixed-base MSM, quotient construction and
commitments, and resident multi-open stages. CPU retains witness synthesis,
transcript/orchestration, and one combined quotient-opening materialization per
proof. The route remains named `metal-hybrid` until that materialization and the
remaining host hot-loop ownership are removed.

## Reproduction

Rust 1.95.0 was used. The acceptance binary contains the production `metal`
feature, not `metal-diagnostics`:

```bash
cargo build --release \
  -p midnight-proof-server \
  --bin proof-backend-parity \
  --features deterministic-test-rng,metal

MIDNIGHT_LEDGER_TEST_STATIC_DIR="$PWD/scripts" \
  target/release/proof-backend-parity \
  --backend cpu-metal-hybrid \
  --iterations 20 \
  --warmup 2 \
  --json /tmp/metal-production-final-paired20.json
```

No Metal acceleration or tuning environment variable is required. Selecting
the route enables native MSM, resident quotient substitution, resident
multi-open, zero coalescing delay, and the bounded four-proof host scheduler by
default. Environment overrides remain diagnostic controls, not production
requirements.

The compact machine-readable counters for the acceptance and final soak are in
`reports/ledger-9-metal-production-evidence-2026-07-17.json`.

The fixture is the generated-key balanced zswap transaction: five proof jobs,
1,397 instructions, maximum `k=15`, and 56,950 aggregate rows. Paired rounds
alternate which backend executes first and use identical seed order.

## End-to-end result

| Metric | CPU | `metal-hybrid` | Speedup |
| --- | ---: | ---: | ---: |
| Mean | 4,565.406 ms | 1,407.204 ms | 3.244x |
| Median | 4,553.227 ms | 1,404.859 ms | **3.241x** |
| Nearest-rank p95 | 4,632.661 ms | 1,426.223 ms | **3.248x** |

All 20 serialized transactions, containing 100 proofs per backend, were
byte-identical. Canonical proof self-verification remained in the timed path.
The Metal route recorded 820 command buffers, 8,500 compute dispatches, zero
post-selection fallback, and zero failed MSM submissions.

The earlier accepted warm CPU reference was 5,218.394 ms. Relative to that
frozen reference the new median is approximately 3.71x, but the paired 3.241x
number above is the primary claim because it controls for the current binary
and machine state.

## Authentic MSM coverage

Each transaction issues the complete 147-output commitment workload:

- 142 nonempty outputs execute in the standalone Metal MSM library;
- five all-zero outputs use the constant identity result and report zero
  nonzero CPU scalars; and
- no nonempty MSM falls back to BLST or another CPU curve implementation.

Across the 20 samples the worker produced 2,840 Metal outputs in 767 coalesced
submissions. Aggregate Metal MSM service time was 18,935.778 ms; service time
overlaps independent proof jobs and therefore must not be added to transaction
latency. The fixed-base v10 route uses Metal scalar canonicalization,
histogram/scan/scatter routing, compact buckets, a 512-record long-bucket
threshold, 2,048-record tiles, direct first-point XYZZ initialization,
cooperative reduction, block integration, and window collection.

## Residency and transfers

The 20 Metal samples reported 2,584,827,200 logical host-written bytes and
676,344,320 logical host-read bytes: approximately 123.254 MiB H2D and
32.251 MiB D2H per transaction on unified memory. These counters describe
logical CPU mappings/copies, not PCIe traffic.

There were 100 resident quotient epochs and 100 resident quotient MSM leases.
No quotient commitment scalar lease was exposed to CPU MSM. There were also 100
whole-polynomial materializations: one combined quotient opening for each of
the five proofs in every transaction. This is the largest explicit remaining
residency seam and is why the route is not labeled pure `metal`.

The resident multi-open arena keeps q/f/pi work and its two commitments on
Metal, returning only compact transcript evaluations and points. A candidate
that moved q linear combinations to Metal from host-resident source slices was
byte-exact but regressed the five-sample median from 1,402.275 ms to 1,418.232
ms and increased logical writes from about 123 MiB to about 327 MiB per
transaction. It was rejected. The next version must consume sources that are
already resident rather than uploading 73 source polynomials in place of five
precombined q polynomials.

## Reliability and lifecycle evidence

The final production-default 100-pair soak completed 100/100 byte-exact
transactions. CPU median/p95 were 4,583.512/5,087.574 ms; Metal median/p95 were
1,403.524/1,454.612 ms. The paired median speedup was **3.266x**, and the p95
ratio was 3.498x. It executed 4,100 command buffers and 42,500 compute
dispatches, produced 14,200 Metal MSM outputs in 3,797 coalesced submissions,
and recorded zero fallback, MSM failure, quotient failure, clear failure,
quarantine, or poison event. All 500 quotient commitment leases stayed
resident; the known combined-opening seam materialized exactly once per proof.

The standalone Metal differential suite passed 11/11 device tests, covering all
variable windows, all fixed preparation widths, prefix/ragged batches,
multi-source resident Montgomery inputs, long-bucket tiling, complete-formula
exceptional cases, invalid-residue rejection, and engine reuse after rejected
input. The Metal prover runtime passed 117 sequential unit tests plus all field,
G1, NTT, and digest integrations. Injected completion, clear, panic, quarantine,
and poisoned-session tests passed. The runtime test audit also corrected stale
radix-8 expectations after the production NTT ceiling moved to radix 9.

## Theoretical and practical ceiling

The previously measured infinite-device Amdahl ceiling is approximately 10.2x:
about 512 ms of witness synthesis, request work, transcript/orchestration, and
other host work remains when all measured accelerator-eligible work is treated
as free. This is a structural upper bound, not an M1 Max forecast.

The current 1,405 ms median captures roughly four-fifths of the removable time
between the frozen 5,218 ms CPU reference and that 512 ms structural floor.
The remaining practical opportunity is concentrated in:

1. removing the combined quotient readback and passing its resident opening
   directly into evaluation and multi-open;
2. reusing resident advice/fixed/permutation polynomials as multi-open sources;
3. further reducing fixed-base MSM service, which remains the dominant device
   interval; and
4. overlapping or moving the remaining witness-derived host polynomial work.

The theoretical 10.2x figure must not be reported as expected hardware
performance. The measured production result is 3.241x paired median; further
work should be judged by end-to-end deltas on this same fixture.

## Security qualification

The accepted MSM uses the explicit trusted-local data-dependent observation
profile. Scalar digits affect bucket occupancy, addresses, and work. It is
fail-closed and erases owned scalar-derived scratch, but it does not claim
constant observation against a hostile local GPU co-tenant. Production use
outside that threat model requires a separate security decision or an
oblivious fixed-work implementation.

CUDA and sppark remain semantic and regression references only. No CUDA runtime
is linked into this macOS route, and no source was copied from sppark or zkMetal.
