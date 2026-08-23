# Ledger 9 Metal batch-affine MSM decision plan — 2026-07-15

> Handling: internal sensitive diagnostic plan. It references aggregate
> witness-derived workload data and is not approved for public release.

## Verdict

The measured scalar CIOS branch and the lower-operation global batch-affine
branch are both closed. The 20-sample authentic-shape Metal replay reached only
30.633M reductions/s and projects the 46.039M-reduction bucket phase at
1,502.943 ms. That is 4.12x slower than the frozen immediate-stop ceiling of
364.6 ms and about 5.01x slower than the 300 ms promotion threshold.

The earlier isolated-rate projection of 347.1 ms for bucket reduction and
382.5 ms after unrealistically cheap projective integration was too optimistic
because it did not capture the complete product-tree/device dataflow. The
bounded probe has now falsified the design. Do not implement histogram/scatter,
proving-key integration, or witness-derived proof routing around it. Metal
proof routing remains disabled. Full evidence and raw samples are in
`ledger-9-metal-batch-affine-result-2026-07-15.md`.

The one remaining materially different layout experiment is also closed. A
recursive 32-point tiled reducer with 128-/256-lane threadgroup-local product
trees passed differential tests but measured only 3.180M/2.593M useful
reductions/s. Its faster G128 path projects the complete reduction workload at
14,479.771 ms. The required rate was 153.5M/s and the required projection was
300 ms. See `ledger-9-metal-flat-segmented-result-2026-07-15.md`.

An exact register-local ceiling probe then executed the complete per-reduction
`7 M/S + 9 A/S` operation population without materializing the scan. Its
20-sample median was 164.206M/s, projecting 280.374 ms of arithmetic alone and
leaving only 19.626 ms for every omitted dependency and dataflow stage. It
failed the predeclared 200M/s headroom gate, so a hierarchical global scan is
also rejected under this plan. See
`ledger-9-metal-arithmetic-ceiling-result-2026-07-15.md`.

## Evidence boundary

The persisted seed-24302 timers contained MSM input lengths but no scalar limbs
or digit histograms. That gap is now closed by the aggregate-only capture in
`reports/ledger-9-metal-msm-authentic-runs-2026-07-15.md`. It reproduces the
147-call logical/nonzero anchors and records only per-window run histograms—no
scalar, digit, sign, bucket identity, point index, or point value.

Those histograms and replay call fingerprints remain sensitive witness-derived
diagnostic data under `docs/metal-msm-security.md`. Aggregation, hashing, and
mode `0600` minimize and control the content; they do not declassify it.

The prior 54.153M record / 52.886M reduction numbers remain useful as the exact
uniform-Fr probability model, not as an observed witness trace. The authentic
structured workload contains 47.093M records and 46.039M reductions.

## Reuse anchors and boundaries

The strongest local algorithm and differential-oracle anchor is
`vendor/midnight-curves-0.3.1/src/msm.rs`:

- `batch_add` (starting at line 57) constructs numerator/denominator products,
  performs one batch inversion, and propagates inverse products in reverse;
- `Schedule` (starting at line 275) prevents two pending updates from targeting
  the same CPU bucket; and
- `msm_best` (starting at line 477) combines affine scheduled buckets,
  projective collision buckets, and weighted integration.

The reviewed file has SHA-256
`4391b2a54b58d2849272f512da74fe7db842f78a69ce1dca3be042cc7b42db76`.
Midnight Curves 0.3.1 is MIT/Apache-2.0. Treat it as an arithmetic reference,
not as a GPU scheduling template: its exceptional-case branches must become
mask selection and its finite CPU scheduler should become balanced bucket-local
trees batched across buckets.

Reuse these additional contracts, without copying their hardware topology:

- Preserve the CUDA ragged descriptor semantics—scalar offset, point offset,
  logical length, and nonzero length—from
  `midnight-sppark-msm/src/bls12_381_g1_msm.cu`. A Metal key context owns
  immutable, validated, batch-normalized affine SRS buffers; a separate request
  arena owns secret digits, records, tree partials, and erasure generations.
- Reuse sppark's signed-recoding, sorted-record, and weighted-reduction
  identities from `vendor/sppark-0.1.15/sppark/msm/pippenger.cuh`, but not its
  warp-32, cooperative-grid, CUB, stream, event, or shared-memory assumptions.
- Preserve the existing commitment/KZG proof batching boundaries. Introduce a
  target-selected resident handle rather than passing Metal through
  `Bls12_381FrDeviceVec` or another CUDA-shaped pointer type.
- Bind the immutable Metal key context to a complete proving-key identity or
  fingerprint. Do not copy the sampled raw-pointer cache key in the vendored
  CPU curve adapter; allocation reuse or interior mutation could alias a stale
  context.

The design is consistent with the parallel Montgomery trick and batched affine
bucket methods in [PipeMSM](https://eprint.iacr.org/2022/999.pdf) and the
scheduling discussion in [CycloneMSM](https://eprint.iacr.org/2022/1396.pdf).
The CUDA XYZZ implementation remains a performance comparison, not a source for
a one-for-one Metal port.

If formulas or scheduling code are materially copied, extend
`midnight-metal-prover/PROVENANCE.md` with exact source files, versions,
revisions, and licenses. Cargo-registry BLST sources are audit references, not
stable project inputs.

## Authentic workload

The selected 147 MSMs contain 3,235,835 logical entries and 1,988,928 nonzero
entries. The current signed-window plan exposes:

```text
23 * 32 * 128 =  94,208  bucket/window slots (8K class)
62 * 29 * 256 = 460,288  bucket/window slots (16K class)
62 * 26 * 512 = 825,344  bucket/window slots (32K class)
total          1,379,840
```

The validated seed-24302 capture gives:

```text
nonzero digit records               47.093M
nonempty bucket runs                 1.054M
bucket-reduction pairs              46.039M
  8K / 16K / 32K                     3.299M / 15.488M / 27.252M
```

The structured scalars are much less Poisson-like than the probability model.
Observed maximum runs are 8,049 / 9,732 / 16,797 for the 8K / 16K / 32K
classes. Several runs are close to a complete logical polynomial. Sequential
rank-by-rank bucket updates would therefore serialize nearly an entire
32K-class MSM.

Use a balanced binary tree inside every bucket and batch all same-level pairs
across the chunk. Authentic maximum and fixed public-capacity depths are
13/14/15 for the 8K/16K/32K classes. With chunk capacities of
8/8/4 MSMs, the 23/62/62 calls require 3/8/16 chunks. Dispatching every
public-capacity level requires 391 global inversion levels. The topology
compiler must report the authentic active count; omitting neutral tails is
allowed only by the approved trusted-local profile.

Weighted bucket integration still requires up to 1,379,840 mixed additions and
1,379,840 complete additions. Window collection adds about 3,999 group
additions and 36,828 doublings.

## Arithmetic feasibility

A branchless affine-or-infinity pair reducer computes ordinary-add and doubling
candidates and masks identity, equality, inverse-pair, and zero-doubling cases.
Excluding inversion, the fixed schedule is:

```text
2M + 2S + 9A per reduced pair
```

A classical product-tree batch inversion of `n` adjusted nonzero denominators
costs `3(n - 1)M + 1I`. One globally batched tree level therefore costs:

```text
(5n - 3)M + 2nS + 9nA + 1I
```

The current fixed-exponent inverse executes 229 multiplications and 384 squares
(613 dependent field operations), not 609. Across `N` reduced pairs and `L`
global level/chunk batches, the schedule is approximately:

```text
M = 5N + 226L
S = 2N + 384L
A = 9N
```

At `N = 46.039M`, root-inverse operation count is small; dependency latency and
the hierarchical product scan still have to be measured. The arithmetic
projection is:

| Rate assumption | M/S time | add/sub time | bucket phase | with optimistic integration |
| --- | ---: | ---: | ---: | ---: |
| Current measured six-`u64` M/S rates | 544.2 ms | 137.9 ms | **682.1 ms** | >717 ms |
| Ideal 1.540404B/s wide rate for both M and S; six-`u64` add/sub rate as an optimistic proxy | 209.2 ms | 137.9 ms | **347.1 ms** | **382.5 ms** |

The 35.4 ms integration allowance uses the temporary synthetic 77.98M
complete-add/s result. That result is not an on-curve G1 implementation and is
deliberately over-optimistic. The table also omits doublings, record routing,
dispatch overhead, erasure, and errors. It is a feasibility projection, not a
mathematical lower bound: a fused kernel can overlap instruction classes, so the
bounded probe is the decisive measurement.

A threadgroup-local inversion shortcut is not free. With local inversion batch
size `G`, bucket work grows to approximately `(7 + inverse_ops/G) * 46.039M`
M/S equivalents before the nine add/sub operations per pair. Including about
33.4M projective integration equivalents gives:

| `G` | Current 613-op inverse | BLST 460-op inverse |
| ---: | ---: | ---: |
| 128 | 576.2M / 374.0 ms | 521.1M / 338.3 ms |
| 256 | 465.9M / 302.5 ms | 438.4M / 284.6 ms |
| 512 | 410.8M / 266.7 ms | 397.0M / 257.7 ms |

The optimized chain has 379 squares, 81 multiplies, and 16 Fp temporaries.
Generated lower-storage alternatives cost 378S+85M with nine temporaries and
379S+109M with five. Benchmark all three because register pressure can outweigh
operation count. The table still excludes `9N` add/sub operations, routing,
irregular gathers, and synchronization. Do not pursue `G=64`.

## Memory and command design

A balanced tree trades latency for resident scratch. For the 8/8/4 chunk plan,
the first partial layer is approximately 100/173/156 MiB for the
8K/16K/32K classes. Two affine ping-pong layers require approximately
200/347/311 MiB, before compact records and denominator/product scratch. This
invalidates the earlier 40 MiB wave-accumulator estimate. Capacity must be
explicit, checked before encoding, owned by the request arena, and frozen only
after the diagnostic reports peak residency.

Structural affine-tree traffic is at least:

```text
affine SRS leaves read              about 4.52 GB
partial affine nodes written        about 4.42 GB
non-root partial nodes reread       about 4.31 GB
compact records                     about 0.19 GB
base bucket-reduction traffic       at least 13.4 GB
```

Materialized global denominator/prefix scratch adds at least about 8.9 GB. A
literal unfused prepare/reread/finish plus product-tree implementation is closer
to 44.1 GB of logical traffic. A local hierarchical tree can reduce that
estimate toward 33.1 GB, but rebuilding products adds roughly one multiplication
per denominator and projects about 377.0 ms bucket-only / 412.4 ms including
optimistic integration.

These byte counts are layout estimates, not additive elapsed-time bounds:
caches, pass fusion, and compute/traffic overlap change device DRAM traffic.
They do show that memory layout is part of the feasibility test. The probe must
report logical bytes and profiler-observed bandwidth rather than claiming a
rate from nominal Apple bandwidth.

Encode all chunks into one command buffer and wait once. Within each balanced
level:

1. gather signed affine SRS leaves or prior-level partials using the resident
   run descriptors;
2. build ordinary-add and doubling numerators/adjusted denominators using a
   fixed source schedule;
3. replace inactive or zero denominators with Montgomery one;
4. run either the hierarchical global product/inverse/reverse passes or the
   declared local `G` variant;
5. mask-select the canonical affine result and write the next tree layer; and
6. after the roots are complete, use the existing complete mixed/full formulas
   for weighted projective integration and export one compact result per MSM.

There is no intermediate host wait/readback. Empty lanes and public-capacity
tail levels use neutral denominators and masked identity outputs.

## Correctness and security contract

The CPU algorithm is branchy and cannot be transcribed literally. Differential
tests against BLST and the vendored schedule must cover:

- identity/point, equal, inverse, signed point, zero denominator, and
  noncanonical infinity inputs;
- randomized subgroup, rescaled projective-derived, and authoritative on-curve
  non-subgroup points;
- batches of 1, 31/32/33, 63/64/65, and exact `G` boundaries;
- odd-run carry, zero-heavy, concentrated top-window, and exact recorded
  8K/16K/32K shapes;
- contamination of one denominator without allowing zero into the product
  tree; and
- injected failure before/after inversion, with completion retention followed
  by GPU erasure and session quarantine/poison.

Exceptional masks must not change control flow or addresses. Invalid SRS points
are rejected at immutable key import. No mid-proof CPU retry is allowed.

Bucket addresses, run lengths, pairing, and timing remain witness-derived. The
optimized route is eligible only after explicit approval of the trusted-local
profile in `docs/metal-msm-security.md`; deterministic scrambling is not a
confidentiality control. Strict mode still requires fixed/oblivious work and is
not expected to meet the M1 Max 3x target.

## Completed bounded probe and decision gates

The correctness-first implementation completed the requested one-command,
one-wait diagnostic over four deterministic representative 32K chunks. It used
6,794,123 authentic-shape reductions, two warmups, and 20 release samples. The
median/p95 were 221.794/223.972 ms; command encoding was only 0.264 ms and the
device interval was 221.154 ms. The full-workload projection is 1,502.943 ms.

The immediate-stop condition fired. The subsequently authorized
local-inversion/layout falsifier also ran and failed: G128/G256 medians were
2,136.823/2,620.323 ms for the representative sample, versus the 44.261 ms
promotion limit. Every promotion step below is now closed for the M1 Max. The
following list is retained as the predeclared decision contract:

Implement diagnostics only:

1. Retain the completed seed-24302 aggregate capture and its normalized workload
   hash as the replay oracle.
2. Add a wide `AffineOrInfinity`, adjusted-denominator preparation,
   hierarchical batch inversion, and finish pipeline behind the diagnostic
   surface.
3. Replay all 27 resident chunks using the authentic ragged run descriptors,
   deterministic SRS-index permutation/signs, and balanced bucket trees.
4. Compare `G={128,256,512}` and the current versus an independently verified
   optimized inversion chain.
5. Time 20 runs with one command/one wait per sample. Report per-level and
   aggregate median/p95, dispatches, waits, logical bytes, observed bandwidth,
   maximum threadgroup width, allocations, and peak Metal-owned memory.

Apply these gates in order:

- **Correctness stop:** any BLST mismatch, value-dependent exceptional path,
  zero entering the product tree, intermediate readback, fallback, capacity
  truncation, or uncleared failure rejects the design.
- **Immediate performance stop:** reject below about 126.3M weighted reduced
  pairs/s or above 364.6 ms projected bucket reduction. At that point even the
  optimistic 35.4 ms integration allowance exhausts the 400 ms MSM budget with
  zero time for routing.
- **Review-only band:** 126.3–153.5M pairs/s permits one bounded layout/inversion
  review. It does not authorize the record router or proof integration.
- **Promotion gate:** build histogram/scatter only at or above 153.5M pairs/s
  and at or below 300 ms projected bucket reduction, leaving about 100 ms for
  integration, routing, dispatch, and erasure.
- **Final MSM gate:** after routing exists, at least 20 authentic samples must
  show aggregate median at most 400 ms and p95 at most 500 ms, with zero CPU
  fallback/readback and a frozen capacity manifest.
- **Security gate:** no witness-derived production route before the
  trusted-local leakage and failure-erasure contracts have explicit owners.

Both bounded probes failed. Stop work on this Metal MSM design for the M1 Max
and revisit the field/formula engine, performance target, hardware, or backend
scope. Additional command queues, scalar limb variants, root-retirement tuning,
or a backend label cannot close the measured 48.3x final-falsifier promotion
gap.
