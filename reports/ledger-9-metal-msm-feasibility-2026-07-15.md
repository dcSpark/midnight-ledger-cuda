# Ledger 9 Metal MSM feasibility — 2026-07-15

## Conclusion

> Outcome update: both subsequently reviewed lower-operation candidates failed.
> The global product-tree reducer projected 1,502.943 ms for bucket reduction;
> the final recursive flat-segmented candidate projected 14,479.771 ms. See
> `ledger-9-metal-batch-affine-result-2026-07-15.md` and
> `ledger-9-metal-flat-segmented-result-2026-07-15.md`. The standalone
> radix-256 FP32 field candidate then exceeded the optimistic product roofline;
> its carry-normalized radix-4096 successor measured only 1.002B median and
> 0.941B p5 raw product-count equivalents/s versus 2.40B/2.25B gates. That
> schedule is closed, but the result is not a global FP32 upper bound. A later
> complete two-lane `u32` FIPS Montgomery kernel passed raw-limb BLST
> differentials but reached only 0.987B median / 0.940B p5 complete Fp/s versus
> 2.0B/1.8B gates. That mapping is closed as well. The generated
> named-register scalar FIOS schedule removed private arrays, inner loops,
> shuffles, and threadgroup scratch, yet reached only 1.529B median / 1.455B p5;
> a stronger sweep confirmed 1.535B median. The selected M1 Max field mappings
> are closed. A later 30×13 native-`u32` product schedule also missed its
> necessary-only gate at 1.901T/1.823T raw products/s; this is a schedule result,
> not a published device-wide `u32` peak. The materially different separated
> 12-limb subtractive-Karatsuba REDC candidate with lazy `[0,4p)` residues also
> stops at Milestone 1. Its portable exact model and generated Metal correctness
> corpus pass, but the authoritative flattened helper reaches 70 source-live
> `u32` equivalents versus the limit of 56. The earlier 48-word value covers
> only top-level aggregates. The field gate is therefore `NO_GO` before a
> performance run; no formula, tree, MSM, or residency work proceeded.

A Metal-native histogram/scan/scatter Pippenger is the appropriate
leakage-accepted MSM architecture for the existing fixed KZG bases. It is not,
however, honest to claim that the current six-`u64` Fp implementation can meet
the complete transaction's `1,757.940 ms` Metal gate.

The persisted seed-24302 proof trace contains 147 MSMs over 3,235,835 logical
scalar positions, of which 1,988,928 are nonzero. The initial Metal window
profiles produce about 54.215 million nonzero signed-digit records under the
original full-window approximation. A later exact uniform-Fr model including
truncated top windows gives 54.153M expected records; the persisted timers do
not contain scalar limbs, so neither value is an observed digit trace. The
later aggregate-only capture in
`ledger-9-metal-msm-authentic-runs-2026-07-15.md` supersedes both estimates with
47.093M records and 46.039M balanced reductions. Before
histograms, scans, memory traffic, synchronization, NTTs, or any other PLONK
stage, complete group addition requires an optimistic 1.17--1.24 seconds at
the measured 554.243 million isolated Fp multiplications/s rate.

The fused complete G1 addition gate has now rejected the current
six-`u64`-limb, one-point-per-thread representation. After replacing
value-dependent field reductions with fixed subtract/select schedules, two
release runs reached only 9.80--10.17 million complete additions/s, equivalent
to 117.57--122.03 million scheduled Fp multiplications/s at 12M per addition.
That is about 14x below the 1.7 billion/s minimum for a 400 ms aggregate MSM
budget. Even the 57.1 million-add arithmetic workload alone projects to at
least 5.6 seconds at the best observed chain rate.

The first twelve-`u32` replacement has also failed its early gate: the
repository's correctness-checked explicit-`mulhi` path peaks at 1.186 billion
raw Montgomery multiplications/s, only 59.28% of the 2.0-billion/s target. The
BLST-checked repository wide-accumulator sibling improves this to 1.540 billion
raw/s, 77.02% of the target, and is still `NO_GO`. A separate synthetic
Algorithm 7 scouting schedule reaches only 77.98 million additions/s or 935.77
million effective multiplications/s, 55.0% of the 1.7-billion/s minimum. An
issue-count bound rejects SIMD-cooperative limbs as the primary rescue path.

Explicit two-/three-state CIOS interleaving subsequently regressed total raw
throughput to 0.963B/s and 0.467B/s, so that interleaving branch is closed. The later
global-tree and recursive flat-segmented lower-operation designs also failed
their predeclared gates. The histogram/scan/scatter pipeline therefore does not
proceed. More command scheduling around the measured arithmetic cannot remove
this dependency-chain floor.

This is a feasibility gate, not a proof-route result. Metal proof execution
remains unavailable.

## Trace workload

The source is the same persisted diagnostic used by
`ledger-9-metal-amdahl-profile-2026-07-15.md`:

```text
/tmp/metal-plan-timers-seed24302.log
SHA-256 0d82c87f34be074a69438a64189960343a3c904a35d8f12025cc9e0727ae079c
```

Filtering MSM leaves to the five enclosing `create_proof` intervals gives:

| Class | Calls | Logical positions | Nonzero scalars | Density | CPU naive sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 23 | 188,415 | 117,871 | 62.56% | 328.007 ms |
| 16K | 62 | 1,015,806 | 625,897 | 61.62% | 1,805.751 ms |
| 32K | 62 | 2,031,614 | 1,245,160 | 61.29% | 4,115.171 ms |
| **Total** | **147** | **3,235,835** | **1,988,928** | **61.47%** | **6,248.929 ms** |

The exact logical shapes are 22×8192 plus 1×8191, 60×16384 plus
2×16383, and 60×32768 plus 2×32767. Median nonzero lengths are 4,626,
8,393, and 16,649; every class has a full-length p95.

## Initial M1 Max profiles

The first profiles target approximately 40 observed points per signed bucket,
or 64 points for a dense MSM:

| Logical class | Window bits | Windows | Signed buckets | Observed average occupancy |
| --- | ---: | ---: | ---: | ---: |
| 8K | 8 | 32 | 128 | 40.0 |
| 16K | 9 | 29 | 256 | 39.4 |
| 32K | 10 | 26 | 512 | 39.2 |

Metal converts each Fr value from Montgomery form once before recoding.
Signed digits are extracted low-to-high with a fixed public window count and
must recompose the canonical scalar exactly. Zero digits produce no record.
The trusted-local profile permits the resulting digit-indexed atomics and
scatter addresses; this behavior is forbidden by the strict profile.

The selected profiles generate approximately 3.757 million, 18.116 million,
and 32.343 million nonzero digit records respectively, 54.215 million total.
Alternative windows are benchmark candidates only; the CUDA window schedule is
not assumed optimal for Apple GPU execution.

## Six-pass pipeline

Each same-profile descriptor chunk records the following ordered compute
passes into one transcript-epoch command buffer:

1. Convert Fr values to canonical form and build per-window magnitude
   histograms.
2. Exclusively scan each histogram into bucket offsets and reset its storage
   for use as scatter cursors.
3. Recode scalars and atomically scatter signed point-index records.
4. Accumulate each bucket with complete homogeneous G1 formulas. Short runs
   use one lane; a benchmarked long-run threshold selects cooperative SIMD-32
   reduction.
5. Integrate descending buckets into one weighted point per window using
   parallel segment summaries.
6. Collect windows high-to-low, convert the final homogeneous point to the
   public Jacobian ABI without inversion, and export one point per MSM.

The chunk has six compute dispatches, two GPU clears, no internal host wait,
and no internal readback. All chunks for a protocol epoch complete before its
single transcript wait. The 147 final projective results total only 21,168
bytes.

Every internal accumulator is initialized to homogeneous identity
`(0, R, 0)`. A byte-zero buffer is not a valid identity-filled bucket arena.
Zero fill is used for erasure only; an identity-fill kernel prepares bucket,
row, partial, and accumulator storage before group arithmetic.

## Resident ownership

`MetalKeyContext` owns validated coefficient and Lagrange SRS buffers plus
public profile plans. `MetalProofArena` owns scalar sources and every secret
scratch allocation. Descriptors contain a generation-checked resident Fr
slice, SRS kind, point offset, public logical length, and output index.

The initial fixed chunk capacities are eight 8K MSMs, eight 16K MSMs, or four
32K MSMs. Estimated scratch is approximately 14.8 MiB, 27.1 MiB, and 24.7 MiB
respectively. Production scratch and SRS live in private Metal buffers;
shared resources are limited to compact staging and result egress.

The production contract clears all canonical scalars, records, histograms,
offsets, buckets, row points, and proof polynomials on the GPU after the final
referencing command and before reuse. Failure and cancellation must quarantine
resources through real command completion; inability to enqueue the clear must
poison the session. Public SRS/key buffers do not require erasure. This lifecycle
is designed but not implemented by the current diagnostic session.

## Arithmetic and fused-chain gate

RCB Algorithm 7 costs approximately 12 Fp multiplications per nonzero digit
record. Records alone therefore require roughly 650.6 million Fp
multiplications:

```text
650.6M / 554.243M per second = 1.174 seconds
```

Approximate bucket integration raises the workload to about 57.1 million
complete additions and the multiplication floor to roughly 1.236 seconds.
This is a throughput lower bound: it assumes independent primitive saturation
and ignores dependencies, register pressure, atomics, scans, traffic,
dispatches, NTTs, the remainder of PLONK, transcript work, and verification.

For a 400 ms aggregate MSM budget:

```text
57.1M additions × 12 Fp multiplications / 0.400 s
    = 1.713 billion effective Fp multiplications/s
```

The implementation should target at least 2.0 billion/s to leave margin. The
fixed-schedule fused diagnostic used independent in-register homogeneous chains,
so it excludes histogram, scan, scatter, bucket traffic, conversion, readback,
and inversion from the timed region:

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  --test metal_g1 report_g1_complete_add_chain_throughput \
  -- --ignored --nocapture
```

| Elements | Adds/chain | Timed dispatches | Timed command/waits | Complete adds/s | Effective Fp M/s | Timed ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 64 | 4 | 1 / 1 | 9,797,487 | 117,569,839 | 214.050 |
| 8,192 | 64 | 8 | 1 / 1 | 10,169,312 | 122,031,746 | 412.447 |

The setup and egress commands convert public Jacobian inputs to the internal
homogeneous form once and convert the result back once; they are deliberately
outside the timed gate. At the best measured chain rate:

```text
54.215M record additions / 10.169M per second = 5.33 seconds
57.1M total additions    / 10.169M per second = 5.62 seconds
required 142.75M additions/s / measured 10.169M = 14.0x gap
```

This is a conservative rejection, not a production forecast: real bucket runs
add routing, memory, and reduction overhead and cannot improve the serialized
field dependency chains measured here. The result stops the Pippenger slice
before its full scratch and routing implementation.

## Replacement-representation gate

The repository's twelve-`u32` Fp diagnostic preserves the public six-`u64` ABI,
uses the same `R = 2^384` Montgomery representation internally, and passes BLST
boundary, randomized, and repeated-chain differential tests. Its best raw result
is 1,185,515,361 multiplications/s at threadgroup width 64. That is a clean
`NO_GO` against the 2.0-billion/s pre-G1 threshold, so no Fp32 G1 port was
started.

The BLST-checked repository wide-accumulator sibling replaces explicit
`mulhi(uint)` carry handling with an exact `ulong` accumulator. It reaches
1.540404 billion raw multiplications/s at width 64, about 29.94% faster than the
explicit path but still 1.298x short of the raw gate. A separate temporary
synthetic Algorithm 7 schedule reaches 77.980 million additions/s (935.766
million effective scheduled multiplications/s). Those synthetic coordinates
establish a schedule ceiling, not G1 correctness, and still leave a 1.82x gap
to the fused gate. Four- and sixteen-lane cooperative arithmetic have
unfavorable ideal issue-count ceilings before shuffle/carry overhead, so they
are not the next full implementation.

Explicitly alternating independent CIOS states inside every limb step does not
rescue the wide path. The two-state kernel peaks at 963,456,555 total
multiplications/s at width 128; the three-state kernel peaks at 467,233,045/s at
width 64. Both pipelines expose a maximum threadgroup width of 384 rather than
the single-state path's 512, consistent with register pressure, and both are
`NO_GO`.

The independently extractable package later implemented the complete two-lane
`u32` FIPS mapping. It passed raw-limb BLST, final-subtraction, chain, ABI, and
SIMD-padding checks, but twenty selected-width M1 Max samples reached only
0.986960B median and 0.939896B nearest-rank p5 complete Fp/s versus 2.0B/1.8B
gates. The direct-XYZZ multiply/square workload alone projects to 504.0 ms
median and 529.2 ms p5 at those rates, before 341.9M field additions, routing,
traffic, integration, or erasure. The two-lane mapping must not proceed to G1.

The named-register field falsifier generated a fully unrolled,
named-register scalar FIOS schedule because the earlier 1.540B/s wide scalar
AIR retained addressable state and inner loops. Both loop-carried orientations
passed raw BLST and all width/padding checks. The formal 20-sample result was
1.529069B median and 1.455061B nearest-rank p5 complete Fp/s, and a stronger
orientation/width sweep confirmed a 1.534806B median. Removing the suspected
compiler structure did not produce the required 1.298x gain.
The standalone decision contract and raw timings are in
`bls12-381-metal-msm/NAMED_REGISTER_FIOS_GATE.md` and
`bls12-381-metal-msm/evidence/m1-max-named-register-fios-field-2026-07-15.json`.

The follow-up radix-`2^13`, 30-limb schoolbook assessment used a
strength-reduced 1,800-product reference schedule per complete multiply. Its
observable `uint4` issue
schedule reached 1.900900T median and 1.822508T p5 raw products/s, equal to only
1.056055B/1.012504B reference-schedule equivalents/s. That tested
schedule is stopped. Apple publishes no native-`u32` issue peak, so the result
does not close every possible 30-limb mapping.

The final bounded candidate changed both the multiplier and the group
representation. A portable exact-integer model validates `K12 = 63`,
three-product separated
REDC at 189 base products, `-p^-1 mod 2^384`, and closure of lazy `[0,4p)`
residues over boundary plus 2,048 deterministic randomized canonical/L4 pairs.
Under complete homogeneous RCB formulas with fused two-product reductions, the
authentic aggregate has 87.890B base products, 38.65% fewer than the 143.262B
named-FIOS direct-XYZZ count, and 20.28% less logical point-value traffic. The
counter-risk is 1.032B fixed linear operations. The authoritative flattened
helper reaches 70 source-live `u32` equivalents, exceeding the predeclared
56-word limit. The 48-word count is only a top-level aggregate subset and is
not a full-kernel liveness result. The structural product and traffic reductions
are therefore unrealized, not a speedup forecast. The package-neutral
arithmetic contract is in
`bls12-381-metal-msm/KARATSUBA_L4_CANDIDATE.md`; internal workload projections
and stop/go gates are in
`reports/ledger-9-metal-karatsuba-l4-candidate-2026-07-15.md`.

Direct XYZZ counting leaves only a paper arithmetic margin: its 497.439M M/S
operations cost 325.3/341.9 ms at the formal median/p5. Granting every 341.877M
add and 47.093M sign operation an optimistic 3.0B/s raises that to 455.0/471.5
ms before at least 23.7 GB of naive tree traffic, routing, exceptional cases,
dispatches, integration, output, or erasure. This is not a 400/500 ms benchmark
pass, so G1 and the six-pass router do not proceed on this mapping.

A CPU+Metal split does not create hidden median headroom. Combining the
optimistic 455.0 ms Metal arithmetic projection with the measured 6,248.929 ms
CPU MSM sum under perfect linear partitioning gives a harmonic lower bound of
about 424.1 ms. The analogous p5 projection is 438.4 ms. Both omit coordination,
CPU/GPU work partitioning, extra representations, Metal traffic, routing, and
erasure; the median already exceeds 400 ms. This rules out the current
CPU+Metal MSM split as a rescue while preserving CPU+Metal as an acceptable
architecture for a future faster native engine. There is still no CUDA+Metal
route.

## Security gate

Histogram destinations, scatter addresses, record occupancy, long-run routing,
and execution timing depend on secret-derived digits. The design is eligible
only under the leakage-accepted trusted-local profile in
`docs/metal-msm-security.md`. Deterministic scalar scrambling may be benchmarked
as an optional load-balancing plan, but is disabled initially because it makes
the observed sparse workload nearly dense and is not a confidentiality control.

Before production routing, the decision record must explicitly cover trusted
host/OS/driver/local users/profilers, remote timing, treatment of existing CUDA
leakage, and cryptography/security-owner approval. There is no CPU fallback.

## Acceptance order

1. The generated Karatsuba/L4 field falsifier passed exact portable and Metal
   correctness but failed the 56-word source-liveness condition at 70 words.
   It is `NO_GO` before the 2.0B/1.8B performance and physical-spill runs.
2. Canonical recoding, routing, bucket accumulation, tree reduction, full MSM,
   resident integration, and end-to-end acceptance were not implemented or run
   for this candidate.
3. A future materially different field implementation must restart at the
   complete Milestone 1 gate. Only a pass may reopen the retained downstream
   correctness, 400/500 ms MSM, lifecycle, and proof-parity contracts.
