# Ledger 9 Metal Karatsuba/L4 candidate result — 2026-07-15

> Handling: internal sensitive engineering report. The exact aggregate counts
> below are derived from a witness-dependent diagnostic capture. Aggregation
> does not declassify them. Do not copy these counts, projections, or workload
> gates into the standalone crate or a public release without explicit
> security-owner declassification.

## Decision

The bounded separated twelve-limb subtractive-Karatsuba REDC falsifier with
lazy `[0, 4p)` residues has stopped at Milestone 1. Its portable exact-integer
model and generated Metal correctness corpus pass, but the authoritative
flattened-helper source-liveness analysis peaks at **70 `u32`-equivalent live
words**, exceeding the predeclared limit of **56**. The previously reported
**48** words cover only the top-level aggregate subset; they omit helper
temporaries and inputs and are not an acceptance metric.

For the authentic complete-formula workload, the exact model uses
**87,890,218,020** base `u32` products, **38.65% fewer** than the
**143,262,326,592** terms in the prior direct-XYZZ/named-FIOS model. It also
models **70.56% fewer** multiplicative final-normalization steps and **20.28%
less** logical point traffic.

Those theoretical workload reductions are unrealized and are not a speedup
result. The field falsifier failed its liveness gate before performance, so no
throughput scout or formal run was taken and no complete-formula kernel, bucket
tree, full MSM, or proof-residency work proceeded on this representation. The
larger live state, carry dependencies, and 1.032 billion L4 add/sub operations
remain precisely the risks that the stop rule was intended to catch.

The reusable arithmetic definition is
[`bls12-381-metal-msm/KARATSUBA_L4_CANDIDATE.md`](../bls12-381-metal-msm/KARATSUBA_L4_CANDIDATE.md).
That document intentionally contains no Ledger workload, trace, security, or
hardware-comparison material.

This is a Metal-native candidate. The existing CUDA backend is an arithmetic
and correctness reference only; there is no CUDA+Metal execution design. CPU
code may own setup, command submission, and out-of-band diagnostics, but the
timed field, formula, reduction-tree, and MSM work must stay on Metal with no
CPU arithmetic fallback.

## Security and formula boundary

The aggregate source is the controlled capture described in
[`ledger-9-metal-msm-authentic-runs-2026-07-15.md`](ledger-9-metal-msm-authentic-runs-2026-07-15.md).
Its record, run, and workload totals remain witness-derived under
[`docs/metal-msm-security.md`](../docs/metal-msm-security.md). Raw captures and
replay manifests retain their existing access controls; this report is not a
substitute for a declassification review.

The current security contract requires fixed-schedule, branchless complete RCB
formulas for both the strict local-oblivious and leakage-accepted trusted-local
profiles. Consequently, the promotable arithmetic path in this report uses:

- complete homogeneous + affine addition: `5M + 3D2 + 18A`;
- complete homogeneous + homogeneous addition: `6M + 3D2 + 24A`; and
- complete homogeneous doubling: `4M + 2S + D2 + 12A`.

`M`, `S`, and signed fused `D2` are the L4 operations defined by the standalone
note. The diagnostic incomplete-projective schedule later in this report is a
performance ceiling only. It cannot become a proof route without a separate
cryptography and security-contract revision.

This arithmetic decision does not approve a leakage profile. Secret-dependent
Pippenger routing, compaction, addresses, or dispatch sizes still require the
trusted-local profile decision described by the security contract. Both
profiles prohibit mid-proof fallback, and both require GPU-side scratch erasure
before reuse.

## Authentic operation population

The balanced reduction topology for the selected 147 MSMs contains:

| Stage | Complete group operations |
| --- | ---: |
| First-level signed affine-pair reductions | 23,266,406 |
| Higher-level homogeneous reductions | 22,772,660 |
| Integration and window-collection full additions | 2,763,679 |
| Doublings | 36,828 |
| **Total** | **48,839,573** |

The higher-level and integration rows combine to **25,536,339** complete
homogeneous additions. Applying the required formulas to the exact mix gives:

| L4 operation | Derivation | Count |
| --- | --- | ---: |
| `M` | `5*23,266,406 + 6*25,536,339 + 4*36,828` | **269,697,376** |
| `S` | `2*36,828` | **73,656** |
| `D2` | `3*(23,266,406 + 25,536,339) + 36,828` | **146,445,063** |
| `A` | `18*23,266,406 + 24*25,536,339 + 12*36,828` | **1,032,109,380** |

With the standalone exact counts `M=189`, `S=180`, and `D2=252` base products:

```text
269,697,376*189 + 73,656*180 + 146,445,063*252
    = 87,890,218,020 base u32 products
```

Ordinary L4 multiplication and squaring return below `3p` and need no final
range subtraction. Only `D2` can return at or above `4p`, so this population
needs **146,445,063** multiplicative final-normalization steps. The prior
direct-XYZZ population needs **497,438,634**, a reduction of **350,993,571
steps**, or **70.56% fewer**.

The prior named-FIOS workload model assigns 288 base terms to each of its
497,438,634 multiplication/square operations:

```text
497,438,634*288 = 143,262,326,592 base u32 products
```

The complete RCB/L4 model is therefore **38.65% lower** in base-product count.
Its L4 add/sub population is approximately three times the earlier linear-field
operation population, so the base-product ratio must not be reported as an
expected wall-time ratio.

At most **47,093,103** selected signed inputs require a point-negation mask.
That count does not include digit routing, histogram/scan/scatter work,
descriptor handling, identity padding, canary checks, or erasure.

## Diagnostic-only incomplete ceiling

For diagnosis only, an incomplete projective schedule would use:

```text
first affine reduction: 3M + 2S + D2 + 6A
full reduction:        10M + 2S + D2 + 6A
doubling:               4M + 2S + D2 + 12A
```

Across the authentic population this is:

| L4 operation | Count |
| --- | ---: |
| `M` | 325,309,920 |
| `S` | 97,679,146 |
| `D2` | 48,839,573 |
| `A` | 293,258,406 |

Its product model is:

```text
325,309,920*189 + 97,679,146*180 + 48,839,573*252
    = 91,373,393,556 base u32 products
```

This uses more base products than complete RCB/L4 but far fewer additions. A
large result gap between the two diagnostics would identify addition pressure;
it would not authorize incomplete formulas.

## Homogeneous tree and traffic model

The high-value layout is a balanced compact tree, not a literal port of the
existing GPU topology:

1. The first tree level consumes signed affine SRS leaves and writes complete
   homogeneous points.
2. Every higher level reads and writes homogeneous points; an odd survivor is
   copied to the next compact level without an identity addition.
3. Dense ping-pong level buffers avoid sparse fixed-capacity tree storage.
4. Bucket integration and window collection retain the same homogeneous
   representation.
5. Only the 147 compact final points cross the GPU boundary.

With a 96-byte affine point and a 144-byte homogeneous point, the logical
read-plus-write traffic is:

| Stage | Bytes per operation | Logical bytes |
| --- | ---: | ---: |
| First affine-pair level | 336 | 7,817,512,416 |
| Higher homogeneous levels | 432 | 9,837,789,120 |
| Integration/full additions | 432 | 1,193,909,328 |
| Doublings | 288 | 10,606,464 |
| **Total** |  | **18,859,817,328** |

The equivalent prior XYZZ traffic model is **23,657,373,120 bytes**. The
homogeneous plan saves **4,797,555,792 bytes**, or **20.28%**. These are logical
layout bytes, not observed DRAM traffic or an elapsed-time lower bound; cache
reuse, pass fusion, atomics, and command ordering determine physical traffic.

This tree is only the group-arithmetic core. A useful MSM backend still needs a
Metal-native recoding, histogram/scan/scatter or alternative routing layer,
bucket integration, window collection, resident SRS/request ownership, and
failure-safe erasure. Recreating the CUDA library as a compatibility-shaped
Metal clone is not the goal; the reusable field/MSM crate should expose a
backend-neutral resident-buffer API while the Metal implementation owns a
Metal-appropriate schedule.

Functionally, that backend replaces the MSM roles currently supplied by
sppark—recode, route, reduce buckets, integrate, and collect—but it should not
recreate sppark's CUDA execution topology or pointer-shaped API. The standalone
crate owns the portable contracts and Metal implementation; a thin application
adapter owns proof-system batching and lifecycle policy.

## Ordered falsification result

The bounded sequence stopped on its first miss.

### 1. Standalone field gate

**NO_GO before performance.** The portable exact model and generated Metal
correctness corpus pass. The authoritative flattened helper reaches 70
source-live `u32` equivalents, above the limit of 56. The 48-word top-level
aggregate subset is not the full-kernel liveness value. Because the source
liveness condition failed, the issue-rate scout, 20-sample throughput gate,
and physical-spill acceptance run were not performed.

### 2. Exact complete-formula gate

**Not run.** This successor gate is retained only as the acceptance contract
for a materially different field implementation that first clears Milestone 1.
It would run the exact authentic mix of 23,266,406 mixed additions, 25,536,339
full additions, and 36,828 doublings as dependent complete formulas, with no
routing or point-buffer traffic in the timestamp.

```text
median rate >= 166.7 million complete group operations/s
p5 rate     >= 150.0 million complete group operations/s
```

For 48,839,573 operations, those thresholds correspond to **292.979 ms** and
**325.597 ms** respectively. A pass permits the tree falsifier; it is not an
MSM result. The diagnostic incomplete schedule may run only after the complete
result and only to attribute a miss.

### 3. Homogeneous tree gate

**Not run.** A future candidate that clears the two earlier gates would
materialize the balanced affine-to-homogeneous tree and integration arithmetic
with authentic level sizes and compact ping-pong buffers.

```text
GPU median <= 300 ms
GPU p95    <= 375 ms
```

The timestamp includes all point reads/writes, odd-node propagation, level
barriers, integration, and window collection. It excludes scalar routing only.

### 4. Full 147-call MSM gate

**Not run.** A future candidate that clears the earlier gates would add
Metal-native scalar conversion, signed recoding, routing, bucket formation,
tree reduction, integration, collection, and GPU-side erasure.

```text
GPU median <= 400 ms
GPU p95    <= 500 ms
```

Correctness must include exact call boundaries `8191/8192`, `16383/16384`, and
`32767/32768`, plus empty, singleton, ragged-tail, concentrated-bucket, and
zero-heavy cases. Compare all 147 compact outputs with BLST. No intermediate
host readback, CPU arithmetic fallback, or mid-proof backend switch is allowed.
Failure and cancellation must retain allocations through real command
completion, erase secret buffers, and poison the session if safe erasure cannot
be enqueued.

## Scout context, not evidence

Two recent local issue-rate scouts explain why this candidate is prioritized,
but they were ephemeral and are not frozen decision artifacts:

- SIMD 8x8 matrix paths observed roughly 4.17T raw FP32 products/s median and
  roughly 4.48T half-by-half-to-float products/s, with low samples around
  3.1--3.6T. They missed the necessary 4.6584T/4.36725T raw gates before exact
  packing, carries, or Montgomery reduction.
- The formal observable `uint4` schedule measured 1.900900T median and
  1.822508T p5 raw `u32` products/s. Dividing by the strength-reduced
  1,800-product reference count gives 1.056055B/1.012504B schedule equivalents
  before carry and reduction.

These observations reject those concrete schedules; they do not establish
device-wide issue limits and must not be used as benchmark claims without a
reproducible evidence record. Neither observation is a Karatsuba/L4 performance
run; that candidate stopped at source liveness before any throughput scout.

## Interpretation boundary

The exact arithmetic model identifies the largest remaining modeled wins:
38.65% fewer base products, 70.56% fewer multiplicative final normalizations,
and 20.28% less logical point traffic. None includes carry latency, the larger
Karatsuba live set, the 1.032B L4 additions/subtractions, compiler scheduling,
register allocation, barriers, routing, atomics, cache behavior, command
overhead, or erasure.

This candidate did not produce a field-gate pass: exact arithmetic correctness
alone does not overcome the 70-versus-56 liveness miss. The complete-formula,
tree, full-workload, and residency gates were therefore not run, and none of
the modeled reductions should be presented as realized M1 Max performance.
