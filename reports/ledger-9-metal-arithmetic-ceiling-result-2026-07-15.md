# Ledger 9 Metal batch-affine arithmetic ceiling — 2026-07-15

> Handling: internal engineering review. This result references the sensitive
> aggregate authentic-workload count and requires security-owner review before
> public release.

## Decision

**Reject the hierarchical global affine-scan implementation on Apple M1 Max.**
The deliberately optimistic register-only kernel reached a 20-sample median of
164.206M pair-iterations/s at threadgroup width 64. That is only 6.97% above the
frozen 153.5M/s bucket-arithmetic gate and below the predeclared 200M/s
research gate needed before materializing a hierarchical prefix/suffix scan.

Applied to the authentic 46,039,066 useful reductions, the measured ceiling
projects 280.374 ms for arithmetic alone. It leaves only 19.626 ms of the 300 ms
bucket gate for real denominator/prefix buffers, the root inversions, global
dependencies, barriers, gathers, routing, and output writes. The p95-equivalent
projection is about 285.347 ms. A hierarchical implementation could expose a
different instruction schedule, but it must also materialize every omitted
dependency and would need to improve the fused rate by 21.8% merely to reach
the 200M/s research gate. This result therefore fails the predeclared
implementation criterion; it is not a proof that every possible affine scan
must run at or below this exact rate.

This closes the one layout question not executed by the global materialized
tree or threadgroup-local tiled reducer. It does not change the decision that no
measured M1 Max MSM path supports the requested 3x proof backend.

## What the probe measures

Each pair iteration executes the exact field-operation population of a globally
batched complete affine reduction:

```text
prepare ordinary/double candidates     1 M/S + 5 A/S
surrogate global product/inverse work  3 M/S
finish the selected affine formula     3 M/S + 4 A/S
                                      ----------------
total                                  7 M/S + 9 A/S
```

The kernel uses the same twelve-`u32`, wide-accumulator Montgomery primitive and
the same branchless affine prepare/finish helpers as the BLST-backed reducer.
All three surrogate scan products feed both the next iteration and an observable
output, so they cannot be discarded. The exceptional-case masks remain runtime
values and the source schedule does not skip any formula candidate.

This is intentionally an optimistic fused ceiling probe. Pair iterations stay
in registers. It does not build or traverse a prefix/suffix structure, execute
a field inversion, route digits, integrate buckets, erase buffers, or wire a
proof. Inputs and the two observable outputs touch device memory only at
dispatch boundaries.

The decision thresholds were fixed before measurement:

- below 153.5M/s: reject every affine Montgomery-scan layout;
- 153.5M/s through 200M/s: reject for insufficient dataflow headroom; and
- at least 200M/s: permit one bounded hierarchical-scan falsifier, not a proof
  route.

At exactly 200M/s the arithmetic projection would be 230.195 ms, reserving only
69.805 ms for all omitted work. The measured 164.206M/s does not reach that
research threshold.

## Measured result

The initial legal-width sweep covered every execution-width multiple from 32
through 384. Width 64 won at 165.735M/s in the scouting sample. The decision run
then used two warmups and 20 samples at that width.

| Metric | Result |
| --- | ---: |
| Elements | 262,144 |
| Pair iterations per sample | 16,777,216 |
| Timed dispatches / waits | 4 / 1 |
| GPU median / p95 | **102.172 / 103.984 ms** |
| Wall median / p95 | **102.523 / 104.491 ms** |
| GPU median rate | **164,205,648 pairs/s** |
| Wall median rate | 163,642,804 pairs/s |
| Ratio to 153.5M/s | **1.0697x** |
| Ratio to 200M/s research gate | **0.8210x** |
| Full arithmetic projection | **280.374 ms** |
| Decision | **Reject: insufficient headroom** |

Raw GPU samples in milliseconds:

```text
102.432 102.011 102.116 101.812 102.581
100.950 102.171 102.844 102.173 102.612
101.741 102.100 101.755 101.685 103.135
104.170 103.980 103.582 103.984 101.516
```

Every sample used one command buffer and one completion wait, with zero timed
Metal-buffer allocation, upload, blit, or execution failure. A 144-byte
point/scan observation occurs after timing. Repeated execution is bitwise
deterministic, and the separately versioned Rust/MSL parameter ABI is
device-probed.

## Alternative-algorithm closure

The follow-up bound also reviewed changes that alter representation or point
count rather than just scan layout:

- BLS12-381 G1 GLV reduces the authentic window-record model by only about
  8.5% after doubling the scalar streams. The endomorphism and decomposition
  context are described by [Bowe](https://eprint.iacr.org/2019/814.pdf) and the
  original [GLV paper](https://www.iacr.org/archive/crypto2001/21390189.pdf).
- Published fixed-base BGMW/LFG counts save at most about 26% from ordinary
  Pippenger at the relevant size, and only a few percent from a pre-shifted
  variant; see [Luo, Fu, and Gong](https://d-nb.info/1286957532/34).
- XYZZ, extended-Jacobian, and extended-Hessian mixed additions have
  multiplication-only floors at or above approximately 299 ms with the
  measured field multiplier, before additions and integration. The operation
  counts come from the [XYZZ Explicit-Formulas
  Database](https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html),
  [Moody and Wu](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=909543),
  and the [extended-Hessian Explicit-Formulas
  Database](https://www.hyperelliptic.org/EFD/g1p/auto-hessian-extended.html).
- The reviewed Comba, Karatsuba, specialized-squaring, lazy-Montgomery, RNS,
  and PMNS directions did not produce a separately bounded multi-x field
  engine capable of closing the measured dataflow gap.
- Larger fixed-base subset tables require tens of gigabytes per SRS before they
  approach the existing point-addition population.

These remain useful on faster hardware or after a new field engine independently
clears its gate. None authorizes another M1 Max implementation branch under the
current target. The operation-count derivations and rejected global dataflow
measurement are recorded in
`ledger-9-metal-batch-affine-result-2026-07-15.md`.

## Reproduction

```bash
MIDNIGHT_METAL_AFFINE_CEILING_THREADGROUP_WIDTH=64 \
MIDNIGHT_METAL_AFFINE_CEILING_SAMPLES=20 \
CARGO_INCREMENTAL=0 cargo test --release \
  -p midnight-metal-prover --lib \
  metal::batch_affine_ceiling::tests::report_batch_affine_arithmetic_ceiling \
  -- --ignored --exact --nocapture --test-threads=1
```

A reported rejection is a successful decision measurement, not a Rust test
failure.
