# Ledger 9 Metal primitive diagnostics — 2026-07-15

This report records early correctness and throughput diagnostics on an Apple
M1 Max. These are primitive engineering measurements, not proof-backend or 3x
acceptance results. All probes include one command submission and wait unless
the command deliberately repeats a resident dispatch for throughput.

## Correctness status

- BLS12-381 Fr: reduced Montgomery add, subtract, multiply, and square match
  BLST over boundary cases and 4,096 deterministic random pairs.
- BLS12-381 Fp: the same four operations match BLST over 4,352 total pairs.
- BLS12-381 G1: complete homogeneous full addition, mixed affine addition, and
  doubling match BLST over exceptional boundaries, 512 deterministic
  scalar-derived/rescaled-Jacobian cases, and authoritative on-curve
  non-subgroup cases.
- The host/shader ABI probe validates Fr, Fp, public affine/projective G1, and
  internal homogeneous G1 sizes, alignments, and every member offset. It also
  checks a pinned layout fingerprint (`0x108cb9731d5097c4`) and the generated
  constants fingerprint whenever a session starts.
- The same public API compiles as explicitly unavailable on non-Apple targets.

RCB Algorithms 7, 8, and 9 and the underlying field add/subtract/reduce paths
now use fixed source-level schedules and mask selection rather than
value-dependent control flow. They remain correctness and performance probes:
the optimized metallib has not passed a GPU side-channel review, and no MSM
digit/bucket strategy has been approved. Witness-derived MSM and proof routing
remain prohibited.

## Field throughput

Command:

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  report_field64_throughput -- --ignored --nocapture
```

The probe used 262,144 elements and 32 resident dispatch rounds.

| Field | Operation | Operations/second |
| --- | --- | ---: |
| Fr | add | 2,302,243,317 |
| Fr | subtract | 1,836,737,116 |
| Fr | Montgomery multiply | 536,090,364 |
| Fr | square | 2,053,012,237 |
| Fp | add | 3,136,611,830 |
| Fp | subtract | 3,004,515,759 |
| Fp | Montgomery multiply | 554,243,108 |
| Fp | square | 714,615,894 |

These are post-rewrite rates for fixed subtract/select reduction. They support
the four-`u64` Fr representation in the resident NTT diagnostic, but isolated
independent Fp operations do not predict a dependent G1 formula. The fused
complete-add result below rejects the six-`u64`, one-point-per-thread Fp design
for production MSM.

## Isolated 12x32-bit Fp gate

The repository also contains an isolated twelve-`u32` CIOS experiment. It
retains the public six-`u64` ABI and exact `R = 2^384` Montgomery layout, with
split/rejoin kernels outside the timed command. Boundary cross-products, 4,096
deterministic randomized pairs, and a repeated internal multiplication chain
match BLST for add, subtract, multiply, and square.

Command:

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  report_fp32_raw_mul_throughput -- --ignored --nocapture
```

The probe used 262,144 independent lanes, 64 dependent in-register
multiplications per lane, and four timed dispatches in one command buffer with
one wait. Conversion and egress ran in separate commands. A threadgroup-width
sweep measured:

| Width | Raw Montgomery multiplications/second |
| ---: | ---: |
| 32 | 1,011,967,934 |
| 64 | **1,185,515,361** |
| 128 | 1,168,151,003 |
| 256 | 1,178,367,510 |
| 448 | 1,110,681,601 |

The best result is 59.28% of the 2.0-billion/s optimistic pre-G1 gate. This
representation is therefore `NO_GO`; it was not extended into G1 or the proof
pipeline.

### Wide-accumulator repository sibling

The second repository path retains the twelve limbs, public ABI, fixed
reduction, and add/subtract implementation but computes each inner CIOS step as
`ulong(a) * ulong(b) + ulong(t) + carry`. Its BLST boundary, 4,096 randomized,
square, and repeated-chain differential tests pass.

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  report_fp32_wide_raw_mul_throughput -- --ignored --nocapture
```

| Width | Raw wide-accumulator multiplications/second | Timed ms |
| ---: | ---: | ---: |
| 32 | 1,284,316,682 | 52.253 |
| 64 | **1,540,404,194** | **43.566** |
| 128 | 1,528,397,194 | 43.908 |
| 256 | 1,514,966,821 | 44.297 |
| 512 | 1,472,517,134 | 45.574 |

The workload and timing boundary are identical to the explicit-`mulhi` probe:
262,144 lanes, 64 iterations, four timed dispatches, one command, and one wait;
split/rejoin remains outside the timed region. The best result is about 29.94%
faster than the explicit path but only 77.02% of the 2.0B gate, still 1.298x
short. It is also `NO_GO`; no wide-accumulator G1 or proof port was attempted.

### Explicit CIOS-state interleaving

The final scalar experiment alternates every multiply/accumulate limb across
two or three independent wide CIOS states inside one Metal thread. This tests
integer instruction-level parallelism explicitly rather than relying on the
compiler to schedule across complete inlined multiplications. Every state is
combined into observable output. A 128-pair deterministic test checks seven
repeated multiplications per state plus the final field combinations against
BLST.

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  report_fp32_wide_interleaved_2_raw_mul_throughput -- --ignored --nocapture
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  report_fp32_wide_interleaved_3_raw_mul_throughput -- --ignored --nocapture
```

Both probes use the same 262,144 lanes, 64 iterations per state, four timed
dispatches, one command, and one wait. Rates count every independent field
multiplication across all states.

| States | Best width | Total multiplications/s | Timed ms | 2.0B ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 128 | **963,456,555** | 139.309 | 48.17% |
| 3 | 64 | **467,233,045** | 430.891 | 23.36% |

Two-state interleaving is 37.45% slower than the 1.540B/s single-state path;
three-state interleaving is 69.67% slower. Both interleaved pipelines report a
maximum threadgroup width of 384 rather than 512 for the single-state path,
consistent with increased register pressure. Both are `NO_GO`, closing the
single-thread scalar CIOS branch.

### Exploratory limb and fused-schedule scouting

A separate `/tmp` research probe tested the cheapest remaining scalar variants
before committing more repository code. It is decision evidence, not an
authoritative backend implementation: field outputs were compared byte-for-byte
across implementations on deterministic reduced inputs, rather than directly
against BLST, and the synthetic Algorithm 7 coordinates were not asserted
on-curve.

| Internal path | Best raw Fp/s | Synthetic Algorithm 7 adds/s | Effective scheduled Fp/s |
| --- | ---: | ---: | ---: |
| 6x64 CIOS | 885,407,000 | — | — |
| 12x32 explicit `mulhi(uint)` | 1,178,251,000 | 56,982,000 | 683,778,000 |
| 12x32 with `ulong` accumulator | 1,584,874,000 | **77,980,000** | **935,766,000** |
| 24x16 | 1,059,987,000 | — | — |
| 24x16 with `mad24` | 1,046,687,000 | — | — |

The repository result above supersedes the temporary raw estimate; the
temporary fused schedule remains useful as an optimistic scouting bound. Even
that schedule reaches only 55.0% of the 1.7-billion-effective complete-add gate
and still needs a 1.82x full-formula improvement. An idealized SIMD-cooperative
issue-count bound is also unfavorable: four-lane cooperation cannot exceed
about 89% of the scalar multiplier efficiency before shuffles and carry
propagation, and sixteen-lane cooperation cannot exceed about 52%.

The ephemeral exploratory sources and output retained on the measured host are
`/tmp/fp_limb_probe.{metal,swift,out}` with SHA-256
`6bcd6bf597304933d497b5e85c135fb24b1ae7faf42b8dd3c3ef02953ea6fe3b`,
`167664b190144936a39bd9beb52fe2b78d6781aa9a62a565d94da157d05879e7`,
and `889e35bb9e8bf1e39deb74efcc6d29fbd35de1602e1f7c740ec194e180114349`
respectively. They are not checkout-reproducible acceptance evidence; the
repository BLST-checked probes above are authoritative. Scalar arithmetic has
now failed the raw, fused, and explicit-interleaving gates. Both subsequent
lower-operation batch-affine candidates also failed their measured gates; see
`ledger-9-metal-batch-affine-result-2026-07-15.md` and
`ledger-9-metal-flat-segmented-result-2026-07-15.md`. Full Pippenger routing
does not proceed around a rejected formula.

## G1 probe throughput

Command:

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  --test metal_g1 report_g1_probe_throughput -- --ignored --nocapture
```

For 2,048 points and four rounds, the combined normalization, doubling, and
mixed-add probe completed in 112.709 ms:

| Metric | Result |
| --- | ---: |
| Input point triplets/second | 72,683 |
| Affine/projective outputs/second | 218,049 |

This is not an MSM forecast. The diagnostic normalizes every projective point
with an individual fixed-exponent inversion, whereas a production MSM returns
only compact final commitments and uses reviewed bucket arithmetic plus batch
normalization where needed.

## Fixed-schedule complete-add gate

Command:

```bash
CARGO_INCREMENTAL=0 cargo test -p midnight-metal-prover --release \
  --test metal_g1 report_g1_complete_add_chain_throughput \
  -- --ignored --nocapture
```

The timed command holds independent homogeneous points in registers and applies
64 RCB Algorithm 7 additions per chain. Jacobian-to-homogeneous setup and
homogeneous-to-Jacobian egress run once outside the timed region; there is no
affine inversion or readback inside it.

| Elements | Timed dispatches | Commands / dispatches / waits | Complete adds/s | Effective Fp M/s | Timed ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4 | 1 / 4 / 1 | 9,797,487 | 117,569,839 | 214.050 |
| 8,192 | 8 | 1 / 8 / 1 | 10,169,312 | 122,031,746 | 412.447 |

Algorithm 7 schedules 12 general Fp multiplications and 27 field
additions/subtractions. The measured path is therefore about 14x below the
1.7-billion-effective-multiplication/s gate required for the initial trusted-
local Pippenger budget. A faster internal Fp representation or cooperative
schedule must pass the same complete-formula differential suite and fused gate
before the full MSM pipeline is built.

## Resident NTT throughput

Command:

```bash
CARGO_INCREMENTAL=0 \
MIDNIGHT_METAL_NTT_BENCH_LOG_SIZE=17 \
MIDNIGHT_METAL_NTT_BENCH_BATCHES=4 \
MIDNIGHT_METAL_NTT_BENCH_ROUNDS=16 \
cargo test -p midnight-metal-prover --release --lib \
  metal::throughput_tests::report_resident_ntt_throughput \
  -- --ignored --exact --nocapture
```

The diagnostic warms the validated domain plan, then records sixteen forward
and inverse pairs in one command buffer over four 131,072-element batches. Two
local runs measured 32.62M and 32.76M element-transforms/second (514.334 ms and
512.078 ms for 32 transforms). Each run reported 96 dispatches, one host wait,
one 16 MiB logical host write, and one 16 MiB logical host read.

The profiled CPU proof sample contains 13,410,304 transformed elements across
239 calls. Dividing that element count by this large-batch phase rate gives an
illustrative 0.41-second kernel floor, versus the sample's 0.844-second CPU FFT
union. It is not an integrated prediction: the real calls have mixed sizes,
must be grouped around transcript dependencies, and should remain resident
instead of reproducing this diagnostic's outer write/read boundary.

## Validation commands

The primitive slice passed native package tests, changed-crate Clippy with
`-D warnings`, formatting, and an `x86_64-unknown-linux-gnu` package check using
`CARGO_INCREMENTAL=0`. Proof readiness remains false and proof-server continues
to reject the Metal route before fixture construction.
