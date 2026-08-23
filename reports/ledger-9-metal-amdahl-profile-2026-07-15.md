# Ledger 9 Metal Amdahl profile — 2026-07-15

## Conclusion

On one instrumented Apple M1 Max diagnostic sample, CPU FFT and MSM leaf work
occupied a combined interval union of `2975.371 ms` inside a `5817.190 ms`
balanced zswap transaction. Removing all of those intervals would leave an
optimistic `2841.819 ms` residual and cap leaf-only acceleration at `2.047x`.

The production Metal gate is `1757.940 ms`, derived from the accepted CPU
median of `5273.820 ms`. Consequently, Metal NTT and MSM must be resident
foundations for later argument, numerator, quotient, and multi-open stages.
Implementing them as host round-trip leaf replacements cannot reach the `3.0x`
end-to-end requirement.

This report contains one diagnostic timing sample, not an acceptance result.
The release decision continues to use paired, alternating CPU/Metal medians,
normal self-verification, byte-exact output, zero fallback, and the full
benchmark contract in `docs/metal-backend.md`.

## Reproduction

The diagnostic used the existing release binary without rebuilding it:

```bash
MIDNIGHT_PROVER_TIMERS=1 \
MIDNIGHT_LEDGER_TEST_STATIC_DIR="$PWD/scripts" \
./target/release/proof-backend-parity \
  --backend cpu \
  --fixture zswap-transaction \
  --iterations 1 \
  --warmup 0 \
  --seed 24302 \
  --json /tmp/metal-plan-seed24302.json \
  2>&1 | tee /tmp/metal-plan-timers-seed24302.log >/dev/null
```

The deterministic fixture contains five proof jobs: two zswap spends, two
zswap outputs, and one Dust fee spend. It has 56,950 aggregate rows, 1,397 IR
instructions, and a maximum circuit size of `k=15`.

### Diagnostic artifacts

| Artifact | Size | Lines | SHA-256 |
| --- | ---: | ---: | --- |
| `/tmp/metal-plan-seed24302.json` | 1,343 bytes | — | `0c6f5750a65e60c9061f92050feb97f93045d6d232a6233cd451fa61232c5059` |
| `/tmp/metal-plan-timers-seed24302.log` | 182,186 bytes | 1,833 | `0d82c87f34be074a69438a64189960343a3c904a35d8f12025cc9e0727ae079c` |
| `benchmark-results/metal-cpu-baseline-m1-max.json` | 1,976 bytes | — | `f11b706995341d689b2e8b40cee23f6af28d90449675ab0ff05975a9378591eb` |

The diagnostic JSON reports `5817.190041 ms`, 23,476 serialized bytes, and the
proof digest:

```text
len:23476 xor:191 sum64:2924713 rolling:105419955334794057
```

## Interval parser

Timer completion lines were parsed with:

```text
^\[midnight-prover\] (\S+?)(?: count=(\d+))? ([0-9.]+)ms @(\d+)$
```

For every record:

```text
end_ns   = captured final integer
start_ns = end_ns - round(elapsed_ms * 1_000_000)
```

The printed duration has millisecond precision to three decimal places, so
reconstructed starts have corresponding rounding precision. This does not
materially affect the millisecond conclusions below.

The analysis found each
`proof_server.prover.plonk.create_proof` interval and retained only fully
contained leaf records named:

- `proof_server.prover.curves.best_fft`
- `proof_server.prover.kzg.msm_specific.filtered`

Parent commitment timers were excluded to avoid double counting, as were
verification MSMs after `create_proof` completed. For each proof and leaf set,
the parser sorted intervals by start time, merged every next interval whose
start was no later than the current end, and summed the merged interval
lengths. The combined union repeated the same operation over FFT and MSM
intervals together.

This union is required because Rayon executes many MSM leaves concurrently.
Their naive duration sum is much greater than their elapsed critical interval.

## Results

### Per-proof intervals

All values are milliseconds.

| Proof | `create_proof` | `compute_trace` | `finalise_proof` | FFT union | MSM union | Combined union |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 280.975 | 128.328 | 152.630 | 42.087 | 148.585 | 190.672 |
| 2 | 1549.892 | 611.289 | 938.595 | 314.240 | 677.345 | 969.336 |
| 3 | 1416.369 | 677.464 | 738.896 | 248.012 | 682.033 | 911.354 |
| 4 | 767.859 | 378.866 | 388.966 | 115.938 | 336.720 | 452.658 |
| 5 | 758.831 | 372.449 | 386.363 | 123.276 | 337.628 | 451.351 |
| **Total** | **4773.926** | **2168.396** | **2605.450** | **843.553** | **2182.311** | **2975.371** |

### Aggregate leaf work

| Leaf | Calls inside proofs | Naive duration sum | Interval union |
| --- | ---: | ---: | ---: |
| FFT | 239 | 859.976 ms | 843.553 ms |
| MSM | 147 | 6248.929 ms | 2182.311 ms |

The separate FFT and MSM unions total `3025.864 ms`. Their combined union is
`2975.371 ms`, showing `50.493 ms` of overlap between the two leaf classes.

The FFT size distribution was:

| Elements | Calls | Naive duration sum |
| ---: | ---: | ---: |
| 8,192 | 17 | 9.350 ms |
| 16,384 | 50 | 58.286 ms |
| 32,768 | 68 | 165.331 ms |
| 65,536 | 52 | 186.535 ms |
| 131,072 | 52 | 440.474 ms |

MSM work is primarily full or nearly full 8K, 16K, and 32K commitments. Exact
full-size calls include 8 at 8,192 points, 22 at 16,384 points, and 22 at
32,768 points, in addition to `n-1` and structured sparse or ragged variants.

## Amdahl budget

For this diagnostic sample:

```text
measured transaction       = 5817.190041 ms
combined FFT/MSM union      = 2975.371000 ms
optimistic residual floor  = 2841.819041 ms
leaf-only maximum speedup  = 5817.190041 / 2841.819041
                           = 2.046995x
```

The accepted five-sample CPU baseline is:

```text
CPU median                 = 5273.819625 ms
3.0x Metal gate            = 1757.939875 ms
```

The diagnostic residual floor is `1083.879166 ms` above that gate. This gap is
not a cross-run performance prediction: the diagnostic and accepted median are
different samples. It is structural evidence that the implementation must
accelerate and retain more of the proof dataflow than the FFT and MSM leaves.

## Resident implementation direction

### NTT and iNTT

- Store polynomials as reduced little-endian four-`u64` Montgomery Fr values.
- Keep input and output in natural order.
- Use a batched mixed-radix Stockham transform with arena-owned ping-pong
  buffers. A radix-256 tile occupies only 8 KiB of threadgroup memory.
- Cache roots, inverse roots, twiddles, inverse domain sizes, and pass plans in
  `MetalKeyContext`; keep proof scratch in `MetalProofArena`.
- Forward NTT is unnormalized. Inverse NTT applies `n^-1` exactly once in the
  final write.
- For the Halo2 coset, ingress multiplies by the repeating pattern
  `[1, zeta, zeta^2]`; egress uses `[1, zeta^2, zeta]`.
- Provide semantic resident operations for `lagrange_to_coeff`,
  `coeff_to_extended`, `extended_to_coeff`, and
  `divide_vanishing_to_coeff`. Do not make host `Vec` round trips the main
  integration boundary.
- Product transforms are `k=13` through `k=17` and should take two or three
  ordered dispatches per batch without an intervening host wait.

### G1 MSM

- Cache coefficient and Lagrange SRS bases in fixed-base `MetalKeyContext`
  plans and accept resident Montgomery Fr batches.
- Convert every scalar to canonical limbs before signed-window digit
  extraction. Raw Montgomery bits are not scalar bits.
- Use a histogram, scan, and scatter Pippenger specialized to the bounded
  signed-window bucket range. The ec-gpu OpenCL MSM scheduling, with private
  global bucket arrays and CPU collection, is not suitable for the production
  M1 path.
- Route normal short bucket runs to one-thread accumulation and skewed long
  runs to a cooperative threadgroup path. Benchmark the run-length
  distribution on actual Lagrange and advice scalars.
- Treat deterministic scalar scrambling and its precomputed correction point as
  an optional performance experiment only. It is not a confidentiality control
  and remains disabled initially because it can make the observed sparse
  workload nearly dense.
- Fold all buckets and windows on-device. Read back only one projective
  commitment per MSM; any `z=0` projective value represents identity.
- Tune window bits on M1 data. Initial candidates that target roughly 64
  points per bucket are `w=8`, `w=9`, and `w=10` for 8K, 16K, and 32K MSMs,
  respectively; these are benchmark candidates, not inherited CUDA settings.

The subsequent fixed-schedule complete-add diagnostic rejected the initial
six-`u64`, one-point-per-thread implementation at only 9.80--10.17 million
complete additions/s, about 14x below the minimum MSM arithmetic gate. The
pipeline description above remains the leakage-accepted algorithmic target,
but implementation now pauses before histogram/scatter work while a faster
internal Fp representation or cooperative schedule is measured. Exact workload
and gate evidence are in
`reports/ledger-9-metal-msm-feasibility-2026-07-15.md`.

### First useful resident boundary

The first implementation slice should land resident batched NTT/iNTT and its
coset operations, including command recording without automatic waits and
arena-owned ping-pong buffers. The following MSM slice should join it at one
complete advice boundary:

```text
resident advice in Lagrange form
    |-- lagrange-SRS MSM commitments --> compact transcript egress
    `-- normalized iNTT --> resident coefficients for later proof stages
```

After that gate, lookup/permutation products, numerator and quotient
construction, evaluations, and KZG multi-open must consume the same resident
polynomials. This is the path capable of reducing the measured residual; a
collection of isolated primitive bridges is not.
