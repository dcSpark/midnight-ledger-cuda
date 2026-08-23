# zkMetal BLS12-381 MSM evaluation — 2026-07-15

## Decision

Do not vendor, wrap, or adopt zkMetal's BLS12-381 MSM. Untouched commit
[`443127161417b415f70d5de74a186a1820e557e5`](https://github.com/carni-ships/zkMetal/tree/443127161417b415f70d5de74a186a1820e557e5)
misses the authentic aggregate budget by roughly two orders of magnitude, is a
CPU+Metal hybrid rather than a mostly-resident Metal engine, and does not provide
credible large-input GPU correctness evidence. The reviewed commit also has no
repository license or source SPDX declarations, so it is clean-room research
input only.

## Reproduction

The repository was shallow-cloned outside the project and remained unmodified:

```bash
git clone --depth 1 https://github.com/carni-ships/zkMetal.git /tmp/codex-zkMetal
cd /tmp/codex-zkMetal
/usr/bin/time -lp swift build -c release --product zkbench
mkdir -p /tmp/zkmetal-bench-home
/usr/bin/time -lp env HOME=/tmp/zkmetal-bench-home \
  CFFIXED_USER_HOME=/tmp/zkmetal-bench-home \
  .build/release/zkbench bls381msm
```

The benchmark command ran twice with an isolated home to avoid a pre-existing
metallib or tuning cache. `git diff --exit-code` passed and `git status --short`
remained empty.

- Device: Apple M1 Max, 32-core GPU, 64 GiB
- Swift: 6.3.3
- Xcode: 26.6
- Release build: 431.17 seconds wall, 7.91 GB peak RSS

## Measurements

| Measurement | Sweep 1 | Sweep 2 |
| --- | ---: | ---: |
| point generation | 546.9 ms | 559.9 ms |
| 1,024 non-GLV | 7.1 ms | 6.3 ms |
| 1,024 apparent GLV | 10.8 ms | 5.9 ms |
| 2^8 | 2.9 ms | 2.0 ms |
| 2^10 | 10.0 ms | 6.0 ms |
| 2^12 | 28.4 ms | 29.7 ms |
| 2^14 | 375.4 ms | 377.8 ms |
| 2^16 | 180.7 ms | 173.8 ms |
| 2^17 | 232.8 ms | 223.7 ms |
| 2^18 | 459.6 ms | 440.0 ms |

The non-monotonic 2^14 result reproduced. The built-in benchmark has no 2^13 or
2^15 case and exposes no BLS12-381 size override. These are wall times from the
external program, not package acceptance measurements.

The authentic Ledger-9 capture has 60 exact 2^14 calls. Their mean zkMetal time
alone projects to:

```text
60 * 376.6 ms = 22,596 ms
```

That is 56.49 times the complete 400 ms median budget and 45.19 times the 500 ms
p95 budget before the two 16,383-point calls or any 8K/32K call. Across all 147
authentic calls, the allowed mean is only 2.721 ms for the median budget and
3.401 ms for the p95 budget.

## Architecture mismatch

The [engine](https://github.com/carni-ships/zkMetal/blob/443127161417b415f70d5de74a186a1820e557e5/Sources/zkMetal/MSM/BLS12381MSMEngine.swift)
routes at most 2,048 points to CPU Pippenger. Larger inputs still perform scalar
reduction, signed-digit extraction, and count-sort on the CPU, then run Metal
bucket reduction/integration and a separate Metal Horner command with two host
waits. This is neither GPU-resident routing nor the mostly-Metal design required
here.

Its base-field shader is ordinary twelve-`u32` CIOS with `ulong` products. That
is the same scalar arithmetic family already measured below the standalone
field gate; it is not a new cooperative field engine.

## Correctness and safety limitations

The timings are useful only as negative evidence:

1. The printed eight-point “GLV correctness” comparison runs below the 2,048
   threshold, so both sides use CPU code. Timed large outputs are discarded
   without an independent reference comparison.
2. The non-GLV `w=15` path used at 2^17 and 2^18 allocates exactly 17 windows and
   discards the final balanced-digit carry. For the deterministic benchmark,
   53,206/131,072 and 106,163/262,144 scalars are affected; the resulting error
   coefficient is nonzero modulo the BLS12-381 scalar order. Those two reported
   timings therefore produce incorrect MSMs.
3. The large GLV host path allocates `n * windows` `u16` digit slots, then copies
   `2n * windows` slots. Its shader layout also overlaps one window's second GLV
   stream with the next window's first stream.
4. The reviewed snapshot contains no `LICENSE`, `COPYING`, `NOTICE`, or SPDX
   declaration.

## Useful clean-room lessons

Only neutral concepts remain useful: package-owned kernels, prepared buffer
reuse, signed-window staging, explicit size tuning, and adversarial tests for
final carries and stream offsets. No source, constants, ABI, or performance
claim from zkMetal enters `bls12-381-metal-msm`.
