# Blocked-radix-128 Karatsuba/L4 field result

Algorithm version 3 is `NO_GO` for the Apple M1 Max. It clears the portable
arithmetic, generated-source liveness, Metal correctness, and optimized-AIR
structural gates, but fails the decisive both-operands-varying throughput gate:

| Decision metric | Measured | Required | Gate fraction |
| --- | ---: | ---: | ---: |
| Median GPU complete L4 REDCs/s | 465,444,165.041 | 2,000,000,000 | 23.27% |
| Nearest-rank p5 GPU complete L4 REDCs/s | 455,317,995.237 | 1,800,000,000 | 25.30% |
| Median wall complete L4 REDCs/s | 461,162,042.514 | 2,000,000,000 | 23.06% |
| Nearest-rank p5 wall complete L4 REDCs/s | 446,254,477.748 | 1,800,000,000 | 24.79% |

The median GPU result is 4.30 times below its gate and the p5 GPU result is
3.95 times below its gate. This is the final field stop for this M1 Max plan.
It authorizes no point-formula, reduction-tree, routing, bucket, or MSM work.
It does not expose a production field or MSM API.

## Frozen implementation

The package-owned generator emits a flattened blocked Montgomery recurrence
over twelve radix-`2^32` limbs. With `Q = 2^128`, it consumes the left operand
in three four-limb blocks and performs 63 base products per round:

```text
three a_i * b K4 products = 27
one low quotient K4       =  9
three q_i * p K4 products = 27
                               --
per round                   = 63
three rounds                = 189 base products
```

The accepted generated schedule has:

```text
algorithm version                 = 3
input range                       = [0, 4p)
raw output bound                  = < 3p
schedule FNV-1a fingerprint       = 0x0ead48b66cf3f532
ABI fingerprint                  = 0x98e696de1bd11618
generated schedule SHA-256        = 14bca956d56fbeebe0faed78f4f7084201c1bd75c499eb634894895740b46a5f
typed last-use peak               = 54 u32-equivalent words
tokenized-source last-use peak    = 54 u32-equivalent words
conservative allocation-time peak = 55 u32-equivalent words
hard source-live limit            = 56 u32-equivalent words
```

The typed operation graph and an independent definition/last-use scan over the
emitted Metal agree at 54 words. The stricter allocation-time scan counts a
new definition before killing values whose last use occurs in that statement;
it reaches 55 and is the acceptance value. A `ulong` counts as two words in all
four reported scans. The result therefore passes the source gate by one word.
These are source-level accounting results, not physical Apple GPU register
counts.

This result is separate from, and does not overwrite, the algorithm-version-2
`NO_GO` in [`KARATSUBA_L4_RESULT.md`](KARATSUBA_L4_RESULT.md). Version 2 failed
before timing because its complete source-live bounds exceeded 56 words.

## Correctness result

The portable exact-integer model passes the three-round recurrence against an
independent `R^-1 mod p` oracle. It verifies the full multiplier identity, the
truncated low-K4 quotient, exact low-block cancellation, the optimized
cancellation-carry identity, the `< 3p` output bound, and the fixed count of 189
base products. Its deterministic cases include word and block boundaries,
all-ones and alternating patterns, carry chains, random `[0, 4p)` pairs, and
forced `q = 0` and `q = 2^128 - 1` in each of the three rounds.

The official Metal suite passes against BLST before the performance samples are
collected. Coverage includes:

- all 144 Cartesian pairs from `x in {0, 1, p - 1}`, each represented as
  `x + k*p` for `k = 0..3`;
- the forced quotient extremes in every blocked round;
- the fixed-right and rotating recurrence kernels;
- every reported legal threadgroup width, zero and ragged lengths around both
  SIMD and selected threadgroup boundaries, and untouched output padding;
- a 4,113-element, seven-iteration fixed-right dependent chain and a
  4,113-element, eleven-iteration rotating chain; and
- ABI validation plus fail-closed rejection of mismatched lengths and either
  operand at the excluded upper bound `4p`.

Passing this suite validates the tested diagnostic and recurrence. It does not
provide a constant-time public API, in-place alias contract, specialized
squaring, fused D2/add-sub primitive, point formula, or resident MSM runtime.

## Optimized AIR result

The final release artifacts were rebuilt on 2026-07-16. Isolating the two
decision kernels with `xcrun air-objdump -d` gives:

| Optimized AIR property | Fixed RHS | Rotating operands |
| --- | ---: | ---: |
| Function IR lines | 4,315 | 4,327 |
| Basic blocks / branch instructions | 7 / 6 | 7 / 6 |
| Multiply instructions | 189 | 189 |
| `i64` / `i32` multiplies | 177 / 12 | 177 / 12 |
| Variable-by-variable / variable-by-constant multiplies | 81 / 108 | 81 / 108 |
| Loads / stores / GEPs | 26 / 12 / 38 | 26 / 12 / 38 |
| `i32` PHIs | 25 | 37 |

Both functions have one outer chain loop. Their arithmetic loop contains no
load, store, GEP, call, or invoke; the twelve stores occur after the loop. The
isolated functions contain no `alloca`, out-of-line call, private or
threadgroup address-space reference, vector/SIMD operation, or dynamic limb
GEP. The rotating function's twelve additional `i32` PHIs are the loop-carried
right operand, which is the intended non-hoistable decision shape.

This AIR is optimized intermediate representation, not final Apple GPU ISA.
Apple's available tools did not expose physical register allocation or
spill/fill traffic. Consequently, the AIR result is useful structural evidence
but **does not prove physical no-spill execution**. The throughput failure is
decisive regardless.

## Performance result

The favorable short fixed-right scout used 131,072 fields, eight iterations,
one dispatch, one warmup, and three samples per width. Its best width-median
was 341,347,222.787 complete L4 REDCs/s at width 32. This was explicitly
scout-only and optimistic because the invariant right operand permits compiler
hoisting.

The formal gate instead uses the recurrence
`(left, right) <- (right, left * right * R^-1 mod p)`, so both operands are
loop-carried. Its frozen settings are:

```text
logical fields                 = 131,072
iterations per dispatch        = 32
dispatches per sample          = 4
complete operations per sample = 16,777,216
selected threadgroup width     = 64
decision warmups               = 2
decision samples               = 20
GPU statistic                  = median and nearest-rank p5
```

Allocation, upload, and readback are outside the timed interval. Each sample
contains one command submission and one completion wait; output and padding are
checked after every warmup and sample. All 20 raw GPU and wall timings are
frozen in
[`evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json`](evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json).

The result also closes an important projection error. The earlier native-`u32`
product probe measured about 1.901 trillion products/s; dividing that unrelated
schedule by 189 suggests about 10.06 billion product-only equivalents/s, but it
is neither this schedule nor a complete-field/device ceiling. Likewise, scaling
the earlier 1.529B named-FIOS result from roughly 300 AIR multiply-class
operations to 189 suggested about 2.427B only as prioritization context. The
actual rotating v3 result is 0.465B. Product counts were never a forecast:
carry/control work, dependency depth, and register pressure dominate this
mapping.

## Stop decision

The failure is large enough that incremental tuning of this field kernel is not
a credible path to the required Metal MSM on this machine. Under the candidate
contract, squaring, fused two-product point primitives, complete formulas, and
the standalone Metal recode/route/reduce/integrate/collect pipeline remain
unauthorized. Recreating sppark's useful *semantic capabilities* in a
Metal-native standalone library remains the right eventual boundary, but this
M1 Max arithmetic result does not justify starting that work.

Machine-readable samples, exact gate fractions, source and artifact hashes,
toolchain identity, correctness scope, and AIR limitations are in the evidence
record linked above. The predeclared contract remains in
[`BLOCKED128_KARATSUBA_L4_CANDIDATE.md`](BLOCKED128_KARATSUBA_L4_CANDIDATE.md).
