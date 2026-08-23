# Blocked radix-128 Karatsuba/L4 field candidate

Status: **completed `NO_GO` on Apple M1 Max**. The portable model, generated
source-liveness gate, Metal correctness suite, and optimized-AIR structural
checks pass, but the required rotating-operands decision run reaches only
0.465B median and 0.455B nearest-rank p5 complete L4 operations/s versus the
2.0B/1.8B gates. The frozen result and all 20 timing samples are in
[`BLOCKED128_KARATSUBA_L4_RESULT.md`](BLOCKED128_KARATSUBA_L4_RESULT.md) and
[`evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json`](evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json).

This document defines algorithm version 3, the final bounded field experiment
for the standalone Metal MSM package. It does not expose a production field or
MSM API and does not supersede the algorithm-version-2 `NO_GO` recorded in
[`KARATSUBA_L4_RESULT.md`](KARATSUBA_L4_RESULT.md).

## Direct blocked recurrence

Let `B = 2^32`, `Q = B^4`, and `R = Q^3 = 2^384`. Split the left operand into
three four-limb blocks,

```text
a = a_0 + a_1 Q + a_2 Q^2.
```

Starting with `t_0 = 0`, run three fixed rounds:

```text
s_i     = t_i + a_i * b
q_i     = (s_i mod Q) * (-p^-1 mod Q) mod Q
u_i     = s_i + q_i * p
t_(i+1) = u_i / Q
```

The low block of `u_i` is zero because `q_i * p = -s_i (mod Q)`. After three
rounds, `t_3 = a*b*R^-1 (mod p)`.

For inputs in `[0, 4p)`, every pre-shift `u_i` is strictly below `Q^4`, so a
16-word accumulator is sufficient and its top carry is zero. The final raw
result is below `3p`, preserving the lazy-L4 closure required by repeated
chains. The exact inequalities and executable checks are in
[`tests/blocked128_karatsuba_l4_model.rs`](tests/blocked128_karatsuba_l4_model.rs).

## Exact multiplication count

A four-limb subtractive Karatsuba product uses nine base
`u32 x u32 -> u64` products:

```text
K(1) = 1
K(2) = 3
K(4) = 3 * K(2) = 9
```

Each blocked round performs:

```text
a_i * b across three K4 blocks = 27
low four-limb quotient product =  9
q_i * p across three K4 blocks = 27
                                      --
per round                         = 63
three rounds                      = 189
```

The quotient product may discard its high four words, but it still requires
nine exact base products. Additive Karatsuba evaluations that require a
33-bit temporary are outside this count and are not permitted by this
candidate.

Algorithm version 3 has the same base-product count as the rejected separated
K12 schedule, but it replaces three deeply nested K12 products and a retained
wide numerator with smaller sink-oriented K4 phases. That structural change is
the reason it merits a separate liveness and performance decision.

## Required generated schedule

The Metal generator must emit one fully flattened, fixed-schedule function
with all of the following properties:

- named scalar words only: no private arrays, dynamic limb indexing, runtime
  recursion, SIMD shuffles, or threadgroup scratch;
- K2 products consumed into a K4 sink without retaining nested K4/K12 result
  trees;
- every mutable word represented as a new SSA value in the generator;
- a typed operation-graph liveness audit that includes both 12-word inputs,
  the complete accumulator, quotient, result, flags, and carry values;
- an independent definition/last-use audit over the emitted Metal source;
- `ulong` values counted as two `u32`-equivalent live words; and
- a frozen algorithm and source fingerprint for every accepted schedule.

Both source audits must agree and report at most 56 live `u32` equivalents.
That source gate is necessary, not sufficient: optimized AIR inspection and
device-level spill/performance evidence remain separate obligations.

## Stop gates

The experiment proceeds in this order and stops on the first miss:

1. Portable exact arithmetic, cancellation, range, and 189-product accounting.
2. Generated whole-function source liveness at or below 56 words.
3. Generated Metal boundary, randomized, ragged-length, padding, ABI, and
   dependent-chain correctness.
4. Optimized AIR structure and the strongest available physical spill check.
5. Fixed-right-operand scout.
6. Both-operands-varying, 20-sample decision run at 2.0 billion median and
   1.8 billion nearest-rank p5 complete L4 operations per second.

Only a pass through step 6 authorizes work on squaring, fused two-product
operations, point formulas, or MSM routing. In particular, a direct extension
of this scan to a fused two-product operation costs 270 base products; a point
implementation must independently derive and gate a better fused schedule
rather than extrapolating from multiplication.

## Independence boundary

This is a package-owned Metal design. CUDA and sppark may supply test vectors
or high-level semantic requirements, but neither is an execution dependency or
topology template. zkMetal is not a source dependency. The eventual standalone
library must own recoding, routing, bucket reduction, integration, collection,
resident-buffer contracts, completion lifetime, and erasure without an
implicit CPU fallback.
