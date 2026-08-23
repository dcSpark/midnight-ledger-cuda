# Branch-completed XYZZ promotion gate

Status: **not authorized on the current M1 Max mappings; retained as a future
hardware/arithmetic promotion contract**.

This document freezes the next curve-level decision without coupling the
standalone library to Midnight's prover or workload types. It is a bounded
falsifier for the fast trusted-local accumulation path, not a production API.
The rejected two-lane and generated named-register field mappings must not be
extended into this work. A materially different engine must first pass its own
2.0B median / 1.8B p5 complete-field gate.

## Point formats and common paths

The diagnostic uses affine inputs and XYZZ bucket state `(X, Y, ZZ, ZZZ)`.
The first tree level must specialize affine-plus-affine; converting both inputs
to a generic projective formula discards the only remaining arithmetic
headroom. Higher levels retain XYZZ state, and bucket integration uses full
XYZZ-plus-XYZZ additions unless a separately measured normalization cost is
included.

For BLS12-381 G1 with `a = 0`, the reviewed schedules are:

| Operation | Normal | Double after equality test | Inverse |
| --- | ---: | ---: | ---: |
| affine + affine -> XYZZ | `4M + 2S + 7A` | `4M + 3S + 9A` | `2A` |
| XYZZ + affine | `8M + 2S + 7A` | `6M + 3S + 9A` | `2M + 2A` |
| XYZZ + XYZZ | `12M + 2S + 7A` | `10M + 3S + 9A` | `4M + 2A` |
| known XYZZ double | -- | `6M + 3S + 7A` | -- |

Identity operands are handled first by copying the other operand. For the
remaining path, `H = 0, r != 0` selects canonical identity and `H = r = 0`
selects doubling. Doubling a zero-`Y` point must also produce canonical
identity. Signed affine materialization uses a fixed subtract/select schedule.

The fast form is explicitly part of a leakage-accepted trusted-local profile:
exception selection and digit-indexed routing are not strict-oblivious. A
version that always evaluates normal and double candidates costs approximately
`18M + 5S + 14A` for a full XYZZ addition and is outside the current budget.
The eventual public API must require an explicit leakage policy rather than
silently choosing the fast profile.

## Correctness gate

The package-owned diagnostic must compare canonical affine results against BLST
for:

- affine-pair, mixed, full, and known-double entry points;
- left, right, and double identity;
- equal, inverse, and zero-`Y` inputs;
- positive and negative signed affine inputs;
- randomized subgroup points and independently rescaled XYZZ representatives;
- on-curve non-subgroup inputs used only by the diagnostic; and
- ragged dispatch widths with normal and exceptional lanes in one SIMD group.

Every result remains observable and checked after every command. There is no
CPU fallback, post-selection retry, unchecked infinity encoding, or
unmeasured normalization.

## Performance gates

Allocation, upload, reference computation, and readback occur outside the GPU
timestamp. Retain two warmups and at least twenty completed-command samples,
report GPU and wall clocks, use nearest-rank p5, freeze artifact hashes, and
record final register/spill evidence. Spill-dependent scratch is `NO_GO`.

Promotion is sequential:

1. A chained full, branch-completed XYZZ kernel must complete at least
   **166.7M actual additions/s median** and **150M/s p5**. Do not relabel this as
   a `12M` field-equivalent rate: the normal formula also contains two squares
   and seven additions/subtractions.
2. A specialized affine-first/full-higher tiled replay supplied through a
   neutral benchmark descriptor must complete its complete bucket phase within
   the application adapter's predeclared budget. The current parent-project
   decision gate is 300 ms median / 375 ms p95, but that application threshold
   is not part of the future library API.
3. Only after both passes may the library implement canonical scalar recoding,
   routing, integration, compact output, and erasure. The parent adapter then
   applies its own complete-batch acceptance gate.

Failure at any stage leaves the production execution surface unavailable.
