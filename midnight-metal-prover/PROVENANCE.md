# Metal backend provenance

The host-runtime architecture and engineering checklist were informed by
Nightstream's Apache-2.0 `neo-prover-metal` backend. The reproducible upstream
reference used for attribution is Nightstream commit
`29ef5a53a665f7b0347cee790bebbfcdcc0c4d69`.

The user also supplied an uncommitted local guide,
`METAL_IMPLEMENTATION_BEST_PRACTICES.md`, whose SHA-256 at review time was
`40e9ede592f887716937b9949a9c6ffd5cc98207c8ffd4f8a5f5f562e0a77bde`. The
Nightstream working tree containing that guide had other uncommitted Metal
changes, so those working-tree files are not treated as a reproducible source
revision.

Reusable ideas include long-lived session ownership, offline metallib builds,
explicit resident/materialized boundaries, retained command resources, and
activity accounting. Midnight's shaders, BLS12-381 arithmetic, proof-stage
integration, constants, and correctness tests are implemented independently;
Nightstream uses a different field and proof system.

## BLS12-381 G1 formulas and differential reference

The G1 shader implements the complete homogeneous formulas from
Renes-Costello-Batina, [Algorithms 7, 8, and
9](https://eprint.iacr.org/2015/1060), specialized to BLS12-381 G1 with `a = 0`,
`b = 4`, and `3b = 12`. Algorithm 7 full addition costs `12M + 27A`, Algorithm
8 homogeneous-plus-affine addition costs `11M + 21A`, and Algorithm 9 doubling
costs `6M + 2S + 13A`; these counts include the four-addition chain used for
each multiplication by `3b`. The shader follows the operation ordering in the
repository's dual MIT/Apache-2.0 vendored `midnight-curves` 0.3.1 source at
commit `64f5d8b7099079ac6c7f8d53723e559cdf9f56d3`. The reviewed `derive/curve.rs`
had SHA-256
`f6933e2771b6b762b0ff00e23783821a6711562ba20beac90bf4df15ff4b40c4`.

The public buffer ABI remains BLST/CUDA-compatible Jacobian. Kernel boundaries
convert between Jacobian `(X, Y, Z)` and the Metal-only homogeneous
representation of `(X/Z, Y/Z)` using the conversion formulas in Apache-2.0
[`blst` v0.3.16 `ec_ops.h`](https://github.com/supranational/blst/blob/v0.3.16/src/ec_ops.h).
The reviewed file had SHA-256
`b4ace5a1efe5354fc622a11b05eb65f4cc81aa91f6a0baa1912caa34600d4fd0`.
Public Jacobian infinity is canonicalized to all zeroes; internal homogeneous
infinity is `(0, R, 0)` and therefore requires an explicit identity fill after
storage erasure.

Differential tests use BLST v0.3.16's public G1 operations in `src/e1.c` as an
independent CPU oracle. That code uses different Jacobian add/double formulas;
the reviewed file had SHA-256
`4939fe48680b3f3aad94501aacbfd081907a1561cb66d39559885f9d0025770e`.
Tests cover canonical and noncanonical infinity, equal and inverse pairs,
independently non-unit-scaled Jacobian inputs, and deterministic scalar-derived
subgroup points. They also reproduce the vendored curve's authoritative
`g1_non_torsion_point` search and verify Algorithms 7, 8, and 9 on a point that
BLST reports is on-curve but outside the prime-order subgroup. The reviewed
`bls12_381/g1.rs` containing that search had SHA-256
`28f61cc7859ffa478f82b188929b085132346b4abe980665805bae4d33232280`.
Supranational's BLST copyright and Apache-2.0 terms remain applicable to the
reviewed reference source.

The formula and mask schedules contain no exceptional-case branches. Fr/Fp
addition, subtraction, and Montgomery final reduction use fixed limb schedules:
unconditional subtraction followed by a mask/select for reduction, and a masked
modulus add for underflow. The exponent branch in diagnostic affine inversion is
over a fixed public exponent. This code has not received an independent
side-channel audit, and these G1 probe kernels remain correctness diagnostics.
Production proof MSM is owned by the separate `bls12-381-metal-msm` package
under its documented trusted-local data-dependent profile.

The startup ABI probe directly reports every Fr/Fp/affine/projective/homogeneous
member offset in addition to size and alignment. Rust validates those words and
the pinned FNV-1a layout fingerprint `0x108cb9731d5097c4` before any diagnostic
arithmetic runs. The internal homogeneous type is not exposed through the public
Rust ABI.

## Experimental 12x32-bit Fp diagnostic

`shaders/fp32.metal` is an isolated performance experiment, not a proof-field
replacement. Its twelve little-endian `uint` limbs preserve the public
six-`ulong` Fp value's exact R=2^384 Montgomery bit layout. The build derives
the 32-bit modulus, Montgomery `R`, `R²`, and `-p⁻¹ mod 2³²` from the canonical
six-limb constants; Rust tests rejoin every generated pair and verify the
inverse identity. Public buffers are split/rejoined only in diagnostic kernels,
and the existing Fp64, NTT, G1, and proof routes are unchanged.

The 12-limb CIOS schedule uses `uint` low multiplication plus `mulhi`, fixed
carry/borrow loops, unconditional final subtraction, and mask/select reduction.
BLST v0.3.16 differential tests cover boundary cross-products, 4,096
deterministic randomized pairs, and an internal repeated-multiplication chain
for add, subtract, multiply, and square.

On the reviewed Apple M1 Max run, 262,144 independent lanes with 64 in-register
iterations and four timed dispatches reached a best 1,185,515,361 raw
multiplications/s at threadgroup width 64. The timed region contained one
command, four dispatches, and one wait, with conversion and readback afterward.
This is 59.28% of the required 2.0B optimistic upper-bound gate, so the result is
`NO_GO`; no Fp32 G1 or proof implementation follows from this experiment.

A sibling single-thread Fp path keeps the same limbs, constants, add/subtract operations,
and fixed reduction, but computes each CIOS inner step as an exact wide
accumulator: `ulong(a) * ulong(b) + ulong(t) + carry`. BLST boundary,
deterministic randomized, square, and repeated-chain coverage also passes for
this path. The ignored release command is:

```text
cargo test -p midnight-metal-prover --release report_fp32_wide_raw_mul_throughput -- --ignored --nocapture
```

On the reviewed M1 Max run, the same 262,144 lanes, 64 iterations, and four
timed dispatches reached 1,540,404,194 multiplications/s at threadgroup width
64 in 43.566 ms. The timed boundary was one command, four dispatches, and one
wait; conversion and readback occurred afterward, for three commands, seven
dispatches, and three waits total. This is 77.02% of the 2.0B gate and remains
`NO_GO`, so no wide-accumulator G1 or proof port was attempted.

The repository then explicitly alternates two and three independent wide CIOS
states at every limb step. A deterministic 128-pair repeated-chain test checks
all observable state combinations against BLST. On the same M1 Max timing
boundary, the best total rates are 963,456,555/s at width 128 and 467,233,045/s
at width 64. These `NO_GO` results close the single-thread CIOS branch; they are
diagnostic transcriptions of the same Montgomery algorithm, not copied
third-party implementation code.

## Mixed-radix Stockham NTT

The pass factorization in `shaders/ntt.metal` is adapted from the dual
MIT/Apache-2.0 `ec-gpu-gen` v0.7.1 `src/cl/fft.cl` implementation at upstream
commit `0a3e5f50ef24ef7681eeb297fc82262e12d3b605`. The reviewed crate source had
SHA-256
`0cf8b4b718ef713b9af358f688cde761d93e391095e00bacaa6fb28fb519d4be`.
Its source credits the Stockham GPU FFT presentation at
<http://www.bealto.com/gpu-fft_group-1.html>.

Midnight's Metal specialization adds a fixed raw BLS12-381 Fr ABI, natural-order
batched two-dimensional dispatch, balanced radix schedules, explicit dynamic
threadgroup memory, generation-checked resident ping-pong ownership, fused
exactly-once inverse normalization, and order-three coset ingress/egress. Host
plan constants are validated against the field identities before caching.

The two-adic root, inverse root, inverse-two, and cube-root constants used by
tests come from the repository's Apache-2.0 vendored
`midnight-curves-0.3.1/src/bls12_381/fq.rs`; the reviewed file SHA-256 was
`f12967e1d672175304ca4cac302a19e1e50d13d030759bd4a81003660174e874`.
Differential tests implement an independent iterative CPU NTT from BLST v0.3.16
field operations and cover all small domains plus the product sizes `k=13..17`.
