# Subtractive Karatsuba and L4 Montgomery candidate

This note defines a reusable arithmetic candidate for BLS12-381 `Fp` on
Metal. It freezes the representation, exact product counts, generator
constraints, correctness obligations, and the field-level performance gate.
It does not promote an implementation: every gate below must pass before this
candidate can support a point or MSM API.

## Representation and range

Let

```text
B = 2^32
R = B^12 = 2^384
p = 0x1a0111ea397fe69a4b1ba7b6434bacd7
    64774b84f38512bf6730d2a0f6b0f624
    1eabfffeb153ffffb9feffffffffaaab
```

An L4 value is a twelve-limb, little-endian integer in `[0, 4p)`, in the
usual Montgomery domain. L4 changes only the accepted representative range;
it does not change the encoding or Montgomery factor.

The useful bounds are:

```text
p / R       = 0.1015788266...
4p / R      = 0.4063153064... < 1
8p / R      = 0.8126306128... < 1
16p / R + 1 = 2.6252612256... < 3
32p / R + 1 = 4.2505224513... < 5
```

Consequently, `4p` and every temporary below `8p` fit in the existing
twelve-`u32` ABI.

L4 addition computes `a + b < 8p`, then conditionally subtracts `4p`.
L4 subtraction computes `a + 4p - b < 8p`, then conditionally subtracts
`4p`. Both selections must be branch-free in the generated shader.

Canonical egress from `[0, 4p)` takes two branch-free steps: conditionally
subtract `2p`, then conditionally subtract `p`. Equality and zero tests must
either canonicalize first or compare modulo `p`, because zero has the valid
L4 representatives `0`, `p`, `2p`, and `3p`.

## Separated Montgomery REDC

For a non-negative double-width integer `T`, define

```text
n' = -p^-1 mod R
m  = (T mod R) * n' mod R
u  = (T + m*p) / R
```

The low half of `T + m*p` is zero by construction. The implementation must
preserve the cancellation carry before reusing the low-half storage.

For ordinary multiplication with `a,b < 4p`, `T = a*b < 16p^2`, so

```text
u < (16p/R + 1)p < 3p.
```

No final subtraction is required to return an L4 result.

Two products can be fused before reduction. For addition, use
`T = a*b + c*d`; its REDC result is below `5p`. For subtraction, use the
non-negative representative

```text
T = a*b - c*d + 2pR.
```

Because `8p < R`, this `T` is positive. Moreover,

```text
-16p^2 + 2pR < T < 16p^2 + 2pR
T/R < (16p/R + 2)p
u   < (16p/R + 3)p < 5p.
```

In either fused case, one branch-free conditional subtraction of `4p`
restores the L4 invariant. The operation called `D2` below is this signed
two-product dot operation, including REDC and L4 normalization.

## Subtractive Karatsuba product

For an even `n = 2h`, split operands as

```text
A = A0 + B^h A1
C = C0 + B^h C1
```

and compute

```text
z0    = A0*C0
z2    = A1*C1
d     = abs(A1 - A0) * abs(C0 - C1)
sign  = sign((A1 - A0) * (C0 - C1))
cross = z0 + z2 + sign*d
A*C   = z0 + B^h*cross + B^(2h)*z2.
```

The sign is a mask, not a signed wide integer. Recombination uses explicit
guard limbs and carry/borrow propagation; no intermediate may rely on signed
overflow.

For an uneven split, the low half has `ceil(n/2)` limbs and the high half has
`floor(n/2)` limbs. If `K(n)` is the number of base `u32 * u32 -> u64`
products, the selected recursion gives:

```text
K(1)  = 1
K(2)  = 3
K(3)  = K(2) + K(1) + K(2) = 7
K(6)  = 21
K(12) = 63
```

The specialized square uses the same subtractive identity with three
recursive squares. A three-limb triangular leaf has six products, so

```text
S(3)  = 6
S(6)  = 18
S(12) = 54.
```

Separated REDC requires one product for the numerator, one for
`(T mod R)*n'`, and one for `m*p`. The exact base-product model is therefore:

| L4 operation | Numerator products | REDC products | Total base products |
| --- | ---: | ---: | ---: |
| `M` | `K(12) = 63` | `2*K(12) = 126` | **189** |
| `S` | `S(12) = 54` | `2*K(12) = 126` | **180** |
| `D2` | `2*K(12) = 126` | `2*K(12) = 126` | **252** |

These counts are an implementation invariant, not a speed claim. Carry
propagation, mask selection, addition, register pressure, instruction issue,
and memory behavior remain part of the measured cost.

For evaluating complete short-Weierstrass formulas, the corresponding generic
field-operation model is:

| Formula | Field operations |
| --- | --- |
| homogeneous + affine | `5M + 3D2 + 18A` |
| homogeneous + homogeneous | `6M + 3D2 + 24A` |
| homogeneous doubling | `4M + 2S + D2 + 12A` |

Here `A` is one L4 add or subtract. These are accounting targets for a direct
implementation of Algorithms 7--9 from the complete addition formulas of
Renes, Costello, and Batina; they must be checked symbolically and by
differential tests rather than inferred from the table.

## Straight-line generator contract

The Metal source must be generated deterministically by `build.rs` as a
depth-first, in-place, straight-line schedule.

- Emit all 63 base products for a 12-limb product and all 54 for a square
  exactly once. A generated `M`, `S`, and `D2` must therefore contain exactly
  189, 180, and 252 base products respectively.
- Write `z0` and `z2` directly into their final wide destinations where their
  lifetimes permit it. Reuse dead operand slots for absolute differences and
  keep only one guarded cross-term scratch region.
- Preserve the low-half cancellation carry before overwriting `T_low`, and
  retain `T_high` until every required high `m*p` term has been accumulated.
- Use fixed scalar names and fixed indices. The emitted hot path must contain
  no recursion, arrays, dynamic subscripting, inner loops, out-of-line multiply
  helper, SIMD shuffle, or threadgroup scratch.
- Emit a stable source fingerprint and the operation counts so tests can fail
  on generator drift.

A full flattened source scan of the current expanded helper reaches 70 word
slots, so liveness is a design constraint rather than cleanup work. The generated
source must remain at or below 56 simultaneously live `u32`-equivalent values;
count a `ulong` as two. Source liveness is only an early reject. AIR inspection
must show no `alloca`, private array, dynamic subscript, out-of-line multiply
helper, SIMD shuffle, or threadgroup scratch in the scalar field kernel.
Compiler statistics or a GPU capture must separately show zero physical
spill/fill traffic for every accepted width and orientation; the absence of an
AIR `alloca` alone is not sufficient evidence.

## Correctness gate

Before performance testing, a portable exact-integer model and the generated
Metal implementation must pass all of the following:

- Exercise canonical representatives and every valid `x + k*p`, `k = 0..3`,
  with explicit boundaries `0`, `1`, `p - 1`, and `4p - 1`.
- Cover every recursive split and carry/borrow boundary, both subtractive
  Karatsuba signs, and maximum guard-limb propagation.
- Check exactly that `(T + m*p) mod R == 0` before the radix division.
- Check raw bounds: `M` and `S` below `3p`, `D2` before L4 normalization below
  `5p`, and every public L4 result below `4p`.
- Run long dependent `M`, `S`, `D2`, add, and subtract chains. Canonicalize the
  result and compare raw Montgomery limbs with BLST.
- Differential-test the point formulas with identity, equal, inverse,
  zero-`Y`, signed affine, rescaled homogeneous, random subgroup, and on-curve
  non-subgroup inputs. Compare canonical final points with BLST.
- Cover in-place aliases, zero- and one-element launches, ragged lengths,
  dispatch tails, untouched canaries, ABI layout, source fingerprint, and all
  fail-closed error paths.

The gate permits no CPU arithmetic fallback.

## Field throughput gate

The decision benchmark measures complete valid L4 Montgomery operations, not
raw products or canonical field operations.

The first latency falsifier may use a dependent accumulator with one invariant
right operand so it remains comparable with the named-FIOS gate. That shape is
optimistic: the compiler may hoist right-operand difference work. Failure still
rejects the candidate, but a pass advances only to a second stream gate in which
both operands vary per operation. Point/formula work requires both gates.

- Use 131,072 independent fields.
- Execute 32 dependent operations per field per dispatch and four dispatches
  per command buffer: 16,777,216 operations per retained sample.
- Run one scout warmup and three scout samples for every legal thread width and
  operand orientation.
- For the selected configuration, run two decision warmups followed by 20
  retained samples.
- Keep allocation, upload, oracle work, canonical readback, and validation
  outside the GPU timestamp.

The candidate passes only if all of these hold:

```text
median throughput             >= 2.0 billion L4 operations/s
nearest-rank p5 throughput    >= 1.8 billion L4 operations/s
source live u32 equivalents   <= 56
physical spill/fill traffic   == 0
```

Every retained sample must still complete final canonicalization and BLST
equality checking. Any mismatch, range violation, REDC cancellation failure,
spill, or gate miss is `NO_GO` for this schedule. A fixed-multiplier pass is not
reported as general field throughput until the dynamic-operand stream also
passes the same bounds.

## Decision boundary

Passing the exact-integer model is necessary but does not justify point work.
Passing the product-count and source-liveness checks is also necessary but does
not predict shader throughput. Only a spill-free generated shader that clears
the correctness and field-throughput gates advances this candidate to formula
and MSM integration.

Primary formula reference: [Complete addition formulas for prime order elliptic
curves](https://eprint.iacr.org/2015/1060.pdf).
