# PMNS/Karatsuba complete-field candidate

Status: **portable arithmetic prerequisite passed; Metal implementation not yet
admitted**.

This is the first materially different field representation after the scalar
FIOS and blocked-radix results. It does not tune algorithm v3. A field element
is represented by 25 signed coefficients in

```text
(Z / pZ)  <-  Z[X] / (X^25 - 2), evaluated at gamma
```

and coefficient reduction uses a Montgomery-like radix `phi = 2^32`. Carries
do not propagate between the 25 coefficients. That replaces the measured
12-round limb dependency with three polynomial products modulo the sparse
external polynomial:

```text
C = A * B mod (X^25 - 2)
Q = C * M' mod (X^25 - 2, 2^32)
S = (C + Q * M mod (X^25 - 2)) / 2^32
```

Here `M(gamma) = 0 mod p` and `M' = -M^-1` in
`(Z / 2^32 Z)[X] / (X^25 - 2)`. This is a PMNS Montgomery product: evaluation
of `S` is evaluation of `A` times evaluation of `B` times `2^-32` modulo `p`.

The construction follows the public PMNS lattice and Montgomery-reduction
method described in Dosso et al.,
[PMNS arithmetic for elliptic curve cryptography](https://eprint.iacr.org/2025/467),
but uses independently generated 32-bit parameters. No source from the
authors' unlicensed reference repository is copied or linked.

## Frozen parameters

```text
n                         = 25
E                         = X^25 - 2
phi                       = 2^32
delta                     = 2
rho                       = 410,240
det(multiplication by M)  = -19,431 * p
required phi bound        = 361,830,798 < 2^32
```

The evaluation point is the unique root selected by
`gamma = 2^(25^-1 mod (p-1)) mod p`:

```text
511066730143548652923330221406862432842513716081412796986393940547425281101465886366786490203512647746495335967402
```

The reduction polynomial is:

```text
[5265, -6042, -550, -3841, -46309,
 -3691, -13917, 54, 11281, -18175,
 -73, 6377, 11698, -9308, 1330,
 2745, -40, 8179, -3550, -11129,
 -22495, 9238, 6654, 3275, 2537]
```

It was selected deterministically from an `fpylll 0.6.4` LLL reduction with
`delta=0.99` and `eta=0.501`. The package-owned derivation script rebuilds the
lattice from `p` and `gamma`, selects the odd-constant reduced row with the
smallest multiplication-matrix one-norm, derives `M'` by Newton lifting, and
checks every frozen invariant. The Rust model independently checks the lattice
root, inverse, exact determinant, coefficient bound, and field identity.

## Frozen arithmetic graph

The selected unbalanced recursive Karatsuba graph for two 25-coefficient
polynomials contains 205 scalar coefficient products. Using that graph for all
three polynomial products gives 615 coefficient products:

```text
A * B     = 205
C * M'    = 205
Q * M     = 205
              ---
              615
```

This is an exact count for the frozen graph, not an algebraic lower bound:
constant products may be strength-reduced and other bilinear schedules remain
possible. Of the 615 coefficient products, 410 belong to exact signed widened products
and 205 belong to the wrapping low-word quotient product. They are not one
homogeneous Metal instruction class. The earlier observable M1 Max wrapping
`u32` multiply-add probe measured
1.900899837T median and 1.822507737T p5 scalar products/s. Dividing those
different-schedule measurements by 615 gives favorable prioritization heuristics
of 3.091B median and 2.963B p5 equivalents/s. Those figures are neither a
necessary arithmetic bound nor a PMNS forecast. A generated schedule must
separately measure its actual widened and wrapping instruction mix. Karatsuba
recombination, signed accumulation, coefficient exchange, reduction,
representation traffic, and compiler behavior remain unmeasured.

The representation is 100 bytes per field element, versus 48 bytes for the
existing Montgomery ABI. Both inputs alone occupy 50 `u32` words, so a scalar
one-thread implementation has no credible live-state margin. A Metal successor
must therefore be cooperative and independently gate per-lane liveness and
shuffle cost. This is an intentional architecture change, not permission to
hide arrays or spill state to private memory.

## Admission gates

Before generated Metal source is accepted:

1. the exact portable model must pass lattice, inverse, determinant, lazy-range,
   randomized, boundary, and repeated-chain checks;
2. a generated cooperative schedule must prove complete coverage of all three
   205-product graphs and preserve `X^25 = 2` exactly;
3. conventional and conservative generated-source liveness must be at most 56
   `u32`-equivalent words **per participating lane**;
4. a favorable observable issue scout for the actual generated widened/wrapping
   and linear-operation mix must reach at least 2.4B median and 2.2B p5 complete
   schedule equivalents/s; and
5. no production API, point formula, router, or MSM code is exposed.

If admitted, the complete Metal decision gate remains unchanged: BLST-backed
evaluation equality, all widths and ragged lengths, both operands loop-varying,
two warmups, twenty samples, at least 2.0B median and 1.8B nearest-rank p5
complete PMNS field operations/s. Conversion is outside the dependent field
chain because resident point coordinates remain PMNS; point traffic and
conversion costs return in the later formula and authentic-MSM gates.

Any prerequisite or throughput miss is `NO_GO` for this representation.
