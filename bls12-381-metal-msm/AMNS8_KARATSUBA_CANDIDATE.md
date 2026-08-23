# AMNS8/Karatsuba complete-field candidate

Status: **portable prerequisites passed; direct generated eight-lane Metal
topology measured `NO_GO` on M1 Max**.

The original contract below is retained as the design record. The completed
decision, including all raw samples, is in
[AMNS8_KARATSUBA_RESULT.md](AMNS8_KARATSUBA_RESULT.md). The measured 6.798M
median and 6.722M p5 complete field operations/s miss the frozen 2.0B/1.8B
gates by approximately 294x/268x. No point or MSM work is authorized by this
candidate on the measured device.

The subsequent explicit-`mulhi` v2 preserves the v1 output corpus byte-for-byte
and retains its 32/33-word source liveness, but primitive scouts found no
repeatable material benefit over widened multiplication. That narrow
substitution is therefore pruned against the existing 294.188x field gap. It
was not a formal v2 complete-field gate and does not close materially
hierarchical arithmetic; see
[AMNS8_MULHI_PRIMITIVE_RESULT.md](AMNS8_MULHI_PRIMITIVE_RESULT.md) and the
[raw primitive evidence](evidence/m1-max-metal-mulhi-primitive-2026-07-16.json).

This is the compact successor to the 25-coefficient PMNS experiment.  A field
element has eight signed 64-bit coefficients in

```text
(Z / pZ)  <-  Z[X] / (X^8 + 2), evaluated at gamma
```

and coefficient reduction uses the Montgomery-like radix `phi = 2^64`:

```text
C = A * B mod (X^8 + 2)
Q = C * M' mod (X^8 + 2, 2^64)
S = (C + Q * M mod (X^8 + 2)) / 2^64
```

Here `M(gamma) = 0 mod p`, `M' = -M^-1` in
`(Z / 2^64 Z)[X] / (X^8 + 2)`, and `gamma^8 = -2 mod p`.  Therefore
evaluation of `S` equals evaluation of `A` times evaluation of `B` times
`2^-64` modulo `p`.

The package-owned Python derivation and Rust arithmetic model are standalone:
they do not depend on ledger, proof-system, CUDA, sppark, or zkMetal source.
They are deliberately shaped so this field engine can later sit inside an
independently open-sourceable Metal MSM library.

## Frozen parameters

```text
n                              = 8
E                              = X^8 + 2
X^8                            = -2
phi                            = 2^64
delta                          = 10
sigma                          = 143,766,863,997,100
rho                            = 32*sigma
                               = 4,600,539,647,907,200
lazy input limit               = 11*(rho-1)
                               = 50,605,936,126,979,189
capacity floor                 = floor(phi/(32*rho)) = 125
required lazy capacity         = (delta+1)^2 = 121
det(multiplication by M)       = 33*p
```

The frozen evaluation point is:

```text
gamma = 0x89413f1d151a1a6e4666ea4ed3dc47ab1cb92213acc505243052888f88ccf35f60b63def51da8deac4a8cbd2f9ae056
      = 1320341115756334504903338613189055857314888584589395389559536083285462048544550127191972802724328330560637937311830
```

The reduction polynomial, in ascending coefficient order, is:

```text
[80042099451959,
 -438404363789,
 -104528439513579,
 91613899373051,
 -7504120839914,
 125847515758073,
 -143766863997100,
 -95095973594275]
```

Its frozen negative inverse modulo `2^64` is:

```text
[0x0010c6fb6f1f4907,
 0x7a708f072a6fc213,
 0xe6af80b15b4d0bc4,
 0xa5eeda87343cdd86,
 0x516fcfe42de2b795,
 0x33826e370315ed0a,
 0xc7af1c653afff478,
 0x1db4899b375f66c0]
```

## Independent derivation and coverage certificate

`research/derive_amns8_basis.py` reconstructs the conventional kernel lattice
from `p` and the frozen root, then runs `fpylll 0.6.4` with
`LLL(delta=0.99, eta=0.501)`.  It freezes all eight output rows, not only the
chosen polynomial.  Among rows with odd constant coefficient, the selected row
uniquely minimizes `(infinity norm, one-norm, coefficient tuple)` and is frozen
exactly as emitted.  The script derives `M'` by Newton lifting rather than
accepting it as an input.

The complete reduced basis has exact determinant `p`.  Its largest row and
column one-norms are:

```text
max row one-norm       = 677,018,084,045,712
max column one-norm    = 843,285,565,234,751
```

The coefficient box has substantial coverage margin:

```text
2*rho / max row one-norm       = 13.5905...
2*rho / max column one-norm    = 10.9109...
```

In particular, the half-open fundamental parallelepiped of the row lattice
fits coordinate-wise inside the `rho` box, while `det(basis)=p` proves that the
lattice has exactly the required index.  Thus every base-field residue has an
eight-coefficient representative in the frozen box.  This is a representation
coverage certificate; it is not yet a constant-time conversion algorithm.

The earlier `670,478,210,175,899 / 811,732,356,571,504` norm pair came from a
`delta=0.75` scouting basis.  It is intentionally not used by this candidate.

## Exact 27-leaf K8 graph

For two length-eight coefficient vectors, balanced recursive Karatsuba has
`3^3 = 27` scalar leaves.  The complete AMNS product has three such polynomial
products:

```text
A * B       27 coefficient products
C * M'      27 coefficient products
Q * M       27 coefficient products
            --
            81 coefficient products
```

An exact signed 64-by-64 leaf is decomposed by a second two-way Karatsuba step.
Writing each magnitude as two `u32` halves, it uses the low product, high
product, and product of the signed half-differences.  Exact widened
recombination gives three `u32` products per scalar leaf, including the
`i64::MIN` boundary.  A wrapping low-64-bit product also needs exactly the
three relevant `u32` products.  The frozen graph therefore contains:

```text
3 polynomial products * 27 leaves * 3 mixed u32 products = 243
```

This count is concrete for the modeled graph, including `Q*M`.  It is not based
on assuming that quotient-transform sums happen to fit `i64`:

```text
q_transform = q_low + q_carry * 2^64
q_low        is centered i64
|q_carry|    <= 4 for a sum of at most eight centered coefficients
```

Every `Q*M` leaf computes `q_low*m_transform` with the exact three-product
signed-64 decomposition, then adds
`(q_carry*m_transform)<<64`.  The carry multiplier is in `[-4, 4]` and is
implemented as signed small-constant arithmetic, not another general scalar
product.  Every frozen `M` transform fits signed 64 bits; its largest possible
magnitude is bounded by `sum(abs(M_i)) = 648,837,316,891,740`.  The Rust model
checks this recomposition on worst-case all-`i64::MIN`, all-`i64::MAX`, random,
boundary, lazy-range, and repeated-chain inputs.

The 243 count excludes additions, subtractions, external folds, the small carry
corrections, coefficient exchange, and final exact divisions by `2^64`.  It is
an arithmetic graph count, not a claim that every operation has the same Metal
issue cost.

The product classes are also not homogeneous:

```text
A * B exact signed leaves                  81 widened u32 products
C * M' low*low part                        27 widened u32 products
C * M' two low-word cross parts            54 wrapping u32 products
Q * M exact range-managed leaves            81 widened u32 products
                                             ---
                                             189 widened + 54 wrapping = 243
```

Those are logical graph products.  They must be reported separately from SIMD
slot issue.  The simplest eight-lane mapping needs four waves for 27 leaves, so
each polynomial convolution executes 32 lane slots unless optimized assembly
eliminates the five inactive slots.  Its conservative physical accounting is
therefore 224 widened plus 64 wrapping products, 288 total, per field multiply.
The benchmark reports both counts and uses completed field operations—not
logical products—as its decision metric.

Eight lanes was selected as the safest initial liveness mapping. It passed the
source gate at 33 allocation-time words but failed throughput by approximately
294x. Four lanes (two coefficients per lane) and two lanes (four coefficients
per lane) would reduce conservative product slots only from 288 to 252, a
maximum product-wave ratio of 1.143x before their additional per-lane state.
That cannot materially change the decision, so the simple lane-packing sweep is
pruned. A future retest must use materially different scheduling or hardware;
it is not open-ended tuning of this topology.

## Closure and 128-bit safety

For `X^8=-2`, an externally reduced output coefficient has absolute-weight
factor at most:

```text
1 + 7*2 = 15
```

For lazy inputs bounded by `B = 11*(rho-1)` and centered quotient coefficients,
the portable certificate is:

```text
|C_i| bound              = 15*B^2
                         = 38,414,411,569,318,461,597,832,836,586,465,815

|(Q*M)_i| bound          = 15*2^63*sigma
                         = 19,890,229,098,257,336,436,485,208,932,352,000

combined numerator bound = 58,304,640,667,575,798,034,318,045,518,817,815
ceil(bound/2^64)         = 3,160,700,903,888,619 < rho
```

This proves coefficient closure for every lazy input in scope, not only for the
random cases in the model.

The recursive implementation also checks a deliberately conservative bound on
temporary values.  If a K8 input coefficient is bounded by `A` and the other by
`B`, every held convolution temporary is bounded by `512*A*B`; folding
`X^8=-2` costs at most another factor of three.  Across `A*B`, range-managed
`Q*M`, and their final addition, this gives:

```text
5,970,395,204,359,761,718,714,167,861,126,944,256 < 2^127
```

Thus the scalar oracle can use checked `i128` throughout.  A Metal generator
must lower these widened values into named `u32` words; the proof does not
assume native fast 128-bit GPU arithmetic.

## Complete-formula compatibility

The capacity bound was checked against the intended complete point formulas,
not chosen only for isolated field multiplication.  A fixed isomorphism maps
BLS12-381 G1 from `y^2 = x^3 + 4` to `y^2 = x^3 - 3`.  With

```text
kappa = 0x18cccb8f6dbc6d2c6a3f657c444c53a4aba6efa5332d0d60549e1bc9ae14c29be771dd57afac430286a29ef641420cc6
```

the package-owned oracle verifies `kappa^6 = -4/3 mod p`; the forward map is
`(x, y) -> (x/kappa^2, y/kappa^3)` and the inverse multiplies by those powers.
For homogeneous coordinates on the transformed curve, define

```text
A = X1*X2                   D = X1*Y2 + Y1*X2
B = Y1*Y2                   E1 = Y1*Z2 + Z1*Y2
C = Z1*Z2                   F = X1*Z2 + Z1*X2
```

Expanding complete RCB Algorithm 7 at `b=-3` gives:

```text
X3 = D*B + 9*D*C + 9*E1*F       weight 19
Y3 = B^2 - 81*C^2 - 27*F*A      weight 109
Z3 = E1*B - 9*E1*C + 3*A*D      weight 13
```

The largest weight, 109, is below the AMNS capacity 125.  The portable test
checks the isomorphism, the expanded sequence against the existing RCB
schedule, identity/equal/inverse cases, noncanonical homogeneous scales, and
random subgroup pairs against an independent affine oracle.  This is only a
candidate-compatibility proof: complete point Metal work remains closed until
the field gate passes.

## Performance result

No valid complete-field rate could be derived by dividing the old
wrapping-`u32` probe by 243: the authentic graph has 189 widened and only 54
wrapping products, in addition to its non-product work. A named-FIOS
instruction-scaling heuristic had suggested roughly:

```text
median    1.888B complete field operations/s
p5        1.796B complete field operations/s
```

That heuristic was not a forecast or admission result. The authentic generated
kernel is decisive: it reached 6.798M median and 6.722M nearest-rank p5 complete
field operations/s, versus the 2.0B/1.8B gates. The result includes the signed
recombination graph, range-managed carry path, K8 additions, shuffle/exchange,
and exact coefficient accumulation.

An explicit-`mulhi` v2 tested whether changing only widened leaf multiplication
could rescue this topology. Its fingerprint is `4cddfae73658b39f`; it preserves
v1 output byte-for-byte, retains 32 last-use and 33 allocation-time source
words, and compiles to 28 `mulhi` calls plus 36 variable `i32` multiplies, zero
variable `i64` leaf multiplies, and 494 shuffles per lane. The constant-RHS
primitive scout measured 292.740B/s explicit versus 342.844B/s widened, with
explicit 17.50% longer at the paired median. A higher-variability rotating scout
measured 213.761B/s versus 226.033B/s and a paired widened/mulhi time ratio of
0.950362. This shows no repeatable material primitive benefit and prunes the
substitution; it is not a formal v2 field result or hardware roofline.

An AMNS8 field element occupies 64 bytes (`16 u32` words), versus 48 bytes for
the existing Montgomery ABI.  In the required eight-lane-per-field layout, each
lane persistently owns two words per field, so two inputs plus one output are six
words per participating lane—not 48.  This cooperative topology keeps the
persistent state compact. The generated source's typed and tokenized scans both
report 32 last-use words and 33 allocation-time words, below the 56-live-word
**per-lane** gate. Available Apple tooling did not expose final physical
register allocation or spill/fill traffic, so no physical no-spill claim is
made.

## Standalone-library boundary and next gate

The Metal implementation should preserve the current separation:

1. a reproducible parameter generator and machine-readable frozen constants;
2. a pure Rust checked oracle with no platform or ledger dependencies;
3. generated, straight-line MSL using named words and the exact frozen K8 graph;
4. a narrow conversion layer between the 48-byte external Montgomery ABI and
   resident 64-byte AMNS8 coordinates; and
5. Metal diagnostics and benchmarks outside the arithmetic API.

The field module should expose representation and multiplication primitives,
not proof routing or Midnight-specific commitment types.  That keeps it usable
as the field engine of a separately published Metal MSM library.

The bounded Metal experiment completed these steps:

1. generate all three straight-line K8 products, including the explicit
   `q_low + q_carry*2^64` path;
2. prove generated-source coverage and exact agreement with the Rust oracle and
   BLST-backed field evaluation for boundaries, random cases, ragged workloads,
   and repeated dependent chains;
3. demonstrate at most 56 live `u32`-equivalent words per participating lane,
   with no source-private arrays, while recording physical spill status as
   unknown if the available tooling cannot expose it;
4. measure the actual widened/wrapping/add/carry mix with both operands varying;
   and
5. apply the unchanged complete-field gate of at least 2.0B median and 1.8B p5
   operations/s before any point formula or MSM integration begins.

Steps 1--4 passed. Step 5 failed at 6.798M median and 6.722M p5, so the
diagnostic source is retained only as reproducible decision evidence and is not
admitted as production field arithmetic.

Reproduction uses Rust 1.95 and the pinned Python package:

```text
PYTHONPATH=<fpylll-install> python3 research/derive_amns8_basis.py
cargo +1.95.0 test --test amns8_karatsuba_model
cargo +1.95.0 test --test amns8_generator
```

The exact benchmark command and its intentionally fail-closed outcome are in
[AMNS8_KARATSUBA_RESULT.md](AMNS8_KARATSUBA_RESULT.md). AMNS8 remains a
reproducible research candidate, not production arithmetic.

The independent primitive comparison and its raw evidence are in
[AMNS8_MULHI_PRIMITIVE_RESULT.md](AMNS8_MULHI_PRIMITIVE_RESULT.md) and
[evidence/m1-max-metal-mulhi-primitive-2026-07-16.json](evidence/m1-max-metal-mulhi-primitive-2026-07-16.json).
