# Packed-FP32 complete-Montgomery gate

Status: **structurally rejected before shader implementation**.

This report evaluates one specific candidate for BLS12-381 Fp multiplication
on an Apple M1 Max. It does not claim that every FP32, integer, or Metal field
engine is impossible. It says this operation count cannot meet the declared
2.0-billion/s gate, so implementing and timing it would be dead-end work.

## Declared target

The parent MSM program's bare arithmetic requirement is approximately 1.713
billion complete Fp operations per second. Its pre-G1 engineering gate is 2.0
billion/s so a field engine has some headroom before point formulas, scalar
routing, memory traffic, synchronization, and command overhead. This assessment
uses that established 2.0-billion/s gate. Even against the bare requirement, the
candidate's 1.783-billion/s product-only ceiling would leave only about 4.1% for
every omitted cost.

## Candidate arithmetic

A reduced BLST-compatible Fp value is a 384-bit Montgomery integer. Split it
into 48 radix-256 digits and recode them into balanced digits in `[-128, 127]`.
A three-level Karatsuba split produces 27 length-six leaf convolutions. Each
leaf uses 36 scalar FP32 products, so one 48-by-48 convolution uses:

```text
27 * 6 * 6 = 972 scalar FP32 products
```

A complete Montgomery multiplication—not merely an unreduced polynomial
convolution—needs three such products in this construction:

1. `t = a * b`;
2. `m = (t_low * (-p^-1 mod 2^384)) mod 2^384`;
3. `u = (t + m * p) / 2^384`, followed by one fixed subtraction of `p`.

Balanced recoding of a 384-bit intermediate may represent `x - 2^384`.
The complete design would track that carry and add the corresponding shifted
multiple of `p`; dropping it would not be a Montgomery multiplication. Signed
carry propagation must also prove that the low 48 result digits are zero before
the exact division. These requirements are included in the three-convolution
count; no incomplete convolution is credited as a field multiplication.

Thus the optimistic product floor is:

```text
3 * 972 = 2,916 scalar FP32 products / complete Fp multiplication
```

## Exactness bounds

FP32 represents consecutive integers through `2^24 = 16,777,216`. With
balanced radix-256 digits:

- a deepest leaf input digit is the sum of at most eight base digits, so its
  absolute value is at most `8 * 128 = 1,024`;
- a six-term leaf coefficient is at most
  `6 * 1,024^2 = 6,291,456`;
- the largest evaluation-order intermediate during the first Karatsuba
  recombination is at most `12,582,912`;
- a final 48-digit convolution coefficient is at most
  `48 * 128^2 = 786,432`; and
- including the signed carry recurrence, the value entering digit extraction
  is at most `786,432 + ceil(786,432 / 255) = 789,517`.

Every value is an integer and all listed bounds are strictly below `2^24`.
Therefore exactness is plausible under precise FP32 compilation with no fast
math or reassociation. Exactness is not the rejection reason.

## M1 Max roofline

The gate counts complete reduced Montgomery multiplications:

```text
2,000,000,000 * 2,916 = 5,832,000,000,000 scalar products/s
```

The commonly quoted M1 Max FP32 peak is about 10.4 TFLOP/s. GPU peak FLOP
figures count one fused multiply-add as two operations, so an optimistically
perfect schedule has at most about 5.2 trillion scalar product issues/s. This
gives the candidate every possible benefit: perfect occupancy, no dependencies,
no additions, no integer work, no barriers, no traffic, and no command cost.

```text
maximum complete Fp/s = floor(5.2e12 / 2,916)
                      = 1,783,264,746

required / maximum    = 1.1215384...
                      = 112.16% of the product-issue roofline
```

If the products are described as product-accumulates, the target requires
11.664 TFLOP/s, likewise above 10.4 TFLOP/s before the remaining arithmetic.
Real implementation overhead can only lower the ceiling.

## Decision

No shader, benchmark, BLST vector harness, build script, or unsafe resident
buffer API is provided for this candidate. Such artifacts would look usable
despite a proof that the declared gate cannot pass. The rejection is frozen in
`src/lib.rs` and checked by pure-Rust unit tests.

A successor may proceed only with a materially lower product count or a
different supported hardware primitive. Its roofline must count complete
Montgomery field multiplications; a raw-convolution number must never be
reported as Fp throughput.
