# Carry-normalized FP32 Montgomery gate

Status: **tested eight-`float4` schedule rejected; do not implement the full
kernel on this mapping**.

This is the second arithmetic candidate for a standalone BLS12-381 Metal MSM.
It changes the first candidate's radix and, critically, normalizes carries at
every Karatsuba boundary. That reduced the modeled FP32 product floor enough to
justify one native feasibility kernel. The measured schedule failed its
predeclared necessary floor, so it does not authorize a field, G1, or MSM API.
This is a schedule decision, not a proved upper bound over every possible FP32
mapping.

## Arithmetic shape

A 384-bit Montgomery value uses 32 balanced radix-4096 digits in
`[-2048, 2047]`. At every Karatsuba node:

1. split the normalized inputs into low and high halves;
2. normalize each low-plus-high sum back to balanced radix 4096 before its
   recursive product;
3. normalize every child product immediately; and
4. recombine normalized integer digits, then normalize again.

Every recursive value carries at least two guard digits. Recombination must
retain shifted high terms until their exact cancellation is complete; truncating
each child to the nominal `2n` digits before recombination is incorrect.

A normalized half-sum can gain one carry digit. Therefore an idealized
`3^4 * 4^2` count is invalid. For `m = ceil(n/2)` and `h = floor(n/2)`, the
product recurrence is:

```text
P(n, 0) = n^2
P(n, d) = P(m, d-1) + P(h, d-1) + P(m+1, d-1)
```

For 32 digits and four levels, the 81 leaves contain 36 length-two, 31
length-three, and 14 length-four products:

```text
36 * 2^2 + 31 * 3^2 + 14 * 4^2 = 647 products / convolution
3 * 647                              = 1,941 products / complete Montgomery Fp
```

The factor of three retains the complete construction: `a*b`, the low-half
Montgomery correction product, and the correction-times-modulus product before
the exact radix division and fixed final subtraction.

## Exactness boundary

Every normalized digit is an exact FP32 integer. A product has absolute value
at most `2048^2 = 2^22`; a four-term leaf coefficient is at most `2^24`, the
inclusive consecutive-integer limit for FP32. The kernel must therefore:

- compile without fast math or reassociation;
- accumulate no more than four leaf products before normalization;
- convert the exact coefficient to signed integer form immediately; and
- perform carry propagation and Karatsuba recombination with checked integer
  bounds before converting normalized digits back to FP32; and
- normalize signed integers, not merely residues modulo the current radix power.

Hitting the inclusive limit leaves no room for an extra add. Shader-side ABI
probes and boundary vectors must detect any compiler or schedule change that
violates this shape.

## Product roofline

At the historical Ledger-9 gate:

```text
2.0B complete Fp/s * 1,941 products = 3.882T products/s
3.882T / 5.2T modeled M1 Max ceiling = 74.66%
product-only complete-Fp ceiling      = 2.679B/s
```

This is a necessary improvement over the rejected 2,916-product candidate, but
only 25.34% of modeled product-issue capacity remains for every normalization,
recombination, conversion, reduction, register, traffic, and scheduling cost.
The exact schedule audit counts 1,307 arithmetic normalization steps and 1,468
emitted store-or-check slots per convolution in addition to the 647 products.
A straightforward depth-first implementation also needs about 286 temporary
digit slots beyond its inputs and output. That storage makes a one-thread
implementation likely to spill; a cooperative implementation instead introduces
carry dependencies and synchronization. This is why the product roofline was
never a performance forecast.

For the Montgomery correction, the balanced correction factor represents the
standard value modulo `R = 2^384` but may be negative. After proving all 32 low
digits of `t + m*p` are zero, the divided value lies in `(-p, 2p)`: add `p` when
negative, then conditionally subtract `p`. A single unsigned final subtraction
is insufficient.

## Predeclared native gates and result

The first standalone Metal diagnostic must use one retained device/queue,
compile precise FP32, allocate and upload outside the timed region, submit one
command and wait once per sample, and make every accumulator observable.

The first stage intentionally measured a more favorable operation than the real
algorithm: eight observable `float4` FMAs per iteration, or 32 scalar product
issues, with no carry, normalization, Karatsuba, Montgomery, or routing work.
Dividing that raw issue rate by 1,941 produces only a **product-count
equivalent**, never complete Fp throughput.

After two warmups and twenty samples on the M1 Max, the necessary first-stage
bounds were:

1. the raw precise-FP32 schedule must reach at least **2.40B product-count
   equivalents/s median** and **2.25B/s observed p5 throughput**, corresponding
   to 4.6584T and 4.36725T raw products/s;
2. deterministic shader outputs must match the same precise-FMA CPU oracle; and
3. pipeline width, command errors, and raw timings must be recorded inside this
   package.

The 2026-07-15 result was:

| Metric | Observed | Required | Fraction of gate |
| --- | ---: | ---: | ---: |
| median raw products/s | 1.945454T | 4.6584T | 41.76% |
| p5 raw products/s | 1.827160T | 4.36725T | 41.84% |
| median 1,941-product equivalents/s | 1.002295B | 2.40B | 41.76% |
| p5 1,941-product equivalents/s | 0.941350B | 2.25B | 41.84% |

The median raw issue rate is only 37.41% of the optimistic 5.2T modeled product
ceiling. The shader passed its ABI and bit-exact output oracle, used one retained
command with four dispatches and one wait per sample, and selected 64 threads per
threadgroup. The historical decision run used only one scout sample per width
and did not record compiler register/spill metadata. Its runner also selected
the second-smallest sample as “p5”; the table corrects that label by recomputing
nearest-rank p5 (the minimum for twenty samples) from the unchanged raw samples.
The correction does not change the failed decision. Those limitations prevent
using the result as a global FP32 ceiling; reopening this mapping requires a
saturation sweep and still needs a 2.39x median schedule improvement before any
omitted work. The full raw samples and toolchain identity are frozen in
[`evidence/m1-max-precise-fp32-product-issue-2026-07-15.json`](evidence/m1-max-precise-fp32-product-issue-2026-07-15.json).

Failure stops this mapping before full Montgomery implementation. The unrun
second stage would have included every normalization, guard, divisibility check,
signed correction, and final reduction; BLST differentials; and gates of 2.0B
complete reduced Fp/s median and 1.8B/s observed p5. Those numbers remain useful
requirements for a materially different arithmetic engine, but they are not
results for this candidate.

The production library surface remains fail closed. The diagnostic shader and
host harness are retained only as reproducible schedule evidence behind the
`metal-diagnostics` feature. The materially different two-lane cooperative
integer successor was subsequently implemented as a complete reduced
Montgomery operation and also rejected: it reached 0.987B median and 0.940B p5
complete Fp/s against 2.0B/1.8B gates. See
[`TWO_LANE_U32_COMBA_GATE.md`](TWO_LANE_U32_COMBA_GATE.md). That result closes
the two-lane FIPS mapping, not every integer schedule.
