# AMNS8 `mulhi` primitive result

Status: **PRUNED as a prioritization hypothesis: replacing widened products
with explicit `low + metal::mulhi` is not a formal AMNS8-v2 complete-field
`NO_GO`; hierarchical arithmetic remains untested**.

The optimized Apple AIR and two package-owned runtime probes agree on the
decision for this Apple M1 Max and compiler. An explicit complete unsigned
32x32-to-64 product remains two operations in AIR:

```text
low  = mul i32
high = air.mul_hi.u.i32
```

The widened expression remains one `mul nuw i64` followed by low/high
extraction. The compiler does not canonicalize either form into the other.
Neither timed kernel has an alloca or threadgroup-memory reference, and its
loop-carried result is stored, so the product work is observable.

## Frozen measurements

Both populations use 20 alternating pairs after three warm-up pairs. Medians
average the two center observations; throughput is the median of the 20
per-sample rates. P5 is nearest-rank over rates, which is the minimum for 20
samples. All GPU outputs matched exact CPU `UInt64` oracles.

| Population | Form | median GPU time | median complete products/s | p5 products/s |
|---|---|---:|---:|---:|
| Constant RHS, 8.590B products/sample | explicit low + `mulhi` | 29.343 ms | 292.740B | 221.763B |
| Constant RHS, 8.590B products/sample | widened `ulong` | 25.055 ms | 342.844B | 332.479B |
| Rotating two-state, 34.360B products/sample | explicit low + `mulhi` | 160.760 ms | 213.761B | 172.516B |
| Rotating two-state, 34.360B products/sample | widened `ulong` | 152.012 ms | 226.033B | 176.578B |

For the paired ratio `widened_time / mulhi_time`, the constant-RHS median is
`0.851027`: widened time is 14.90% lower, or explicit low-plus-`mulhi` takes
about 17.50% longer. The rotating-two-state median is `0.950362`, favoring
widened multiplication by 4.96%. The rotating population is noisy
(`0.597659..1.260054`), so it is used only to establish that varying both
operands reveals no material `mulhi` advantage. It is not used as a hardware
ceiling.

Correctness coverage is:

- 16,384 direct products, including the Cartesian product of seven boundary
  values and 16,335 deterministic random pairs;
- 257 constant-RHS chains of 19 iterations against a CPU oracle; and
- 257 rotating two-state chains of 37 iterations against a CPU oracle, with
  `left_next = right` and `right_next = mix(low, high)`.

## Required-rate comparison

The direct AMNS8 topology issues 224 widened product slots per complete field
multiply. Meeting the frozen 2.0B-field/s gate therefore asks that topology to
sustain:

```text
2.0B fields/s * 224 widened slots/field = 448B widened products/s
```

That is before its 64 wrapping-product slots per field, shuffles, additions,
folds, carries, loads, stores, and control. The favorable constant-RHS scout
measures 342.844B/s for the widened expression and 292.740B/s for explicit
low-plus-`mulhi`.

This is prioritization evidence, not a strict device roofline: the scout has a
constant second operand and is not a published hardware limit or a proof that
every possible schedule stays below 448B/s. The actionable observation is
simpler and stronger for the proposed edit: explicit `mulhi` makes the
favorable primitive chain slower and provides no material benefit when both
operands rotate. It therefore cannot credibly rescue the measured direct
field kernel's approximately 294x miss against 2.0B field operations/s.

The result prunes this substitution only. A materially different hierarchical
schedule, representation, compiler generation, or Apple GPU remains a new
hypothesis requiring its own gate.

## Reproduction and evidence

The complete raw samples, exact statistics, correctness corpus, AIR counts,
device/toolchain contract, and source/artifact hashes are frozen in
[`evidence/m1-max-metal-mulhi-primitive-2026-07-16.json`](evidence/m1-max-metal-mulhi-primitive-2026-07-16.json).

Run both package-owned probes with:

```sh
sh research/metal_mulhi_probe/run.sh
```

The runner compiles with the crate-aligned flags
`-std=metal3.0 -mmacosx-version-min=13.0 -fno-fast-math -O3`. Sources are in
[`research/metal_mulhi_probe`](research/metal_mulhi_probe).
