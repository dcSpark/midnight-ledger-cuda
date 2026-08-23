# AMNS8/Karatsuba complete-field result

Status: **`NO_GO` for the direct generated eight-lane topology on this Apple
M1 Max**.

The AMNS8 experiment passed its portable arithmetic, generated-source
correctness, and source-liveness prerequisites. Its formal rotating-operands
Metal gate then measured only:

```text
median GPU     6,798,372.789 complete field operations/s
p5 GPU         6,721,714.410 complete field operations/s
median wall    6,797,376.196 complete field operations/s
p5 wall        6,720,719.148 complete field operations/s

median gate    2,000,000,000 complete field operations/s
p5 gate        1,800,000,000 complete field operations/s
decision       NO_GO
```

The median is 0.3399% of its gate, a 294.19x miss. The nearest-rank p5 is
0.3734% of its gate, a 267.79x miss. This is not a marginal tuning failure.
Complete point formulas, bucket reduction, MSM, and resident proof integration
remain closed on this M1 Max plan.

All 20 decision samples and both GPU warmup timestamps are frozen in
[evidence/m1-max-amns8-k8-field-2026-07-16.json](evidence/m1-max-amns8-k8-field-2026-07-16.json).

## What was measured

The tested field representation has eight signed 64-bit coefficients modulo
`X^8 + 2`. The generated K8 schedule performs all three coefficient products
needed by the AMNS Montgomery-like reduction:

```text
A * B
C * M'
Q * M
```

Its logical graph contains 243 `u32` products per complete field operation:
189 widened and 54 wrapping. The direct eight-lane mapping executes 288
physical product slots because each 27-leaf K8 convolution occupies four
SIMD waves. Every benchmark iteration rotates both operands:

```text
left_next  = right
right_next = amns8_multiply(left, right)
```

This prevents a fixed right-hand operand from turning the result into a
constant-hoisting benchmark.

The decision run used:

```text
device registry id        4294970105
thread execution width    32
threadgroup width         32
threads per field         8
logical fields            131,072
iterations                32
dispatches per sample     4
operations per sample     16,777,216
decision warmups          2
decision samples          20
threadgroup memory        0 bytes
```

Allocation, upload, and readback were outside the timed region. Each sample
used one command and one wait. The exact output was checked after every warmup
and decision sample, every check passed, and the padded output canary remained
untouched.

## Prerequisites that passed

The package-owned derivation and exact Rust model pass. The generated Metal
differential corpus passes before the performance run. Independent typed and
tokenized last-use scans both report 32 live `u32`-equivalent words per lane;
their conservative allocation-time scans both report 33, below the frozen
limit of 56. The schedule fingerprint is `f8a2079cfdcc25d5`.

An independent optimized-AIR audit of the one-shot kernel found 28 variable
widened `i64` multiply sites and eight variable wrapping `i32` multiply sites
per Metal lane, matching the generated graph after lane packing. It also found
494 SIMD-shuffle calls, two global input loads, one constant load, one global
store, zero allocations, zero private/threadgroup address-space references,
and zero out-of-line calls. This rules out an obvious source-array or helper-call
artifact, but it does not prove a physical register count or absence of
spill/fill traffic in final Apple GPU ISA; the available tooling did not expose
either. Throughput is decisive regardless. Exact MSL, AIR, and metallib hashes
are frozen with the samples in the JSON evidence record.

## Explicit-`mulhi` follow-up

A generated v2 tested one narrow rescue hypothesis: replace each variable
widened leaf product with an explicit low-word `u32` multiply plus
`metal::mulhi`. It preserves the v1 outputs byte-for-byte over the retained
corpus and retains the same source-liveness result:

```text
v2 schedule fingerprint       4cddfae73658b39f
last-use source words         32
allocation-time source words  33
AIR mulhi calls/lane          28
AIR variable i32 muls/lane    36
AIR variable i64 leaf muls    0
AIR shuffles/lane             494
```

The 36 variable `i32` sites comprise the 28 explicit low-word leaf multiplies
plus the eight wrapping multiplies already present in v1. The v2 source changes
the leaf primitive; it does not eliminate the exchange topology.

Two package-owned primitive scouts then compared explicit `mulhi` with widened
unsigned multiplication:

| Population | Explicit `mulhi` | Widened | Paired result |
|---|---:|---:|---:|
| Constant RHS | 292.740B products/s | 342.844B products/s | explicit 17.50% longer |
| Rotating two-state | 213.761B products/s | 226.033B products/s | widened/mulhi time 0.950362 |

The rotating population had substantial variability, so it supports only the
absence of a repeatable material `mulhi` benefit. Together the AIR and runtime
scouts prune this leaf substitution as a credible rescue for the existing
294.188x median field-gate gap.

This is **not** a formal v2 complete-field `NO_GO`, no v2 complete-field result
replaces the v1 measurements above, and neither primitive scout is a hardware
roofline. A materially different hierarchical schedule remains untested. See
[AMNS8_MULHI_PRIMITIVE_RESULT.md](AMNS8_MULHI_PRIMITIVE_RESULT.md) for the full
method and
[evidence/m1-max-metal-mulhi-primitive-2026-07-16.json](evidence/m1-max-metal-mulhi-primitive-2026-07-16.json)
for every raw sample and artifact hash.

## Why simple two- and four-lane variants are pruned

Packing the same 27 leaves into two- or four-lane field mappings reduces the
conservative product-slot count only from 288 to 252. The most favorable ratio
is therefore:

```text
288 / 252 = 1.142857...
```

That 1.143x ratio applies only to product-wave packing, not to the complete
kernel. Even granting it as an impossible end-to-end speedup leaves an
approximately 257x median gap to 2.0B/s. Implementing those variants cannot
credibly change this decision, so they are not the next milestone.

## Scope of the decision

This result rejects one concrete, direct generated AMNS8/K8 topology on the
measured M1 Max. It does **not** prove that AMNS is globally unsuitable, that
Metal cannot implement BLS12-381 arithmetic, or that a materially different
hierarchical schedule or later Apple GPU cannot pass a separately declared
gate. Such a retest must change the execution strategy or hardware by orders of
magnitude; lane repacking alone is not a materially new experiment.

The candidate derivation remains useful research evidence, but the standalone
crate continues to expose no production field, point, or MSM execution API.

## Reproduction

The correctness run succeeds normally:

```bash
cargo +1.95.0 test --release --features metal-diagnostics \
  metal_diagnostic::validate_amns8_field_correctness \
  -- --ignored --nocapture --test-threads=1
```

On an otherwise idle Apple-silicon benchmark host, the formal gate is:

```bash
CARGO_INCREMENTAL=0 cargo +1.95.0 test --release \
  --features metal-diagnostics \
  metal_diagnostic::report_amns8_field_gate \
  -- --ignored --nocapture --test-threads=1
```

The gate intentionally exits nonzero after printing `NO_GO`; its final
assertion fails closed when either frozen throughput threshold is missed.
