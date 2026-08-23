# Two-lane cooperative `u32` Montgomery gate

Status: **NO_GO on Apple M1 Max: 0.987B median / 0.940B p5 complete Fp/s**.

This was the prioritized M1 Max field experiment for the standalone library.
The complete implementation is correct, but it misses both predeclared
throughput gates. It therefore does not authorize G1 or MSM work on this
mapping. Any continuation requires a materially different, separately gated
arithmetic or direct group-operation design.

## Why this topology

The best exact twelve-`u32` scalar CIOS diagnostic reached 1.540B complete Fp
multiplications/s against a 2.0B/s engineering requirement, a 29.8% gap.
Interleaving two and three complete scalar states regressed to 0.963B/s and
0.467B/s. Four- and sixteen-lane cooperative layouts have unfavorable ideal
issue-count bounds before carry exchange. A pair of lanes is the only narrow
topology that can materially reduce per-thread live state without multiplying
the field across too many SIMD lanes.

The rejected FP32 issue schedule would need a 2.39x improvement before carry or
reduction work. The two-lane integer route therefore has the smaller quantified
gap and receives priority.

## Candidate mapping

- One 32-lane SIMD group owns 16 independent Fp multiplications.
- Each adjacent lane pair owns one 12-limb Montgomery value. Lane parity owns
  alternating input and output limbs; Montgomery correction digits use the
  opposite parity to balance work.
- The pair performs 24-column finely integrated product scanning. Each lane
  accumulates its assigned products into an exact 96-bit value, then exchanges
  a packed `uint4` with its peer. Column 23 explicitly emits the final limb and
  carries no product term.
- `build.rs` generates 144 synchronized pair slots. At the source level, each
  slot issues one lane-local product in both lanes: 288 lane-issued slots cover
  144 full-product terms and 132 non-`p0` reduction terms, with 12 dummy slots
  balancing the lanes. The build fails unless every useful term occurs exactly
  once. There is no dynamic private array, threadgroup barrier, or device
  scratch.
- Low and high halves of every 32-by-32-bit product remain exact. The source
  uses a widening `ulong` product, and the frozen evidence records the optimized
  AIR operations emitted for that exact artifact. A truncated `uint` product is
  not sufficient.
- Montgomery correction, radix division, final conditional subtraction, and
  reduced canonical range are part of the timed operation.
- Inputs and outputs retain the package's neutral twelve-`u32` Montgomery ABI.

The implementation must compile as a package-owned shader and use the private
diagnostic system boundary. It may not import Midnight, CUDA, sppark, zkMetal,
or a CPU field fallback.

## Predeclared field gate

Run on an otherwise idle M1 Max with allocation/upload/reference/readback outside
the timed interval:

1. ABI and metallib fingerprints pass before dispatch;
2. zero, one, `p-1`, limb/carry boundaries, deterministic random pairs, repeated
   chains, and adversarial final-subtraction cases match BLST exactly;
3. two warmups and at least 20 completed-command samples are retained, with a
   nearest-rank p5, wall and GPU time, selected width, compiler register/spill
   evidence, device identity, and metallib hash;
4. median throughput is at least **2.0B complete reduced Fp/s**; and
5. p5 throughput is at least **1.8B complete reduced Fp/s**.

Any mismatch, command failure, spill-driven unbounded scratch, median miss, or
p5 miss is `NO_GO`. Raw products, unreduced convolution, or Montgomery steps may
not be counted as completed Fp operations.

## Result

The package-owned implementation passed:

- raw twelve-`u32` BLST 0.3.16 Montgomery equality without canonicalizing GPU
  output;
- zero, one, `p-1`, carry boundaries, a forced final-subtraction vector, and
  a 4,113-field repeated-chain corpus comprising 14 boundary fields and 4,099
  deterministic random fields, chained through seven multiplications;
- logical lengths 0, 1, 15, 16, and 17, plus both sides of every candidate
  threadgroup boundary;
- every output after every scout, warmup, and decision command; and
- the ABI fingerprint, SIMD32 requirement, reduced-output check, padding-store
  canary, command status, and zero static threadgroup-memory requirements.

The decision run used 131,072 logical fields, 32 dependent multiplications per
field, four dispatches per command, two warmups, and twenty samples. Width 32
won the independent three-sample-per-width scout.

| Metric | Observed | Required | Fraction |
| --- | ---: | ---: | ---: |
| Median GPU complete Fp/s | 0.986960B | 2.0B | 49.35% |
| Nearest-rank p5 GPU complete Fp/s | 0.939896B | 1.8B | 52.22% |
| Median wall complete Fp/s | 0.966377B | — | — |
| Nearest-rank p5 wall complete Fp/s | 0.922351B | — | — |

One lane pair, not two threads, counts as one completed operation. Allocation,
upload, BLST reference work, output checking, and readback are outside each GPU
timestamp. Full raw samples and artifact hashes are frozen in
[`evidence/m1-max-two-lane-u32-fips-field-2026-07-15.json`](evidence/m1-max-two-lane-u32-fips-field-2026-07-15.json).

The installed offline tools expose optimized AIR but not final Apple GPU
register allocation or spill counts. That evidence remains unknown; it cannot
rescue a schedule already missing the median gate by 2.03x.

### Frozen optimized-AIR audit

For AIR SHA-256
`1dcb004c76f64f77497fe6732ba77c9ad5a12f35dbb9193b489aeca355854bba`,
`xcrun air-objdump -d` reports the following inside
`u32_field_fips_montgomery_multiply`:

| Optimized AIR IR observation | Count |
| --- | ---: |
| `mul nuw i64` widening products | 122 |
| `mul i32` low-word correction products | 12 |
| `air.mul_hi.u.i32` correction products | 12 |
| `air.simd_shuffle_xor.u.v4i32` calls | 26 |
| `air.simd_shuffle_xor.u.v2i32` calls | 2 |

The complete AIR contains zero `addrspace(3)` threadgroup-memory references.
These are optimized AIR IR observations, not final Apple GPU ISA counts:
constant and zero products are subject to strength reduction, and the installed
tools do not expose final register allocation or spill metadata.

## Promotion order that did not unlock

A field pass permits only a complete curve-formula replay. The performance route
is branch-completed XYZZ with identity, doubling, and inverse exceptions handled
entirely on Metal; complete RCB remains a correctness slow lane, not the common
bucket formula.

1. Fused complete-add equivalent: at least **166.7M/s median** and **150M/s p5**
   under the declared 12-M-equivalent gate.
2. Authentic-shape tiled XYZZ replay: at most **300 ms median** and **375 ms
   p95**.
3. Full neutral equivalent of the 147-call resident batch: at most **400 ms
   median** and **500 ms p95**, with exact BLST equality, compact output only,
   zero CPU fallback/readback, and successful secret erasure.

Only all three passes permit a public safe MSM API and a one-way application
adapter.
