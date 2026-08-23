# Ledger 9 Metal maximum-attainable plan — 2026-07-15

> **Historical checkpoint.** This report records the state and candidate gates
> on 2026-07-15. The later fixed-base Metal MSM and resident proof integration
> supersede its statements about the implemented maximum: the production route
> now passes exact 20-pair acceptance above 3x. See
> `reports/ledger-9-metal-production-acceptance-2026-07-17.md`. The 10.2x
> infinite-device Amdahl bound remains a structural upper bound, not a forecast.

## Decision

The direct answer to “how much” depends on the boundary:

- **Implemented routed CPU+Metal bridge:** `1.010x` median end to end.
- **Measured resident quotient diagnostic:** an optimistic substituted
  `1.077x`; its actual safe epoch costs `1.975x` its budget, so the route is
  rejected.
- **Maximum while retaining CPU MSM:** a loose `2.125x` ceiling, so this route
  cannot satisfy the `3x` target.
- **Maximum after making warm FFT and MSM intervals free:** `2.610x`; this
  leaf-only ceiling still cannot satisfy the `3x` target.
- **Maximum structural opportunity after replacing MSM and nearly all measured
  proof work:** about `10.2x` with an infinitely fast device. This is an Amdahl
  bound, not an M1 Max performance forecast.

The current M1 Max does not have a promoted native-MSM forecast between those
bounds. The best complete generated field schedule reaches 1.529B median /
1.455B p5 against 2.0B/1.8B gates, and even an optimistic perfect CPU+Metal MSM
partition remains above the 400 ms median budget before traffic. Thus `10.2x`
is the maximum theoretical structural opportunity; the only implemented result
remains `1.010x`, and no measured current-Mac route supports a `3x` claim.

The algorithm-v2 separated subtractive-Karatsuba/L4 candidate also stopped at
Milestone 1. Its portable exact-integer model and generated Metal correctness
corpus pass, and its complete-RCB model has 38.65% fewer base products and
20.28% less logical point-value traffic than the historical named-FIOS/direct-XYZZ
diagnostic. But its
authoritative flattened helper peaks at 70 source-live `u32` equivalents versus
the limit of 56; the earlier 48-word count covers only top-level aggregates.

The algorithm-v3 blocked-radix-128 successor supersedes only the earlier
statement that v3 was pending. It passes generated Metal correctness,
optimized-AIR structure, and source liveness at 55/56 words, but reaches only
465.444M median / 455.318M p5 complete L4 field operations/s against the
2.0B/1.8B gates. It is `NO_GO` on throughput. The point-formula, compact-tree,
full-MSM, and additional residency stages did not run. Consequently the maximum
theoretical answer stays `10.2x`, while the current attainable answer stays
unproven above `1.077x` and the routed result stays `1.010x`. No measured M1 Max
Metal candidate justifies a `3x` claim. See
`reports/ledger-9-metal-blocked128-field-result-2026-07-16.md`.

The direct generated eight-lane AMNS8/Karatsuba Milestone-1 gate has now run on
the Apple M1 Max and is `NO_GO`. The kernel uses eight signed coefficients modulo
`X^8 + 2`, `phi=2^64`, `rho=4,600,539,647,907,200`, `delta=10`, and a 64-byte
field. Each retained sample executed
`131,072 fields * 32 iterations * 4 dispatches = 16,777,216` complete
multiplications. After two warmups, all 20 measured samples were correct, left
padding untouched, and used zero bytes of static threadgroup memory. Median GPU
throughput was 6,798,372.789 operations/s and nearest-rank p5 was
6,721,714.410, versus the 2.0B/1.8B gates. The median shortfall is about 294x.

The selected K8 graph still has 243 logical mixed `u32` products—189 widened
and 54 wrapping—while the direct eight-lane topology issues 288 physical
product slots. Two-/four-lane packing can reduce that only to 252 slots, a
1.143x improvement. That is nowhere near the measured gap, so the packing-only
topology sweep is pruned. The formal result supersedes the earlier
1.888B/1.796B instruction-scaling heuristic. It rejects this direct AMNS8/K8
topology on the M1 Max, not all AMNS arithmetic or Metal devices globally.

The agreed stop rule now closes RCB, point, compact-tree, full-MSM, and further
residency work on this machine. Reopening requires a materially
hierarchical/new arithmetic schedule or newer hardware, and the successor must
pass Milestone 1 before any higher layer resumes.
The standalone decision and all raw samples are frozen in
`bls12-381-metal-msm/AMNS8_KARATSUBA_RESULT.md` and
`bls12-381-metal-msm/evidence/m1-max-amns8-k8-field-2026-07-16.json`.

A bounded explicit-`mulhi` follow-up does not reopen that result. The v2
generator preserves the topology, liveness schedule, and same 494 shuffles;
optimized AIR replaces 28 variable `i64` multiply leaves with 28 `mulhi`
operations plus 36 variable `i32` multiplies. The constant-RHS scout measured
292.740B products/s for explicit low+`mulhi` versus 342.844B/s for widened
multiplication. With both operands rotating, it showed no material advantage:
213.761B versus 226.033B/s, with high variability. This prunes the substitution
only as prioritization against v1's 294.188x full-field miss; it is not a formal
v2 complete-field `NO_GO` or a Metal/M1 hardware bound. See the
[AMNS8 mulhi primitive result](../bls12-381-metal-msm/AMNS8_MULHI_PRIMITIVE_RESULT.md)
and [raw evidence](../bls12-381-metal-msm/evidence/m1-max-metal-mulhi-primitive-2026-07-16.json).

The earlier 25-coefficient PMNS/Karatsuba model remains lower-priority research.
Its portable bounds pass, its 615 graph leaves still split into 410 exact signed
widened and 205 wrapping products, and its 3.091B/2.963B wrapping-probe
normalization remains only an optimistic prioritization heuristic—not a
necessary bound or a Metal result.

Relative to the best measured complete-field result, named FIOS at
1.529B/1.455B median/p5, the next hardware/arithmetic candidate must improve by
at least `1.308x` at the median and `1.237x` at p5 merely to reopen G1. Relative
to v3 specifically, the required improvements are 4.297x and 3.953x. These are
prerequisite uplifts, not an MSM forecast; a candidate must still pass complete
homogeneous RCB point formulas, tiled buckets, and the full 147-call resident
gates. The branch-completed XYZZ path is permitted only as an explicitly labeled
trusted-local diagnostic and is not the production formula contract.

In time rather than ratios, the measured bridge moves the median from
`4580.986 ms` to `4535.683 ms`. The `3x` acceptance line is `1757.940 ms` for
the accepted CPU baseline. The backend-scope structural bound retains roughly
`512 ms` of CPU circuit-synthesis and request work, so
`5218.394 ms / ~512 ms ~= 10.2x`. It does not assume that arbitrary circuit
synthesis can be translated into Metal kernels.

Therefore `3x` requires a mostly Metal resident prover and a native Metal MSM.
The product target remains `>=3x`; `10.2x` is not a promised implementation
result.

Build a Metal-native semantic counterpart to the capabilities this repository
currently obtains from `midnight-sppark-msm`. Do not translate vendor sppark or
its CUDA kernel topology line by line.

The implementation has two deliberately distinct routes:

- The CPU+Metal bridge, currently named `metal-hybrid`, is an executable
  development route. Today Metal owns eligible batched coefficient-to-extended
  NTTs while the CPU owns the rest of the proof, including BLST MSM. The bridge
  is intended to grow into resident field epochs.
- `metal` remains unavailable until the resident field engine, arguments,
  quotient, multi-open, and production Metal MSM are all connected without
  whole-polynomial materialization or hidden fallback.

There is no CUDA+Metal execution route. CUDA remains a separate Linux reference
implementation; it is neither linked into nor used by the macOS CPU+Metal path.

This superseded the earlier decision to stop all proof integration after the
initial MSM designs failed their 3x feasibility gates. The resulting resident
quotient experiment is now measured: its CPU-BLST safe epoch failed the 600 ms
gate at 1,184.911 ms and projects only 1.077x under optimistic substitution.
The resident implementation remains an integration oracle. All selected M1 Max
standalone field/MSM mappings subsequently failed. The direct eight-lane
AMNS8/K8 gate then failed at 6.798M/6.722M median/p5 operations/s, about 294x
below its median gate. Packing-only 2/4-lane variants cannot bridge the gap and
are pruned. PMNS-25 remains research, not an active proof-route justification.

## What “Metal sppark” means here

The relevant specification is not only `vendor/sppark-0.1.15`. The repository's
custom CUDA layer in `midnight-sppark-msm/src/lib.rs` and
`midnight-sppark-msm/src/bls12_381_g1_msm.cu` is a resident proving runtime. Its
semantic surface includes:

- typed Fr storage, checked slices, upload, concatenate, gather, scatter, and
  controlled materialization;
- batched NTT/iNTT, coset ingress/egress, quotient division, and vector
  arithmetic;
- canonical ordering, deterministic sorting, segmented inversion, and prefix
  products;
- lookup permutation and permutation, lookup, and trash products;
- expression graphs, wrapped rotations, Horner accumulation, and numerator
  evaluation;
- reusable proving-key/numerator contexts;
- quotient splitting/blinding, ordered evaluations, and resident multi-open;
- Poseidon advice expansion; and
- cached neutral prepared-base collections (assigned coefficient/Lagrange
  meaning by the Midnight adapter) plus same-base, ragged, suffix, and
  resident-scalar G1 MSM.

Metal must reproduce those contracts and transcript-visible results. CUDA warp
shuffles, cooperative grids, streams, device pointers, CUB sorting, and kernel
boundaries are implementation details and are not part of the contract.

## zkMetal disposition

[`carni-ships/zkMetal`](https://github.com/carni-ships/zkMetal) was audited at
commit `443127161417b415f70d5de74a186a1820e557e5`. It is useful as research
evidence, not as an implementation dependency or correctness oracle:

- it does not expose a Rust BLS12-381 resident-prover API, NTT engine, quotient
  path, expression evaluator, or multi-open implementation;
- its Rust wrapper selects CPU/NEON work rather than the repository's Metal G1
  MSM path;
- the Metal MSM sorts scalars on the CPU and reduces each bucket serially, while
  its cooperative bucket-sum path is disabled after hangs;
- the GLV path contains an incorrect scalar-order constant and allocates storage
  for `n * windows` records before copying `2 * n * windows` records; and
- no root license was present at the audited commit, independently preventing a
  code-copying or vendoring decision without clarification.

Pinned source anchors: [primitive/API status](https://github.com/carni-ships/zkMetal/blob/443127161417b415f70d5de74a186a1820e557e5/README.md),
[Rust MSM wrapper](https://github.com/carni-ships/zkMetal/blob/443127161417b415f70d5de74a186a1820e557e5/bindings/rust/src/msm.rs),
[Swift BLS12-381 engine](https://github.com/carni-ships/zkMetal/blob/443127161417b415f70d5de74a186a1820e557e5/Sources/zkMetal/MSM/BLS12381MSMEngine.swift),
and [GLV kernels](https://github.com/carni-ships/zkMetal/blob/443127161417b415f70d5de74a186a1820e557e5/Sources/Shaders/msm/bls12381_glv_kernels.metal).

The stock non-GLV benchmark built on the M1 Max, but an independent harness
linked to that implementation failed against the repository's own CPU
Pippenger reference for tested sizes from 2,049 points upward. Its best measured
window-13 dense-random times were 24.648 ms at 8K points, 44.587 ms at 16K, and
88.006 ms at 32K. Replaying the authentic 23/62/62-call MSM shape as an
optimistic sequential sum projects about 8,175.611 ms, over 20x the 400 ms MSM
gate before accounting for the authentic size skew. These measurements reject
integration of the current design; they do not prove that a different Metal MSM
architecture is impossible.

No source from zkMetal is copied into this tree. The actionable lessons are the
negative ones: keep scalar preprocessing resident, distribute long buckets
cooperatively without a grid-wide barrier assumption, validate every large
size against an independent group implementation, and gate GLV decomposition
with exact scalar-field constants and allocation-shape tests.

## Measured performance bands

### Warm timing profile

The warmup-1/repeat-5 timer profile of seeds 24302–24306 produced these metric
medians on the M1 Max:

| Interval | Median |
| --- | ---: |
| Complete transaction | 5218.394 ms |
| Proof execution | 5154.881 ms |
| Outside proof | 58.546 ms |
| MSM union | 2404.229 ms |
| FFT union | 839.682 ms |
| FFT/MSM union | 3227.136 ms |
| `compute_nu_poly` | 1460.296 ms |
| `evaluate_numerator` | 1171.559 ms |

The useful schedule bounds are:

| Architecture | Bound / expectation | Meaning |
| --- | ---: | --- |
| Current host-round-trip Metal NTT hook | measured 1.010x | Exact five-sample, safety-complete result below. |
| Current resident quotient plus CPU MSM | optimistic measured substitution 1.077x | Uses the isolated 1,184.911 ms safe epoch in place of the prior CPU quotient interval; the route failed its gate. |
| Same sequential hybrid, infinite remaining field speed | loose about 1.55x | Retains CPU MSM, advice synthesis, measured Metal NTT boundary, and outside-proof work; not an implementation plan. |
| Two-arena CPU-MSM/Metal-field overlap | obsolete stretch about 1.82x | Superseded by the failed CPU-BLST quotient result; do not implement this route. |
| CPU MSM retained, every other measured proof interval free | loose 2.125x | Omits handoff/filter/gather costs and is not an implementation forecast. |
| FFT and MSM made free | 2.610x | Warm-trace leaf-only Amdahl ceiling; about 1999 ms remains. |
| Only advice synthesis and outside-proof work retained | loose 10.2x | Infinite-device structural bound. It proves 3x is possible after native MSM, not that the M1 Max can attain it. |

The old 2.047x free-FFT/MSM figure came from a cold diagnostic containing about
882 ms of proving-key initialization outside `create_proof`. The warm 2.610x
number is the relevant steady-state leaf bound. Neither number is a hardware
throughput prediction.

### Executable CPU+Metal bridge (`metal-hybrid`) result

The first production-shaped route batches eligible `coeffs_to_extended` calls
through one shared Metal session and intentionally returns their complete
polynomials to the CPU prover. Five alternating timed samples after one warmup
produced:

| Metric | CPU | `metal-hybrid` |
| --- | ---: | ---: |
| Median | 4580.986 ms | 4535.683 ms |
| p95 | 5167.139 ms | 4858.933 ms |
| Mean | 4734.544 ms | 4595.026 ms |
| Median speedup | — | **1.010x** |
| p95 speedup | — | 1.063x |
| Native commands / dispatches | 0 / 0 | 40 / 110 |
| Post-selection CPU retries | 0 | 0 |
| Logical host writes | 0 | 135 MiB |
| Logical host reads | 0 | 540 MiB |

Every seed produced byte-identical serialized transactions and passed the
normal canonical self-verification path. Each Metal sample executed four work
commands plus four mandatory clear commands, for twenty-two total dispatches;
it wrote 27 MiB and read 108 MiB. The two full-capacity resident buffers are
cleared on the GPU after each transform before they can return to the bounded
pool. An unprovable clear completion quarantines the buffers and poisons the
session instead of permitting reuse.

The zero retry count has a narrow meaning: once an eligible transform selects
Metal, a Metal error aborts rather than rerunning that transform on CPU. Calls
that miss the bridge's field, size, or batch eligibility checks remain on CPU
before selection and are not counted by the current metric. Add separate
eligible-call and ineligible-bypass counters before promoting any resident
route; zero retries alone is not evidence of mostly-Metal execution.

This safety-complete result supersedes the earlier 1.019x measurement, which
predated mandatory resident-buffer erasure and pooling lifecycle accounting.

The timer profile and paired acceptance diagnostic have different
instrumentation and timing conditions. Their absolute CPU medians must not be
mixed to manufacture a speedup. The former supplies interval bounds; the latter
is the authoritative paired result for the current executable route.

Artifacts:

- `/tmp/metal-hybrid-lifecycle-repeat5.json`
  - SHA-256 `5c525dab247c66eaafbf604c05e2de056215f1fc2b01d1cbf6ba39242d3395ab`
- `target/release/proof-backend-parity`
  - SHA-256 `e651f0a4d99706e7c499e47ee0e98bc766e0833e4f1f03bb758c807cd95cfac2`

Reproduction command:

```bash
MIDNIGHT_LEDGER_TEST_STATIC_DIR="$PWD/scripts" \
./target/release/proof-backend-parity \
  --backend cpu-metal-hybrid \
  --fixture zswap-transaction \
  --iterations 5 \
  --warmup 1 \
  --seed 24301 \
  --json /tmp/metal-hybrid-lifecycle-repeat5.json
```

Here `cpu-metal-hybrid` is the parity binary's legacy spelling for two separate,
alternating routes: canonical CPU and the CPU+Metal bridge. The profiler wrapper
accepts the clearer equivalent `--backend cpu-vs-metal-hybrid`; neither spelling
describes a CUDA+Metal process.

Five samples establish direction and parity, not a production statistical
gate. The release gate remains at least twenty alternating samples after two
warmups plus repeat-100 lifecycle testing.

## Target architecture

```text
midnight-proofs / midnight-curves
              |
       resident phase semantics
              |
      midnight-gpu-prover
         /             \
midnight-sppark-msm   midnight-metal-prover
       CUDA                 Metal
```

The diagram shows mutually exclusive source implementations behind one facade,
not a CUDA+Metal process. Linux/CUDA and Apple-silicon/Metal artifacts are built
and executed separately.

`midnight-gpu-prover` becomes the compile-time-selected resident facade. The
proof crates use concrete target-selected handles rather than trait objects,
CUDA pointer types, or Metal framework objects:

- `ExecutionGuard`;
- `ResidentFrBuffer` and checked resident slices;
- `ResidentKeyContext`;
- `ResidentProofArena`;
- `ResidentMsmContext`;
- `ResidentPolynomialId`; and
- common activity, materialization, and error records.

The proof-facing operations describe semantic phases—lookup permutation,
argument products, quotient construction, query evaluation, and multi-open
epochs. Backend-private code decides how many Metal kernels and command buffers
implement each phase.

## Implementation history and gated continuation

Steps 1--4 below record foundation and diagnostic work that already exists;
they are not authorization to extend proof routing while native MSM is closed.
The field decision point inside step 5 has now failed. V3 failed first; the
direct generated eight-lane AMNS8/K8 successor then measured only
6.798M/6.722M median/p5 operations/s and is also `NO_GO`. Its 294x median gap
cannot be rescued by the 1.143x physical-slot gain available to two-/four-lane
packing, so that branch is pruned. Steps 6--9 remain deferred. There is no active
point/MSM or resident-proof implementation queue on this M1 Max.

### 1. Stabilize the CPU+Metal boundary

- Keep `metal` fail-closed and expose `metal-hybrid` explicitly.
- Use one process session and one unified accelerator permit.
- Count Metal command buffers, dispatches, waits, writes, reads, failures, and
  post-selection retries per proof. Add explicit eligible-call and
  ineligible-bypass counters as the required follow-up to the current bridge.
- Retain the current hook only as a measurable bridge; do not add more
  host-round-trip leaf calls without an end-to-end win.

Exit: byte-exact proof output, canonical verification, native dispatches, zero
post-selection CPU retries, and the paired result above. This slice is complete;
eligibility counters remain required before resident-route promotion.

### 2. Stabilize the Metal runtime

Status: the lifecycle is implemented and failure-injection tested for the
resident Fr NTT arena and diagnostic quotient epoch owner, including callback
error, unwind, quarantine, and poison behavior. Generalization to every
proof-secret region and the host module split remain open.

- Split `midnight-metal-prover/src/metal.rs` by ownership into session, ABI,
  arena, field, NTT, G1, diagnostics, and later proof-phase modules.
- Retain one device, queue owner, library, and pipeline cache.
- Add `Healthy`, `Clearing`, and `Poisoned` session states.
- Quarantine resources through actual command completion, GPU-clear secret
  arenas after errors, and poison the session if clearing cannot be guaranteed.
- Add generation-checked buffer pooling and proof-scoped monotonic counters.

Exit: injected failures prove completion, clearing, rejection while clearing,
and permanent poison behavior; repeat-100 shows no post-warmup arena growth.

### 3. Neutralize CUDA-owned proof seams

This removes CUDA-specific types from generic proof code so Metal can own the
same protocol boundaries; it does not combine CUDA and Metal execution.

Status: the neutral `RawFr`, request-owned `ProofArena`, opaque generation-
checked buffer token, error contract, and `gpu-resident` feature boundary are
implemented and tested. Existing proof carrier fields still name CUDA types;
the facade therefore remains unused by proof execution and Metal remains
fail-closed.

- Introduce the resident facade in `midnight-gpu-prover`.
- Replace concrete `Bls12_381FrDeviceVec`, numerator-context, and multi-open
  types in `ProverTrace`, proving keys, PCS hooks, and argument structures one
  seam at a time.
- Keep CUDA's implementation and tuning backend-private.
- Key SRS/key contexts by stable proving-key fingerprint, device identity,
  backend version, and configuration—not a host pointer.

Exit: CPU and CUDA deterministic outputs remain byte-exact before and after
each seam migration; the CUDA error-isolation regression passes on NVIDIA.

### 4. Build the resident numerator/quotient engine

This is the highest-return field slice. The complete field-side epoch costs
1,513.724 ms median in the warm trace, versus 97.673 ms for ordered evaluations
and 196.686 ms for the non-commitment part of multi-open. A 600 ms Metal epoch
would save about 914 ms and imply approximately 1.212x end-to-end on its own;
making this epoch free caps its isolated contribution at about 1.409x.

The atomic proof-visible boundary starts immediately after the CPU squeezes
`y` and ends only after it writes the ordered quotient-limb commitments, before
squeezing `x`:

```text
y challenge -> dynamic coefficients -> extended cosets -> numerator identities
            -> vanishing division + inverse NTT -> n-1 split/blind
            -> quotient-limb commitments -> x challenge
```

Stopping at numerator coefficients is not a protocol epoch because it changes
no transcript state. Evaluations and multi-open should not land first: both
consume the dynamic/fixed arena and source map created by quotient construction,
and isolated host-upload bridges would target much smaller intervals.

Status: the diagnostic implementation and representative performance decision
are complete. The target-neutral graph IR and logical-target
compiler, six specialized Metal expression pipelines, pinned startup ABI probe,
checked dispatch, CPU/BLST semantic corpus, and deterministic key-wide graph
family compaction are implemented. Required permutation and trash-selector
columns outside graph operands participate in the compact maps and fingerprint.
The session-cached schema-v2 Metal context binds the exact PK, domain, NTT ABI,
and coefficient/Lagrange SRS identities; uploads every immutable array; and
exposes checked field, periodic-table, and G1 views. Dynamic views reject
cross-session ID collisions and carry explicit basis, exact domain, source, and
partial-Stockham state.

The diagnostic quotient epoch now accepts the complete schema-v2 binding,
uploads dynamic coefficients once, and performs extended-coset NTTs, all four
numerator categories, vanishing division, inverse coset NTT, and exact `n-1`
split/blinding in one non-clear Metal command with one intentional wait. It then
exposes exactly `q*n` `RawFr` values through a callback-scoped, non-cloneable CPU
MSM lease and GPU-clears every secret allocation before returning. It does not
materialize the 108 MiB extended-coset working set. Error, panic, clear-failure,
quarantine, and poison paths are tested.

Independent q=2/3/4 CPU differentials cover every field phase. An authentic
private K=4 prover harness matches ordered quotient commitment bytes, `x`, and
the next 64 RNG bytes across custom, cross-kind permutation/equality, lookup,
table, and additive-selector trash constraints. The existing CPU BLST adapter
still makes scalar/base/digit preprocessing copies that are not zeroized, so the
path is trusted-local diagnostic support rather than production routing. No
production route selects it.

The representative two-spend, two-output, one-dust transaction measured the
complete five-proof safe epoch at a **1,184.911 ms median** after two warmups.
All 25 retained quotient epochs and CPU-MSM leases completed with byte-exact
serialized output, normal proof verification, zero quotient-local whole-
polynomial materializations, and zero clear, poison, or quarantine failures.
The result is **1.975x the budget**—97.5% over the predeclared 600 ms gate. Even replacing the
prior 1,513.724 ms CPU quotient interval optimistically would move the
4,580.182 ms CPU transaction only to about 4,251.369 ms, or **1.077x**.

Decision: reject this CPU-BLST continuation as a 3x-enabling implementation
path. Preserve the epoch as the correctness/lifecycle oracle for a future native
MSM, and do not connect it to proof finalization. The full command, samples,
counter contract, and decision are recorded in
`reports/ledger-9-metal-resident-quotient-result-2026-07-15.md`.

### 5. Qualify a standalone native Metal MSM

A real Metal backend requires:

- validated, device-bound neutral `PreparedBases` collections and cache
  identities; the Midnight adapter assigns coefficient/Lagrange semantics;
- canonical scalar recoding, never digit extraction from Montgomery limbs;
- the explicitly approved strict-oblivious or trusted-local leakage profile;
- histogram/scan/scatter or another measured Apple-native routing strategy;
- complete accumulation, integration, ragged/suffix batches, compact G1
  egress, and secret erasure; and
- exact BLST/canonical-CPU differential tests at 8191/8192, 16383/16384, and
  32767/32768 boundaries.

The production point contract is complete homogeneous RCB with a fixed reviewed
schedule. Branch-completed XYZZ may be retained only as an optional
trusted-local diagnostic; its results cannot promote the production MSM.

The one-way Midnight adapter separately preserves CUDA parity and may consume
immutable external fixtures; the standalone library never imports or links the
CUDA implementation.

The already measured global batch-affine, flat-segmented, and register-only
branches remain rejected. A new design must independently clear the aggregate
MSM median/p95 gates of 400/500 ms before proof integration.

The exact-FP32 packed-convolution falsifier has now stopped before shader work.
Its balanced-radix-256, three-level-Karatsuba construction needs three
972-product convolutions, or 2,916 scalar FP32 products, for one complete
reduced Montgomery Fp multiplication. The established 2.0B/s pre-G1 gate would
therefore require 5.832T scalar products/s. Interpreting the M1 Max's optimistic
10.4-TFLOP/s FP32 peak in the conventional two-FLOP FMA accounting gives at
most about 5.2T product issues/s: the candidate needs 112.16% of the
product-issue roofline before additions, carries, barriers, memory, or command
cost. Its most
optimistic complete-Fp ceiling is 1.783B/s.

The independently extractable `bls12-381-metal-msm` package freezes that result
as a fail-closed assessment with its own license, security policy, and
provenance. A second carry-normalized radix-4096 candidate lowered the modeled
floor to 1,941 products per complete Fp. Its package-owned raw precise-FP32 gate
then measured 1.945T median and 1.827T p5 product issues/s: only 1.002B/0.941B
1,941-product equivalents/s against 2.40B/2.25B predeclared gates. The shader
passed ABI and bit-exact float checks, but performed no normalization, reduction,
or cryptography, so this favorable roughly 2.4x miss rejects the tested schedule
before complete Montgomery or G1 work. It is not an upper bound on every FP32
mapping.

The diagnostic runtime, shader, and raw evidence remain inside the standalone
package behind an unstable feature. It still exposes no production resident
interop, G1, GLV, router, or MSM execution API. A successor requires a materially
different arithmetic primitive and a new predeclared complete-field gate;
relabeling raw products or unreduced convolution as Fp throughput is not
permitted.

The next successor used two SIMD lanes per field and a generated,
coverage-checked FIPS schedule. Unlike the FP32 probes, it implemented the
complete Montgomery correction and fixed final subtraction and matched BLST at
the raw twelve-`u32` Montgomery-limb boundary. Its twenty-sample M1 Max decision
run reached only 0.986960B median and 0.939896B nearest-rank p5 complete Fp/s,
against 2.0B/1.8B gates. The two-lane mapping is therefore rejected. At those
rates, the authentic trusted-local direct-XYZZ diagnostic workload's
multiply/square component alone
projects to about 504 ms median and 529 ms p5, before field additions, routing,
traffic, or integration. The result is schedule-specific, so one materially
different generated named-register scalar FIOS gate was authorized with complete
reduction and the same BLST oracle. It removed private arrays, inner loops,
shuffles, and threadgroup scratch, but reached only 1.529B median / 1.455B p5;
a stronger sweep confirmed a 1.535B median. The selected M1 Max field mappings
therefore close before G1. At the formal rates, optimistic diagnostic
direct-XYZZ arithmetic
already projects to about 455.0/471.5 ms median/p5 before traffic or routing, so
the 400/500 ms batch gate is not an implementation forecast.

A later native-`u32` product-only schedule also failed its declared gate: its
1.900900T/1.822508T median/p5 raw issue rates correspond to only
1.056055B/1.012504B 1,800-product reference-schedule equivalents/s. This
rejects that tested 30×13 mapping, not the entire family, because Apple
publishes no device-wide native-`u32` issue peak.

The algorithm-v2 materially different candidate evaluated in the standalone
package uses exact 12-limb subtractive Karatsuba (`K12 = 63`), separated
Montgomery REDC, and lazy
`[0,4p)` residues. Its portable model validates 189-product multiplication and
closure below `3p` over canonical, boundary, and deterministic randomized L4
inputs. Combining it with complete homogeneous RCB formulas and fused
two-product reductions reduces the authentic structural count from 143.262B to
87.890B base products and logical point traffic from 23.657GB to 18.860GB. The
portable and generated Metal correctness checks pass, but its authoritative
flattened helper requires 70 source-live `u32` equivalents, above the frozen
limit of 56. The 48-word top-level aggregate subset is not a full-kernel
liveness value. Milestone 1 is therefore `NO_GO` before throughput and physical
spill testing. The complete-formula, compact-tree, 400/500 ms full-MSM, and
residency gates were not run. See
`reports/ledger-9-metal-karatsuba-l4-candidate-2026-07-15.md`.

Algorithm v3 then replaced the separated wide recurrence with a generated
blocked-radix-128 recurrence. It passed the exact portable model, generated
Metal differential corpus, optimized-AIR structural inspection, and the source
gate at 55 live `u32`-equivalent words versus the 56-word limit. Its
both-operands-varying decision run nevertheless reached only 465.444M median /
455.318M nearest-rank p5 complete L4 operations/s, versus the frozen
2.0B/1.8B gates. The median and p5 miss by factors of 4.297x and 3.953x,
respectively. This is the Milestone 1 stop: do not implement complete point
formulas, a compact reduction tree, routing, the standalone full MSM, or further
resident-proof integration for this schedule. The full decision is in
`reports/ledger-9-metal-blocked128-field-result-2026-07-16.md`.

The favorable product-count projections did not survive complete-field
measurement. Scaling named FIOS's 1.529B rate by `300/189` suggested about
2.427B operations/s only as a prioritization heuristic; v3 measured 0.465B.
Likewise, dividing the earlier 1.9009T native-`u32` product-only issue result by
189 gives about 10.06B mathematical product equivalents/s, but that probe omits
the complete recurrence's additions, carries, dependencies, and reduction.
Neither number is a complete-field forecast, and neither is related to the
separate proof-level 10.2x infinite-device structural bound.

The direct generated AMNS8/Karatsuba experiment is now complete and rejected for
this M1 Max. It ran 131,072 fields for 32 dependent iterations across four
dispatches, or 16,777,216 complete multiplications per sample. After two
warmups, 20 measured samples produced a 6,798,372.789 median and
6,721,714.410 p5 GPU operations/s. Every sample was correct, padding remained
untouched, and static threadgroup memory was zero, so the failure is the measured
throughput rather than correctness or hidden scratch. Against 2.0B/1.8B, the
direct eight-lane topology is `NO_GO` by about 294x at the median.

The logical graph remains 189 widened plus 54 wrapping products, or 243, while
the topology issues 288 physical product slots. The best packing-only change to
two or four lanes yields 252 slots, merely 1.143x better. The proposed topology
sweep is therefore pruned rather than benchmarked. This does not establish a
global AMNS or Metal upper bound; a materially hierarchical/new schedule or
newer hardware may reopen the question, but only by passing a fresh Milestone-1
gate first. No RCB, point, MSM, or residency work proceeds from this result.

The earlier `n=25`, `X^25-2`, `phi=2^32` PMNS/Karatsuba experiment remains
lower-priority research. Its strict lazy bound of 1,230,717, phi bound of
361,830,798, and portable model remain valid. Its 615-product graph count splits
into 410 widened and 205 wrapping leaves; the 3.091B/2.963B normalization is only
a prioritization heuristic, not a bound, admission pass, or throughput forecast.

Even a perfect CPU+Metal MSM partition does not recover the median gate. Treating
the optimistic 455.0 ms Metal arithmetic projection and the measured 6,248.929
ms CPU MSM sum as ideal linearly divisible engines yields a 424.1 ms harmonic
lower bound (438.4 ms from the p5 projection), before coordination, traffic,
routing, or erasure. CPU+Metal remains an allowed future architecture, but the
current arithmetic does not justify implementing that split. No CUDA+Metal
execution route exists.

zkMetal does not supply a shortcut. Its untouched BLS12-381 engine averaged
376.6 ms for one 16K call on this M1 Max. The authentic trace has 60 exact 16K
calls, so those alone project to 22.596 seconds. The engine also performs CPU
recoding/count-sort and its unverified large paths contain correctness and buffer
layout defects. See
`ledger-9-zkmetal-bls12-381-evaluation-2026-07-15.md`.

All arithmetic, routing, prepared-base ownership, Metal runtime, shaders,
correctness tests, and benchmarks for this step live in
`bls12-381-metal-msm`. Midnight supplies only a one-way adapter after the
standalone API clears its gates.

### 6. Replace the quotient CPU-MSM seam

- Start at the existing exact boundary:
  `midnight-metal-prover/src/metal/quotient_epoch.rs::MetalSession::with_diagnostic_quotient_epoch`
  produces blinded resident limbs,
  `midnight-gpu-prover/src/lib.rs::metal::with_diagnostic_quotient_epoch`
  currently converts them into a callback-scoped CPU lease, and
  `vendor/midnight-proofs-0.8.1/src/poly/kzg/mod.rs::diagnostic_commit_quotient_lease`
  invokes CPU `msm_specific` once per limb.
- Replace that lease with a checked resident scalar view bound to the standalone
  engine's coefficient `PreparedBases`; do not route through generic host-slice
  `msm_specific` or duplicate the SRS allocation.
- Commit every quotient limb natively before the transcript writes and `x`
  challenge, preserving exact ordering and proof bytes; return only ordered
  compact G1 commitments and retain the quotient for evaluations and multi-open.
- Keep the current CPU-BLST callback only as a trusted-local differential oracle;
  do not make it part of the routed proof.

Exit: native and CPU commitment bytes, transcript prefix, and next RNG state
match; there is no quotient polynomial readback, fallback, or post-selection
retry.

### 7. Move remaining trace producers resident

- Port vector arithmetic, segmented inversion/prefix scans, exact canonical Fr
  ordering, and deterministic radix sort.
- Port Poseidon advice expansion, lookup compression/permutation, and
  permutation, lookup, and trash products.
- Preserve transcript/RNG order and return only compact commitments or
  challenge-dependent scalar egress at real host epochs.

Exit: per-stage buffers and transcript prefixes match CPU/CUDA for lookup
duplicates, modulus boundaries, negative rotations, and boundary domain sizes.

### 8. Port field-side multi-open

- Neutralize stable dynamic/fixed source IDs and point-set ordering.
- Keep `q`, `f`, the final polynomial, reverse affine-transform Kate scans,
  batched evaluations, and `pi` preparation resident.
- Use only the standalone native MSM adapter for transcript-visible commitments;
  do not revive the failed CPU-BLST overlap route.

Exit: every multi-open intermediate and transcript epoch matches CPU/CUDA, with
zero whole-polynomial readback.

### 9. Enable pure `metal`

Add the pure route only when every selected operation is native, production
whole-polynomial materialization is zero, eligibility bypass is zero,
post-selection retry is zero, failures erase or poison correctly, and the guard
constructs a real Metal proof arena.

## Final acceptance

- At least twenty paired, alternating CPU/Metal samples after two warmups.
- Exact deterministic proof/transaction bytes and independent canonical
  verification for every sample.
- `compute_call_count > 0`, `fallback_count == 0` for post-selection retries,
  zero ineligible CPU bypass, and no hidden retry.
- No production whole-polynomial materialization.
- End-to-end median at most 1757.940 ms and p95 at most 2000 ms for the original
  3x acceptance target; paired CPU/Metal median ratio also at least 3.0x.
- Repeat-100 passes every proof, includes stress seed 24320, has no post-warmup
  arena growth, and stays within a checked memory manifest.
- Metal System Trace confirms waits only at transcript or intentional hybrid
  CPU-MSM boundaries.
- The MSM leakage model, remote-timing policy, profiler boundary, and secret
  erasure contract have explicit security-owner approval.

The 10.2x structural bound shows that 3x is mathematically possible once MSM
and the resident field pipeline are native. It is not an M1 Max forecast: the
current measured candidates, including algorithm v2's Karatsuba/L4 Milestone 1
liveness failure and algorithm v3's blocked-radix-128 Milestone 1 throughput
failure, do not justify a production MSM or a 3x claim on this machine. The
direct eight-lane AMNS8/Karatsuba successor now reinforces that conclusion with
a measured 6.798M/6.722M median/p5 `NO_GO`, about 294x below the median gate.
Packing-only variants are pruned; PMNS-25 remains research. The maximum
theoretical structural opportunity remains 10.2x and the routed result remains
1.010x, but neither is an attained native-Metal forecast.
