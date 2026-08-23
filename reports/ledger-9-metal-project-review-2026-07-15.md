# Ledger 9 Metal backend project review — 2026-07-15

> **Historical checkpoint.** This report preserves the evidence available on
> 2026-07-15. Its project-level no-go conclusion was overturned by the later
> fixed-base Metal MSM and resident proof integration, which pass exact 20-pair
> production acceptance above 3x. See
> `reports/ledger-9-metal-production-acceptance-2026-07-17.md`. Candidate-specific
> arithmetic measurements below remain valid for those candidates.

> Scope update: the MSM no-go findings below remain valid for the measured
> candidates. A subsequent maximum-attainable experiment implemented the
> resident quotient epoch with an intentional CPU-BLST seam, but its 1,184.911 ms
> representative median failed the 600 ms gate and projects only 1.077x under
> optimistic substitution. The resident field engine remains an oracle. The
> subsequent standalone Metal arithmetic campaign rejected two complete field
> mappings, including generated named-register FIOS at 1.529B/1.455B median/p5.
> The algorithm-v2 separated subtractive-Karatsuba/L4 candidate passes its
> portable exact-integer model and generated Metal correctness corpus, but
> fails the Milestone 1 source-liveness gate at 70 `u32`-equivalent live words
> versus the limit of 56. The later algorithm-v3 blocked-radix-128 schedule
> passes correctness, optimized-AIR structure, and source liveness at 55/56
> words, but reaches only 465.444M median / 455.318M p5 complete L4 field
> operations/s against the 2.0B/1.8B gates. Milestone 1 is therefore `NO_GO` on
> throughput; no formula, tree, MSM, or further residency work ran. This
> supersedes only the earlier statement that v3 was pending; the v2 result and
> all historical measurements remain unchanged. CUDA remains a separate Linux
> reference; no CUDA+Metal execution route exists. See
> `reports/ledger-9-metal-maximum-attainable-2026-07-15.md`.
> The independently audited direct eight-lane AMNS8/Karatsuba Milestone-1 gate
> has now run on the Apple M1 Max and is `NO_GO`. Each retained sample executed
> `131,072 * 32 * 4 = 16,777,216` complete field multiplications. After two
> warmups, all 20 measured samples were correct, left padding untouched, and used
> zero bytes of static threadgroup memory. Median GPU throughput was only
> 6,798,372.789 operations/s and nearest-rank p5 was 6,721,714.410, versus the
> frozen 2.0B/1.8B gates. The median shortfall is about 294x. This supersedes the
> earlier 1.888B/1.796B instruction-scaling heuristic.
> The logical K8 graph still contains 189 widened plus 54 wrapping products, or
> 243 total, while the direct eight-lane topology issues 288 base-product slots.
> Reducing slot rounding with two or four lanes can improve 288 only to 252, or
> 1.143x, nowhere near the measured gap; that packing-only sweep is pruned.
> This rejects the direct generated AMNS8/K8 topology on this M1 Max, not all
> AMNS arithmetic or Metal hardware globally. The stop rule now forbids RCB,
> point, tree, MSM, and additional residency work on this machine. Reopening
> requires a materially hierarchical/new arithmetic schedule or newer hardware,
> and that successor must pass Milestone 1 first.
> The standalone decision and all raw samples are frozen in
> `bls12-381-metal-msm/AMNS8_KARATSUBA_RESULT.md` and
> `bls12-381-metal-msm/evidence/m1-max-amns8-k8-field-2026-07-16.json`.
> A bounded explicit-`mulhi` follow-up does not reopen that result. The v2
> generator preserves the topology, liveness schedule, and same 494 shuffles;
> optimized AIR replaces 28 variable `i64` multiply leaves with 28 `mulhi`
> operations plus 36 variable `i32` multiplies. The constant-RHS scout measured
> 292.740B products/s for explicit low+`mulhi` versus 342.844B/s for widened
> multiplication. With both operands rotating, it showed no material advantage:
> 213.761B versus 226.033B/s, with high variability. This prunes the substitution
> only as prioritization against v1's 294.188x full-field miss; it is not a formal
> v2 complete-field `NO_GO` or a Metal/M1 hardware bound. See the
> [AMNS8 mulhi primitive result](../bls12-381-metal-msm/AMNS8_MULHI_PRIMITIVE_RESULT.md)
> and [raw evidence](../bls12-381-metal-msm/evidence/m1-max-metal-mulhi-primitive-2026-07-16.json).
> The earlier `n=25` PMNS/Karatsuba model remains useful lower-priority research:
> its strict lazy bound is 1,230,717, its required phi bound is 361,830,798, and
> its 615 leaves split into 410 exact signed widened and 205 wrapping products.
> Its 3.091B/2.963B wrapping-probe normalization remains only an optimistic
> prioritization heuristic, not a necessary bound or throughput result.
> The current executable CPU+Metal bridge measures 1.010x. The warm leaf-only
> free-FFT/MSM ceiling is 2.610x, the loose structural infinite-device ceiling
> is about 10.2x, and retaining CPU MSM limits even an otherwise free measured
> proof to 2.125x. These bounds answer different questions and must not be
> presented as one measured speedup.

> Handling: internal engineering review. It references sensitive aggregate MSM
> diagnostics and requires security-owner review before public release.

## Executive verdict

The correct Metal architecture is a target-selected, resident PLONK backend that
reuses the canonical Midnight prover, transcript, proof encoding, and verifier.
It must not be a collection of host-round-trip NTT/MSM leaf calls and must never
run CPU work under a Metal label.

The project is ready to continue reusable foundation maintenance, but it is not
ready to route a Metal proof. Runtime ownership, ABI validation, resident NTT,
complete G1 correctness probes, fail-closed routing, profiling, and benchmark
contracts are in place. The production MSM arithmetic design is not: the
six-`u64` path, the existing scalar twelve-`u32` paths, explicit two-/three-state
interleaving, and the standalone two-lane `u32` FIPS path all fail their measured
feasibility gates. The algorithm-v2 Karatsuba/L4 design failed its 56-word
source-liveness condition at 70 words. Its blocked-radix-128 v3 successor passed
correctness, optimized-AIR structure, and source liveness at 55 words, but
failed the 2.0B/1.8B field-throughput gates at 465.444M/455.318M median/p5. It
is not a viable MSM candidate on this path; point, bucket, full-MSM, and further
residency stages were not started.

The post-v3 representation restart is now measured. The generated direct
AMNS8/Karatsuba kernel uses eight signed coefficients modulo `X^8 + 2`,
`phi=2^64`, `rho=4,600,539,647,907,200`, `delta=10`, and a 64-byte field spread
across eight lanes. Its portable exact, randomized, repeated-chain,
curve-isomorphism, and complete homogeneous RCB oracles pass, including
`W=109 < 125` capacity. The formal M1 Max run then executed 16,777,216 complete
multiplications per sample, with two warmups and 20 measured samples. Every
sample was correct, padding remained untouched, and static threadgroup memory
was zero. Nevertheless, median/p5 GPU throughput was only
6,798,372.789/6,721,714.410 operations/s against 2.0B/1.8B.

The direct eight-lane topology is therefore `NO_GO`, about 294x short at the
median. Its logical graph remains 189 widened plus 54 wrapping products, while
the topology issues 288 product slots. Two-/four-lane slot packing can reduce
that only to 252, a 1.143x improvement, so it is not a credible rescue and is
pruned. The result is topology- and M1-Max-specific rather than a global AMNS or
Metal impossibility result. On this machine, the agreed stop rule closes point
formulas, compact trees, MSM, and further resident-proof integration. Only a
materially hierarchical/new arithmetic schedule or newer hardware can reopen
the project, and it must restart at Milestone 1.

The older 25-coefficient PMNS/Karatsuba experiment remains lower-priority
representation research. Its portable model still passes; its strict lazy bound
is 1,230,717 and its required phi bound is 361,830,798. Its 615
coefficient-product graph count still splits into 410 exact signed
widened and 205 wrapping leaves. Its 3.091B/2.963B normalization remains only an
optimistic prioritization heuristic, not a necessary bound or Metal result.

The lower-operation review is now measured and rejected. A correctness-first,
one-command global batch-affine replay reduced 6.794M representative
authentic-shape pairs at only 30.633M/s. Its 20-sample median/p95 were
221.794/223.972 ms, projecting the complete 46.039M-pair bucket phase at
1,502.943 ms before routing or integration. It missed the immediate-stop rate
by 4.12x and the promotion rate by 5.01x. The isolated-rate projection was not
representative of the complete device dataflow. No record router or proof
integration may be built around this design.

The final recursive flat-segmented alternative was then implemented with
32-point same-run tiles and threadgroup-local product trees. It is correct but
slower: G128/G256 reached only 3.180M/2.593M useful reductions/s and project
14,479.771/17,756.113 ms for the complete reduction population. G128 misses the
promotion rate by 48.3x. The final layout/inversion research branch is closed.

The remaining hierarchical-scan question was bounded with an exact,
register-local `7 M/S + 9 A/S` ceiling probe. Its 20-sample median was
164.206M pair-iterations/s, which projects 280.374 ms for the authentic
46.039M reductions before prefix/suffix storage, root inversions, barriers,
routing, gathers, integration, or writes. It cleared the bare 153.5M/s rate by
only 6.97% and failed the predeclared 200M/s headroom gate. A different schedule
could change the fused rate, but it would also have to materialize all omitted
work; no hierarchical implementation branch is authorized by this result.

The original 3x target is `1,757.940 ms` against the same-Mac cold diagnostic
CPU median of `5,273.820 ms`. That run's `2.047x` free-FFT/MSM value included
about 882 ms of proving-key initialization outside `create_proof`; the relevant
warm steady-state leaf-only ceiling is `2.610x`. The paired executable bridge
uses a different timing population and measures only `1.010x`. The
implementation must therefore keep polynomials resident through arguments,
numerator, quotient, evaluations, and KZG multi-open and eventually replace CPU
MSM to reach 3x.

## Inputs reviewed

- the workspace proof-server, proof crates, CUDA/sppark adapter, CUDA-oxide
  facade, feature graph, CI, profiler, and proof-flow map;
- the native Metal crate, generated ABI/constants, shaders, tests, and reports;
- the complete Nightstream guide at
  `/Users/nijaar/starstream/Nightstream/METAL_IMPLEMENTATION_BEST_PRACTICES.md`,
  SHA-256
  `40e9ede592f887716937b9949a9c6ffd5cc98207c8ffd4f8a5f5f562e0a77bde`;
  and
- same-host CPU, Amdahl, primitive, NTT, and MSM feasibility measurements.

Nightstream is useful architectural evidence for lifecycle, residency,
instrumentation, and benchmarking. It is not a source for BLS12-381 field,
curve, MSM, PLONK, or side-channel decisions.

## Current state

| Area | Status | Evidence / consequence |
| --- | --- | --- |
| Toolchain | Ready | Rust 1.95.0 is pinned; native Metal build uses one root MSL translation unit and an embedded metallib. |
| Runtime/lifecycle | Resident-foundation ready | One long-lived device/queue/session, serialized execution, retained command resources, authoritative per-generation view state, bounded exact-size secret pooling, post-completion GPU erasure, quarantine, and durable session poison are implemented and failure-injection tested for the resident Fr arena. General proof-arena ownership remains open. |
| ABI/constants | Ready | Fr/Fp/G1 sizes, alignments, member offsets, generated constants, and pinned fingerprints are checked at session startup. |
| Fr and NTT | Foundation-ready | Mixed-radix resident Stockham NTT/iNTT passes differential tests and reaches about 32.6–32.8M element-transforms/s in the large-batch diagnostic. |
| Fp/G1 correctness | Diagnostic-ready | Fixed-schedule Fp and complete homogeneous RCB add/mixed-add/double match BLST across boundaries, randomized subgroup points, rescaling, and on-curve non-subgroup cases. |
| Fp/G1 performance | All measured candidates rejected; direct AMNS8 is NO_GO | Six-`u64`, twelve-`u32`, two-lane FIPS, named FIOS, v2, and v3 all failed their gates. The direct generated eight-lane AMNS8/K8 formal run used 131,072 fields, 32 iterations, four dispatches, two warmups, and 20 measured samples. All samples were correct, padding remained untouched, and static threadgroup memory was zero. Median/p5 throughput was only 6,798,372.789/6,721,714.410 operations/s versus 2.0B/1.8B—about 294x short at the median. Its logical graph is 189 widened + 54 wrapping = 243 products, while the topology issues 288 slots. Two-/four-lane packing reaches only 252 slots, a 1.143x improvement, so that sweep is pruned. This rejects the direct AMNS8/K8 topology on M1 Max, not AMNS or Metal globally. PMNS-25 remains research. Milestone 1 is closed before RCB/G1. |
| MSM workload | Authentic capture ready | Aggregate-only seed-24302 capture proves 147 calls, 47.093M records, 1.054M runs, and 46.039M balanced reductions without persisting scalars, digits, bucket identities, or points. |
| MSM arithmetic | No promoted M1 Max route; stop before MSM | Scalar accumulation, global-tree, flat-segmented, fused-ceiling, and every complete-field candidate failed. Direct eight-lane AMNS8 is about 294x below its median field gate; 2/4-lane slot packing offers only 1.143x and is pruned. The agreed stop rule forbids RCB, point, tree, full-MSM, and further residency work on this M1 Max. Reopening requires materially new hierarchical arithmetic or newer hardware and a fresh Milestone-1 pass. The CPU+Metal resident-field epoch remains only a diagnostic oracle. |
| PLONK residency | Diagnostic quotient epoch measured and rejected for integration | Exact CUDA seams, arena layouts, source IDs, transcript epochs, and required operations are documented. A checked neutral graph compiler, deterministic key-wide family compaction, six Metal expression PSOs, startup ABI validation, and CPU/BLST diagnostic corpora pass. PK-local preparation and ParamsKZG-owned coefficient/Lagrange SRS preparation feed a session-cached schema-v2 Metal context that binds exact PK, domain, NTT ABI, and SRS identities and exposes checked field/periodic/G1 views. The diagnostic epoch performs resident extended transforms, all numerator categories, vanishing division, inverse transform, and exact split/blinding in one non-clear command, then exposes an exact callback-scoped CPU-MSM lease and clears its secret allocations. Independent q=2/3/4 differentials and an authentic private K=4 transcript/RNG parity harness pass. The representative five-proof safe-epoch median was 1,184.911 ms, 1.975x the 600 ms gate; optimistic end-to-end substitution is only 1.077x. Keep it as a native-MSM oracle, not a proof route. The general proof arena, remaining epochs, and production routing are deferred behind a fresh Milestone-1 pass. |
| Routing | Correctly unavailable | Proof-server preflights every entry point and rejects Metal before proof execution; runtime smoke success cannot unlock it. |
| Parity/performance gate | Instrumented, not enforcing | Paired samples and route/native/fallback metadata exist, but verification can be disabled and the CLI does not enforce sample count, 3x, absolute median/p95, materialization, or memory gates. A strict acceptance mode is required before routing. |
| CUDA regression | Source-level guard added | Hidden TLS arena handoff was replaced by explicit ownership and a narrow ignored CUDA error-isolation test. It still requires an NVIDIA runner and has not run locally. |
| CI | Compile gate ready | Portable lanes avoid blanket accelerator features and check the unsupported Metal surface; Apple silicon compiles the real metallib. Standard hosted macOS may not expose a GPU, so device/performance gates require an xlarge or physical runner. |

## Highest-priority findings

### 1. MSM arithmetic is the current critical path

The balanced transaction has approximately 57.1 million complete bucket and
integration additions under the initial trusted-local window plan. A 400 ms MSM
budget requires roughly 1.713B scheduled Fp multiplications/s, with a 2.0B/s
engineering target. Current complete-add throughput projects about 5.6 seconds
of arithmetic before routing, memory, scans, NTT, or the rest of PLONK.

The bounded scalar falsification campaign is complete. The authoritative
wide-accumulator result is a 1.540B/s no-go; explicitly alternating two and
three independent CIOS states worsens total throughput to 0.963B/s and 0.467B/s.
The tested attempt to change group-operation count also failed: the complete
global batch-affine device dataflow reached only 30.633M pairs/s.

The follow-up local-tree attempt passed its ABI and BLST-backed differential
suite but made the inversion dependency an occupancy bottleneck. Its faster
G128 variant reduced the representative sample in 2,136.823 ms, 48.3x above
the 44.261 ms promotion limit. Reducing dispatches from 1,569 to 17 and nominal
buffer traffic from 8.946 GB to 0.834 GB did not rescue the arithmetic mapping.

The final exact-operation probe removed dataflow to answer whether a
hierarchical scan merited implementation. At 164.206M/s it projects 280.374 ms
of arithmetic alone and leaves 19.626 ms inside the 300 ms bucket gate. It
failed the frozen 200M/s research threshold, so the omitted prefix/suffix
structure, inversion, synchronization, routing, and output path were not built.

That review is recorded in
`reports/ledger-9-metal-batch-affine-plan-2026-07-15.md`, and the authentic
capture in `reports/ledger-9-metal-msm-authentic-runs-2026-07-15.md` supersedes
its probability model. The real workload has 47.093M digit records and 46.039M
balanced-tree reductions, but maximum runs reach 16,797. The wide/global-tree
projection is about 382.5 ms arithmetic-only after optimistic integration, so
the pre-measurement projection was a likely `NO_GO`. The completed diagnostic
confirmed that verdict by a wide margin: 1,502.943 ms projected bucket-only,
versus the 364.6 ms immediate stop and 300 ms promotion gates. The raw samples
and complete boundary are in
`reports/ledger-9-metal-batch-affine-result-2026-07-15.md`.

### 2. The MSM leakage profile needs an owner decision

Witness-derived Pippenger digits expose bucket addresses, occupancy, work, and
timing. The strict fixed-base radix-16 design is reviewable but infeasible for
the 3x M1 Max target. A production-speed design therefore needs explicit
approval for the leakage-accepted trusted-local profile, including the trusted
host/OS/driver/profiler boundary and remote timing policy. Existing CUDA leakage
is context, not automatic authorization.

The run capture and replay need the same classification. Occupancy, run lengths,
and per-call fingerprints remain secret-derived even when scalar values, bucket
IDs, signs, and point indices are omitted. Mode `0600`, aggregation, and hashing
reduce exposure but do not declassify the data. Capture/replay artifacts remain
outside the checkout; tracked code must not freeze new per-call or
selected-chunk fingerprints without an explicit security-owner decision.
Concurrent recapture also needs a process-wide permit or proof token because
the current active-scope counter can observe matching MSMs from any thread while
a scope is open.

### 3. The acceptance command is not yet an acceptance gate

Canonical self-verification can currently be disabled with
`MIDNIGHT_PROOF_SELF_VERIFY=off`, and that environment is neither rejected nor
recorded in parity provenance. The parity binary defaults to three samples,
prints speedup, and exits successfully without enforcing the documented sample,
median, p95, native-compute, fallback, materialization, or memory limits.

Before routing Metal, add a strict gate mode that rejects disabled verification,
records verification count/time, verifies every measured output independently,
and enforces all absolute and relative thresholds. Diagnostics may continue to
print non-passing results without treating the Rust process as failed.

### 4. Durable diagnostic lifecycle is implemented

The resident Fr and quotient owners retain command resources through completion,
GPU-clear secret allocations before reuse, quarantine allocations whose erasure
cannot be proved, and durably poison the session after an unprovable clear.
Callback error and unwind paths are covered. Production still needs the same
contract generalized to the complete proof arena.

### 5. Preserve the host and standalone-MSM ownership split

`midnight-metal-prover` owns its session/lifecycle, proof ABI, arena, field/NTT
proof kernels, G1 diagnostics, PLONK epochs, standalone-MSM adapter, and
diagnostics. `bls12-381-metal-msm` separately owns all native MSM arithmetic,
routing, prepared-base state, Metal runtime/shaders, tests, and benchmarks. Each
package may have one coherent Metal translation unit; neither may import the
other in the reverse direction or duplicate device/command ownership implicitly.

The highest-value first adapter is exact and already has a correctness oracle.
`midnight-metal-prover/src/metal/quotient_epoch.rs::MetalSession::with_diagnostic_quotient_epoch`
produces the blinded quotient limbs, the facade currently exposes them through
`midnight-gpu-prover/src/lib.rs::metal::with_diagnostic_quotient_epoch`, and
`vendor/midnight-proofs-0.8.1/src/poly/kzg/mod.rs::diagnostic_commit_quotient_lease`
then invokes CPU `msm_specific` once per limb. After the standalone MSM passes,
replace that callback-scoped CPU lease with a checked resident scalar view and
coefficient `PreparedBases`; encode one ordered native batch before the `x`
challenge, return only compact G1 commitments, retain the quotient for
evaluations and multi-open, and erase it only after its final consumer. Do not
insert Metal into generic host-slice `msm_specific`, duplicate SRS buffers, or
create a second implicit device/queue.

Production point arithmetic is complete homogeneous RCB. Branch-completed XYZZ
may remain an optional trusted-local diagnostic for comparative research, but it
is not the production formula contract and cannot satisfy the production gate.

### 6. Finish backend neutralization at owned-resource seams

The proof-server selection contract is neutral, but the existing CUDA proof
route still owns concrete sppark types inside patched proof crates. Introduce
target-selected Fr buffer, key context, proof arena, and execution guard types
at the documented seams. Refactor one seam at a time and require CUDA proof-byte
parity before and after each change. Do not put Metal objects or dynamic trait
objects into generic protocol code.

### 7. Treat native CI as two different gates

The standard Apple-silicon lane is a compiler/linker and fail-closed-routing
gate. Native correctness requires a runner that exposes `MTLDevice`; sustained
performance requires a stable physical benchmark host or a declared
GPU-enabled runner. CUDA ownership/error-isolation needs a separately
provisioned NVIDIA lane. A skipped device test is not a performance result.

## Implementation plan and exit gates

### Phase A — Arithmetic decision (complete: direct AMNS8 no-go)

1. Six-`u64`, twelve-`u32`, wide-accumulator, and explicit two-/three-state
   scalar paths are retained as rejected baselines.
2. The seed-24302 run capture is retained as sensitive diagnostic evidence;
   aggregation and hashing do not declassify witness-derived occupancy.
3. The BLST-checked global batch-affine replay completed 20 release samples in
   one command/one wait with resident roots and no timed allocation or upload.
4. Its 30.633M pairs/s and 1,502.943 ms projection fired the predeclared
   immediate stop.
5. The separately bounded recursive flat-segmented branch completed paired
   G128/G256 20-sample runs with the same one-command/one-wait/status-only
   boundary. Both failed; G128 won at only 3.180M/s and a 14,479.771 ms full
   projection.
6. The exact register-local `7 M/S + 9 A/S` probe completed a 20-sample run at
   164.206M/s. Its 280.374 ms arithmetic-only projection failed the predeclared
   200M/s headroom gate for building a hierarchical scan.

Exit result: **failed**. There is no approved M1 Max MSM arithmetic route. The
algorithm-v2 separated Karatsuba/L4 candidate passed its portable model and
generated Metal correctness corpus but failed the standalone field gate's
56-word liveness condition at 70 words before performance. The 48-word
top-level subset is not an acceptance result. The algorithm-v3
blocked-radix-128 successor then passed correctness, optimized-AIR structure,
and source liveness at 55 words, but failed its 2.0B/1.8B performance gates at
465.444M median / 455.318M p5 complete L4 operations/s. The later resident
quotient experiment also failed its CPU-BLST performance gate. Pure Metal
routing and the 3x claim remain closed until materially different
hardware/arithmetic clears the standalone field, formula, and 400 ms aggregate
gates. No formula, tree, MSM, or further residency work was authorized after
the v3 failure. The existing CPU+Metal field implementation remains an
explicitly labeled diagnostic oracle. See
`reports/ledger-9-metal-blocked128-field-result-2026-07-16.md`.

Phase A is now closed on this M1 Max. The direct generated eight-lane AMNS8/K8
kernel completed the formal gate with 16,777,216 multiplications per sample,
two warmups, and 20 measured samples. Correctness, padding, and zero static
threadgroup-memory checks all passed, but median/p5 throughput was only
6,798,372.789/6,721,714.410 operations/s versus 2.0B/1.8B. The median miss is
about 294x. Packing the 243 logical products more tightly changes 288 physical
slots to only 252, a 1.143x improvement, so the prior 2/4-lane sweep is pruned.
PMNS-25 remains research, not an active implementation path. Phase C is not
authorized on this machine.

### Phase B — Stabilize host architecture

These items are retained as reusable architecture and maintenance guidance, not
an active residency implementation phase on this M1 Max. New work here resumes
only after a materially new Milestone-1 pass.

1. Split the native host module along the ownership boundaries above.
2. Add an explicit `Healthy/Clearing/Poisoned` session state. On execution
   failure, retain resources to completion, clear secret arenas, reject new
   encoding, and permanently poison the session if clearing fails.
3. Move probe-only readbacks and throughput kernels behind a diagnostics
   surface that production routing cannot call.
4. Replace global resettable activity snapshots with monotonic, proof-scoped
   counter baselines tied to the execution guard. Include live/peak bytes,
   materializations, stage dispatches/waits, retries, and fallbacks.
5. Add stable encoder/pipeline labels and profile-only shader source/line
   attribution, plus a required GPU-backed correctness lane.
6. Make proof verification non-optional in acceptance runs. Record verification
   count/time and add a strict parity mode that enforces sample count, absolute
   median/p95, 3x speedup, native compute, zero fallback/materialization, and
   memory gates instead of merely printing them.

Exit: exact primitive parity, no more than 5% same-host median throughput
regression across five alternating before/after samples, no duplicate
queue/session ownership, non-Apple checks pass, and injected failures prove
post-completion clearing plus session quarantine/poison behavior.

### Phase C — Resident MSM

This phase has no promoted implementation candidate. Its independently
extractable package boundary is frozen in `bls12-381-metal-msm`; both FP32
schedules and the complete two-lane `u32` FIPS mapping are rejected, and the
package intentionally has no execution-shaped API. Generated named-register
FIOS also failed its field gate. The algorithm-v2 subtractive-Karatsuba/L4
falsifier passed exactness but failed source liveness at 70 words versus the
56-word limit. The algorithm-v3 blocked-radix-128 successor passed correctness,
optimized-AIR structure, and the source gate at 55 words, but its
465.444M/455.318M median/p5 rates failed the 2.0B/1.8B throughput gates. Neither
enters this phase. Direct eight-lane AMNS8/K8 is also rejected at
6.798M/6.722M median/p5 operations/s. The 2/4-lane slot-packing branch is pruned
because its best physical improvement is only 288 to 252 slots. PMNS-25 remains
research. The following sequence is retained only for materially hierarchical
new arithmetic or newer hardware after a fresh Milestone-1 pass:

1. obtain the explicit strict-oblivious or trusted-local security decision;
2. implement canonical Fr recoding and exact recomposition;
3. implement histogram/scan/scatter only for the approved leakage profile;
4. implement accumulation/integration, ragged same-base batches, compact
   projective egress, and GPU erasure; and
5. differential-test 8191/8192, 16383/16384, and 32767/32768 shapes against BLST
   and the canonical CPU implementation.

Complete homogeneous RCB is mandatory for the production sequence. A
branch-completed XYZZ variant may be benchmarked only as an explicitly labeled
trusted-local diagnostic and does not replace the RCB correctness or throughput
gate.

The one-way Midnight adapter owns CUDA parity and may use immutable external
fixtures; neither CUDA nor sppark is a dependency of the standalone package.

Exit: across at least 20 authentic-trace samples, aggregate MSM median is at
most 400 ms and p95 is at most 500 ms, with zero intermediate polynomial
readback, zero fallback, and all lifecycle/security tests passing.

### Phase D — Neutral resident prover seams

The first neutral ownership seams are reusable foundation. Additional proof-
pipeline migrations are deferred until Phase C clears its standalone native-MSM
gate; completing them would not waive that gate or authorize pure Metal routing.

1. Replace concrete resident types in `ProverTrace`, proving-key caches, PCS
   hooks, and multi-open ownership with target-selected handles.
2. Preserve transcript/RNG order and CUDA output byte-for-byte after every
   seam migration.
3. Keep public proof code unaware of Metal resources.

Exit: CPU and CUDA parity remains exact; target feature graphs select one native
backend only; ownership tests show no TLS or cross-request contamination.

### Phase E — Resident PLONK epochs

After Phase C clears, implement in dependency order: advice boundary; lookup permutation;
permutation/lookup/trash products; expression graphs and numerator; quotient
division/splitting/blinding; ordered evaluations; Poseidon advice; resident
multi-open `q`, `f`, and `pi`. Wait only at the nine documented transcript
epochs and return only compact commitments/evaluations.

The expression/key substrate now includes neutral compilation, six specialized
Metal pipelines, ABI validation, checked diagnostic dispatch, CPU/BLST parity,
key-wide compaction, and a validated session-cached schema-v2 PK/SRS context.
Dynamic and immutable Fr sources have checked physical identity, algebraic
basis, exact NTT domain, and partial-Stockham progress, while NTT ABI v2 handles
packed-to-extended strides. The diagnostic resident numerator/quotient epoch is
implemented with one non-clear work command, one intentional CPU-MSM wait, exact
`q*n` lease egress, and GPU cleanup. Its independent q=2/3/4 field differential
and authentic private K=4 commitment/transcript/RNG parity harness pass. It has
now been measured on representative Ledger-9: the complete five-proof safe-epoch
median is 1,184.911 ms, or 1.975x the 600 ms budget (97.5% over). An optimistic
substitution model yields only 1.077x end-to-end, so its CPU-BLST continuation
will not be connected to proof finalization. It remains a correctness/lifecycle
oracle for the standalone native MSM. The BLST adapter still does not zeroize
its preprocessing copies, and no proof route uses it.

Exit: per-stage buffer parity, transcript-prefix parity after every epoch,
materialization count zero in production, and exact deterministic proof bytes.

### Phase F — Route and accept

1. Add `ProofExecutionRoute::Metal` only after every selected proof operation is
   native and the counter bridge is connected.
2. Run paired CPU/Metal zswap samples with independently enforced canonical
   verification, then
   repeat-100 lifecycle and bounded-memory stress.
3. Capture Metal System Trace and counters under the profiling build.

Exit: after two warmups, at least 20 paired alternating performance samples have
no hidden fallback/retry or whole-polynomial readback, Metal median is at most
`1,757.940 ms`, and Metal p95 is at most `2,000 ms` against the same-host CPU
baseline. The paired-run CPU median divided by the Metal median must also be at
least `3.0x`; the absolute limit prevents a slower CPU regression from creating
a false pass. A separate 100-attempt seed/lifecycle run must verify 100/100 proofs,
perform zero post-warmup arena growth, and keep measured Metal-owned bytes at or
below the checked arena/key-context capacity manifest.

## Recommended review and merge series

The current worktree deliberately contains the evidence needed to make the
architecture decision, but it should not merge as one change. Split it into:

1. contracts, CPU/Amdahl baselines, proof-flow map, and Rust 1.95 pin;
2. fail-closed backend selection, parity/profiler contracts, portable CI, and
   the minimal `midnight-metal-prover` build/unsupported-target scaffold needed
   by the facade's `metal` feature;
3. native Metal runtime, ABI, arena, Fr/NTT, Fp/G1 diagnostics, and their reports;
4. CUDA resident ownership and activity-counter hardening.

Series 4 is not mergeable until an NVIDIA lane compiles `gpu-blitzar` and runs
the ignored resident multi-open error-isolation regression. The repository has
no suitable runner label today; source review is not a substitute for that gate.

## Explicit non-goals and stop rules

- Do not route the proof-byte digest smoke kernel as a Metal prover.
- Do not port the CUDA kernel layout one-for-one without Apple measurements.
- Do not build full Pippenger around an arithmetic path that missed its fused
  gate.
- Do not claim unified memory makes host/device ownership transitions free.
- Do not add queues, concurrency, fusion, or SIMD variants without a trace or
  counter hypothesis that can falsify the change.
- Do not accept simulator, standard hosted-runner, or single-sample timing as
  3x evidence.
- Do not allow mid-proof CPU fallback; a service retry is a new, visible proof
  attempt from authoritative inputs.

The detailed contracts live in `docs/metal-backend.md`,
`docs/metal-resident-plonk-plan.md`, and `docs/metal-msm-security.md`. Primitive,
Amdahl, and MSM evidence lives in the other `ledger-9-metal-*` reports beside
this file.
