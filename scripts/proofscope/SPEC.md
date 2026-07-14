# proofscope - proof-server GPU-resident prover diagnostics, spec v2

One-sentence spec: a reproducible proof-server diagnostics harness where every
timing number is tied to deterministic correctness evidence, every CUDA event is
mapped to proof ownership, and every claimed speedup preserves the timed
contract.

## North Star

The target architecture is not a CPU prover with CUDA helpers. The target is a
GPU-owned proving pipeline:

- SRS bases, fixed columns, advice columns, lookup/permutation products,
  quotient/intermediate polynomials, and opening inputs live in device memory
  for the lifetime of a proof, and ideally across repeated proofs for stable
  key material.
- CPU remains the owner of request decode, key lookup, Fiat-Shamir transcript
  boundaries, response encoding, and byte-exact parity checks.
- CPU/GPU synchronization is allowed only at protocol boundaries that need
  transcript-visible bytes: commitments, evaluations, challenges, final proof
  bytes, and explicit validation artifacts.
- MSM, FFT/NTT, prefix product, inversion, and vector kernels are internal
  primitives inside the resident proof flow. They are not the architecture.

The sppark integration implements that resident architecture for eligible large
domains. Advice, instance, lookup/permutation/trash products, numerator,
quotient, transcript evaluations, and multi-open state remain on device between
transcript gates. The synthetic k=17 fixture demonstrates `15.586x`, but it is
not the optimization score. The primary gate is a complete ledger zswap
transaction: actual wallet and Dust fee balancing, two shielded spends, payment
and change outputs, one Dust spend, transaction proving, binding seal, and final
serialization. The accepted baseline is CPU 3753.733 ms versus CUDA 392.970 ms,
`9.552x`, byte-exact across five seeds. Zswap spend and output use the resident
path; the accepted run keeps the k=13 Dust relation on the leaf bridge. The
current tree contains an opt-in k=13 resident candidate with size-aware
sequential argument scheduling and lane-unique sppark device cursors. It is not
enabled by default because CPU-first repeat parity still fails at later seeds.

The concrete proof computation graph is maintained in
`scripts/proofscope/PROOF_FLOW.md`. Treat it as the implementation map for the
next CUDA slices: transcript gates are hard dependencies, while whole-polynomial
host/device round trips between those gates are implementation debt.

## Scope

`proofscope` is the ledger-side port of the Nightstream GPU workflow. NVIDIA
tools acquire facts; this repo-specific layer interprets them against the
proof-server dataflow and the resident-prover contract above.

Implemented as of 2026-07-11: the offline analyzer below, plus nsys
collection (`gpuprof.sh nsys`, builds with perf-timers and sets
`MIDNIGHT_PROVER_TIMERS=1`, exports SQLite, analyzes, appends
`benchmark-results/gpuprof-history.jsonl`), static kernel lint
(spill/occupancy/warp/grid from launch configs; no NCU counters needed),
per-stage and whole-run transfer ledgers, provenance stamping (git, GPU,
driver, MIDNIGHT_* env), a terminal report, `diff` with structural causes,
the exit-coded `check` regression gate, `trend`, deterministic parity, transfer
counters, and an accepted resident-prover baseline. NCU and sanitizers remain
follow-ups.

The analyzer slice:

- read a Nsight Systems SQLite export from a bundle;
- use NVTX or stamped timer ranges as stage ownership;
- decompose each stage into GPU busy, transfer wait, sync wait, CUDA API,
  host gap, and unattributed time;
- write machine-readable `analysis/stages.json` and `analysis/levers.json`;
- fail when reconciliation leaves hidden time.

NCU and sanitizer orchestration are follow-up steps. The
shell wrapper keeps those command names visible, but it should not pretend each
mode is wired before resident proof stages emit stable NVTX/perf-timer ranges
and bundle metadata. Nsight Systems collection, `diff`, `check`, and `trend`
are wired for the parity gate.
The first wired gate is `proof-backend-parity`, which defaults to the complete
`zswap-transaction` workload, times deterministic CPU and sppark-backed CUDA
execution, and compares the final sealed ledger transaction exactly.
That gate also reports per-iteration and total H2D/D2H byte counters from the
sppark FFI boundary. Those counters are a residency-pressure diagnostic for the
normal parity loop; Nsight Systems remains the source of truth for actual CUDA
transfer timing in collected bundles.

## Large-Domain Substrate Milestones

These substrate milestones landed for the eligible large-domain diagnostic;
they do not mean the zswap product gate has reached 10x:

1. **Device polynomial store - complete.** Key-scoped SRS/fixed/permutation
   material and proof-scoped advice/instance arenas have explicit device
   ownership and basis-aware operations.
2. **Resident proof blocks - complete.** Lookup/permutation/trash products,
   numerator evaluation, quotient construction, transcript evaluations, and
   KZG multi-open consume resident handles without normal-path polynomial
   readback.
3. **Dependency-aware scheduling - first latency slice complete.** Independent
   permutation and lookup products overlap, same-base MSM uses four lanes, and
   transcript egress remains ordered. Cross-proof streams and CUDA graphs are
   the next throughput/launch-overhead milestone.
4. **Direct resident advice - complete.** The terminal Poseidon fixture uploads
   a compact 16 MiB witness, expands eight columns in device memory, applies
   exact host-RNG blinding cells by indexed scatter, and commits without the old
   32 MiB host advice allocation/upload.

The large-domain result is accepted only with exact output and the fair timing
contract. Domains below `2^15` use the validated CUDA leaf path because the
resident finalise layout is not enabled there.

## Proof-Server Taxonomy

Stage labels describe ownership and dependency, not incidental function names.
The current taxonomy starts with:

- `proof_server.request.*` - payload read/decode and key or IR resolution.
- `proof_server.prove.*` - IR dispatch, provider setup, proof execution, and
  proof checking.
- `proof_server.prove.cuda.*` - GPU-owned proof stages: residency setup,
  device polynomial transforms, expression evaluation, device commitments,
  transcript egress/ingress, and final proof egress.
- `proof_server.transaction.*` - transaction proving path through ledger/zswap.
- `zkir.*`, `zswap.*`, `ledger.*` - lower-layer ownership when instrumentation
  lands below the HTTP wrapper.

The source of truth is `scripts/proofscope/taxonomy.py`. Unknown NVTX labels are
reported explicitly rather than silently folded into known rows.

## Fair Speedup Contract

A CUDA candidate only counts as faster when all of these stay true:

- deterministic CPU-vs-GPU parity is exact for the main gate's complete sealed
  and serialized ledger transaction;
- public inputs, skip-output metadata, and proof check results match;
- validation/replay work is outside the fast timing window, or the CPU and GPU
  benchmarks both include the same validation contract;
- H2D/D2H traffic inside the repeated hot loop is visible and does not regress;
- host synchronization inside the repeated hot loop is visible and does not
  regress;
- repeat medians improve against an accepted baseline, not merely the previous
  run;
- skipped export, serialization, or response work is either skipped by both
  sides or reported as a separate contract change.

The deterministic test RNG plumbing in `proof-server` is part of this contract:
it gives parity and replay tests a stable input source without changing
production `OsRng` behavior.

## Artifact Bundle

The target bundle shape is:

- `metadata.json` - git/config/tool/GPU identity.
- `stdout.txt` and `stderr.txt` - complete process output.
- `nsys/report.sqlite` and the matching `.nsys-rep` - raw Nsight Systems truth.
- `ncu/` - Nsight Compute reports for selected kernels.
- `sanitizer/` - Compute Sanitizer logs.
- `analysis/stages.json` - stage attribution and reconciliation.
- `analysis/levers.json` - ranked optimization levers.

`proofscope analyze <bundle-dir>` currently accepts `nsys/report.sqlite`,
`report.sqlite`, `proofscope.sqlite`, or `nsys.sqlite`.

Nsight Systems collection is wired only for the parity gate:

```bash
scripts/gpuprof.sh nsys proof-backend-parity --repeat 1
```

The collector builds `target/release/proof-backend-parity` first, then profiles
the binary directly so Cargo compilation is not included in the trace.

## Parity Benchmark Gate

Run the one primary optimization gate with:

```bash
scripts/gpuprof.sh parity --repeat 3 \
  --json benchmark-results/gpuprof-parity-zswap-transaction-candidate.json
```

`parity` defaults to `--backend both --fixture zswap-transaction` and enables
the required CUDA feature. Setup creates a deterministic funded `TestState`
with shielded UTXOs of 60 and 70 plus a Dust-generating fee UTXO. Each timed
iteration then:

1. Constructs an unbalanced ledger transaction paying 100 shielded units.
2. Runs the repository's actual `TestState::balance_tx`, selecting both
   shielded UTXOs, creating 30 units of change, computing fees, and selecting
   one Dust UTXO.
3. Runs ledger structural well-formedness with balancing and limits enforced.
4. Calls `Transaction::prove`, which fans out two spend proofs, two output
   proofs, and the Dust fee-spend proof.
5. Self-verifies every proof, seals transaction bindings, serializes the final
   ledger transaction, and compares every CPU/CUDA byte.

Funded state and current-stack key generation are outside the timing window.
All transaction construction, balancing, validation, proving, sealing, and
serialization above are inside it. The accepted repeat-5 median baseline is:

| Fixture | CPU | CUDA | Exact | CUDA transfer per iteration |
| --- | ---: | ---: | ---: | ---: |
| `zswap-transaction`, 2 spends + 2 outputs + 1 Dust spend | 3753.733 ms | 392.970 ms (`9.552x`) | 23474-23476 bytes, 5/5 | 94.444 MiB H2D, 23.529 MiB D2H |

Source:
`benchmark-results/accepted-baselines/proof-backend-parity-zswap-transaction-sppark.json`.
At this CPU time, 10x requires at most 375.373 ms and 15x requires at most
250.249 ms on CUDA. The ratio is reported together with absolute CUDA latency.
Relative to the previous accepted artifact, CUDA fell from 467.749 ms to
392.970 ms. H2D increased slightly because residual 8K-32K MSMs remain on CUDA;
the transfer ledger makes that scope explicit.

The accepted MSM reducer keeps weighted window integration on device and folds
only weighted rows plus the small batch-partial array on the host. A fully
device-side final collector reached 373-379 ms in short runs but failed later
deterministic seeds and CPU-first cache-lifecycle parity, so it is not enabled.
Resident ragged lookup-input MSM is also implemented behind
`MIDNIGHT_GPU_LOOKUP_RESIDENT_RAGGED=on`; stable device compaction and base
offsets are exact, but the k=14/15 live sets are too small to amortize separate
Pippenger launches and regress this gate to roughly 412-421 ms. The mixed path
remains the measured default.

The production wallet rebalancer is not present in this repository;
`TestState::balance_tx` is the closest in-repo implementation and operates on
the real ledger transaction, zswap wallet state, fee model, and Dust state. The
fixture uses the checked-in zswap and Dust relations with generated
current-stack keys and an explicit current-prover adapter. Normal production
key dispatch remains unchanged and still needs coordinated re-keying before
this path can replace compatibility proving.

The other fixtures are diagnostics, not competing benchmark scores:

- `zswap-spend` validates deterministic liveness compaction of its 401 cached
  intermediates into the CUDA evaluator's bounded live-slot arena. It now
  measures 1157.565 ms CPU versus 271.581 ms CUDA (`4.262x`) byte-exact.
- `zswap-output` validates the coherent k=14 resident finalise, argument, and
  multi-open policy. The shared default remains k=14 so those ownership modes
  cannot diverge.
- The accepted k=13 Dust result remains the non-resident leaf path. A resident
  candidate is implemented and passed short exact runs, but post-cursor-fix
  CPU-first repeat parity still fails and the candidate remains opt-in.
- `hash-chain-v2` demonstrates eligible large-domain residency and reached
  `15.586x`, but cannot be used as the product headline.
- `minimal-v2` and the small hash-chain gate remain smoke/safety checks.

The emitted JSON includes transfer counters and per-operation breakdowns so a
candidate cannot hide whole-polynomial movement. The CUDA transaction provider
schedules the five deterministic proof jobs as real blocking tasks. The current
tree includes opt-in k=13 residency and fixes an sppark batch-addition cursor
alias that allowed concurrent proof lanes to corrupt shared device work state.
The remaining architecture must resolve the later-seed k=13 failure, then batch
ready work across proofs without changing the sealed transaction bytes.

The cuda-oxide toolchain path is exercised separately with:

```bash
scripts/gpuprof.sh cuda-adapter
```

That gate builds `proof-server-cuda` with `cargo +nightly-2026-04-03 oxide`,
runs a real GPU digest kernel over deterministic bytes, and checks the digest
against the CPU adapter implementation. It is an adapter/toolchain smoke gate,
not a proof-prover speedup claim. If cuda-oxide becomes strategic, the first
real kernel it should own is the resident numerator evaluator; otherwise it
should stay out of speedup claims.

## Current Attribution Targets

The next main-gate profile should attribute the complete transaction rather
than a single synthetic proof:

- unbalanced transaction construction and `balance_tx` ownership;
- shielded coin selection, change construction, fee calculation, and Dust
  selection;
- both spend proofs and both output proofs as separate ownership ranges;
- the Dust fee-spend proof as a separate ownership range;
- the current 94.444 MiB H2D and 23.529 MiB D2H per iteration;
- time waiting on each proof's sppark lane lock versus useful kernels;
- the Dust relation's k=13 resident parity boundary;
- overlap across five proof execution lanes and ordered transaction joining;
- binding seal, serialization, self-verification, and key-context reuse for all
  three circuit types.

Nsight Systems remains first for ownership, overlap, transfers, and host gaps.
Use Nsight Compute only after Systems identifies a hot kernel, and keep
Compute Sanitizer in the correctness loop for any new pointer-owning kernel.

## First Things That Happen

The primary transaction gate begins in this order:

1. Clone the pre-funded deterministic ledger/wallet state.
2. Construct the unbalanced shielded payment transaction.
3. Run shielded and Dust fee balancing.
4. Check structural well-formedness, balancing, and limits.
5. Fan out transaction proof jobs by their actual dependencies.
6. Self-verify each proof, join the proven transaction, and seal bindings.
7. Serialize the final transaction and compare CPU/CUDA bytes.

Each individual `/prove` request used by a production proof provider begins in
this order:

1. Read the HTTP payload.
2. Deserialize `(ProofPreimageVersioned, Option<ProvingKeyMaterial>, Option<Fr>)`.
3. Build a resolver over public params, dust params, and optional request key
   material.
4. Optionally rewrite the proof preimage binding input.
5. Resolve proving key material if the request did not inline it.
6. Dispatch by proof preimage and IR version.
7. Load the IR source and construct a local proving provider.
8. Execute the prover.
9. Reuse the skip-output metadata produced by that prover execution.
10. Serialize the versioned proof response.

The lower `transient_crypto::ProofPreimage::prove` path resolves key material,
uses cached parsed IR/prover-key objects when the source bytes match, executes
the ZKIR prover, fetches verifier params by `k`, and self-verifies the proof.
The full raw key bytes participate in cache equality, so replacing key material
cannot silently reuse a stale parsed key.

The resident prover then follows this flow:

1. Decode request and resolve/cached key material on CPU.
2. Create or reuse a device proof context for the proving key/SRS/fixed columns.
3. Upload witness/advice and public input data once.
4. Run device-owned proof phases, emitting only transcript-visible commitments
   and evaluations to CPU.
5. CPU samples Fiat-Shamir challenges and sends challenge scalars back.
6. Run device-owned opening/finalization phases.
7. Finalize proof bytes on CPU and run the normal self-verification.

Any design that repeatedly downloads whole polynomials, rebuilds SRS bases, or
calls the GPU only for individual leaves is a compatibility bridge, not the
target flow.

## Independent Work

Current code evidence for parallelism:

- `/prove` is single-proof at the HTTP layer; most useful early parallelism is
  inside key/material decode and GPU-resident proof phases, not across multiple
  proofs unless the caller batches requests externally.
- `/prove-tx` already exposes coarse parallelism: intents, guaranteed coins,
  and fallible coins are joined concurrently.
- Each intent proves contract actions with `join_all`; dust actions appear
  independent of those actions and are a candidate for an explicit concurrent
  split.
- `zswap::Offer::prove` proves inputs, outputs, and transients concurrently.
- Each transient proves its input proof and output proof concurrently.
- Dust spends are independent across spends and already use `join_all`.

The machine-readable version of this map is emitted as
`analysis/dependencies.json`. It should evolve with the code: when a stage is
forced by transcript/challenge order, mark it as a dependency; when old CPU call
order is accidental, put the leaves in a parallel group.

## Latency vs Throughput

The headline is final-transaction latency: currently 392.970 ms, `9.552x`
against the matching 3753.733 ms CPU gate. The 10x and 15x latency lines are
375.373 ms and 250.249 ms respectively.

Device throughput remains a supporting diagnostic. This fixture already
contains five independent proof jobs, so stream overlap should reduce the same
client-visible transaction latency rather than being reported as a separate
synthetic batch win. Never substitute an isolated circuit, offer, or hash-chain
result for the final-transaction score.

## Kernel Stack Policy

ICICLE is not an assumed dependency for this repo unless licensing/access
changes. The current open path is sppark plus local CUDA/C++ or cuda-oxide
kernels, informed by OSS halo2 GPU prover designs. Before hand-rolling each
large subsystem, study open implementations for device-resident polynomial
stores, expression evaluators, sort/permutation construction, and batched
opening flows; copy architecture only when license and curve support fit.

The sppark fork exposes immutable per-context MSM configuration rather than
process-global tuning variables. `Bls12_381MsmConfig` is selected once from the
proof domain k and travels with the proof execution lane and context-cache key.
It owns the residual-MSM floor, unfiltered-MSM floor, window bits, sort grid and
window lanes, internal batch lanes, event-reuse policy, and final-reducer mode.
For k=13/14/15 the
accepted RTX 4090 profile is 8192 residual scalars, 16384 unfiltered scalars,
window 10, sort divisor 3 with two window lanes, six batch lanes, and fenced
per-engine event reuse. Larger/unknown domains retain conservative thresholds.
Environment overrides remain diagnostic only; concurrent proofs cannot mutate
one another's configuration.
