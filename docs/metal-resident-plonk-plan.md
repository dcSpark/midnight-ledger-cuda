# Metal-resident PLONK integration

This document is the current residency ledger and continuation plan for the
Apple Metal prover. It describes executable code, not a feasibility gate. Dated
research reports preserve rejected arithmetic and reducer candidates.

## Current result

The final settled seven-sample `metal-hybrid` optimization run measured a
1,128.954 ms median and 1,181.488 ms p95 on the five-proof balanced zswap
transaction. Against the frozen clean CPU median of 4,591.330 ms, that is a
4.067x absolute median speedup. All measured and warm-up transactions were
correct, the complete 147-output MSM workload ran, and there was no nonempty
CPU MSM fallback or failed Metal submission. The run produced 966 nonempty
Metal outputs in 240 coalesced submissions.

This is a short-run optimization result, not a replacement for the existing
20-pair production acceptance record. The last paired CPU median implies a
1,133.657 ms 4x line; the final short run is 4.703 ms below that proxy
threshold. A fresh 20-pair run and 100-transaction soak were intentionally not
run before handoff, so this remains a settled optimization result rather than
a new paired production acceptance record.

The final improvement narrows the process-session execution lock to proof
runtime commands and secure cleanup. Callback-scoped quotient and multi-open
buffers remain pinned while their standalone MSM commitments run, allowing the
independent proof-runtime and MSM queues to overlap. All 117 active Metal
runtime tests passed, including panic cleanup, clear-failure quarantine,
resident quotient parity, and resident multi-open parity.

Two broader overlap attempts were rejected and removed. Two complete Metal
sessions regressed to 1,144.959 ms median and 1,307.473 ms p95 while increasing
uploads and key-context duplication. Removing the proof-facing quotient audit
lock remained byte-exact in its falsifier but measured 1,185.233 ms, so the
serialized exact-counter audit was restored.

The route is mostly accelerated but not fully resident. One combined quotient
opening polynomial per proof crosses from Metal to CPU, and multi-open receives
host-precombined q polynomials. Those boundaries are explicit and prevent a
pure `metal` label.

## Architectural boundaries

The implementation has three ownership layers:

1. `bls12-381-metal-msm` is an application-neutral BLS12-381 G1 library. It
   owns fixed/variable-base MSM, prepared bases, resident scalar views, command
   submission, validation, and erasure.
2. `midnight-metal-prover` owns reusable field/NTT primitives, immutable proving
   key contexts, proof-scoped resident arenas, quotient and multi-open kernels,
   scheduling, counters, and lifecycle enforcement.
3. `midnight-proofs` and `midnight-gpu-prover` own PLONK semantics, transcript
   order, typed adapter handles, and backend selection.

Dependencies point downward. The standalone MSM crate must never import PLONK,
transcript, proving-key, or ledger types.

## Residency invariant

A resident handle binds all of the following:

- one Metal device registry identity and process session;
- an allocation generation and checked byte range;
- scalar representation, polynomial basis, exact domain, stride, and batch
  shape;
- immutable or mutable intent and aliasing rules;
- producer completion and callback lifetime; and
- the owner responsible for full-capacity erasure.

Foreign devices, stale generations, basis/domain relabeling, out-of-bounds
subviews, unproven producer completion, or ambiguous cleanup fail closed. Raw
`MTLBuffer` pointers are not a proof-system API.

## Implemented pipeline

### Key preparation

The schema-v2 key context binds the exact domain and roots, classified
expression graphs, fixed and permutation columns, periodic selectors, cosets,
and both KZG SRS bases. Cache identity includes content fingerprints; cache
misses repeat field, curve, and subgroup validation before upload.

Prepared coefficient and Lagrange SRS prefixes are shared by clones and bound
to their requested domain size. Standalone MSM preparation expands public
window-shifted fixed bases and validates the shader ABI before proof work.

### Trace and commitments

Witness synthesis remains on CPU. Independent proof jobs can synthesize and
prepare concurrently. Eligible coefficient/Lagrange conversions use the
resident Metal NTT arena. Advice, permutation, lookup, and trash commitments
use the standalone fixed-base MSM worker, which batches compatible requests
across proofs.

The production worker now owns two Metal MSM queues. Requests are sharded by
prepared basis rather than round-robin: Lagrange commitments remain on one
queue while coefficient and resident-opening commitments use the other. This
preserves same-key coalescing while allowing the two commitment streams to
overlap. `MIDNIGHT_METAL_MSM_WORKERS=1` remains the diagnostic control.

### Quotient

The Metal quotient epoch consumes checked trace/key sources and transcript
challenges. It evaluates the classified numerator graphs, performs the domain
transforms and vanishing division, splits and blinds quotient pieces, and lends
their resident Montgomery scalars directly to fixed-base MSM. Quotient pieces
are never materialized for CPU commitment.

After their commitments are transcript-bound and `x` is sampled, Metal combines
the pieces with the required splitting factor. The current continuation reads
that one combined `n`-coefficient opening polynomial to CPU. It is then used by
the ordinary finalizer and uploaded with the other multi-open inputs.

### Evaluations and multi-open

Ordered PLONK evaluations and transcript writes remain CPU-owned where their
inputs are host polynomials. The Metal multi-open path precombines each point
set's q polynomial on CPU, then runs q division, f construction, compact q
evaluation, pi construction, and both f/pi commitments inside a callback-scoped
resident arena. Only compact evaluations and curve outputs cross the boundary.

Precombining q on CPU is currently faster than uploading every source
polynomial. A tested Metal lincomb version increased logical H2D from about
123 MiB to about 327 MiB per transaction and regressed the median. This result does
not reject resident q construction; it requires direct reuse of already-resident
sources.

## Transcript epochs

Residency must not reorder Fiat–Shamir operations. The important boundaries are:

1. trace commitments are written before beta/gamma/theta/trash challenges;
2. quotient pieces are blinded and committed before `x` is sampled;
3. ordered polynomial evaluations are written before multi-open challenges;
4. f is committed before `x3`;
5. q evaluations are written before `x4`; and
6. pi is committed last.

Metal callbacks may execute polynomial work between these gates, but compact
outputs are written by the canonical transcript implementation in exactly the
CPU order. Byte-exact paired tests are the final oracle for this contract.

## Failure and cancellation

Proof-scoped secret allocations cannot escape their callback or arena. Cleanup
runs on success, error, and panic. Every leased capacity is cleared before pool
reuse. If completion or clearing cannot be proven, the capacity is quarantined
and the session is permanently poisoned. No selected Metal operation retries on
CPU.

The standalone MSM worker separately retains resident scalar buffers through
command completion and erases its own routing, bucket, integration, output, and
status scratch. The producer remains responsible for erasing caller-owned
resident source buffers after their final consumer completes.

## Remaining path to pure `metal`

### 1. Transfer the quotient continuation

Replace the host `Vec<F>` continuation with an owned, generation-checked
resident quotient-opening handle. Ownership must transfer from the quotient
epoch into final evaluation/multi-open and retain the buffer until both finish.
The final owner clears it. This removes five whole-polynomial materializations
per transaction and avoids re-uploading the same opening.

The safe API must not allow the handle to outlive its session, alias mutable
scratch, or skip erasure on a transcript or MSM error. A callback that encloses
the rest of finalization is acceptable if an owned transfer type becomes too
complex.

### 2. Build a shared resident polynomial catalog

Assign stable proof-local IDs to advice, instance, lookup, permutation, trash,
fixed, and quotient polynomials. Multi-open metadata should reference those IDs
plus weights and point sets. Host-only polynomials may be uploaded once into the
same catalog; already-resident sources must not be copied through CPU memory.

The q lincomb kernel already supports general source metadata. The missing part
is lifetime/identity plumbing from trace and key contexts, not arithmetic.

### 3. Resident ordered evaluations

Evaluate catalog polynomials at the required rotations on Metal and read only
the ordered scalar vector. Bind every evaluation request to the exact query
order and domain rotation to prevent transcript permutation bugs.

### 4. Continue MSM optimization by end-to-end delta

MSM is still the largest device interval. Optimize the standalone library, then
accept a change only when the paired transaction improves without reliability,
transfer, or lifecycle regression. Candidate-specific field or reducer failures
do not stop this work; they reject only that representation and schedule.

## Completion gates for pure `metal`

Promotion from `metal-hybrid` to `metal` requires:

- no whole-polynomial quotient materialization;
- no nonempty CPU MSM or hidden retry;
- no host reconstruction of a polynomial that already exists in a resident
  catalog;
- exact proof/transcript/serialized-result parity;
- all transfer, materialization, fallback, clear, quarantine, and poison
  counters audited by the acceptance harness;
- at least 20 paired samples and a 100-transaction reliability soak; and
- no regression below the accepted 3x end-to-end floor.

The CPU may continue to own orchestration, transcript/challenge logic, witness
synthesis, compact inputs/outputs, and self-verification. Pure `metal` means the
high-cost polynomial/commitment hot loop is resident; it does not mean a
GPU-only application process.

## Performance ceiling

The historical Amdahl analysis leaves about 512 ms of host-only work and gives
an infinite-device structural ceiling around 10.2x. The current measured median
is about 1.405 s. The ceiling is useful for prioritizing residency work but is
not a promised M1 Max result. Every optimization claim must use the complete
paired fixture, including synchronization, mappings, cleanup, transcript work,
and proof verification.
