# Metal G1 MSM security contract

This contract applies to witness-derived BLS12-381 G1 commitments executed by
the standalone Metal MSM and its Midnight adapter. Correct group results and
proof parity are necessary but do not by themselves establish confidentiality.

## Supported observation profile

The production implementation exposes
`TrustedLocalDataDependentV1`. It assumes the process, host OS, Metal driver,
profiling facilities, and local GPU co-tenants are trusted.

The following values or effects depend on witness-derived scalars:

- signed-window digits and zero positions;
- bucket indices, occupancy, scatter destinations, and long-bucket tiling;
- indirect work counts and the number of point additions; and
- command duration and memory traffic.

The route is therefore not constant-time or constant-observation at the Metal
software/command level. Random PLONK blinding hides published commitments; it
does not declassify scalar digits or the local execution trace. Do not use this
profile where an attacker can observe GPU addresses, occupancy, counters, or
fine-grained timing. Remote request timing also needs a separate policy if an
untrusted client can collect repeated samples.

Erasure, complete exceptional-case handling, and fail-closed routing are still
required under this trusted-local profile. Accepting trace leakage does not
permit malformed-input behavior, hidden fallback, stale resident handles, or
secret buffer reuse without clearing.

## Confidential values

Unless a caller proves otherwise, all of these are secret-derived:

- canonical or Montgomery scalar limbs;
- recoded digits, histogram counts, offsets, cursors, and signed point records;
- bucket, tile, integration, and window-collection points;
- quotient and multi-open polynomials and intermediate evaluations; and
- status/output staging that may encode a secret-dependent execution result.

Prepared KZG bases, domain metadata, job lengths, and circuit shapes are public
for the current threat model. A different application must revisit that
classification before reusing the standalone library.

## Input and SRS validation

The public scalar type accepts only four little-endian limbs strictly below the
BLS12-381 scalar modulus. Resident Montgomery input is canonicalized and checked
on Metal before routing; a residue at or above the modulus rejects the command.

Prepared affine points must have reduced Fp Montgomery coordinates, must not use
the all-zero identity encoding, must be on BLS12-381 G1, and must be in the
prime-order subgroup. BLST 0.3.16 supplies independent host validation before
upload. Prepared allocations are bound to their engine context, Metal device,
base count, and fixed-window expansion.

Midnight adds a second preparation layer. The key-context identity binds the
coefficient and Lagrange SRS fingerprints, exact domain, and proving-key
content. A cache miss repeats field/curve/subgroup validation. A cache hit can
reuse only that immutable validated identity.

These checks validate representation and content identity; they do not prove
the cryptographic authenticity of the trusted setup. Production key ingestion
must separately authenticate a known-good setup or verify the required
coefficient/Lagrange and G2 consistency relations.

## Versioned ABI and range checks

Engine initialization probes the production metallib for:

- public ABI version;
- generated field/scalar constant fingerprint;
- layout fingerprint, sizes, alignments, and offsets;
- fixed-route v10 parameters and tile thresholds; and
- status masks and sentinel values.

Mismatch fails before proof data is submitted. All counts, products, byte
ranges, base/scalar offsets, job outputs, bucket ranges, tile capacities,
resident-source extents, and reserved fields are checked with overflow handling.
Kernel status is atomic and validated before output acceptance.

The production `metal` feature compiles the production shader root. Historical
arithmetic probes and experimental reducers require `metal-diagnostics` and are
not part of the accepted proof artifact.

## Point arithmetic

The fixed-base v10 route uses XYZZ state to retain `Z^2` and `Z^3`. Addition
handles identity, doubling, and opposite-point cases and is differentially
tested against BLST. Lane and tile accumulators initialize directly from their
first valid affine point; this removes an addition with identity without using
an incomplete formula for later terms.

The implementation has scalar-dependent branches and work, which is permitted
only by the selected trusted-local profile. A future strict profile must replace
this with fixed-work formulas and public-address table/reduction schedules; the
current correctness evidence must not be relabeled as an oblivious trace.

Output is converted to the raw Montgomery Jacobian ABI and checked through the
independent proof/transcript oracle. The CPU adapter performs no nonempty group
arithmetic after Metal selection.

## No hidden fallback

Selecting the production Metal route enables native MSM by default. Any
initialization, preparation, validation, allocation, command, kernel-status,
completion, or erasure failure returns an error to the proof. The standalone
library has no CPU curve implementation and cannot silently retry.

The application adapter has one explicit all-zero shortcut: an MSM with no
nonzero scalars returns the public group identity. Acceptance counters record
these outputs and their zero nonzero-scalar count. Every nonempty authentic
commitment executes on Metal.

Environment controls that explicitly select CPU work are diagnostic overrides.
Results collected with such an override cannot support a production Metal
fallback or residency claim.

## Secret lifecycle

Host scalar-derived vectors use zeroizing ownership. GPU buffers for
canonicalized scalars, histograms, offsets, cursors, records, jobs, bucket/tile
roots, integrations, outputs, and status are erased after the work command
completes. Erasure covers full allocated capacity, not only logical length.

The engine serializes one scratch lifecycle so it can prove command completion
before reuse. If erasure completion cannot be established, the engine is
poisoned and rejects subsequent work. Validation errors that submit no command
do not poison an otherwise healthy engine.

The Metal prover runtime applies the same rule to proof arenas, quotient epochs,
NTT scratch, and multi-open buffers:

```text
allocate/lease -> encode -> submit -> wait -> consume compact output
               -> clear full capacity -> wait -> recycle
```

Error and panic paths also clear. A failed clear quarantines the capacity and
poisons the session. Drop is a cleanup backstop, not the sole success-path
mechanism.

Caller-owned resident scalar buffers are retained through MSM completion but
remain the producer's erasure responsibility. Checked resident views bind
device identity, storage range, representation, and synchronization. A foreign,
misaligned, out-of-range, or nonreduced view is rejected.

## Concurrency

`MetalMsmEngine` is deliberately thread-affine and uses `Rc`; it does not claim
`Send` or `Sync` for Objective-C ownership. Midnight runs it on one dedicated
worker. Compatible requests can be coalesced into one batched command, while
independent proof jobs overlap CPU preparation.

Request senders retain their inputs until the worker replies. Multi-source
resident batching retains every source buffer until the shared command
completes. Queue closure or worker failure returns an error; it does not detach
an in-flight buffer or invoke CPU MSM.

## Diagnostic data

MSM workload captures, digit/run histograms, bucket occupancies, per-call
fingerprints, and exact timing traces remain sensitive even when scalar values
are removed or hashed. Untracked captures must use owner-only permissions and
must not be committed without explicit declassification. Normal logs expose
aggregate counts and durations, not digit/bucket contents or witness-derived
fingerprints.

The process-global diagnostic scope can observe concurrent proof workers. A
future recapture must serialize the process or propagate a proof-specific
capture token.

## Verification evidence

Current device coverage includes:

- scalar modulus boundaries and malformed resident residues;
- every variable window and fixed preparation width;
- empty, single, prefix-batched, ragged, and multi-source resident jobs;
- repeated, opposite, identity-result, cancellation, and long-bucket cases;
- cross-engine/preparation identity rejection;
- command completion, clear failure, panic cleanup, quarantine, and poison
  injection; and
- paired byte-exact proof/transcript/serialized-transaction comparison.

The 20-pair production acceptance produced 2,840 Metal MSM outputs with zero
MSM failure and zero fallback. A 100-pair soak produced 14,200 outputs with the
same result. Exact commands and latency data are in the
[acceptance report](../reports/ledger-9-metal-production-acceptance-2026-07-17.md).

## Known exclusions

The current security claim excludes:

- hostile local GPU co-tenants, driver/kernel compromise, and physical
  power/EM/fault observation;
- a constant-work or constant-address MSM trace;
- authenticated trusted-setup ingestion inside the standalone library;
- asynchronous process termination during a Metal command; and
- a pure-Metal proof claim while the combined quotient opening is materialized
  to host memory.

These exclusions are deployment constraints, not permission for silent
fallback or incomplete cleanup.

## Future strict profile

A strict local-oblivious profile would require public-shape dispatches,
fixed-count scalar recoding, no zero compaction, public-address table scans or a
reviewed oblivious selection mechanism, and fixed reduction/addition schedules.
It needs a separate profile enum, kernels, benchmarks, threat-model approval,
and negative trace tests. It must not replace the trusted-local implementation
silently because its performance and leakage contracts are materially
different.
