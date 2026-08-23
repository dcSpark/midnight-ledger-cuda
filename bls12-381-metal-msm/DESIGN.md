# Standalone design contract

`bls12-381-metal-msm` owns application-neutral BLS12-381 G1 MSM for Apple
Metal. A ledger or proof system is a consumer, never a dependency.

## Layers

1. `msm_api` defines portable canonical/Montgomery scalar, raw point, window,
   routing, and ragged-job values.
2. `router` implements deterministic overlapping-bit Booth recoding and checked
   host routing for the variable-base API. Scalar-derived allocations erase on
   every exit path.
3. `engine` validates bases/jobs, owns device preparations, selects variable or
   fixed execution plans, and exposes host/resident APIs.
4. `engine::abi` freezes private parameter and record layouts shared with the
   production shaders.
5. `engine::sys` is the audited Objective-C, `MTLBuffer`, command-lifetime,
   shared-memory, status, and erasure boundary. Unsafe host code is confined
   there.

The dependency direction is always:

```text
application adapter -> bls12-381-metal-msm
```

Transcript, quotient, proving-key, proof-routing, and application activity
types are prohibited in this crate.

## Variable-base path

The portable router supports exact 8/9/10-bit overlapping Booth semantics,
single/prefix/ragged jobs, deterministic count-sort ordering, and compact signed
records. Metal performs point reduction, integration, collection, output
conversion, and erasure. Multiple reducer strategies remain available for
differential testing and general callers.

Variable routing is host-owned and data-dependent. Midnight's production KZG
path does not use it because its bases are fixed and reusable.

## Fixed-base path

`PreparedFixedBases` validates public affine bases once and expands them by
window powers for widths 11 through 16. Fixed-base execution moves scalar
canonicalization, recoding, routing, and all group arithmetic to Metal.

```text
host canonical | host Montgomery | resident Montgomery scalars
  -> checked ragged jobs
  -> optional resident Montgomery canonicalization
  -> histogram / exclusive offsets / compact scatter
  -> ordinary or tiled XYZZ bucket reduction
  -> blockwise weighted integration
  -> high-to-low window collection
  -> raw Montgomery Jacobian outputs
```

The v10 topology classifies runs longer than 512 records as long buckets and
describes them with compact 2,048-record tiles. Ordinary reduction uses eight
lanes per bucket. Long tiles reduce independently and combine into one dense
root. Integration splits each output into 64-bucket blocks, uses 16 lanes per
block and four blocks per group, then finishes block prefixes and collects the
window-shifted result.

Lane/tile accumulators use the first valid affine point as their initial XYZZ
state. All subsequent additions handle identity, equality/doubling, and
opposite-point cancellation. The direct initialization removes known-identity
work without weakening later exceptional-case behavior.

## Batching and resident sources

Prefix batches are convenience wrappers over general `MsmJobV1` arrays. Ragged
jobs may choose independent base/scalar ranges and dense output indices.

Resident multi-source execution accepts several retained scalar buffers and one
logical job array. It builds one command without host scalar concatenation. The
caller/worker may coalesce compatible requests that share preparation, basis,
window, and device identity. Replies are split back into their original request
boundaries only after the shared command completes.

The engine owns all derived scratch and output staging. It only borrows caller
resident sources and retains them through completion.

## Validation and failure

Prepared bases are reduced, finite, on-curve, and prime-subgroup checked with
BLST before upload. Engine/preparation/device identities, scalar residues, job
ranges, capacities, descriptor density, and shader launch parameters fail
closed. Startup probes bind generated BLS12-381 constants and every production
ABI layout.

The library contains no CPU group implementation. An error never triggers a
retry. Validation errors before submission leave the engine usable. A command,
kernel-status, completion, or erasure state that cannot be proven poisons the
engine and rejects later work.

## Secret lifecycle

Host-derived digit/routing arrays use zeroizing ownership. GPU canonical scalar,
histogram, offset/cursor, signed record, bucket/tile, integration, collection,
output, indirect, and status capacities are cleared after completion. Execution
is serialized inside one engine so scratch ownership and cleanup ordering are
auditable.

Caller-owned resident inputs are not erased by the MSM engine. Their producer
must clear them after all retained consumers finish.

## Observation profile

`TrustedLocalDataDependentV1` permits scalar-dependent bucket addresses,
occupancy, indirect work, and timing. It makes no hostile-co-tenant or remote
timing claim. A strict fixed-observation implementation requires a distinct
profile, kernels, tests, and performance contract; it cannot be substituted
under the existing name.

## Build separation and extraction

`metal` builds only the production shader root. `metal-diagnostics` adds
historical arithmetic/reducer experiments. Research failures are retained as
candidate evidence but do not control the executable API.

The package is extractable because it owns its manifest, generated constants,
shaders, tests, ABI, security policy, license, provenance, and release process.
Publication remains disabled until repository governance, security reporting,
and Apple-hardware CI are available.
