# bls12-381-metal-msm

`bls12-381-metal-msm` is a standalone BLS12-381 G1 multi-scalar
multiplication library for Apple Metal. It is application-neutral and intended
to be extractable into an independent open-source repository.

The crate owns complete executable MSM paths, not proxy benchmarks:

- validated reusable variable and fixed prepared bases;
- canonical host, Montgomery host, and checked resident Montgomery scalars;
- variable 8/9/10-bit and fixed 11–16-bit windows;
- single, prefix-batched, ragged, and multi-source resident jobs;
- Metal recoding/routing for the fixed-base production path;
- compact bucket reduction, long-bucket tiling, weighted integration, and
  window collection;
- raw Montgomery Jacobian output;
- versioned ABI probes, status validation, secure scratch erasure, and poisoned
  context handling; and
- no hidden CPU curve fallback.

Midnight uses the fixed-base path because the KZG coefficient and Lagrange SRS
bases are immutable and reused. CUDA, sppark, and zkMetal are not dependencies
and no source was copied from them.

## Platform and build

The portable default build exposes value types and routing models without Metal
objects. Apple execution requires Apple-silicon macOS, the Xcode Metal compiler,
and Rust 1.95:

```bash
cargo test -p bls12-381-metal-msm
cargo build --release -p bls12-381-metal-msm --features metal
```

`metal` compiles the production shader root. `metal-diagnostics` additionally
builds retained research kernels and is not the production proof artifact.

## API

The principal types are:

- `MetalMsmEngine`: one thread-affine device/queue/pipeline owner;
- `PreparedBases`: validated variable-base allocation;
- `PreparedFixedBases`: validated bases expanded by public window powers;
- `CanonicalScalar` and `RawFrMontgomery`: checked scalar representations;
- `ResidentMontgomeryScalars`: retained caller-owned Metal scalar view; and
- `MsmJobV1`: versioned ragged base/scalar/output ranges.

Execution methods include `msm`, `msm_batch`, `msm_ragged`, `msm_fixed`,
`msm_fixed_batch`, `msm_fixed_ragged`, host-Montgomery variants, resident
variants, and `msm_fixed_resident_sources_ragged` for coalescing independent
resident buffers without concatenating them on the CPU.

An engine and its preparations are bound to one engine context and one Metal
device. The engine deliberately uses `Rc` and is neither `Send` nor `Sync`; an
application should keep it on one worker thread. Unsupported inputs, foreign
preparations, invalid resident ranges, ABI mismatches, command failures, kernel
status failures, or unconfirmed erasure return an error.

## Fixed-base production pipeline

The fixed-base v10 path is fully Metal-routed:

```text
checked scalar source + prepared window-shifted bases
  -> optional Montgomery-to-canonical conversion
  -> signed-window histogram / scan / scatter
  -> cooperative XYZZ bucket reduction
  -> 512-record long-bucket classification
  -> compact 2,048-record tile reduction and root combination
  -> blockwise weighted integration
  -> window collection and raw Jacobian output
  -> status validation and full-capacity scratch erasure
```

Ordinary reduction assigns eight lanes to each of eight buckets in a 64-thread
group. Integration uses 16 lanes per 64-bucket block and four blocks per group.
Lane and tile accumulators initialize from their first valid affine point,
avoiding an addition with identity while preserving exceptional-case handling
for all later additions.

The variable-base API retains deterministic host routing and several Metal
reducer strategies for general callers and differential research. It is not on
Midnight's accepted KZG path.

## Correctness and reliability

The production differential suite compares every output with independent BLST
sums. On the M1 Max, all 11 device tests pass, covering:

- scalar boundaries and all supported windows;
- empty, identity, repeated, opposite, and cancellation cases;
- prefix and general ragged batches;
- every fixed preparation width;
- host and resident Montgomery inputs;
- multi-source resident batching;
- long skewed buckets and adaptive/tiled reducers; and
- malformed resident residue rejection followed by successful engine reuse.

```bash
cargo test --release -p bls12-381-metal-msm

cargo test --release -p bls12-381-metal-msm \
  --features metal-diagnostics \
  --test metal_msm_differential \
  -- --ignored --test-threads=1
```

In Midnight's accepted transaction workload, the library produces 142 nonempty
MSM outputs per transaction. The production 20-pair run produced 2,840 outputs
in 767 coalesced submissions with zero library failure; the 100-pair soak
produced 14,200 outputs with the same result. End-to-end transaction latency is
documented in the parent acceptance report rather than claimed as a standalone
microbenchmark.

## Security profile

Version 0.1 exposes only `TrustedLocalDataDependentV1`. Digits affect bucket
occupancy, addresses, indirect work, and timing. Use it only when the local GPU
execution environment is trusted. The profile is not constant-time or
constant-observation.

Scalar-derived host storage and owned GPU scratch are erased. Command or
erasure states that cannot be confirmed poison the engine. The library never
selects a CPU MSM behind the caller's back. See [SECURITY.md](SECURITY.md).

## Standalone boundary

The package has its own manifest, shader roots, generated constants, ABI,
license, tests, security policy, provenance, and release checklist. It has no
dependency on Midnight ledger types, PLONK, transcript code, CUDA, sppark, or
zkMetal. Application adapters may depend on this crate; this crate must not
import application scheduling or proof-system types.

Publication remains disabled until the standalone repository, maintainers,
security-reporting channel, macOS CI, and release ownership are established.
Licensed under Apache-2.0; see [LICENSE-APACHE](LICENSE-APACHE).
