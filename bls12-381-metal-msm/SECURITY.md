# Security policy

## Executable scope

Version 0.1 exposes executable variable- and fixed-base Apple Metal MSM under
the explicit `TrustedLocalDataDependentV1` observation profile. It is not
constant-time or constant-observation: signed digits affect bucket occupancy,
scatter addresses, indirect work, additions, traffic, and duration.

Use it only when the host process, OS, driver, profiling facilities, and local
GPU execution environment are trusted. Remote timing and hostile local
co-tenants are outside the current claim.

## No fallback

The crate has no CPU group-arithmetic implementation. Initialization,
preparation, validation, routing, ABI, allocation, command, kernel-status,
completion, and erasure failures return an error. A consumer may explicitly
select another backend outside this library; this library never does so after
an MSM is selected.

## Input validation

- `CanonicalScalar` rejects limbs greater than or equal to the BLS12-381 Fr
  modulus.
- Resident Montgomery scalars are bounds/device checked on the host and reduced
  residue checked by Metal canonicalization.
- Prepared affine coordinates must be reduced Fp Montgomery residues, finite,
  on curve, and in prime-order G1. BLST 0.3.16 supplies independent host
  predicates.
- Preparations bind engine context, device, base count, fixed window, and
  expanded allocation identity.
- Job ranges, output density, indices, descriptors, capacities, launch geometry,
  reserved words, and ABI arithmetic are checked before output acceptance.
- Startup probes verify curve constants, public/private layouts, fixed-route
  v10 topology, and status protocol.

These checks do not authenticate a KZG trusted setup. Applications must bind
prepared bases to an authenticated setup and its semantic basis.

## Secret lifecycle

Scalar-derived host vectors use zeroizing ownership. Owned GPU scalar, route,
bucket, tile, integration, collection, output, indirect, and status capacities
are wiped after work completes. Full capacity is cleared, not only logical
length.

Execution is synchronous and serialized inside an engine. If command completion
or erasure cannot be proven, the engine is poisoned and rejects later work.
Rejected inputs that submit no command do not poison the engine.

`ResidentMontgomeryScalars` retains caller-owned input through command
completion but does not erase it. The producer owns source erasure after its
last consumer. Foreign-device, misaligned, out-of-range, and nonreduced resident
views fail closed.

Unsafe Objective-C and shared-memory operations are confined to `engine::sys`.
`MetalMsmEngine` is thread-affine (`Rc`, not `Send`/`Sync`) so applications do
not rely on undocumented Objective-C cross-thread ownership.

## Point arithmetic

The fixed production reducer maintains XYZZ state and handles identity,
doubling, opposite points, and cancellation. Direct first-point initialization
avoids addition with a known identity but does not bypass exceptional handling
for later inputs. Independent BLST differentials cover boundary and skewed
cases.

Branches and addresses remain scalar-dependent and are permitted only by the
trusted-local profile. They must not be described as oblivious or constant-time.

## Validation evidence

The M1 Max device suite currently passes 11/11 differential tests across every
supported window and fixed preparation width, prefix/ragged/multi-source
resident batches, scalar boundaries, exceptional point cases, long-bucket
tiling, malformed resident residues, and post-rejection engine reuse.

The Midnight production integration has completed 20 paired byte-exact
transactions (2,840 Metal outputs) and a 100-pair soak (14,200 Metal outputs)
with zero standalone MSM failure and zero fallback.

## Known exclusions

Version 0.1 does not claim:

- protection from hostile GPU co-tenants, driver/kernel compromise, or physical
  power/EM/fault observation;
- fixed-work or fixed-address execution;
- authenticated trusted-setup ingestion;
- asynchronous cancellation/process-kill erasure; or
- safety for writable foreign buffers or external command events.

Adding a strict observation profile, asynchronous API, writable resident view,
or external synchronization primitive requires a new reviewed contract and
negative tests.

## Reporting and release

Report suspected vulnerabilities privately to the future standalone
maintainers. Until a repository owner and security-reporting channel exist,
`publish = false` remains mandatory. A release must document supported macOS,
Apple GPU, Xcode/Metal compiler, Rust, BLST, ABI, and shader fingerprints and
must preserve fail-closed negative tests.
