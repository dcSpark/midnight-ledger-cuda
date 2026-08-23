# Midnight Metal prover

This crate owns Midnight's native Metal runtime and proof-kernel boundary. The
production `metal-hybrid` route uses it for validated batched NTTs, immutable
schema-v2 proving-key/SRS contexts, expression evaluation, quotient construction
and blinding, resident quotient commitment leases, and the resident multi-open
arena. It also retains deterministic proof-byte, Fr/Fp, G1, and performance
probes for differential testing and research.

The runtime loads specialized expression-evaluator pipelines and validates
their complete layout/tag ABI at startup. Its NTT path caches checked domain
plans, executes balanced mixed-radix Stockham passes through generation-checked
resident ping-pong buffers, normalizes inverse transforms once, and supports
resident order-three coset extension/interpolation. Stable raw Fr/Fp/G1 sizes,
alignments, member offsets, a pinned layout fingerprint, and build-generated
Rust/Metal constants are validated before proof work.

`metal_status()` reports runtime and kernel readiness; proof-server separately
owns selection of the honest CPU+Metal route. Requesting selected Metal work
returns an error when unavailable or invalid, and this crate never substitutes
CPU execution. The production route is documented as `metal-hybrid` because it
still materializes one combined quotient-opening polynomial per proof.

The resident secret lifecycle covers NTT, quotient, multi-open, and shared
scratch arenas. Successful, error, and panic paths clear full allocation
capacities before reuse. An unconfirmed clear quarantines the capacity and
permanently poisons the session. Dropping a buffer or overwriting only its
logical length is never treated as erasure.

The earlier CPU-BLST quotient lease remains only as explicit diagnostic test
support. Its representative five-proof shadow benchmark measured 1,184.911 ms
and was rejected. Production instead lends quotient pieces directly to the
standalone Metal MSM, then materializes only the combined opening needed by the
current host continuation. See the historical
`../reports/ledger-9-metal-resident-quotient-result-2026-07-15.md` and current
`../reports/ledger-9-metal-production-acceptance-2026-07-17.md`.

Native MSM is isolated in the sibling `bls12-381-metal-msm` package. This
runtime adapts checked resident scalar and SRS handles into that library, but
the standalone package does not depend back on this proof system.

The G1 correctness kernels use complete RCB Algorithms 7, 8, and 9 for full
addition, mixed addition, and doubling through a Metal-only homogeneous
representation. Their point-operation and identity-mask schedules do not branch
on exceptional cases. Fr/Fp addition, subtraction, and Montgomery final
reduction use fixed subtract-and-mask/select schedules instead of value-dependent
branches. These diagnostic probe kernels have not received an independent
side-channel audit and are not the production MSM path. The standalone MSM uses
its own reviewed trusted-local data-dependent profile. The public
BLST/CUDA-compatible Jacobian ABI is unchanged, and the internal `(0, R, 0)`
homogeneous identity is covered by a
dedicated fill diagnostic.

Apple-silicon macOS builds require Xcode's optional Metal toolchain. One root
translation unit includes the phase shaders, is compiled to AIR, and is linked
into the embedded `midnight-metal.metallib`. The compiler is explicitly pinned
to a macOS 13 deployment target, and the build rejects AIR that does not contain
the `apple-macosx13.0.0` target stamp. A macOS 13 Apple-silicon build-and-load
runner remains the compatibility CI gate when updating Xcode. Other targets
build the same public API with an unavailable `MetalSession`.

After selecting the intended Xcode, install a missing compiler with:

```bash
xcodebuild -downloadComponent metalToolchain
```

The ignored release diagnostic reports per-operation Fr64 and Fp64 throughput:

```bash
cargo test -p midnight-metal-prover --release report_field64_throughput -- --ignored --nocapture
```

Set `MIDNIGHT_METAL_BENCH_ELEMENTS` or `MIDNIGHT_METAL_BENCH_ROUNDS` to override
the default batch and repetition counts.

The isolated Fp32 experiment keeps the public six-`u64` ABI but splits values
inside diagnostic kernels into twelve `u32` limbs of the same R=2^384
Montgomery integer. Its raw multiplication benchmark excludes split/rejoin and
readback from the timed command:

```bash
cargo test -p midnight-metal-prover --release report_fp32_raw_mul_throughput -- --ignored --nocapture
```

Set `MIDNIGHT_METAL_FP32_ELEMENTS`, `MIDNIGHT_METAL_FP32_ITERATIONS`, or
`MIDNIGHT_METAL_FP32_DISPATCHES` to override its defaults. The reviewed Apple
M1 Max run reached 1,185,515,361 multiplications/s, below the required 2.0B
optimistic gate, so this representation is not extended into G1, NTT, or proof
execution.

The sibling wide-accumulator experiment uses the same twelve limbs and fixed
reduction, but evaluates each 32x32 product plus limb and carry in one `ulong`:

```bash
cargo test -p midnight-metal-prover --release report_fp32_wide_raw_mul_throughput -- --ignored --nocapture
```

Its reviewed M1 Max run reached 1,540,404,194 multiplications/s at threadgroup
width 64. That is 77.02% of the same 2.0B raw gate and remains `NO_GO`; no G1,
NTT, or proof route uses the wide path.

Explicit two- and three-state variants alternate every CIOS limb step to test
integer instruction-level parallelism. Their reviewed M1 Max totals regress to
963,456,555 and 467,233,045 multiplications/s respectively, so the
single-thread Fp32 branch is closed:

```bash
cargo test -p midnight-metal-prover --release report_fp32_wide_interleaved_2_raw_mul_throughput -- --ignored --nocapture
cargo test -p midnight-metal-prover --release report_fp32_wide_interleaved_3_raw_mul_throughput -- --ignored --nocapture
```

The corresponding ignored end-to-end G1 diagnostic reports normalization,
doubling, and mixed-add result triplets:

```bash
cargo test -p midnight-metal-prover --release report_g1_probe_throughput -- --ignored --nocapture
```

Set `MIDNIGHT_METAL_G1_BENCH_ELEMENTS` or `MIDNIGHT_METAL_G1_BENCH_ROUNDS` to
override its defaults.

The MSM-feasibility diagnostic converts inputs before timing, runs in-register
Algorithm 7 chains entirely in internal homogeneous storage, and converts and
reads back only after the timed command. Each complete addition is exactly
`12M + 27A`:

```bash
cargo test -p midnight-metal-prover --release report_g1_complete_add_chain_throughput -- --ignored --nocapture
```

Set `MIDNIGHT_METAL_G1_CHAIN_ELEMENTS`,
`MIDNIGHT_METAL_G1_CHAIN_ADDITIONS`, or
`MIDNIGHT_METAL_G1_CHAIN_DISPATCHES` to override its 8,192-element, 64-addition,
four-dispatch defaults.

The host-side, test-only batch-affine diagnostics replay a content-minimized,
witness-derived run topology and require explicit acknowledgement. Their MSL
symbols remain embedded in the normal metallib, but no non-test Rust path loads
or dispatches them; they are not proof routes. On the reviewed M1 Max, the
global-tree reducer reached 30.633M useful reductions/s; the final recursive
threadgroup-local reducer reached only 3.180M/s, versus the 153.5M/s promotion
gate. Both are rejected. A separate topology-independent, register-local ceiling probe
then executed the exact `7 M/S + 9 A/S` population at 164.206M/s. Its
280.374 ms arithmetic-only projection left 19.626 ms for all real dataflow and
failed the predeclared 200M/s implementation gate. Reproduce that decision with:

```bash
MIDNIGHT_METAL_AFFINE_CEILING_THREADGROUP_WIDTH=64 \
MIDNIGHT_METAL_AFFINE_CEILING_SAMPLES=20 \
CARGO_INCREMENTAL=0 cargo test --release -p midnight-metal-prover --lib \
  metal::batch_affine_ceiling::tests::report_batch_affine_arithmetic_ceiling \
  -- --ignored --exact --nocapture --test-threads=1
```

The sensitive flat-segmented result can be reproduced only after regenerating
the mode-`0600` manifest outside the checkout:

```bash
MIDNIGHT_METAL_SENSITIVE_DIAGNOSTIC=1 CARGO_INCREMENTAL=0 cargo test --release \
  -p midnight-metal-prover --lib \
  metal::flat_segmented_authentic::authentic_four_chunk_flat_segmented_decision_gate \
  -- --ignored --exact --nocapture --test-threads=1
```

See [`../docs/metal-msm-security.md`](../docs/metal-msm-security.md) and the
owner-only local report
[`../reports/ledger-9-metal-flat-segmented-result-2026-07-15.md`](../reports/ledger-9-metal-flat-segmented-result-2026-07-15.md)
for the handling contract, exact sensitive counters, and decision. The ceiling
result, which references the same aggregate authentic-workload total, is in
[`../reports/ledger-9-metal-arithmetic-ceiling-result-2026-07-15.md`](../reports/ledger-9-metal-arithmetic-ceiling-result-2026-07-15.md).
Do not stage or publish the flat-segmented report without security-owner
declassification.

The ignored resident NTT phase diagnostic warms the cached plan, then encodes
all forward/inverse pairs into one command with one final wait and readback:

```bash
cargo test -p midnight-metal-prover --release report_resident_ntt_throughput -- --ignored --nocapture
```

Set `MIDNIGHT_METAL_NTT_BENCH_LOG_SIZE`,
`MIDNIGHT_METAL_NTT_BENCH_BATCHES`, or `MIDNIGHT_METAL_NTT_BENCH_ROUNDS` to
override its `k=17`, four-batch, sixteen-roundtrip defaults.

The host runtime and build organization are informed by the Apache-2.0
[`neo-prover-metal`](https://github.com/LFDT-Nightstream/Nightstream/tree/29ef5a53a665f7b0347cee790bebbfcdcc0c4d69/crates/neo-prover-metal)
implementation. Midnight's BLS12-381 kernels and protocol integration are
independent because Nightstream uses a different field and proof system.
The exact reviewed inputs and working-tree caveat are recorded in
[`PROVENANCE.md`](PROVENANCE.md).
