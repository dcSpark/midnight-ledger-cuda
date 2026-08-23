# Metal prover backend

The Apple-silicon backend is an executable CPU+Metal proof route named
`metal-hybrid`. It runs the complete production proof and is not a CUDA+Metal
hybrid. CUDA/Rust Oxide remains a separately built Linux backend and a semantic
reference only.

The current production result on the 32-GPU-core M1 Max is a 1,404.859 ms
median and 1,426.223 ms p95 for the five-proof balanced zswap transaction. The
paired CPU median and p95 from the same binary are 4,553.227 ms and 4,632.661 ms,
giving 3.241x median and 3.248x p95 speedups. Twenty paired transactions were
byte-exact with zero fallback and zero Metal failure. See the
[production acceptance report](../reports/ledger-9-metal-production-acceptance-2026-07-17.md).

## Build and selection

Rust 1.95 is the checked toolchain. Build the production backend with:

```bash
cargo build --release -p midnight-proof-server --features metal
```

The parity/acceptance executable additionally needs the deterministic fixture
feature:

```bash
cargo build --release \
  -p midnight-proof-server \
  --bin proof-backend-parity \
  --features deterministic-test-rng,metal
```

Selecting `metal-hybrid` enables the native MSM, resident quotient prefix, and
resident multi-open path. No acceleration environment variable is required.
If Metal initialization, ABI validation, command completion, kernel validation,
or secret erasure fails after selection, the proof fails. The route never
retries that operation on CPU.

The pure `metal` selector remains fail-closed. It is reserved for a future route
that removes the remaining whole-polynomial host seam; using `metal-hybrid`
keeps the current ownership claim honest.

## Architecture

```text
proof-server route guard
        |
        v
midnight-proofs / midnight-curves accelerator seams
        |
        v
midnight-gpu-prover                 backend-neutral types and checked handles
        |
        +-- midnight-metal-prover   NTT, field, quotient, multi-open, lifecycle
        |
        +-- bls12-381-metal-msm     standalone fixed/variable-base G1 MSM
```

`bls12-381-metal-msm` does not depend on PLONK, transcript, ledger, CUDA,
sppark, or the Metal prover runtime. The application adapter depends on the
standalone library, never the reverse. This boundary is intended to permit an
independent open-source repository later.

One process-level Metal session owns the device, command queues, immutable key
contexts, resident field arena, and secret-buffer pool. A dedicated thread owns
the thread-affine standalone MSM engine. Bounded request queues coalesce
compatible fixed-base jobs from independent proofs. Four proof jobs may prepare
CPU work concurrently while Metal command ownership remains serialized and
auditable.

## Current ownership

| Prover work | Current owner | Residency notes |
| --- | --- | --- |
| Decode, request construction | CPU | Compact/public inputs |
| Witness synthesis | CPU | Independent proof jobs overlap |
| Advice/instance transforms | Metal NTT | Checked domain-tagged resident arena |
| Advice, permutation, lookup, trash commitments | Metal MSM | Fixed prepared coefficient/Lagrange bases |
| Numerator and quotient pieces | Metal | Immutable key context plus uploaded trace columns |
| Quotient blinding and piece commitments | Metal | Resident Montgomery scalars go directly to MSM |
| Combined quotient opening | Metal then CPU | One full polynomial is currently materialized per proof |
| Multi-open q inputs | CPU | Five compact q polynomials are uploaded; uploading all 73 sources was slower |
| Multi-open q/f/pi arithmetic | Metal | Arena remains resident until transcript callback returns |
| f and pi commitments | Metal MSM | No scalar readback |
| Transcript/challenges/self-verification | CPU | Only compact values and G1 outputs are required |

The authentic transaction requests 147 commitment outputs. In the accepted
run, 142 nonempty outputs execute on Metal and five all-zero outputs return the
constant identity. The all-zero path records no nonzero CPU scalar and is not a
BLST fallback.

## Standalone MSM path

The performance-critical fixed-base path uses prepared window-shifted bases at
widths 11 through 16. It accepts canonical host scalars, Montgomery host
scalars, or checked caller-owned resident Montgomery buffers. Batched, ragged,
and multi-source resident requests use one versioned job ABI.

The v10 pipeline is:

1. validate the engine/device/preparation identities and every job range;
2. canonicalize Montgomery scalars on Metal when needed;
3. recode and histogram signed fixed-window digits on Metal;
4. scan dense bucket offsets and scatter compact signed point records;
5. reduce ordinary buckets cooperatively in XYZZ coordinates;
6. split buckets longer than 512 records into 2,048-record tiles and combine
   their roots;
7. integrate 64-bucket blocks using 16 lanes per block and four blocks per
   group; and
8. finish integration, collect outputs, validate status, and erase scratch.

The first valid affine point initializes each lane directly, avoiding a
complete addition with identity. Long-bucket root combination likewise starts
from the first tile. These are representation-preserving initializations, not
incomplete exceptional-case formulas; differential tests cover identity,
repeated/opposite points, cancellation, and long skewed buckets.

The variable-base API remains available with 8/9/10-bit routing and several
diagnostic reducer strategies. Midnight uses the fixed-base v10 path because
the KZG SRS is immutable and reused.

## Resident quotient and multi-open

The key-context schema binds domain roots, expression graphs, every immutable
field column, permutation data, and the coefficient/Lagrange SRS fingerprints.
Invalid dimensions, unreduced values, foreign sessions, stale generations, or
SRS mismatches fail before execution.

The quotient epoch evaluates the classified expression graphs, constructs the
numerator, divides/interpolates, splits and blinds quotient limbs, and lends a
callback-scoped resident MSM view. Quotient commitments therefore never expose
their scalar pieces to CPU MSM. The existing finalizer still needs the combined
opening polynomial, so the epoch combines the pieces on Metal and reads one
domain-sized polynomial to the host. That transfer is explicit in counters and
is the primary blocker for promoting the route to pure `metal`.

The multi-open arena currently receives host-precombined q polynomials. It
performs division, f construction, compact q evaluations, pi construction, and
both commitments on Metal. Only transcript evaluations and G1 commitments
leave the arena. A measured attempt to upload all original source polynomials
and form q on Metal increased writes by about 198 MiB per transaction and
regressed latency; the next design must reference already-resident sources.

## Failure and secret lifecycle

Every secret-capacity allocation has one owner and one cleanup path. Successful,
error, and unwind paths issue full-capacity clear commands before buffers can
return to the pool. An unconfirmed clear quarantines the capacity and poisons
the session. A poisoned session rejects future work.

The standalone MSM similarly erases scalar-derived host storage, GPU route
records, bucket state, output staging, and status storage. A rejected input that
never submitted work does not poison the engine; a completion or erasure state
that cannot be proven does.

Activity counters expose command buffers, dispatches, logical host mappings,
materializations, resident/CPU MSM leases, clear failures, quarantined bytes,
poison events, worker submissions, outputs, and failures. The acceptance
harness treats nonzero fallback/failure/lifecycle counters as an error.

## Observation profile

The production route explicitly uses the trusted-local data-dependent MSM
profile. Digit values affect bucket occupancy, scatter addresses, indirect
work, and timing. Erasure and fail-closed behavior do not make that trace
oblivious. Do not deploy it against a hostile local GPU co-tenant without a
separate threat-model approval or a fixed-observation implementation.

## Diagnostic overrides

These controls are for controlled comparisons; production uses their defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIDNIGHT_METAL_MSM` | `on` when Metal is selected | Explicit `off`/`cpu` comparison override |
| `MIDNIGHT_METAL_MSM_CPU_MAX_NONZERO` | `0` | Optional bounded CPU partition experiment |
| `MIDNIGHT_METAL_MSM_COALESCE_US` | `0` | Worker queue wait before compatible batching |
| `MIDNIGHT_METAL_MSM_WINDOW` | adaptive fixed 11/13 policy | Force a supported window for experiments |
| `MIDNIGHT_METAL_QUOTIENT_SUBSTITUTE` | `on` when Metal is selected | Compare canonical CPU quotient with `off` |
| `MIDNIGHT_METAL_RESIDENT_MULTIOPEN` | `on` | Disable only for diagnostic comparison |
| `MIDNIGHT_METAL_PARALLEL_PROOFS` | `4` in the parity harness | Bound independent proof preparation |

Overrides that move work to CPU must not be used for a production Metal
performance or residency claim.

## Verification

Core commands used for acceptance are:

```bash
cargo test --release -p bls12-381-metal-msm

cargo test --release -p bls12-381-metal-msm \
  --features metal-diagnostics \
  --test metal_msm_differential \
  -- --ignored --test-threads=1

cargo test --release -p midnight-metal-prover -- --test-threads=1

cargo test --release -p midnight-metal-prover \
  metal::msm_worker::tests::worker_api_matches_blst_for_single_and_ragged_batch_requests \
  -- --ignored --exact --test-threads=1
```

Metal device tests run sequentially to avoid macOS interactive-GPU watchdog
errors from unrelated long kernels executing concurrently. The CI workflow
already uses `--test-threads=1` for this package.

## Performance ceiling and next work

The historical warm Amdahl model leaves roughly 512 ms of CPU synthesis,
request, transcript, and orchestration work and therefore gives an
infinite-device structural ceiling near 10.2x. It is not a hardware forecast.
The measured route is 3.241x paired median and has captured roughly four-fifths
of the removable time relative to the frozen CPU reference.

The next meaningful work, in order, is:

1. transfer ownership of the resident combined quotient into evaluation and
   multi-open instead of materializing it;
2. let multi-open reference resident trace/key polynomials rather than uploading
   all source columns;
3. continue end-to-end-driven fixed-base MSM optimization; and
4. move additional witness-derived polynomial work only when its measured union
   interval justifies the residency complexity.

Historical candidate failures remain in the dated reports and standalone MSM
research documents. They reject those schedules only and no longer describe
the current backend status.
