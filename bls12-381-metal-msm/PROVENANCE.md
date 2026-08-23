# Provenance

The standalone package structure, public API, CPU router, private shader ABI,
runtime, build split, and evidence records were developed for Midnight in July
2026 under Apache-2.0.

## Executable MSM sources

The 6x64 CIOS Fp implementation and complete homogeneous G1 implementation were
extracted from the package-local parent project `midnight-metal-prover`, then
renamed, isolated from application types, and integrated behind a new versioned
standalone ABI. The field path had already been differentially tested against
[BLST 0.3.16](https://github.com/supranational/blst/tree/v0.3.16). The point
schedules implement Renes, Costello, and Batina,
[Complete addition formulas for prime order elliptic curves](https://eprint.iacr.org/2015/1060),
Algorithms 7--9, following BLST's operation order.

The variable-base CPU router independently implements the exact overlapping-bit
signed-window semantics used by `midnight-curves` for compatibility. The
standalone code adds canonical-Fr construction, frozen 8/9/10-bit variable and
11--16-bit fixed window shapes, deterministic batched/ragged count-sort routing,
compact signed records, checked overflow handling, and explicit erasure. It
imports no curve arithmetic or CPU fallback.

The Metal pipeline, public/private ABI translation, status protocol, production
shader root, engine/prepared-base API, fixed-base histogram/scan/scatter,
cooperative and tiled XYZZ reducers, blockwise integration, window collection,
resident scalar views, and multi-source batching were written for this
standalone package. The Objective-C runtime patterns were adapted from reviewed
retained-command and shared-buffer patterns in `midnight-metal-prover`; parent
application types are not imported.

BLST 0.3.16 is an optional Apple-platform dependency for prepared-point
validation and a development oracle. The accepted M1 Max differential run
executed all 11 device tests successfully. That is correctness evidence, not a
standalone throughput claim.

## External projects

No source was copied from CUDA, sppark, zkMetal, or another external MSM
library, and none is a dependency. CUDA/sppark were used only to understand MSM
semantics and workload expectations. zkMetal was reviewed as research evidence
at commit
[`443127161417b415f70d5de74a186a1820e557e5`](https://github.com/carni-ships/zkMetal/commit/443127161417b415f70d5de74a186a1820e557e5);
it contributed no code, constants, or benchmark claims.

The separate parent review at
`reports/ledger-9-zkmetal-bls12-381-evaluation-2026-07-15.md` has SHA-256
`2189187c63d2f2debcefcf806d8c8bbb9191e3337ae89bc792ab2216746387ce`.

## Historical candidate evidence

Earlier arithmetic work is retained because a failed experiment rejects only
its concrete representation and schedule:

| Candidate | M1 Max result | Evidence |
|---|---|---|
| precise FP32 issue schedule | 1.002B median / 0.941B p5 equivalents/s | `evidence/m1-max-precise-fp32-product-issue-2026-07-15.json` |
| two-lane u32 FIPS | 0.987B / 0.940B complete Fp/s | `evidence/m1-max-two-lane-u32-fips-field-2026-07-15.json` |
| named-register FIOS | 1.529B / 1.455B complete Fp/s | `evidence/m1-max-named-register-fios-field-2026-07-15.json` |
| native-u32 30-limb issue probe | 1.901T / 1.823T raw products/s | `evidence/m1-max-u32-low-limb-product-gate-2026-07-15.json` |
| subtractive Karatsuba/L4 v2 | 70 live words versus 56 gate; no timing | `evidence/m1-max-karatsuba-l4-liveness-gate-2026-07-15.json` |
| blocked-radix-128 v3 | 0.465B / 0.455B complete REDCs/s | `evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json` |
| AMNS8 direct K8 | 6.798M / 6.722M field operations/s | `evidence/m1-max-amns8-k8-field-2026-07-16.json` |
| explicit `mulhi` substitution | no repeatable primitive benefit | `evidence/m1-max-metal-mulhi-primitive-2026-07-16.json` |

These figures retain their original gate definitions and limitations in the
named candidate reports. They are not device-wide ceilings and do not veto the
executable MSM or materially different algorithms. The standalone baseline uses
the validated CIOS/RCB path while parallel alternatives are developed.

The package also retains independently generated PMNS (`X^25 - 2`) and AMNS
(`X^8 + 2`) models. Their lattice derivations use the cited public mathematics;
the associated public source repository supplied no code or constants.

## Build and dependency record

`build.rs` generates BLS12-381 constants, ABI fingerprints, and research
schedules. With `metal`, it compiles only `shaders/msm_root.metal`; with
`metal-diagnostics`, it compiles the separate `shaders/root.metal`. Generated
AIR and metallib artifacts live in `OUT_DIR` and are not vendored.

Apple execution uses pinned `dispatch2`, `objc2`, `objc2-foundation`, and
`objc2-metal` dependencies. `zeroize` supplies no-std host-vector erasure.
`num-bigint` is a test-only exact-integer oracle. The portable default build has
no Metal object and compiles no shader artifact.

Historical parent inputs include
`reports/ledger-9-metal-msm-feasibility-2026-07-15.md` (SHA-256
`4606d06d9c76ff68648ddc2a5a9d515a9ae9645de97f14ba02a15f117090e784`) and
`reports/ledger-9-metal-primitives-2026-07-15.md` (SHA-256
`c33b378d29aae26782f3d9a43ce81039fe850a511506e18f8a2c29a23d8de303`).

The package now has checked resident-buffer interop and is integrated into the
authentic proof path. Per transaction, it supplies all 142 nonempty MSM outputs;
the remaining five logical calls are all-zero identity shortcuts with zero
nonzero CPU scalars. The accepted 20-pair application run produced 2,840 outputs
in 767 submissions with no library failure, and the 100-pair soak produced
14,200 outputs with the same result. The parent report
`reports/ledger-9-metal-production-acceptance-2026-07-17.md` owns the end-to-end
3x claim because proof latency is not a standalone MSM benchmark.
