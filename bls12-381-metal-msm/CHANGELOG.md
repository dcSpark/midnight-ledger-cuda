# Changelog

## 0.1.0 - unreleased

### Production MSM

- Added public canonical Fr, raw Montgomery Fp/G1, fixed-window, compact routed
  point, bucket-run, and versioned ragged-job types with frozen layouts.
- Added exact overlapping-bit signed recoding for variable 8/9/10-bit and fixed
  11--16-bit windows, checked capacities and ranges, and scalar-derived host-data
  erasure.
- Added `MetalMsmEngine` plus reusable, validated, device-bound `PreparedBases`
  and `PreparedFixedBases`.
- Added single, prefix-batched, ragged, host-Montgomery, resident-Montgomery, and
  multi-source resident APIs under the explicitly selected
  `TrustedLocalDataDependentV1` profile.
- Added a fully Metal-routed fixed-base v10 path: scalar conversion, histogram,
  scan, scatter, cooperative XYZZ bucket reduction, compact long-bucket tiling,
  blockwise weighted integration, window collection, and raw Jacobian output.
  CPU code performs no group arithmetic and the engine has no hidden fallback.
- Added the production worker adapter used by Midnight. It coalesces resident
  jobs, retains prepared KZG bases, validates output policy, and exposes failure
  counters without importing proof-system types into this crate.
- Added private shader-ABI translation, status validation, startup ABI/constants/
  layout probes, serialized scratch ownership, erasure verification, and context
  poisoning on unconfirmed cleanup.
- Kept the engine intentionally thread-affine with `Rc`; version 1 does not claim
  `Send` or `Sync` for Metal/Objective-C ownership.
- Split the production `msm_root.metal` metallib from the historical diagnostic
  `root.metal` build.
- Passed all 11 ignored Metal/BLST differential tests on the 32-core M1 Max,
  covering every supported window, scalar boundaries, complete cases, host and
  resident inputs, multi-source batches, long buckets, malformed input recovery,
  prefix batches, and ragged batches.
- Produced all 142 nonempty MSM outputs per authentic transaction. The accepted
  20-pair proof run produced 2,840 outputs in 767 coalesced submissions with no
  library failure; the 100-pair soak produced 14,200 outputs with the same
  result.

### Fixes

- The fixed-base v10 integration and bucket-reduction kernels no longer do
  group arithmetic inside a lane-dependent branch between SIMD shuffles. Every
  lane computes the combination and only the owning lanes keep it. Before,
  batched fixed-base MSMs returned wrong points for some outputs on some runs
  (observed on an M5 Pro, macOS 27.0.1): the integration kernel at widths 12
  to 16, and the bucket reduction for a SIMD group of buckets on about one
  output in 1,000. Proofs made with Metal then failed to verify. Six-output
  batches over 2^16 bases take about 5% longer at width 13 and about 20%
  longer at width 16. Covered by
  `fixed_base_batches_match_blst_on_repeated_runs_at_every_width`,
  `fixed_base_batch_soak_matches_blst` and the
  `metal-fixed-base-msm-regression` tests.

### Current limits and release boundary

- Version 0.1 is intentionally thread-affine and exposes only the trusted-local,
  data-dependent execution profile; it is not constant-observation.
- End-to-end proof speed belongs to the parent application acceptance report,
  not the standalone crate's API contract.
- Publication remains disabled until maintainers, an independent repository,
  private vulnerability reporting, Apple-GPU CI, and release review exist.

### Historical research

- Created the independently extractable package, governance boundary, generated
  constants, diagnostic metallib, and reproducible evidence format.
- Rejected the tested packed-FP32 and carry-normalized precise-product schedules;
  the latter measured 1.002B median / 0.941B p5 equivalents/s.
- Implemented and measured the two-lane u32 FIPS field path at 0.987B / 0.940B
  complete Fp/s and named-register FIOS at 1.529B / 1.455B.
- Measured the 30-limb native-u32 issue probe at 1.901T / 1.823T raw products/s.
- Recorded the subtractive Karatsuba/L4 v2 liveness failure at 70 words versus
  56, then implemented the blocked-radix-128 v3 model and measured 0.465B /
  0.455B complete REDCs/s.
- Added independently generated PMNS and AMNS models. The direct AMNS8 K8 path
  measured 6.798M / 6.722M field operations/s; explicit low-plus-`mulhi` showed
  no repeatable primitive benefit.
- Retained every failure as candidate-specific diagnostic evidence. None is a
  universal Metal, field, point, or MSM stop condition.
