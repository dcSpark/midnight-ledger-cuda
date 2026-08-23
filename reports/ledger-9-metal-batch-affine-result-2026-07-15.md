# Ledger 9 Metal batch-affine MSM decision result — 2026-07-15

> Handling: internal sensitive diagnostic. This report contains aggregate
> witness-derived workload counts needed to interpret the measurement. It is
> not approved for public release without security-owner review.

## Decision

**Reject the global batch-affine bucket reducer on the Apple M1 Max.** It does
not enter the review-only band and must not advance to digit routing, weighted
bucket integration, proving-key integration, or proof routing.

The correctness-first Metal implementation reduced 6,794,123 representative
authentic-shape pairs in a median 221.794 ms, or 30.633M pairs/s. Scaling by the
observed seed-24302 total of 46,039,066 reductions gives a 1,502.943 ms bucket
phase before record routing, integration, erasure, or failure handling. The
frozen immediate-stop gate was 126.3M pairs/s and 364.6 ms. The measured path is
4.12x too slow for that stop gate and 5.01x below the 153.5M/s promotion gate.

This closes the bounded branch described in
`ledger-9-metal-batch-affine-plan-2026-07-15.md`. The scalar CIOS/projective
branches were already rejected; no measured Metal MSM design currently meets
the M1 Max budget. Metal proof routing remains unavailable.

The separately bounded flat-segmented follow-up is also complete. Its faster
G128 variant reached only 3.180M useful reductions/s, with a 2,136.823 ms
representative median and a 14,479.771 ms full projection. The G256 variant was
slower. See `ledger-9-metal-flat-segmented-result-2026-07-15.md`. That result
closes the final layout/inversion falsifier; it does not reopen this rejected
global-tree branch.

The last unmeasured hierarchical-scan question is now closed under a separate
predeclared decision contract. A register-local kernel executing the exact
`7 M/S + 9 A/S` per-reduction population reached 164.206M/s and projects
280.374 ms for arithmetic alone. That is above the bare 153.5M/s promotion rate
but below the 200M/s required before implementing real prefix/suffix dataflow.
It leaves only 19.626 ms for every omitted dependency, inversion, barrier,
gather, route, integration, and write. See
`ledger-9-metal-arithmetic-ceiling-result-2026-07-15.md`.

## Measured result

Hardware and software:

- Apple M1 Max, 32-core GPU, 64 GiB unified memory;
- Rust 1.95.0 and Apple Metal 32023.883;
- optimized Cargo `release` test profile; and
- one long-lived `MetalSession`, embedded metallib, and retained-reference
  command buffers.

The timed workload used four deterministic load strata, covering 15 of the 62
seed-24302 32K calls. The source population contains both 32,767 and 32,768
logical lengths. The ephemeral manifest and its call fingerprints/run
histograms remain sensitive witness-derived diagnostic data under
`docs/metal-msm-security.md`; they are mode `0600`, outside the checkout, and
are not declassified by aggregation or hashing. SRS points, point indices, and
signs used by the replay are synthetic.

| Metric | Result |
| --- | ---: |
| Warmups / timed samples | 2 / 20 |
| Sample reductions | 6,794,123 |
| Median / p95 | **221.794 / 223.972 ms** |
| Device interval median | 221.154 ms |
| Command encoding median | 0.264 ms |
| Required status read median | 0.000604 ms |
| Reduction rate | **30,632,600/s** |
| Full 46,039,066-pair projection | **1,502.943 ms** |
| Commands / waits per sample | 1 / 1 |
| Dispatches per sample | 1,569 |
| Carry copies per sample | 1,539,333 |
| Nominal device-buffer traffic | 8,946,476,208 B |
| Timed Metal allocations / uploads | 0 / 0 B |
| Timed host read | 16 B status |
| Prepared Metal allocations | 19 |
| Prepared allocation bytes | 548,303,692 B |
| Immutable SRS/record/map upload | 64,242,952 B |
| Shared scratch / resident roots | 469,974,948 / 14,085,792 B |
| Driver-recommended working set | 55,662,788,608 B |

Raw total wall samples in milliseconds:

```text
222.137417 221.246792 221.690792 221.630083 221.246292
221.777667 222.670166 224.626584 223.971709 223.929458
222.133042 222.190667 221.023583 221.517833 222.437083
221.810083 221.254125 222.157334 221.234959 221.227334
```

The median is the mean of sorted positions 10 and 11. The p95 is the nearest
rank, sorted position 19. The full projection uses the measured pair rate, not
an isolated field-operation estimate:

```text
rate = 6,794,123 / 0.221793875 = 30,632,600 pairs/s
projection = 46,039,066 / rate = 1,502.943 ms
```

Pair scaling is favorable to the implementation. The four selected chunks have
53 active balanced levels, whereas the full 8K/16K/32K chunk plan has 391; the
level-count scale is larger than the 6.776x pair scale. Fixed root inversions
and dispatch overhead therefore make the true all-chunk replay no better than
the reported projection.

This is not a final full-authentic MSM median/p95 and does not prove that every
possible Metal MSM is impossible. It rejects this literal global product-tree
algorithm. The selection is slightly optimistic: selected record/pair ratio is
1.0216 versus 1.0229 for the full workload, selected carry/pair ratio is 22.66%
versus 24.98% across all 32K bins, and dispatch density differs by only about
1.3%. At the measured rate, the authentic 32K reductions alone project to about
889.6 ms before the other classes or omitted MSM stages.

## Timing and residency boundary

Preparation happens once and is excluded from samples. It validates the
complete manifest and all packed SRS indices before any Metal call, compiles
balanced run topology, reorders records, allocates one shared maximum-sized
scratch arena, and uploads immutable SRS/record/map data. Preparation took
198.387 ms in this run.

Each sample then:

1. creates one command buffer and compute encoder;
2. clears one 16-byte status on the GPU;
3. records gather, prepare, product reduction, root inversion, reverse inverse
   expansion, affine finish, carry, and root-export kernels for all four chunks;
4. commits once and waits once;
5. reads only the required 16-byte status; and
6. retains all bucket roots on the device.

The reported total includes encoding, commit/completion wait, and status read.
GPU timestamp lookup occurs afterward and is reported separately. Activity
counters prove zero timed buffer allocations, uploads, blits, fallback, or
execution failures. The 548.3 MB arena is below one percent of the reported
55.7 GB recommended working set, so memory pressure is not the explanation.

## Correctness boundary

This remains a diagnostic, not an MSM implementation. It deliberately omits
digit histogram/scatter, weighted bucket integration, window collection,
erasure, and proof wiring. Those omissions make the performance rejection
stronger: every omitted stage adds work.

The arithmetic path itself is not synthetic:

- the Rust/MSL ABI is probed at session startup;
- branchless affine-or-infinity formulas cover ordinary add, double, inverse,
  both identity orders, infinity plus infinity, signed inputs, odd carries, and
  intermediate cancellation;
- product-tree sizes 1, 31/32/33, 63/64/65, 127/128/129, 255/256/257, and
  511/512/513 match BLST;
- deterministic ragged subgroup runs include a 16,797-point run;
- authoritative on-curve non-subgroup points match BLST; and
- invalid record indices are rejected before command creation, dispatch,
  allocation, or wait.

The authentic-shape benchmark uses valid synthetic subgroup SRS points and
irregular deterministic signed indices. Every measured status was zero. It
does not read 146,727 roots in the timed path; the separate differential suite
owns value correctness.

## Why this is a hard stop

The result is not close enough for a layout or inversion-chain review:

- the review band starts at 126.3M pairs/s; measured throughput is 24.25% of it;
- promotion requires 153.5M pairs/s; measured throughput is 20.0% of it;
- command encoding is only 0.12% of the sample median;
- the GPU interval accounts for essentially the complete wall time; and
- even an impossible free removal of all carry-copy work, assuming cost scaled
  directly with pair count, would remain more than 3x below the stop rate. Carry
  traffic is only about 302 MB, or 3.37% of the nominal 8.946 GB device-buffer
  traffic.

Reaching the immediate gate would require reducing the representative sample
from 221.794 ms to 53.794 ms, a 168.000 ms or 75.7% reduction. Any fused or
threadgroup-local proposal capable of that is a materially different algorithm
and needs a new bound and decision contract; it is not a tuning continuation of
this rejected implementation.

Adding the prior unrealistically optimistic 35.4 ms weighted-integration
allowance gives about 1,538 ms for MSM alone. Against the 1,757.940 ms whole-proof
3x gate, that would leave only about 220 ms for routing, NTTs, every PLONK argument,
quotient construction, evaluations, multi-open, transcript boundaries, and
verification. That is structurally impossible, and the estimate still omits
real MSM stages.

## Alternative-algorithm disposition

A follow-up review did not find a credible M1 Max path from this result to the
5.01x promotion improvement:

| Direction | Optimistic bound | Disposition |
| --- | ---: | --- |
| Threadgroup batch inversion, `G=128/256/512/4096` | about 454.5 / 400.8 / 374.0 / 350.5 ms before routing/integration | Does not reach 300 ms; infinite amortization approaches the already insufficient 347.1 ms arithmetic model. |
| Threadgroup product tree | `(2G-1)*48` bytes: 24.5 KB at 256, 49.1 KB at 512 | M1 exposes 32 KB threadgroup memory; 512 does not fit and 256 leaves little occupancy headroom. |
| Measured recursive flat-segmented tree, `G=128/256` | 14,479.771 / 17,756.113 ms full projections | Both fail decisively; lane-zero inversion plus group barriers make G128 48.3x slower than the promotion rate. |
| Exact register-local `7 M/S + 9 A/S` ceiling | 164.206M/s; 280.374 ms arithmetic-only | Clears the bare rate by only 6.97% and fails the predeclared 200M/s headroom gate; do not materialize a hierarchical scan. |
| Cached Jacobian mixed addition | roughly ten multiplication equivalents, or 298.9 ms multiplication-only | No allowance for additions, routing, memory, or integration. |
| Homogeneous two-point batching | published GPU gain around 3% | Far below the required 5.01x. |
| Fixed-base precomputation | published single-digit percentage savings at relevant sizes | Even 9.6% applied to the optimistic 347.1 ms model is about 313.8 ms before omitted work. |
| BLS12-381 G1 GLV | two approximately 128-bit scalar streams replace one 255-bit stream | Shortens serial scalar multiplication but leaves Pippenger digit/bucket-update count approximately unchanged. |
| CPU+GPU combination | idealized combined rate about 51.8M pairs/s, roughly 888.5 ms | Misses the gate and would require a separately named heterogeneous backend. |

Routing changes cannot fix this reducer: it already starts from grouped runs and
excludes histogram/scatter/radix work. Wider windows, run-overhead removal, or
root-retirement tuning are also single-digit effects relative to the required
improvement.

The hardware and algorithm references are Apple's
[Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf),
[compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/),
and [reduction guidance](https://developer.apple.com/videos/play/tech-talks/10858/);
[PipeMSM](https://eprint.iacr.org/2022/999.pdf) and
[CycloneMSM](https://eprint.iacr.org/2022/1396.pdf) for FPGA batch-affine and
extended-coordinate mappings; [Chen et al., TCHES
2024](https://doi.org/10.46586/tches.v2024.i2.522-544) for GPU point-arithmetic
costs; [Luo, Fu, and Gong, TCHES
2023](https://doi.org/10.46586/tches.v2023.i2.358-380) for fixed-base MSM; and
[Bowe](https://eprint.iacr.org/2019/814.pdf) plus the original
[GLV paper](https://www.iacr.org/archive/crypto2001/21390189.pdf) for the
BLS12-381 endomorphism and scalar-decomposition context.

## Consequence and next decision

Do not optimize this global batch-affine design further and do not build its
record router. Do not treat NTT throughput or runtime readiness as evidence of
a usable proof backend. The safe paths are now an explicit owner decision:

1. fund a new MSM-algorithm research spike with a separately falsifiable
   arithmetic/traffic bound;
2. move the 3x target to materially faster Apple GPU hardware and rerun the
   same frozen gates;
3. relax the M1 Max performance target; or
4. stop the Metal proof route while retaining the backend-neutral architecture,
   correctness probes, and NTT work for future hardware.

Do not start another M1 Max MSM implementation branch under the current gates.
The final flat per-lane segmented Montgomery-trick falsifier has now run and
failed at 3.180M/s, versus the required 153.5M/s. The subsequent exact fused
arithmetic ceiling reached 164.206M/s but failed its 200M/s implementation
threshold and left no credible allowance for real scan dataflow. Further work
requires a new hardware, target, heterogeneous-backend, or field/formula-engine
decision.

Host-module splitting, failure quarantine/erasure, neutral resident handles,
strict verification accounting, and native CI are still valid architecture
work, but full resident PLONK implementation should not proceed speculatively
until an MSM design clears its independent gate.

## Reproduction

Generate the owner-only sensitive replay manifest from the separately captured
seed-24302 aggregate diagnostic:

```bash
python3 scripts/metal_msm_replay.py \
  /tmp/metal-msm-runs-seed24302.jsonl \
  --expect-seed-24302 \
  --output /tmp/metal-msm-replay-seed24302.json
```

Run the exact release diagnostic:

```bash
CARGO_INCREMENTAL=0 cargo test --release \
  -p midnight-metal-prover --lib \
  metal::batch_affine_authentic::authentic_four_chunk_batch_affine_decision_gate \
  -- --ignored --exact --nocapture --test-threads=1
```

The test validates status, command/wait/dispatch counts, zero timed
allocations/uploads/blits/failures, host byte boundaries, and arena accounting
before reporting performance. A `reject` result is a successful execution of
the decision test, not a Rust test failure.
