# Ledger 9 Metal flat-segmented MSM decision result — 2026-07-15

> Handling: internal sensitive diagnostic. Aggregate run topology and workload
> counts are witness-derived. This report is not approved for public release
> without security-owner review. Keep this working copy untracked and mode
> `0600`; Git cannot preserve that owner-only mode, so do not stage or publish it
> without declassification.

## Decision

**Reject the final flat-segmented batch-affine candidate on Apple M1 Max.** The
faster 128-lane variant reduced the representative 6,794,123-pair workload in a
2,136.823 ms median, or 3.180M useful reductions/s. The frozen promotion gate
was 44.261 ms and 153.5M/s. The measured candidate is 48.3x too slow before
digit routing, weighted integration, erasure, or proof integration.

The 256-lane variant was slower still: 2,620.323 ms and 2.593M/s. This closes
the last bounded M1 Max falsifier left open by
`ledger-9-metal-batch-affine-result-2026-07-15.md`. No implemented or measured
Metal MSM arithmetic path satisfies the current 3x backend target.

## Candidate

Every nonempty run is recursively split into same-run tiles of at most 32
points. One lane owns one affine accumulator. At each group-uniform round, 128
or 256 lanes prepare adjusted denominators, build a complete product tree in
threadgroup memory, invert the root once, expand individual inverses, and
finish the affine additions. Tile partials ping-pong through distinct resident
point buffers until each original run has one root.

The canonical host planner proves that:

- tiles never cross run boundaries;
- input and output ranges are dense, unique, and in bounds;
- recursive useful additions telescope to `records - roots`;
- real tile lengths are 1 through 32 and padding descriptors are neutral;
- descriptor sorting changes scheduling only, not semantic output order; and
- all offsets, counts, byte extents, packed SRS indices, pipeline widths, and
  threadgroup-memory limits are checked before Metal allocation.

The timed path uses one retained-reference command buffer, one encoder, one
completion wait, zero timed `MTLBuffer` allocations/uploads/blits, resident
roots, and one 16-byte status read. Command-buffer and encoder driver-object
creation are included in wall time but not in the `MTLBuffer` counter. The two
widths were prepared sequentially, retained concurrently, warmed twice, and
alternated sample-by-sample for 20 paired measurements (20 per width) to reduce
ordering and thermal bias.

## Measured result

| Metric | G128 | G256 |
| --- | ---: | ---: |
| Median / p95 | **2,136.823 / 2,205.504 ms** | **2,620.323 / 2,783.973 ms** |
| Device interval median | 2,135.716 ms | 2,619.967 ms |
| Command encoding median | 0.031 ms | 0.029 ms |
| Useful reductions/s | **3,179,544** | **2,592,857** |
| Full 46,039,066-pair projection | **14,479.771 ms** | **17,756.113 ms** |
| Gap from 153.5M/s promotion rate | **48.28x** | **59.20x** |
| Passes / dispatches per sample | 12 / 17 | 12 / 17 |
| Real / padded tiles | 591,532 / 592,256 | 591,532 / 592,640 |
| Scheduled lane-rounds / batch inversions | 6,833,280 / 53,385 | 6,849,792 / 26,757 |
| Threadgroup product-tree bytes | 12,240 | 24,528 |
| Nominal device-buffer traffic | 833,590,072 B | 833,597,752 B |
| Prepared allocations / bytes | 20 / 70,796,292 B | 20 / 70,793,188 B |
| Timed host read | 16 B | 16 B |

The promotion-time identity is:

```text
6,794,123 / 153,500,000 = 44.261 ms
```

The four selected chunks contain 6,940,850 records, 146,727 roots, and exactly
6,794,123 useful reductions. The complete seed-24302 population contains
46,039,066 reductions. Projection uses the measured useful-reduction rate, not
an isolated arithmetic estimate.

Raw G128 total wall samples in milliseconds:

```text
2127.584875 2196.029917 2077.692833 2135.197500 2303.424542
2097.823500 2159.732084 2171.862584 2094.014167 2138.449083
2110.725250 2098.217334 2189.897084 2092.646625 2036.304792
2119.063875 2205.504417 2157.697916 2142.876084 2143.920834
```

Raw G256 total wall samples in milliseconds:

```text
2608.911625 3086.723708 2590.429250 2666.744750 2640.112875
2602.906500 2783.972916 2616.739833 2663.876042 2623.905875
2569.847917 2633.566584 2697.085042 2630.267708 2534.060542
2604.198041 2607.405333 2536.343458 2651.353125 2615.741834
```

## Correctness and scope

The MSL parameter ABI is separately versioned and device-probed. The global
product-tree reducer remains the BLST-backed value oracle. Both widths match it
for ordinary addition, doubling, inverse pairs, identity in both orders,
signed inputs, intermediate cancellation, tile and group boundaries, recursive
passes, a cancellation exactly across a 32-point tile boundary, and a ragged
16,797-point run. Every measured status was zero.

This is still a favorable diagnostic. It starts from run topology, uses
synthetic subgroup SRS points and indices, and omits digit routing, weighted
bucket integration, erasure, and proof wiring. The replay expansion sorts
histogram lengths before grouping, giving unusually coherent lane work. That
makes a passing result insufficient for production, but makes this failure a
conservative rejection.

Only the Rust planner, pipeline loader, session field, benchmark, and dispatch
entry points are `cfg(test)`. The two MSL kernel symbols are still compiled into
the normal embedded metallib; normal host code cannot load or dispatch them,
but this is artifact inclusion rather than symbol absence. A production build
that requires diagnostic-symbol exclusion needs a separate conditional
translation unit or metallib. Diagnostic buffers are dropped without an
explicit GPU erasure pass and are therefore ineligible for witness-derived
production use.

The manifest loader validates its schema and digest but does not enforce file
mode, and SHA-256 detects changes rather than authenticating provenance. The
operator must regenerate the manifest through the reviewed capture process and
verify mode `0600`. Nominal traffic below accounts for explicit buffer-value
traffic implied by the kernel ABI; it is not a physical DRAM or cache-traffic
measurement.

## Why bounded tuning cannot close the gap

The kernel already removes the global product-tree buffers and cuts nominal
device traffic by about 90.7% relative to the rejected 8.946 GB global-tree
sample. Dispatches fall from 1,569 to 17 and encoding falls below 0.04 ms. The
GPU interval still accounts for essentially the complete wall time.

Each scheduled round performs a 613-operation dependent inversion in lane zero
while the rest of the group waits, plus complete product-tree reduction and
inverse expansion barriers. G256 consumes 24,528 B of the M1 Max 32 KB
threadgroup store, permitting only one resident group per core from this
resource alone. G128 permits more occupancy and wins despite twice as many
inversions. An optimized 460-operation inversion chain changes the arithmetic
model by only a few percent; it cannot close a 48.3x measured gap.

The result is not sensitive to host overhead, dispatch count, status traffic,
or allocation strategy. Reaching the gate would require removing 97.9% of the
winner's sample time while retaining all useful affine reductions. That is a
new field/formula/hardware research program, not a layout optimization of this
candidate.

## Consequence

Do not implement routing, weighted integration, proving-key ownership, or
proof-server routing around either batch-affine candidate. Preserve the
backend-neutral facade, fail-closed route, Metal runtime, ABI probes, resident
NTT, G1 correctness work, canonical replay planner, and measurements as future
hardware foundations.

Continuing toward the requested 3x M1 Max backend now requires an explicit
scope decision: materially newer Apple GPU hardware, a relaxed target, a
separately named heterogeneous backend, or funding for a genuinely new
field/formula engine that first demonstrates at least 153.5M complete useful
reductions/s.

## Reproduction

The manifest must be regenerated locally and retained outside the checkout at
mode `0600`. Then run:

```bash
MIDNIGHT_METAL_SENSITIVE_DIAGNOSTIC=1 \
CARGO_INCREMENTAL=0 cargo test --release \
  -p midnight-metal-prover --lib \
  metal::flat_segmented_authentic::authentic_four_chunk_flat_segmented_decision_gate \
  -- --ignored --exact --nocapture --test-threads=1
```

A reported `reject` is a successful decision measurement, not a Rust test
failure.
