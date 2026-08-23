# Ledger 9 Metal blocked-radix-128 field result — 2026-07-16

## Decision

Algorithm version 3, the generated blocked-radix-128 Karatsuba/L4 field
schedule, is **`NO_GO` on the M1 Max**. It passed correctness, optimized-AIR
structure, and the frozen source-liveness limit, but failed both throughput
gates by a wide margin against the frozen 2.0B/1.8B median/p5 requirements:

| Milestone 1 check | Required | Measured | Result |
| --- | ---: | ---: | --- |
| Generated Metal correctness | Exact differential pass | Pass | Pass |
| Source liveness | At most 56 `u32`-equivalent words | 55 words | Pass |
| Optimized AIR structure | No rejected alloca/private-array, dynamic-GEP, or out-of-line structure | Pass | Pass |
| Median complete L4 field throughput | At least 2.0B operations/s | 465,444,165.041 operations/s (465.444M) | **Fail** |
| Nearest-rank p5 complete L4 field throughput | At least 1.8B operations/s | 455,317,995.237 operations/s (455.318M) | **Fail** |

The median reached 23.27% of its gate, a 4.297x miss. The p5 reached 25.30%
of its gate, a 3.953x miss. Source liveness and AIR inspection were necessary
conditions, not evidence that the schedule could sustain the required device
rate. Apple does not expose authoritative physical register-allocation metadata
for this shader, so the AIR result must not be restated as proof of zero
physical spilling; the throughput failure is independently decisive.

This result supersedes only the earlier statement that algorithm version 3 was
pending. It does not revise the algorithm-version-2 result: the separated
Karatsuba/L4 schedule remains rejected at 70 source-live words versus the
56-word limit. It also does not revise any CPU, CPU+Metal, quotient, MSM, or
Amdahl measurement.

The formal both-operands-varying gate used 131,072 logical fields, 32
iterations per dispatch, four dispatches and 16,777,216 complete operations per
sample, threadgroup width 64, two warmups, and twenty decision samples. Each
sample used one command submission and one completion wait; allocation, upload,
and readback were outside the timed interval, and every output was checked. The
[standalone result](../bls12-381-metal-msm/BLOCKED128_KARATSUBA_L4_RESULT.md)
records the complete correctness/AIR analysis, and its
[machine-readable evidence](../bls12-381-metal-msm/evidence/m1-max-blocked128-karatsuba-l4-field-2026-07-16.json)
contains all raw GPU and wall timings.

## Stop point

The milestone order remains the right one:

1. generated standalone field arithmetic;
2. complete point formulas;
3. compact bucket-reduction tree;
4. full standalone Metal MSM;
5. removal of the CPU commitment seam; and
6. resident PLONK integration and production acceptance.

Version 3 failed step 1. Work therefore stops before complete point formulas,
the reduction tree, routing, a full MSM, or further proof-residency integration.
Building those layers around a field mapping that misses its median gate by
4.297x would not be a credible route to the 400/500 ms authentic-MSM gates or
the end-to-end 3x target.

The measurement also falsifies product-count scaling as a performance forecast.
Scaling the prior named-FIOS rate by `300/189` suggested about 2.427B
operations/s only as a prioritization heuristic, but v3 measured 0.465B.
Similarly, dividing the earlier 1.9009T native-`u32` product-only issue rate by
189 gives about 10.06B mathematical product equivalents/s while omitting the
complete recurrence's additions, carries, dependencies, and reduction. Neither
figure predicts complete-field throughput, and neither is the proof-level 10.2x
structural maximum below.

## Project-level performance context

These figures answer different questions and remain separate:

| Boundary | Result or bound | Interpretation |
| --- | ---: | --- |
| Current executable CPU+Metal bridge | 1.010x | Measured routed implementation |
| Resident quotient with CPU MSM | optimistic 1.077x | Measured epoch substituted optimistically; rejected |
| Retain CPU MSM and make every other measured proof interval free | loose 2.125x | Cannot reach 3x even under this idealization |
| Make warm FFT and MSM intervals free | 2.610x | Leaf-only Amdahl ceiling; still below 3x |
| Retain only advice synthesis and outside-proof work | about 10.2x | Infinite-device structural opportunity, not an M1 Max forecast |

The accepted CPU baseline is 5,273.820 ms, so the 3x median line is
1,757.940 ms. The v3 failure does not change the approximately 10.2x maximum
theoretical structural opportunity; it shows that this field mapping cannot
realize it on the tested M1 Max. The only current executable result remains
1.010x.

## Architecture boundary

There is no CUDA+Metal execution design. CPU+Metal is allowed only as an
explicit development architecture while Metal progressively owns resident
work; the intended production route is mostly Metal and has no hidden CPU MSM
fallback.

Any future Metal MSM remains an independently open-sourceable standalone
library. It owns field and group arithmetic, scalar recoding, routing, bucket
reduction, integration, collection, prepared-base state, runtime lifetime, and
erasure. CUDA and sppark provide semantic requirements and independent
fixtures—not execution dependencies or a kernel-topology template. Midnight's
proof system may depend on the library through a one-way resident adapter; the
library must not depend back on ledger, PLONK, transcript, or prover types.

The audited [`carni-ships/zkMetal`](https://github.com/carni-ships/zkMetal)
revision does not alter this decision. Its CPU recoding/count-sort boundary,
measured authentic-shape projection, large-path correctness and allocation
defects, and missing root license reject it as a dependency or shortcut. See
the [clean-room evaluation](ledger-9-zkmetal-bls12-381-evaluation-2026-07-15.md).

## Reopen condition

Do not resume point, tree, MSM, or resident-proof implementation for this
schedule. Reopen Milestone 1 only for materially different arithmetic or
hardware with a newly frozen, both-operands-varying complete-field gate. It
must independently pass correctness, bounded source liveness, optimized-AIR
inspection, the strongest available spill evidence, and at least 2.0B median /
1.8B p5 complete L4 operations/s before any higher layer is authorized.
