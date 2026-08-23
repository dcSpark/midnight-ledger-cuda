# Ledger 9 Metal project CPU baseline — 2026-07-15

## Result

The balanced zswap transaction workload completed in a CPU median of
`5273.820 ms` on the Apple M1 Max development host. The Metal production gate
is therefore `1757.940 ms` or faster for a 3.0x end-to-end speedup on this
machine.

This is a CPU-only baseline. It does not claim Metal correctness or performance.
It establishes the initial budget; the acceptance run must remeasure CPU in the
paired, alternating CPU/Metal harness and record thermal state alongside both
backends.

## Workload

| Property | Value |
| --- | ---: |
| Fixture | `zswap-transaction` |
| Zswap spends | 2 |
| Zswap outputs | 2 |
| Dust spends | 1 |
| Proof jobs per transaction | 5 |
| Measured iterations | 5 (25 internal proof jobs) |
| Aggregate circuit rows | 56,950 |
| IR instructions | 1,397 |
| Maximum circuit `k` | 15 |

The timed region includes transaction construction, wallet and Dust balancing,
proof generation, normal self-verification, binding seal generation, and final
transaction serialization. Current-stack proving-key generation and funded
fixture setup are outside the timed region.

## Host

| Component | Value |
| --- | --- |
| Model | MacBook Pro, Apple M1 Max |
| CPU | 10 cores: 8 performance, 2 efficiency |
| GPU | 32-core integrated Apple GPU |
| Memory | 64 GB unified memory |
| Metal | Metal 4; compiler 32023.883 |
| macOS | 26.5.2 (25F84) |
| Xcode | 26.6 (17F113) |
| Rust | rustc 1.95.0 |
| Source revision | `e5c65ae5e2ff2f6a382c3140141e552f224f46bf` |

## Reproduction

```bash
./scripts/gpuprof.sh parity \
  --backend cpu \
  --fixture zswap-transaction \
  --repeat 5 \
  --warmup 1 \
  --json benchmark-results/metal-cpu-baseline-m1-max.json
```

The warmup uses seed `24301`; measured iterations use seeds `24302` through
`24306`.

| Iteration | CPU ms | Serialized bytes |
| ---: | ---: | ---: |
| 1 | 5273.820 | 23,476 |
| 2 | 5364.630 | 23,476 |
| 3 | 5257.681 | 23,474 |
| 4 | 5499.574 | 23,476 |
| 5 | 5259.447 | 23,476 |
| **Median** | **5273.820** | |
| **Mean** | **5331.030** | |

All five CPU iterations passed normal self-verification for all 25 internal
proof jobs. The ignored JSON artifact retains raw timings and deterministic
proof digests for local comparison.

## Metal release gate

The backend is accepted only when the same fixture and timing boundary meet all
of the following on this host:

- after two warmups, at least 20 paired alternating CPU/Metal samples;
- Metal median at or below `1757.940 ms` and p95 at or below `2000 ms`;
- paired CPU median divided by Metal median at least `3.0x`;
- every measured proof self-verifies;
- deterministic CPU and Metal transactions are byte-identical;
- fallback count is zero;
- residency and synchronization counters satisfy `docs/metal-backend.md`;
- a separate repeat-100 seed/lifecycle run verifies 100/100 proofs with zero
  post-warmup arena growth and no capacity-manifest overrun.
