# Ledger 9 CUDA Prover Benchmark

**Run date:** 2026-07-14  
**Status:** Repeat-5 benchmark passed; extended concurrency stress follow-up remains open  
**Source checkpoint:** `a0c9ae2d55466c25976f96190cb39458534967b4`

## Result

The Ledger 9 proof server proved a balanced zswap transaction in a median
`385.982 ms` on CUDA versus `3751.063 ms` on CPU, a **9.718x median speedup**.
All five timed CPU/CUDA outputs were byte-for-byte identical and every generated
proof passed the normal in-process verifier.

```text
Benchmark: Ledger 9 proof server, CUDA 13.0, RTX 4090
Workload: 2 zswap spends + 2 outputs + 1 Dust rebalancing spend
Proofs: 5 proofs, 56,950 aggregate circuit rows

CPU median:   3,751.063 ms
CUDA median:    385.982 ms
Speedup:          9.718x
Parity:           5/5 byte-exact and self-verified
```

## Test System

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9 |
| NVIDIA driver | 580.126.18 |
| CUDA toolkit | 13.0, nvcc 13.0.88 |
| Nsight Systems | 2025.3.2 |
| Nsight Compute | 2025.3.1 |
| Rust | rustc 1.95.0 |
| Source repository | `git@github.com:dcSpark/midnight-ledger-cuda.git` |
| Source branch | `main` |

## Workload Contract

The fixture starts with an unbalanced ledger transaction and uses the actual
Ledger 9 test wallet and fee-balancing path. It selects two shielded UTXOs,
creates payment and change outputs, adds one Dust input for fees, and proves the
five resulting relations through `Transaction::prove`.

| Property | Value |
| --- | ---: |
| Zswap spends | 2 |
| Zswap outputs | 2 |
| Dust spends | 1 |
| Proof jobs | 5 |
| Aggregate circuit rows | 56,950 |
| IR instructions | 1,397 |
| Maximum circuit `k` | 15 |
| Input value | 130 |
| Payment | 100 |
| Change | 30 |

The timed region includes transaction construction, wallet/Dust balancing,
proof generation, proof self-verification, binding seal generation, and final
transaction serialization. Current-stack key generation and funded fixture
setup are outside the timed region for both backends.

## Reproduction

Run from the repository root on the system described above:

```bash
./scripts/gpuprof.sh parity \
  --backend both \
  --fixture zswap-transaction \
  --repeat 5 \
  --warmup 1 \
  --json benchmark-results/pre-push-parity-safe-repeat5.json
```

The gate uses deterministic seed `24301` for the warmup. Timed iterations use
seeds `24302` through `24306`. CPU runs first, followed by CUDA in the same
process, exercising proving-key and GPU-context lifecycle behavior.

## Iteration Log

| Iteration | Seed | CPU ms | CUDA ms | Speedup | Serialized bytes | Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 24302 | 3749.616 | 386.342 | 9.705x | 23,476 | yes |
| 2 | 24303 | 3765.423 | 378.288 | 9.954x | 23,476 | yes |
| 3 | 24304 | 3751.063 | 377.054 | 9.948x | 23,474 | yes |
| 4 | 24305 | 3758.524 | 386.359 | 9.728x | 23,476 | yes |
| 5 | 24306 | 3739.010 | 385.982 | 9.687x | 23,476 | yes |
| **Median** | | **3751.063** | **385.982** | **9.718x** | | **5/5** |
| **Mean** | | **3752.727** | **382.805** | **9.803x** | | |

Parity is a direct byte comparison of the complete sealed and serialized
transaction, not a digest-only comparison. No proof outputs were skipped.

## CUDA Transfer Accounting

| Direction | Five-run total | Average per transaction |
| --- | ---: | ---: |
| Host to device | 495,050,200 bytes | 94.423 MiB |
| Device to host | 123,361,600 bytes | 23.529 MiB |

The largest transfer owners across the five timed CUDA iterations were device
polynomial uploads (`288.229 MiB` H2D), NTT bridge traffic (`90.001 MiB` in
each direction), MSM context batches (`29.005 MiB` H2D), device polynomial
downloads (`26.177 MiB` D2H), and MSM context creation (`24.687 MiB` H2D).
These counters are emitted at the CUDA FFI boundary and are part of the parity
JSON contract.

## Implementation Under Test

The CUDA backend is integrated through locally vendored and modified
`midnight-proofs`, `midnight-curves`, `midnight-zk-stdlib`, and `sppark` code.
The prover keeps eligible PLONK polynomial state resident across MSM, NTT,
lookup/permutation, numerator, quotient, evaluation, and KZG multi-open stages.
The proof-server backend is selected explicitly and does not silently fall back
to CPU when CUDA is requested.

Correctness support includes caller-controlled deterministic proof randomness,
normal proof self-verification, complete serialized-output comparison, and
primitive-level CPU/GPU tests. Before this benchmark, all 14
`midnight-sppark-msm` tests and all 22 `proofscope` analyzer tests passed.

## Validation Boundary

This document reports the completed repeat-5 benchmark above. A subsequent
CUDA-only repeat-20 stress command exposed an intermittent concurrency failure
at round index 19, seed `24320`, where a zswap spend proof was rejected by
self-verification. No invalid transaction was emitted, but the result means the
current multi-lane path still requires a concurrency fix before this benchmark
should be treated as a production reliability certification. The accepted
k=13 residency experiment also remains opt-in; the default eligibility boundary
is k=14.

## Traceability

- CUDA implementation: `midnight-sppark-msm/`
- Prover modifications: `vendor/midnight-proofs-0.8.1/`
- Proof-server parity gate: `proof-server/src/bin/proof_backend_parity.rs`
- Profiler entrypoint: `scripts/gpuprof.sh`
- Proof-flow architecture: `scripts/proofscope/PROOF_FLOW.md`
- Profiling contract: `scripts/proofscope/SPEC.md`
- Generated benchmark artifacts: intentionally ignored by Git
