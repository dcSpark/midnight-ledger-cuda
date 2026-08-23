# Native `u32` low-limb product gate

Status: **the tested 30×13-bit schoolbook-CIOS product schedule is rejected on
this M1 Max**.

This gate evaluates the only materially different open BLS12-381 field family
found in the public Apple/WebGPU implementations reviewed after the named FIOS
result. It uses 30 radix-`2^13` limbs so every limb product is a native wrapping
`u32` multiply. It is not derived from or coupled to any application or MSM
router.

## Reference product schedule

The conventional product schedule for one complete CIOS multiplication has:

- 900 operand products;
- 900 Montgomery correction-by-modulus terms; and
- 30 low-digit inverse steps.

For BLS12-381 at radix `2^13`, `p0 = 0x0aab` and `n0' = 0x1ffd = -3 mod
8192`, with `p0 * n0' mod 8192 = 8191`. Each inverse step can therefore be
implemented as shifts/adds and a mask, so it is not counted as a native
multiply. The reference count is 1,800 mathematical products before carry
propagation, shifts, masks, comparison, final subtraction, loads, or stores.
The correction operand is compile-time constant, so an optimizer may also trade
some of those products for other instructions; this is an engineering schedule
count, not an algebraic or hardware lower bound.

The existing complete-field gates imply these raw required rates for the
declared reference schedule:

```text
2.0B complete Fp/s × 1,800 = 3.600T u32 products/s median
1.8B complete Fp/s × 1,800 = 3.240T u32 products/s p5
```

Failure of this favorable product-only test rejects promotion of this schedule;
passing it would only authorize a correctness-complete implementation. Because
Apple does not publish a native-`u32` issue peak, the observed rate is not a
hardware-wide upper bound over every possible 30-limb instruction schedule.

## Formal result

The package-owned kernel maintains eight independent observable `uint4`
multiply-add chains per physical thread. Optimized AIR contains eight vector
multiplies in the loop, representing exactly 32 counted scalar products; the
compiler does not reuse a permuted product. The Rust harness probes the ABI,
selects a threadgroup width, performs two warmups, records twenty completed
commands, and checks a wrapping-`u32` oracle after every decision command.

On the Apple M1 Max:

| Metric | Observed | Required | Fraction |
| --- | ---: | ---: | ---: |
| Median raw products/s | **1.901T** | 3.600T | 52.80% |
| P5 raw products/s | **1.823T** | 3.240T | 56.25% |
| Median reference-schedule equivalent | **1.056B Fp/s** | 2.0B | 52.80% |
| P5 reference-schedule equivalent | **1.013B Fp/s** | 1.8B | 56.25% |

The tested mapping must also execute the omitted carry and reduction work, so it
is stopped before field, G1, or MSM code. An implementation that strength-reduces
constant products or otherwise changes the issue mix is a different schedule
and needs its own evidence; this result does not prove a device-wide peak.

Reproduce the decision explicitly:

```bash
CARGO_INCREMENTAL=0 cargo +1.95 test \
  --manifest-path bls12-381-metal-msm/Cargo.toml \
  --features metal-diagnostics \
  report_u32_low_limb_product_issue_gate --release -- \
  --ignored --nocapture --test-threads=1
```

Complete samples and artifact hashes are in
[`evidence/m1-max-u32-low-limb-product-gate-2026-07-15.json`](evidence/m1-max-u32-low-limb-product-gate-2026-07-15.json).

## SIMD-group matrix disposition

An additional non-decision scout tested Apple-family-7 SIMD-group matrix
multiply. The local mixed half×half→FP32 probe reached approximately 4.5T raw
products/s, but a convolution mapping cannot use every dense-GEMM product and
still needs normalization and Montgomery reduction. As an additional
prioritization check, Apple's maintained [MLX](https://github.com/ml-explore/mlx)
dense GEMM on this machine reached about 4.56T products/s at full dense
utilization. That still misses the existing 4.6584T median raw gate for the
1,941-product FP32 candidate before any cryptographic work. The MLX raw samples
and command are not packaged, so this is unfrozen scout context rather than
decision evidence. SIMD-group matrix multiplication is not promoted to a field
implementation on this M1 Max.

This decision does not reject subquadratic integer multiplication. A separated
subtractive-Karatsuba Montgomery engine has a different product bound and must
be evaluated independently.
