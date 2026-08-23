# Generated named-register FIOS field gate

Status: **NO_GO on Apple M1 Max: 1.529B median / 1.455B p5 complete Fp/s**.

This is the final bounded field experiment selected for the current M1 Max
campaign. It removes the addressable temporary array and inner loops retained
in the earlier 1.540B/s wide scalar AIR, but it does not clear the predeclared
2.0B/1.8B complete-field gates. It therefore does not authorize G1, XYZZ,
routing, or an MSM execution API.

## Candidate

- One Metal thread owns one reduced twelve-`u32` Montgomery field element.
- The internal type has named `x0` through `x11` members; the neutral packed
  twelve-limb type exists only at the buffer boundary.
- `build.rs` emits twelve straight-line FIOS rounds with named `t0` through
  `t12` state. There is no inner loop, dynamic indexing, SIMD shuffle, or
  threadgroup scratch inside the multiplication.
- Every round computes
  `T' = (T + fixed * scanned_digit + q * p) / 2^32`. The low `p0` cancellation
  uses `mulhi(q, p0) + (low != 0)`.
- The generated coverage audit contains 288 mathematical matrix terms: 144
  input-product terms and 144 Montgomery-correction terms. This is not a claim
  of 288 widened AIR instructions; constant cancellation lowers the compiled
  classes to 276 widened multiplies, 12 unsigned `mulhi` operations, and 12
  low-word correction-digit multiplies.
- The final comparison propagates borrow through `t12`, subtracts the modulus
  once, and mask-selects a reduced result. Inputs must already be reduced raw
  Montgomery residues.
- Both loop-carried operand orientations are compiled and measured. They are
  mathematically identical but expose different invariant values to the
  compiler.

The optimized AIR contains no out-of-line multiply helper, `alloca`, private
array, SIMD shuffle, or threadgroup address space in either variant. Each has
only bounds handling, the zero-iteration case, and the dependent outer-loop
backedge. Final Apple GPU register allocation and spill metadata remain
unavailable; that unknown cannot rescue a failed throughput decision.

## Correctness

The portable exact-integer recurrence differential covers boundary inputs and
deterministic reduced random pairs. The Apple-silicon suite additionally checks
both orientations and every candidate width (32, 64, 128, 256, and 512) against
BLST 0.3.16 at the raw Montgomery-limb boundary, including:

- a forced final-subtraction vector;
- lengths 0, 1, 15, 16, and 17 and both sides of every threadgroup boundary;
- a 4,113-field repeated chain containing 14 boundary and 4,099 deterministic
  random fields for seven multiplications;
- every output after every scout, warmup, and decision command;
- strict output reduction, ABI/fingerprint equality, command completion, and a
  zero padding-store canary.

An independent exact-integer audit also replayed 49 boundary and 200,000 random
reduced products with no mismatch.

## Predeclared decision run

The host allocates, uploads, computes the BLST oracle, and reads back outside
the timed interval. One retained command encodes four dispatches and waits once.
One logical field, not one product term, counts as one completed operation.

| Parameter | Value |
| --- | ---: |
| Logical fields | 131,072 |
| Multiplications per field and dispatch | 32 |
| Dispatches per command | 4 |
| Complete operations per sample | 16,777,216 |
| Width/orientation scout | one warmup + three samples per candidate |
| Decision warmups / samples | 2 / 20 |
| Selected orientation | accumulator fixed, multiplier scanned |
| Selected width / execution width | 128 / 32 |
| Static threadgroup memory | 0 bytes |

## Result

| Metric | Observed | Required | Fraction of gate |
| --- | ---: | ---: | ---: |
| Median GPU complete Fp/s | 1.529069B | 2.0B | 76.45% |
| Nearest-rank p5 GPU complete Fp/s | 1.455061B | 1.8B | 80.84% |
| Median wall complete Fp/s | 1.470124B | -- | -- |
| Nearest-rank p5 wall complete Fp/s | 1.410710B | -- | -- |

A stronger follow-up used two warmups and five 64-iteration scouts for every
orientation/width pair. It selected the opposite orientation at width 128 and
confirmed a 1.534806B/s median, less than 0.4% above the formal result. Its p5
was 1.388598B/s during a CPU-busy run, so it is confirmatory saturation evidence,
not the frozen decision population. Neither result approaches a pass.

Raw samples, toolchain identity, artifact hashes, and both sweep protocols are
frozen in
[`evidence/m1-max-named-register-fios-field-2026-07-15.json`](evidence/m1-max-named-register-fios-field-2026-07-15.json).

## MSM consequence

This failed prerequisite does not justify implementing the curve or router.
Application-specific workload counts and projections remain in the consuming
project's internal reports; they are deliberately not embedded in this
independently publishable package. The current M1 Max field mappings are closed.
A future design must change hardware or arithmetic materially and rerun a
complete-field gate; this result is schedule-specific, not a universal upper
bound on Metal or every integer representation.
