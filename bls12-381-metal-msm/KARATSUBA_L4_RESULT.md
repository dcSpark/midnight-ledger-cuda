# Karatsuba/L4 field-gate result

The generated subtractive-Karatsuba/L4 field candidate is `NO_GO` for this
Apple M1 Max schedule. Its portable exact-integer model and Metal correctness
suite pass, but the predeclared source-liveness gate fails before performance
testing: one flattened multiply helper already has a 70-word local-temporary
live set, above the limit of 56 `u32` equivalents. Including its inputs raises
that helper to 82 words; retaining the numerator raises separated REDC to at
least 94, and the dependent chain's loop-carried right operand raises the
complete source-level lower bound to 106.

This is an early rejection of one generated schedule, not a throughput result
and not a device-wide impossibility proof. No point formula, routing, bucket,
or MSM execution surface is authorized by this result.

## Frozen implementation

Algorithm version 2 generates a twelve-limb, radix-`2^32` subtractive
Karatsuba product and separated Montgomery REDC for lazy `[0, 4p)` inputs. The
frozen invariants are:

```text
K(1)                 = 1
K(2)                 = 3
K(3)                 = 7
K(6)                 = 21
K(12)                = 63 base products
complete REDC        = 3 * K(12) = 189 base products
schedule fingerprint = 0x757db3673700e544
```

The generator emits named scalar temporaries, fixed indices, and no source
arrays or recursive runtime calls. The product counts are mathematical
schedule invariants; they do not account for carry, borrow, selection, or
register-allocation costs.

## Correctness result

The portable model passes the recursive product identity at widths 2, 3, 6,
and 12, checks the 384-bit Montgomery inverse and exact low-half cancellation,
and differentially validates canonical and L4 residues with deterministic
boundary and random inputs. It observes every permitted zero-, one-, and
two-subtraction canonicalization path.

The generated Metal multiplication kernel also passes its implemented
deterministic correctness suite. The suite includes all 144 Cartesian pairs
formed from canonical
`x in {0, 1, p - 1}` with each valid representative `x + k*p`, `k = 0..3`.
Thus `p`, `2p`, and `3p` are exercised as valid noncanonical representatives
of zero. It also checks dependent chains, legal threadgroup widths, ragged and
zero lengths, output padding canaries, and the generated ABI fingerprint.

Host validation fails closed before device setup for mismatched input lengths
and for either operand equal to the excluded upper bound `4p`. The exact `4p`
case is outside the `[0, 4p)` contract; `4p - 1` remains valid.

Passing these tests establishes arithmetic correctness for the tested
multiplication diagnostic. Direct cancellation instrumentation, a specialized
square, fused D2/add-sub chains, in-place aliases, and point formulas were not
implemented or passed. The result does not make the kernel production-safe or
expose a public field API.

## Liveness result

The predeclared gate counts each live `u32` as one word and each live `ulong`
as two:

| Source-level quantity | `u32` equivalents | Acceptance role |
| --- | ---: | --- |
| Top-level aggregate subset | 48 | Diagnostic only |
| One K12 helper's local temporaries | 70 | Already fails |
| Product helper including its inputs | 82 | Layered audit |
| Separated REDC retaining the numerator | at least 94 | Complete-operation lower bound |
| Dependent chain retaining the fixed RHS | **at least 106** | Chain lower bound |
| Required maximum | **56** | Hard limit |

The 48-word number tracks only the outer `product`, `multiplier`, and
`correction` aggregate transitions. It omits the flattened Karatsuba helper's
inner temporaries and therefore is not an acceptance metric. A reviewed
kill-at-last-use scan reaches 70 local-temporary words inside one K12 helper.
The same audit reaches 82 after including that helper's 24 input components,
at least 94 for separated REDC while its numerator is retained, and at least
106 in the dependent chain while the fixed right operand remains live for the
next iteration. The larger values are conservative source-live lower bounds,
not physical Apple GPU register counts; any one of them is already well beyond
the predeclared limit.

The 6,818-line optimized AIR function confirms that the diagnostic kernel is
fully inlined and contains no AIR `alloca`, out-of-line call, SIMD operation,
threadgroup-address space, private-address-space reference, or dynamic limb
GEP. Its 36 limb GEPs are all fixed. The function retains 181 multiply
instructions after constant strength reduction. Those facts do not repair the
source-liveness failure and do not prove physical residency. Apple tooling did
not expose compiler register allocation or physical spill/fill traffic for
this run, so physical spill status remains unknown.

## Stop decision

Neither the short scout nor the 20-sample field-throughput gate was run. Both
manual performance tests check the 56-word precondition before submitting
timed work and therefore fail closed for this schedule. There are no median or
p5 field-throughput numbers to report.

The proposed dependent-chain benchmark holds one right operand invariant. That
shape is favorable because the compiler can hoist right-operand setup; even a
future pass would still require a second gate with both operands varying. Here,
the earlier liveness failure makes that qualification moot.

Reopening the field milestone requires a materially different, sink-oriented
schedule whose complete dependent-chain live set is at most 56 words, followed by
zero-spill evidence and both field-throughput gates. Refactoring only the
top-level aggregates does not address the failed helper live set.

Machine-readable evidence, artifact hashes, and the exact scope of the AIR
inspection are frozen in
[`evidence/m1-max-karatsuba-l4-liveness-gate-2026-07-15.json`](evidence/m1-max-karatsuba-l4-liveness-gate-2026-07-15.json).
The candidate contract remains in
[`KARATSUBA_L4_CANDIDATE.md`](KARATSUBA_L4_CANDIDATE.md).
