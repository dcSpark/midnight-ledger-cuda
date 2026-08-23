# Metal resident expression and quotient epoch

This document freezes the backend-neutral expression representation and Metal
ABI for the first transcript-visible resident PLONK epoch. It complements
`metal-resident-plonk-plan.md`; the CPU prover remains the semantic oracle and
CUDA remains a separate Linux reference. The macOS production route is
CPU+Metal, never CUDA+Metal.

The IR and ABI described here are now used by the production `metal-hybrid`
route. Historical CPU-BLST diagnostic measurements remain below because they
explain a rejected continuation, not the current backend status.

## Epoch boundary and budget

The production transcript-visible Metal epoch is:

```text
y challenge
  -> upload/gather dynamic coefficient sources
  -> extended-coset transforms
  -> custom, permutation, lookup, and trash identities
  -> vanishing division and inverse NTT
  -> n-1 quotient split and blinding
  -> ordered quotient-limb commitments through resident Metal MSM leases
  -> x challenge
  -> resident limb combination and one explicit host opening materialization
```

The earlier CPU-BLST continuation measured 1,184.911 ms on the representative
five-proof transaction, cost 1.975x its 600 ms budget, and was rejected. The
replacement lends quotient pieces directly to the standalone Metal MSM. In the
accepted complete transaction, quotient commitments have zero CPU-MSM leases;
the remaining materialization is the combined opening after `x`.

## Semantic source and owned IR

`GraphEvaluator<F>` remains the logical builder and CPU oracle. Its liveness
allocator is now shared by the target-neutral lowering and compiles once into
pointer-free owned data. Logical column IDs remain intact in each graph until
the key-wide graph family performs one deterministic compaction. CUDA's
`RawGraph` is not the neutral ABI because it contains host pointers and
platform-sized lengths.

```rust,ignore
#[repr(C)]
struct GraphValueRef {
    kind: u32,
    a: u32,
    b: u32,
}

#[repr(C)]
struct GraphOp {
    opcode: u32,
    target: u32,
    parts_offset: u32,
    parts_len: u32,
    lhs: GraphValueRef,
    rhs: GraphValueRef,
}

struct CompiledGraph {
    constants: Box<[RawFr]>,
    rotations: Box<[i32]>,
    operations: Box<[GraphOp]>,
    horner_parts: Box<[GraphValueRef]>,
    scratch_class: ScratchClass,
}
```

The compiler is the only constructor. It validates every tag, index, range,
logical target, and source before allocation or command creation.
`CompiledProtocolGraphFamily` then accepts the five protocol categories
separately, derives their section counts, and concatenates custom, lookup,
lookup-input, lookup-table, and trash graphs in that exact order. Immutable key
import accepts only this classified type; no caller may relabel a packed family
with an independent section table.

### Value tags

| Tag | Source | `a` | `b` |
| ---: | --- | --- | --- |
| 0 | constant | constant index | zero |
| 1 | intermediate | physical scratch slot | zero |
| 2 | fixed | compact column | rotation index |
| 3 | advice | compact column | rotation index |
| 4 | instance | compact column | rotation index |
| 5 | challenge | challenge index | zero |
| 6 | beta | zero | zero |
| 7 | gamma | zero | zero |
| 8 | theta | zero | zero |
| 9 | trash challenge | zero | zero |
| 10 | y | zero | zero |
| 11 | previous accumulated value | zero | zero |

### Operation tags

Tags `0..7` remain Add, Sub, Mul, Square, Double, Negate, Horner, and
Store. Horner is exactly:

```text
factor = rhs
value = lhs
for part in stored order:
    value = value * factor + part
```

An empty graph yields zero. Otherwise the graph result is the value produced
by the final operation. Logical duplicate targets and forward intermediate
reads are rejected; reuse of a physical slot after its logical value's last
use is intentional.

Peak liveness selects one of six pipelines: 8, 16, 32, 64, 128, or 256 field
elements. A graph requiring 257 slots fails before device work. One universal
`Fr[256]` private array is forbidden unless Metal counter evidence later proves
it superior.

The compiled private arrays cost 256, 512, 1,024, 2,048, 4,096, and 8,192
bytes per thread respectively. On the initial M1 Max all six pipelines report
SIMD width 32 and a maximum threadgroup size of 384, versus 1,024 for the simple
Fr-add pipeline. This establishes resource pressure, not measured spilling.
The authentic transaction benchmark now exercises this epoch through native
Metal commitments. Scratch-class tuning remains subordinate to the complete
paired proof result.

## Metal ABI

The host/shader ABI probe fixes these layouts:

- `GraphValueRef`: size 12, alignment 4, fields at 0/4/8;
- `GraphOp`: size 40, alignment 4, `lhs` at 16 and `rhs` at 28; and
- `GraphEvalParameters`: size 256, alignment 8, field challenges at
  96/128/160/192/224.

Recommended bindings:

| Index | Resource |
| ---: | --- |
| 0 | concatenated constants |
| 1 | concatenated rotations |
| 2 | concatenated operations |
| 3 | concatenated Horner parts |
| 4 | immutable fixed cosets |
| 5 | dynamic cosets |
| 6 | challenges |
| 7 | packed `l0`, `l_last`, `l_active` |
| 8 | read/write numerator |
| 9 | `GraphEvalParameters` |

The parameter block contains row length/stride/rotation scale, graph-array
offsets and counts, compact fixed/advice/instance ranges, challenge count,
argument-column indices, and the five scalar challenges. For custom graphs,
`previous_mode=0` supplies zero and `previous_mode=1` captures
`numerator[row]` before overwrite. Lookup and trash graphs receive zero for
`PreviousValue`; their fixed post-processing independently resumes from the
resident numerator.

One templated evaluator has six named scratch-capacity entry points. A small
`post_kind` controls only the fixed fold after graph evaluation:

- 0: store the graph result;
- 1: apply the five lookup folds; and
- 2: apply the trash fold.

Permutation identities remain a separate fixed kernel. Optional empty arrays
bind retained dummy buffers with a logical count of zero.

### Rotation arithmetic

Both CPU and Metal use signed 64-bit arithmetic equivalent to:

```text
(row + i64(rotation) * i64(rot_scale)).rem_euclid(len)
```

No implementation may call `abs`; `i32::MIN.abs()` overflows. The CPU oracle
now widens the multiply/add to signed 64-bit arithmetic. The Metal differential
corpus must still exercise the extreme rotations before proof integration.

## Exact numerator order

The queue order is protocol state:

1. custom graph;
2. permutation first-set `l0`, last-set Boolean `l_last`, set links, then each
   set's product identity;
3. every lookup in argument order, with initial, Boolean, product, `a-s`, and
   adjacency folds; and
4. every trash argument in argument order.

Every term applies `value = value * y + term`. Lookup or trash work cannot be
reordered merely because its input columns are otherwise independent.

## Buffer views and layouts

Polynomial storage is column-major `[column][row]`, matching the existing
batched Metal NTT stride. The persistent coefficient/source order is advice,
instance, permutation products, lookup products, lookup inputs, lookup tables,
then trash.

The current private `ResidentFrArena` stores one stride and batch count for the
whole two-buffer arena. That is insufficient: after transforming many source
columns, the same storage cannot safely be reinterpreted as one numerator batch
for quotient iNTT. The general proof arena therefore needs checked views with
allocation, offset, logical length, stride, batch count, and generation, or a
dedicated one-batch numerator ping-pong pair.

Initial safe allocation shape:

```text
dynamic coefficients:      source_columns * n
coset ping:                 coset_columns * n_ext
coset pong:                 coset_columns * n_ext
numerator/quotient ping:    n_ext
numerator/quotient pong:    n_ext
```

The numerator allocation must not overlap constants, fixed cosets, dynamic
cosets, challenges, or selector buffers. Metal lowers distinct `device`
pointer arguments under non-alias assumptions; a checked view alone does not
make overlapping bindings valid. Before every dispatch the host also proves
`len > 0`, `len <= stride`, every packed `offset + count`, complete column and
selector extents, `post_kind` in `0..=2`, and `previous_mode` in `0..=1`.

For multiple proof instances, queue order remains outermost by instance:
custom -> permutation -> all lookups -> all trash, then the next instance. It
must not be transposed into all custom passes followed by all permutation
passes across instances.

Reuse the private Metal NTT encoders. Public NTT probes are forbidden in this
epoch because they wait and materialize complete vectors.

## Immutable key cache

`CompiledGraphFamily` hashes a versioned, fixed-width, fieldwise little-endian
serialization of the protocol-ordered graphs and their compact column maps.
That digest is only the graph-family component of the key identity. The key
context separately binds `n`, `n_ext`, the actual fixed/permutation columns,
Lagrange values, selector polynomials, full fixed coefficient source arena,
graph/backend ABI version, metallib/constants fingerprint, and Metal device
registry identity. Cache by:

```text
(content fingerprint,
 n,
 n_ext,
 graph/backend ABI version,
 metallib/constants fingerprint,
 MTLDevice registryID)
```

Do not key by a proving-key pointer or transcript representation alone.
`objc2-metal` exposes `registryID()`. If it is ever zero, keep the cache local
to that `MetalSession` rather than substituting the device name.

## Differential gates

Tests cover:

- zero rows without dispatch, one row, proof-sized power-of-two rows, and a
  diagnostic non-power-of-two length;
- every operation and source tag over zero/one/two/minus-one, modulus carry and
  borrow boundaries, and deterministic random values;
- empty/one/many-part Horner cases whose order is distinguishable;
- rotations `0`, `+/-1`, `+/-3`, `i32::MAX`, and `i32::MIN` at scales 1 and 4;
- liveness fan-out and capacity boundaries 8/9, 16/17, 32/33, 64/65,
  128/129, 256, plus pre-dispatch rejection at 257;
- every malformed tag, forward read, duplicate/missing target, index, and
  offset-plus-length overflow;
- snapshots after custom, permutation, each lookup, and each trash fold with a
  nontrivial `y`;
- deterministic quotient blinds, coefficients, limb commitments, and the
  transcript prefix through `x`; and
- authentic k13/k14/k15 byte-exact proofs with canonical verification, zero
  fallback, zero CPU-MSM quotient leases, and exactly the declared combined
  quotient-opening materialization.

Before removing the old compiler, the neutral arrays must match the current
CUDA lowering byte-for-byte and preserve the existing CUDA reference result.
That is an oracle check, not a reason to refactor or execute CUDA on macOS.
Metal rows then compare with both a scalar IR interpreter and
`GraphEvaluator::evaluate`.

## Ordered implementation

1. **Complete:** checked owned IR and the shared logical-target/liveness
   compiler now feed CUDA's existing raw arrays without changing the separate
   CUDA runtime. The runnable Metal-feature regression covers exact tags,
   Horner order, source bounds, slot reuse, and malformed logical targets.
2. **Complete:** `expression.metal`, all six eager
   pipelines, the 75-word startup ABI probe, checked pre-dispatch validation,
   and the CPU/BLST semantic corpus pass on the M1 Max. The expanded corpus has
   eight tests covering every operation/source tag, empty and ordered Horner,
   extreme signed rotations, custom `PreviousValue`, lookup/trash folds, all
   scratch classes, honest counters, and malformed-request rejection. Every
   class reports SIMD width 32 and maximum threadgroup size 384.
3. **Complete:** key-wide deterministic compaction and
   the combined schema-v2 `MetalKeyContext` bind canonical graph-family, PK,
   domain, NTT ABI, coefficient-SRS, and Lagrange-SRS fingerprints. The context
   validates and caches packed graphs, compact maps, exact fixed/selector
   columns, periodic tables, SRS bases, and typed generation-checked views. The
   NTT ABI accepts separate packed-source and extended-destination strides.
4. **Historical diagnostic, retained:** canonical dynamic sources were uploaded
   once; custom -> permutation -> lookup -> trash evaluation, vanishing
   division, inverse NTT, quotient split/blind, ordered CPU BLST commitments,
   and transcript parity through `x` executed under one owned lifecycle.
5. **Rejected continuation:** the representative
   five-proof safe-epoch median is 1,184.911 ms, 1.975x the 600 ms gate. Preserve
   this path as a correctness/lifecycle oracle.
6. **Complete in production:** `with_resident_quotient_epoch` lends opaque
   quotient pieces directly to the standalone fixed-base Metal MSM, preserves
   transcript order, and materializes only the combined opening required by the
   current finalizer.
7. **Complete in production:** the resident multi-open arena performs q/f/pi
   arithmetic and f/pi commitments before returning compact transcript outputs.
   The complete route passes the authentic paired gate above 3x. Removing the
   declared combined-opening seam is tracked in `metal-resident-plonk-plan.md`.
