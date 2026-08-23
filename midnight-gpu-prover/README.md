# Midnight GPU prover facade

This facade does not define a CUDA+Metal hybrid. CUDA is a separate Linux
backend. On macOS the production route is CPU+Metal: CPU retains witness
synthesis and transcript/orchestration while Metal owns NTT, fixed-base MSM,
quotient construction and commitments, and resident multi-open. The `cuda` and
`metal` implementation features are therefore mutually exclusive.

This crate is the compile-time backend boundary for the resident BLS12-381
prover. Linux/CUDA and Apple-silicon/Metal implementations live in symmetric
`cuda` and `metal` modules. Platform-selected consumers use `target`.

The facade is intentionally BLS12-381-specific and does not use trait objects.
Resident buffers and contexts cross the proof pipeline, so a dynamic backend
abstraction would spread associated types through the PLONK implementation.
Platform-selected concrete types keep the hot path monomorphic.

The neutral storage contract is `ProofArena` plus non-owning,
generation-checked `ResidentFrBuffer` tokens. Tokens expose no backend pointer,
queue, download, or arithmetic methods. Concrete Metal arenas use the proven
clear/quarantine/poison lifecycle in `midnight-metal-prover`; this crate does
not label CPU execution as resident Metal work.

Immutable graph import accepts only `CompiledProtocolGraphFamily`. Its
constructor takes custom, lookup, lookup-input, lookup-table, and trash groups
separately, derives their counts, and packs them in protocol order. The facade
preflights the exact domain, field extents, proving-key identity, and SRS
identity before making fieldwise cross-crate copies.

`metal::prepare_resident_key_context` imports reusable public key state and
prepares both KZG SRS bases in the standalone MSM worker. The production
quotient boundary, `metal::with_resident_quotient_epoch`, keeps split quotient
pieces opaque and lends them only to
`msm_prepared_resident_quotient_batch`. No CPU scalar slice is exposed for
commitment. `materialize_resident_quotient_combination` is the one explicit
whole-polynomial host continuation and prevents a pure-Metal claim.

`metal::with_resident_multiopen` keeps q/f/pi arithmetic and the f/pi
commitments in a callback-scoped Metal arena. Only compact evaluations and G1
outputs leave that arena. Native MSM belongs wholly to
`bls12-381-metal-msm`; this facade adapts neutral handles into it, while the
standalone package cannot import this facade or proof-system types.

The earlier `with_diagnostic_quotient_epoch` CPU-BLST lease and shadow route are
retained as explicit test support and historical evidence. They are not
selected by production.

`ExecutionGuard` preserves the existing Linux/CUDA per-domain MSM profile,
diagnostic overrides, affinity key, and sppark lane lifetime. The current
Metal-hybrid proof guard is owned by proof-server and the patched proof entry
points. The facade's `execution_routed` status remains reserved for the future
fully resident target; it must not be read as denying the separately named and
measured `metal-hybrid` route.

`BackendStatus` keeps runtime readiness, proof-pipeline completeness, and the
future pure-Metal application route separate. A working smoke kernel alone
cannot make proof-server execute CPU work under a Metal label. Shared APIs must
not require CUDA stream, pointer, or tuning details. See
`../docs/metal-backend.md` for lifecycle and validation requirements.

The existing CUDA proof route still spans patched proof crates and proof-server
rather than this facade. It remains a semantic and performance reference, not a
runtime dependency of the Metal route.
