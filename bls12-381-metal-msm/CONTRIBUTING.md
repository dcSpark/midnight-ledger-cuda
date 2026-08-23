# Contributing

Use Rust 1.95 or newer and keep the crate buildable independently of its current
parent workspace.

Before submitting a change, run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo doc --no-deps
cargo package --allow-dirty
```

Changes to Metal execution must additionally run strict Clippy and the ignored
differential suite on a physical Apple GPU. Performance changes must include raw
samples, toolchain/metallib identity, and a clear statement of what was not
measured.

Do not add ledger, PLONK, CUDA, sppark, or application routing dependencies.
Arithmetic candidates need a written feasibility gate before shader work.
Executable cryptography additionally requires differential vectors, failure and
erasure tests, leakage-profile review, and measured Apple-GPU evidence.

Never commit witness material, proving keys, secret scalars, or private traces.
