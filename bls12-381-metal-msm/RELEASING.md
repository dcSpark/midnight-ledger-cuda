# Releasing

Publication remains blocked by `publish = false` until all of these exist:

1. a standalone repository and final repository URL;
2. named maintainers/CODEOWNERS and private vulnerability reporting;
3. portable MSRV/stable CI plus required Apple-GPU execution for releases;
4. package-archive, license, dependency, semver, and API checks;
5. self-contained public provenance and reproducible benchmark artifacts; and
6. completed security review for every executable API and shader.

Never remove `publish = false` merely to test packaging; `cargo package` works
while publication is disabled.

## Packaging while embedded in another workspace

This directory currently lives inside a parent workspace that uses Cargo
resolver 2. Cargo ignores a member package's resolver selection while invoked
through that workspace, and `cargo package` will normalize such an archive to
resolver 2 even though this standalone manifest selects resolver 3. That
warning is expected for parent-workspace development, but such an archive is
not the standalone release candidate.

Before producing or signing a release archive, copy this directory outside the
parent workspace and run from the copied package root:

```bash
cargo +1.95 metadata --locked --no-deps --format-version=1
cargo +1.95 test --locked --all-targets --release
cargo +1.95 package --locked
```

Inspect the normalized `Cargo.toml` inside the resulting `.crate` and require
`resolver = "3"`. Release CI must build and test that exact extracted archive,
not an archive normalized through an embedding workspace.
