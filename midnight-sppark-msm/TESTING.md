# GPU correctness regression tests

Run on a CUDA-capable host:

```bash
cargo test --release -p midnight-sppark-msm --features test-hooks -- --test-threads=1
```

The `test-hooks` feature compiles producer-delay hooks used only by the backend tests.
It is not included in the proof server's `gpu-blitzar` feature or in default builds.
Do not enable it in production packages.

The four `*_waits_for_delayed_*_producer` tests check the resident MSM stream boundary:

- Direct device MSM must observe pending writes on the primary producer stream.
- Dense and sparse ragged device MSM must observe those writes before compaction or accumulation.
- Scrambled device MSM must observe the scalar-mask kernel before reading its temporary buffer.

The test hooks delay the producer by 100 ms, while independent consumer streams remain free
to run. Inputs and expected CPU results are fixed. This exposes missing dependencies without
depending on random proof blinding. The scalar-mask test also initializes the temporary buffer
so premature reads have defined contents.

In sppark's `gpu_t`, implicit conversion to `cudaStream_t` selects the primary stream, whereas
`gpu[0]` selects a different nonblocking flip-flop stream. Events must be recorded on the
actual producer stream and waited on by every MSM engine's input stream. Recording an event
on `gpu[0]` does not order kernels previously launched on `gpu`.

The dependency is a device-side stream wait, not a host/device-wide synchronization. Keep the
temporary buffer and event alive until all consumers finish. Normal proof verification remains
enabled; retries, disabled verification, and switching the accelerated operation to CPU are
not substitutes for this ordering guarantee.

For memory diagnostics, obtain the test executable with `cargo test ... --no-run`, then run:

```bash
compute-sanitizer --tool memcheck --error-exitcode=87 TEST_EXECUTABLE delayed_ --test-threads=1
```

A clean memory-checker result is supporting evidence, not a guarantee that every cross-stream
ordering bug is absent. Also exercise complete proofs, process restarts, key-cache eviction,
and concurrent server requests using the exact production circuit keys and proof verification.
