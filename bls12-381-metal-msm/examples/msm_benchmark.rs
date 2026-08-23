// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod benchmark {
    use std::env;
    use std::hint::black_box;
    use std::process::ExitCode;
    use std::time::Instant;

    use bls12_381_metal_msm::{
        BLS12_381_SCALAR_MODULUS, CanonicalScalar, FixedWindowBits, MetalMsmEngine,
        MetalMsmObservationProfile, MetalMsmReductionStrategy, MsmJobV1, RawFpMontgomery,
        RawFrMontgomery, RawG1Affine, RawG1Jacobian, WindowBits, route_jobs,
    };
    use blst::MultiPoint;

    const SCHEMA: &str = "bls12-381-metal-msm/full-msm-benchmark/v1";
    const FIXED_SCHEMA: &str = "bls12-381-metal-msm/fixed-base-benchmark/v1";

    #[derive(Clone, Copy)]
    enum Mode {
        Single,
        Batch,
        Ragged,
    }

    impl Mode {
        const fn name(self) -> &'static str {
            match self {
                Self::Single => "single",
                Self::Batch => "batch",
                Self::Ragged => "ragged",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum FixedScalarInput {
        CanonicalHost,
        ResidentMontgomery,
    }

    impl FixedScalarInput {
        const fn name(self) -> &'static str {
            match self {
                Self::CanonicalHost => "canonical-host",
                Self::ResidentMontgomery => "resident-montgomery",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum ScalarPattern {
        Dense,
        Sparse,
        RandomQuarter,
        RandomSparse,
        RandomHalf,
        RandomFiveEighths,
        RandomThreeQuarters,
        Repeated,
    }

    impl ScalarPattern {
        const fn name(self) -> &'static str {
            match self {
                Self::Dense => "dense",
                Self::Sparse => "sparse_1_of_8_single_bit",
                Self::RandomQuarter => "random_sparse_1_of_4",
                Self::RandomSparse => "random_sparse_3_of_8",
                Self::RandomHalf => "random_sparse_1_of_2",
                Self::RandomFiveEighths => "random_sparse_5_of_8",
                Self::RandomThreeQuarters => "random_sparse_3_of_4",
                Self::Repeated => "repeated_scalar",
            }
        }
    }

    struct Config {
        sizes: Vec<usize>,
        windows: Vec<WindowBits>,
        fixed_windows: Vec<FixedWindowBits>,
        fixed_path: bool,
        fixed_scalar_input: FixedScalarInput,
        repeats: usize,
        warmups: usize,
        mode: Mode,
        pattern: ScalarPattern,
        batch_count: usize,
        seed: u64,
        reduction_strategy: MetalMsmReductionStrategy,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                sizes: vec![8_192],
                windows: vec![
                    WindowBits::W6,
                    WindowBits::W7,
                    WindowBits::W8,
                    WindowBits::W9,
                    WindowBits::W10,
                ],
                fixed_windows: vec![
                    FixedWindowBits::W11,
                    FixedWindowBits::W12,
                    FixedWindowBits::W13,
                    FixedWindowBits::W14,
                    FixedWindowBits::W15,
                    FixedWindowBits::W16,
                ],
                fixed_path: false,
                fixed_scalar_input: FixedScalarInput::CanonicalHost,
                repeats: 5,
                warmups: 1,
                mode: Mode::Single,
                pattern: ScalarPattern::Dense,
                batch_count: 4,
                seed: 0x4d53_4d2d_4d45_5441,
                reduction_strategy: MetalMsmReductionStrategy::GlobalBatchAffineV1,
            }
        }
    }

    struct Workload {
        scalars: Vec<CanonicalScalar>,
        montgomery_scalars: Vec<RawFrMontgomery>,
        jobs: Vec<MsmJobV1>,
        batch_ranges: Vec<std::ops::Range<usize>>,
    }

    pub fn main() -> ExitCode {
        let config = match parse_args() {
            Ok(Some(config)) => config,
            Ok(None) => return ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}\n\n{}", usage());
                return ExitCode::from(2);
            }
        };
        match run(&config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("benchmark failed: {error}");
                ExitCode::FAILURE
            }
        }
    }

    fn run(config: &Config) -> Result<(), String> {
        let engine = MetalMsmEngine::new_with_strategy(
            MetalMsmObservationProfile::TrustedLocalDataDependentV1,
            config.reduction_strategy,
        )
        .map_err(|error| error.to_string())?;

        for &size in &config.sizes {
            let (raw_bases, reference_bases) = deterministic_bases(size);
            let workload = make_workload(config, size)?;

            if config.fixed_path {
                for &window in &config.fixed_windows {
                    let preparation_started = Instant::now();
                    let prepared = engine
                        .prepare_fixed_bases(&raw_bases, window)
                        .map_err(|error| error.to_string())?;
                    let preparation_ms = preparation_started.elapsed().as_secs_f64() * 1_000.0;
                    let reference_started = Instant::now();
                    let expected =
                        expected_first_output(&reference_bases, &workload.scalars, &workload.jobs);
                    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

                    for _ in 0..config.warmups {
                        black_box(execute_fixed(
                            &engine,
                            &prepared,
                            &workload,
                            config.mode,
                            config.fixed_scalar_input,
                        )?);
                    }
                    let mut samples_ms = Vec::with_capacity(config.repeats);
                    let mut first_output = None;
                    for _ in 0..config.repeats {
                        let started = Instant::now();
                        let outputs = execute_fixed(
                            &engine,
                            &prepared,
                            &workload,
                            config.mode,
                            config.fixed_scalar_input,
                        )?;
                        samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                        if first_output.is_none() {
                            first_output = outputs.first().copied();
                        }
                        black_box(outputs);
                    }
                    verify_same_point(
                        first_output.ok_or_else(|| "execution returned no output".to_owned())?,
                        expected,
                    )?;
                    print_fixed_result(
                        config,
                        &engine,
                        size,
                        window,
                        prepared.expanded_len(),
                        preparation_ms,
                        reference_ms,
                        &samples_ms,
                    );
                }
                continue;
            }

            let prepared = engine.prepare_bases(&raw_bases).map_err(|error| error.to_string())?;

            for &window in &config.windows {
                let mut routing_samples_ms = Vec::with_capacity(config.repeats);
                let mut route_shape = None;
                for _ in 0..config.repeats {
                    let started = Instant::now();
                    let mut plan =
                        route_jobs(prepared.len(), &workload.scalars, &workload.jobs, window)
                            .map_err(|error| error.to_string())?;
                    routing_samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                    route_shape.get_or_insert((plan.records().len(), plan.runs().len()));
                    plan.erase();
                }
                let (record_count, run_count) =
                    route_shape.ok_or_else(|| "routing produced no sample".to_owned())?;

                let reference_started = Instant::now();
                let expected =
                    expected_first_output(&reference_bases, &workload.scalars, &workload.jobs);
                let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

                for _ in 0..config.warmups {
                    black_box(execute(&engine, &prepared, &workload, config.mode, window)?);
                }

                let mut samples_ms = Vec::with_capacity(config.repeats);
                let mut first_output = None;
                for _ in 0..config.repeats {
                    let started = Instant::now();
                    let outputs = execute(&engine, &prepared, &workload, config.mode, window)?;
                    samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                    if first_output.is_none() {
                        first_output = outputs.first().copied();
                    }
                    black_box(outputs);
                }
                let actual =
                    first_output.ok_or_else(|| "execution returned no output".to_owned())?;
                verify_same_point(actual, expected)?;

                print_result(
                    config,
                    &engine,
                    size,
                    window,
                    record_count,
                    run_count,
                    reference_ms,
                    &routing_samples_ms,
                    &samples_ms,
                );
            }
        }
        Ok(())
    }

    fn execute(
        engine: &MetalMsmEngine,
        prepared: &bls12_381_metal_msm::PreparedBases,
        workload: &Workload,
        mode: Mode,
        window: WindowBits,
    ) -> Result<Vec<RawG1Jacobian>, String> {
        match mode {
            Mode::Single => engine
                .msm(prepared, &workload.scalars, window)
                .map(|point| vec![point])
                .map_err(|error| error.to_string()),
            Mode::Batch => {
                let batches = workload
                    .batch_ranges
                    .iter()
                    .map(|range| &workload.scalars[range.clone()])
                    .collect::<Vec<_>>();
                engine.msm_batch(prepared, &batches, window).map_err(|error| error.to_string())
            }
            Mode::Ragged => engine
                .msm_ragged(prepared, &workload.scalars, &workload.jobs, window)
                .map_err(|error| error.to_string()),
        }
    }

    fn execute_fixed(
        engine: &MetalMsmEngine,
        prepared: &bls12_381_metal_msm::PreparedFixedBases,
        workload: &Workload,
        mode: Mode,
        scalar_input: FixedScalarInput,
    ) -> Result<Vec<RawG1Jacobian>, String> {
        if matches!(scalar_input, FixedScalarInput::ResidentMontgomery) {
            return execute_fixed_montgomery(engine, prepared, workload, mode);
        }
        match mode {
            Mode::Single => engine
                .msm_fixed(prepared, &workload.scalars)
                .map(|point| vec![point])
                .map_err(|error| error.to_string()),
            Mode::Batch => {
                let batches = workload
                    .batch_ranges
                    .iter()
                    .map(|range| &workload.scalars[range.clone()])
                    .collect::<Vec<_>>();
                engine.msm_fixed_batch(prepared, &batches).map_err(|error| error.to_string())
            }
            Mode::Ragged => engine
                .msm_fixed_ragged(prepared, &workload.scalars, &workload.jobs)
                .map_err(|error| error.to_string()),
        }
    }

    fn execute_fixed_montgomery(
        engine: &MetalMsmEngine,
        prepared: &bls12_381_metal_msm::PreparedFixedBases,
        workload: &Workload,
        mode: Mode,
    ) -> Result<Vec<RawG1Jacobian>, String> {
        let values = workload.montgomery_scalars.as_slice();
        match mode {
            Mode::Single => engine
                .msm_fixed_montgomery(prepared, values)
                .map(|point| vec![point])
                .map_err(|error| error.to_string()),
            Mode::Batch | Mode::Ragged => engine
                .msm_fixed_montgomery_ragged(prepared, values, &workload.jobs)
                .map_err(|error| error.to_string()),
        }
    }

    fn make_workload(config: &Config, size: usize) -> Result<Workload, String> {
        let output_count = match config.mode {
            Mode::Single => 1,
            Mode::Batch | Mode::Ragged => config.batch_count,
        };
        let mut scalars = Vec::new();
        let mut jobs = Vec::with_capacity(output_count);
        let mut batch_ranges = Vec::with_capacity(output_count);

        for output in 0..output_count {
            let (base_offset, term_count) = match config.mode {
                Mode::Single | Mode::Batch => (0, size),
                Mode::Ragged => {
                    let shrink = output.saturating_mul(size / 2) / output_count.max(1);
                    (shrink, size - shrink)
                }
            };
            let scalar_offset = scalars.len();
            scalars.extend(deterministic_scalars(
                term_count,
                config.pattern,
                config.seed ^ (output as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            ));
            let scalar_end = scalars.len();
            jobs.push(MsmJobV1 {
                base_offset: u32::try_from(base_offset)
                    .map_err(|_| "base offset exceeds the version-1 ABI")?,
                scalar_offset: u32::try_from(scalar_offset)
                    .map_err(|_| "scalar offset exceeds the version-1 ABI")?,
                term_count: u32::try_from(term_count)
                    .map_err(|_| "term count exceeds the version-1 ABI")?,
                output_index: u32::try_from(output)
                    .map_err(|_| "output index exceeds the version-1 ABI")?,
            });
            batch_ranges.push(scalar_offset..scalar_end);
        }

        let montgomery_scalars = scalars
            .iter()
            .map(|scalar| {
                let canonical = scalar.to_le_limbs();
                let mut raw = blst::blst_fr::default();
                // SAFETY: BLST receives four initialized canonical limbs.
                unsafe { blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr()) };
                RawFrMontgomery { limbs: raw.l }
            })
            .collect();

        Ok(Workload { scalars, montgomery_scalars, jobs, batch_ranges })
    }

    fn deterministic_bases(count: usize) -> (Vec<RawG1Affine>, Vec<blst::blst_p1_affine>) {
        let generator = unsafe { *blst::blst_p1_generator() };
        let mut projective = Vec::with_capacity(count);
        let mut current = generator;
        for _ in 0..count {
            projective.push(current);
            let mut next = blst::blst_p1::default();
            // SAFETY: initialized inputs and a disjoint output satisfy BLST's contract.
            unsafe { blst::blst_p1_add_or_double(&mut next, &current, &generator) };
            current = next;
        }
        let pointers = projective.iter().map(std::ptr::from_ref).collect::<Vec<_>>();
        let mut affine = vec![blst::blst_p1_affine::default(); count];
        // SAFETY: every pointer remains live and the output has `count` slots.
        unsafe { blst::blst_p1s_to_affine(affine.as_mut_ptr(), pointers.as_ptr(), count) };
        let raw = affine
            .iter()
            .map(|point| RawG1Affine {
                x: RawFpMontgomery { limbs: point.x.l },
                y: RawFpMontgomery { limbs: point.y.l },
            })
            .collect();
        (raw, affine)
    }

    fn deterministic_scalars(
        count: usize,
        pattern: ScalarPattern,
        seed: u64,
    ) -> Vec<CanonicalScalar> {
        let mut state = seed;
        (0..count)
            .map(|index| match pattern {
                ScalarPattern::Dense => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::Sparse if index.is_multiple_of(8) => {
                    let bit = (splitmix64(&mut state) % 254) as usize;
                    let mut limbs = [0_u64; 4];
                    limbs[bit / 64] = 1_u64 << (bit % 64);
                    CanonicalScalar::from_le_limbs(limbs).expect("single-bit scalar is canonical")
                }
                ScalarPattern::Sparse => CanonicalScalar::ZERO,
                ScalarPattern::RandomQuarter if index.is_multiple_of(4) => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::RandomQuarter => CanonicalScalar::ZERO,
                ScalarPattern::RandomSparse if index % 8 < 3 => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::RandomSparse => CanonicalScalar::ZERO,
                ScalarPattern::RandomHalf if index.is_multiple_of(2) => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::RandomHalf => CanonicalScalar::ZERO,
                ScalarPattern::RandomFiveEighths if index % 8 < 5 => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::RandomFiveEighths => CanonicalScalar::ZERO,
                ScalarPattern::RandomThreeQuarters if index % 4 < 3 => {
                    let limbs = [
                        splitmix64(&mut state) | 1,
                        splitmix64(&mut state),
                        splitmix64(&mut state),
                        splitmix64(&mut state) % BLS12_381_SCALAR_MODULUS[3],
                    ];
                    CanonicalScalar::from_le_limbs(limbs).expect("top limb makes scalar canonical")
                }
                ScalarPattern::RandomThreeQuarters => CanonicalScalar::ZERO,
                ScalarPattern::Repeated => CanonicalScalar::from_le_limbs([
                    0x0123_4567_89ab_cdef,
                    0xfedc_ba98_7654_3210,
                    0x1357_9bdf_2468_ace0,
                    0x0123_4567_89ab_cdef,
                ])
                .expect("fixed repeated scalar is canonical"),
            })
            .collect()
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn expected_first_output(
        bases: &[blst::blst_p1_affine],
        scalars: &[CanonicalScalar],
        jobs: &[MsmJobV1],
    ) -> blst::blst_p1_affine {
        let job = jobs
            .iter()
            .find(|job| job.output_index == 0)
            .expect("validated workload has output zero");
        let base_start = job.base_offset as usize;
        let scalar_start = job.scalar_offset as usize;
        let count = job.term_count as usize;
        let mut bytes = Vec::with_capacity(count * 32);
        for scalar in &scalars[scalar_start..scalar_start + count] {
            bytes.extend_from_slice(&scalar.to_le_bytes());
        }
        let projective = bases[base_start..base_start + count].mult(&bytes, 255);
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: `projective` is initialized and `affine` is a disjoint output.
        unsafe { blst::blst_p1_to_affine(&mut affine, &projective) };
        affine
    }

    fn verify_same_point(
        actual: RawG1Jacobian,
        expected: blst::blst_p1_affine,
    ) -> Result<(), String> {
        let projective = blst::blst_p1 {
            x: blst::blst_fp { l: actual.x.limbs },
            y: blst::blst_fp { l: actual.y.limbs },
            z: blst::blst_fp { l: actual.z.limbs },
        };
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: the Metal output uses BLST's Jacobian Montgomery representation.
        unsafe { blst::blst_p1_to_affine(&mut affine, &projective) };
        if affine.x.l == expected.x.l && affine.y.l == expected.y.l {
            Ok(())
        } else {
            Err("Metal output did not match independent BLST Pippenger".to_owned())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn print_result(
        config: &Config,
        engine: &MetalMsmEngine,
        size: usize,
        window: WindowBits,
        record_count: usize,
        run_count: usize,
        reference_ms: f64,
        routing_samples_ms: &[f64],
        samples_ms: &[f64],
    ) {
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut sorted_routing = routing_samples_ms.to_vec();
        sorted_routing.sort_by(f64::total_cmp);
        let samples =
            samples_ms.iter().map(|value| format!("{value:.6}")).collect::<Vec<_>>().join(",");
        let device = engine.device_identity();
        println!(
            concat!(
                "{{\"schema\":\"{}\",\"device\":{{\"name\":{},\"registry_id\":\"0x{:016x}\"}},",
                "\"observation_profile\":\"trusted_local_data_dependent_v1\",",
                "\"reduction_strategy\":\"{}\",",
                "\"build_profile\":\"{}\",",
                "\"mode\":\"{}\",\"scalar_pattern\":\"{}\",\"size\":{},\"batch_count\":{},",
                "\"window_bits\":{},\"warmups\":{},\"repeats\":{},\"setup_in_timed_samples\":false,",
                "\"verification\":\"blst_0_3_16_pippenger_output_0\",",
                "\"blst_reference_ms\":{:.6},",
                "\"routing_record_count\":{},\"routing_run_count\":{},",
                "\"cpu_routing_median_ms\":{:.6},",
                "\"samples_ms\":[{}],\"quantile_method\":\"linear_r7\",",
                "\"p5_ms\":{:.6},\"median_ms\":{:.6},\"p95_ms\":{:.6}}}"
            ),
            SCHEMA,
            json_string(device.name()),
            device.registry_id(),
            reduction_strategy_name(engine.reduction_strategy()),
            if cfg!(debug_assertions) { "debug" } else { "release" },
            config.mode.name(),
            config.pattern.name(),
            size,
            match config.mode {
                Mode::Single => 1,
                Mode::Batch | Mode::Ragged => config.batch_count,
            },
            window.get(),
            config.warmups,
            config.repeats,
            reference_ms,
            record_count,
            run_count,
            quantile(&sorted_routing, 0.50),
            samples,
            quantile(&sorted, 0.05),
            quantile(&sorted, 0.50),
            quantile(&sorted, 0.95),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn print_fixed_result(
        config: &Config,
        engine: &MetalMsmEngine,
        size: usize,
        window: FixedWindowBits,
        expanded_base_count: usize,
        preparation_ms: f64,
        reference_ms: f64,
        samples_ms: &[f64],
    ) {
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(f64::total_cmp);
        let samples =
            samples_ms.iter().map(|value| format!("{value:.6}")).collect::<Vec<_>>().join(",");
        let device = engine.device_identity();
        println!(
            concat!(
                "{{\"schema\":\"{}\",\"device\":{{\"name\":{},\"registry_id\":\"0x{:016x}\"}},",
                "\"observation_profile\":\"trusted_local_data_dependent_v1\",",
                "\"strategy\":\"pre-shifted-xyzz-v10\",",
                "\"build_profile\":\"{}\",",
                "\"mode\":\"{}\",\"scalar_input\":\"{}\",\"scalar_pattern\":\"{}\",\"size\":{},\"batch_count\":{},",
                "\"window_bits\":{},\"windows_per_scalar\":{},\"buckets_per_output\":{},",
                "\"expanded_base_count\":{},\"preparation_ms\":{:.6},",
                "\"warmups\":{},\"repeats\":{},\"setup_in_timed_samples\":false,",
                "\"verification\":\"blst_0_3_16_pippenger_output_0\",",
                "\"blst_reference_ms\":{:.6},",
                "\"samples_ms\":[{}],\"quantile_method\":\"linear_r7\",",
                "\"p5_ms\":{:.6},\"median_ms\":{:.6},\"p95_ms\":{:.6}}}"
            ),
            FIXED_SCHEMA,
            json_string(device.name()),
            device.registry_id(),
            if cfg!(debug_assertions) { "debug" } else { "release" },
            config.mode.name(),
            config.fixed_scalar_input.name(),
            config.pattern.name(),
            size,
            match config.mode {
                Mode::Single => 1,
                Mode::Batch | Mode::Ragged => config.batch_count,
            },
            window.get(),
            window.windows_per_scalar(),
            window.buckets_per_output(),
            expanded_base_count,
            preparation_ms,
            config.warmups,
            config.repeats,
            reference_ms,
            samples,
            quantile(&sorted, 0.05),
            quantile(&sorted, 0.50),
            quantile(&sorted, 0.95),
        );
    }

    fn quantile(sorted: &[f64], probability: f64) -> f64 {
        let position = probability * (sorted.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }

    fn json_string(value: &str) -> String {
        let mut output = String::with_capacity(value.len() + 2);
        output.push('"');
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                character if character <= '\u{1f}' => {
                    output.push_str(&format!("\\u{:04x}", character as u32));
                }
                character => output.push(character),
            }
        }
        output.push('"');
        output
    }

    fn parse_args() -> Result<Option<Config>, String> {
        let mut config = Config::default();
        let mut sizes_set = false;
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--help" || argument == "-h" {
                println!("{}", usage());
                return Ok(None);
            }
            let value =
                arguments.next().ok_or_else(|| format!("missing value after {argument}"))?;
            match argument.as_str() {
                "--size" => {
                    if !sizes_set {
                        config.sizes.clear();
                        sizes_set = true;
                    }
                    for item in value.split(',') {
                        let size = parse_usize(item, "size")?;
                        if size == 0 {
                            return Err("size must be non-zero".to_owned());
                        }
                        config.sizes.push(size);
                    }
                }
                "--window" => {
                    config.fixed_path = false;
                    config.windows = match value.as_str() {
                        "all" | "6,7,8,9,10" => {
                            vec![
                                WindowBits::W6,
                                WindowBits::W7,
                                WindowBits::W8,
                                WindowBits::W9,
                                WindowBits::W10,
                            ]
                        }
                        "6" => vec![WindowBits::W6],
                        "7" => vec![WindowBits::W7],
                        "8" => vec![WindowBits::W8],
                        "9" => vec![WindowBits::W9],
                        "10" => vec![WindowBits::W10],
                        _ => return Err("window must be 6, 7, 8, 9, 10, or all".to_owned()),
                    };
                }
                "--fixed-window" => {
                    config.fixed_path = true;
                    config.fixed_windows = match value.as_str() {
                        "all" | "11,12,13,14,15,16" => vec![
                            FixedWindowBits::W11,
                            FixedWindowBits::W12,
                            FixedWindowBits::W13,
                            FixedWindowBits::W14,
                            FixedWindowBits::W15,
                            FixedWindowBits::W16,
                        ],
                        "11" => vec![FixedWindowBits::W11],
                        "12" => vec![FixedWindowBits::W12],
                        "13" => vec![FixedWindowBits::W13],
                        "14" => vec![FixedWindowBits::W14],
                        "15" => vec![FixedWindowBits::W15],
                        "16" => vec![FixedWindowBits::W16],
                        _ => {
                            return Err(
                                "fixed window must be 11, 12, 13, 14, 15, 16, or all".to_owned()
                            );
                        }
                    };
                }
                "--fixed-input" => {
                    config.fixed_scalar_input = match value.as_str() {
                        "canonical" | "canonical-host" => FixedScalarInput::CanonicalHost,
                        "resident" | "resident-montgomery" | "montgomery" => {
                            FixedScalarInput::ResidentMontgomery
                        }
                        _ => {
                            return Err(
                                "fixed input must be canonical-host or resident-montgomery"
                                    .to_owned(),
                            );
                        }
                    };
                }
                "--repeats" => {
                    config.repeats = parse_usize(&value, "repeats")?;
                    if config.repeats == 0 {
                        return Err("repeats must be non-zero".to_owned());
                    }
                }
                "--warmups" => config.warmups = parse_usize(&value, "warmups")?,
                "--mode" => {
                    config.mode = match value.as_str() {
                        "single" => Mode::Single,
                        "batch" => Mode::Batch,
                        "ragged" => Mode::Ragged,
                        _ => return Err("mode must be single, batch, or ragged".to_owned()),
                    };
                }
                "--pattern" => {
                    config.pattern = match value.as_str() {
                        "dense" => ScalarPattern::Dense,
                        "sparse" => ScalarPattern::Sparse,
                        "random-quarter" | "quarter-random" => ScalarPattern::RandomQuarter,
                        "random-sparse" | "sparse-random" => ScalarPattern::RandomSparse,
                        "random-half" | "half-random" => ScalarPattern::RandomHalf,
                        "random-five-eighths" | "five-eighths-random" => {
                            ScalarPattern::RandomFiveEighths
                        }
                        "random-three-quarters" | "three-quarters-random" => {
                            ScalarPattern::RandomThreeQuarters
                        }
                        "repeated" => ScalarPattern::Repeated,
                        _ => {
                            return Err(
                                "pattern must be dense, sparse, random-quarter, random-sparse, random-half, random-five-eighths, random-three-quarters, or repeated"
                                    .to_owned(),
                            );
                        }
                    };
                }
                "--batch-count" => {
                    config.batch_count = parse_usize(&value, "batch-count")?;
                    if config.batch_count == 0 {
                        return Err("batch-count must be non-zero".to_owned());
                    }
                }
                "--reducer" => {
                    config.reduction_strategy = match value.as_str() {
                        "global-batch-affine-v1" | "global" | "affine" => {
                            MetalMsmReductionStrategy::GlobalBatchAffineV1
                        }
                        "parallel-projective-v1" | "parallel" | "projective" => {
                            MetalMsmReductionStrategy::ParallelProjectiveV1
                        }
                        "parallel-branch-completed-jacobian-v2" | "jacobian-v2" | "jacobian" => {
                            MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2
                        }
                        "gpu-routed-dense-jacobian-v3" | "gpu-routed" | "dense-v3" => {
                            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3
                        }
                        "gpu-routed-dense-xyzz-v4" | "xyzz-v4" | "xyzz" => {
                            MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4
                        }
                        "gpu-routed-adaptive-xyzz-v5" | "adaptive-xyzz-v5" | "xyzz-v5" => {
                            MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5
                        }
                        "gpu-routed-tiled-xyzz-v6" | "tiled-xyzz-v6" | "xyzz-v6" => {
                            MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6
                        }
                        "gpu-routed-packed-tiled-xyzz-v7" | "packed-tiled-xyzz-v7" | "xyzz-v7" => {
                            MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7
                        }
                        "gpu-routed-fine-tiled-xyzz-v8" | "fine-tiled-xyzz-v8" | "xyzz-v8" => {
                            MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8
                        }
                        "gpu-routed-batch-affine-xyzz-v9" | "batch-affine-xyzz-v9" | "xyzz-v9" => {
                            MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9
                        }
                        _ => {
                            return Err(
                                "reducer must be global-batch-affine-v1, parallel-projective-v1, parallel-branch-completed-jacobian-v2, gpu-routed-dense-jacobian-v3, gpu-routed-dense-xyzz-v4, gpu-routed-adaptive-xyzz-v5, gpu-routed-tiled-xyzz-v6, gpu-routed-packed-tiled-xyzz-v7, gpu-routed-fine-tiled-xyzz-v8, or gpu-routed-batch-affine-xyzz-v9"
                                    .to_owned(),
                            );
                        }
                    };
                }
                "--seed" => config.seed = parse_u64(&value, "seed")?,
                _ => return Err(format!("unknown option {argument}")),
            }
        }
        if config.sizes.is_empty() {
            return Err("at least one size is required".to_owned());
        }
        Ok(Some(config))
    }

    fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
        value.parse().map_err(|_| format!("invalid {label}: {value}"))
    }

    fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
        let parsed = value.strip_prefix("0x").unwrap_or(value);
        if value.starts_with("0x") {
            u64::from_str_radix(parsed, 16).map_err(|_| format!("invalid {label}: {value}"))
        } else {
            parsed.parse().map_err(|_| format!("invalid {label}: {value}"))
        }
    }

    const fn reduction_strategy_name(strategy: MetalMsmReductionStrategy) -> &'static str {
        match strategy {
            MetalMsmReductionStrategy::GlobalBatchAffineV1 => "global-batch-affine-v1",
            MetalMsmReductionStrategy::ParallelProjectiveV1 => "parallel-projective-v1",
            MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2 => {
                "parallel-branch-completed-jacobian-v2"
            }
            MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3 => "gpu-routed-dense-jacobian-v3",
            MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4 => "gpu-routed-dense-xyzz-v4",
            MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5 => "gpu-routed-adaptive-xyzz-v5",
            MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6 => "gpu-routed-tiled-xyzz-v6",
            MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7 => {
                "gpu-routed-packed-tiled-xyzz-v7"
            }
            MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8 => "gpu-routed-fine-tiled-xyzz-v8",
            MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9 => {
                "gpu-routed-batch-affine-xyzz-v9"
            }
        }
    }

    fn usage() -> &'static str {
        "Usage: cargo run --release -p bls12-381-metal-msm --features metal --example msm_benchmark -- [options]\n\
         Options:\n\
           --size N[,N...]       term counts (default: 8192; option may repeat)\n\
           --window 6|7|8|9|10|all Booth windows (default: all)\n\
           --fixed-window 11|12|13|14|15|16|all pre-shifted fixed-base path\n\
           --fixed-input canonical-host|resident-montgomery (default: canonical-host)\n\
           --repeats N           timed synchronous samples (default: 5)\n\
           --warmups N           untimed executions (default: 1)\n\
           --mode single|batch|ragged (default: single)\n\
           --pattern dense|sparse|random-quarter|random-sparse|random-half|random-five-eighths|random-three-quarters|repeated (default: dense)\n\
           --batch-count N       outputs for batch/ragged modes (default: 4)\n\
           --reducer NAME        global-batch-affine-v1, parallel-projective-v1,\n\
                                 parallel-branch-completed-jacobian-v2, gpu-routed-dense-jacobian-v3,\n\
                                 gpu-routed-dense-xyzz-v4, gpu-routed-adaptive-xyzz-v5,\n\
                                 gpu-routed-tiled-xyzz-v6, gpu-routed-packed-tiled-xyzz-v7,\n\
                                 gpu-routed-fine-tiled-xyzz-v8, or gpu-routed-batch-affine-xyzz-v9\n\
           --seed N|0xHEX        deterministic scalar seed"
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn main() -> std::process::ExitCode {
    benchmark::main()
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn main() -> std::process::ExitCode {
    eprintln!("msm_benchmark requires --features metal on Apple silicon macOS");
    std::process::ExitCode::FAILURE
}
