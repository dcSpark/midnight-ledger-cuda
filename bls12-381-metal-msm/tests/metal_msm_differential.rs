// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use bls12_381_metal_msm::{
    BLS12_381_SCALAR_MODULUS, CanonicalScalar, FixedWindowBits, MetalMsmEngine,
    MetalMsmObservationProfile, MetalMsmReductionStrategy, MsmJobV1, RawFpMontgomery,
    RawFrMontgomery, RawG1Affine, RawG1Jacobian, ResidentMontgomeryScalars, WindowBits,
};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

fn engine() -> MetalMsmEngine {
    MetalMsmEngine::new(MetalMsmObservationProfile::TrustedLocalDataDependentV1)
        .expect("create standalone Metal MSM engine")
}

fn prepared_points(count: usize) -> (Vec<RawG1Affine>, Vec<blst::blst_p1_affine>) {
    let mut raw = Vec::with_capacity(count);
    let mut reference = Vec::with_capacity(count);
    for index in 0..count {
        let multiplier = (index as u64 + 1).to_le_bytes();
        let mut projective = blst::blst_p1::default();
        let mut affine = blst::blst_p1_affine::default();
        // SAFETY: BLST's process-lifetime generator, initialized scalar bytes,
        // and distinct output remain valid for each call.
        unsafe {
            blst::blst_p1_mult(
                &mut projective,
                blst::blst_p1_generator(),
                multiplier.as_ptr(),
                u64::BITS as usize,
            );
            blst::blst_p1_to_affine(&mut affine, &projective);
        }
        raw.push(RawG1Affine {
            x: RawFpMontgomery { limbs: affine.x.l },
            y: RawFpMontgomery { limbs: affine.y.l },
        });
        reference.push(affine);
    }
    (raw, reference)
}

fn expected_msm(
    bases: &[blst::blst_p1_affine],
    scalars: &[CanonicalScalar],
) -> blst::blst_p1_affine {
    assert_eq!(bases.len(), scalars.len());
    let mut accumulator = blst::blst_p1::default();
    for (base, scalar) in bases.iter().zip(scalars) {
        let bytes = scalar.to_le_bytes();
        let mut projective_base = blst::blst_p1::default();
        let mut term = blst::blst_p1::default();
        let mut sum = blst::blst_p1::default();
        // SAFETY: all inputs are initialized, destinations are disjoint, and
        // canonical Fr uses at most 255 little-endian bits.
        unsafe {
            blst::blst_p1_from_affine(&mut projective_base, base);
            blst::blst_p1_mult(&mut term, &projective_base, bytes.as_ptr(), 255);
            blst::blst_p1_add_or_double(&mut sum, &accumulator, &term);
        }
        accumulator = sum;
    }
    let mut affine = blst::blst_p1_affine::default();
    // SAFETY: `accumulator` is initialized and `affine` is a distinct output.
    unsafe { blst::blst_p1_to_affine(&mut affine, &accumulator) };
    affine
}

fn normalize_metal(point: RawG1Jacobian) -> blst::blst_p1_affine {
    let projective = blst::blst_p1 {
        x: blst::blst_fp { l: point.x.limbs },
        y: blst::blst_fp { l: point.y.limbs },
        z: blst::blst_fp { l: point.z.limbs },
    };
    let mut affine = blst::blst_p1_affine::default();
    // SAFETY: the Metal output has the exact BLST Jacobian memory ABI and the
    // completed kernel initialized all coordinates.
    unsafe { blst::blst_p1_to_affine(&mut affine, &projective) };
    affine
}

fn assert_same_point(actual: RawG1Jacobian, expected: blst::blst_p1_affine) {
    let actual = normalize_metal(actual);
    assert_eq!(actual.x.l, expected.x.l);
    assert_eq!(actual.y.l, expected.y.l);
}

fn scalar(limbs: [u64; 4]) -> CanonicalScalar {
    CanonicalScalar::from_le_limbs(limbs).expect("test scalar is canonical")
}

fn modulus_minus_one() -> CanonicalScalar {
    let mut limbs = BLS12_381_SCALAR_MODULUS;
    limbs[0] -= 1;
    scalar(limbs)
}

fn resident_montgomery_scalars(values: &[CanonicalScalar]) -> ResidentMontgomeryScalars {
    let montgomery =
        host_montgomery_scalars(values).into_iter().map(|value| value.limbs).collect::<Vec<_>>();
    resident_montgomery_limbs(&montgomery)
}

fn host_montgomery_scalars(values: &[CanonicalScalar]) -> Vec<RawFrMontgomery> {
    values
        .iter()
        .map(|value| {
            let canonical = value.to_le_limbs();
            let mut raw = blst::blst_fr::default();
            // SAFETY: BLST receives four initialized canonical little-endian limbs.
            unsafe { blst::blst_fr_from_uint64(&mut raw, canonical.as_ptr()) };
            RawFrMontgomery { limbs: raw.l }
        })
        .collect()
}

fn resident_montgomery_limbs(values: &[[u64; 4]]) -> ResidentMontgomeryScalars {
    let device = MTLCreateSystemDefaultDevice().expect("create default Metal device");
    let byte_len = std::mem::size_of_val(values).max(std::mem::size_of::<[u64; 4]>());
    let buffer = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        .expect("allocate resident scalar test buffer");
    if !values.is_empty() {
        // SAFETY: the shared allocation has exactly enough bytes for this POD slice,
        // and no command can observe it until construction and this copy complete.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().as_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            )
        };
    }
    ResidentMontgomeryScalars::from_synchronized_mtl_buffer(buffer, 0, values.len())
        .expect("retain synchronized resident scalar view")
}

#[test]
fn resident_montgomery_scalar_views_are_worker_transferable() {
    fn require_send<T: Send>() {}
    require_send::<ResidentMontgomeryScalars>();
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn metal_msm_matches_blst_for_boundaries_all_windows_and_complete_cases() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine.prepare_bases(&raw_bases).expect("prepare validated bases");
    assert_eq!(prepared.len(), raw_bases.len());

    let cases = [
        Vec::new(),
        vec![CanonicalScalar::ZERO],
        vec![scalar([1, 0, 0, 0])],
        vec![scalar([2, 0, 0, 0]), scalar([255, 0, 0, 0])],
        vec![scalar([256, 0, 0, 0]), scalar([u64::MAX, 0, 0, 0]), modulus_minus_one()],
    ];

    for window in [WindowBits::W6, WindowBits::W7, WindowBits::W8, WindowBits::W9, WindowBits::W10]
    {
        for scalars in &cases {
            let actual = engine.msm(&prepared, scalars, window).expect("Metal MSM succeeds");
            let expected = expected_msm(&reference_bases[..scalars.len()], scalars);
            assert_same_point(actual, expected);
        }
    }

    let (repeated_raw, repeated_reference) = {
        let raw = vec![raw_bases[0], raw_bases[0]];
        let reference = vec![reference_bases[0], reference_bases[0]];
        (raw, reference)
    };
    let repeated = engine.prepare_bases(&repeated_raw).expect("prepare repeated bases");
    for scalars in [
        vec![scalar([1, 0, 0, 0]), scalar([1, 0, 0, 0])],
        vec![scalar([1, 0, 0, 0]), modulus_minus_one()],
    ] {
        let actual = engine.msm(&repeated, &scalars, WindowBits::W8).expect("complete MSM");
        assert_same_point(actual, expected_msm(&repeated_reference, &scalars));
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn metal_ragged_and_prefix_batches_match_independent_blst_sums() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine.prepare_bases(&raw_bases).expect("prepare validated bases");
    let scalars = [
        scalar([1, 0, 0, 0]),
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
        CanonicalScalar::ZERO,
    ];
    let jobs = [
        MsmJobV1 { base_offset: 2, scalar_offset: 0, term_count: 3, output_index: 2 },
        MsmJobV1 { base_offset: 0, scalar_offset: 3, term_count: 0, output_index: 0 },
        MsmJobV1 { base_offset: 1, scalar_offset: 3, term_count: 3, output_index: 1 },
    ];
    let actual =
        engine.msm_ragged(&prepared, &scalars, &jobs, WindowBits::W9).expect("ragged Metal MSM");
    assert_same_point(actual[0], expected_msm(&[], &[]));
    assert_same_point(actual[1], expected_msm(&reference_bases[1..4], &scalars[3..6]));
    assert_same_point(actual[2], expected_msm(&reference_bases[2..5], &scalars[0..3]));

    let batches: [&[CanonicalScalar]; 3] = [&scalars[..1], &scalars[..4], &[]];
    let actual =
        engine.msm_batch(&prepared, &batches, WindowBits::W10).expect("prefix Metal MSM batch");
    for (index, batch) in batches.iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_msm_matches_blst_for_every_preparation_width() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let scalars = [
        CanonicalScalar::ZERO,
        scalar([1, 0, 0, 0]),
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        scalar([u64::MAX, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
        scalar([42, 0, 0, 0]),
    ];
    let expected = expected_msm(&reference_bases, &scalars);

    for window in [
        FixedWindowBits::W11,
        FixedWindowBits::W12,
        FixedWindowBits::W13,
        FixedWindowBits::W14,
        FixedWindowBits::W15,
        FixedWindowBits::W16,
    ] {
        let prepared =
            engine.prepare_fixed_bases(&raw_bases, window).expect("prepare shifted fixed bases");
        assert_eq!(prepared.len(), raw_bases.len());
        assert_eq!(prepared.window_bits(), window);
        assert_eq!(prepared.expanded_len(), raw_bases.len() * window.windows_per_scalar() as usize);
        let actual = engine.msm_fixed(&prepared, &scalars).expect("fixed-base Metal MSM");
        assert_same_point(actual, expected);
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_ragged_and_prefix_batches_match_blst() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine
        .prepare_fixed_bases(&raw_bases, FixedWindowBits::W13)
        .expect("prepare shifted fixed bases");
    let scalars = [
        scalar([1, 0, 0, 0]),
        CanonicalScalar::ZERO,
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
    ];
    let jobs = [
        MsmJobV1 { base_offset: 2, scalar_offset: 0, term_count: 3, output_index: 2 },
        MsmJobV1 { base_offset: 0, scalar_offset: 3, term_count: 0, output_index: 0 },
        MsmJobV1 { base_offset: 1, scalar_offset: 3, term_count: 3, output_index: 1 },
    ];
    let actual =
        engine.msm_fixed_ragged(&prepared, &scalars, &jobs).expect("ragged fixed-base Metal MSM");
    assert_same_point(actual[0], expected_msm(&[], &[]));
    assert_same_point(actual[1], expected_msm(&reference_bases[1..4], &scalars[3..6]));
    assert_same_point(actual[2], expected_msm(&reference_bases[2..5], &scalars[0..3]));

    let batches: [&[CanonicalScalar]; 3] = [&scalars[..1], &scalars[..5], &[]];
    let actual = engine.msm_fixed_batch(&prepared, &batches).expect("fixed-base prefix batch");
    for (index, batch) in batches.iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_resident_batch_and_ragged_match_blst_without_host_scalar_conversion() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine
        .prepare_fixed_bases(&raw_bases, FixedWindowBits::W13)
        .expect("prepare shifted fixed bases");
    let scalars = [
        scalar([1, 0, 0, 0]),
        CanonicalScalar::ZERO,
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
    ];

    let resident = resident_montgomery_scalars(&scalars);
    let jobs = [
        MsmJobV1 { base_offset: 2, scalar_offset: 0, term_count: 3, output_index: 2 },
        MsmJobV1 { base_offset: 0, scalar_offset: 3, term_count: 0, output_index: 0 },
        MsmJobV1 { base_offset: 1, scalar_offset: 3, term_count: 3, output_index: 1 },
    ];
    let actual = engine
        .msm_fixed_resident_ragged(&prepared, &resident, &jobs)
        .expect("ragged resident fixed-base Metal MSM");
    assert_same_point(actual[0], expected_msm(&[], &[]));
    assert_same_point(actual[1], expected_msm(&reference_bases[1..4], &scalars[3..6]));
    assert_same_point(actual[2], expected_msm(&reference_bases[2..5], &scalars[0..3]));

    let resident_batches = [
        resident_montgomery_scalars(&scalars[..1]),
        resident_montgomery_scalars(&scalars[..5]),
        resident_montgomery_scalars(&[]),
    ];
    let actual = engine
        .msm_fixed_resident_batch(&prepared, &resident_batches)
        .expect("resident fixed-base prefix batch");
    for (index, batch) in [&scalars[..1], &scalars[..5], &[]].iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }

    let resident_sources =
        [resident_montgomery_scalars(&scalars[..3]), resident_montgomery_scalars(&scalars[3..])];
    let source_jobs = [
        MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 3, output_index: 1 },
        MsmJobV1 { base_offset: 1, scalar_offset: 3, term_count: 3, output_index: 0 },
        MsmJobV1 { base_offset: 2, scalar_offset: 1, term_count: 4, output_index: 2 },
    ];
    let actual = engine
        .msm_fixed_resident_sources_ragged(&prepared, &resident_sources, &source_jobs)
        .expect("multi-source ragged resident fixed-base Metal MSM");
    assert_same_point(actual[0], expected_msm(&reference_bases[1..4], &scalars[3..6]));
    assert_same_point(actual[1], expected_msm(&reference_bases[..3], &scalars[..3]));
    assert_same_point(actual[2], expected_msm(&reference_bases[2..6], &scalars[1..5]));
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_host_montgomery_batch_and_ragged_match_blst() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine
        .prepare_fixed_bases(&raw_bases, FixedWindowBits::W13)
        .expect("prepare shifted fixed bases");
    let scalars = [
        scalar([1, 0, 0, 0]),
        CanonicalScalar::ZERO,
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
    ];
    let montgomery = host_montgomery_scalars(&scalars);
    let jobs = [
        MsmJobV1 { base_offset: 2, scalar_offset: 0, term_count: 3, output_index: 2 },
        MsmJobV1 { base_offset: 0, scalar_offset: 3, term_count: 0, output_index: 0 },
        MsmJobV1 { base_offset: 1, scalar_offset: 3, term_count: 3, output_index: 1 },
    ];
    let actual = engine
        .msm_fixed_montgomery_ragged(&prepared, &montgomery, &jobs)
        .expect("ragged host-Montgomery fixed-base Metal MSM");
    assert_same_point(actual[0], expected_msm(&[], &[]));
    assert_same_point(actual[1], expected_msm(&reference_bases[1..4], &scalars[3..6]));
    assert_same_point(actual[2], expected_msm(&reference_bases[2..5], &scalars[0..3]));

    let batches = [&montgomery[..1], &montgomery[..5], &[]];
    let actual = engine
        .msm_fixed_montgomery_batch(&prepared, &batches)
        .expect("host-Montgomery fixed-base prefix batch");
    for (index, batch) in [&scalars[..1], &scalars[..5], &[]].iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_resident_path_rejects_nonreduced_montgomery_residues() {
    let engine = engine();
    let (raw_bases, _) = prepared_points(1);
    let prepared = engine
        .prepare_fixed_bases(&raw_bases, FixedWindowBits::W13)
        .expect("prepare shifted fixed base");
    let invalid = resident_montgomery_limbs(&[BLS12_381_SCALAR_MODULUS]);
    let error = engine
        .msm_fixed_resident(&prepared, &invalid)
        .expect_err("the scalar modulus is not a reduced residue");
    assert!(error.to_string().contains("kernel rejected its input"));

    let invalid = [RawFrMontgomery { limbs: BLS12_381_SCALAR_MODULUS }];
    let error = engine
        .msm_fixed_montgomery(&prepared, &invalid)
        .expect_err("the host Montgomery scalar modulus is not a reduced residue");
    assert!(error.to_string().contains("kernel rejected its input"));

    let valid = host_montgomery_scalars(&[scalar([1, 0, 0, 0])]);
    engine
        .msm_fixed_montgomery(&prepared, &valid)
        .expect("a rejected submission must not poison the engine");
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn fixed_base_long_bucket_tiling_matches_blst_for_batched_skew() {
    let engine = engine();
    let (raw_bases, reference_bases) = prepared_points(2_048);
    let prepared = engine
        .prepare_fixed_bases(&raw_bases, FixedWindowBits::W13)
        .expect("prepare shifted fixed bases");
    let repeated = scalar([
        0x0123_4567_89ab_cdef,
        0x0fed_cba9_8765_4321,
        0x1357_9bdf_2468_ace0,
        0x0123_4567_89ab_cdef,
    ]);
    let scalars = vec![repeated; raw_bases.len()];
    let batches: [&[CanonicalScalar]; 2] = [&scalars, &scalars[..1_537]];

    let actual = engine.msm_fixed_batch(&prepared, &batches).expect("tiled fixed-base MSM");
    for (index, batch) in batches.iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn parallel_projective_reducer_matches_blst_across_windows_and_batches() {
    for strategy in [
        MetalMsmReductionStrategy::ParallelProjectiveV1,
        MetalMsmReductionStrategy::ParallelBranchCompletedJacobianV2,
        MetalMsmReductionStrategy::GpuRoutedDenseJacobianV3,
        MetalMsmReductionStrategy::GpuRoutedDenseXyzzV4,
        MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5,
        MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6,
        MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7,
        MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8,
        MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9,
    ] {
        assert_parallel_strategy(strategy);
    }
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn adaptive_xyzz_rebalances_a_skewed_bucket_pair() {
    let engine = MetalMsmEngine::new_with_strategy(
        MetalMsmObservationProfile::TrustedLocalDataDependentV1,
        MetalMsmReductionStrategy::GpuRoutedAdaptiveXyzzV5,
    )
    .expect("create adaptive XYZZ Metal MSM engine");
    let (raw_bases, reference_bases) = prepared_points(256);
    let prepared = engine.prepare_bases(&raw_bases).expect("prepare validated bases");
    let repeated = scalar([
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x1357_9bdf_2468_ace0,
        0x0123_4567_89ab_cdef,
    ]);
    let scalars = vec![repeated; raw_bases.len()];

    let actual = engine.msm(&prepared, &scalars, WindowBits::W8).expect("adaptive XYZZ MSM");
    assert_same_point(actual, expected_msm(&reference_bases, &scalars));
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn tiled_xyzz_strategies_split_a_long_bucket_without_changing_the_group_sum() {
    let (raw_bases, reference_bases) = prepared_points(8_192);
    let repeated = scalar([
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x1357_9bdf_2468_ace0,
        0x0123_4567_89ab_cdef,
    ]);
    let scalars = vec![repeated; raw_bases.len()];

    for strategy in [
        MetalMsmReductionStrategy::GpuRoutedTiledXyzzV6,
        MetalMsmReductionStrategy::GpuRoutedPackedTiledXyzzV7,
        MetalMsmReductionStrategy::GpuRoutedFineTiledXyzzV8,
        MetalMsmReductionStrategy::GpuRoutedBatchAffineXyzzV9,
    ] {
        let engine = MetalMsmEngine::new_with_strategy(
            MetalMsmObservationProfile::TrustedLocalDataDependentV1,
            strategy,
        )
        .expect("create tiled XYZZ Metal MSM engine");
        let prepared = engine.prepare_bases(&raw_bases).expect("prepare validated bases");
        let actual = engine.msm(&prepared, &scalars, WindowBits::W8).expect("tiled XYZZ MSM");
        assert_same_point(actual, expected_msm(&reference_bases, &scalars));
    }
}

fn assert_parallel_strategy(strategy: MetalMsmReductionStrategy) {
    let engine = MetalMsmEngine::new_with_strategy(
        MetalMsmObservationProfile::TrustedLocalDataDependentV1,
        strategy,
    )
    .expect("create parallel-projective Metal MSM engine");
    let (raw_bases, reference_bases) = prepared_points(8);
    let prepared = engine.prepare_bases(&raw_bases).expect("prepare validated bases");
    let scalars = [
        scalar([1, 0, 0, 0]),
        scalar([255, 0, 0, 0]),
        scalar([256, 0, 0, 0]),
        modulus_minus_one(),
        scalar([0xdead_beef, 0x1234, 0, 0]),
        CanonicalScalar::ZERO,
        scalar([u64::MAX, 0, 0, 0]),
        scalar([42, 0, 0, 0]),
    ];

    let empty = engine.msm(&prepared, &[], WindowBits::W8).expect("empty parallel MSM");
    assert_same_point(empty, expected_msm(&[], &[]));

    for window in [WindowBits::W6, WindowBits::W7, WindowBits::W8, WindowBits::W9, WindowBits::W10]
    {
        let actual = engine.msm(&prepared, &scalars, window).expect("parallel Metal MSM");
        assert_same_point(actual, expected_msm(&reference_bases, &scalars));
    }

    let batches: [&[CanonicalScalar]; 3] = [&scalars[..1], &scalars[..5], &scalars];
    let actual =
        engine.msm_batch(&prepared, &batches, WindowBits::W8).expect("parallel prefix batch");
    for (index, batch) in batches.iter().enumerate() {
        assert_same_point(actual[index], expected_msm(&reference_bases[..batch.len()], batch));
    }

    let repeated_raw = [raw_bases[0], raw_bases[0]];
    let repeated_reference = [reference_bases[0], reference_bases[0]];
    let repeated = engine.prepare_bases(&repeated_raw).expect("prepare repeated bases");
    for exceptional in
        [[scalar([1, 0, 0, 0]), scalar([1, 0, 0, 0])], [scalar([1, 0, 0, 0]), modulus_minus_one()]]
    {
        let actual = engine
            .msm(&repeated, &exceptional, WindowBits::W8)
            .expect("branch-completed exceptional MSM");
        assert_same_point(actual, expected_msm(&repeated_reference, &exceptional));
    }
}
