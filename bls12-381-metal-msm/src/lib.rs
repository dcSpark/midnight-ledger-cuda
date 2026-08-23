// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]

//! Standalone BLS12-381 multi-scalar multiplication for Apple Metal.
//!
//! The first executable path uses deterministic CPU Booth routing and performs
//! every G1 bucket reduction, weighted integration, and window collection on
//! Metal. It has an explicit data-dependent trusted-local leakage profile and
//! never falls back to CPU curve arithmetic. Historical arithmetic experiments
//! remain available under [`feasibility`]; their results reject only the named
//! candidates and are not global verdicts on this MSM implementation.

extern crate alloc;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
extern crate std;

/// Versioned raw values and job descriptors shared by standalone callers.
pub mod msm_api;
/// Deterministic signed-window recoding and CPU routing.
pub mod router;

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64",
    bls12_381_metal_shaders
))]
mod engine;

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64",
    bls12_381_metal_shaders
))]
pub use engine::{
    MetalDeviceIdentity, MetalMsmEngine, MetalMsmError, MetalMsmObservationProfile,
    MetalMsmReductionStrategy, PreparedBaseIssue, PreparedBases, PreparedFixedBases,
    ResidentMontgomeryScalars,
};

pub use msm_api::{
    BLS12_381_SCALAR_MODULUS, CanonicalScalar, FixedWindowBits, MsmJobV1, RawFpMontgomery,
    RawFrMontgomery, RawG1Affine, RawG1Jacobian, ScalarEncodingError, WindowBits,
};
pub use router::{BucketRunV1, RouteError, RoutePlan, RoutedPointV1, route_jobs};

#[cfg(all(
    test,
    feature = "metal-diagnostics",
    target_os = "macos",
    target_arch = "aarch64",
    bls12_381_metal_diagnostics
))]
mod amns8_oracle;

#[cfg(all(
    test,
    feature = "metal-diagnostics",
    target_os = "macos",
    target_arch = "aarch64",
    bls12_381_metal_diagnostics
))]
mod metal_diagnostic;

/// Hardware- and target-specific feasibility assessments.
///
/// These results apply only to the named arithmetic candidate, declared
/// throughput target, and device roofline. They are not a general verdict on
/// FP32 arithmetic, Metal, or BLS12-381 MSM.
pub mod feasibility {
    /// Historical workload gate for complete, reduced Montgomery Fp
    /// multiplications.
    ///
    /// This freezes the workload-specific decision that rejected the first
    /// candidate. It is not a universal throughput policy for the future library.
    pub const HISTORICAL_TARGET_COMPLETE_FP_MULTIPLICATIONS_PER_SECOND: u64 = 2_000_000_000;

    /// Optimistic modeled FP32 peak used for the M1 Max roofline.
    ///
    /// As is conventional for GPU peak figures, one fused multiply-add counts as
    /// two floating-point operations.
    pub const M1_MAX_FP32_PEAK_FLOPS_PER_SECOND: u64 = 10_400_000_000_000;

    /// Most optimistic scalar product-issue ceiling derived from the FP32 peak.
    ///
    /// This assumes every FP32 lane issues one useful product on every available
    /// FMA slot. It gives the rejected candidate credit for perfect occupancy and
    /// ignores all additions, integer carries, barriers, memory traffic, and launch
    /// overhead.
    pub const M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND: u64 = M1_MAX_FP32_PEAK_FLOPS_PER_SECOND / 2;

    /// Exact-integer limit supplied by the 24-bit FP32 significand.
    ///
    /// Every integer in `[-2^24, 2^24]` is representable. The rejected radix-256
    /// design keeps strict headroom; the carry-normalized candidate reaches this
    /// inclusive boundary and therefore permits no extra leaf accumulation.
    pub const FP32_EXACT_INTEGER_CEILING: u64 = 1 << 24;

    const BALANCED_DIGIT_ABSOLUTE_BOUND: u64 = 128;

    const fn coefficient_bound(width: u64, aggregated_base_digits: u64) -> u64 {
        let digit_bound = aggregated_base_digits * BALANCED_DIGIT_ABSOLUTE_BOUND;
        width * digit_bound * digit_bound
    }

    // For one Karatsuba combine evaluated as
    // `cross = (z_sum - z_low) - z_high`, followed by
    // `output = (z_low + cross) + z_high`, interval arithmetic gives eight times
    // the side-product coefficient bound. `outer_sources` records aggregation that
    // happened in the enclosing splits; the local sum doubles it for `z_sum`.
    const fn recombination_evaluation_bound(half_width: u64, outer_sources: u64) -> u64 {
        let side = coefficient_bound(half_width, outer_sources);
        let sum = coefficient_bound(half_width, outer_sources * 2);
        let cross = sum + side + side;
        side + cross + side
    }

    const DEEPEST_RECOMBINATION_BOUND: u64 = recombination_evaluation_bound(6, 4);
    const MIDDLE_RECOMBINATION_BOUND: u64 = recombination_evaluation_bound(12, 2);
    const TOP_RECOMBINATION_BOUND: u64 = recombination_evaluation_bound(24, 1);
    const MAXIMUM_RECOMBINATION_BOUND: u64 =
        if DEEPEST_RECOMBINATION_BOUND > MIDDLE_RECOMBINATION_BOUND {
            if DEEPEST_RECOMBINATION_BOUND > TOP_RECOMBINATION_BOUND {
                DEEPEST_RECOMBINATION_BOUND
            } else {
                TOP_RECOMBINATION_BOUND
            }
        } else if MIDDLE_RECOMBINATION_BOUND > TOP_RECOMBINATION_BOUND {
            MIDDLE_RECOMBINATION_BOUND
        } else {
            TOP_RECOMBINATION_BOUND
        };

    const FINAL_CONVOLUTION_COEFFICIENT_BOUND: u64 = coefficient_bound(48, 1);
    const MAXIMUM_CARRY_UNPACK_VALUE: u64 =
        FINAL_CONVOLUTION_COEFFICIENT_BOUND + FINAL_CONVOLUTION_COEFFICIENT_BOUND.div_ceil(255);

    /// Decision produced by a feasibility gate before implementation begins.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum CandidateDecision {
        /// This candidate misses its declared target on the modeled M1 Max.
        MissesDeclaredTargetOnM1Max,
        /// The product count is feasible, but omitted work still needs a native
        /// full-kernel measurement.
        ClearsProductRooflineOnly,
    }

    /// Outcome of a native, predeclared candidate benchmark.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeGateDecision {
        /// The tested schedule missed at least one required native throughput bound.
        TestedScheduleMissesPredeclaredGate,
    }

    /// Outcome of a complete reduced-field implementation gate.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum CompleteFieldGateDecision {
        /// The complete implementation missed at least one throughput bound.
        MissesPredeclaredThroughputGate,
    }

    /// Disposition of a primitive substitution that did not reach a complete
    /// field benchmark.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum PrimitiveSubstitutionDecision {
        /// Representative primitive scouts provide no credible path to closing
        /// the measured complete-field gap, so the substitution is deprioritized.
        PrunedByPrioritizationEvidence,
    }

    /// Outcome of the generated Karatsuba/L4 field-candidate gate.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum KaratsubaL4FieldGateDecision {
        /// The flattened helper exceeded the source-liveness limit, so the
        /// schedule stopped before performance testing.
        MissesSourceLivenessGateBeforePerformance,
    }

    /// Pure-data description of the rejected packed-FP32 Montgomery candidate.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PackedFp32MontgomeryAssessment {
        /// Balanced digit radix (`2^8`).
        pub radix: u64,
        /// Number of radix-256 digits in a 384-bit Montgomery value.
        pub digits: u64,
        /// Number of Karatsuba split levels.
        pub karatsuba_levels: u64,
        /// Number of length-six leaf convolutions (`3^3`).
        pub leaf_convolutions: u64,
        /// Scalar FP32 products in one 48-by-48 convolution.
        pub scalar_products_per_convolution: u64,
        /// Convolutions required by the complete Montgomery construction.
        pub convolutions_per_complete_multiply: u64,
        /// Scalar FP32 products required by one complete Montgomery multiply.
        pub scalar_products_per_complete_multiply: u64,
        /// Largest absolute leaf-convolution accumulator.
        pub maximum_leaf_accumulator: u64,
        /// Largest absolute intermediate in Karatsuba recombination.
        pub maximum_recombination_intermediate: u64,
        /// Largest value entering signed carry extraction.
        pub maximum_carry_unpack_value: u64,
        /// Required complete Fp multiplication rate.
        pub required_complete_multiplications_per_second: u64,
        /// Optimistic hardware scalar-product issue ceiling.
        pub peak_scalar_products_per_second: u64,
        /// Scalar-product issue rate the throughput gate would require.
        pub required_scalar_products_per_second: u64,
        /// Optimistic maximum complete multiply rate before any non-product work.
        pub roofline_complete_multiplications_per_second: u64,
        /// Ceiling percentage required in basis points, rounded upward.
        ///
        /// `10_000` means 100%; this candidate requires `11_216`, or 112.16%.
        pub required_product_issue_basis_points: u64,
        /// Fail-closed outcome.
        pub decision: CandidateDecision,
    }

    impl PackedFp32MontgomeryAssessment {
        /// Returns whether this candidate misses its declared M1 Max target.
        #[must_use]
        pub const fn misses_declared_target(self) -> bool {
            self.required_scalar_products_per_second > self.peak_scalar_products_per_second
        }

        /// Returns exact-integer headroom below the FP32 accumulator ceiling.
        #[must_use]
        pub const fn exact_integer_headroom(self) -> u64 {
            FP32_EXACT_INTEGER_CEILING - self.maximum_recombination_intermediate
        }
    }

    /// Frozen roofline result for the balanced-radix-256, three-level-Karatsuba
    /// candidate.
    ///
    /// Exactness itself is feasible: its largest FP32 intermediate is 12,582,912,
    /// below `2^24`. Performance is not: three 972-product convolutions require
    /// 2,916 scalar products per complete reduced Montgomery multiplication.
    pub const M1_MAX_PACKED_FP32_MONTGOMERY_ASSESSMENT: PackedFp32MontgomeryAssessment = {
        const RADIX: u64 = 1 << 8;
        const DIGITS: u64 = 48;
        const KARATSUBA_LEVELS: u64 = 3;
        const LEAF_CONVOLUTIONS: u64 = 27;
        const LEAF_WIDTH: u64 = 6;
        const PRODUCTS_PER_CONVOLUTION: u64 = LEAF_CONVOLUTIONS * LEAF_WIDTH * LEAF_WIDTH;
        const CONVOLUTIONS_PER_COMPLETE_MULTIPLY: u64 = 3;
        const PRODUCTS_PER_COMPLETE_MULTIPLY: u64 =
            PRODUCTS_PER_CONVOLUTION * CONVOLUTIONS_PER_COMPLETE_MULTIPLY;
        const REQUIRED_PRODUCTS: u64 = HISTORICAL_TARGET_COMPLETE_FP_MULTIPLICATIONS_PER_SECOND
            * PRODUCTS_PER_COMPLETE_MULTIPLY;
        const ROOFLINE_COMPLETE_MULTIPLIES: u64 =
            M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND / PRODUCTS_PER_COMPLETE_MULTIPLY;
        const REQUIRED_BASIS_POINTS: u64 =
            (REQUIRED_PRODUCTS * 10_000).div_ceil(M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND);

        PackedFp32MontgomeryAssessment {
            radix: RADIX,
            digits: DIGITS,
            karatsuba_levels: KARATSUBA_LEVELS,
            leaf_convolutions: LEAF_CONVOLUTIONS,
            scalar_products_per_convolution: PRODUCTS_PER_CONVOLUTION,
            convolutions_per_complete_multiply: CONVOLUTIONS_PER_COMPLETE_MULTIPLY,
            scalar_products_per_complete_multiply: PRODUCTS_PER_COMPLETE_MULTIPLY,
            maximum_leaf_accumulator: coefficient_bound(6, 8),
            maximum_recombination_intermediate: MAXIMUM_RECOMBINATION_BOUND,
            maximum_carry_unpack_value: MAXIMUM_CARRY_UNPACK_VALUE,
            required_complete_multiplications_per_second:
                HISTORICAL_TARGET_COMPLETE_FP_MULTIPLICATIONS_PER_SECOND,
            peak_scalar_products_per_second: M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND,
            required_scalar_products_per_second: REQUIRED_PRODUCTS,
            roofline_complete_multiplications_per_second: ROOFLINE_COMPLETE_MULTIPLIES,
            required_product_issue_basis_points: REQUIRED_BASIS_POINTS,
            decision: CandidateDecision::MissesDeclaredTargetOnM1Max,
        }
    };

    const fn carry_normalized_karatsuba_products(digits: u64, levels: u64) -> u64 {
        if levels == 0 {
            return digits * digits;
        }

        let low_digits = digits.div_ceil(2);
        let high_digits = digits / 2;
        carry_normalized_karatsuba_products(low_digits, levels - 1)
            + carry_normalized_karatsuba_products(high_digits, levels - 1)
            + carry_normalized_karatsuba_products(low_digits + 1, levels - 1)
    }

    /// Product-count assessment for carry-normalized radix-4096 Karatsuba.
    ///
    /// This is a prerequisite result, not an executable-field verdict. It omits
    /// carry normalization, Karatsuba recombination, Montgomery correction,
    /// register pressure, conversion, traffic, and command costs.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CarryNormalizedFp32Assessment {
        /// Balanced digit radix (`2^12`).
        pub radix: u64,
        /// Number of radix-4096 digits in a 384-bit value.
        pub digits: u64,
        /// Number of carry-normalized Karatsuba levels.
        pub karatsuba_levels: u64,
        /// Number of schoolbook leaf convolutions (`3^4`).
        pub leaf_convolutions: u64,
        /// Largest leaf width after normalized sums gain at most one digit.
        pub maximum_leaf_width: u64,
        /// Largest exact integer accumulated by one FP32 leaf coefficient.
        pub maximum_leaf_accumulator: u64,
        /// Scalar FP32 products in one complete 32-digit convolution.
        pub scalar_products_per_convolution: u64,
        /// Convolutions required by the complete Montgomery construction.
        pub convolutions_per_complete_multiply: u64,
        /// Scalar FP32 products required by one complete Montgomery multiply.
        pub scalar_products_per_complete_multiply: u64,
        /// Required complete Fp multiplication rate from the historical workload.
        pub required_complete_multiplications_per_second: u64,
        /// Scalar-product issue rate required at that complete-operation rate.
        pub required_scalar_products_per_second: u64,
        /// Optimistic maximum complete rate before any non-product work.
        pub roofline_complete_multiplications_per_second: u64,
        /// Product-issue ceiling required in basis points, rounded upward.
        pub required_product_issue_basis_points: u64,
        /// Necessary-only outcome of the arithmetic roofline.
        pub decision: CandidateDecision,
    }

    impl CarryNormalizedFp32Assessment {
        /// Returns whether the product count fits under the modeled issue ceiling.
        #[must_use]
        pub const fn clears_product_roofline(self) -> bool {
            self.required_scalar_products_per_second <= M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND
        }

        /// Returns unused modeled product-issue capacity in basis points.
        #[must_use]
        pub const fn product_issue_headroom_basis_points(self) -> u64 {
            10_000 - self.required_product_issue_basis_points
        }
    }

    /// Necessary-only roofline result for the carry-normalized radix-4096
    /// candidate.
    ///
    /// Every Karatsuba sum is normalized before entering the next product, and
    /// every child product is normalized before recombination. A normalized sum
    /// can gain one digit, so the recurrence is not the idealized `32^2 * (3/4)^4`.
    /// The exact recurrence yields 647 products per convolution and 1,941 per
    /// complete Montgomery multiplication.
    pub const M1_MAX_CARRY_NORMALIZED_FP32_ASSESSMENT: CarryNormalizedFp32Assessment = {
        const RADIX: u64 = 1 << 12;
        const DIGITS: u64 = 32;
        const KARATSUBA_LEVELS: u64 = 4;
        const LEAF_CONVOLUTIONS: u64 = 81;
        const MAXIMUM_LEAF_WIDTH: u64 = 4;
        const BALANCED_DIGIT_ABSOLUTE_BOUND: u64 = RADIX / 2;
        const MAXIMUM_LEAF_ACCUMULATOR: u64 =
            MAXIMUM_LEAF_WIDTH * BALANCED_DIGIT_ABSOLUTE_BOUND * BALANCED_DIGIT_ABSOLUTE_BOUND;
        const PRODUCTS_PER_CONVOLUTION: u64 =
            carry_normalized_karatsuba_products(DIGITS, KARATSUBA_LEVELS);
        const CONVOLUTIONS_PER_COMPLETE_MULTIPLY: u64 = 3;
        const PRODUCTS_PER_COMPLETE_MULTIPLY: u64 =
            PRODUCTS_PER_CONVOLUTION * CONVOLUTIONS_PER_COMPLETE_MULTIPLY;
        const REQUIRED_PRODUCTS: u64 = HISTORICAL_TARGET_COMPLETE_FP_MULTIPLICATIONS_PER_SECOND
            * PRODUCTS_PER_COMPLETE_MULTIPLY;
        const ROOFLINE_COMPLETE_MULTIPLIES: u64 =
            M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND / PRODUCTS_PER_COMPLETE_MULTIPLY;
        const REQUIRED_BASIS_POINTS: u64 =
            (REQUIRED_PRODUCTS * 10_000).div_ceil(M1_MAX_FP32_PRODUCT_ISSUE_PER_SECOND);

        CarryNormalizedFp32Assessment {
            radix: RADIX,
            digits: DIGITS,
            karatsuba_levels: KARATSUBA_LEVELS,
            leaf_convolutions: LEAF_CONVOLUTIONS,
            maximum_leaf_width: MAXIMUM_LEAF_WIDTH,
            maximum_leaf_accumulator: MAXIMUM_LEAF_ACCUMULATOR,
            scalar_products_per_convolution: PRODUCTS_PER_CONVOLUTION,
            convolutions_per_complete_multiply: CONVOLUTIONS_PER_COMPLETE_MULTIPLY,
            scalar_products_per_complete_multiply: PRODUCTS_PER_COMPLETE_MULTIPLY,
            required_complete_multiplications_per_second:
                HISTORICAL_TARGET_COMPLETE_FP_MULTIPLICATIONS_PER_SECOND,
            required_scalar_products_per_second: REQUIRED_PRODUCTS,
            roofline_complete_multiplications_per_second: ROOFLINE_COMPLETE_MULTIPLIES,
            required_product_issue_basis_points: REQUIRED_BASIS_POINTS,
            decision: CandidateDecision::ClearsProductRooflineOnly,
        }
    };

    /// Frozen result from one carry-normalized candidate schedule's raw
    /// precise-FP32 product-issue gate on an Apple M1 Max.
    ///
    /// This diagnostic measured only 32 observable scalar products per loop
    /// iteration, with no carries, normalization, Karatsuba recombination,
    /// Montgomery correction, or memory routing. Rates divided by 1,941 are
    /// product-count equivalents, not completed field multiplications. The result
    /// rejects this concrete issue schedule; it is not an upper bound on every
    /// possible FP32 instruction schedule.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PreciseFp32ProductIssueGateResult {
        /// Number of timed samples after warmup.
        pub samples: u64,
        /// Independent shader lanes in each dispatch.
        pub elements: u64,
        /// FMA loop iterations per lane.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Observed median scalar-product issue rate.
        pub median_scalar_products_per_second: u64,
        /// Observed fifth-percentile scalar-product issue rate.
        pub p5_scalar_products_per_second: u64,
        /// Median issue rate divided by the 1,941-product arithmetic floor.
        pub median_complete_product_equivalents_per_second: u64,
        /// Fifth-percentile rate divided by the 1,941-product floor.
        pub p5_complete_product_equivalents_per_second: u64,
        /// Required median product-count equivalent rate.
        pub required_median_equivalents_per_second: u64,
        /// Required fifth-percentile product-count equivalent rate.
        pub required_p5_equivalents_per_second: u64,
        /// Fail-closed outcome.
        pub decision: NativeGateDecision,
    }

    impl PreciseFp32ProductIssueGateResult {
        /// Returns whether both predeclared throughput bounds were met.
        #[must_use]
        pub const fn passes(self) -> bool {
            self.median_complete_product_equivalents_per_second
                >= self.required_median_equivalents_per_second
                && self.p5_complete_product_equivalents_per_second
                    >= self.required_p5_equivalents_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-15 with Rust 1.95.0 and Apple
    /// Metal compiler 32023.883.
    pub const M1_MAX_PRECISE_FP32_PRODUCT_ISSUE_GATE_RESULT: PreciseFp32ProductIssueGateResult =
        PreciseFp32ProductIssueGateResult {
            samples: 20,
            elements: 262_144,
            iterations: 256,
            dispatches: 4,
            threadgroup_width: 64,
            median_scalar_products_per_second: 1_945_454_089_254,
            p5_scalar_products_per_second: 1_827_159_711_141,
            median_complete_product_equivalents_per_second: 1_002_294_739,
            p5_complete_product_equivalents_per_second: 941_349_671,
            required_median_equivalents_per_second: 2_400_000_000,
            required_p5_equivalents_per_second: 2_250_000_000,
            decision: NativeGateDecision::TestedScheduleMissesPredeclaredGate,
        };

    /// Frozen complete-field result for the parity-owned, two-lane `u32` FIPS
    /// Montgomery implementation on an Apple M1 Max.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct TwoLaneU32FipsFieldGateResult {
        /// Generated arithmetic-schedule version.
        pub algorithm_version: u64,
        /// Product/reduction slots emitted across the 24 integrated columns.
        pub generated_product_slots: u64,
        /// SIMD lanes assigned to one logical field operation.
        pub threads_per_field: u64,
        /// Logical fields in the decision run.
        pub logical_fields: u64,
        /// Dependent multiplications per field and dispatch.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Timed samples after warmup.
        pub samples: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Observed median complete field multiplications per second.
        pub median_complete_fp_per_second: u64,
        /// Observed nearest-rank fifth-percentile complete rate.
        pub p5_complete_fp_per_second: u64,
        /// Required median complete field rate.
        pub required_median_complete_fp_per_second: u64,
        /// Required fifth-percentile complete field rate.
        pub required_p5_complete_fp_per_second: u64,
        /// Fail-closed result.
        pub decision: CompleteFieldGateDecision,
    }

    impl TwoLaneU32FipsFieldGateResult {
        /// Returns whether both complete-field throughput gates were met.
        #[must_use]
        pub const fn passes(self) -> bool {
            self.median_complete_fp_per_second >= self.required_median_complete_fp_per_second
                && self.p5_complete_fp_per_second >= self.required_p5_complete_fp_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-15 with Rust 1.95.0, BLST
    /// 0.3.16 differentials, and Apple Metal compiler 32023.883.
    pub const M1_MAX_TWO_LANE_U32_FIPS_FIELD_GATE_RESULT: TwoLaneU32FipsFieldGateResult =
        TwoLaneU32FipsFieldGateResult {
            algorithm_version: 2,
            generated_product_slots: 144,
            threads_per_field: 2,
            logical_fields: 131_072,
            iterations: 32,
            dispatches: 4,
            samples: 20,
            threadgroup_width: 32,
            median_complete_fp_per_second: 986_960_401,
            p5_complete_fp_per_second: 939_895_686,
            required_median_complete_fp_per_second: 2_000_000_000,
            required_p5_complete_fp_per_second: 1_800_000_000,
            decision: CompleteFieldGateDecision::MissesPredeclaredThroughputGate,
        };

    /// Loop-carried operand orientation selected for a named-register FIOS
    /// benchmark.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NamedFiosOperandOrientation {
        /// The loop-carried accumulator occupies the generated function's
        /// `fixed` operand while the invariant multiplier supplies scanned
        /// digits.
        AccumulatorFixed,
        /// The loop-carried accumulator supplies scanned digits while the
        /// invariant multiplier occupies the generated function's `fixed`
        /// operand.
        AccumulatorScanned,
    }

    /// Frozen complete-field result for the generated named-register scalar
    /// FIOS implementation on an Apple M1 Max.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NamedRegisterFiosFieldGateResult {
        /// Generated arithmetic-schedule version.
        pub algorithm_version: u64,
        /// Mathematical operand and Montgomery-correction matrix terms.
        ///
        /// This is source-level schedule accounting, not a count of widened AIR
        /// multiply instructions.
        pub matrix_terms: u64,
        /// Compiled loop-carried operand orientations.
        pub variant_count: u64,
        /// Metal threads assigned to one logical field operation.
        pub threads_per_field: u64,
        /// Logical fields in the decision run.
        pub logical_fields: u64,
        /// Dependent multiplications per field and dispatch.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Timed samples after warmup.
        pub samples: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Selected loop-carried operand orientation.
        pub selected_orientation: NamedFiosOperandOrientation,
        /// Observed median complete field multiplications per second.
        pub median_complete_fp_per_second: u64,
        /// Observed nearest-rank fifth-percentile complete rate.
        pub p5_complete_fp_per_second: u64,
        /// Required median complete field rate.
        pub required_median_complete_fp_per_second: u64,
        /// Required fifth-percentile complete field rate.
        pub required_p5_complete_fp_per_second: u64,
        /// Fail-closed result.
        pub decision: CompleteFieldGateDecision,
    }

    impl NamedRegisterFiosFieldGateResult {
        /// Returns whether both complete-field throughput gates were met.
        #[must_use]
        pub const fn passes(self) -> bool {
            self.median_complete_fp_per_second >= self.required_median_complete_fp_per_second
                && self.p5_complete_fp_per_second >= self.required_p5_complete_fp_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-15 with Rust 1.95.0, BLST
    /// 0.3.16 differentials, and Apple Metal compiler 32023.883.
    pub const M1_MAX_NAMED_REGISTER_FIOS_FIELD_GATE_RESULT: NamedRegisterFiosFieldGateResult =
        NamedRegisterFiosFieldGateResult {
            algorithm_version: 1,
            matrix_terms: 288,
            variant_count: 2,
            threads_per_field: 1,
            logical_fields: 131_072,
            iterations: 32,
            dispatches: 4,
            samples: 20,
            threadgroup_width: 128,
            selected_orientation: NamedFiosOperandOrientation::AccumulatorFixed,
            median_complete_fp_per_second: 1_529_069_153,
            p5_complete_fp_per_second: 1_455_060_905,
            required_median_complete_fp_per_second: 2_000_000_000,
            required_p5_complete_fp_per_second: 1_800_000_000,
            decision: CompleteFieldGateDecision::MissesPredeclaredThroughputGate,
        };

    /// Frozen raw-product result for one 30-limb radix-`2^13`
    /// schoolbook-CIOS schedule on an Apple M1 Max.
    ///
    /// The reference schedule counts 900 bilinear operand terms and 900
    /// correction-by-modulus terms. Its 30 low-limb inverse steps are omitted
    /// because `n0' = -3 mod 8192` is strength-reducible. The diagnostic omits
    /// every carry, shift, comparison, and memory-routing cost. Dividing its
    /// observed issue rate by 1,800 gives a schedule equivalent, not completed
    /// field throughput or a device-wide ceiling.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct U32LowLimbProductIssueGateResult {
        /// Mathematical products in the tested conventional reference schedule.
        pub products_per_complete_multiply: u64,
        /// Number of timed samples after warmup.
        pub samples: u64,
        /// Independent shader lanes in each dispatch.
        pub elements: u64,
        /// Multiply-add loop iterations per lane.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Observed median native-`u32` product issue rate.
        pub median_products_per_second: u64,
        /// Observed nearest-rank fifth-percentile product issue rate.
        pub p5_products_per_second: u64,
        /// Median product rate divided by the candidate's 1,800-product count.
        pub median_product_only_complete_equivalents_per_second: u64,
        /// Fifth-percentile product rate divided by the candidate's product count.
        pub p5_product_only_complete_equivalents_per_second: u64,
        /// Required median product-only equivalent rate.
        pub required_median_equivalents_per_second: u64,
        /// Required fifth-percentile product-only equivalent rate.
        pub required_p5_equivalents_per_second: u64,
        /// Fail-closed outcome.
        pub decision: NativeGateDecision,
    }

    impl U32LowLimbProductIssueGateResult {
        /// Returns whether both favorable product-only throughput bounds were met.
        #[must_use]
        pub const fn passes(self) -> bool {
            self.median_product_only_complete_equivalents_per_second
                >= self.required_median_equivalents_per_second
                && self.p5_product_only_complete_equivalents_per_second
                    >= self.required_p5_equivalents_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-15 with Rust 1.95.0 and Apple
    /// Metal compiler 32023.883.
    pub const M1_MAX_U32_LOW_LIMB_PRODUCT_ISSUE_GATE_RESULT: U32LowLimbProductIssueGateResult =
        U32LowLimbProductIssueGateResult {
            products_per_complete_multiply: 1_800,
            samples: 20,
            elements: 262_144,
            iterations: 256,
            dispatches: 4,
            threadgroup_width: 128,
            median_products_per_second: 1_900_899_837_349,
            p5_products_per_second: 1_822_507_737_124,
            median_product_only_complete_equivalents_per_second: 1_056_055_465,
            p5_product_only_complete_equivalents_per_second: 1_012_504_298,
            required_median_equivalents_per_second: 2_000_000_000,
            required_p5_equivalents_per_second: 1_800_000_000,
            decision: NativeGateDecision::TestedScheduleMissesPredeclaredGate,
        };

    /// Frozen pre-performance result for the generated subtractive-Karatsuba
    /// and lazy-L4 field candidate on an Apple M1 Max.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct KaratsubaL4SourceLivenessGateResult {
        /// Generated arithmetic-schedule version.
        pub algorithm_version: u64,
        /// Base `u32` products in one twelve-limb Karatsuba product.
        pub base_products_per_k12: u64,
        /// Base `u32` products in multiplication plus separated REDC.
        pub base_products_per_redc: u64,
        /// Frozen generated-schedule fingerprint.
        pub schedule_fingerprint: u64,
        /// Cartesian Metal boundary pairs checked successfully.
        pub metal_cartesian_boundary_pairs: u64,
        /// Narrow outer-aggregate peak; this is not the acceptance metric.
        pub top_level_aggregate_subset_words: u64,
        /// Local-temporary peak inside one flattened K12 helper.
        pub full_flattened_helper_peak_words: u64,
        /// Product-helper peak after including its 24 input components.
        pub product_helper_including_inputs_peak_words: u64,
        /// Separated-REDC source-live lower bound with its numerator retained.
        pub separated_redc_peak_lower_bound_words: u64,
        /// Dependent-chain lower bound with the fixed right operand retained.
        pub dependent_chain_peak_lower_bound_words: u64,
        /// Maximum accepted source-live `u32`-equivalent words.
        pub maximum_source_live_words: u64,
        /// Whether the performance scout was run.
        pub performance_scout_run: bool,
        /// Whether the formal field-throughput gate was run.
        pub formal_performance_gate_run: bool,
        /// Whether physical register spill/fill evidence was available.
        pub physical_spill_evidence_available: bool,
        /// Whether this result authorizes point or MSM implementation work.
        pub point_or_msm_work_authorized: bool,
        /// Fail-closed outcome.
        pub decision: KaratsubaL4FieldGateDecision,
    }

    impl KaratsubaL4SourceLivenessGateResult {
        /// Returns whether the complete dependent-chain lower bound fits.
        #[must_use]
        pub const fn passes_source_liveness(self) -> bool {
            self.dependent_chain_peak_lower_bound_words <= self.maximum_source_live_words
        }
    }

    /// Apple M1 Max result frozen on 2026-07-15 with Rust 1.95.0 and Apple
    /// Metal compiler 32023.883.
    pub const M1_MAX_KARATSUBA_L4_SOURCE_LIVENESS_GATE_RESULT: KaratsubaL4SourceLivenessGateResult =
        KaratsubaL4SourceLivenessGateResult {
            algorithm_version: 2,
            base_products_per_k12: 63,
            base_products_per_redc: 189,
            schedule_fingerprint: 0x757d_b367_3700_e544,
            metal_cartesian_boundary_pairs: 144,
            top_level_aggregate_subset_words: 48,
            full_flattened_helper_peak_words: 70,
            product_helper_including_inputs_peak_words: 82,
            separated_redc_peak_lower_bound_words: 94,
            dependent_chain_peak_lower_bound_words: 106,
            maximum_source_live_words: 56,
            performance_scout_run: false,
            formal_performance_gate_run: false,
            physical_spill_evidence_available: false,
            point_or_msm_work_authorized: false,
            decision: KaratsubaL4FieldGateDecision::MissesSourceLivenessGateBeforePerformance,
        };

    /// Frozen complete-field result for the direct blocked-radix-128,
    /// subtractive-Karatsuba schedule on an Apple M1 Max.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockedRadix128KaratsubaL4FieldGateResult {
        /// Generated arithmetic-schedule version.
        pub algorithm_version: u64,
        /// Base `u32` products in one four-limb Karatsuba product.
        pub base_products_per_k4: u64,
        /// Base `u32` products in one complete blocked REDC.
        pub base_products_per_redc: u64,
        /// Frozen generated-schedule fingerprint.
        pub schedule_fingerprint: u64,
        /// Conventional typed and emitted-source live-word peak.
        pub conventional_source_peak_live_words: u64,
        /// Conservative peak that also reserves the current destination.
        pub conservative_source_peak_live_words: u64,
        /// Maximum accepted source-live `u32`-equivalent words.
        pub maximum_source_live_words: u64,
        /// Best median complete-REDC rate from the optimistic fixed-RHS scout.
        pub fixed_rhs_scout_complete_redcs_per_second: u64,
        /// Logical fields in the rotating-operand decision run.
        pub logical_fields: u64,
        /// Dependent REDCs per field and dispatch.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Timed decision samples after warmup.
        pub samples: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Observed median rotating-operand complete REDCs per second.
        pub median_complete_redcs_per_second: u64,
        /// Observed nearest-rank fifth-percentile rotating-operand rate.
        pub p5_complete_redcs_per_second: u64,
        /// Required median complete-REDC rate.
        pub required_median_complete_redcs_per_second: u64,
        /// Required nearest-rank fifth-percentile complete-REDC rate.
        pub required_p5_complete_redcs_per_second: u64,
        /// Whether the portable and native Metal correctness suites passed.
        pub correctness_passed: bool,
        /// Whether compiler-exposed physical spill metadata was available.
        pub physical_spill_evidence_available: bool,
        /// Whether this result authorizes point or MSM implementation work.
        pub point_or_msm_work_authorized: bool,
        /// Fail-closed result.
        pub decision: CompleteFieldGateDecision,
    }

    impl BlockedRadix128KaratsubaL4FieldGateResult {
        /// Returns whether the conservative generated-source liveness gate passed.
        #[must_use]
        pub const fn passes_source_liveness(self) -> bool {
            self.conservative_source_peak_live_words <= self.maximum_source_live_words
        }

        /// Returns whether both rotating-operand throughput gates passed.
        #[must_use]
        pub const fn passes_throughput(self) -> bool {
            self.median_complete_redcs_per_second >= self.required_median_complete_redcs_per_second
                && self.p5_complete_redcs_per_second >= self.required_p5_complete_redcs_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-16 with Rust 1.95.0, BLST
    /// 0.3.16 differentials, and Apple Metal compiler 32023.883.
    pub const M1_MAX_BLOCKED_RADIX128_KARATSUBA_L4_FIELD_GATE_RESULT:
        BlockedRadix128KaratsubaL4FieldGateResult = BlockedRadix128KaratsubaL4FieldGateResult {
        algorithm_version: 3,
        base_products_per_k4: 9,
        base_products_per_redc: 189,
        schedule_fingerprint: 0x0ead_48b6_6cf3_f532,
        conventional_source_peak_live_words: 54,
        conservative_source_peak_live_words: 55,
        maximum_source_live_words: 56,
        fixed_rhs_scout_complete_redcs_per_second: 341_347_223,
        logical_fields: 131_072,
        iterations: 32,
        dispatches: 4,
        samples: 20,
        threadgroup_width: 64,
        median_complete_redcs_per_second: 465_444_165,
        p5_complete_redcs_per_second: 455_317_995,
        required_median_complete_redcs_per_second: 2_000_000_000,
        required_p5_complete_redcs_per_second: 1_800_000_000,
        correctness_passed: true,
        physical_spill_evidence_available: false,
        point_or_msm_work_authorized: false,
        decision: CompleteFieldGateDecision::MissesPredeclaredThroughputGate,
    };

    /// Frozen complete-field result for the generated eight-lane AMNS8/Karatsuba
    /// schedule on an Apple M1 Max.
    ///
    /// The product counts are generated-source topology counts, not device
    /// instructions. Optimized AIR contains 28 variable widened-`i64` multiply
    /// sites and eight variable wrapping-`i32` multiply sites per Metal lane,
    /// matching the generated schedule after SIMD lane packing. The AIR also
    /// contains 494 SIMD-shuffle calls, no allocations, no private or threadgroup
    /// address-space references, and no out-of-line calls in the one-shot kernel.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Amns8KaratsubaFieldGateResult {
        /// Generated arithmetic-schedule version.
        pub algorithm_version: u64,
        /// Frozen generated-schedule fingerprint.
        pub schedule_fingerprint: u64,
        /// Mathematical widened and wrapping base products per field operation.
        pub logical_products_per_field: u64,
        /// Product slots executed after eight-lane padding.
        pub issued_products_per_field: u64,
        /// SIMD lanes assigned to one logical field operation.
        pub threads_per_field: u64,
        /// Conventional generated-source live-word peak.
        pub source_peak_live_words: u64,
        /// Conservative peak including the destination allocation.
        pub allocation_peak_live_words: u64,
        /// Maximum accepted source-live `u32`-equivalent words.
        pub maximum_source_live_words: u64,
        /// Logical fields in the decision run.
        pub logical_fields: u64,
        /// Dependent multiplications per field and dispatch.
        pub iterations: u64,
        /// Dispatches encoded into each timed command.
        pub dispatches: u64,
        /// Complete field operations in each timed sample.
        pub complete_operations_per_sample: u64,
        /// Timed decision samples after warmup.
        pub samples: u64,
        /// Selected threads per threadgroup.
        pub threadgroup_width: u64,
        /// Observed median complete field multiplications per second.
        pub median_complete_fp_per_second: u64,
        /// Observed nearest-rank fifth-percentile complete rate.
        pub p5_complete_fp_per_second: u64,
        /// Required median complete-field rate.
        pub required_median_complete_fp_per_second: u64,
        /// Required nearest-rank fifth-percentile complete-field rate.
        pub required_p5_complete_fp_per_second: u64,
        /// Whether the portable and native Metal correctness suites passed.
        pub correctness_passed: bool,
        /// Whether every checked padding field retained its canary.
        pub padding_untouched: bool,
        /// Static threadgroup memory reported by the pipeline.
        pub static_threadgroup_memory_bytes: u64,
        /// Whether this result authorizes point or MSM implementation work.
        pub point_or_msm_work_authorized: bool,
        /// Fail-closed result.
        pub decision: CompleteFieldGateDecision,
    }

    impl Amns8KaratsubaFieldGateResult {
        /// Returns whether the conservative generated-source liveness gate passed.
        #[must_use]
        pub const fn passes_source_liveness(self) -> bool {
            self.allocation_peak_live_words <= self.maximum_source_live_words
        }

        /// Returns whether both complete-field throughput gates passed.
        #[must_use]
        pub const fn passes_throughput(self) -> bool {
            self.median_complete_fp_per_second >= self.required_median_complete_fp_per_second
                && self.p5_complete_fp_per_second >= self.required_p5_complete_fp_per_second
        }
    }

    /// Apple M1 Max result measured on 2026-07-16 with Rust 1.95.0 and Apple
    /// Metal compiler 32023.883.
    pub const M1_MAX_AMNS8_KARATSUBA_FIELD_GATE_RESULT: Amns8KaratsubaFieldGateResult =
        Amns8KaratsubaFieldGateResult {
            algorithm_version: 1,
            schedule_fingerprint: 0xf8a2_079c_fdcc_25d5,
            logical_products_per_field: 243,
            issued_products_per_field: 288,
            threads_per_field: 8,
            source_peak_live_words: 32,
            allocation_peak_live_words: 33,
            maximum_source_live_words: 56,
            logical_fields: 131_072,
            iterations: 32,
            dispatches: 4,
            complete_operations_per_sample: 16_777_216,
            samples: 20,
            threadgroup_width: 32,
            median_complete_fp_per_second: 6_798_372,
            p5_complete_fp_per_second: 6_721_714,
            required_median_complete_fp_per_second: 2_000_000_000,
            required_p5_complete_fp_per_second: 1_800_000_000,
            correctness_passed: true,
            padding_untouched: true,
            static_threadgroup_memory_bytes: 0,
            point_or_msm_work_authorized: false,
            decision: CompleteFieldGateDecision::MissesPredeclaredThroughputGate,
        };

    /// Frozen prioritization result for replacing AMNS8 widened products with
    /// explicit low-`u32` multiplication plus `metal::mulhi`.
    ///
    /// This is deliberately not a complete-field gate or a hardware roofline.
    /// The fixed-RHS and rotating-two-state probes isolate one unsigned
    /// 32-by-32-to-64 product, retain both halves in observable dependent state,
    /// and validate against CPU oracles. They show no material explicit-`mulhi`
    /// advantage. The generated v2 field source remains available for future
    /// research, but the current plan does not spend a full field-gate run on a
    /// substitution that preserves all 494 shuffles while expanding its
    /// multiply-like instruction mix.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Amns8MulhiPrimitiveDisposition {
        /// Generated explicit-`mulhi` schedule version.
        pub algorithm_version: u64,
        /// Frozen generated v2 callable fingerprint.
        pub schedule_fingerprint: u64,
        /// Logical products in the unchanged K8 graph.
        pub logical_products_per_field: u64,
        /// Lane-packed conceptual product slots in the unchanged topology.
        pub issued_products_per_field: u64,
        /// Widened conceptual product slots per field.
        pub issued_widened_products_per_field: u64,
        /// Wrapping conceptual product slots per field.
        pub issued_wrapping_products_per_field: u64,
        /// Source last-use peak in `u32`-equivalent words.
        pub source_peak_live_words: u64,
        /// Conservative source allocation-time peak.
        pub allocation_peak_live_words: u64,
        /// SIMD-shuffle calls in optimized one-shot AIR.
        pub optimized_air_simd_shuffle_calls: u64,
        /// Explicit high-half calls in optimized one-shot AIR.
        pub optimized_air_mulhi_calls: u64,
        /// Variable `i32` multiply sites in optimized one-shot AIR.
        pub optimized_air_variable_i32_multiply_sites: u64,
        /// Variable `i64` leaf multiply sites in optimized one-shot AIR.
        pub optimized_air_variable_i64_multiply_sites: u64,
        /// Direct-topology widened-product rate implied by the 2.0B field gate.
        pub required_widened_products_per_second_before_other_work: u64,
        /// Products in each fixed-RHS timed sample.
        pub fixed_rhs_products_per_sample: u64,
        /// Alternating timed pairs in the fixed-RHS population.
        pub fixed_rhs_sample_pairs: u64,
        /// Fixed-RHS median explicit low-plus-`mulhi` product rate.
        pub fixed_rhs_mulhi_products_per_second: u64,
        /// Fixed-RHS median widened-expression product rate.
        pub fixed_rhs_widened_products_per_second: u64,
        /// Products in each rotating-two-state timed sample.
        pub rotating_products_per_sample: u64,
        /// Alternating timed pairs in the rotating population.
        pub rotating_sample_pairs: u64,
        /// Rotating median explicit low-plus-`mulhi` product rate.
        pub rotating_mulhi_products_per_second: u64,
        /// Rotating median widened-expression product rate.
        pub rotating_widened_products_per_second: u64,
        /// Whether a complete AMNS8-v2 field gate ran.
        pub complete_field_gate_run: bool,
        /// Whether this result authorizes point or MSM implementation work.
        pub point_or_msm_work_authorized: bool,
        /// Scoped prioritization outcome.
        pub decision: PrimitiveSubstitutionDecision,
    }

    impl Amns8MulhiPrimitiveDisposition {
        /// Returns whether explicit `mulhi` won both representative median-rate
        /// comparisons.
        #[must_use]
        pub const fn explicit_mulhi_wins_both_scouts(self) -> bool {
            self.fixed_rhs_mulhi_products_per_second > self.fixed_rhs_widened_products_per_second
                && self.rotating_mulhi_products_per_second
                    > self.rotating_widened_products_per_second
        }
    }

    /// Apple M1 Max prioritization result measured on 2026-07-16 with Apple
    /// Metal compiler 32023.883.
    pub const M1_MAX_AMNS8_MULHI_PRIMITIVE_DISPOSITION: Amns8MulhiPrimitiveDisposition =
        Amns8MulhiPrimitiveDisposition {
            algorithm_version: 2,
            schedule_fingerprint: 0x4cdd_fae7_3658_b39f,
            logical_products_per_field: 243,
            issued_products_per_field: 288,
            issued_widened_products_per_field: 224,
            issued_wrapping_products_per_field: 64,
            source_peak_live_words: 32,
            allocation_peak_live_words: 33,
            optimized_air_simd_shuffle_calls: 494,
            optimized_air_mulhi_calls: 28,
            optimized_air_variable_i32_multiply_sites: 36,
            optimized_air_variable_i64_multiply_sites: 0,
            required_widened_products_per_second_before_other_work: 448_000_000_000,
            fixed_rhs_products_per_sample: 8_589_934_592,
            fixed_rhs_sample_pairs: 20,
            fixed_rhs_mulhi_products_per_second: 292_740_437_434,
            fixed_rhs_widened_products_per_second: 342_843_712_446,
            rotating_products_per_sample: 34_359_738_368,
            rotating_sample_pairs: 20,
            rotating_mulhi_products_per_second: 213_760_985_655,
            rotating_widened_products_per_second: 226_032_774_367,
            complete_field_gate_run: false,
            point_or_msm_work_authorized: false,
            decision: PrimitiveSubstitutionDecision::PrunedByPrioritizationEvidence,
        };

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn exactness_bounds_fit_strictly_below_fp32_integer_ceiling() {
            let assessment = M1_MAX_PACKED_FP32_MONTGOMERY_ASSESSMENT;
            assert_eq!(coefficient_bound(6, 8), 6_291_456);
            assert_eq!(DEEPEST_RECOMBINATION_BOUND, 12_582_912);
            assert_eq!(MIDDLE_RECOMBINATION_BOUND, 6_291_456);
            assert_eq!(TOP_RECOMBINATION_BOUND, 3_145_728);
            assert_eq!(MAXIMUM_RECOMBINATION_BOUND, 12_582_912);
            assert_eq!(FINAL_CONVOLUTION_COEFFICIENT_BOUND, 786_432);
            assert_eq!(MAXIMUM_CARRY_UNPACK_VALUE, 789_517);
            assert!(assessment.maximum_leaf_accumulator < FP32_EXACT_INTEGER_CEILING);
            assert!(assessment.maximum_recombination_intermediate < FP32_EXACT_INTEGER_CEILING);
            assert_eq!(assessment.exact_integer_headroom(), 4_194_304);
            assert_eq!(assessment.maximum_carry_unpack_value, 789_517);
        }

        #[test]
        fn complete_montgomery_operation_count_is_frozen() {
            let assessment = M1_MAX_PACKED_FP32_MONTGOMERY_ASSESSMENT;
            assert_eq!(assessment.leaf_convolutions, 27);
            assert_eq!(assessment.scalar_products_per_convolution, 972);
            assert_eq!(assessment.convolutions_per_complete_multiply, 3);
            assert_eq!(assessment.scalar_products_per_complete_multiply, 2_916);
        }

        #[test]
        fn roofline_rejects_candidate_before_shader_work() {
            let assessment = M1_MAX_PACKED_FP32_MONTGOMERY_ASSESSMENT;
            assert_eq!(assessment.required_scalar_products_per_second, 5_832_000_000_000);
            assert_eq!(assessment.peak_scalar_products_per_second, 5_200_000_000_000);
            assert_eq!(assessment.roofline_complete_multiplications_per_second, 1_783_264_746);
            assert_eq!(assessment.required_product_issue_basis_points, 11_216);
            assert!(assessment.misses_declared_target());
            assert_eq!(assessment.decision, CandidateDecision::MissesDeclaredTargetOnM1Max);
        }

        #[test]
        fn carry_normalized_recurrence_includes_sum_carry_digits() {
            let assessment = M1_MAX_CARRY_NORMALIZED_FP32_ASSESSMENT;
            assert_eq!(assessment.digits, 32);
            assert_eq!(assessment.karatsuba_levels, 4);
            assert_eq!(assessment.leaf_convolutions, 81);
            assert_eq!(assessment.maximum_leaf_width, 4);
            assert_eq!(assessment.scalar_products_per_convolution, 647);
            assert_eq!(36 * 2 * 2 + 31 * 3 * 3 + 14 * 4 * 4, 647);
        }

        #[test]
        fn carry_normalized_leaves_fit_the_inclusive_fp32_integer_limit() {
            let assessment = M1_MAX_CARRY_NORMALIZED_FP32_ASSESSMENT;
            assert_eq!(assessment.maximum_leaf_accumulator, FP32_EXACT_INTEGER_CEILING);
            assert_eq!(assessment.radix, 4_096);
        }

        #[test]
        fn carry_normalized_candidate_clears_only_the_product_count_gate() {
            let assessment = M1_MAX_CARRY_NORMALIZED_FP32_ASSESSMENT;
            assert_eq!(assessment.convolutions_per_complete_multiply, 3);
            assert_eq!(assessment.scalar_products_per_complete_multiply, 1_941);
            assert_eq!(assessment.required_scalar_products_per_second, 3_882_000_000_000);
            assert_eq!(assessment.roofline_complete_multiplications_per_second, 2_679_031_427);
            assert_eq!(assessment.required_product_issue_basis_points, 7_466);
            assert_eq!(assessment.product_issue_headroom_basis_points(), 2_534);
            assert!(assessment.clears_product_roofline());
            assert_eq!(assessment.decision, CandidateDecision::ClearsProductRooflineOnly);
        }

        #[test]
        fn native_precise_fp32_gate_rejects_tested_schedule() {
            let result = M1_MAX_PRECISE_FP32_PRODUCT_ISSUE_GATE_RESULT;
            assert_eq!(result.samples, 20);
            assert_eq!(result.threadgroup_width, 64);
            assert_eq!(result.median_scalar_products_per_second, 1_945_454_089_254);
            assert_eq!(result.p5_scalar_products_per_second, 1_827_159_711_141);
            assert_eq!(result.median_complete_product_equivalents_per_second, 1_002_294_739);
            assert_eq!(result.p5_complete_product_equivalents_per_second, 941_349_671);
            assert!(!result.passes());
            assert_eq!(result.decision, NativeGateDecision::TestedScheduleMissesPredeclaredGate);
        }

        #[test]
        fn complete_two_lane_u32_field_gate_is_frozen() {
            let result = M1_MAX_TWO_LANE_U32_FIPS_FIELD_GATE_RESULT;
            assert_eq!(result.algorithm_version, 2);
            assert_eq!(result.generated_product_slots, 144);
            assert_eq!(result.threads_per_field, 2);
            assert_eq!(result.median_complete_fp_per_second, 986_960_401);
            assert_eq!(result.p5_complete_fp_per_second, 939_895_686);
            assert!(!result.passes());
            assert_eq!(result.decision, CompleteFieldGateDecision::MissesPredeclaredThroughputGate);
        }

        #[test]
        fn complete_named_register_fios_field_gate_is_frozen() {
            let result = M1_MAX_NAMED_REGISTER_FIOS_FIELD_GATE_RESULT;
            assert_eq!(result.algorithm_version, 1);
            assert_eq!(result.matrix_terms, 288);
            assert_eq!(result.variant_count, 2);
            assert_eq!(result.threads_per_field, 1);
            assert_eq!(result.threadgroup_width, 128);
            assert_eq!(result.selected_orientation, NamedFiosOperandOrientation::AccumulatorFixed);
            assert_eq!(result.median_complete_fp_per_second, 1_529_069_153);
            assert_eq!(result.p5_complete_fp_per_second, 1_455_060_905);
            assert!(!result.passes());
            assert_eq!(result.decision, CompleteFieldGateDecision::MissesPredeclaredThroughputGate);
        }

        #[test]
        fn native_u32_low_limb_product_gate_is_frozen() {
            let result = M1_MAX_U32_LOW_LIMB_PRODUCT_ISSUE_GATE_RESULT;
            assert_eq!(result.products_per_complete_multiply, 1_800);
            assert_eq!(result.threadgroup_width, 128);
            assert_eq!(result.median_products_per_second, 1_900_899_837_349);
            assert_eq!(result.p5_products_per_second, 1_822_507_737_124);
            assert_eq!(result.median_product_only_complete_equivalents_per_second, 1_056_055_465);
            assert_eq!(result.p5_product_only_complete_equivalents_per_second, 1_012_504_298);
            assert!(!result.passes());
            assert_eq!(result.decision, NativeGateDecision::TestedScheduleMissesPredeclaredGate);
        }

        #[test]
        fn karatsuba_l4_source_liveness_gate_is_frozen() {
            let result = M1_MAX_KARATSUBA_L4_SOURCE_LIVENESS_GATE_RESULT;
            assert_eq!(result.algorithm_version, 2);
            assert_eq!(result.base_products_per_k12, 63);
            assert_eq!(result.base_products_per_redc, 189);
            assert_eq!(result.schedule_fingerprint, 0x757d_b367_3700_e544);
            assert_eq!(result.metal_cartesian_boundary_pairs, 144);
            assert_eq!(result.top_level_aggregate_subset_words, 48);
            assert_eq!(result.full_flattened_helper_peak_words, 70);
            assert_eq!(result.product_helper_including_inputs_peak_words, 82);
            assert_eq!(result.separated_redc_peak_lower_bound_words, 94);
            assert_eq!(result.dependent_chain_peak_lower_bound_words, 106);
            assert_eq!(result.maximum_source_live_words, 56);
            assert!(!result.passes_source_liveness());
            assert!(!result.performance_scout_run);
            assert!(!result.formal_performance_gate_run);
            assert!(!result.physical_spill_evidence_available);
            assert!(!result.point_or_msm_work_authorized);
            assert_eq!(
                result.decision,
                KaratsubaL4FieldGateDecision::MissesSourceLivenessGateBeforePerformance
            );
        }

        #[test]
        fn blocked_radix128_field_gate_passes_prerequisites_but_misses_throughput() {
            let result = M1_MAX_BLOCKED_RADIX128_KARATSUBA_L4_FIELD_GATE_RESULT;
            assert_eq!(result.algorithm_version, 3);
            assert_eq!(result.base_products_per_k4, 9);
            assert_eq!(result.base_products_per_redc, 189);
            assert_eq!(result.schedule_fingerprint, 0x0ead_48b6_6cf3_f532);
            assert_eq!(result.conventional_source_peak_live_words, 54);
            assert_eq!(result.conservative_source_peak_live_words, 55);
            assert!(result.passes_source_liveness());
            assert!(result.correctness_passed);
            assert_eq!(result.samples, 20);
            assert_eq!(result.threadgroup_width, 64);
            assert_eq!(result.median_complete_redcs_per_second, 465_444_165);
            assert_eq!(result.p5_complete_redcs_per_second, 455_317_995);
            assert!(!result.passes_throughput());
            assert!(!result.physical_spill_evidence_available);
            assert!(!result.point_or_msm_work_authorized);
            assert_eq!(result.decision, CompleteFieldGateDecision::MissesPredeclaredThroughputGate);
        }

        #[test]
        fn amns8_field_gate_passes_prerequisites_but_misses_throughput() {
            let result = M1_MAX_AMNS8_KARATSUBA_FIELD_GATE_RESULT;
            assert_eq!(result.algorithm_version, 1);
            assert_eq!(result.schedule_fingerprint, 0xf8a2_079c_fdcc_25d5);
            assert_eq!(result.logical_products_per_field, 243);
            assert_eq!(result.issued_products_per_field, 288);
            assert_eq!(result.threads_per_field, 8);
            assert_eq!(result.source_peak_live_words, 32);
            assert_eq!(result.allocation_peak_live_words, 33);
            assert!(result.passes_source_liveness());
            assert!(result.correctness_passed);
            assert!(result.padding_untouched);
            assert_eq!(result.complete_operations_per_sample, 16_777_216);
            assert_eq!(result.samples, 20);
            assert_eq!(result.threadgroup_width, 32);
            assert_eq!(result.median_complete_fp_per_second, 6_798_372);
            assert_eq!(result.p5_complete_fp_per_second, 6_721_714);
            assert!(!result.passes_throughput());
            assert!(!result.point_or_msm_work_authorized);
            assert_eq!(result.decision, CompleteFieldGateDecision::MissesPredeclaredThroughputGate);
        }

        #[test]
        fn amns8_mulhi_is_pruned_as_a_primitive_substitution_not_a_field_result() {
            let result = M1_MAX_AMNS8_MULHI_PRIMITIVE_DISPOSITION;
            assert_eq!(result.algorithm_version, 2);
            assert_eq!(result.schedule_fingerprint, 0x4cdd_fae7_3658_b39f);
            assert_eq!(result.logical_products_per_field, 243);
            assert_eq!(result.issued_products_per_field, 288);
            assert_eq!(result.optimized_air_simd_shuffle_calls, 494);
            assert_eq!(result.optimized_air_mulhi_calls, 28);
            assert_eq!(result.optimized_air_variable_i32_multiply_sites, 36);
            assert_eq!(result.optimized_air_variable_i64_multiply_sites, 0);
            assert_eq!(result.fixed_rhs_sample_pairs, 20);
            assert_eq!(result.rotating_sample_pairs, 20);
            assert!(!result.explicit_mulhi_wins_both_scouts());
            assert!(!result.complete_field_gate_run);
            assert!(!result.point_or_msm_work_authorized);
            assert_eq!(
                result.decision,
                PrimitiveSubstitutionDecision::PrunedByPrioritizationEvidence
            );
        }
    }
}
