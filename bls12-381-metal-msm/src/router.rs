// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Deterministic CPU recoding and routing for the first CPU+Metal MSM path.
//!
//! This module performs no curve arithmetic. It turns canonical scalars into
//! compact signed base references grouped by dense `(output, window, bucket)`
//! indices; Metal owns bucket reduction, integration, and collection.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use zeroize::{Zeroize, Zeroizing};

use crate::msm_api::{CanonicalScalar, MsmJobV1, WindowBits};

const NEGATIVE_BIT: u32 = 1 << 31;
const POINT_INDEX_MASK: u32 = NEGATIVE_BIT - 1;
const MAX_PREPARED_BASES: usize = 1 << 31;

/// One signed prepared-base reference in routing ABI version 1.
///
/// Bit 31 is one for a negative Booth digit; bits 0 through 30 hold the
/// absolute prepared-base index.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct RoutedPointV1(u32);

impl RoutedPointV1 {
    /// Packs a prepared-base index and sign.
    pub fn new(point_index: u32, negative: bool) -> Result<Self, RouteError> {
        if point_index > POINT_INDEX_MASK {
            return Err(RouteError::PointIndexOutOfRange { point_index });
        }
        Ok(Self(point_index | if negative { NEGATIVE_BIT } else { 0 }))
    }

    /// Returns the packed ABI word.
    #[must_use]
    pub const fn packed(self) -> u32 {
        self.0
    }

    /// Returns the absolute prepared-base index.
    #[must_use]
    pub const fn point_index(self) -> u32 {
        self.0 & POINT_INDEX_MASK
    }

    /// Returns whether the routed point must be negated.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 & NEGATIVE_BIT != 0
    }
}

impl fmt::Debug for RoutedPointV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RoutedPointV1(REDACTED)")
    }
}

impl Zeroize for RoutedPointV1 {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

/// One non-empty, contiguous bucket run in routing ABI version 1.
#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct BucketRunV1 {
    /// First record in [`RoutePlan::records`].
    pub record_offset: u32,
    /// Number of records reduced into this bucket.
    pub record_count: u32,
    /// Dense `(output, window, magnitude - 1)` bucket index.
    pub dense_bucket_index: u32,
    /// Must be zero in ABI version 1.
    pub reserved: u32,
}

impl fmt::Debug for BucketRunV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BucketRunV1(REDACTED)")
    }
}

impl Zeroize for BucketRunV1 {
    fn zeroize(&mut self) {
        self.record_offset.zeroize();
        self.record_count.zeroize();
        self.dense_bucket_index.zeroize();
        self.reserved.zeroize();
    }
}

impl Zeroize for MsmJobV1 {
    fn zeroize(&mut self) {
        self.base_offset.zeroize();
        self.scalar_offset.zeroize();
        self.term_count.zeroize();
        self.output_index.zeroize();
    }
}

/// A validated, deterministic routing plan that owns all submission arrays.
///
/// Records within each run preserve input term order. Runs are ordered by
/// increasing dense bucket index, making routing reproducible across machines.
pub struct RoutePlan {
    window_bits: WindowBits,
    prepared_base_count: u32,
    scalar_count: u32,
    output_count: u32,
    dense_bucket_count: u32,
    jobs: Vec<MsmJobV1>,
    records: Vec<RoutedPointV1>,
    runs: Vec<BucketRunV1>,
}

impl RoutePlan {
    /// Returns the selected Booth window width.
    #[must_use]
    pub const fn window_bits(&self) -> WindowBits {
        self.window_bits
    }

    /// Returns the number of windows routed for every scalar.
    #[must_use]
    pub const fn windows_per_scalar(&self) -> u32 {
        self.window_bits.windows_per_scalar()
    }

    /// Returns the number of magnitude buckets in every window.
    #[must_use]
    pub const fn buckets_per_window(&self) -> u32 {
        self.window_bits.buckets_per_window()
    }

    /// Returns the validated prepared-base count.
    #[must_use]
    pub const fn prepared_base_count(&self) -> u32 {
        self.prepared_base_count
    }

    /// Returns the canonical scalar-array length.
    #[must_use]
    pub const fn scalar_count(&self) -> u32 {
        self.scalar_count
    }

    /// Returns the number of dense output slots.
    #[must_use]
    pub const fn output_count(&self) -> u32 {
        self.output_count
    }

    /// Returns the total dense-bucket address space.
    #[must_use]
    pub const fn dense_bucket_count(&self) -> u32 {
        self.dense_bucket_count
    }

    /// Returns the validated ragged jobs owned by this plan.
    #[must_use]
    pub fn jobs(&self) -> &[MsmJobV1] {
        &self.jobs
    }

    /// Returns compact signed point references grouped by run.
    #[must_use]
    pub fn records(&self) -> &[RoutedPointV1] {
        &self.records
    }

    /// Returns non-empty bucket runs ordered by dense bucket index.
    #[must_use]
    pub fn runs(&self) -> &[BucketRunV1] {
        &self.runs
    }

    /// Explicitly clears all scalar-derived routing state.
    ///
    /// `Drop` calls the same operation. The zeroization implementation uses
    /// compiler fences and volatile writes supplied by the audited `zeroize`
    /// crate, including the vectors' spare capacity.
    pub fn erase(&mut self) {
        self.wipe_sensitive();
    }

    /// Clears records and scalar-derived occupancy metadata before reuse/drop.
    pub(crate) fn wipe_sensitive(&mut self) {
        self.records.zeroize();
        self.runs.zeroize();
        self.jobs.zeroize();
        self.prepared_base_count.zeroize();
        self.scalar_count.zeroize();
        self.output_count.zeroize();
        self.dense_bucket_count.zeroize();
    }
}

impl fmt::Debug for RoutePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RoutePlan")
            .field("window_bits", &self.window_bits)
            .field("prepared_base_count", &self.prepared_base_count)
            .field("output_count", &self.output_count)
            .field("job_count", &self.jobs.len())
            .field("routing_state", &"REDACTED")
            .finish()
    }
}

impl Drop for RoutePlan {
    fn drop(&mut self) {
        self.wipe_sensitive();
    }
}

/// A fail-closed routing validation or capacity error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteError {
    /// Prepared-base indices cannot share bit 31 with the sign flag.
    TooManyPreparedBases {
        /// Requested prepared-base count.
        count: usize,
    },
    /// The scalar array cannot be represented by the version-1 ABI.
    TooManyScalars {
        /// Requested canonical-scalar count.
        count: usize,
    },
    /// The job array cannot be represented by the version-1 ABI.
    TooManyJobs {
        /// Requested MSM job count.
        count: usize,
    },
    /// A job output is outside the dense output array.
    OutputIndexOutOfBounds {
        /// Index of the malformed job.
        job_index: usize,
        /// Out-of-range output destination.
        output_index: u32,
    },
    /// Two jobs target the same dense output slot.
    DuplicateOutputIndex {
        /// Output destination used more than once.
        output_index: u32,
    },
    /// A job's prepared-base range is invalid.
    BaseRangeOutOfBounds {
        /// Index of the malformed job.
        job_index: usize,
    },
    /// A job's scalar range is invalid.
    ScalarRangeOutOfBounds {
        /// Index of the malformed job.
        job_index: usize,
    },
    /// A packed point index would overlap the sign bit.
    PointIndexOutOfRange {
        /// Point index that overlaps the packed sign bit.
        point_index: u32,
    },
    /// A version-1 offset/count calculation exceeded `u32`.
    AbiOverflow,
    /// Memory for a validated route could not be reserved.
    AllocationFailed,
}

impl fmt::Display for RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyPreparedBases { count } => {
                write!(formatter, "prepared-base count {count} must be below 2^31")
            }
            Self::TooManyScalars { count } => {
                write!(formatter, "scalar count {count} exceeds the v1 ABI")
            }
            Self::TooManyJobs { count } => {
                write!(formatter, "job count {count} exceeds the v1 ABI")
            }
            Self::OutputIndexOutOfBounds { job_index, output_index } => {
                write!(formatter, "job {job_index} output index {output_index} is not dense")
            }
            Self::DuplicateOutputIndex { output_index } => {
                write!(formatter, "output index {output_index} is used by multiple jobs")
            }
            Self::BaseRangeOutOfBounds { job_index } => {
                write!(formatter, "job {job_index} prepared-base range is out of bounds")
            }
            Self::ScalarRangeOutOfBounds { job_index } => {
                write!(formatter, "job {job_index} scalar range is out of bounds")
            }
            Self::PointIndexOutOfRange { point_index } => {
                write!(formatter, "prepared-base index {point_index} overlaps the sign bit")
            }
            Self::AbiOverflow => {
                formatter.write_str("routing plan exceeds a version-1 ABI offset or count")
            }
            Self::AllocationFailed => formatter.write_str("routing-plan allocation failed"),
        }
    }
}

/// Routes a batch of ragged MSM jobs over shared bases and scalar storage.
pub fn route_jobs(
    prepared_base_count: usize,
    scalars: &[CanonicalScalar],
    jobs: &[MsmJobV1],
    window_bits: WindowBits,
) -> Result<RoutePlan, RouteError> {
    if prepared_base_count >= MAX_PREPARED_BASES {
        return Err(RouteError::TooManyPreparedBases { count: prepared_base_count });
    }
    let scalar_count = u32::try_from(scalars.len())
        .map_err(|_| RouteError::TooManyScalars { count: scalars.len() })?;
    let output_count =
        u32::try_from(jobs.len()).map_err(|_| RouteError::TooManyJobs { count: jobs.len() })?;
    let prepared_base_count_u32 =
        u32::try_from(prepared_base_count).map_err(|_| RouteError::AbiOverflow)?;

    validate_jobs(prepared_base_count, scalars.len(), jobs)?;

    let windows = window_bits.windows_per_scalar();
    let buckets = window_bits.buckets_per_window();
    let dense_bucket_count = output_count
        .checked_mul(windows)
        .and_then(|value| value.checked_mul(buckets))
        .ok_or(RouteError::AbiOverflow)?;

    let mut counts = Zeroizing::new(try_zeroed_vec::<u32>(dense_bucket_count as usize)?);
    let mut record_count = 0_u32;
    for job in jobs {
        for term in 0..job.term_count {
            let scalar_index =
                job.scalar_offset.checked_add(term).ok_or(RouteError::AbiOverflow)?;
            let scalar = scalars[scalar_index as usize];
            for window_index in 0..windows {
                let digit = booth_digit(scalar, window_index, window_bits);
                if digit == 0 {
                    continue;
                }
                let dense =
                    dense_bucket_index(job.output_index, window_index, digit, windows, buckets)?;
                counts[dense as usize] =
                    counts[dense as usize].checked_add(1).ok_or(RouteError::AbiOverflow)?;
                record_count = record_count.checked_add(1).ok_or(RouteError::AbiOverflow)?;
            }
        }
    }

    let mut offsets = Zeroizing::new(try_zeroed_vec::<u32>(counts.len())?);
    let mut next_offset = 0_u32;
    for (offset, count) in offsets.iter_mut().zip(counts.iter()) {
        *offset = next_offset;
        next_offset = next_offset.checked_add(*count).ok_or(RouteError::AbiOverflow)?;
    }
    debug_assert_eq!(next_offset, record_count);

    let mut records = Zeroizing::new(try_zeroed_vec::<RoutedPointV1>(record_count as usize)?);
    let mut cursors = Zeroizing::new(try_copy_vec(offsets.as_slice())?);
    for job in jobs {
        for term in 0..job.term_count {
            let scalar_index =
                job.scalar_offset.checked_add(term).ok_or(RouteError::AbiOverflow)?;
            let point_index = job.base_offset.checked_add(term).ok_or(RouteError::AbiOverflow)?;
            let scalar = scalars[scalar_index as usize];
            for window_index in 0..windows {
                let digit = booth_digit(scalar, window_index, window_bits);
                if digit == 0 {
                    continue;
                }
                let dense =
                    dense_bucket_index(job.output_index, window_index, digit, windows, buckets)?;
                let cursor = &mut cursors[dense as usize];
                records[*cursor as usize] = RoutedPointV1::new(point_index, digit < 0)?;
                *cursor = cursor.checked_add(1).ok_or(RouteError::AbiOverflow)?;
            }
        }
    }

    let non_empty = counts.iter().filter(|count| **count != 0).count();
    let mut runs = Zeroizing::new(Vec::new());
    runs.try_reserve_exact(non_empty).map_err(|_| RouteError::AllocationFailed)?;
    for (dense, (&record_offset, &record_count)) in offsets.iter().zip(counts.iter()).enumerate() {
        if record_count != 0 {
            runs.push(BucketRunV1 {
                record_offset,
                record_count,
                dense_bucket_index: u32::try_from(dense).map_err(|_| RouteError::AbiOverflow)?,
                reserved: 0,
            });
        }
    }

    let mut owned_jobs = Zeroizing::new(Vec::new());
    owned_jobs.try_reserve_exact(jobs.len()).map_err(|_| RouteError::AllocationFailed)?;
    owned_jobs.extend_from_slice(jobs);

    Ok(RoutePlan {
        window_bits,
        prepared_base_count: prepared_base_count_u32,
        scalar_count,
        output_count,
        dense_bucket_count,
        jobs: core::mem::take(&mut *owned_jobs),
        records: core::mem::take(&mut *records),
        runs: core::mem::take(&mut *runs),
    })
}

fn validate_jobs(
    prepared_base_count: usize,
    scalar_count: usize,
    jobs: &[MsmJobV1],
) -> Result<(), RouteError> {
    let mut outputs_seen = Zeroizing::new(try_zeroed_vec::<bool>(jobs.len())?);
    for (job_index, job) in jobs.iter().enumerate() {
        let output_index = job.output_index as usize;
        if output_index >= jobs.len() {
            return Err(RouteError::OutputIndexOutOfBounds {
                job_index,
                output_index: job.output_index,
            });
        }
        if outputs_seen[output_index] {
            return Err(RouteError::DuplicateOutputIndex { output_index: job.output_index });
        }
        outputs_seen[output_index] = true;

        let base_end = (job.base_offset as usize)
            .checked_add(job.term_count as usize)
            .ok_or(RouteError::BaseRangeOutOfBounds { job_index })?;
        if base_end > prepared_base_count {
            return Err(RouteError::BaseRangeOutOfBounds { job_index });
        }
        let scalar_end = (job.scalar_offset as usize)
            .checked_add(job.term_count as usize)
            .ok_or(RouteError::ScalarRangeOutOfBounds { job_index })?;
        if scalar_end > scalar_count {
            return Err(RouteError::ScalarRangeOutOfBounds { job_index });
        }
    }
    Ok(())
}

fn try_zeroed_vec<T: Default + Clone>(len: usize) -> Result<Vec<T>, RouteError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| RouteError::AllocationFailed)?;
    values.resize(len, T::default());
    Ok(values)
}

fn try_copy_vec<T: Copy>(values: &[T]) -> Result<Vec<T>, RouteError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len()).map_err(|_| RouteError::AllocationFailed)?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn dense_bucket_index(
    output_index: u32,
    window_index: u32,
    digit: i32,
    windows: u32,
    buckets: u32,
) -> Result<u32, RouteError> {
    debug_assert_ne!(digit, 0);
    let magnitude = digit.unsigned_abs();
    debug_assert!(magnitude <= buckets);
    output_index
        .checked_mul(windows)
        .and_then(|value| value.checked_add(window_index))
        .and_then(|value| value.checked_mul(buckets))
        .and_then(|value| value.checked_add(magnitude - 1))
        .ok_or(RouteError::AbiOverflow)
}

/// Exact overlapping-bit Booth encoding used by Midnight's CPU MSM.
fn booth_digit(scalar: CanonicalScalar, window_index: u32, window_bits: WindowBits) -> i32 {
    let window_size = window_bits.get();
    let element = scalar.to_le_bytes();
    let skip_bits = (window_index * window_size).saturating_sub(1);
    let skip_bytes = skip_bits / 8;

    let mut bytes = [0_u8; 4];
    for (destination, source) in bytes.iter_mut().zip(element.iter().skip(skip_bytes as usize)) {
        *destination = *source;
    }
    let mut value = u32::from_le_bytes(bytes);
    if window_index == 0 {
        value <<= 1;
    }
    value >>= skip_bits - skip_bytes * 8;
    value &= (1 << (window_size + 1)) - 1;

    let positive = value & (1 << window_size) == 0;
    value = (value + 1) >> 1;
    if positive { value as i32 } else { -((!(value - 1) & ((1 << window_size) - 1)) as i32) }
}

const _: [(); 4] = [(); core::mem::size_of::<RoutedPointV1>()];
const _: [(); 16] = [(); core::mem::size_of::<BucketRunV1>()];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msm_api::BLS12_381_SCALAR_MODULUS;

    fn scalar(limbs: [u64; 4]) -> CanonicalScalar {
        CanonicalScalar::from_le_limbs(limbs).unwrap()
    }

    #[test]
    fn exact_window_counts_and_scalar_recomposition() {
        let mut modulus_minus_one = BLS12_381_SCALAR_MODULUS;
        modulus_minus_one[0] -= 1;
        let corpus = [
            [0; 4],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [u64::MAX, 0, 0, 0],
            [0, 1, 0, 0],
            modulus_minus_one,
        ];
        for window in
            [WindowBits::W6, WindowBits::W7, WindowBits::W8, WindowBits::W9, WindowBits::W10]
        {
            for limbs in corpus {
                let value = scalar(limbs);
                assert_eq!(recompose(value, window), limbs);
            }
        }
    }

    #[test]
    fn deterministic_ragged_routing_has_expected_signs_and_runs() {
        let scalars = [scalar([1, 0, 0, 0]), scalar([255, 0, 0, 0]), scalar([256, 0, 0, 0])];
        let jobs = [
            MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 2, output_index: 1 },
            MsmJobV1 { base_offset: 2, scalar_offset: 2, term_count: 1, output_index: 0 },
        ];
        let first = route_jobs(3, &scalars, &jobs, WindowBits::W8).unwrap();
        let second = route_jobs(3, &scalars, &jobs, WindowBits::W8).unwrap();
        assert_eq!(first.records(), second.records());
        assert_eq!(first.runs(), second.runs());
        assert_eq!(first.windows_per_scalar(), 32);
        assert_eq!(first.buckets_per_window(), 128);
        assert_eq!(first.output_count(), 2);
        assert!(first.records().iter().any(|record| record.is_negative()));
        assert!(
            first
                .runs()
                .windows(2)
                .all(|pair| pair[0].dense_bucket_index < pair[1].dense_bucket_index)
        );
        for run in first.runs() {
            assert_ne!(run.record_count, 0);
            assert_eq!(run.reserved, 0);
            assert!(
                (run.record_offset as usize + run.record_count as usize) <= first.records().len()
            );
        }
    }

    #[test]
    fn rejects_non_dense_outputs_and_out_of_bounds_ranges() {
        let scalars = [CanonicalScalar::ZERO];
        assert!(matches!(
            route_jobs(
                1,
                &scalars,
                &[MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 1, output_index: 1 }],
                WindowBits::W8
            ),
            Err(RouteError::OutputIndexOutOfBounds { job_index: 0, output_index: 1 })
        ));
        assert!(matches!(
            route_jobs(
                1,
                &scalars,
                &[MsmJobV1 { base_offset: 1, scalar_offset: 0, term_count: 1, output_index: 0 }],
                WindowBits::W8
            ),
            Err(RouteError::BaseRangeOutOfBounds { job_index: 0 })
        ));
        assert!(matches!(
            route_jobs(MAX_PREPARED_BASES, &[], &[], WindowBits::W8),
            Err(RouteError::TooManyPreparedBases { .. })
        ));
    }

    #[test]
    fn explicit_erase_clears_owned_arrays() {
        let mut plan = route_jobs(
            1,
            &[scalar([1, 0, 0, 0])],
            &[MsmJobV1 { base_offset: 0, scalar_offset: 0, term_count: 1, output_index: 0 }],
            WindowBits::W8,
        )
        .unwrap();
        assert!(!plan.records().is_empty());
        plan.erase();
        assert!(plan.records().is_empty());
        assert!(plan.runs().is_empty());
        assert!(plan.jobs().is_empty());
        assert_eq!(plan.output_count(), 0);
    }

    #[test]
    fn deterministic_full_width_corpus_recomposes_and_stays_in_bucket_range() {
        let mut state = 0x6d73_6d2d_626f_6f74_u64;
        for _ in 0..4_096 {
            let mut limbs = [0_u64; 4];
            for limb in &mut limbs {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *limb = state;
            }
            limbs[3] &= 0x7fff_ffff_ffff_ffff;
            if super::super::msm_api::CanonicalScalar::from_le_limbs(limbs).is_err() {
                limbs = subtract_modulus_once(limbs);
            }
            let value = scalar(limbs);
            for window in
                [WindowBits::W6, WindowBits::W7, WindowBits::W8, WindowBits::W9, WindowBits::W10]
            {
                assert_eq!(recompose(value, window), limbs);
                for index in 0..window.windows_per_scalar() {
                    let digit = booth_digit(value, index, window);
                    assert!(digit.unsigned_abs() <= window.buckets_per_window());
                }
            }
        }
    }

    fn recompose(scalar: CanonicalScalar, window: WindowBits) -> [u64; 4] {
        let width = window.get();
        let mut positive = [0_u64; 5];
        let mut negative = [0_u64; 5];
        for index in 0..window.windows_per_scalar() {
            let digit = booth_digit(scalar, index, window);
            let destination = if digit < 0 { &mut negative } else { &mut positive };
            add_shifted(destination, digit.unsigned_abs(), index * width);
        }
        subtract_wide(positive, negative)[..4].try_into().unwrap()
    }

    fn add_shifted(value: &mut [u64; 5], addend: u32, shift: u32) {
        let limb = (shift / 64) as usize;
        let bits = shift % 64;
        let shifted = (addend as u128) << bits;
        let mut carry = shifted;
        let mut index = limb;
        while carry != 0 {
            let sum = value[index] as u128 + (carry & u64::MAX as u128);
            value[index] = sum as u64;
            carry = (carry >> 64) + (sum >> 64);
            index += 1;
        }
    }

    fn subtract_wide(mut lhs: [u64; 5], rhs: [u64; 5]) -> [u64; 5] {
        let mut borrow = 0_u128;
        for index in 0..lhs.len() {
            let subtrahend = rhs[index] as u128 + borrow;
            let minuend = lhs[index] as u128;
            lhs[index] = minuend.wrapping_sub(subtrahend) as u64;
            borrow = u128::from(minuend < subtrahend);
        }
        assert_eq!(borrow, 0);
        lhs
    }

    fn subtract_modulus_once(mut limbs: [u64; 4]) -> [u64; 4] {
        let mut borrow = 0_u128;
        for index in 0..limbs.len() {
            let subtrahend = BLS12_381_SCALAR_MODULUS[index] as u128 + borrow;
            let minuend = limbs[index] as u128;
            limbs[index] = minuend.wrapping_sub(subtrahend) as u64;
            borrow = u128::from(minuend < subtrahend);
        }
        assert_eq!(borrow, 0);
        limbs
    }
}
