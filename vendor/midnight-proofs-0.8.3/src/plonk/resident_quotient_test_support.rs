// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Feature-gated proof-facing diagnostic for the resident Metal quotient epoch.
//!
//! This module is intentionally not production routing. It borrows one already
//! computed CPU trace, preserves its transcript/RNG order, and compares the
//! transcript-visible quotient boundary. CUDA is neither linked nor invoked by
//! this path. Quotient limbs are submitted directly to the standalone Metal
//! MSM, then combined on Metal; only the resulting opening polynomial is
//! materialized for the current host multi-open continuation.

#[cfg(feature = "metal-resident-quotient-diagnostic")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use ff::Field;
use midnight_curves::{Bls12, Fq, G1Projective};
use midnight_gpu_prover::{
    DiagnosticQuotientEpochInput, PreparedResidentProvingKey, PreparedResidentSrs,
    QuotientCoefficientSources, RawFr, ResidentError,
    metal::{self, MetalKeyContext, MetalSession},
};
#[cfg(feature = "metal-resident-quotient-diagnostic")]
use rand_core::OsRng;
use rand_core::{CryptoRng, RngCore};

use super::{Error, ProvingKey, traces::ProverTrace};
#[cfg(feature = "metal-resident-quotient-diagnostic")]
use crate::transcript::{CircuitTranscript, TranscriptHash};
use crate::{
    poly::kzg::{KZGCommitmentScheme, commit_resident_quotient_lease, params::ParamsKZG},
    transcript::{Hashable, Sampleable, Transcript},
};

#[cfg(feature = "metal-resident-quotient-diagnostic")]
static EPOCH_AUDIT: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) struct MetalQuotientPrefix {
    pub(crate) x: Fq,
    #[cfg_attr(not(feature = "metal-resident-quotient-diagnostic"), allow(dead_code))]
    pub(crate) h_poly: Vec<Fq>,
}

/// Runs the diagnostic resident quotient boundary for one already-computed trace.
///
/// The caller must use this instead of CPU quotient construction with the same
/// transcript and RNG state. Exactly `q - 1` blinds are sampled in the existing
/// ascending boundary order. All resident Metal commitments are completed
/// before the first transcript write, and the returned challenge is the
/// resulting `x`.
#[allow(dead_code)] // Invoked by the feature-gated authentic proof harness in the next slice.
pub(crate) fn run_diagnostic_resident_quotient_prefix<T: Transcript>(
    session: &MetalSession,
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fq, KZGCommitmentScheme<Bls12>>,
    trace: &ProverTrace<Fq>,
    transcript: &mut T,
    rng: &mut (impl RngCore + CryptoRng),
) -> Result<Fq, Error>
where
    Fq: Sampleable<T::Hash>,
    G1Projective: Hashable<T::Hash>,
{
    require_one_proof(trace)?;
    let key = pk.prepared_resident_proving_key().map_err(Error::Backend)?;
    let prepared_srs = params
        .prepared_resident_srs(key.domain().n())
        .map_err(Error::Backend)?;
    let context = metal::prepare_resident_key_context(session, &key, &prepared_srs)
        .map_err(|error| Error::Backend(error.to_string()))?;

    run_diagnostic_resident_quotient_prefix_with_context(
        session,
        &context,
        params,
        &key,
        &prepared_srs,
        trace,
        transcript,
        rng,
        false,
    )
    .map(|prefix| prefix.x)
}

/// Replaces the CPU quotient prefix while retaining an exact host continuation.
///
/// The field epoch, quotient commitments, and challenge-dependent limb
/// combination execute on Metal. One combined opening polynomial is read for
/// the existing multi-open path; no failure after selection retries the CPU
/// quotient implementation.
#[cfg(feature = "metal-resident-quotient-diagnostic")]
pub(crate) fn run_shared_resident_quotient_prefix<H: TranscriptHash>(
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fq, KZGCommitmentScheme<Bls12>>,
    trace: &ProverTrace<Fq>,
    transcript: &mut CircuitTranscript<H>,
    rng: &mut (impl RngCore + CryptoRng),
) -> Result<MetalQuotientPrefix, Error>
where
    Fq: Sampleable<H>,
    G1Projective: Hashable<H>,
{
    require_one_proof(trace)?;
    let session =
        metal::diagnostic_shared_session().map_err(|error| Error::Backend(error.to_string()))?;
    let key = pk.prepared_resident_proving_key().map_err(Error::Backend)?;
    let prepared_srs = params
        .prepared_resident_srs(key.domain().n())
        .map_err(Error::Backend)?;
    let context = shared_key_context(session, &key, &prepared_srs)?;
    let _audit = EPOCH_AUDIT
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| Error::Backend("resident quotient audit lock is poisoned".to_owned()))?;

    run_diagnostic_resident_quotient_prefix_with_context(
        session,
        &context,
        params,
        &key,
        &prepared_srs,
        trace,
        transcript,
        rng,
        true,
    )
}

/// Runs the quotient prefix as a discarded shadow of the real CPU proof.
///
/// The proof transcript and RNG are not modified: this path deep-clones the
/// audited built-in transcript and uses an independent OS-backed diagnostic
/// RNG. It therefore measures the safe epoch through mandatory GPU cleanup
/// while leaving the canonical CPU quotient and final proof bytes unchanged.
#[cfg(feature = "metal-resident-quotient-diagnostic")]
pub(crate) fn run_shared_diagnostic_resident_quotient_shadow<H: TranscriptHash>(
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fq, KZGCommitmentScheme<Bls12>>,
    trace: &ProverTrace<Fq>,
    transcript: &CircuitTranscript<H>,
) -> Result<(), Error>
where
    Fq: Sampleable<H>,
    G1Projective: Hashable<H>,
{
    require_one_proof(trace)?;
    let session =
        metal::diagnostic_shared_session().map_err(|error| Error::Backend(error.to_string()))?;
    let key = pk.prepared_resident_proving_key().map_err(Error::Backend)?;
    let prepared_srs = params
        .prepared_resident_srs(key.domain().n())
        .map_err(Error::Backend)?;
    let context = shared_key_context(session, &key, &prepared_srs)?;
    let mut shadow_transcript = transcript.clone();
    let mut shadow_rng = OsRng;

    // The epoch itself is serialized by MetalSession, but the before/after
    // counter audit surrounds that internal lock. Keep this wider diagnostic
    // window exclusive so another proof cannot contaminate its safety deltas.
    let _audit = EPOCH_AUDIT
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| Error::Backend("diagnostic quotient audit lock is poisoned".to_owned()))?;

    run_diagnostic_resident_quotient_prefix_with_context(
        session,
        &context,
        params,
        &key,
        &prepared_srs,
        trace,
        &mut shadow_transcript,
        &mut shadow_rng,
        false,
    )?;
    Ok(())
}

#[cfg(feature = "metal-resident-quotient-diagnostic")]
fn shared_key_context(
    session: &MetalSession,
    key: &PreparedResidentProvingKey,
    prepared_srs: &PreparedResidentSrs,
) -> Result<MetalKeyContext, Error> {
    type CacheKey = ([u8; 32], [u8; 32]);
    static CONTEXTS: OnceLock<Mutex<HashMap<CacheKey, MetalKeyContext>>> = OnceLock::new();

    let cache_key = (key.fingerprint(), prepared_srs.fingerprint());
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(context) = contexts
        .lock()
        .map_err(|_| Error::Backend("diagnostic Metal context cache is poisoned".to_owned()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(context);
    }

    let context = metal::prepare_resident_key_context(session, key, prepared_srs)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let mut contexts = contexts
        .lock()
        .map_err(|_| Error::Backend("diagnostic Metal context cache is poisoned".to_owned()))?;
    Ok(contexts
        .entry(cache_key)
        .or_insert_with(|| context.clone())
        .clone())
}

#[allow(clippy::too_many_arguments)]
fn run_diagnostic_resident_quotient_prefix_with_context<T: Transcript>(
    session: &MetalSession,
    context: &MetalKeyContext,
    params: &ParamsKZG<Bls12>,
    key: &PreparedResidentProvingKey,
    prepared_srs: &PreparedResidentSrs,
    trace: &ProverTrace<Fq>,
    transcript: &mut T,
    rng: &mut (impl RngCore + CryptoRng),
    materialize_opening: bool,
) -> Result<MetalQuotientPrefix, Error>
where
    Fq: Sampleable<T::Hash>,
    G1Projective: Hashable<T::Hash>,
{
    let advice = compact_polynomial_slices(
        "advice",
        &trace.advice_polys[0],
        key.graphs().packed().advice_compact_to_logical(),
    )?;
    let instance = compact_polynomial_slices(
        "instance",
        &trace.instance_polys[0],
        key.graphs().packed().instance_compact_to_logical(),
    )?;
    let permutation_products = trace.permutations[0]
        .sets
        .iter()
        .map(|set| raw_scalar_slice(&set.permutation_product_poly.values))
        .collect::<Vec<_>>();
    let lookup_products = trace.lookups[0]
        .iter()
        .map(|lookup| raw_scalar_slice(&lookup.product_poly.values))
        .collect::<Vec<_>>();
    let lookup_permuted_inputs = trace.lookups[0]
        .iter()
        .map(|lookup| raw_scalar_slice(&lookup.permuted_input_poly.values))
        .collect::<Vec<_>>();
    let lookup_permuted_tables = trace.lookups[0]
        .iter()
        .map(|lookup| raw_scalar_slice(&lookup.permuted_table_poly.values))
        .collect::<Vec<_>>();
    let trash = trace.trashcans[0]
        .iter()
        .map(|trash| raw_scalar_slice(&trash.trash_poly.values))
        .collect::<Vec<_>>();
    let sources = QuotientCoefficientSources::new(
        &advice,
        &instance,
        &permutation_products,
        &lookup_products,
        &lookup_permuted_inputs,
        &lookup_permuted_tables,
        &trash,
    );

    let total_columns = advice.len()
        + instance.len()
        + permutation_products.len()
        + lookup_products.len()
        + lookup_permuted_inputs.len()
        + lookup_permuted_tables.len()
        + trash.len();
    require_request_memory_budget(session, context, key, total_columns, trace.challenges.len())?;

    let piece_count = key.domain().quotient_piece_count() as usize;
    if piece_count < 2 {
        return Err(Error::Backend(
            "diagnostic resident quotient requires at least two limbs".to_owned(),
        ));
    }
    // This is the existing CPU prover order from `blind_quotient_limbs`: one
    // field sample for each boundary, from the first boundary to the last.
    let quotient_blinds = (0..piece_count - 1)
        .map(|_| raw_scalar(Fq::random(&mut *rng)))
        .collect::<Vec<_>>();
    let input = DiagnosticQuotientEpochInput::new(
        &key,
        &prepared_srs,
        sources,
        raw_scalar_slice(&trace.challenges),
        raw_scalar(trace.beta),
        raw_scalar(trace.gamma),
        raw_scalar(trace.theta),
        raw_scalar(trace.trash_challenge),
        raw_scalar(trace.y),
        &quotient_blinds,
    );

    let expected_lease_bytes = (key.domain().n() as u64)
        .checked_mul(key.domain().quotient_piece_count() as u64)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<RawFr>() as u64))
        .ok_or_else(|| Error::Backend("diagnostic quotient lease byte count overflowed".into()))?;
    let activity_before = session.activity();
    let commitments_and_continuation =
        metal::with_resident_quotient_epoch(session, &context, input, |lease| {
            if lease.prepared_key_fingerprint() != key.fingerprint()
                || lease
                    .piece_len()
                    .checked_mul(lease.piece_count())
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<RawFr>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    != Some(expected_lease_bytes)
            {
                return Err(ResidentError::Backend(
                    "resident quotient lease differs from the prepared request".to_owned(),
                ));
            }
            let commitments = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.quotient.commit_resident_limbs",
                lease.piece_len().saturating_mul(lease.piece_count()),
                || commit_resident_quotient_lease(session, params, &prepared_srs, &lease),
            )
            .map_err(ResidentError::Backend)?;
            if !materialize_opening {
                return Ok((commitments, None));
            }

            let mut staged = transcript.clone();
            for commitment in &commitments {
                staged.write(commitment).map_err(|_| {
                    ResidentError::Backend(
                        "could not bind a resident quotient commitment to the transcript"
                            .to_owned(),
                    )
                })?;
            }
            let x: Fq = staged.squeeze_challenge();
            let splitting_factor = x.pow_vartime([u64::from(key.domain().n() - 1)]);
            let h_poly = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.quotient.materialize_opening",
                lease.piece_len().saturating_mul(lease.piece_count()),
                || {
                    metal::materialize_resident_quotient_combination(
                        session,
                        &lease,
                        raw_scalar(splitting_factor),
                    )
                },
            )?
            .into_iter()
            .map(|value| Fq::from(blst::blst_fr { l: value.limbs }))
            .collect::<Vec<_>>();
            *transcript = staged;
            Ok((Vec::new(), Some((x, h_poly))))
        });
    let (commitments, continuation) =
        commitments_and_continuation.map_err(|error| Error::Backend(error.to_string()))?;
    let activity_after = session.activity();
    require_counter_delta(
        "quotient materializations",
        activity_before.quotient_materializations,
        activity_after.quotient_materializations,
        u64::from(materialize_opening),
    )?;
    require_counter_delta(
        "quotient whole-polynomial materializations",
        activity_before.quotient_whole_polynomial_materializations,
        activity_after.quotient_whole_polynomial_materializations,
        u64::from(materialize_opening),
    )?;
    require_counter_delta(
        "successful quotient epochs",
        activity_before.quotient_epochs,
        activity_after.quotient_epochs,
        1,
    )?;
    require_counter_delta(
        "failed quotient epochs",
        activity_before.quotient_epoch_failures,
        activity_after.quotient_epoch_failures,
        0,
    )?;
    require_positive_counter_delta(
        "successful quotient epoch elapsed nanoseconds",
        activity_before.quotient_epoch_elapsed_ns,
        activity_after.quotient_epoch_elapsed_ns,
    )?;
    require_counter_delta(
        "failed quotient epoch elapsed nanoseconds",
        activity_before.quotient_epoch_failure_elapsed_ns,
        activity_after.quotient_epoch_failure_elapsed_ns,
        0,
    )?;
    require_counter_value(
        "quarantined secret bytes",
        activity_after.quarantined_secret_bytes,
        0,
    )?;
    require_counter_delta(
        "CPU-MSM leases",
        activity_before.cpu_msm_leases,
        activity_after.cpu_msm_leases,
        0,
    )?;
    require_counter_delta(
        "CPU-MSM lease bytes",
        activity_before.cpu_msm_lease_bytes,
        activity_after.cpu_msm_lease_bytes,
        0,
    )?;
    require_counter_delta(
        "resident MSM leases",
        activity_before.resident_msm_leases,
        activity_after.resident_msm_leases,
        1,
    )?;
    require_counter_delta(
        "resident MSM lease bytes",
        activity_before.resident_msm_lease_bytes,
        activity_after.resident_msm_lease_bytes,
        expected_lease_bytes,
    )?;

    let (x, h_poly) = if let Some(continuation) = continuation {
        continuation
    } else {
        let x = transact_commitments_and_challenge(
            transcript,
            Ok::<_, Error>(commitments),
            |staged, commitment| staged.write(commitment).map_err(Error::from),
            |staged| staged.squeeze_challenge(),
        )?;
        (x, Vec::new())
    };
    Ok(MetalQuotientPrefix { x, h_poly })
}

fn require_request_memory_budget(
    session: &MetalSession,
    context: &MetalKeyContext,
    key: &PreparedResidentProvingKey,
    total_columns: usize,
    challenge_count: usize,
) -> Result<(), Error> {
    let domain = key.domain();
    let n = u64::from(domain.n());
    let n_ext = u64::from(domain.n_ext());
    let columns = u64::try_from(total_columns)
        .map_err(|_| Error::Backend("diagnostic quotient column count overflowed".to_owned()))?;
    let challenges = u64::try_from(challenge_count)
        .map_err(|_| Error::Backend("diagnostic quotient challenge count overflowed".to_owned()))?;
    let pieces = u64::from(domain.quotient_piece_count());
    let ntt_tile_columns = columns.min(8);
    let secret_elements = columns
        .checked_mul(
            n.checked_add(n_ext).ok_or_else(|| {
                Error::Backend("diagnostic quotient domain size overflowed".into())
            })?,
        )
        .and_then(|value| value.checked_add(n_ext.checked_mul(ntt_tile_columns)?.checked_mul(2)?))
        .and_then(|value| value.checked_add(challenges))
        .and_then(|value| value.checked_add(pieces))
        .ok_or_else(|| Error::Backend("diagnostic quotient memory estimate overflowed".into()))?;
    let secret_bytes = secret_elements
        .checked_mul(std::mem::size_of::<RawFr>() as u64)
        .ok_or_else(|| Error::Backend("diagnostic quotient secret bytes overflowed".into()))?;
    let immutable_bytes = u64::try_from(context.immutable_bytes())
        .map_err(|_| Error::Backend("diagnostic immutable bytes overflowed".to_owned()))?;
    let required = secret_bytes
        .checked_add(immutable_bytes)
        .ok_or_else(|| Error::Backend("diagnostic working-set estimate overflowed".into()))?;
    let recommended = session
        .device_info()
        .map_err(|error| Error::Backend(error.to_string()))?
        .recommended_working_set_bytes;
    let budget = recommended / 2;
    if recommended == 0 || required > budget {
        return Err(Error::Backend(format!(
            "diagnostic quotient requires {required} Metal bytes, exceeding the conservative {budget}-byte budget"
        )));
    }
    Ok(())
}

fn require_one_proof(trace: &ProverTrace<Fq>) -> Result<(), Error> {
    let outer_lengths = [
        trace.advice_polys.len(),
        trace.instance_polys.len(),
        trace.lookups.len(),
        trace.trashcans.len(),
        trace.permutations.len(),
    ];
    if outer_lengths != [1; 5] {
        return Err(Error::Backend(
            "diagnostic resident quotient currently requires one proof per trace".to_owned(),
        ));
    }
    Ok(())
}

fn compact_polynomial_slices<'a, B>(
    label: &str,
    polynomials: &'a [crate::poly::Polynomial<Fq, B>],
    compact_to_logical: &[u32],
) -> Result<Vec<&'a [RawFr]>, Error> {
    compact_to_logical
        .iter()
        .copied()
        .enumerate()
        .map(|(compact_index, logical_index)| {
            polynomials
                .get(logical_index as usize)
                .map(|polynomial| raw_scalar_slice(&polynomial.values))
                .ok_or_else(|| {
                    Error::Backend(format!(
                        "diagnostic {label} compact column {compact_index} names missing logical column {logical_index}"
                    ))
                })
        })
        .collect()
}

fn require_counter_delta(label: &str, before: u64, after: u64, expected: u64) -> Result<(), Error> {
    if after.checked_sub(before) != Some(expected) {
        return Err(Error::Backend(format!(
            "diagnostic resident quotient {label} delta was not {expected}"
        )));
    }
    Ok(())
}

fn require_positive_counter_delta(label: &str, before: u64, after: u64) -> Result<(), Error> {
    if after.checked_sub(before).is_none_or(|delta| delta == 0) {
        return Err(Error::Backend(format!(
            "diagnostic resident quotient {label} did not increase"
        )));
    }
    Ok(())
}

fn require_counter_value(label: &str, actual: u64, expected: u64) -> Result<(), Error> {
    if actual != expected {
        return Err(Error::Backend(format!(
            "diagnostic resident quotient {label} was not {expected}"
        )));
    }
    Ok(())
}

fn transact_commitments_and_challenge<T, C, E, X>(
    transcript: &mut T,
    commitments: Result<Vec<C>, E>,
    mut write: impl FnMut(&mut T, &C) -> Result<(), E>,
    squeeze: impl FnOnce(&mut T) -> X,
) -> Result<X, E>
where
    T: Clone,
{
    let commitments = commitments?;
    let mut staged = transcript.clone();
    for commitment in &commitments {
        write(&mut staged, commitment)?;
    }
    let challenge = squeeze(&mut staged);
    *transcript = staged;
    Ok(challenge)
}

fn raw_scalar(value: Fq) -> RawFr {
    raw_scalar_slice(std::slice::from_ref(&value))[0]
}

#[allow(unsafe_code)]
fn raw_scalar_slice(values: &[Fq]) -> &[RawFr] {
    assert_eq!(std::mem::size_of::<Fq>(), std::mem::size_of::<RawFr>());
    assert_eq!(std::mem::align_of::<Fq>(), std::mem::align_of::<RawFr>());
    // SAFETY: `Fq` is the BLS12-381 scalar field in this codebase and is
    // repr(transparent) over four reduced little-endian Montgomery u64 limbs,
    // exactly matching `RawFr`. The returned borrow cannot outlive `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<RawFr>(), values.len()) }
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::{
        Error, G1Projective, KZGCommitmentScheme, MetalSession, ParamsKZG, ProverTrace,
        raw_scalar_slice, run_diagnostic_resident_quotient_prefix,
        transact_commitments_and_challenge,
    };
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Constraints, Expression, Fixed, Instance,
            Selector, TableColumn, keygen_pk, keygen_vk_with_k,
        },
        poly::{ExtendedLagrangeCoeff, Polynomial, Rotation, kzg::diagnostic_scalar_slice},
        transcript::{Blake2b256, CircuitTranscript, Hashable, Sampleable, Transcript},
    };

    type Scheme = KZGCommitmentScheme<midnight_curves::Bls12>;

    #[derive(Clone, Debug, Default)]
    struct QuotientParityCircuit;

    #[derive(Clone, Debug)]
    struct QuotientParityConfig {
        graph_advice: Column<Advice>,
        graph_fixed: Column<Fixed>,
        graph_instance: Column<Instance>,
        permutation_advice: Column<Advice>,
        permutation_fixed: Column<Fixed>,
        lookup_advice: Column<Advice>,
        lookup_selector: Selector,
        lookup_table: TableColumn,
        trash_advice: Column<Advice>,
        trash_selector: Selector,
    }

    impl Circuit<midnight_curves::Fq> for QuotientParityCircuit {
        type Config = QuotientParityConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<midnight_curves::Fq>) -> Self::Config {
            let graph_advice = meta.advice_column();
            let permutation_advice = meta.advice_column();
            let lookup_advice = meta.advice_column();
            let trash_advice = meta.advice_column();
            let graph_fixed = meta.fixed_column();
            let permutation_fixed = meta.fixed_column();
            let graph_instance = meta.instance_column();

            // Exercise the three permutation column kinds and retain an
            // additional advice column in the exact insertion order.
            meta.enable_equality(permutation_advice);
            meta.enable_equality(permutation_fixed);
            meta.enable_equality(graph_instance);
            meta.enable_equality(graph_advice);

            meta.create_gate("diagnostic quotient parity", |meta| {
                let advice = meta.query_advice(graph_advice, Rotation::cur());
                let fixed = meta.query_fixed(graph_fixed, Rotation::cur());
                let instance = meta.query_instance(graph_instance, Rotation::cur());
                Constraints::without_selector(vec![advice + fixed - instance])
            });

            let lookup_selector = meta.complex_selector();
            let lookup_table = meta.lookup_table_column();
            meta.lookup("diagnostic quotient lookup", |meta| {
                let selector = meta.query_selector(lookup_selector);
                let inactive = Expression::from(1) - selector.clone();
                let advice = meta.query_advice(lookup_advice, Rotation::cur());
                vec![(selector * advice + inactive, lookup_table)]
            });

            let trash_selector = meta.complex_selector();
            meta.create_gate("diagnostic quotient trash", |meta| {
                let advice = meta.query_advice(trash_advice, Rotation::cur());
                Constraints::with_additive_selector(
                    trash_selector,
                    vec![advice.clone().square() - advice],
                )
            });

            QuotientParityConfig {
                graph_advice,
                graph_fixed,
                graph_instance,
                permutation_advice,
                permutation_fixed,
                lookup_advice,
                lookup_selector,
                lookup_table,
                trash_advice,
                trash_selector,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<midnight_curves::Fq>,
        ) -> Result<(), Error> {
            layouter.assign_table(
                || "diagnostic quotient lookup table",
                |mut table| {
                    for (row, value) in [1u64, 3].into_iter().enumerate() {
                        table.assign_cell(
                            || format!("lookup value {value}"),
                            config.lookup_table,
                            row,
                            || Value::known(midnight_curves::Fq::from(value)),
                        )?;
                    }
                    Ok(())
                },
            )?;

            let permutation_advice = layouter.assign_region(
                || "diagnostic quotient parity witness",
                |mut region| {
                    config.lookup_selector.enable(&mut region, 0)?;
                    config.trash_selector.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "graph advice",
                        config.graph_advice,
                        0,
                        || Value::known(midnight_curves::Fq::from(3u64)),
                    )?;
                    region.assign_fixed(
                        || "graph fixed",
                        config.graph_fixed,
                        0,
                        || Value::known(midnight_curves::Fq::from(5u64)),
                    )?;
                    let permutation_advice = region.assign_advice(
                        || "permutation advice",
                        config.permutation_advice,
                        0,
                        || Value::known(midnight_curves::Fq::from(8u64)),
                    )?;
                    let permutation_fixed = region.assign_fixed(
                        || "permutation fixed",
                        config.permutation_fixed,
                        0,
                        || Value::known(midnight_curves::Fq::from(8u64)),
                    )?;
                    region.constrain_equal(permutation_advice.cell(), permutation_fixed.cell())?;
                    region.assign_advice(
                        || "lookup advice",
                        config.lookup_advice,
                        0,
                        || Value::known(midnight_curves::Fq::from(3u64)),
                    )?;
                    region.assign_advice(
                        || "trash advice",
                        config.trash_advice,
                        0,
                        || Value::known(midnight_curves::Fq::ONE),
                    )?;
                    Ok(permutation_advice)
                },
            )?;
            layouter.constrain_instance(permutation_advice.cell(), config.graph_instance, 0)
        }
    }

    fn run_cpu_quotient_prefix<T: Transcript>(
        params: &ParamsKZG<midnight_curves::Bls12>,
        pk: &super::ProvingKey<midnight_curves::Fq, Scheme>,
        trace: ProverTrace<midnight_curves::Fq>,
        numerator: Polynomial<midnight_curves::Fq, ExtendedLagrangeCoeff>,
        transcript: &mut T,
        rng: &mut (impl RngCore + rand_core::CryptoRng),
    ) -> Result<midnight_curves::Fq, Error>
    where
        midnight_curves::Fq: Sampleable<T::Hash> + Hashable<T::Hash>,
        G1Projective: Hashable<T::Hash>,
    {
        let ProverTrace { vanishing, .. } = trace;
        let _constructed = vanishing.construct::<Scheme, T>(
            params,
            pk.get_vk().get_domain(),
            numerator,
            transcript,
            rng,
        )?;
        Ok(transcript.squeeze_challenge())
    }

    fn appended_proof_bytes<T: Transcript>(
        baseline: &[u8],
        transcript: &T,
    ) -> Result<Vec<u8>, Error> {
        transcript
            .clone()
            .finalize()
            .strip_prefix(baseline)
            .map(ToOwned::to_owned)
            .ok_or_else(|| {
                Error::Backend(
                    "diagnostic quotient transcript no longer has its trace prefix".into(),
                )
            })
    }

    #[test]
    fn raw_fr_fq_cast_preserves_montgomery_abi_and_field_values() {
        let scalars = [
            midnight_curves::Fq::ZERO,
            midnight_curves::Fq::ONE,
            midnight_curves::Fq::from(7u64),
        ];
        let raw = raw_scalar_slice(&scalars);

        assert_eq!(raw.as_ptr().cast::<midnight_curves::Fq>(), scalars.as_ptr());
        assert_eq!(raw[0].limbs, [0; 4]);
        assert_eq!(
            raw[1].limbs,
            [
                0x0000_0001_ffff_fffe,
                0x5884_b7fa_0003_4802,
                0x998c_4fef_ecbc_4ff5,
                0x1824_b159_acc5_056f,
            ]
        );

        let round_trip = diagnostic_scalar_slice(raw);
        assert_eq!(round_trip.as_ptr(), scalars.as_ptr());
        assert!(round_trip == scalars.as_slice());
        assert!(
            round_trip[2] * midnight_curves::Fq::from(3u64) == midnight_curves::Fq::from(21u64)
        );
    }

    #[test]
    fn commitment_error_writes_nothing() {
        let mut transcript = vec![3u64];
        let result = transact_commitments_and_challenge(
            &mut transcript,
            Err::<Vec<u64>, _>("msm failed"),
            |staged, commitment| {
                staged.push(*commitment);
                Ok(())
            },
            |staged| staged.len(),
        );
        assert_eq!(result, Err("msm failed"));
        assert_eq!(transcript, [3]);
    }

    #[test]
    fn transcript_write_error_rolls_back_every_staged_commitment() {
        let mut transcript = vec![3u64];
        let result = transact_commitments_and_challenge(
            &mut transcript,
            Ok(vec![7, 11, 13]),
            |staged, commitment| {
                if *commitment == 11 {
                    return Err("write failed");
                }
                staged.push(*commitment);
                Ok(())
            },
            |staged| staged.len(),
        );
        assert_eq!(result, Err("write failed"));
        assert_eq!(transcript, [3]);
    }

    #[test]
    fn successful_commitments_preserve_order() {
        let mut transcript = vec![3u64];
        let challenge = transact_commitments_and_challenge::<_, _, &str, _>(
            &mut transcript,
            Ok(vec![7, 11, 13]),
            |staged, commitment| {
                staged.push(*commitment);
                Ok(())
            },
            |staged| staged.iter().sum::<u64>(),
        )
        .expect("write all prepared commitments and squeeze");
        assert_eq!(transcript, [3, 7, 11, 13]);
        assert_eq!(challenge, 34);
    }

    #[cfg(all(
        not(feature = "gpu-blitzar"),
        not(all(feature = "gpu-metal", target_os = "macos", target_arch = "aarch64"))
    ))]
    #[test]
    fn authentic_trace_metal_and_cpu_quotient_prefixes_match() {
        const K: u32 = 4;
        let params = ParamsKZG::<midnight_curves::Bls12>::unsafe_setup(
            K,
            ChaCha20Rng::from_seed([0x51; 32]),
        );
        let circuit = QuotientParityCircuit;
        let verifying_key =
            keygen_vk_with_k::<midnight_curves::Fq, Scheme, _>(&params, &circuit, K)
                .expect("build diagnostic parity verifying key");
        let proving_key =
            keygen_pk(verifying_key, &circuit).expect("build diagnostic parity proving key");
        let prepared = proving_key
            .prepared_resident_proving_key()
            .expect("prepare diagnostic parity key");
        let sections = prepared.graphs().sections();
        assert_eq!(sections.custom(), 1);
        assert_eq!(sections.lookups(), 1);
        assert_eq!(sections.lookup_inputs(), 1);
        assert_eq!(sections.lookup_tables(), 1);
        assert_eq!(sections.trash(), 1);
        assert_eq!(
            prepared
                .permutation_columns()
                .iter()
                .map(|column| column.kind())
                .collect::<Vec<_>>(),
            [
                midnight_gpu_prover::ResidentColumnKind::Advice,
                midnight_gpu_prover::ResidentColumnKind::Fixed,
                midnight_gpu_prover::ResidentColumnKind::Instance,
                midnight_gpu_prover::ResidentColumnKind::Advice,
            ]
        );
        let session = MetalSession::new().expect("initialize Metal for quotient parity");

        let graph_instance = [midnight_curves::Fq::from(8u64)];
        let instance_columns = [graph_instance.as_slice()];
        let proof_instances = [instance_columns.as_slice()];
        let mut transcript = CircuitTranscript::<Blake2b256>::init();
        let mut trace_rng = ChaCha20Rng::from_seed([0xa7; 32]);
        let trace = crate::plonk::prover::compute_trace(
            &params,
            &proving_key,
            std::slice::from_ref(&circuit),
            #[cfg(feature = "committed-instances")]
            0,
            &proof_instances,
            &mut transcript,
            &mut trace_rng,
        )
        .expect("compute an authentic proof trace through y");
        assert_eq!(trace.advice_polys.len(), 1);
        assert!(!trace.advice_polys[0].is_empty());
        assert_eq!(trace.instance_polys.len(), 1);
        assert_eq!(trace.instance_polys[0].len(), 1);
        assert_eq!(trace.permutations.len(), 1);
        assert!(!trace.permutations[0].sets.is_empty());
        assert_eq!(trace.lookups.len(), 1);
        assert_eq!(trace.lookups[0].len(), 1);
        assert_eq!(trace.trashcans.len(), 1);
        assert_eq!(trace.trashcans[0].len(), 1);
        let cpu_numerator = crate::plonk::prover::compute_nu_poly(&proving_key, &trace);

        let baseline = transcript.clone().finalize();
        let mut metal_transcript = transcript.clone();
        let mut cpu_transcript = transcript;
        let mut metal_rng = trace_rng.clone();
        let mut cpu_rng = trace_rng;

        let metal_x = run_diagnostic_resident_quotient_prefix(
            &session,
            &params,
            &proving_key,
            &trace,
            &mut metal_transcript,
            &mut metal_rng,
        )
        .expect("run diagnostic Metal quotient prefix");
        let cpu_x = run_cpu_quotient_prefix(
            &params,
            &proving_key,
            trace,
            cpu_numerator,
            &mut cpu_transcript,
            &mut cpu_rng,
        )
        .expect("run reference CPU quotient prefix");

        let metal_commitment_bytes = appended_proof_bytes(&baseline, &metal_transcript)
            .expect("extract ordered Metal commitment bytes");
        let cpu_commitment_bytes = appended_proof_bytes(&baseline, &cpu_transcript)
            .expect("extract ordered CPU commitment bytes");
        let piece_count = proving_key.get_vk().get_domain().get_quotient_poly_degree();
        let commitment_width =
            <G1Projective as Hashable<Blake2b256>>::to_bytes(&G1Projective::default()).len();
        assert_eq!(metal_commitment_bytes.len(), piece_count * commitment_width);
        assert_eq!(cpu_commitment_bytes.len(), piece_count * commitment_width);
        for (piece_index, (metal, cpu)) in metal_commitment_bytes
            .chunks_exact(commitment_width)
            .zip(cpu_commitment_bytes.chunks_exact(commitment_width))
            .enumerate()
        {
            assert_eq!(metal, cpu, "quotient commitment limb {piece_index}");
        }
        assert!(metal_x == cpu_x);

        let mut metal_next = [0u8; 64];
        let mut cpu_next = [0u8; 64];
        metal_rng.fill_bytes(&mut metal_next);
        cpu_rng.fill_bytes(&mut cpu_next);
        assert_eq!(metal_next, cpu_next);
    }
}
