#[cfg(feature = "gpu-blitzar")]
use std::{any::TypeId, sync::Arc};
use std::{hash::Hash, iter};

use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::ff::BatchInvert;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;

use super::{
    super::{Error, ProvingKey, circuit::Expression},
    Argument,
};
use crate::{
    plonk::evaluation::evaluate,
    poly::{
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
        commitment::PolynomialCommitmentScheme,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Permuted<F: PrimeField> {
    compressed_input_expression: Polynomial<F, LagrangeCoeff>,
    permuted_input_expression: Polynomial<F, LagrangeCoeff>,
    permuted_input_poly: Polynomial<F, Coeff>,
    compressed_table_expression: Polynomial<F, LagrangeCoeff>,
    permuted_table_expression: Polynomial<F, LagrangeCoeff>,
    permuted_table_poly: Polynomial<F, Coeff>,
    #[cfg(feature = "gpu-blitzar")]
    resident_permuted_coefficients: Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    #[cfg(feature = "gpu-blitzar")]
    resident_trace_lagrange: Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    #[cfg(feature = "gpu-blitzar")]
    resident_input_index: Option<usize>,
    #[cfg(feature = "gpu-blitzar")]
    resident_table_index: Option<usize>,
}

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) permuted_input_poly: Box<Polynomial<F, Coeff>>,
    pub(crate) permuted_table_poly: Box<Polynomial<F, Coeff>>,
    pub(crate) product_poly: Box<Polynomial<F, Coeff>>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_product_coefficients:
        Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    #[cfg(feature = "gpu-blitzar")]
    resident_product_index: Option<usize>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_permuted_coefficients:
        Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_input_index: Option<usize>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_table_index: Option<usize>,
}

pub(crate) struct Evaluated<F: PrimeField> {
    constructed: Committed<F>,
}

#[cfg(feature = "gpu-blitzar")]
fn lookup_product_backend_enabled<F: PrimeField + 'static>(len: usize) -> bool {
    let mode = std::env::var("MIDNIGHT_GPU_LOOKUP_PRODUCT_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "false" | "no" | "off" | "none") {
        return false;
    }
    let explicitly_enabled = matches!(
        mode.as_str(),
        "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "device"
    );
    let min_rows = std::env::var("MIDNIGHT_GPU_LOOKUP_PRODUCT_MIN_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1 << 15);
    (explicitly_enabled || (mode == "auto" && len >= min_rows))
        && midnight_curves::gpu_sppark::enabled()
        && TypeId::of::<F>() == TypeId::of::<midnight_curves::Fq>()
}

#[cfg(feature = "gpu-blitzar")]
fn resident_lookup_permuted_enabled<
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    pk: &ProvingKey<F, CS>,
) -> bool {
    let len = pk.vk.domain.n as usize;
    midnight_curves::gpu_sppark::enabled()
        && TypeId::of::<F>() == TypeId::of::<midnight_curves::Fq>()
        && crate::plonk::resident_argument_without_host_mirror(
            crate::plonk::ResidentArgumentKind::Lookup,
            len,
            crate::plonk::prover::resident_finalise_supported(pk),
        )
}

#[cfg(feature = "gpu-blitzar")]
fn resident_lookup_permuted_debug_enabled() -> bool {
    matches!(
        std::env::var("MIDNIGHT_GPU_RESIDENT_LOOKUP_PERMUTED_DEBUG")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "gpu-blitzar")]
fn resident_lookup_permute_backend_enabled<
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    pk: &ProvingKey<F, CS>,
) -> bool {
    let mode = std::env::var("MIDNIGHT_GPU_LOOKUP_PERMUTE_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    !matches!(mode.as_str(), "0" | "false" | "no" | "off" | "none" | "cpu")
        && resident_lookup_permuted_enabled::<F, CS>(pk)
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
fn retain_permuted_coefficients_sppark<F: PrimeField + 'static>(
    lookups: &mut [Permuted<F>],
    domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
    if lookups.is_empty() {
        return Ok(());
    }
    let len = domain.n as usize;
    let raw = |value: F| unsafe { std::mem::transmute_copy(&value) };
    let lookup_count = lookups.len();
    let mut values = Vec::with_capacity(lookup_count * len * 4);
    for lookup in lookups.iter() {
        values.extend(lookup.compressed_input_expression.iter().copied().map(raw));
    }
    for lookup in lookups.iter() {
        values.extend(lookup.compressed_table_expression.iter().copied().map(raw));
    }
    for lookup in lookups.iter() {
        values.extend(lookup.permuted_input_expression.iter().copied().map(raw));
    }
    for lookup in lookups.iter() {
        values.extend(lookup.permuted_table_expression.iter().copied().map(raw));
    }

    let trace_lagrange = crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.trace_resident.upload",
        values.len(),
        || midnight_sppark_msm::Bls12_381FrDeviceVec::upload(&values),
    )
    .map_err(|error| Error::Backend(error.to_string()))?;
    let coefficient_indices = ((2 * lookup_count)..(4 * lookup_count))
        .map(|index| u32::try_from(index).expect("lookup index fits u32"))
        .collect::<Vec<_>>();
    let mut coefficients = trace_lagrange
        .gather_polys(len, &coefficient_indices)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let count = lookup_count * 2;
    let one = raw(F::ONE);
    crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.permuted_resident.ntt_batch",
        values.len(),
        || coefficients.ntt_zeta_batch(len, count, true, one, one, one),
    )
    .map_err(|error| Error::Backend(error.to_string()))?;

    let coefficients = Arc::new(coefficients);
    let trace_lagrange = Arc::new(trace_lagrange);
    for (index, lookup) in lookups.iter_mut().enumerate() {
        lookup.resident_permuted_coefficients = Some(coefficients.clone());
        lookup.resident_trace_lagrange = Some(trace_lagrange.clone());
        lookup.resident_input_index = Some(index);
        lookup.resident_table_index = Some(lookup_count + index);
    }
    Ok(())
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code, clippy::too_many_arguments)]
fn construct_permuted_batch_sppark<
    F: WithSmallOrderMulGroup<3> + Ord + Hash + FromUniformBytes<64> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    lookups: &[Argument<F>],
    pk: &ProvingKey<F, CS>,
    domain: &EvaluationDomain<F>,
    theta: F,
    advice_values: &[Polynomial<F, LagrangeCoeff>],
    resident_advice: Option<&midnight_sppark_msm::Bls12_381FrDeviceVec>,
    fixed_values: &[Polynomial<F, LagrangeCoeff>],
    instance_values: &[Polynomial<F, LagrangeCoeff>],
    challenges: &[F],
    blinds: &[(Vec<F>, Vec<F>)],
) -> Result<Vec<(Permuted<F>, Vec<Polynomial<F, LagrangeCoeff>>)>, Error> {
    let len = domain.n as usize;
    let usable_rows = len - (pk.vk.cs.blinding_factors() + 1);
    let debug = matches!(
        std::env::var("MIDNIGHT_GPU_LOOKUP_PERMUTE_BACKEND")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "debug" | "validate" | "check"
    );
    let raw = |value: F| unsafe { std::mem::transmute_copy(&value) };
    let input_blinds = blinds
        .iter()
        .flat_map(|(input, _)| input.iter().copied())
        .collect::<Vec<_>>();
    let table_blinds = blinds
        .iter()
        .flat_map(|(_, table)| table.iter().copied())
        .collect::<Vec<_>>();
    let resident_trace = if let (Some(resident_advice), Some(context)) = (
        resident_advice,
        crate::plonk::prover::try_sppark_resident_context(pk)?,
    ) {
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.lookup.compress_permute.resident_context",
            lookups.len() * len * 2,
            || {
                context.trace_lookup_permutation(
                    advice_values,
                    Some(resident_advice),
                    instance_values,
                    challenges,
                    theta,
                    usable_rows,
                    &input_blinds,
                    &table_blinds,
                )
            },
        )
        .map_err(Error::Backend)?
    } else {
        None
    };
    let compressed = if resident_trace.is_none() || debug {
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.lookup.compress.resident_batch",
            lookups.len() * len * 2,
            || {
                lookups
                    .par_iter()
                    .map(|lookup| {
                        lookup.compress_expressions::<CS>(
                            pk,
                            domain,
                            theta,
                            advice_values,
                            fixed_values,
                            instance_values,
                            challenges,
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )
    } else {
        (0..lookups.len())
            .map(|_| (Polynomial::init(0), Polynomial::init(0)))
            .collect()
    };
    let trace_lagrange = if let Some(resident_trace) = resident_trace {
        resident_trace
    } else {
        let mut compressed_inputs = Vec::with_capacity(lookups.len() * len);
        let mut compressed_tables = Vec::with_capacity(lookups.len() * len);
        for (input, table) in &compressed {
            compressed_inputs.extend(input.iter().copied().map(raw));
            compressed_tables.extend(table.iter().copied().map(raw));
        }
        let input_blinds = input_blinds.iter().copied().map(raw).collect::<Vec<_>>();
        let table_blinds = table_blinds.iter().copied().map(raw).collect::<Vec<_>>();
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.lookup.permute.resident_gpu_sppark",
            compressed_inputs.len() + compressed_tables.len(),
            || {
                midnight_sppark_msm::bls12_381_fr_lookup_permute_batch(
                    &compressed_inputs,
                    &compressed_tables,
                    len,
                    usable_rows,
                    &input_blinds,
                    &table_blinds,
                )
            },
        )
        .map_err(|error| Error::Backend(error.to_string()))?
    };
    let lookup_count = lookups.len();
    let permuted_values = if debug {
        let downloaded_indices = ((2 * lookup_count)..(4 * lookup_count))
            .map(|index| u32::try_from(index).expect("lookup index fits u32"))
            .collect::<Vec<_>>();
        trace_lagrange
            .gather_polys(len, &downloaded_indices)
            .map_err(|error| Error::Backend(error.to_string()))?
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?
    } else {
        Vec::new()
    };

    let permuted_indices = ((2 * lookup_count)..(4 * lookup_count))
        .map(|index| u32::try_from(index).expect("lookup index fits u32"))
        .collect::<Vec<_>>();
    let mut coefficients = trace_lagrange
        .gather_polys(len, &permuted_indices)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let one = raw(F::ONE);
    coefficients
        .ntt_zeta_batch(len, lookup_count * 2, true, one, one, one)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let coefficients = Arc::new(coefficients);
    let trace_lagrange = Arc::new(trace_lagrange);

    compressed
        .into_iter()
        .enumerate()
        .map(
            |(index, (compressed_input_expression, compressed_table_expression))| {
                let permuted_input_expression = if debug {
                    domain.lagrange_from_vec(
                        permuted_values[index * len..(index + 1) * len]
                            .iter()
                            .copied()
                            .map(|value| unsafe { std::mem::transmute_copy(&value) })
                            .collect(),
                    )
                } else {
                    Polynomial::init(0)
                };
                let table_index = lookup_count + index;
                let permuted_table_expression = if debug {
                    domain.lagrange_from_vec(
                        permuted_values[table_index * len..(table_index + 1) * len]
                            .iter()
                            .copied()
                            .map(|value| unsafe { std::mem::transmute_copy(&value) })
                            .collect(),
                    )
                } else {
                    Polynomial::init(0)
                };
                if debug {
                    let expected = permute_expression_pair(
                        pk,
                        domain,
                        &compressed_input_expression,
                        &compressed_table_expression,
                        &blinds[index].0,
                        &blinds[index].1,
                    )?;
                    if expected.0[..] != permuted_input_expression[..]
                        || expected.1[..] != permuted_table_expression[..]
                    {
                        return Err(Error::Backend(format!(
                            "resident lookup permutation mismatch at lookup {index}"
                        )));
                    }
                }
                let commitment_values = if debug {
                    vec![
                        permuted_input_expression.clone(),
                        permuted_table_expression.clone(),
                    ]
                } else {
                    Vec::new()
                };
                Ok((
                    Permuted {
                        compressed_input_expression,
                        permuted_input_expression,
                        permuted_input_poly: domain.empty_coeff(),
                        compressed_table_expression,
                        permuted_table_expression,
                        permuted_table_poly: domain.empty_coeff(),
                        resident_permuted_coefficients: Some(coefficients.clone()),
                        resident_trace_lagrange: Some(trace_lagrange.clone()),
                        resident_input_index: Some(index),
                        resident_table_index: Some(lookup_count + index),
                    },
                    commitment_values,
                ))
            },
        )
        .collect()
}

#[cfg(feature = "gpu-blitzar")]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn commit_product_batch_sppark<
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    lookups: Vec<Permuted<F>>,
    pk: &ProvingKey<F, CS>,
    params: &CS::Parameters,
    beta: F,
    gamma: F,
    blind_values: &[F],
) -> Result<(Vec<Committed<F>>, Vec<CS::Commitment>), Error> {
    let len = pk.vk.n() as usize;
    let blinding_factors = pk.vk.cs.blinding_factors();
    let raw = |value: F| unsafe { std::mem::transmute_copy(&value) };
    if blind_values.len() != lookups.len() * blinding_factors {
        return Err(Error::Backend(
            "lookup blind count does not match the argument".to_string(),
        ));
    }
    let blind_values = blind_values.iter().copied().map(raw).collect::<Vec<_>>();
    let resident_trace = lookups
        .first()
        .and_then(|lookup| lookup.resident_trace_lagrange.as_deref())
        .filter(|resident| {
            resident.len() == lookups.len().saturating_mul(len).saturating_mul(4)
                && lookups.iter().all(|lookup| {
                    lookup
                        .resident_trace_lagrange
                        .as_ref()
                        .is_some_and(|candidate| {
                            Arc::ptr_eq(
                                candidate,
                                lookups[0].resident_trace_lagrange.as_ref().unwrap(),
                            )
                        })
                })
        });

    let mut products = crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.products.gpu_sppark_device",
        lookups.len() * len,
        || {
            if let Some(resident_trace) = resident_trace {
                midnight_sppark_msm::bls12_381_fr_lookup_products_device(
                    resident_trace,
                    len,
                    lookups.len(),
                    raw(beta),
                    raw(gamma),
                    &blind_values,
                    blinding_factors,
                )
            } else {
                let mut compressed_inputs = Vec::with_capacity(lookups.len() * len);
                let mut compressed_tables = Vec::with_capacity(lookups.len() * len);
                let mut permuted_inputs = Vec::with_capacity(lookups.len() * len);
                let mut permuted_tables = Vec::with_capacity(lookups.len() * len);
                for lookup in &lookups {
                    compressed_inputs
                        .extend(lookup.compressed_input_expression.iter().copied().map(raw));
                    compressed_tables
                        .extend(lookup.compressed_table_expression.iter().copied().map(raw));
                    permuted_inputs
                        .extend(lookup.permuted_input_expression.iter().copied().map(raw));
                    permuted_tables
                        .extend(lookup.permuted_table_expression.iter().copied().map(raw));
                }
                midnight_sppark_msm::bls12_381_fr_lookup_products(
                    &compressed_inputs,
                    &compressed_tables,
                    &permuted_inputs,
                    &permuted_tables,
                    len,
                    raw(beta),
                    raw(gamma),
                    &blind_values,
                    blinding_factors,
                )
            }
        },
    )
    .map_err(|error| Error::Backend(error.to_string()))?;

    let device_commitments =
        CS::commit_lagrange_device_batch(params, &products, len, lookups.len())
            .map_err(Error::Backend)?;
    let commitments = if let Some(commitments) = device_commitments {
        commitments
    } else {
        let product_values = products
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?
            .chunks_exact(len)
            .map(|values| {
                pk.vk.domain.lagrange_from_vec(
                    values
                        .iter()
                        .copied()
                        .map(|value| unsafe { std::mem::transmute_copy(&value) })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        CS::commit_lagrange_batch(params, &product_values)
    };

    let one = raw(F::ONE);
    products
        .ntt_zeta_batch(len, lookups.len(), true, one, one, one)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let resident_products = Arc::new(products);
    let coefficients = if crate::plonk::resident_argument_without_host_mirror(
        crate::plonk::ResidentArgumentKind::Lookup,
        len,
        crate::plonk::prover::resident_finalise_supported(pk),
    ) {
        None
    } else {
        Some(
            resident_products
                .download()
                .map_err(|error| Error::Backend(error.to_string()))?,
        )
    };
    let committed = lookups
        .into_iter()
        .enumerate()
        .map(|(lookup_index, lookup)| Committed {
            permuted_input_poly: Box::new(lookup.permuted_input_poly),
            permuted_table_poly: Box::new(lookup.permuted_table_poly),
            product_poly: Box::new(coefficients.as_ref().map_or_else(
                || pk.vk.domain.empty_coeff(),
                |values| {
                    pk.vk.domain.coeff_from_vec(
                        values[lookup_index * len..(lookup_index + 1) * len]
                            .iter()
                            .copied()
                            .map(|value| unsafe { std::mem::transmute_copy(&value) })
                            .collect(),
                    )
                },
            )),
            resident_product_coefficients: Some(resident_products.clone()),
            resident_product_index: Some(lookup_index),
            resident_permuted_coefficients: lookup.resident_permuted_coefficients,
            resident_input_index: lookup.resident_input_index,
            resident_table_index: lookup.resident_table_index,
        })
        .collect();
    Ok((committed, commitments))
}

impl<F: WithSmallOrderMulGroup<3> + Ord + Hash + 'static> Argument<F> {
    #[allow(clippy::too_many_arguments)]
    fn compress_expressions<CS: PolynomialCommitmentScheme<F>>(
        &self,
        pk: &ProvingKey<F, CS>,
        domain: &EvaluationDomain<F>,
        theta: F,
        advice_values: &[Polynomial<F, LagrangeCoeff>],
        fixed_values: &[Polynomial<F, LagrangeCoeff>],
        instance_values: &[Polynomial<F, LagrangeCoeff>],
        challenges: &[F],
    ) -> ExpressionPair<F> {
        let compress = |expressions: &[Expression<F>]| {
            expressions
                .iter()
                .map(|expression| {
                    pk.vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        domain.n as usize,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * theta + &expression
                })
        };
        (
            compress(&self.input_expressions),
            compress(&self.table_expressions),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn construct_permuted<'a, CS: PolynomialCommitmentScheme<F>>(
        &self,
        pk: &ProvingKey<F, CS>,
        domain: &EvaluationDomain<F>,
        theta: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        compressed: Option<ExpressionPair<F>>,
        input_blinds: &[F],
        table_blinds: &[F],
        defer_permuted_coefficients: bool,
    ) -> Result<(Permuted<F>, Vec<Polynomial<F, LagrangeCoeff>>), Error>
    where
        F: FromUniformBytes<64>,
    {
        let (compressed_input_expression, compressed_table_expression) =
            compressed.unwrap_or_else(|| {
                self.compress_expressions::<CS>(
                    pk,
                    domain,
                    theta,
                    advice_values,
                    fixed_values,
                    instance_values,
                    challenges,
                )
            });

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
            pk,
            domain,
            &compressed_input_expression,
            &compressed_table_expression,
            input_blinds,
            table_blinds,
        )?;

        let commitment_values = vec![
            permuted_input_expression.clone(),
            permuted_table_expression.clone(),
        ];

        // Convert permuted input expression to coefficient form.
        let permuted_input_poly = if defer_permuted_coefficients {
            pk.vk.domain.empty_coeff()
        } else {
            pk.vk
                .domain
                .lagrange_to_coeff(permuted_input_expression.clone())
        };

        // Convert permuted table expression to coefficient form.
        let permuted_table_poly = if defer_permuted_coefficients {
            pk.vk.domain.empty_coeff()
        } else {
            pk.vk
                .domain
                .lagrange_to_coeff(permuted_table_expression.clone())
        };

        Ok((
            Permuted {
                compressed_input_expression,
                permuted_input_expression,
                permuted_input_poly,
                compressed_table_expression,
                permuted_table_expression,
                permuted_table_poly,
                #[cfg(feature = "gpu-blitzar")]
                resident_permuted_coefficients: None,
                #[cfg(feature = "gpu-blitzar")]
                resident_trace_lagrange: None,
                #[cfg(feature = "gpu-blitzar")]
                resident_input_index: None,
                #[cfg(feature = "gpu-blitzar")]
                resident_table_index: None,
            },
            commitment_values,
        ))
    }

    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table
    /// expressions [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... +
    ///   \theta A_{m-2} + A_{m-1} and S_compressed = \theta^{m-1} S_0 +
    ///   theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair()
    ///   helper, obtaining A' and S', and
    /// - constructs `Permuted<C>` struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    ///
    /// The `Permuted<C>` struct is used to update the Lookup, and is then
    /// returned.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn commit_permuted<
        'a,
        'params: 'a,
        CS: PolynomialCommitmentScheme<F>,
        T: Transcript,
    >(
        &self,
        pk: &ProvingKey<F, CS>,
        params: &'params CS::Parameters,
        domain: &EvaluationDomain<F>,
        theta: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        rng: &mut impl RngCore,
        transcript: &mut T,
    ) -> Result<Permuted<F>, Error>
    where
        F: FromUniformBytes<64>,
        CS::Commitment: Hashable<T::Hash>,
    {
        let blind_count = pk.vk.cs.blinding_factors() + 1;
        let input_blinds = (0..blind_count)
            .map(|_| F::random(&mut *rng))
            .collect::<Vec<_>>();
        let table_blinds = (0..blind_count)
            .map(|_| F::random(&mut *rng))
            .collect::<Vec<_>>();
        let (permuted, commitment_values) = self.construct_permuted::<CS>(
            pk,
            domain,
            theta,
            advice_values,
            fixed_values,
            instance_values,
            challenges,
            None,
            &input_blinds,
            &table_blinds,
            false,
        )?;
        for commitment in CS::commit_lagrange_batch(params, &commitment_values) {
            transcript.write(&commitment)?;
        }
        Ok(permuted)
    }

    #[allow(clippy::too_many_arguments, unsafe_code)]
    pub(crate) fn commit_permuted_batch<
        'a,
        'params: 'a,
        CS: PolynomialCommitmentScheme<F>,
        T: Transcript,
    >(
        lookups: &[Self],
        pk: &ProvingKey<F, CS>,
        params: &'params CS::Parameters,
        domain: &EvaluationDomain<F>,
        theta: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        #[cfg(feature = "gpu-blitzar")] resident_advice: Option<
            &midnight_sppark_msm::Bls12_381FrDeviceVec,
        >,
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        rng: &mut impl RngCore,
        transcript: &mut T,
    ) -> Result<Vec<Permuted<F>>, Error>
    where
        F: FromUniformBytes<64> + 'static,
        CS::Commitment: Hashable<T::Hash>,
        CS::Parameters: Sync,
    {
        #[cfg(feature = "gpu-blitzar")]
        let mut gpu_compressions = {
            let graphs = pk
                .ev
                .lookup_inputs
                .iter()
                .zip(pk.ev.lookup_tables.iter())
                .flat_map(|(input, table)| [input, table])
                .collect::<Vec<_>>();
            let enabled = matches!(
                std::env::var("MIDNIGHT_GPU_LOOKUP_COMPRESSION_BACKEND")
                    .unwrap_or_else(|_| "off".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on" | "gpu" | "cuda" | "sppark" | "batch"
            );
            let values = if enabled {
                crate::perf_timer::timed_count_detailed(
                    "proof_server.prover.plonk.lookup.compress.gpu_sppark_batch",
                    domain.n as usize * graphs.len(),
                    || {
                        crate::plonk::evaluation::try_evaluate_graphs_sppark(
                            &graphs,
                            domain,
                            fixed_values,
                            advice_values,
                            instance_values,
                            challenges,
                            F::ZERO,
                            F::ZERO,
                            theta,
                            F::ZERO,
                            F::ZERO,
                        )
                    },
                )
                .map_err(Error::Backend)?
            } else {
                None
            };
            values
                .map(|values| {
                    let mut values = values.into_iter();
                    (0..lookups.len())
                        .map(|_| {
                            (
                                values.next().expect("lookup input graph result"),
                                values.next().expect("lookup table graph result"),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .map(Vec::into_iter)
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let mut gpu_compressions: Option<std::vec::IntoIter<ExpressionPair<F>>> = None;

        let compressed = (0..lookups.len())
            .map(|_| {
                gpu_compressions
                    .as_mut()
                    .map(|values| values.next().expect("one graph pair per lookup"))
            })
            .collect::<Vec<_>>();
        let blind_count = pk.vk.cs.blinding_factors() + 1;
        let blinds = (0..lookups.len())
            .map(|_| {
                let input = (0..blind_count)
                    .map(|_| F::random(&mut *rng))
                    .collect::<Vec<_>>();
                let table = (0..blind_count)
                    .map(|_| F::random(&mut *rng))
                    .collect::<Vec<_>>();
                (input, table)
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "gpu-blitzar")]
        let retain_permuted_coefficients = resident_lookup_permuted_enabled::<F, CS>(pk)
            || resident_lookup_permuted_debug_enabled();
        #[cfg(feature = "gpu-blitzar")]
        let defer_permuted_coefficients =
            retain_permuted_coefficients && !resident_lookup_permuted_debug_enabled();
        #[cfg(not(feature = "gpu-blitzar"))]
        let defer_permuted_coefficients = false;
        #[cfg(feature = "gpu-blitzar")]
        let gpu_permute = resident_lookup_permute_backend_enabled::<F, CS>(pk);
        #[cfg(feature = "gpu-blitzar")]
        let constructed = if gpu_permute {
            construct_permuted_batch_sppark::<F, CS>(
                lookups,
                pk,
                domain,
                theta,
                advice_values,
                resident_advice,
                fixed_values,
                instance_values,
                challenges,
                &blinds,
            )?
        } else {
            lookups
                .par_iter()
                .zip(compressed.into_par_iter())
                .zip(blinds.into_par_iter())
                .map(|((lookup, compressed), (input_blinds, table_blinds))| {
                    lookup.construct_permuted::<CS>(
                        pk,
                        domain,
                        theta,
                        advice_values,
                        fixed_values,
                        instance_values,
                        challenges,
                        compressed,
                        &input_blinds,
                        &table_blinds,
                        defer_permuted_coefficients,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let constructed = lookups
            .par_iter()
            .zip(compressed.into_par_iter())
            .zip(blinds.into_par_iter())
            .map(|((lookup, compressed), (input_blinds, table_blinds))| {
                lookup.construct_permuted::<CS>(
                    pk,
                    domain,
                    theta,
                    advice_values,
                    fixed_values,
                    instance_values,
                    challenges,
                    compressed,
                    &input_blinds,
                    &table_blinds,
                    defer_permuted_coefficients,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut permuted = Vec::with_capacity(constructed.len());
        let mut commitment_values = Vec::with_capacity(constructed.len() * 2);
        for (lookup, mut values) in constructed {
            permuted.push(lookup);
            commitment_values.append(&mut values);
        }

        #[cfg(feature = "gpu-blitzar")]
        if retain_permuted_coefficients && !gpu_permute {
            retain_permuted_coefficients_sppark(&mut permuted, domain)?;
        }

        #[cfg(feature = "gpu-blitzar")]
        let commitments = if gpu_permute {
            let resident_trace = permuted
                .first()
                .and_then(|lookup| lookup.resident_trace_lagrange.as_deref())
                .filter(|resident| {
                    resident.len()
                        == lookups
                            .len()
                            .saturating_mul(domain.n as usize)
                            .saturating_mul(4)
                })
                .ok_or_else(|| {
                    Error::Backend(
                        "resident lookup permutation has no shared Lagrange arena".into(),
                    )
                })?;
            let table_indices = ((3 * lookups.len())..(4 * lookups.len()))
                .map(|index| u32::try_from(index).expect("lookup index fits u32"))
                .collect::<Vec<_>>();
            let resident_tables = resident_trace
                .gather_polys(domain.n as usize, &table_indices)
                .map_err(|error| Error::Backend(error.to_string()))?;
            let input_scalar_offset = 2 * lookups.len() * domain.n as usize;
            let execution_lane = midnight_sppark_msm::current_execution_lane();
            let commit_tables = || {
                let _lane = (execution_lane != 0)
                    .then(|| midnight_sppark_msm::bind_execution_lane(execution_lane))
                    .transpose()
                    .map_err(|error| error.to_string())?;
                CS::commit_lagrange_device_batch(
                    params,
                    &resident_tables,
                    domain.n as usize,
                    lookups.len(),
                )
            };
            let resident_ragged = matches!(
                std::env::var("MIDNIGHT_GPU_LOOKUP_RESIDENT_RAGGED")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on" | "resident" | "ragged"
            );
            let (input_commitments, table_commitments) = if resident_ragged {
                let (input_starts, input_nonzero_counts) = crate::perf_timer::timed_count_detailed(
                    "proof_server.prover.plonk.lookup.permuted_inputs.stats_download",
                    lookups.len(),
                    || {
                        resident_trace.nonzero_suffix_stats(
                            input_scalar_offset,
                            domain.n as usize,
                            lookups.len(),
                        )
                    },
                )
                .map_err(|error| Error::Backend(error.to_string()))?;
                rayon::join(
                    || {
                        let _lane = (execution_lane != 0)
                            .then(|| midnight_sppark_msm::bind_execution_lane(execution_lane))
                            .transpose()
                            .map_err(|error| error.to_string())?;
                        CS::commit_lagrange_device_suffix_batch(
                            params,
                            resident_trace,
                            input_scalar_offset,
                            &input_starts,
                            &input_nonzero_counts,
                            domain.n as usize,
                        )
                    },
                    &commit_tables,
                )
            } else {
                let (input_starts, downloaded) = crate::perf_timer::timed_count_detailed(
                    "proof_server.prover.plonk.lookup.permuted_inputs.suffix_download",
                    lookups.len(),
                    || {
                        resident_trace.download_nonzero_suffixes(
                            input_scalar_offset,
                            domain.n as usize,
                            lookups.len(),
                        )
                    },
                )
                .map_err(|error| Error::Backend(error.to_string()))?;
                let mut cursor = 0usize;
                let input_suffixes = input_starts
                    .iter()
                    .map(|&start| {
                        let len = domain.n as usize - start;
                        let end = cursor + len;
                        let values = downloaded[cursor..end]
                            .iter()
                            .copied()
                            .map(|value| unsafe { std::mem::transmute_copy(&value) })
                            .collect::<Vec<F>>();
                        cursor = end;
                        (start, values)
                    })
                    .collect::<Vec<_>>();
                if cursor != downloaded.len() {
                    return Err(Error::Backend(
                        "resident lookup suffix download returned the wrong length".into(),
                    ));
                }
                rayon::join(
                    || {
                        let _lane = (execution_lane != 0)
                            .then(|| midnight_sppark_msm::bind_execution_lane(execution_lane))
                            .transpose()
                            .map_err(|error| error.to_string())?;
                        CS::commit_lagrange_suffix_batch(params, &input_suffixes, domain.n as usize)
                    },
                    &commit_tables,
                )
            };
            let input_commitments =
                input_commitments.map_err(Error::Backend)?.ok_or_else(|| {
                    Error::Backend(
                        "commitment scheme does not support sparse lookup-input MSMs".into(),
                    )
                })?;
            let table_commitments =
                table_commitments.map_err(Error::Backend)?.ok_or_else(|| {
                    Error::Backend(
                        "commitment scheme does not support resident lookup-table MSMs".into(),
                    )
                })?;
            if input_commitments.len() != table_commitments.len()
                || input_commitments.len() != lookups.len()
            {
                return Err(Error::Backend(
                    "hybrid lookup commitment batch returned the wrong number of points".into(),
                ));
            }
            input_commitments
                .into_iter()
                .zip(table_commitments)
                .flat_map(|(input, table)| [input, table])
                .collect::<Vec<_>>()
        } else {
            CS::commit_lagrange_batch(params, &commitment_values)
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let commitments = CS::commit_lagrange_batch(params, &commitment_values);

        for commitment in commitments {
            transcript.write(&commitment)?;
        }

        Ok(permuted)
    }
}

impl<F: WithSmallOrderMulGroup<3>> Permuted<F> {
    fn construct_product<CS: PolynomialCommitmentScheme<F>>(
        self,
        pk: &ProvingKey<F, CS>,
        beta: F,
        gamma: F,
        blind_values: &[F],
    ) -> Result<(Committed<F>, Polynomial<F, LagrangeCoeff>), Error>
    where
        F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    {
        let blinding_factors = pk.vk.cs.blinding_factors();
        if blind_values.len() != blinding_factors {
            return Err(Error::Backend(
                "lookup blind count does not match the argument".to_string(),
            ));
        }
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... +
        // \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... +
        //              \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.
        let mut lookup_product = vec![F::ZERO; pk.vk.n() as usize];
        // Denominator uses the permuted input expression and permuted table expression
        parallelize(&mut lookup_product, |lookup_product, start| {
            for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                .iter_mut()
                .zip(self.permuted_input_expression[start..].iter())
                .zip(self.permuted_table_expression[start..].iter())
            {
                *lookup_product = (beta + permuted_input_value) * &(gamma + permuted_table_value);
            }
        });

        // Batch invert to obtain the denominators for the lookup product
        // polynomials
        crate::perf_timer::timed_count(
            "proof_server.prover.plonk.lookup.product_batch_invert",
            lookup_product.len(),
            || {
                #[cfg(feature = "gpu-blitzar")]
                if midnight_curves::gpu_sppark::try_bls12_381_fr_invert_assign(&mut lookup_product)
                {
                    return;
                }

                lookup_product.iter_mut().batch_invert();
            },
        );

        // Finish the computation of the entire fraction by computing the numerators
        // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta
        // a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta
        //   s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        parallelize(&mut lookup_product, |product, start| {
            for (i, product) in product.iter_mut().enumerate() {
                let i = i + start;

                *product *= &(self.compressed_input_expression[i] + &beta);
                *product *= &(self.compressed_table_expression[i] + &gamma);
            }
        });

        // The product vector is a vector of products of fractions of the form
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... +
        // \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... +
        //              \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where there are m input expressions and m table expressions,
        // a_j(\omega^i) is the jth input expression in this lookup,
        // a'j(\omega^i) is the permuted input expression,
        // s_j(\omega^i) is the jth table expression in this lookup,
        // s'(\omega^i) is the permuted table expression,
        // and i is the ith row of the expression.

        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let z = iter::once(F::ONE)
            .chain(lookup_product)
            .scan(F::ONE, |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a boolean (and ideally 1, else soundness is broken)
            .take(pk.vk.n() as usize - blinding_factors)
            // Chain random blinding factors.
            .chain(blind_values.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(z.len(), pk.vk.n() as usize);
        let z = pk.vk.domain.lagrange_from_vec(z);

        #[cfg(debug_assertions)]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange representation, check that product is correctly constructed
            let u = (pk.vk.n() as usize) - (blinding_factors + 1);

            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], F::ONE);

            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X)
            //   + ... + s_{m-1}(X) + \gamma)
            for i in 0..u {
                let mut left = z[i + 1];
                let permuted_input_value = &self.permuted_input_expression[i];

                let permuted_table_value = &self.permuted_table_expression[i];

                left *= &(beta + permuted_input_value);
                left *= &(gamma + permuted_table_value);

                let mut right = z[i];
                let mut input_term = self.compressed_input_expression[i];
                let mut table_term = self.compressed_table_expression[i];

                input_term += &(beta);
                table_term += &(gamma);
                right *= &(input_term * &table_term);

                assert_eq!(left, right);
            }

            // l_last(X) * (z(X)^2 - z(X)) = 0
            // Assertion will fail only when soundness is broken, in which
            // case this z[u] value will be zero. (bad!)
            assert_eq!(z[u], F::ONE);
        }

        let product_values = z.clone();
        let z = pk.vk.domain.lagrange_to_coeff(z);

        Ok((
            Committed::<F> {
                permuted_input_poly: Box::new(self.permuted_input_poly),
                permuted_table_poly: Box::new(self.permuted_table_poly),
                product_poly: Box::new(z),
                #[cfg(feature = "gpu-blitzar")]
                resident_product_coefficients: None,
                #[cfg(feature = "gpu-blitzar")]
                resident_product_index: None,
                #[cfg(feature = "gpu-blitzar")]
                resident_permuted_coefficients: self.resident_permuted_coefficients,
                #[cfg(feature = "gpu-blitzar")]
                resident_input_index: self.resident_input_index,
                #[cfg(feature = "gpu-blitzar")]
                resident_table_index: self.resident_table_index,
            },
            product_values,
        ))
    }

    /// Given a Lookup with input expressions, table expressions, and the
    /// permuted input expression and permuted table expression, this method
    /// constructs the grand product polynomial over the lookup. The grand
    /// product polynomial is used to populate the `Product<C>` struct. The
    /// `Product<C>` struct is added to the Lookup and finally returned by the
    /// method.
    #[allow(dead_code)]
    pub(crate) fn commit_product<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        pk: &ProvingKey<F, CS>,
        params: &CS::Parameters,
        beta: F,
        gamma: F,
        rng: &mut (impl RngCore + CryptoRng),
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
        CS::Commitment: Hashable<T::Hash>,
    {
        let blind_values = (0..pk.vk.cs.blinding_factors())
            .map(|_| F::random(&mut *rng))
            .collect::<Vec<_>>();
        let (committed, product_values) =
            self.construct_product::<CS>(pk, beta, gamma, &blind_values)?;
        let product_commitment = CS::commit_lagrange(params, &product_values);
        transcript.write(&product_commitment)?;
        Ok(committed)
    }

    pub(crate) fn commit_product_batch<CS: PolynomialCommitmentScheme<F>>(
        lookups: impl IntoIterator<Item = Self>,
        pk: &ProvingKey<F, CS>,
        params: &CS::Parameters,
        beta: F,
        gamma: F,
        blind_values: &[F],
    ) -> Result<(Vec<Committed<F>>, Vec<CS::Commitment>), Error>
    where
        F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    {
        let lookups = lookups.into_iter().collect::<Vec<_>>();
        #[cfg(feature = "gpu-blitzar")]
        {
            let resident_count = lookups
                .iter()
                .filter(|lookup| lookup.resident_trace_lagrange.is_some())
                .count();
            if resident_count != 0 && resident_count != lookups.len() {
                return Err(Error::Backend(
                    "lookup batch mixes resident and host-owned traces".to_string(),
                ));
            }
            let resident_trace = resident_count != 0;
            if resident_trace
                && matches!(
                    std::env::var("MIDNIGHT_GPU_LOOKUP_PRODUCT_BACKEND")
                        .unwrap_or_else(|_| "auto".to_string())
                        .to_ascii_lowercase()
                        .as_str(),
                    "0" | "false" | "no" | "off" | "none"
                )
            {
                return Err(Error::Backend(
                    "resident lookup trace requires the CUDA lookup-product backend".to_string(),
                ));
            }
            if (resident_trace || lookup_product_backend_enabled::<F>(pk.vk.n() as usize))
                && !lookups.is_empty()
            {
                return commit_product_batch_sppark(lookups, pk, params, beta, gamma, blind_values);
            }
        }

        let blinding_factors = pk.vk.cs.blinding_factors();
        if blind_values.len() != lookups.len() * blinding_factors {
            return Err(Error::Backend(
                "lookup blind count does not match the batch".to_string(),
            ));
        }

        let mut committed = Vec::new();
        let mut product_values = Vec::new();

        for (lookup, blinds) in lookups
            .into_iter()
            .zip(blind_values.chunks_exact(blinding_factors))
        {
            let (lookup, values) = lookup.construct_product::<CS>(pk, beta, gamma, blinds)?;
            committed.push(lookup);
            product_values.push(values);
        }

        let commitments = CS::commit_lagrange_batch(params, &product_values);
        Ok((committed, commitments))
    }
}

impl<F: WithSmallOrderMulGroup<3> + 'static> Committed<F> {
    pub(crate) fn evaluate_batch<T: Transcript, CS: PolynomialCommitmentScheme<F>>(
        committed: Vec<Self>,
        pk: &ProvingKey<F, CS>,
        x: F,
        transcript: &mut T,
    ) -> Result<Vec<Evaluated<F>>, Error>
    where
        F: Hashable<T::Hash>,
    {
        if committed.is_empty() {
            return Ok(Vec::new());
        }
        let domain = &pk.vk.domain;
        let x_inv = domain.rotate_omega(x, Rotation::prev());
        let x_next = domain.rotate_omega(x, Rotation::next());

        #[cfg(feature = "gpu-blitzar")]
        let resident_products = committed[0]
            .resident_product_coefficients
            .as_deref()
            .filter(|resident| {
                committed.iter().all(|lookup| {
                    lookup
                        .resident_product_coefficients
                        .as_ref()
                        .is_some_and(|candidate| {
                            Arc::ptr_eq(
                                candidate,
                                &committed[0].resident_product_coefficients.as_ref().unwrap(),
                            )
                        })
                }) && resident.len() % (domain.n as usize) == 0
            });
        #[cfg(feature = "gpu-blitzar")]
        if committed
            .iter()
            .any(|lookup| lookup.resident_product_coefficients.is_some())
            && resident_products.is_none()
        {
            return Err(Error::Backend(
                "lookup product residency is not shared across the batch".into(),
            ));
        }
        #[cfg(feature = "gpu-blitzar")]
        let resident_product_evals = resident_products
            .map(|resident| {
                let count = resident.len() / (domain.n as usize);
                Ok::<_, Error>((
                    crate::plonk::evaluate_resident_polynomials(
                        resident,
                        domain.n as usize,
                        count,
                        x,
                    )?,
                    crate::plonk::evaluate_resident_polynomials(
                        resident,
                        domain.n as usize,
                        count,
                        x_next,
                    )?,
                ))
            })
            .transpose()?;

        #[cfg(feature = "gpu-blitzar")]
        let resident_permuted = committed[0]
            .resident_permuted_coefficients
            .as_deref()
            .filter(|resident| {
                committed.iter().all(|lookup| {
                    lookup
                        .resident_permuted_coefficients
                        .as_ref()
                        .is_some_and(|candidate| {
                            Arc::ptr_eq(
                                candidate,
                                &committed[0]
                                    .resident_permuted_coefficients
                                    .as_ref()
                                    .unwrap(),
                            )
                        })
                }) && resident.len() % (domain.n as usize) == 0
            });
        #[cfg(feature = "gpu-blitzar")]
        if committed
            .iter()
            .any(|lookup| lookup.resident_permuted_coefficients.is_some())
            && resident_permuted.is_none()
        {
            return Err(Error::Backend(
                "lookup permutation residency is not shared across the batch".into(),
            ));
        }
        #[cfg(feature = "gpu-blitzar")]
        let resident_permuted_evals = resident_permuted
            .map(|resident| {
                let count = resident.len() / (domain.n as usize);
                Ok::<_, Error>((
                    crate::plonk::evaluate_resident_polynomials(
                        resident,
                        domain.n as usize,
                        count,
                        x,
                    )?,
                    crate::plonk::evaluate_resident_polynomials(
                        resident,
                        domain.n as usize,
                        count,
                        x_inv,
                    )?,
                ))
            })
            .transpose()?;

        let mut evaluated = Vec::with_capacity(committed.len());
        for lookup in committed {
            #[cfg(feature = "gpu-blitzar")]
            let (product_eval, product_next_eval) =
                if let Some((at_x, at_next)) = resident_product_evals.as_ref() {
                    let index = lookup.resident_product_index.ok_or_else(|| {
                        Error::Backend("resident lookup product has no batch index".into())
                    })?;
                    (
                        *at_x.get(index).ok_or(Error::BoundsFailure)?,
                        *at_next.get(index).ok_or(Error::BoundsFailure)?,
                    )
                } else {
                    (
                        eval_polynomial(&lookup.product_poly, x),
                        eval_polynomial(&lookup.product_poly, x_next),
                    )
                };
            #[cfg(not(feature = "gpu-blitzar"))]
            let (product_eval, product_next_eval) = (
                eval_polynomial(&lookup.product_poly, x),
                eval_polynomial(&lookup.product_poly, x_next),
            );

            #[cfg(feature = "gpu-blitzar")]
            let (permuted_input_eval, permuted_input_inv_eval, permuted_table_eval) =
                if let Some((at_x, at_inv)) = resident_permuted_evals.as_ref() {
                    let input_index = lookup.resident_input_index.ok_or_else(|| {
                        Error::Backend("resident lookup input has no batch index".into())
                    })?;
                    let table_index = lookup.resident_table_index.ok_or_else(|| {
                        Error::Backend("resident lookup table has no batch index".into())
                    })?;
                    (
                        *at_x.get(input_index).ok_or(Error::BoundsFailure)?,
                        *at_inv.get(input_index).ok_or(Error::BoundsFailure)?,
                        *at_x.get(table_index).ok_or(Error::BoundsFailure)?,
                    )
                } else {
                    (
                        eval_polynomial(&lookup.permuted_input_poly, x),
                        eval_polynomial(&lookup.permuted_input_poly, x_inv),
                        eval_polynomial(&lookup.permuted_table_poly, x),
                    )
                };
            #[cfg(not(feature = "gpu-blitzar"))]
            let (permuted_input_eval, permuted_input_inv_eval, permuted_table_eval) = (
                eval_polynomial(&lookup.permuted_input_poly, x),
                eval_polynomial(&lookup.permuted_input_poly, x_inv),
                eval_polynomial(&lookup.permuted_table_poly, x),
            );

            for eval in iter::empty()
                .chain(Some(product_eval))
                .chain(Some(product_next_eval))
                .chain(Some(permuted_input_eval))
                .chain(Some(permuted_input_inv_eval))
                .chain(Some(permuted_table_eval))
            {
                transcript.write(&eval)?;
            }
            evaluated.push(Evaluated {
                constructed: lookup,
            });
        }

        Ok(evaluated)
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluated<F> {
    pub(crate) fn open<'a, CS: PolynomialCommitmentScheme<F>>(
        &'a self,
        pk: &'a ProvingKey<F, CS>,
        x: F,
    ) -> impl Iterator<Item = ProverQuery<'a, F>> + Clone {
        let x_inv = pk.vk.domain.rotate_omega(x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.product_poly,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.permuted_table_poly,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.product_poly,
            }))
    }
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row that has the
///   corresponding value in S'.
///
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<F, CS: PolynomialCommitmentScheme<F>>(
    pk: &ProvingKey<F, CS>,
    domain: &EvaluationDomain<F>,
    input_expression: &Polynomial<F, LagrangeCoeff>,
    table_expression: &Polynomial<F, LagrangeCoeff>,
    input_blinds: &[F],
    table_blinds: &[F],
) -> Result<ExpressionPair<F>, Error>
where
    F: WithSmallOrderMulGroup<3> + Hash + Ord + FromUniformBytes<64>,
{
    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = pk.vk.n() as usize - (blinding_factors + 1);
    if input_blinds.len() != blinding_factors + 1 || table_blinds.len() != blinding_factors + 1 {
        return Err(Error::BoundsFailure);
    }

    let mut permuted_input_expression: Vec<F> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);
    #[cfg(feature = "gpu-blitzar")]
    let sorted_on_gpu = crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.permute.sort.gpu_sppark",
        permuted_input_expression.len(),
        || {
            midnight_curves::gpu_sppark::try_bls12_381_fr_sort_canonical(
                &mut permuted_input_expression,
            )
        },
    );
    #[cfg(not(feature = "gpu-blitzar"))]
    let sorted_on_gpu = false;
    if !sorted_on_gpu {
        permuted_input_expression.sort();
    }

    let mut sorted_table = table_expression
        .iter()
        .take(usable_rows)
        .copied()
        .collect::<Vec<_>>();
    crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.permute.table_sort",
        sorted_table.len(),
        || sorted_table.sort(),
    );
    let mut permuted_table_coeffs = vec![F::ZERO; usable_rows];
    crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.lookup.permute.multiset_merge",
        usable_rows,
        || {
            let mut repeated_input_rows = Vec::new();
            let mut unique_inputs = Vec::new();
            for (row, input_value) in permuted_input_expression.iter().copied().enumerate() {
                if row == 0 || input_value != permuted_input_expression[row - 1] {
                    permuted_table_coeffs[row] = input_value;
                    unique_inputs.push(input_value);
                } else {
                    repeated_input_rows.push(row);
                }
            }

            let mut removed = vec![false; sorted_table.len()];
            let mut table_index = 0usize;
            for input_value in unique_inputs {
                while table_index < sorted_table.len() && sorted_table[table_index] < input_value {
                    table_index += 1;
                }
                if table_index == sorted_table.len() || sorted_table[table_index] != input_value {
                    return Err(Error::ConstraintSystemFailure);
                }
                removed[table_index] = true;
                table_index += 1;
            }

            let leftovers = sorted_table
                .iter()
                .copied()
                .zip(removed)
                .filter_map(|(value, removed)| (!removed).then_some(value));
            let mut repeated_input_rows = repeated_input_rows.into_iter().rev();
            for value in leftovers {
                let row = repeated_input_rows
                    .next()
                    .ok_or(Error::ConstraintSystemFailure)?;
                permuted_table_coeffs[row] = value;
            }
            if repeated_input_rows.next().is_some() {
                return Err(Error::ConstraintSystemFailure);
            }
            Ok::<(), Error>(())
        },
    )?;

    permuted_input_expression.extend_from_slice(input_blinds);
    permuted_table_coeffs.extend_from_slice(table_blinds);
    assert_eq!(permuted_input_expression.len(), pk.vk.n() as usize);
    assert_eq!(permuted_table_coeffs.len(), pk.vk.n() as usize);

    #[cfg(debug_assertions)]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}
