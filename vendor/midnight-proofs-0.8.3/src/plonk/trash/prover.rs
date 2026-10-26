use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
#[cfg(feature = "gpu-blitzar")]
use std::sync::Arc;

use super::{super::Error, Argument};
use crate::{
    plonk::evaluation::{GraphEvaluator, evaluate},
    poly::{
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery,
        commitment::PolynomialCommitmentScheme,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::eval_polynomial,
};

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) trash_poly: Polynomial<F, Coeff>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_coefficients: Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_index: Option<usize>,
}

pub(crate) struct Evaluated<F: PrimeField>(Committed<F>);

impl<F: WithSmallOrderMulGroup<3> + Ord> Argument<F> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_lagrange(
        &self,
        domain: &EvaluationDomain<F>,
        trash_challenge: F,
        advice_values: &[Polynomial<F, LagrangeCoeff>],
        fixed_values: &[Polynomial<F, LagrangeCoeff>],
        instance_values: &[Polynomial<F, LagrangeCoeff>],
        challenges: &[F],
    ) -> Polynomial<F, LagrangeCoeff> {
        self.constraint_expressions
            .iter()
            .map(|expression| {
                domain.lagrange_from_vec(evaluate(
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
                acc * trash_challenge + &expression
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit<'a, 'params: 'a, CS, T>(
        &self,
        params: &'params CS::Parameters,
        domain: &EvaluationDomain<F>,
        graph: &GraphEvaluator<F>,
        trash_challenge: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        F: FromUniformBytes<64>,
        CS: PolynomialCommitmentScheme<F>,
        CS::Commitment: Hashable<T::Hash>,
        T: Transcript,
    {
        let evaluate_cpu = || {
            self.evaluate_lagrange(
                domain,
                trash_challenge,
                advice_values,
                fixed_values,
                instance_values,
                challenges,
            )
        };
        #[cfg(feature = "gpu-blitzar")]
        let compressed_expression = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.trash.evaluate.gpu_sppark",
            domain.n as usize,
            || {
                graph.try_evaluate_sppark(
                    domain,
                    fixed_values,
                    advice_values,
                    instance_values,
                    challenges,
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                    trash_challenge,
                    F::ZERO,
                )
            },
        )
        .map_err(Error::Backend)?
        .unwrap_or_else(evaluate_cpu);
        #[cfg(not(feature = "gpu-blitzar"))]
        let compressed_expression = {
            let _ = graph;
            evaluate_cpu()
        };

        let trash_commitment = CS::commit_lagrange(params, &compressed_expression);
        let trash_poly = domain.lagrange_to_coeff(compressed_expression);

        // Hash permuted input commitment
        transcript.write(&trash_commitment)?;

        Ok(Committed {
            trash_poly,
            #[cfg(feature = "gpu-blitzar")]
            resident_coefficients: None,
            #[cfg(feature = "gpu-blitzar")]
            resident_index: None,
        })
    }
}

#[cfg(feature = "gpu-blitzar")]
pub(crate) fn prepare_resident_batch<
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    params: &CS::Parameters,
    domain: &EvaluationDomain<F>,
    mut lagrange: midnight_sppark_msm::Bls12_381FrDeviceVec,
    count: usize,
) -> Result<(Vec<Committed<F>>, Vec<CS::Commitment>), Error> {
    let len = domain.n as usize;
    if count == 0 || lagrange.len() != len.saturating_mul(count) {
        return Err(Error::Backend(
            "resident trash polynomial layout does not match the proving key".into(),
        ));
    }
    let debug = matches!(
        std::env::var("MIDNIGHT_GPU_RESIDENT_TRACE_TRASH")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "debug" | "validate" | "check"
    );
    let debug_lagrange = debug
        .then(|| lagrange.download())
        .transpose()
        .map_err(|error| Error::Backend(error.to_string()))?;
    let commitments = CS::commit_lagrange_device_scrambled_batch(params, &lagrange, len, count)
        .map_err(Error::Backend)?
        .ok_or_else(|| Error::Backend("resident trash commitment is unsupported".into()))?;
    if let Some(values) = debug_lagrange.as_ref() {
        let expected_polys = values
            .chunks_exact(len)
            .map(|values| {
                domain.lagrange_from_vec(
                    values
                        .iter()
                        .copied()
                        .map(super::super::evaluation::field_from_raw)
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let expected_commitments = CS::commit_lagrange_batch(params, &expected_polys);
        if commitments != expected_commitments {
            let mismatch = commitments
                .iter()
                .zip(&expected_commitments)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(commitments.len().min(expected_commitments.len()));
            return Err(Error::Backend(format!(
                "resident trash commitment mismatch at {mismatch}"
            )));
        }
    }
    let one = super::super::evaluation::raw_from_field(F::ONE);
    lagrange
        .ntt_zeta_batch(len, count, true, one, one, one)
        .map_err(|error| Error::Backend(error.to_string()))?;
    if let Some(values) = debug_lagrange {
        let actual = lagrange
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?;
        let expected = values
            .chunks_exact(len)
            .flat_map(|values| {
                domain
                    .lagrange_to_coeff(
                        domain.lagrange_from_vec(
                            values
                                .iter()
                                .copied()
                                .map(super::super::evaluation::field_from_raw)
                                .collect(),
                        ),
                    )
                    .values
                    .into_iter()
                    .map(super::super::evaluation::raw_from_field)
            })
            .collect::<Vec<_>>();
        if actual != expected {
            let mismatch = actual
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(actual.len().min(expected.len()));
            return Err(Error::Backend(format!(
                "resident trash coefficient mismatch at {mismatch}"
            )));
        }
        let coefficients = Arc::new(lagrange);
        return Ok((
            actual
                .chunks_exact(len)
                .enumerate()
                .map(|(index, values)| Committed {
                    trash_poly: domain.coeff_from_vec(
                        values
                            .iter()
                            .copied()
                            .map(super::super::evaluation::field_from_raw)
                            .collect(),
                    ),
                    resident_coefficients: Some(coefficients.clone()),
                    resident_index: Some(index),
                })
                .collect(),
            commitments,
        ));
    }
    let coefficients = Arc::new(lagrange);
    Ok((
        (0..count)
            .map(|index| Committed {
                trash_poly: domain.empty_coeff(),
                resident_coefficients: Some(coefficients.clone()),
                resident_index: Some(index),
            })
            .collect(),
        commitments,
    ))
}

#[cfg(feature = "gpu-blitzar")]
pub(crate) fn commit_resident_batch<
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
>(
    params: &CS::Parameters,
    domain: &EvaluationDomain<F>,
    lagrange: midnight_sppark_msm::Bls12_381FrDeviceVec,
    count: usize,
    transcript: &mut T,
) -> Result<Vec<Committed<F>>, Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: Hashable<T::Hash>,
{
    let (committed, commitments) =
        prepare_resident_batch::<F, CS>(params, domain, lagrange, count)?;
    for commitment in commitments {
        transcript.write(&commitment)?;
    }
    Ok(committed)
}

impl<F: WithSmallOrderMulGroup<3> + 'static> Committed<F> {
    pub(crate) fn evaluate<T>(self, x: F, transcript: &mut T) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
        T: Transcript,
    {
        #[cfg(feature = "gpu-blitzar")]
        let trash_eval = if let Some(coefficients) = self.resident_coefficients.as_deref() {
            let count = coefficients.len() / self.trash_poly.len();
            let index = self
                .resident_index
                .ok_or_else(|| Error::Backend("resident trash polynomial has no index".into()))?;
            *crate::plonk::evaluate_resident_polynomials(
                coefficients,
                self.trash_poly.len(),
                count,
                x,
            )?
            .get(index)
            .ok_or(Error::BoundsFailure)?
        } else {
            eval_polynomial(&self.trash_poly, x)
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let trash_eval = eval_polynomial(&self.trash_poly, x);
        transcript.write(&trash_eval)?;

        Ok(Evaluated(self))
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluated<F> {
    pub(crate) fn open(&self, x: F) -> impl Iterator<Item = ProverQuery<'_, F>> + Clone {
        vec![ProverQuery {
            point: x,
            poly: &self.0.trash_poly,
        }]
        .into_iter()
    }
}
