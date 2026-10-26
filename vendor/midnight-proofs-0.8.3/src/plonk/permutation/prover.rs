use std::iter::{self, ExactSizeIterator};
#[cfg(feature = "gpu-blitzar")]
use std::{any::TypeId, sync::Arc};

use super::{super::circuit::Any, Argument, ProvingKey};
use crate::{
    plonk::{self, Error},
    poly::{
        Coeff, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
        commitment::PolynomialCommitmentScheme,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::BatchInvert;

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct CommittedSet<F: PrimeField> {
    pub(crate) permutation_product_poly: Polynomial<F, Coeff>,
}

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) sets: Vec<CommittedSet<F>>,
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) resident_coefficients: Option<Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
}

pub(crate) struct Evaluated<F: PrimeField> {
    constructed: Committed<F>,
}

fn shifted_prefix_product<F: PrimeField>(values: Vec<F>, initial: F) -> Vec<F> {
    let mut z = Vec::with_capacity(values.len());
    z.push(initial);
    for row in 1..values.len() {
        z.push(z[row - 1] * values[row - 1]);
    }
    z
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code, clippy::too_many_arguments)]
fn try_commit_sppark<F, CS>(
    argument: &Argument,
    params: &CS::Parameters,
    pk: &plonk::ProvingKey<F, CS>,
    pkey: &ProvingKey<F>,
    advice: &[Polynomial<F, LagrangeCoeff>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    instance: &[Polynomial<F, LagrangeCoeff>],
    beta: F,
    gamma: F,
    blind_values: &[F],
) -> Result<Option<(Committed<F>, Vec<CS::Commitment>)>, Error>
where
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    let backend = std::env::var("MIDNIGHT_GPU_PERMUTATION_BACKEND")
        .unwrap_or_else(|_| "scrambled-sppark".to_string())
        .to_ascii_lowercase();
    if !midnight_curves::gpu_sppark::enabled()
        || TypeId::of::<F>() != TypeId::of::<midnight_curves::Fq>()
        || argument.columns.is_empty()
        || !matches!(
            backend.as_str(),
            "1" | "true"
                | "yes"
                | "on"
                | "gpu"
                | "cuda"
                | "sppark"
                | "device"
                | "ec-gpu"
                | "ec_gpu"
                | "ecgpu"
                | "scrambled-sppark"
                | "scrambled_sppark"
                | "scrambled"
                | "masked-sppark"
                | "masked_sppark"
                | "masked"
        )
    {
        return Ok(None);
    }

    let domain = &pk.vk.domain;
    let len = domain.n as usize;
    let chunk_len = pk
        .vk
        .cs_degree
        .checked_sub(2)
        .ok_or_else(|| Error::Backend("permutation degree is below three".to_string()))?;
    let set_count = argument.columns.len().div_ceil(chunk_len);
    let blinding_factors = pk.vk.cs.blinding_factors();
    if pkey.permutations.len() != argument.columns.len() {
        return Err(Error::Backend(
            "permutation key column count does not match the argument".to_string(),
        ));
    }

    let raw = |value: F| unsafe { std::mem::transmute_copy(&value) };
    let mut columns = Vec::with_capacity(argument.columns.len() * len);
    for column in &argument.columns {
        let values = match column.column_type() {
            Any::Advice(_) => advice,
            Any::Fixed => fixed,
            Any::Instance => instance,
        };
        let polynomial = values.get(column.index()).ok_or(Error::BoundsFailure)?;
        if polynomial.len() != len {
            return Err(Error::BoundsFailure);
        }
        columns.extend(polynomial.iter().copied().map(raw));
    }
    let permutations = pkey
        .permutations
        .iter()
        .flat_map(|polynomial| polynomial.iter().copied().map(raw))
        .collect::<Vec<_>>();
    let mut delta = F::ONE;
    let delta_powers = (0..argument.columns.len())
        .map(|_| {
            let current = delta;
            delta *= F::DELTA;
            raw(current)
        })
        .collect::<Vec<_>>();
    if blind_values.len() != set_count * blinding_factors {
        return Err(Error::Backend(
            "permutation blind count does not match the argument".to_string(),
        ));
    }
    let blind_values = blind_values.iter().copied().map(raw).collect::<Vec<_>>();

    let mut products = crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.permutation.products.gpu_sppark_device",
        len * set_count,
        || {
            midnight_sppark_msm::bls12_381_fr_permutation_products(
                &columns,
                &permutations,
                len,
                chunk_len,
                raw(beta),
                raw(gamma),
                raw(domain.get_omega()),
                &delta_powers,
                &blind_values,
                blinding_factors,
            )
        },
    )
    .map_err(|error| Error::Backend(error.to_string()))?;

    let commitments = if matches!(
        backend.as_str(),
        "scrambled-sppark"
            | "scrambled_sppark"
            | "scrambled"
            | "masked-sppark"
            | "masked_sppark"
            | "masked"
    ) {
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.permutation.commit.gpu_sppark_device_scrambled",
            len * set_count,
            || CS::commit_lagrange_device_scrambled_batch(params, &products, len, set_count),
        )
        .map_err(Error::Backend)?
        .ok_or_else(|| {
            Error::Backend(
                "scrambled sppark permutation commitment is unsupported by this PCS".to_string(),
            )
        })?
    } else {
        let product_values = products
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?
            .chunks_exact(len)
            .map(|values| {
                domain.lagrange_from_vec(
                    values
                        .iter()
                        .copied()
                        .map(|value| unsafe { std::mem::transmute_copy(&value) })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        if matches!(backend.as_str(), "ec-gpu" | "ec_gpu" | "ecgpu") {
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.permutation.commit.ec_gpu",
                len * set_count,
                || CS::commit_lagrange_ec_gpu_batch(params, &product_values),
            )
            .map_err(Error::Backend)?
            .ok_or_else(|| {
                Error::Backend(
                    "ec-gpu permutation commitment is unsupported by this PCS".to_string(),
                )
            })?
        } else {
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.permutation.commit.gpu_sppark_staged",
                len * set_count,
                || {
                    product_values
                        .iter()
                        .map(|polynomial| CS::commit_lagrange(params, polynomial))
                        .collect::<Vec<_>>()
                },
            )
        }
    };
    let one = raw(F::ONE);
    products
        .ntt_zeta_batch(len, set_count, true, one, one, one)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let sets = if crate::plonk::resident_argument_without_host_mirror(
        crate::plonk::ResidentArgumentKind::Permutation,
        len,
        crate::plonk::prover::resident_finalise_supported(pk),
    ) {
        (0..set_count)
            .map(|_| CommittedSet {
                permutation_product_poly: domain.empty_coeff(),
            })
            .collect()
    } else {
        products
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?
            .chunks_exact(len)
            .map(|values| CommittedSet {
                permutation_product_poly: domain.coeff_from_vec(
                    values
                        .iter()
                        .copied()
                        .map(|value| unsafe { std::mem::transmute_copy(&value) })
                        .collect(),
                ),
            })
            .collect()
    };
    Ok(Some((
        Committed {
            sets,
            resident_coefficients: Some(Arc::new(products)),
        },
        commitments,
    )))
}

#[cfg(feature = "gpu-blitzar")]
#[allow(unsafe_code)]
pub(crate) fn commit_resident_products<
    F: WithSmallOrderMulGroup<3> + 'static,
    CS: PolynomialCommitmentScheme<F>,
>(
    params: &CS::Parameters,
    pk: &plonk::ProvingKey<F, CS>,
    mut products: midnight_sppark_msm::Bls12_381FrDeviceVec,
) -> Result<(Committed<F>, Vec<CS::Commitment>), Error> {
    let domain = &pk.vk.domain;
    let len = domain.n as usize;
    if len == 0 || products.len() % len != 0 {
        return Err(Error::Backend(
            "resident permutation product layout does not match the proving key".into(),
        ));
    }
    let set_count = products.len() / len;
    let commitments = crate::perf_timer::timed_count_detailed(
        "proof_server.prover.plonk.permutation.commit.resident_gpu_sppark",
        products.len(),
        || CS::commit_lagrange_device_scrambled_batch(params, &products, len, set_count),
    )
    .map_err(Error::Backend)?
    .ok_or_else(|| Error::Backend("resident permutation commitment is unsupported".into()))?;
    let raw = |value: F| unsafe { std::mem::transmute_copy(&value) };
    let one = raw(F::ONE);
    products
        .ntt_zeta_batch(len, set_count, true, one, one, one)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let sets = if crate::plonk::resident_argument_without_host_mirror(
        crate::plonk::ResidentArgumentKind::Permutation,
        len,
        crate::plonk::prover::resident_finalise_supported(pk),
    ) {
        (0..set_count)
            .map(|_| CommittedSet {
                permutation_product_poly: domain.empty_coeff(),
            })
            .collect()
    } else {
        products
            .download()
            .map_err(|error| Error::Backend(error.to_string()))?
            .chunks_exact(len)
            .map(|values| CommittedSet {
                permutation_product_poly: domain.coeff_from_vec(
                    values
                        .iter()
                        .copied()
                        .map(|value| unsafe { std::mem::transmute_copy(&value) })
                        .collect(),
                ),
            })
            .collect()
    };
    Ok((
        Committed {
            sets,
            resident_coefficients: Some(Arc::new(products)),
        },
        commitments,
    ))
}

impl Argument {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
        &self,
        params: &CS::Parameters,
        pk: &plonk::ProvingKey<F, CS>,
        pkey: &ProvingKey<F>,
        advice: &[Polynomial<F, LagrangeCoeff>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        beta: F,
        gamma: F,
        blind_values: &[F],
    ) -> Result<(Committed<F>, Vec<CS::Commitment>), Error> {
        #[cfg(feature = "gpu-blitzar")]
        if let Some(committed) = try_commit_sppark(
            self,
            params,
            pk,
            pkey,
            advice,
            fixed,
            instance,
            beta,
            gamma,
            blind_values,
        )? {
            return Ok(committed);
        }

        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs_degree >= 3);
        let chunk_len = pk.vk.cs_degree - 2;
        let blinding_factors = pk.vk.cs.blinding_factors();

        // Each column gets its own delta power.
        let mut deltaomega = F::ONE;

        // Track the "last" value from the previous column set
        let mut last_z = F::ONE;

        let mut sets = vec![];
        let mut product_values = vec![];
        let mut blind_values = blind_values.iter().copied();

        for (columns, permutations) in self
            .columns
            .chunks(chunk_len)
            .zip(pkey.permutations.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            let mut modified_values = vec![F::ONE; domain.n as usize];

            crate::perf_timer::timed_count(
                "proof_server.prover.plonk.permutation.denominator",
                domain.n as usize * columns.len(),
                || {
                    // Iterate over each column of the permutation
                    for (&column, permuted_column_values) in columns.iter().zip(permutations.iter())
                    {
                        let values = match column.column_type() {
                            Any::Advice(_) => advice,
                            Any::Fixed => fixed,
                            Any::Instance => instance,
                        };
                        parallelize(&mut modified_values, |modified_values, start| {
                            for ((modified_values, value), permuted_value) in modified_values
                                .iter_mut()
                                .zip(values[column.index()][start..].iter())
                                .zip(permuted_column_values[start..].iter())
                            {
                                *modified_values *= &(beta * permuted_value + &gamma + value);
                            }
                        });
                    }
                },
            );

            // Invert to obtain the denominator for the permutation product polynomial
            crate::perf_timer::timed_count(
                "proof_server.prover.plonk.permutation.batch_invert",
                modified_values.len(),
                || -> Result<(), Error> {
                    #[cfg(feature = "gpu-blitzar")]
                    if midnight_curves::gpu_sppark::try_bls12_381_fr_invert_assign(
                        &mut modified_values,
                    )
                    .map_err(|error| Error::Backend(error.to_string()))?
                    {
                        return Ok(());
                    }

                    modified_values.batch_invert();
                    Ok(())
                },
            )?;

            // Iterate over each column again, this time finishing the computation
            // of the entire fraction by computing the numerators
            crate::perf_timer::timed_count(
                "proof_server.prover.plonk.permutation.numerator",
                domain.n as usize * columns.len(),
                || {
                    let omega = domain.get_omega();
                    for &column in columns.iter() {
                        let values = match column.column_type() {
                            Any::Advice(_) => advice,
                            Any::Fixed => fixed,
                            Any::Instance => instance,
                        };
                        parallelize(&mut modified_values, |modified_values, start| {
                            let mut deltaomega =
                                deltaomega * &omega.pow_vartime([start as u64, 0, 0, 0]);
                            for (modified_values, value) in modified_values
                                .iter_mut()
                                .zip(values[column.index()][start..].iter())
                            {
                                // Multiply by p_j(\omega^i) + \delta^j \omega^i \beta
                                *modified_values *= &(deltaomega * &beta + &gamma + value);
                                deltaomega *= &omega;
                            }
                        });
                        deltaomega *= &F::DELTA;
                    }
                },
            );

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with the chained value from the
            // previous set.
            let z_values = crate::perf_timer::timed_count(
                "proof_server.prover.plonk.permutation.prefix_product",
                modified_values.len(),
                || -> Result<Vec<F>, Error> {
                    #[cfg(feature = "gpu-blitzar")]
                    let mut values = modified_values;
                    #[cfg(not(feature = "gpu-blitzar"))]
                    let values = modified_values;
                    #[cfg(feature = "gpu-blitzar")]
                    if midnight_curves::gpu_sppark::try_bls12_381_fr_shifted_prefix_product(
                        &mut values,
                        last_z,
                    )
                    .map_err(|error| Error::Backend(error.to_string()))?
                    {
                        Ok(values)
                    } else {
                        Ok(shifted_prefix_product(values, last_z))
                    }
                    #[cfg(not(feature = "gpu-blitzar"))]
                    {
                        Ok(shifted_prefix_product(values, last_z))
                    }
                },
            )?;
            let mut z = domain.lagrange_from_vec(z_values);
            // Set blinding factors
            crate::perf_timer::timed_count(
                "proof_server.prover.plonk.permutation.blind",
                blinding_factors,
                || {
                    for z in &mut z[domain.n as usize - blinding_factors..] {
                        *z = blind_values
                            .next()
                            .expect("validated permutation blind count");
                    }
                },
            );
            // Set new last_z
            last_z = z[domain.n as usize - (blinding_factors + 1)];

            product_values.push(z.clone());
            let permutation_product_poly = domain.lagrange_to_coeff(z);

            sets.push(CommittedSet {
                permutation_product_poly,
            });
        }

        assert!(blind_values.next().is_none());
        let commitments = CS::commit_lagrange_batch(params, &product_values);
        Ok((
            Committed {
                sets,
                #[cfg(feature = "gpu-blitzar")]
                resident_coefficients: None,
            },
            commitments,
        ))
    }
}

impl<F: PrimeField> super::ProvingKey<F> {
    pub(crate) fn open(&self, x: F) -> impl Iterator<Item = ProverQuery<'_, F>> + Clone {
        self.polys
            .iter()
            .map(move |poly| ProverQuery { point: x, poly })
    }

    pub(crate) fn evaluate<T: Transcript>(&self, x: F, transcript: &mut T) -> Result<(), Error>
    where
        F: Hashable<T::Hash>,
    {
        // Hash permutation evals
        for eval in self.polys.iter().map(|poly| eval_polynomial(poly, x)) {
            transcript.write(&eval)?;
        }

        Ok(())
    }
}

impl<F: WithSmallOrderMulGroup<3>> Committed<F> {
    pub(crate) fn evaluate<T: Transcript, CS: PolynomialCommitmentScheme<F>>(
        self,
        pk: &plonk::ProvingKey<F, CS>,
        x: F,
        transcript: &mut T,
    ) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
    {
        let domain = &pk.vk.domain;
        let blinding_factors = pk.vk.cs.blinding_factors();

        {
            #[cfg(feature = "gpu-blitzar")]
            let resident_evals = self
                .resident_coefficients
                .as_deref()
                .map(|coefficients| {
                    let count = self.sets.len();
                    let x_next = domain.rotate_omega(x, Rotation::next());
                    let x_last = domain.rotate_omega(x, Rotation(-((blinding_factors + 1) as i32)));
                    Ok::<_, Error>((
                        crate::plonk::evaluate_resident_polynomials(
                            coefficients,
                            domain.n as usize,
                            count,
                            x,
                        )?,
                        crate::plonk::evaluate_resident_polynomials(
                            coefficients,
                            domain.n as usize,
                            count,
                            x_next,
                        )?,
                        crate::plonk::evaluate_resident_polynomials(
                            coefficients,
                            domain.n as usize,
                            count,
                            x_last,
                        )?,
                    ))
                })
                .transpose()?;

            let mut sets = self.sets.iter();
            #[cfg(feature = "gpu-blitzar")]
            let mut set_index = 0usize;

            while let Some(set) = sets.next() {
                #[cfg(feature = "gpu-blitzar")]
                let (permutation_product_eval, permutation_product_next_eval) = resident_evals
                    .as_ref()
                    .map(|(at_x, at_next, _)| (at_x[set_index], at_next[set_index]))
                    .unwrap_or_else(|| {
                        (
                            eval_polynomial(&set.permutation_product_poly, x),
                            eval_polynomial(
                                &set.permutation_product_poly,
                                domain.rotate_omega(x, Rotation::next()),
                            ),
                        )
                    });
                #[cfg(not(feature = "gpu-blitzar"))]
                let (permutation_product_eval, permutation_product_next_eval) = (
                    eval_polynomial(&set.permutation_product_poly, x),
                    eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(x, Rotation::next()),
                    ),
                );

                // Hash permutation product evals
                for eval in iter::empty()
                    .chain(Some(&permutation_product_eval))
                    .chain(Some(&permutation_product_next_eval))
                {
                    transcript.write(eval)?;
                }

                // If we have any remaining sets to process, evaluate this set at omega^u
                // so we can constrain the last value of its running product to equal the
                // first value of the next set's running product, chaining them together.
                if sets.len() > 0 {
                    #[cfg(feature = "gpu-blitzar")]
                    let permutation_product_last_eval = resident_evals
                        .as_ref()
                        .map(|(_, _, at_last)| at_last[set_index])
                        .unwrap_or_else(|| {
                            eval_polynomial(
                                &set.permutation_product_poly,
                                domain.rotate_omega(x, Rotation(-((blinding_factors + 1) as i32))),
                            )
                        });
                    #[cfg(not(feature = "gpu-blitzar"))]
                    let permutation_product_last_eval = eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(x, Rotation(-((blinding_factors + 1) as i32))),
                    );

                    transcript.write(&permutation_product_last_eval)?;
                }
                #[cfg(feature = "gpu-blitzar")]
                {
                    set_index += 1;
                }
            }
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluated<F> {
    pub(crate) fn open<'a, CS: PolynomialCommitmentScheme<F>>(
        &'a self,
        pk: &'a plonk::ProvingKey<F, CS>,
        x: F,
    ) -> impl Iterator<Item = ProverQuery<'a, F>> + Clone {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());
        let x_last = pk
            .vk
            .domain
            .rotate_omega(x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(self.constructed.sets.iter().flat_map(move |set| {
                iter::empty()
                    // Open permutation product commitments at x and \omega x
                    .chain(Some(ProverQuery {
                        point: x,
                        poly: &set.permutation_product_poly,
                    }))
                    .chain(Some(ProverQuery {
                        point: x_next,
                        poly: &set.permutation_product_poly,
                    }))
            }))
            // Open it at \omega^{last} x for all but the last set. This rotation is only
            // sensical for the first row, but we only use this rotation in a constraint
            // that is gated on l_0.
            .chain(
                self.constructed
                    .sets
                    .iter()
                    .rev()
                    .skip(1)
                    .flat_map(move |set| {
                        Some(ProverQuery {
                            point: x_last,
                            poly: &set.permutation_product_poly,
                        })
                    }),
            )
    }
}
