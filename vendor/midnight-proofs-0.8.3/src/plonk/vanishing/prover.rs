use std::{collections::HashMap, iter};

use ff::{PrimeField, WithSmallOrderMulGroup};
use rand_chacha::ChaCha20Rng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rayon::current_num_threads;

use super::Argument;
use crate::{
    plonk::Error,
    poly::{
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial, ProverQuery,
        commitment::PolynomialCommitmentScheme,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) random_poly: Polynomial<F, Coeff>,
}

#[cfg_attr(feature = "bench-internal", derive(Clone))]
pub(crate) struct Constructed<F: PrimeField> {
    h_pieces: Vec<Polynomial<F, Coeff>>,
    #[cfg(feature = "gpu-blitzar")]
    resident_h_pieces: Option<std::sync::Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    committed: Committed<F>,
}

pub(crate) struct Evaluated<F: PrimeField> {
    h_poly: Polynomial<F, Coeff>,
    #[cfg(feature = "gpu-blitzar")]
    resident_h_poly: Option<std::sync::Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>>,
    committed: Committed<F>,
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> Argument<F, CS> {
    pub(crate) fn sample_random(
        domain: &EvaluationDomain<F>,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> Committed<F> {
        // Sample a random polynomial of degree n - 1
        let n = 1usize << domain.k() as usize;
        let mut rand_vec = vec![F::ZERO; n];

        let num_threads = current_num_threads();
        let chunk_size = n / num_threads;
        let thread_seeds = (0..)
            .step_by(chunk_size + 1)
            .take(n % num_threads)
            .chain(
                (chunk_size != 0)
                    .then(|| ((n % num_threads) * (chunk_size + 1)..).step_by(chunk_size))
                    .into_iter()
                    .flatten(),
            )
            .take(num_threads)
            .zip(iter::repeat_with(|| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                ChaCha20Rng::from_seed(seed)
            }))
            .collect::<HashMap<_, _>>();

        parallelize(&mut rand_vec, |chunk, offset| {
            let mut rng = thread_seeds[&offset].clone();
            chunk.iter_mut().for_each(|v| *v = F::random(&mut rng));
        });

        let random_poly: Polynomial<F, Coeff> = domain.coeff_from_vec(rand_vec);

        Committed { random_poly }
    }

    pub(crate) fn commit<T: Transcript>(
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        rng: &mut (impl RngCore + CryptoRng),
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        F: Hashable<T::Hash>,
    {
        let committed = Self::sample_random(domain, rng);

        // Commit
        let c = CS::commit(params, &committed.random_poly);
        transcript.write(&c)?;

        Ok(committed)
    }
}

impl<F: WithSmallOrderMulGroup<3>> Committed<F> {
    pub(crate) fn construct<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        h_poly: Polynomial<F, ExtendedLagrangeCoeff>,
        transcript: &mut T,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> Result<Constructed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        CS::Parameters: Sync,
        F: Hashable<T::Hash>,
        F: Sync,
    {
        let mut h_poly = h_poly;
        let fused_quotient = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.divide_to_coeff.gpu_sppark",
            h_poly.len(),
            || domain.try_divide_by_vanishing_to_coeff(&mut h_poly),
        );

        let h_poly = if fused_quotient {
            h_poly.values
        } else {
            // Divide by t(X) = X^{params.n} - 1.
            let h_poly = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.vanishing.construct.divide_by_vanishing",
                h_poly.len(),
                || domain.divide_by_vanishing_poly(h_poly),
            );

            // Obtain final h(X) polynomial
            crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.vanishing.construct.extended_to_coeff",
                h_poly.len(),
                || domain.extended_to_coeff(h_poly),
            )
        };

        self.construct_from_coeff_values::<CS, T>(params, domain, h_poly, transcript, rng)
    }

    pub(crate) fn construct_from_coeff_values<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        mut h_poly: Vec<F>,
        transcript: &mut T,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> Result<Constructed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        CS::Parameters: Sync,
        F: Hashable<T::Hash>,
        F: Sync,
    {
        // Let n := size of evaluation domain
        // Let d := degree of constraint system
        // Hence, the degree of the quotient poly is: d*(n-1) - n = (d-1)*(n-1) - 1,
        // and a domain of size (d-1)*(n-1) suffices to correctly represent it
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.truncate",
            h_poly.len(),
            || h_poly.truncate((domain.n - 1) as usize * domain.get_quotient_poly_degree()),
        );

        // Split h(X) up into pieces
        let mut h_pieces = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.split",
            h_poly.len(),
            || {
                h_poly
                    .chunks_exact((domain.n - 1) as usize)
                    .map(|v| v.to_vec())
                    .collect::<Vec<_>>()
            },
        );
        drop(h_poly);

        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.blind_limbs",
            h_pieces.len(),
            || blind_quotient_limbs(&mut h_pieces, rng),
        );

        let h_pieces: Vec<_> = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.coeff_from_limbs",
            h_pieces.iter().map(Vec::len).sum(),
            || {
                h_pieces
                    .into_iter()
                    .map(|h_piece| domain.coeff_from_vec(h_piece))
                    .collect()
            },
        );

        // Compute commitments to each h(X) piece
        let h_commitments: Vec<_> = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.commit_batch",
            h_pieces.iter().map(|poly| poly.values.len()).sum(),
            || CS::commit_batch(params, &h_pieces),
        );

        // Hash each h(X) piece
        crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.write_commitments",
            h_commitments.len(),
            || {
                for c in h_commitments {
                    transcript.write(&c)?;
                }
                Ok::<(), Error>(())
            },
        )?;

        Ok(Constructed {
            h_pieces,
            #[cfg(feature = "gpu-blitzar")]
            resident_h_pieces: None,
            committed: self,
        })
    }

    /// Continues after a backend has committed and transcript-bound quotient limbs.
    ///
    /// The supplied coefficients are their challenge-dependent opening
    /// combination. This method deliberately performs no commitment, RNG, or
    /// second quotient-piece Horner pass.
    pub(crate) fn construct_from_precommitted_h_poly(
        self,
        domain: &EvaluationDomain<F>,
        coefficients: Vec<F>,
    ) -> Result<Constructed<F>, Error> {
        let piece_len = domain.n as usize;
        if coefficients.len() != piece_len {
            return Err(Error::Backend(
                "precombined quotient opening does not match the proof domain".into(),
            ));
        }
        Ok(Constructed {
            h_pieces: vec![domain.coeff_from_vec(coefficients)],
            #[cfg(feature = "gpu-blitzar")]
            resident_h_pieces: None,
            committed: self,
        })
    }

    #[cfg(feature = "gpu-blitzar")]
    pub(crate) fn construct_from_resident_coefficients<
        CS: PolynomialCommitmentScheme<F>,
        T: Transcript,
    >(
        self,
        params: &CS::Parameters,
        domain: &EvaluationDomain<F>,
        quotient: midnight_sppark_msm::Bls12_381FrDeviceVec,
        transcript: &mut T,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> Result<Constructed<F>, Error>
    where
        CS::Commitment: Hashable<T::Hash>,
        CS::Parameters: Sync,
        F: Hashable<T::Hash> + Sync + 'static,
    {
        let piece_count = domain.get_quotient_poly_degree();
        let poly_len = domain.n as usize;
        if piece_count < 2 {
            return Err(Error::Backend(
                "resident quotient requires at least two limbs".into(),
            ));
        }
        let blinds = (0..piece_count - 1)
            .map(|_| F::random(&mut *rng))
            .collect::<Vec<_>>();
        let raw_blinds = blinds
            .iter()
            .copied()
            .map(crate::plonk::evaluation::raw_from_field)
            .collect::<Vec<_>>();
        let pieces = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.resident_split_blind",
            poly_len * piece_count,
            || quotient.quotient_pieces(poly_len, piece_count, &raw_blinds),
        )
        .map_err(|error| Error::Backend(error.to_string()))?;
        let commitments = crate::perf_timer::timed_count_detailed(
            "proof_server.prover.plonk.vanishing.construct.resident_commit_batch",
            poly_len * piece_count,
            || CS::commit_device_scrambled_batch(params, &pieces, poly_len, piece_count),
        )
        .map_err(Error::Backend)?
        .ok_or_else(|| {
            Error::Backend("commitment scheme does not support resident quotient limbs".into())
        })?;
        for commitment in commitments {
            transcript.write(&commitment)?;
        }
        Ok(Constructed {
            h_pieces: Vec::new(),
            resident_h_pieces: Some(std::sync::Arc::new(pieces)),
            committed: self,
        })
    }
}

fn blind_quotient_limbs<F: PrimeField>(
    quotient_limbs: &mut [Vec<F>],
    rng: &mut (impl RngCore + CryptoRng),
) {
    let nr_limbs = quotient_limbs.len();
    assert!(nr_limbs >= 2);

    for i in 1..nr_limbs {
        let t = F::random(&mut *rng);
        quotient_limbs[i - 1].push(t);
        quotient_limbs[i][0] -= t;
    }

    quotient_limbs[nr_limbs - 1].push(F::ZERO);
}

impl<F: WithSmallOrderMulGroup<3>> Constructed<F> {
    pub(crate) fn evaluate<T: Transcript>(
        self,
        x: F,
        domain: &EvaluationDomain<F>,
        transcript: &mut T,
    ) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
    {
        let splitting_factor: F = x.pow_vartime([domain.n - 1]);
        #[cfg(feature = "gpu-blitzar")]
        let resident_h_poly = if let Some(pieces) = self.resident_h_pieces {
            let poly_len = domain.n as usize;
            let piece_count = pieces.len() / poly_len;
            let mut power = F::ONE;
            let scalars = (0..piece_count)
                .map(|_| {
                    let current = power;
                    power *= splitting_factor;
                    crate::plonk::evaluation::raw_from_field(current)
                })
                .collect::<Vec<_>>();
            let combined = crate::perf_timer::timed_count_detailed(
                "proof_server.prover.plonk.vanishing.evaluate.resident_h_lincomb",
                poly_len * piece_count,
                || pieces.poly_lincomb_device(poly_len, &scalars),
            )
            .map_err(|error| Error::Backend(error.to_string()))?;
            Some(std::sync::Arc::new(combined))
        } else {
            None
        };
        #[cfg(feature = "gpu-blitzar")]
        let h_poly = if resident_h_poly.is_some() {
            domain.empty_coeff()
        } else {
            self.h_pieces
                .into_iter()
                .rev()
                .reduce(|acc, eval| acc * splitting_factor + eval)
                .expect("H pieces should not be empty")
        };
        #[cfg(not(feature = "gpu-blitzar"))]
        let h_poly = self
            .h_pieces
            .into_iter()
            .rev()
            .reduce(|acc, eval| acc * splitting_factor + eval)
            .expect("H pieces should not be empty");

        let random_eval = eval_polynomial(&self.committed.random_poly, x);
        transcript.write(&random_eval)?;

        Ok(Evaluated {
            h_poly,
            #[cfg(feature = "gpu-blitzar")]
            resident_h_poly,
            committed: self.committed,
        })
    }
}

impl<F: PrimeField> Evaluated<F> {
    #[cfg(feature = "gpu-blitzar")]
    pub(crate) fn resident_opening(
        &self,
    ) -> Option<(
        &Polynomial<F, Coeff>,
        &midnight_sppark_msm::Bls12_381FrDeviceVec,
    )> {
        self.resident_h_poly
            .as_deref()
            .map(|resident| (&self.h_poly, resident))
    }

    pub(crate) fn open(&self, x: F) -> impl Iterator<Item = ProverQuery<'_, F>> + Clone {
        iter::empty()
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.h_poly,
            }))
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.committed.random_poly,
            }))
    }
}
