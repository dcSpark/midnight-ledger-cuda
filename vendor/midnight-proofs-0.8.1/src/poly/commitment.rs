//! Trait for a commitment scheme
use core::ops::{Add, Mul};
#[cfg(feature = "gpu-blitzar")]
use std::{collections::HashMap, sync::Arc};
use std::{fmt::Debug, hash::Hash};

use ff::{FromUniformBytes, PrimeField};

use crate::{
    plonk::{Circuit, k_from_circuit},
    poly::{Coeff, Error, LagrangeCoeff, Polynomial, ProverQuery, VerifierQuery},
    transcript::{Hashable, Sampleable, Transcript},
    utils::helpers::ProcessedSerdeObject,
};

#[cfg(feature = "gpu-blitzar")]
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct ResidentPolynomialSource {
    pub(crate) source: u32,
    pub(crate) index: u32,
}

#[cfg(feature = "gpu-blitzar")]
#[doc(hidden)]
#[derive(Debug)]
pub struct ResidentMultiopenArena {
    pub(crate) dynamic: midnight_sppark_msm::Bls12_381FrDeviceVec,
    pub(crate) fixed: Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>,
    pub(crate) polynomial_sources: HashMap<usize, ResidentPolynomialSource>,
}

/// Public interface for a additively homomorphic Polynomial Commitment Scheme
/// (PCS)
pub trait PolynomialCommitmentScheme<F: PrimeField>: Clone + Debug {
    /// Parameters needed to generate a proof in the PCS
    type Parameters: Params;

    /// Parameters needed to verify a proof in the PCS
    type VerifierParameters;

    /// Type of a committed polynomial
    type Commitment: Clone
        + Debug
        + Default
        + PartialEq
        + ProcessedSerdeObject
        + Send
        + Sync
        + Add<Output = Self::Commitment>
        + Mul<F, Output = Self::Commitment>;

    /// Verification guard. Allows for batch verification
    type VerificationGuard: Guard<F, Self>;

    /// Generates the parameters of the polynomial commitment scheme
    fn gen_params(k: u32) -> Self::Parameters;

    /// Extract the `VerifierParameters` from `Parameters`
    fn get_verifier_params(params: &Self::Parameters) -> Self::VerifierParameters;

    /// Commit to a polynomial in coefficient form
    fn commit(params: &Self::Parameters, polynomial: &Polynomial<F, Coeff>) -> Self::Commitment;

    /// Commit to a batch of polynomials in coefficient form.
    ///
    /// The default implementation preserves the single-polynomial semantics.
    /// Commitment schemes with a native batch backend can override this method
    /// while returning commitments in input order.
    fn commit_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<F, Coeff>],
    ) -> Vec<Self::Commitment> {
        polynomials
            .iter()
            .map(|polynomial| Self::commit(params, polynomial))
            .collect()
    }

    /// Commit to a polynomial expressed in Lagrange evaluations form (over the
    /// underlying domain specified in params).
    fn commit_lagrange(
        params: &Self::Parameters,
        poly: &Polynomial<F, LagrangeCoeff>,
    ) -> Self::Commitment;

    /// Commit to a batch of polynomials expressed in Lagrange evaluations form.
    ///
    /// The default implementation preserves the single-polynomial semantics.
    /// Commitment schemes with a native batch backend can override this method
    /// while returning commitments in input order.
    fn commit_lagrange_batch(
        params: &Self::Parameters,
        polynomials: &[Polynomial<F, LagrangeCoeff>],
    ) -> Vec<Self::Commitment> {
        polynomials
            .iter()
            .map(|polynomial| Self::commit_lagrange(params, polynomial))
            .collect()
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits a same-size host batch through the alternate ec-gpu CUDA MSM.
    /// Implementations return `Ok(None)` when the curve is unsupported.
    fn commit_lagrange_ec_gpu_batch(
        _params: &Self::Parameters,
        _polynomials: &[Polynomial<F, LagrangeCoeff>],
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Uploads and commits a host batch while returning the resident values.
    fn commit_lagrange_resident_scrambled_batch(
        _params: &Self::Parameters,
        _polynomials: &[Polynomial<F, LagrangeCoeff>],
    ) -> Result<
        Option<(
            Vec<Self::Commitment>,
            Arc<midnight_sppark_msm::Bls12_381FrDeviceVec>,
        )>,
        String,
    > {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits host-resident nonzero suffixes without materializing their zero prefixes.
    fn commit_lagrange_suffix_batch(
        _params: &Self::Parameters,
        _suffixes: &[(usize, Vec<F>)],
        _poly_len: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits ragged nonzero suffixes directly from a resident polynomial arena.
    fn commit_lagrange_device_suffix_batch(
        _params: &Self::Parameters,
        _polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        _scalar_offset: usize,
        _starts: &[usize],
        _nonzero_counts: &[usize],
        _poly_len: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits a same-size batch whose scalar evaluations already reside on
    /// the GPU. Implementations return `Ok(None)` when unsupported.
    fn commit_lagrange_device_batch(
        _params: &Self::Parameters,
        _polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        _poly_len: usize,
        _count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits resident scalars after applying a cached exact MSM scramble.
    fn commit_lagrange_device_scrambled_batch(
        _params: &Self::Parameters,
        _polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        _poly_len: usize,
        _count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    #[cfg(feature = "gpu-blitzar")]
    /// Commits resident coefficient polynomials with the cached MSM scramble.
    fn commit_device_scrambled_batch(
        _params: &Self::Parameters,
        _polynomials: &midnight_sppark_msm::Bls12_381FrDeviceVec,
        _poly_len: usize,
        _count: usize,
    ) -> Result<Option<Vec<Self::Commitment>>, String> {
        Ok(None)
    }

    /// Create a multi-opening proof at a set of [ProverQuery]'s.
    fn multi_open<T: Transcript>(
        params: &Self::Parameters,
        prover_query: &[ProverQuery<F>],
        transcript: &mut T,
    ) -> Result<(), Error>
    where
        F: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        Self::Commitment: Hashable<T::Hash>;

    #[cfg(feature = "gpu-blitzar")]
    /// Creates a multi-opening proof while retaining coefficient polynomials
    /// already owned by the current GPU proof context.
    fn multi_open_with_arena<T: Transcript>(
        params: &Self::Parameters,
        prover_query: &[ProverQuery<F>],
        transcript: &mut T,
        arena: Option<ResidentMultiopenArena>,
    ) -> Result<(), Error>
    where
        F: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        Self::Commitment: Hashable<T::Hash>,
    {
        drop(arena);
        Self::multi_open(params, prover_query, transcript)
    }

    /// Verify an multi-opening proof for a given set of [VerifierQuery]'s.
    /// The function fails if the transcript has trailing bytes.
    fn multi_prepare<'com, T: Transcript>(
        verifier_query: &[VerifierQuery<'com, F, Self>],
        transcript: &mut T,
    ) -> Result<Self::VerificationGuard, Error>
    where
        F: Sampleable<T::Hash> + Hash + Ord + Hashable<T::Hash>,
        Self::Commitment: 'com + Hashable<T::Hash>;
}

/// Interface for verifier finalizer
pub trait Guard<F: PrimeField, CS: PolynomialCommitmentScheme<F>>: Sized {
    /// Finalize the verification guard
    fn verify(self, params: &CS::VerifierParameters) -> Result<(), Error>;

    /// Finalize a batch of verification guards
    fn batch_verify<'a, I, J>(guards: I, params: J) -> Result<(), Error>
    where
        I: ExactSizeIterator<Item = Self>,
        J: ExactSizeIterator<Item = &'a CS::VerifierParameters>,
        CS::VerifierParameters: 'a,
    {
        assert_eq!(guards.len(), params.len());
        guards
            .into_iter()
            .zip(params)
            .try_for_each(|(guard, params)| guard.verify(params))
    }
}

/// Interface for PCS params
pub trait Params {
    /// Returns the max size of polynomials that these parameters can commit to
    fn max_k(&self) -> u32;

    /// Downsize the params to work with a circuit of size `new_k`
    fn downsize(&mut self, new_k: u32);

    /// Downsize the params to work with a circuit of unknown length. The
    /// function first computes the `k` of the provided circuit, and then
    /// downsizes the SRS.
    fn downsize_from_circuit<
        F: PrimeField + Ord + FromUniformBytes<64>,
        ConcreCircuit: Circuit<F>,
    >(
        &mut self,
        circuit: &ConcreCircuit,
    ) {
        let k = k_from_circuit(circuit);
        self.downsize(k);
    }
}
