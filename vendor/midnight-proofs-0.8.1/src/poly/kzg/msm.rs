use std::{any::TypeId, fmt::Debug};

use ff::Field;
use group::{Curve, Group};
use itertools::izip;
use midnight_curves::{
    CurveAffine, Fq, G1Projective,
    msm::msm_best,
    pairing::{Engine, MillerLoopResult, MultiMillerLoop},
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use super::params::ParamsVerifierKZG;
use crate::{
    poly::{
        CommitmentLabel, Error,
        commitment::{Guard, PolynomialCommitmentScheme},
        kzg::KZGCommitmentScheme,
    },
    utils::{
        arithmetic::{CurveExt, MSM},
        helpers::ProcessedSerdeObject,
    },
};

/// A multi-scalar multiplication in the polynomial commitment scheme.
/// For every i, term (bases_i, scalars_i) may be have an optional
/// label_i for debugging or other purposes.
#[derive(Clone, Default, Debug)]
pub struct MSMKZG<E: Engine> {
    pub(crate) scalars: Vec<E::Fr>,
    pub(crate) bases: Vec<E::G1>,
    pub(crate) labels: Vec<CommitmentLabel>,
}

impl<E: Engine> MSMKZG<E> {
    /// Create an empty MSM instance
    pub fn init() -> Self {
        MSMKZG {
            scalars: vec![],
            bases: vec![],
            labels: vec![],
        }
    }

    /// Create an MSM from various MSMs
    pub fn from_many(msms: Vec<Self>) -> Self {
        let len = msms.iter().map(|m| m.scalars.len()).sum();

        let mut scalars = Vec::with_capacity(len);
        let mut bases = Vec::with_capacity(len);
        let mut labels = Vec::with_capacity(len);

        for mut msm in msms {
            scalars.append(&mut msm.scalars);
            bases.append(&mut msm.bases);
            labels.append(&mut msm.labels);
        }

        Self {
            scalars,
            bases,
            labels,
        }
    }

    /// Create a new MSM from a given base (with scalar of 1).
    pub fn from_base(base: &E::G1) -> Self {
        MSMKZG {
            scalars: vec![E::Fr::ONE],
            bases: vec![*base],
            labels: vec![CommitmentLabel::NoLabel],
        }
    }
}

impl<E: Engine + Debug> MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Evaluates the MSM to a single point and replaces all terms with that
    /// single point (scalar = 1, label = `NoLabel`).
    ///
    /// This mirrors `AssignedMsm::collapse` in the circuits crate.
    ///
    /// # Panics (in debug mode)
    ///
    /// If any term carries a label other than `NoLabel` or `Advice`.
    //
    // We only allow `NoLabel` or `Advice` because these types of labels are
    // not relevant for the `verifier_gadget` in `midnight-circuits` (at least for
    // now). Other types of labels may carry information that we do not want to lose
    // when "collapsing".
    pub fn collapse(&mut self) {
        debug_assert!(
            self.labels
                .iter()
                .all(|l| matches!(l, CommitmentLabel::NoLabel | CommitmentLabel::Advice(_))),
            "collapse: all labels must be NoLabel or Advice, found: {:?}",
            self.labels,
        );
        let point = self.eval();
        self.scalars = vec![E::Fr::ONE];
        self.bases = vec![point];
        self.labels = vec![CommitmentLabel::NoLabel];
    }
}

impl<E: Engine + Debug> MSM<E::G1Affine> for MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn append_term(&mut self, scalar: E::Fr, point: E::G1, label: CommitmentLabel) {
        self.scalars.push(scalar);
        self.bases.push(point);
        self.labels.push(label);
    }

    fn add_msm(&mut self, other: &Self) {
        self.scalars.reserve(other.scalars().len());
        self.scalars.extend_from_slice(&other.scalars());

        self.bases.reserve(other.bases().len());
        self.bases.extend_from_slice(&other.bases());

        self.labels.reserve(other.labels().len());
        self.labels.extend_from_slice(&other.labels());
    }

    fn scale(&mut self, factor: E::Fr) {
        self.scalars.par_iter_mut().for_each(|s| {
            *s *= &factor;
        })
    }

    fn check(&self) -> bool {
        bool::from(self.eval().is_identity())
    }

    fn eval(&self) -> E::G1 {
        if self.scalars == vec![E::Fr::ONE] {
            self.bases[0]
        } else {
            msm_specific::<E::G1Affine>(&self.scalars, &self.bases)
        }
    }

    fn bases(&self) -> Vec<E::G1> {
        self.bases.clone()
    }

    fn scalars(&self) -> Vec<E::Fr> {
        self.scalars.clone()
    }

    fn labels(&self) -> Vec<CommitmentLabel> {
        self.labels.clone()
    }
}

#[allow(unsafe_code)]
/// Wrapper over the MSM function to use the blstrs underlying function
pub fn msm_specific<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    if TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>() {
        let n = coeffs.len().min(bases.len());
        let coeffs = unsafe { &*(&coeffs[..n] as *const [C::Scalar] as *const [Fq]) };
        midnight_curves::msm_run_diagnostic::record_g1_msm(coeffs);
    }

    #[cfg(feature = "gpu-blitzar")]
    if super::gpu_blitzar::should_accelerate_sppark(coeffs.len())
        && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
    {
        let n = coeffs.len().min(bases.len());
        if n != 0 {
            let nonzero_scalars = coeffs[..n]
                .iter()
                .filter(|scalar| !scalar.is_zero_vartime())
                .count();
            if super::gpu_blitzar::should_accelerate_sppark_dense_unfiltered(n, nonzero_scalars) {
                let coeffs_slice = &coeffs[..n];
                let bases_slice = &bases[..n];
                let coeffs = unsafe { &*(coeffs_slice as *const _ as *const [Fq]) };
                let bases = unsafe { &*(bases_slice as *const _ as *const [G1Projective]) };
                let res = crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.msm_specific.gpu_sppark_unfiltered",
                    n,
                    || {
                        G1Projective::multi_exp_sppark(bases, coeffs)
                            .expect("sppark BLS12-381 G1 unfiltered MSM failed")
                    },
                );
                return unsafe { std::mem::transmute_copy(&res) };
            }
        }
    }

    // We remove zeros (keep only non-zero coefficients)
    let (coeffs, bases): (Vec<C::Scalar>, Vec<C::Curve>) = coeffs
        .iter()
        .zip(bases)
        .filter(|(s, _)| !s.is_zero_vartime())
        .map(|(s, b)| (*s, *b))
        .unzip();

    crate::perf_timer::timed_count(
        "proof_server.prover.kzg.msm_specific.filtered",
        coeffs.len(),
        || {
            if coeffs.is_empty() {
                return C::Curve::identity();
            }

            #[cfg(feature = "gpu-blitzar")]
            if super::gpu_blitzar::should_accelerate_sppark(coeffs.len())
                && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
            {
                let coeffs_slice = coeffs.as_slice();
                let bases_slice = bases.as_slice();
                let coeffs = unsafe { &*(coeffs_slice as *const _ as *const [Fq]) };
                let bases = unsafe { &*(bases_slice as *const _ as *const [G1Projective]) };
                let res = crate::perf_timer::timed_count(
                    "proof_server.prover.kzg.msm_specific.gpu_sppark",
                    coeffs.len(),
                    || {
                        G1Projective::multi_exp_sppark(bases, coeffs)
                            .expect("sppark BLS12-381 G1 MSM failed")
                    },
                );
                return unsafe { std::mem::transmute_copy(&res) };
            }

            // We empirically checked that for MSMs larger than 2**18, the blstrs
            // implementation regresses.
            if coeffs.len() <= (2 << 18)
                && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
            {
                // Safe: we just checked type
                let coeffs_slice = coeffs.as_slice();
                let bases_slice = bases.as_slice();
                let coeffs = unsafe { &*(coeffs_slice as *const _ as *const [Fq]) };
                let bases = unsafe { &*(bases_slice as *const _ as *const [G1Projective]) };
                let res = G1Projective::multi_exp(bases, coeffs);
                unsafe { std::mem::transmute_copy(&res) }
            } else {
                let mut affine_bases = vec![C::identity(); coeffs.len()];
                C::Curve::batch_normalize(&bases, &mut affine_bases);
                msm_best(&coeffs, &affine_bases)
            }
        },
    )
}

/// Two channel MSM accumulator
#[derive(Debug, Clone)]
pub struct DualMSM<E: Engine> {
    pub(crate) left: MSMKZG<E>,
    pub(crate) right: MSMKZG<E>,
}

/// A [DualMSM] split into left and right vectors of `(Scalar, Point)` tuples
pub type SplitDualMSM<'a, E> = (
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
);

impl<E: MultiMillerLoop + Debug> Default for DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn default() -> Self {
        Self::init()
    }
}

impl<E: MultiMillerLoop> Guard<E::Fr, KZGCommitmentScheme<E>> for DualMSM<E>
where
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn verify(
        self,
        params: &<KZGCommitmentScheme<E> as PolynomialCommitmentScheme<E::Fr>>::VerifierParameters,
    ) -> Result<(), Error> {
        self.check(params).then_some(()).ok_or(Error::OpeningError)
    }
}

impl<E: MultiMillerLoop + Debug> DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Create an empty two channel MSM accumulator instance
    pub fn init() -> Self {
        Self {
            left: MSMKZG::init(),
            right: MSMKZG::init(),
        }
    }

    /// Create a new two channel MSM accumulator instance
    pub fn new(left: MSMKZG<E>, right: MSMKZG<E>) -> Self {
        Self { left, right }
    }

    /// Split the [DualMSM] into `left` and `right`
    pub fn split(&self) -> SplitDualMSM<'_, E> {
        let left = izip!(
            self.left.labels.iter(),
            self.left.scalars.iter(),
            self.left.bases.iter()
        )
        .collect();
        let right = izip!(
            self.right.labels.iter(),
            self.right.scalars.iter(),
            self.right.bases.iter(),
        )
        .collect();
        (left, right)
    }

    /// Scale all scalars in the MSM by some scaling factor
    pub fn scale(&mut self, e: E::Fr) {
        self.left.scale(e);
        self.right.scale(e);
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: Self) {
        self.left.add_msm(&other.left);
        self.right.add_msm(&other.right);
    }

    /// Performs final pairing check with given verifier params and two channel
    /// linear combination
    pub fn check(self, params: &ParamsVerifierKZG<E>) -> bool {
        let Self { left, right } = self;
        let left = crate::perf_timer::timed("proof_server.prover.verify.msm_left", || {
            if left.scalars.len() == 1 && left.scalars[0] == E::Fr::ONE {
                left.bases[0]
            } else {
                left.eval()
            }
        });

        let right =
            crate::perf_timer::timed("proof_server.prover.verify.msm_right", || right.eval());

        let (term_1, term_2) = (
            (&left.into(), &params.s_g2_prepared),
            (&right.into(), &params.n_g2_prepared),
        );
        let terms = &[term_1, term_2];

        let miller = crate::perf_timer::timed("proof_server.prover.verify.miller_loop", || {
            E::multi_miller_loop(&terms[..])
        });
        let result =
            crate::perf_timer::timed("proof_server.prover.verify.final_exponentiation", || {
                miller.final_exponentiation()
            });
        bool::from(result.is_identity())
    }
}
