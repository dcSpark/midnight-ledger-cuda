use std::{collections::HashMap, hash::Hash, sync::Arc};

use blake2b_simd::State as Blake2bState;
use ff::{Field, WithSmallOrderMulGroup};
use midnight_curves::{Bls12, CurveAffine, CurveExt, pairing::MultiMillerLoop, serde::SerdeObject};
use rand_core::OsRng;

use super::{
    KZGCommitmentScheme, gpu_blitzar, gpu_resident_multiopen_enabled,
    gpu_resident_multiopen_pipeline_enabled,
    params::{ParamsKZG, ParamsVerifierKZG},
    raw_scalar,
};
use crate::{
    poly::{
        Coeff, CommitmentLabel, Error, EvaluationDomain, Polynomial,
        commitment::{
            Guard, PolynomialCommitmentScheme, ResidentMultiopenArena, ResidentPolynomialSource,
        },
        query::{ProverQuery, VerifierQuery},
    },
    transcript::{CircuitTranscript, Hashable, Sampleable, Transcript},
    utils::arithmetic::eval_polynomial,
};

#[derive(Clone, Copy)]
enum MultiOpenEntryPoint {
    Standard,
    Resident,
}

/// Asserts that a consumed resident arena cannot leak state into the next
/// multi-open call, and that the fresh resident proof matches the CPU proof.
#[doc(hidden)]
pub fn assert_resident_multiopen_ownership_error_isolation() {
    const K: u32 = 14;
    const POLY_LEN: usize = 1 << K;

    let _gpu = gpu_blitzar::EnableGuard::set(true);
    assert!(gpu_blitzar::enabled());
    assert!(gpu_resident_multiopen_enabled(POLY_LEN));
    assert!(gpu_resident_multiopen_pipeline_enabled());

    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);

    // Force an error after transferring ownership of a real device arena.
    // Returning from multi_open_impl must drop that arena; the successful
    // resident call below demonstrates that no per-call state survived.
    let domain = EvaluationDomain::new(1, K);
    let error_poly = domain.empty_coeff();
    let error_arena = resident_multiopen_arena(&[&error_poly]);
    let mut error_transcript = CircuitTranscript::<Blake2bState>::init();
    let error = KZGCommitmentScheme::<Bls12>::multi_open_with_arena(
        &params,
        &[],
        &mut error_transcript,
        Some(error_arena),
    );
    assert!(matches!(error, Err(Error::OpeningError)));

    let standard =
        create_proof::<_, CircuitTranscript<Blake2bState>>(&params, MultiOpenEntryPoint::Standard);
    let resident =
        create_proof::<_, CircuitTranscript<Blake2bState>>(&params, MultiOpenEntryPoint::Resident);

    assert_eq!(resident, standard);
    verify::<Bls12, CircuitTranscript<Blake2bState>>(&params.verifier_params(), &resident);
}

fn resident_multiopen_arena<F: Field + 'static>(
    polynomials: &[&Polynomial<F, Coeff>],
) -> ResidentMultiopenArena {
    let poly_len = polynomials
        .first()
        .expect("resident multi-open test arena is empty")
        .len();
    assert!(
        polynomials
            .iter()
            .all(|polynomial| polynomial.len() == poly_len)
    );

    let mut values = Vec::with_capacity(poly_len * polynomials.len());
    let mut polynomial_sources = HashMap::with_capacity(polynomials.len());
    for (index, polynomial) in polynomials.iter().enumerate() {
        values.extend(polynomial.iter().copied().map(raw_scalar));
        polynomial_sources.insert(
            *polynomial as *const Polynomial<F, Coeff> as usize,
            ResidentPolynomialSource {
                source: 0,
                index: u32::try_from(index).unwrap(),
            },
        );
    }

    let dynamic = midnight_sppark_msm::Bls12_381FrDeviceVec::upload(&values).unwrap();
    let fixed = Arc::new(
        midnight_sppark_msm::Bls12_381FrDeviceVec::upload(&vec![
            midnight_sppark_msm::Bls12_381Fr::default();
            poly_len
        ])
        .unwrap(),
    );

    ResidentMultiopenArena {
        dynamic,
        fixed,
        polynomial_sources,
    }
}

fn verify<E, T>(verifier_params: &ParamsVerifierKZG<E>, proof: &[u8])
where
    E: MultiMillerLoop,
    T: Transcript,
    E::Fr: Hashable<T::Hash> + Sampleable<T::Hash> + Ord + Hash,
    E::G1: Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + SerdeObject,
{
    let mut transcript = T::init_from_bytes(proof);

    let a: E::G1 = transcript.read().unwrap();
    let b: E::G1 = transcript.read().unwrap();
    let c: E::G1 = transcript.read().unwrap();

    let x: E::Fr = transcript.squeeze_challenge();
    let y: E::Fr = transcript.squeeze_challenge();

    let avx: E::Fr = transcript.read().unwrap();
    let bvx: E::Fr = transcript.read().unwrap();
    let cvy: E::Fr = transcript.read().unwrap();

    use CommitmentLabel::NoLabel;
    let queries = std::iter::empty()
        .chain(Some(VerifierQuery::new(x, NoLabel, &a, avx)))
        .chain(Some(VerifierQuery::new(x, NoLabel, &b, bvx)))
        .chain(Some(VerifierQuery::new(y, NoLabel, &c, cvy)))
        .collect::<Vec<_>>();

    let result = KZGCommitmentScheme::multi_prepare(&queries, &mut transcript).unwrap();
    assert!(result.verify(verifier_params).is_ok());
}

fn create_proof<E, T>(kzg_params: &ParamsKZG<E>, entry_point: MultiOpenEntryPoint) -> Vec<u8>
where
    E: MultiMillerLoop,
    T: Transcript,
    E::Fr: WithSmallOrderMulGroup<3> + Hashable<T::Hash> + Hash + Sampleable<T::Hash> + Ord,
    E::G1: Hashable<T::Hash> + CurveExt<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
    E::G1Affine: SerdeObject + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    let k = (kzg_params.g.len() - 1).ilog2() + 1;
    let domain = EvaluationDomain::new(1, k);

    let mut ax = domain.empty_coeff();
    for (i, a) in ax.iter_mut().enumerate() {
        *a = <E::Fr>::from(10 + i as u64);
    }

    let mut bx = domain.empty_coeff();
    for (i, a) in bx.iter_mut().enumerate() {
        *a = <E::Fr>::from(100 + i as u64);
    }

    let mut cx = domain.empty_coeff();
    for (i, a) in cx.iter_mut().enumerate() {
        *a = <E::Fr>::from(100 + i as u64);
    }

    let mut transcript = T::init();

    let a = KZGCommitmentScheme::commit(kzg_params, &ax);
    let b = KZGCommitmentScheme::commit(kzg_params, &bx);
    let c = KZGCommitmentScheme::commit(kzg_params, &cx);

    transcript.write(&a).unwrap();
    transcript.write(&b).unwrap();
    transcript.write(&c).unwrap();

    let x: E::Fr = transcript.squeeze_challenge();
    let y = transcript.squeeze_challenge();

    let avx = eval_polynomial(&ax, x);
    let bvx = eval_polynomial(&bx, x);
    let cvy = eval_polynomial(&cx, y);

    transcript.write(&avx).unwrap();
    transcript.write(&bvx).unwrap();
    transcript.write(&cvy).unwrap();

    let queries = vec![
        ProverQuery {
            point: x,
            poly: &ax,
        },
        ProverQuery {
            point: x,
            poly: &bx,
        },
        ProverQuery {
            point: y,
            poly: &cx,
        },
    ];

    match entry_point {
        MultiOpenEntryPoint::Standard => {
            KZGCommitmentScheme::multi_open(kzg_params, &queries, &mut transcript)
        }
        MultiOpenEntryPoint::Resident => {
            let arena = resident_multiopen_arena(&[&ax, &bx, &cx]);
            KZGCommitmentScheme::multi_open_with_arena(
                kzg_params,
                &queries,
                &mut transcript,
                Some(arena),
            )
        }
    }
    .unwrap();

    transcript.finalize()
}
