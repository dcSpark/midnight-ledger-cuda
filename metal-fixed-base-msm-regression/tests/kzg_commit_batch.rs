// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

//! Batched KZG commitments with Metal selected against the CPU's, on the same
//! polynomials and repeated: the fixed-base batch kernel once returned a wrong
//! point for some outputs on some runs only.

use ff::Field;
use midnight_curves::{Bls12, Fq as F, G1Projective, gpu_metal::EnableGuard};
use midnight_proofs::poly::{
    EvaluationDomain,
    commitment::PolynomialCommitmentScheme,
    kzg::{KZGCommitmentScheme, params::ParamsKZG},
};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

type Scheme = KZGCommitmentScheme<Bls12>;

const K: u32 = 16;
const BATCH: usize = 6;
const ITERATIONS: u64 = 20;
const RUNS: usize = 8;

fn commit_lagrange_batch(
    params: &ParamsKZG<Bls12>,
    domain: &EvaluationDomain<F>,
    evaluations: &[Vec<F>],
    metal: bool,
) -> Vec<G1Projective> {
    let _guard = EnableGuard::set(metal);
    let polynomials: Vec<_> =
        evaluations.iter().map(|values| domain.lagrange_from_vec(values.clone())).collect();
    Scheme::commit_lagrange_batch(params, &polynomials)
}

#[test]
#[ignore = "requires access to an Apple Metal device"]
fn metal_lagrange_commit_batches_equal_cpu_on_repeated_runs() {
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, ChaCha20Rng::seed_from_u64(1));
    let domain = EvaluationDomain::<F>::new(1, K);
    let mut wrong = Vec::new();
    for iteration in 0..ITERATIONS {
        let mut rng = ChaCha20Rng::seed_from_u64(100 + iteration);
        let evaluations: Vec<Vec<F>> =
            (0..BATCH).map(|_| (0..1 << K).map(|_| F::random(&mut rng)).collect()).collect();
        let cpu = commit_lagrange_batch(&params, &domain, &evaluations, false);
        for run in 0..RUNS {
            let metal = commit_lagrange_batch(&params, &domain, &evaluations, true);
            for (output, (cpu, metal)) in cpu.iter().zip(&metal).enumerate() {
                if cpu != metal {
                    wrong.push(format!("iteration {iteration} run {run} output {output}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "Metal batched commitments differ from the CPU's: {wrong:?}");
}
