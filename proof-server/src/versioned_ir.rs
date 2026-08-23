// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use base_crypto::rng::SplittableRng;
use ledger::prove::Resolver;
use rand::{CryptoRng, Rng, rngs::OsRng};
#[allow(unused_imports)]
use serialize::tagged_deserialize;
use std::io::Cursor;
use transient_crypto::proofs::{Proof, ProofPreimage, Zkir};
use zkir as zkir_v2;

use crate::endpoints::PUBLIC_PARAMS;
use crate::perf_timed;
use crate::prover_backend::ProverBackend;

#[cfg(feature = "experimental")]
pub(crate) fn k(request: &[u8]) -> Result<u8, &'static str> {
    if let Ok(ir_v2) = zkir_v2::IrSource::load_from_tagged(Cursor::new(request)) {
        Ok(ir_v2.k())
    } else if let Ok(ir_v3) = tagged_deserialize::<zkir_v3::IrSource>(request) {
        Ok(ir_v3.k())
    } else {
        Err("Unsupported ZKIR version")
    }
}

#[cfg(not(feature = "experimental"))]
pub(crate) fn k(request: &[u8]) -> Result<u8, &'static str> {
    if let Ok(ir_v2) = zkir_v2::IrSource::load_from_tagged(Cursor::new(request)) {
        Ok(ir_v2.k())
    } else {
        Err("Unsupported ZKIR version")
    }
}

#[cfg(feature = "experimental")]
pub(crate) fn check(ppi: Arc<ProofPreimage>, ir: &[u8]) -> Result<Vec<Option<usize>>, String> {
    if let Ok(ir_v2) = zkir_v2::IrSource::load_from_tagged(Cursor::new(ir)) {
        ppi.check(&ir_v2).map_err(|e| e.to_string())
    } else if let Ok(ir_v3) = tagged_deserialize::<zkir_v3::IrSource>(ir) {
        ppi.check(&ir_v3).map_err(|e| e.to_string())
    } else {
        Err("Unsupported ZKIR version".to_string())
    }
}

#[cfg(not(feature = "experimental"))]
pub(crate) fn check(ppi: Arc<ProofPreimage>, ir: &[u8]) -> Result<Vec<Option<usize>>, String> {
    if let Ok(ir_v2) = zkir_v2::IrSource::load_from_tagged(Cursor::new(ir)) {
        ppi.check(&ir_v2).map_err(|e| e.to_string())
    } else {
        Err("Unsupported ZKIR version".to_string())
    }
}

pub(crate) async fn prove_with_backend(
    backend: ProverBackend,
    ppi: Arc<ProofPreimage>,
    ir_source: &[u8],
    resolver: &Resolver,
) -> Result<(Proof, Vec<Option<usize>>), String> {
    let _execution = backend.enter_proof_execution().map_err(|e| e.to_string())?;
    prove_inner(ppi, ir_source, resolver, OsRng).await
}

/// Test-only proving entry point that accepts caller-controlled proof randomness.
///
/// Production endpoints intentionally use [`prove_with_backend`], which sources randomness from
/// `OsRng`.
/// Supplying a seeded RNG makes the proof blinding deterministic for prover paths that only
/// consume this RNG; it does not by itself guarantee byte-identical proofs for every IR version.
#[cfg(any(test, feature = "deterministic-test-rng"))]
pub async fn prove_with_rng<R>(
    ppi: Arc<ProofPreimage>,
    ir_source: &[u8],
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    let _execution = ProverBackend::Cpu
        .enter_proof_execution()
        .map_err(|e| e.to_string())?;
    prove_inner(ppi, ir_source, resolver, rng).await
}

/// Benchmark/parity entry point that combines backend selection with deterministic RNG.
#[cfg(any(test, feature = "deterministic-test-rng"))]
pub async fn prove_with_backend_and_rng<R>(
    backend: ProverBackend,
    ppi: Arc<ProofPreimage>,
    ir_source: &[u8],
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    let _execution = backend.enter_proof_execution().map_err(|e| e.to_string())?;
    prove_inner(ppi, ir_source, resolver, rng).await
}

/// Benchmark-only entry point that runs an [`IrSource`](zkir_v2::IrSource)
/// through the current prover even when its relation version would normally
/// select the V0/V1 compatibility prover.
///
/// This supports parity-testing legacy production relations after regenerating
/// their keys for the current prover. Production dispatch remains unchanged.
#[cfg(any(test, feature = "deterministic-test-rng"))]
pub async fn prove_current_ir_with_backend_and_rng<R>(
    backend: ProverBackend,
    ppi: Arc<ProofPreimage>,
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    let _execution = backend.enter_proof_execution().map_err(|e| e.to_string())?;
    let provider = zkir_v2::LocalProvingProvider {
        rng,
        resolver,
        params: &*PUBLIC_PARAMS,
    };
    perf_timed!("proof_server.prove.execute", {
        provider.prove_current_with_skips(&ppi, None).await
    })
    .map_err(|e| e.to_string())
}

#[cfg(feature = "experimental")]
async fn prove_inner<R>(
    ppi: Arc<ProofPreimage>,
    ir_source: &[u8],
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    let is_v2 = perf_timed!("proof_server.prove.dispatch", {
        zkir_v2::IrSource::load_from_tagged(Cursor::new(ir_source)).is_ok()
    });
    if is_v2 {
        ppi.prove::<zkir_v2::IrSource>(rng, &*PUBLIC_PARAMS, resolver)
            .await
            .map_err(|e| e.to_string())
    } else if tagged_deserialize::<zkir_v3::IrSource>(ir_source).is_ok() {
        ppi.prove::<zkir_v3::IrSource>(rng, &*PUBLIC_PARAMS, resolver)
            .await
            .map_err(|e| e.to_string())
    } else {
        Err("Unsupported ZKIR version".into())
    }
}

#[cfg(not(feature = "experimental"))]
async fn prove_inner<R>(
    ppi: Arc<ProofPreimage>,
    ir_source: &[u8],
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    prove_v2_with_rng(ppi, ir_source, resolver, rng).await
}

#[cfg(not(feature = "experimental"))]
async fn prove_v2_with_rng<R>(
    ppi: Arc<ProofPreimage>,
    _ir_source: &[u8],
    resolver: &Resolver,
    rng: R,
) -> Result<(Proof, Vec<Option<usize>>), String>
where
    R: Rng + CryptoRng + SplittableRng,
{
    let mut provider = perf_timed!("proof_server.prove.provider", {
        zkir_v2::LocalProvingProvider {
            rng,
            resolver,
            params: &*PUBLIC_PARAMS,
        }
    });
    let (proof, skips) = perf_timed!("proof_server.prove.execute", {
        provider.split().prove_with_skips(&ppi, None).await
    })
    .map_err(|e| e.to_string())?;
    Ok((proof, skips))
}

#[cfg(test)]
mod tests {
    use super::*;

    use base_crypto::data_provider::{FetchMode, MidnightDataProvider, OutputMode};
    use coin_structure::coin;
    use ledger::dust::DustResolver;
    use rand::{SeedableRng, rngs::StdRng};
    use serialize::tagged_serialize;
    use std::borrow::Cow;
    use storage::db::InMemoryDB;
    use transient_crypto::proofs::{KeyLocation, ProvingKeyMaterial, Resolver as _};

    fn builtin_resolver() -> Resolver {
        resolver_with_data(None)
    }

    fn resolver_with_data(data: Option<ProvingKeyMaterial>) -> Resolver {
        Resolver::new(
            PUBLIC_PARAMS.clone(),
            DustResolver(
                MidnightDataProvider::new(
                    FetchMode::OnDemand,
                    OutputMode::Log,
                    ledger::dust::DUST_EXPECTED_FILES.to_owned(),
                )
                .expect("data provider initialization failed"),
            ),
            Box::new(move |_: KeyLocation| Box::pin(std::future::ready(Ok(data.clone())))),
        )
    }

    fn zswap_output_preimage() -> Arc<ProofPreimage> {
        let mut rng = StdRng::seed_from_u64(0x42);
        let sks = zswap::keys::SecretKeys::from_rng_seed(&mut rng);
        let coin = coin::Info::new(&mut rng, 100, Default::default());
        let output = zswap::Output::<_, InMemoryDB>::new(
            &mut rng,
            &coin,
            None,
            &sks.coin_public_key(),
            Some(sks.enc_public_key()),
        )
        .expect("failed to create output");

        Arc::new((*output.proof).clone())
    }

    #[tokio::test]
    async fn prove_with_rng_is_byte_reproducible_for_current_v2_ir() {
        let params = builtin_resolver();
        let ir_raw = r#"{
           "version": { "major": 2, "minor": 2 },
           "num_inputs": 1,
           "do_communications_commitment": false,
           "instructions": [
               { "op": "assert", "cond": 0 }
           ]
        }"#;
        let ir = zkir_v2::IrSource::load(ir_raw.as_bytes()).expect("minimal v2 IR should load");
        let (pk, vk) = ir
            .v2_keygen(&params)
            .await
            .expect("minimal v2 keygen should succeed");

        let mut ir_source = Vec::new();
        ir.serialize_to_tagged(&mut ir_source)
            .expect("IR should serialize");
        let mut prover_key = Vec::new();
        zkir_v2::IrSource::serialize_prover_key_to_tagged(ir.version, &pk, &mut prover_key)
            .expect("prover key should serialize");
        let mut verifier_key = Vec::new();
        tagged_serialize(&vk, &mut verifier_key).expect("verifier key should serialize");

        let resolver = resolver_with_data(Some(ProvingKeyMaterial {
            prover_key: prover_key.into(),
            verifier_key,
            ir_source: ir_source.clone(),
        }));
        let ppi = Arc::new(ProofPreimage {
            binding_input: 42.into(),
            communications_commitment: None,
            inputs: vec![1.into()],
            private_transcript: vec![],
            public_transcript_inputs: vec![],
            public_transcript_outputs: vec![],
            key_location: KeyLocation(Cow::Borrowed("test/minimal-v2")),
        });

        let first = prove_with_rng(
            ppi.clone(),
            &ir_source,
            &resolver,
            StdRng::seed_from_u64(0x5eed),
        )
        .await
        .expect("first proof should succeed");
        let second = prove_with_rng(ppi, &ir_source, &resolver, StdRng::seed_from_u64(0x5eed))
            .await
            .expect("second proof should succeed");

        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }

    #[tokio::test]
    async fn prove_with_rng_accepts_seeded_rng_for_legacy_zswap() {
        let resolver = builtin_resolver();
        let ppi = zswap_output_preimage();
        let proving_data = resolver
            .resolve_key(ppi.key_location.clone())
            .await
            .expect("resolving key should not error")
            .expect("built-in key should exist");

        let first = prove_with_rng(
            ppi.clone(),
            &proving_data.ir_source,
            &resolver,
            StdRng::seed_from_u64(0x5eed),
        )
        .await
        .expect("first proof should succeed");
        let second = prove_with_rng(
            ppi,
            &proving_data.ir_source,
            &resolver,
            StdRng::seed_from_u64(0x5eed),
        )
        .await
        .expect("second proof should succeed");

        // The current built-in zswap fixture routes through the older compatibility prover,
        // where controlling this outer RNG is not enough to guarantee byte-identical proofs.
        // The deterministic CUDA parity harness can use this entry point, but full proof-byte
        // determinism still requires pinning any lower-layer nondeterminism in that path.
        assert_eq!(first.0.0.len(), second.0.0.len());
        assert_eq!(first.1, second.1);
    }
}
