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

#[macro_use]
extern crate tracing;

use base_crypto::rng::SplittableRng;
use rand::{CryptoRng, Rng};
use serialize::tagged_deserialize;
use std::{
    collections::HashMap,
    io::Cursor,
    sync::{Arc, Mutex, OnceLock},
};
use transient_crypto::proofs::{Proof, ProverKey, ProvingKeyMaterial, VerifierKey};

mod ir;
pub mod ir_v1;
mod ir_vm;

pub use ir::{Instruction, IrMinorVersion, IrSource};
pub use ir_vm::Preprocessed;

const PARSED_IR_CACHE_SIZE: usize = 32;
static PARSED_IR_CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<IrSource>>>> = OnceLock::new();
const PARSED_PK_CACHE_SIZE: usize = 8;

struct ParsedProverKey {
    raw: Arc<Vec<u8>>,
    key: ProverKey<IrSource>,
}

static PARSED_PK_CACHE: OnceLock<Mutex<HashMap<String, ParsedProverKey>>> = OnceLock::new();

fn load_cached_ir(bytes: &[u8]) -> anyhow::Result<Arc<IrSource>> {
    let cache = PARSED_IR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ir) = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(bytes)
        .cloned()
    {
        return Ok(ir);
    }

    let parsed = Arc::new(IrSource::load_from_tagged(Cursor::new(bytes))?);
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(ir) = cache.get(bytes) {
        return Ok(ir.clone());
    }
    if cache.len() >= PARSED_IR_CACHE_SIZE {
        cache.clear();
    }
    cache.insert(bytes.to_vec(), parsed.clone());
    Ok(parsed)
}

fn load_cached_prover_key(
    location: &str,
    bytes: &Arc<Vec<u8>>,
) -> anyhow::Result<ProverKey<IrSource>> {
    let cache = PARSED_PK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let mut cache = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = cache.get_mut(location)
            && (Arc::ptr_eq(&entry.raw, bytes) || entry.raw.as_slice() == bytes.as_slice())
        {
            entry.raw = bytes.clone();
            return Ok(entry.key.clone());
        }
    }

    let parsed = <IrSource as transient_crypto::proofs::Zkir>::load_prover_key_from_tagged(
        Cursor::new(bytes.as_slice()),
    )?;
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = cache.get_mut(location)
        && (Arc::ptr_eq(&entry.raw, bytes) || entry.raw.as_slice() == bytes.as_slice())
    {
        entry.raw = bytes.clone();
        return Ok(entry.key.clone());
    }
    if cache.len() >= PARSED_PK_CACHE_SIZE && !cache.contains_key(location) {
        cache.clear();
    }
    cache.insert(
        location.to_owned(),
        ParsedProverKey {
            raw: bytes.clone(),
            key: parsed.clone(),
        },
    );
    Ok(parsed)
}

fn proof_self_verify_enabled() -> bool {
    !matches!(
        std::env::var("MIDNIGHT_PROOF_SELF_VERIFY")
            .unwrap_or_else(|_| "on".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off" | "skip" | "none"
    )
}

/// Implements `ProvingProvider` locally, dispatching V0/V1 circuits through
/// the old (zk-stdlib v1) pipeline and V2+ through the current pipeline.
pub struct LocalProvingProvider<'a, R: Rng + CryptoRng + SplittableRng, S, P> {
    /// The randomness to use for proving
    pub rng: R,
    /// The resolver to use to fetch keys
    pub resolver: &'a S,
    /// The parameters provider to use
    pub params: &'a P,
}

impl<
    'a,
    R: Rng + CryptoRng + SplittableRng,
    S: transient_crypto::proofs::Resolver,
    P: transient_crypto::proofs::ParamsProverProvider,
> LocalProvingProvider<'a, R, S, P>
{
    async fn prove_current_loaded(
        self,
        preimage: transient_crypto::proofs::ProofPreimage,
        proving_data: ProvingKeyMaterial,
        ir: Arc<IrSource>,
    ) -> Result<(Proof, Vec<Option<usize>>), anyhow::Error> {
        let verifier_key =
            transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_vk", || {
                tagged_deserialize::<VerifierKey>(&mut &proving_data.verifier_key[..])
            })?;
        let prover_key =
            transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_pk", || {
                load_cached_prover_key(preimage.key_location.0.as_ref(), &proving_data.prover_key)
            })?;
        let k = transient_crypto::proof_timer::timed("proof_server.prover.zkir.init_pk", || {
            prover_key
                .init()
                .map_err(|e| anyhow::anyhow!("Could not init pk: {e:?}"))
                .map(|pk| pk.k())
        })?;
        let (proof, pis, pi_skips) = ir
            .v2_prove(self.rng, self.params, prover_key, &preimage)
            .await?;
        if proof_self_verify_enabled() {
            debug!("proof created; verifying to make sure");
            if let Err(e) = verifier_key.verify(
                &self.params.get_params(k).await?.as_verifier(),
                &proof,
                pis.iter().copied(),
            ) {
                error!(error = ?e, ?pis, ?ir, "self-verification failed! This may be a bug, check that your keys match!");
                return Err(e);
            }
            debug!("proof ok");
        }
        Ok((proof, pi_skips))
    }

    pub async fn check(
        &self,
        preimage: &transient_crypto::proofs::ProofPreimage,
    ) -> Result<Vec<Option<usize>>, anyhow::Error> {
        let proving_data = self
            .resolver
            .resolve_key(preimage.key_location.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "attempted to check proof for '{}' without circuit data!",
                    preimage.key_location.0
                )
            })?;
        let ir = transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_ir", || {
            load_cached_ir(&proving_data.ir_source)
        })?;
        use transient_crypto::proofs::Zkir;
        ir.check(preimage)
    }

    pub async fn prove_with_skips(
        self,
        preimage: &transient_crypto::proofs::ProofPreimage,
        overwrite_binding_input: Option<transient_crypto::curve::Fr>,
    ) -> Result<(Proof, Vec<Option<usize>>), anyhow::Error> {
        let mut preimage = preimage.clone();
        if let Some(binding_input) = overwrite_binding_input {
            preimage.binding_input = binding_input;
        }

        // Resolve to determine the IR version.
        let proving_data = self
            .resolver
            .resolve_key(preimage.key_location.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "attempted to prove '{}' without circuit data!",
                    preimage.key_location.0
                )
            })?;
        let ir = transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_ir", || {
            load_cached_ir(&proving_data.ir_source)
        })?;

        match ir.version {
            IrMinorVersion::V0 | IrMinorVersion::V1 => {
                // V0/V1: use old Zkir pipeline directly.
                // Load PK via IrSource's multi-tag loader, then convert to old ProverKey.
                use transient_crypto::proofs::Zkir as _;
                let current_pk = IrSource::load_prover_key_from_tagged(std::io::Cursor::new(
                    &proving_data.prover_key[..],
                ))?;
                let mut pk_buf = Vec::new();
                serialize::Serializable::serialize(&current_pk, &mut pk_buf)?;
                let pk: transient_crypto_old::proofs::ProverKey<IrSource> =
                    serialize::Deserializable::deserialize(&mut &pk_buf[..], 0)?;
                let old_preimage = ir_v1::preimage_to_v1(&preimage);
                let v1_params = ir_v1::V1Params(self.params);
                let (proof, _, pi_skips) = transient_crypto_old::proofs::Zkir::prove(
                    ir.as_ref(),
                    self.rng,
                    &v1_params,
                    pk,
                    &old_preimage,
                )
                .await?;
                Ok((Proof(proof.0), pi_skips))
            }
            _ => self.prove_current_loaded(preimage, proving_data, ir).await,
        }
    }

    /// Proves with the current zk-stdlib/PLONK stack regardless of the
    /// relation's compatibility-dispatch version.
    ///
    /// This is intended for current-stack re-keying and deterministic parity
    /// benchmarks. Production dispatch should use [`Self::prove_with_skips`].
    pub async fn prove_current_with_skips(
        self,
        preimage: &transient_crypto::proofs::ProofPreimage,
        overwrite_binding_input: Option<transient_crypto::curve::Fr>,
    ) -> Result<(Proof, Vec<Option<usize>>), anyhow::Error> {
        let mut preimage = preimage.clone();
        if let Some(binding_input) = overwrite_binding_input {
            preimage.binding_input = binding_input;
        }
        let proving_data = self
            .resolver
            .resolve_key(preimage.key_location.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "attempted to prove '{}' without circuit data!",
                    preimage.key_location.0
                )
            })?;
        let ir = transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_ir", || {
            load_cached_ir(&proving_data.ir_source)
        })?;
        self.prove_current_loaded(preimage, proving_data, ir).await
    }

    pub fn split(&mut self) -> Self {
        Self {
            rng: self.rng.split(),
            resolver: self.resolver,
            params: self.params,
        }
    }

    pub fn resolver(&self) -> &impl transient_crypto::proofs::Resolver {
        self.resolver
    }
}

impl<
    'a,
    R: Rng + CryptoRng + SplittableRng,
    S: transient_crypto::proofs::Resolver,
    P: transient_crypto::proofs::ParamsProverProvider,
> transient_crypto::proofs::ProvingProvider for LocalProvingProvider<'a, R, S, P>
{
    async fn check(
        &self,
        preimage: &transient_crypto::proofs::ProofPreimage,
    ) -> Result<Vec<Option<usize>>, anyhow::Error> {
        Self::check(self, preimage).await
    }

    async fn prove(
        self,
        preimage: &transient_crypto::proofs::ProofPreimage,
        overwrite_binding_input: Option<transient_crypto::curve::Fr>,
    ) -> Result<Proof, anyhow::Error> {
        self.prove_with_skips(preimage, overwrite_binding_input)
            .await
            .map(|(proof, _)| proof)
    }

    fn split(&mut self) -> Self {
        Self::split(self)
    }

    fn resolver(&self) -> &impl transient_crypto::proofs::Resolver {
        Self::resolver(self)
    }
}

impl<
    'a,
    R: Rng + CryptoRng + SplittableRng,
    S: transient_crypto_old::proofs::Resolver,
    P: transient_crypto_old::proofs::ParamsProverProvider,
> transient_crypto_old::proofs::ProvingProvider for LocalProvingProvider<'a, R, S, P>
{
    async fn check(
        &self,
        preimage: &transient_crypto_old::proofs::ProofPreimage,
    ) -> Result<Vec<Option<usize>>, anyhow::Error> {
        let proving_data = self
            .resolver
            .resolve_key(preimage.key_location.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "attempted to check proof for '{}' without circuit data!",
                    preimage.key_location.0
                )
            })?;
        let ir = transient_crypto::proof_timer::timed("proof_server.prover.zkir.load_ir", || {
            load_cached_ir(&proving_data.ir_source)
        })?;
        use transient_crypto_old::proofs::Zkir as V1Zkir;
        ir.check(preimage)
    }

    async fn prove(
        self,
        preimage: &transient_crypto_old::proofs::ProofPreimage,
        overwrite_binding_input: Option<transient_crypto_old::curve::Fr>,
    ) -> Result<transient_crypto_old::proofs::Proof, anyhow::Error> {
        let mut preimage = preimage.clone();
        if let Some(binding_input) = overwrite_binding_input {
            preimage.binding_input = binding_input;
        }

        let (proof, _) = preimage
            .prove::<IrSource>(self.rng, self.params, self.resolver)
            .await?;
        Ok(proof)
    }

    fn split(&mut self) -> Self {
        Self {
            rng: self.rng.split(),
            resolver: self.resolver,
            params: self.params,
        }
    }
}
