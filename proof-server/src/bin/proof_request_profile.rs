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

//! Diagnostic only: replay captured MicroDAO v3 requests with seeded proof randomness.
//! Does not change server endpoints, production RNG, keys, witnesses, or verification.
//! "resident" retains typed IR/PK/VK; "provider" repeats the production key-loading path.
//! Both omit HTTP transport and outer request decoding from measured prove time.

use std::{io::Cursor, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Result, anyhow, ensure};
use base_crypto::data_provider::{FetchMode, MidnightDataProvider, OutputMode};
use clap::{Parser, ValueEnum};
use ledger::{dust::DustResolver, prove::Resolver, structure::ProofPreimageVersioned};
use midnight_proof_server::{
    endpoints::PUBLIC_PARAMS, prover_backend::ProverBackend,
    versioned_ir::prove_with_backend_and_rng,
};
use rand::{SeedableRng, rngs::StdRng};
use serialize::tagged_deserialize;
use transient_crypto::{
    curve::Fr,
    proofs::{
        GpuBlitzarGuard, KeyLocation, ParamsProverProvider, Proof, ProofPreimage, ProverKey,
        ProvingKeyMaterial, VerifierKey, Zkir,
    },
};
use zkir_v3::IrSource;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Provider,
    Resident,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    request: PathBuf,
    #[arg(long, value_enum, default_value_t = Mode::Resident)]
    mode: Mode,
    #[arg(long, value_enum, default_value_t = ProverBackend::Cuda)]
    backend: ProverBackend,
    #[arg(long, default_value_t = 12)]
    iterations: u64,
    #[arg(long, default_value_t = 20260904)]
    seed: u64,
    #[arg(long)]
    cpu_parity: bool,
}

struct Fixture {
    ppi: Arc<ProofPreimage>,
    ir_bytes: Vec<u8>,
    ir: IrSource,
    pk: ProverKey<IrSource>,
    vk: VerifierKey,
    resolver: Resolver,
}

impl Fixture {
    async fn run(&self, mode: Mode, backend: ProverBackend, seed: u64) -> Result<Proof> {
        let rng = StdRng::seed_from_u64(seed);
        match mode {
            Mode::Provider => prove_with_backend_and_rng(
                backend,
                self.ppi.clone(),
                &self.ir_bytes,
                &self.resolver,
                rng,
            )
            .await
            .map(|(proof, _)| proof)
            .map_err(|_| anyhow!("provider proof/verification failed")),
            Mode::Resident => {
                backend.ensure_available()?;
                let _backend = GpuBlitzarGuard::set(backend == ProverBackend::Cuda);
                let (proof, pis, _) = self
                    .ir
                    .prove(rng, &*PUBLIC_PARAMS, self.pk.clone(), &self.ppi)
                    .await?;
                let params = PUBLIC_PARAMS.get_params(self.pk.init()?.k()).await?;
                self.vk
                    .verify(&params.as_verifier(), &proof, pis.into_iter())?;
                Ok(proof)
            }
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let raw = std::fs::read(&args.request)?;
    let start = Instant::now();
    let (ppi, data, binding): (
        ProofPreimageVersioned,
        Option<ProvingKeyMaterial>,
        Option<Fr>,
    ) = tagged_deserialize(&raw[..])?;
    let decode_ms = start.elapsed().as_secs_f64() * 1e3;
    let ProofPreimageVersioned::V2(ppi) = ppi else {
        return Err(anyhow!("requires V2 proof preimage"));
    };
    let mut ppi = (*ppi).clone();
    if let Some(binding) = binding {
        ppi.binding_input = binding;
    }
    let data = data.ok_or_else(|| anyhow!("requires embedded custom proving key"))?;
    let ir: IrSource = tagged_deserialize(&data.ir_source[..])?;
    let vk: VerifierKey = tagged_deserialize(&data.verifier_key[..])?;
    let start = Instant::now();
    let pk = IrSource::load_prover_key_from_tagged(Cursor::new(&data.prover_key[..]))?;
    let load_pk_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let k = pk.init()?.k();
    let init_pk_ms = start.elapsed().as_secs_f64() * 1e3;
    let ir_bytes = data.ir_source.clone();
    let resolver = Resolver::new(
        PUBLIC_PARAMS.clone(),
        DustResolver(MidnightDataProvider::new(
            FetchMode::OnDemand,
            OutputMode::Log,
            ledger::dust::DUST_EXPECTED_FILES.to_owned(),
        )?),
        Box::new(move |_: KeyLocation| Box::pin(std::future::ready(Ok(Some(data.clone()))))),
    );
    let fixture = Fixture {
        ppi: Arc::new(ppi),
        ir_bytes,
        ir,
        pk,
        vk,
        resolver,
    };
    println!(
        "{}",
        serde_json::json!({"event":"setup", "request_bytes":raw.len(),"k":k,
        "decode_ms":decode_ms,"load_pk_ms":load_pk_ms,"init_pk_ms":init_pk_ms,
        "mode":format!("{:?}",args.mode),"backend":format!("{:?}",args.backend)})
    );
    drop(raw);
    let mut failures = 0;
    for i in 0..=args.iterations {
        let seed = args.seed + i;
        // Alternate order to avoid systematically granting one backend warm CPU caches.
        let backends = if args.cpu_parity {
            if i % 2 == 0 {
                vec![ProverBackend::Cpu, args.backend]
            } else {
                vec![args.backend, ProverBackend::Cpu]
            }
        } else {
            vec![args.backend]
        };
        let mut results = Vec::new();
        for backend in backends {
            let start = Instant::now();
            let result = fixture.run(args.mode, backend, seed).await;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if result.is_err() {
                failures += 1;
            }
            println!(
                "{}",
                serde_json::json!({"event":"proof","iteration":i,"warmup":i==0,
                "seed":seed,"backend":format!("{:?}",backend),"mode":format!("{:?}",args.mode),
                "ms":ms,"valid":result.is_ok()})
            );
            results.push(result);
        }
        if args.cpu_parity {
            let equal = matches!((&results[0], &results[1]), (Ok(a), Ok(b)) if a == b);
            if !equal {
                failures += 1;
            }
            println!(
                "{}",
                serde_json::json!({"event":"parity","iteration":i,"seed":seed,"equal":equal})
            );
        }
    }
    ensure!(
        failures == 0,
        "{failures} proof or parity failures (private details suppressed)"
    );
    Ok(())
}
