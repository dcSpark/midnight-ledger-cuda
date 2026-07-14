// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use base_crypto::data_provider::{FetchMode, MidnightDataProvider, OutputMode};
use base_crypto::rng::SplittableRng;
use clap::{Parser, ValueEnum};
use coin_structure::{
    coin::{Info as CoinInfo, QualifiedInfo as QualifiedCoinInfo},
    transfer::Recipient,
};
use ledger::dust::DustResolver;
use ledger::prove::Resolver;
use ledger::structure::{
    INITIAL_TRANSACTION_COST_MODEL, ProofPreimageMarker, Signature, Transaction,
};
use ledger::test_utilities::TestState;
use ledger::verify::WellFormedStrictness;
use midnight_proof_server::endpoints::PUBLIC_PARAMS;
use midnight_proof_server::prover_backend::ProverBackend;
use midnight_proof_server::versioned_ir::{
    prove_current_ir_with_backend_and_rng, prove_with_backend_and_rng,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serialize::tagged_serialize;
use storage::db::InMemoryDB;
use transient_crypto::commitment::PedersenRandomness;
use transient_crypto::merkle_tree::MerkleTree;
use transient_crypto::proofs::{
    KeyLocation, Proof, ProofPreimage, ProvingKeyMaterial, ProvingProvider, Resolver as ResolverT,
};

const ZSWAP_OUTPUT_BENCH_KEY: &str = "proof-backend-parity/zswap-output-current";
const ZSWAP_SPEND_BENCH_KEY: &str = "proof-backend-parity/zswap-spend-current";
const ZSWAP_OUTPUT_KEY: &str = "midnight/zswap/output";
const ZSWAP_SPEND_KEY: &str = "midnight/zswap/spend";
const DUST_SPEND_KEY: &str = "midnight/dust/spend";
const REBALANCE_INPUT_VALUES: [u128; 2] = [60, 70];
const REBALANCE_PAYMENT_VALUE: u128 = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchBackend {
    Cpu,
    Cuda,
    Both,
}

impl BenchBackend {
    fn selected(self) -> &'static [ProverBackend] {
        match self {
            Self::Cpu => &[ProverBackend::Cpu],
            Self::Cuda => &[ProverBackend::Cuda],
            Self::Both => &[ProverBackend::Cpu, ProverBackend::Cuda],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum FixtureKind {
    ZswapTransaction,
    MinimalV2,
    HashChainV2,
    ZswapOutput,
    ZswapSpend,
}

impl fmt::Display for FixtureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZswapTransaction => f.write_str("zswap-transaction"),
            Self::MinimalV2 => f.write_str("minimal-v2"),
            Self::HashChainV2 => f.write_str("hash-chain-v2"),
            Self::ZswapOutput => f.write_str("zswap-output"),
            Self::ZswapSpend => f.write_str("zswap-spend"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixturePipeline {
    Versioned,
    CurrentProver,
}

impl fmt::Display for FixturePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Versioned => f.write_str("versioned"),
            Self::CurrentProver => f.write_str("current-prover-generated-key"),
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_enum, default_value_t = BenchBackend::Both)]
    backend: BenchBackend,
    #[arg(long, value_enum, default_value_t = FixtureKind::ZswapTransaction)]
    fixture: FixtureKind,
    #[arg(long, default_value_t = 64)]
    hash_rounds: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 0x5eed)]
    seed: u64,
    #[arg(long)]
    json: Option<PathBuf>,
}

struct Fixture {
    resolver: Arc<Resolver>,
    payload: FixturePayload,
    kind: FixtureKind,
    instructions: usize,
    k: u8,
    rows: usize,
    proof_jobs: usize,
    pipeline: FixturePipeline,
    rebalance: Option<RebalanceMetadata>,
}

enum FixturePayload {
    Single {
        ir_source: Vec<u8>,
        ppi: Arc<ProofPreimage>,
    },
    ZswapTransaction(TransactionTemplate),
}

struct TransactionTemplate {
    state: TestState<InMemoryDB>,
    recipient_keys: zswap::keys::SecretKeys,
    current_materials: Arc<HashMap<String, ProvingKeyMaterial>>,
}

#[derive(Clone, Copy)]
struct RebalanceMetadata {
    input_total: u128,
    payment: u128,
    change: u128,
    spends: usize,
    outputs: usize,
    dust_spends: usize,
}

#[derive(Clone, Copy)]
struct TransactionShape {
    spends: usize,
    outputs: usize,
    dust_spends: usize,
}

type UnprovenTransaction =
    Transaction<Signature, ProofPreimageMarker, PedersenRandomness, InMemoryDB>;

struct BackendReport {
    backend: ProverBackend,
    iterations: usize,
    total_ms: f64,
    iteration_ms: Vec<f64>,
    transfer_h2d_bytes: Vec<u64>,
    transfer_d2h_bytes: Vec<u64>,
    transfer_breakdown: Vec<TransferBucketSnapshot>,
    proof_bytes: Vec<Vec<u8>>,
    skips: Vec<Vec<Option<usize>>>,
    adapter_digests: Vec<Option<String>>,
}

impl BackendReport {
    fn mean_ms(&self) -> f64 {
        self.total_ms / self.iterations as f64
    }

    fn median_ms(&self) -> f64 {
        let mut values = self.iteration_ms.clone();
        values.sort_by(f64::total_cmp);
        let midpoint = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[midpoint - 1] + values[midpoint]) / 2.0
        } else {
            values[midpoint]
        }
    }

    fn total_h2d_bytes(&self) -> u64 {
        self.transfer_h2d_bytes.iter().sum()
    }

    fn total_d2h_bytes(&self) -> u64 {
        self.transfer_d2h_bytes.iter().sum()
    }
}

#[derive(Clone, Debug, Default)]
struct TransferSnapshot {
    h2d_bytes: u64,
    d2h_bytes: u64,
    breakdown: Vec<TransferBucketSnapshot>,
}

#[derive(Clone, Debug, Default)]
struct TransferBucketSnapshot {
    name: String,
    h2d_bytes: u64,
    d2h_bytes: u64,
}

impl TransferBucketSnapshot {
    fn total_bytes(&self) -> u64 {
        self.h2d_bytes.saturating_add(self.d2h_bytes)
    }

    fn add_assign(&mut self, other: &Self) {
        self.h2d_bytes = self.h2d_bytes.saturating_add(other.h2d_bytes);
        self.d2h_bytes = self.d2h_bytes.saturating_add(other.d2h_bytes);
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

async fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.iterations == 0 {
        return Err("--iterations must be greater than zero".into());
    }

    let fixture = build_fixture(&args).await?;
    println!(
        "fixture={} pipeline={} proof_jobs={} instructions={} max_k={} aggregate_rows={}",
        fixture.kind,
        fixture.pipeline,
        fixture.proof_jobs,
        fixture.instructions,
        fixture.k,
        fixture.rows
    );
    if let Some(rebalance) = fixture.rebalance {
        println!(
            "rebalance=spends:{} outputs:{} dust_spends:{} input_total:{} payment:{} change:{}",
            rebalance.spends,
            rebalance.outputs,
            rebalance.dust_spends,
            rebalance.input_total,
            rebalance.payment,
            rebalance.change
        );
    }
    let mut reports = Vec::new();
    for backend in args.backend.selected() {
        let report = run_backend(*backend, &fixture, &args).await?;
        print_report(&report);
        reports.push(report);
    }

    let mut byte_exact = None;
    if reports.len() == 2 {
        assert_byte_exact(&reports[0], &reports[1])?;
        byte_exact = Some(true);
        println!(
            "byte_exact=ok cpu_median_ms={:.3} gpu_median_ms={:.3} speedup={:.3}x",
            reports[0].median_ms(),
            reports[1].median_ms(),
            reports[0].median_ms() / reports[1].median_ms()
        );
    }

    if let Some(path) = &args.json {
        write_json_report(path, &args, &fixture, &reports, byte_exact)?;
        println!("json={}", path.display());
    }

    Ok(())
}

async fn build_fixture(args: &Args) -> Result<Fixture, String> {
    if args.fixture == FixtureKind::ZswapTransaction {
        return build_zswap_transaction_fixture().await;
    }

    let (ir, ppi, instructions, pipeline) = match args.fixture {
        FixtureKind::ZswapTransaction => unreachable!("handled above"),
        FixtureKind::MinimalV2 => {
            let ir = zkir::IrSource::load(minimal_v2_ir().as_bytes())
                .map_err(|error| error.to_string())?;
            let ppi = ProofPreimage {
                binding_input: 42.into(),
                communications_commitment: None,
                inputs: vec![1.into()],
                private_transcript: vec![],
                public_transcript_inputs: vec![],
                public_transcript_outputs: vec![],
                key_location: KeyLocation(Cow::Borrowed("proof-backend-parity/minimal-v2")),
            };
            (ir, ppi, 1, FixturePipeline::Versioned)
        }
        FixtureKind::HashChainV2 => {
            if args.hash_rounds == 0 {
                return Err("--hash-rounds must be greater than zero for hash-chain-v2".into());
            }
            let ir = zkir::IrSource::load(hash_chain_v2_ir(args.hash_rounds).as_bytes())
                .map_err(|error| error.to_string())?;
            let ppi = ProofPreimage {
                binding_input: 42.into(),
                communications_commitment: None,
                inputs: vec![1.into(), 2.into(), 3.into()],
                private_transcript: vec![],
                public_transcript_inputs: vec![],
                public_transcript_outputs: vec![],
                key_location: KeyLocation(Cow::Borrowed("proof-backend-parity/hash-chain-v2")),
            };
            (ir, ppi, args.hash_rounds, FixturePipeline::Versioned)
        }
        FixtureKind::ZswapOutput => {
            let ir =
                zkir::IrSource::load(include_str!("../../../zswap/static/output.zkir").as_bytes())
                    .map_err(|error| error.to_string())?;
            let instructions = ir.instructions.len();
            let mut fixture_rng = StdRng::seed_from_u64(0x42);
            let keys = zswap::keys::SecretKeys::from_rng_seed(&mut fixture_rng);
            let coin = CoinInfo::new(&mut fixture_rng, 100, Default::default());
            let output = zswap::Output::<ProofPreimage, InMemoryDB>::new(
                &mut fixture_rng,
                &coin,
                None,
                &keys.coin_public_key(),
                Some(keys.enc_public_key()),
            )
            .map_err(|error| error.to_string())?;
            let mut ppi = (*output.proof).clone();
            ppi.key_location = KeyLocation(Cow::Borrowed(ZSWAP_OUTPUT_BENCH_KEY));
            (ir, ppi, instructions, FixturePipeline::CurrentProver)
        }
        FixtureKind::ZswapSpend => {
            let ir =
                zkir::IrSource::load(include_str!("../../../zswap/static/spend.zkir").as_bytes())
                    .map_err(|error| error.to_string())?;
            let instructions = ir.instructions.len();
            let mut fixture_rng = StdRng::seed_from_u64(0x42);
            let qualified_coin = QualifiedCoinInfo {
                value: Default::default(),
                type_: Default::default(),
                nonce: fixture_rng.r#gen(),
                mt_index: 0,
            };
            let coin = CoinInfo::from(&qualified_coin);
            let recipient = Recipient::Contract(Default::default());
            let tree = MerkleTree::<(), InMemoryDB>::blank(32)
                .try_update_hash(0, coin.commitment(&recipient).0, ())
                .map_err(|error| error.to_string())?
                .rehash();
            let input = zswap::Input::<ProofPreimage, InMemoryDB>::new_contract_owned(
                &mut fixture_rng,
                &qualified_coin,
                None,
                Default::default(),
                &tree,
            )
            .map_err(|error| error.to_string())?;
            let mut ppi = (*input.proof).clone();
            ppi.key_location = KeyLocation(Cow::Borrowed(ZSWAP_SPEND_BENCH_KEY));
            (ir, ppi, instructions, FixturePipeline::CurrentProver)
        }
    };
    let params = builtin_resolver(None);
    let model = ir.model();
    let material = current_key_material(&ir, &params).await?;
    let ir_source = material.ir_source.clone();
    let resolver = builtin_resolver(Some(material));
    let ppi = Arc::new(ppi);

    Ok(Fixture {
        resolver: Arc::new(resolver),
        payload: FixturePayload::Single { ir_source, ppi },
        kind: args.fixture,
        instructions,
        k: model.k(),
        rows: model.rows(),
        proof_jobs: 1,
        pipeline,
        rebalance: None,
    })
}

async fn build_zswap_transaction_fixture() -> Result<Fixture, String> {
    let spend_ir =
        zkir::IrSource::load(include_str!("../../../zswap/static/spend.zkir").as_bytes())
            .map_err(|error| error.to_string())?;
    let output_ir =
        zkir::IrSource::load(include_str!("../../../zswap/static/output.zkir").as_bytes())
            .map_err(|error| error.to_string())?;
    let dust_ir =
        zkir::IrSource::load(include_str!("../../../ledger/static/dust/spend.zkir").as_bytes())
            .map_err(|error| error.to_string())?;
    let spend_model = spend_ir.model();
    let output_model = output_ir.model();
    let dust_model = dust_ir.model();
    let params = builtin_resolver(None);
    let spend_material = current_key_material(&spend_ir, &params).await?;
    let output_material = current_key_material(&output_ir, &params).await?;
    let dust_material = current_key_material(&dust_ir, &params).await?;
    let resolver = builtin_resolver(None);
    let template = build_transaction_template(HashMap::from([
        (ZSWAP_SPEND_KEY.to_owned(), spend_material),
        (ZSWAP_OUTPUT_KEY.to_owned(), output_material),
        (DUST_SPEND_KEY.to_owned(), dust_material),
    ]))
    .await?;
    let mut preview_rng = StdRng::seed_from_u64(0x7478_7072_6576_6965);
    let preview = build_balanced_transaction(&template, &resolver, &mut preview_rng).await?;
    let shape = transaction_shape(&preview)?;
    if shape.spends != 2 || shape.outputs != 2 || shape.dust_spends == 0 {
        return Err(format!(
            "unexpected zswap transaction shape: spends={} outputs={} dust_spends={}",
            shape.spends, shape.outputs, shape.dust_spends
        ));
    }
    let payload = FixturePayload::ZswapTransaction(template);
    let input_total = REBALANCE_INPUT_VALUES.iter().sum::<u128>();
    let change = input_total - REBALANCE_PAYMENT_VALUE;

    Ok(Fixture {
        resolver: Arc::new(resolver),
        payload,
        kind: FixtureKind::ZswapTransaction,
        instructions: shape.spends * spend_ir.instructions.len()
            + shape.outputs * output_ir.instructions.len()
            + shape.dust_spends * dust_ir.instructions.len(),
        k: spend_model.k().max(output_model.k()).max(dust_model.k()),
        rows: shape.spends * spend_model.rows()
            + shape.outputs * output_model.rows()
            + shape.dust_spends * dust_model.rows(),
        proof_jobs: shape.spends + shape.outputs + shape.dust_spends,
        pipeline: FixturePipeline::CurrentProver,
        rebalance: Some(RebalanceMetadata {
            input_total,
            payment: REBALANCE_PAYMENT_VALUE,
            change,
            spends: shape.spends,
            outputs: shape.outputs,
            dust_spends: shape.dust_spends,
        }),
    })
}

async fn current_key_material(
    ir: &zkir::IrSource,
    params: &Resolver,
) -> Result<ProvingKeyMaterial, String> {
    let (pk, vk) = ir.v2_keygen(params).await.map_err(|e| e.to_string())?;
    let mut ir_source = Vec::new();
    ir.serialize_to_tagged(&mut ir_source)
        .map_err(|e| e.to_string())?;
    let mut prover_key = Vec::new();
    zkir::IrSource::serialize_prover_key_to_tagged(ir.version, &pk, &mut prover_key)
        .map_err(|e| e.to_string())?;
    let mut verifier_key = Vec::new();
    tagged_serialize(&vk, &mut verifier_key).map_err(|e| e.to_string())?;
    Ok(ProvingKeyMaterial {
        prover_key: prover_key.into(),
        verifier_key,
        ir_source,
    })
}

async fn build_transaction_template(
    current_materials: HashMap<String, ProvingKeyMaterial>,
) -> Result<TransactionTemplate, String> {
    if std::env::var_os("MIDNIGHT_LEDGER_TEST_STATIC_DIR").is_none() {
        return Err(
            "MIDNIGHT_LEDGER_TEST_STATIC_DIR must be set for transaction fixture setup; gpuprof sets it automatically"
                .to_owned(),
        );
    }
    let mut rng = StdRng::seed_from_u64(0x5a5a_42);
    let mut state = TestState::new(&mut rng);
    state.give_fee_token(&mut rng, 1).await;
    for value in REBALANCE_INPUT_VALUES {
        state.rewards_shielded(&mut rng, Default::default(), value);
    }
    let recipient_keys = zswap::keys::SecretKeys::from_rng_seed(&mut rng);

    Ok(TransactionTemplate {
        state,
        recipient_keys,
        current_materials: Arc::new(current_materials),
    })
}

fn minimal_v2_ir() -> String {
    r#"{
       "version": { "major": 2, "minor": 2 },
       "num_inputs": 1,
       "do_communications_commitment": false,
       "instructions": [
           { "op": "assert", "cond": 0 }
       ]
    }"#
    .to_owned()
}

fn hash_chain_v2_ir(rounds: usize) -> String {
    let mut out = String::from(
        r#"{
       "version": { "major": 2, "minor": 2 },
       "num_inputs": 3,
       "do_communications_commitment": false,
       "instructions": [
"#,
    );
    let mut prev = 0usize;
    for round in 0..rounds {
        let suffix = if round + 1 == rounds { "\n" } else { ",\n" };
        out.push_str(&format!(
            r#"           {{ "op": "transient_hash", "inputs": [{prev}, 1, 2] }}{suffix}"#
        ));
        prev = 3 + round;
    }
    out.push_str(
        r#"       ]
    }"#,
    );
    out
}

fn builtin_resolver(data: Option<ProvingKeyMaterial>) -> Resolver {
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
        Box::new(move |_| Box::pin(std::future::ready(Ok(data.clone())))),
    )
}

async fn build_balanced_transaction(
    template: &TransactionTemplate,
    resolver: &Resolver,
    rng: &mut StdRng,
) -> Result<UnprovenTransaction, String> {
    let token_type = Default::default();
    let payment_coin = CoinInfo::new(rng, REBALANCE_PAYMENT_VALUE, token_type);
    let payment_output = zswap::Output::new(
        rng,
        &payment_coin,
        None,
        &template.recipient_keys.coin_public_key(),
        Some(template.recipient_keys.enc_public_key()),
    )
    .map_err(|error| error.to_string())?;
    let offer = zswap::Offer {
        inputs: Vec::new().into(),
        outputs: vec![payment_output].into(),
        transient: Vec::new().into(),
        deltas: vec![zswap::Delta {
            token_type,
            value: -(REBALANCE_PAYMENT_VALUE as i128),
        }]
        .into(),
    };
    let transaction: UnprovenTransaction = Transaction::new(
        "local-test",
        Default::default(),
        Some(offer),
        Default::default(),
    );
    let mut state = template.state.clone();
    let balanced = state
        .balance_tx(rng.split(), transaction, resolver)
        .await
        .map_err(|error| error.to_string())?;
    let mut strictness = WellFormedStrictness::default();
    strictness.verify_native_proofs = false;
    strictness.verify_contract_proofs = false;
    strictness.verify_signatures = false;
    balanced
        .well_formed(&state.ledger, strictness, state.time)
        .map_err(|error| error.to_string())?;
    Ok(balanced)
}

fn transaction_shape(transaction: &UnprovenTransaction) -> Result<TransactionShape, String> {
    let Transaction::Standard(transaction) = transaction else {
        return Err("zswap benchmark unexpectedly produced a rewards transaction".to_owned());
    };
    let offer = transaction
        .guaranteed_coins
        .as_ref()
        .ok_or("balanced transaction has no guaranteed zswap offer")?;
    if !transaction.fallible_coins.is_empty() || !offer.transient.is_empty() {
        return Err("zswap benchmark unexpectedly produced fallible or transient coins".to_owned());
    }
    let dust_spends = transaction
        .intents
        .iter()
        .map(|entry| {
            entry
                .1
                .dust_actions
                .as_ref()
                .map_or(0, |actions| actions.spends.len())
        })
        .sum();
    Ok(TransactionShape {
        spends: offer.inputs.len(),
        outputs: offer.outputs.len(),
        dust_spends,
    })
}

#[derive(Clone)]
struct CurrentMaterialResolver {
    materials: Arc<HashMap<String, ProvingKeyMaterial>>,
    fallback: Arc<Resolver>,
}

impl CurrentMaterialResolver {
    fn has_current_material(&self, location: &KeyLocation) -> bool {
        self.materials.contains_key(location.0.as_ref())
    }
}

impl ResolverT for CurrentMaterialResolver {
    async fn resolve_key(&self, key: KeyLocation) -> io::Result<Option<ProvingKeyMaterial>> {
        if let Some(material) = self.materials.get(key.0.as_ref()) {
            Ok(Some(material.clone()))
        } else {
            self.fallback.resolve_key(key).await
        }
    }
}

struct CurrentTransactionProvider<R: Rng + rand::CryptoRng + SplittableRng> {
    rng: R,
    resolver: CurrentMaterialResolver,
    parallel: bool,
}

async fn prove_current_transaction_job<R>(
    rng: R,
    resolver: CurrentMaterialResolver,
    preimage: ProofPreimage,
    overwrite_binding_input: Option<transient_crypto::curve::Fr>,
    use_current: bool,
) -> Result<Proof, anyhow::Error>
where
    R: Rng + rand::CryptoRng + SplittableRng,
{
    let inner = zkir::LocalProvingProvider {
        rng,
        resolver: &resolver,
        params: &*PUBLIC_PARAMS,
    };
    if use_current {
        inner
            .prove_current_with_skips(&preimage, overwrite_binding_input)
            .await
            .map(|(proof, _)| proof)
    } else {
        inner
            .prove_with_skips(&preimage, overwrite_binding_input)
            .await
            .map(|(proof, _)| proof)
    }
}

impl<R> ProvingProvider for CurrentTransactionProvider<R>
where
    R: Rng + rand::CryptoRng + SplittableRng + Clone + Send + 'static,
{
    async fn check(&self, preimage: &ProofPreimage) -> Result<Vec<Option<usize>>, anyhow::Error> {
        zkir::LocalProvingProvider {
            rng: self.rng.clone(),
            resolver: &self.resolver,
            params: &*PUBLIC_PARAMS,
        }
        .check(preimage)
        .await
    }

    async fn prove(
        self,
        preimage: &ProofPreimage,
        overwrite_binding_input: Option<transient_crypto::curve::Fr>,
    ) -> Result<Proof, anyhow::Error> {
        let use_current = self.resolver.has_current_material(&preimage.key_location);
        let key_location = preimage.key_location.0.to_string();
        let result = if self.parallel {
            let preimage = preimage.clone();
            tokio::task::spawn_blocking(move || {
                futures::executor::block_on(prove_current_transaction_job(
                    self.rng,
                    self.resolver,
                    preimage,
                    overwrite_binding_input,
                    use_current,
                ))
            })
            .await
            .map_err(|error| anyhow::anyhow!("CUDA proof worker failed: {error}"))?
        } else {
            prove_current_transaction_job(
                self.rng,
                self.resolver,
                preimage.clone(),
                overwrite_binding_input,
                use_current,
            )
            .await
        };
        result.map_err(|error| anyhow::anyhow!("proof {key_location}: {error:#}"))
    }

    fn split(&mut self) -> Self {
        Self {
            rng: self.rng.split(),
            resolver: self.resolver.clone(),
            parallel: self.parallel,
        }
    }

    fn resolver(&self) -> &impl ResolverT {
        &self.resolver
    }
}

async fn prove_zswap_transaction(
    backend: ProverBackend,
    template: &TransactionTemplate,
    resolver: &Arc<Resolver>,
    seed: u64,
) -> Result<(Vec<u8>, Vec<Option<usize>>), String> {
    backend
        .ensure_available()
        .map_err(|error| error.to_string())?;
    #[cfg(feature = "gpu-blitzar")]
    let _gpu_blitzar =
        transient_crypto::proofs::GpuBlitzarGuard::set(backend == ProverBackend::Cuda);

    let mut root_rng = StdRng::seed_from_u64(seed);
    let mut transaction_rng = root_rng.split();
    let proof_rng = root_rng.split();
    let seal_rng = root_rng.split();
    let transaction =
        build_balanced_transaction(template, resolver.as_ref(), &mut transaction_rng).await?;
    let shape = transaction_shape(&transaction)?;
    if shape.spends != 2 || shape.outputs != 2 || shape.dust_spends == 0 {
        return Err(format!(
            "zswap transaction shape changed during benchmark: spends={} outputs={} dust_spends={}",
            shape.spends, shape.outputs, shape.dust_spends
        ));
    }
    let current_resolver = CurrentMaterialResolver {
        materials: template.current_materials.clone(),
        fallback: resolver.clone(),
    };
    let provider = CurrentTransactionProvider {
        rng: proof_rng,
        resolver: current_resolver,
        parallel: backend == ProverBackend::Cuda,
    };
    let proven_transaction = transaction
        .prove(provider, &INITIAL_TRANSACTION_COST_MODEL.runtime_cost_model)
        .await
        .map_err(|error| error.to_string())?;
    let sealed_transaction = proven_transaction.seal(seal_rng);
    let mut bytes = Vec::new();
    tagged_serialize(&sealed_transaction, &mut bytes).map_err(|error| error.to_string())?;
    Ok((bytes, Vec::new()))
}

async fn run_backend(
    backend: ProverBackend,
    fixture: &Fixture,
    args: &Args,
) -> Result<BackendReport, String> {
    let mut proof_bytes = Vec::with_capacity(args.iterations);
    let mut skips = Vec::with_capacity(args.iterations);
    let mut adapter_digests = Vec::with_capacity(args.iterations);
    let mut transfer_h2d_bytes = Vec::with_capacity(args.iterations);
    let mut transfer_d2h_bytes = Vec::with_capacity(args.iterations);
    let mut iteration_ms = Vec::with_capacity(args.iterations);
    let mut transfer_breakdown: Vec<TransferBucketSnapshot> = Vec::new();
    let mut total_ms = 0.0;

    for round in 0..(args.warmup + args.iterations) {
        let seed = args.seed.wrapping_add(round as u64);
        reset_gpu_transfer_counters();
        let start = Instant::now();
        let (result_bytes, round_skips) = match &fixture.payload {
            FixturePayload::Single { ir_source, ppi } => {
                let rng = StdRng::seed_from_u64(seed);
                let (proof, skips) = match fixture.pipeline {
                    FixturePipeline::Versioned => {
                        prove_with_backend_and_rng(
                            backend,
                            ppi.clone(),
                            ir_source,
                            &fixture.resolver,
                            rng,
                        )
                        .await?
                    }
                    FixturePipeline::CurrentProver => {
                        prove_current_ir_with_backend_and_rng(
                            backend,
                            ppi.clone(),
                            &fixture.resolver,
                            rng,
                        )
                        .await?
                    }
                };
                (proof.0, skips)
            }
            FixturePayload::ZswapTransaction(template) => {
                prove_zswap_transaction(backend, template, &fixture.resolver, seed)
                    .await
                    .map_err(|error| {
                        format!("backend={backend} round={round} seed={seed}: {error}")
                    })?
            }
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let transfer = gpu_transfer_counters();
        if round >= args.warmup {
            total_ms += elapsed_ms;
            iteration_ms.push(elapsed_ms);
            transfer_h2d_bytes.push(transfer.h2d_bytes);
            transfer_d2h_bytes.push(transfer.d2h_bytes);
            merge_transfer_breakdown(&mut transfer_breakdown, &transfer.breakdown);
            adapter_digests.push(adapter_digest(&result_bytes));
            proof_bytes.push(result_bytes);
            skips.push(round_skips);
        }
    }

    Ok(BackendReport {
        backend,
        iterations: args.iterations,
        total_ms,
        iteration_ms,
        transfer_h2d_bytes,
        transfer_d2h_bytes,
        transfer_breakdown,
        proof_bytes,
        skips,
        adapter_digests,
    })
}

fn print_report(report: &BackendReport) {
    let result_len = report.proof_bytes.last().map_or(0, std::vec::Vec::len);
    let adapter_digest = report
        .adapter_digests
        .last()
        .and_then(Option::as_deref)
        .unwrap_or("unlinked");
    let top_transfer = report
        .transfer_breakdown
        .iter()
        .filter(|bucket| bucket.total_bytes() > 0)
        .max_by_key(|bucket| bucket.total_bytes())
        .map(|bucket| {
            format!(
                "{}:{:.3}/{:.3}MB",
                bucket.name,
                bucket.h2d_bytes as f64 / (1024.0 * 1024.0),
                bucket.d2h_bytes as f64 / (1024.0 * 1024.0)
            )
        })
        .unwrap_or_else(|| "none".to_string());
    println!(
        "backend={} iterations={} total_ms={:.3} mean_ms={:.3} median_ms={:.3} h2d_mb={:.3} d2h_mb={:.3} top_transfer={} result_bytes={} adapter_digest={}",
        report.backend,
        report.iterations,
        report.total_ms,
        report.mean_ms(),
        report.median_ms(),
        report.total_h2d_bytes() as f64 / (1024.0 * 1024.0),
        report.total_d2h_bytes() as f64 / (1024.0 * 1024.0),
        top_transfer,
        result_len,
        adapter_digest
    );
}

fn write_json_report(
    path: &PathBuf,
    args: &Args,
    fixture: &Fixture,
    reports: &[BackendReport],
    byte_exact: Option<bool>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mean_speedup = (reports.len() == 2).then(|| reports[0].mean_ms() / reports[1].mean_ms());
    let median_speedup =
        (reports.len() == 2).then(|| reports[0].median_ms() / reports[1].median_ms());
    let reports = reports
        .iter()
        .map(|report| {
            serde_json::json!({
                "backend": report.backend.to_string(),
                "iterations": report.iterations,
                "total_ms": report.total_ms,
                "mean_ms": report.mean_ms(),
                "median_ms": report.median_ms(),
                "iteration_ms": report.iteration_ms,
                "transfer_h2d_bytes": report.transfer_h2d_bytes,
                "transfer_d2h_bytes": report.transfer_d2h_bytes,
                "transfer_total_h2d_bytes": report.total_h2d_bytes(),
                "transfer_total_d2h_bytes": report.total_d2h_bytes(),
                "transfer_breakdown": report.transfer_breakdown.iter().map(|bucket| {
                    serde_json::json!({
                        "name": bucket.name,
                        "h2d_bytes": bucket.h2d_bytes,
                        "d2h_bytes": bucket.d2h_bytes,
                    })
                }).collect::<Vec<_>>(),
                "proof_lengths": report.proof_bytes.iter().map(Vec::len).collect::<Vec<_>>(),
                "proof_digests": report.proof_bytes.iter().map(|bytes| proof_digest(bytes)).collect::<Vec<_>>(),
                "skip_outputs": report.skips,
                "adapter_digests": report.adapter_digests,
            })
        })
        .collect::<Vec<_>>();
    let rebalance = fixture.rebalance.map(|rebalance| {
        serde_json::json!({
            "spends": rebalance.spends,
            "outputs": rebalance.outputs,
            "dust_spends": rebalance.dust_spends,
            "input_values": REBALANCE_INPUT_VALUES,
            "input_total": rebalance.input_total,
            "payment": rebalance.payment,
            "change": rebalance.change,
        })
    });
    let body = serde_json::json!({
        "gate": "proof-backend-parity",
        "fixture": {
            "kind": fixture.kind.to_string(),
            "pipeline": fixture.pipeline.to_string(),
            "instructions": fixture.instructions,
            "k": fixture.k,
            "rows": fixture.rows,
            "proof_jobs": fixture.proof_jobs,
            "result_kind": if fixture.kind == FixtureKind::ZswapTransaction {
                "serialized-ledger-transaction"
            } else {
                "proof"
            },
            "rebalance": rebalance,
            "hash_rounds": if fixture.kind == FixtureKind::HashChainV2 {
                Some(args.hash_rounds)
            } else {
                None
            },
        },
        "args": {
            "backend": format!("{:?}", args.backend).to_lowercase(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "byte_exact": byte_exact,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "reports": reports,
    });
    let bytes = serde_json::to_vec_pretty(&body).map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

fn proof_digest(bytes: &[u8]) -> String {
    let mut xor = 0u8;
    let mut sum64 = 0u64;
    let mut rolling = 0u64;
    for (idx, byte) in bytes.iter().copied().enumerate() {
        xor ^= byte;
        sum64 = sum64.wrapping_add(byte as u64);
        rolling = rolling
            .wrapping_mul(0x100_0000_01b3)
            .wrapping_add(byte as u64)
            .wrapping_add(idx as u64);
    }
    format!(
        "len:{} xor:{} sum64:{} rolling:{}",
        bytes.len(),
        xor,
        sum64,
        rolling
    )
}

fn assert_byte_exact(cpu: &BackendReport, gpu: &BackendReport) -> Result<(), String> {
    if cpu.backend != ProverBackend::Cpu || gpu.backend != ProverBackend::Cuda {
        return Err("byte-exact comparison expects CPU report followed by CUDA report".into());
    }
    if cpu.proof_bytes.len() != gpu.proof_bytes.len() {
        return Err(format!(
            "iteration count mismatch: cpu={} gpu={}",
            cpu.proof_bytes.len(),
            gpu.proof_bytes.len()
        ));
    }
    for (idx, ((cpu_proof, gpu_proof), (cpu_skips, gpu_skips))) in cpu
        .proof_bytes
        .iter()
        .zip(&gpu.proof_bytes)
        .zip(cpu.skips.iter().zip(&gpu.skips))
        .enumerate()
    {
        if cpu_proof != gpu_proof {
            let first_mismatch = cpu_proof
                .iter()
                .zip(gpu_proof)
                .position(|(cpu, gpu)| cpu != gpu)
                .unwrap_or(cpu_proof.len().min(gpu_proof.len()));
            return Err(format!(
                "proof byte mismatch at iteration {idx}: offset={first_mismatch} cpu_len={} gpu_len={}",
                cpu_proof.len(),
                gpu_proof.len()
            ));
        }
        if cpu_skips != gpu_skips {
            return Err(format!(
                "skip-output mismatch at iteration {idx}: cpu={cpu_skips:?} gpu={gpu_skips:?}"
            ));
        }
        if cpu.adapter_digests.get(idx) != gpu.adapter_digests.get(idx) {
            return Err(format!(
                "adapter digest mismatch at iteration {idx}: cpu={:?} gpu={:?}",
                cpu.adapter_digests.get(idx),
                gpu.adapter_digests.get(idx)
            ));
        }
    }
    Ok(())
}

fn merge_transfer_breakdown(
    totals: &mut Vec<TransferBucketSnapshot>,
    next: &[TransferBucketSnapshot],
) {
    if totals.is_empty() {
        totals.extend_from_slice(next);
        return;
    }

    for bucket in next {
        if let Some(total) = totals.iter_mut().find(|total| total.name == bucket.name) {
            total.add_assign(bucket);
        } else {
            totals.push(bucket.clone());
        }
    }
}

fn reset_gpu_transfer_counters() {
    #[cfg(feature = "gpu-blitzar")]
    midnight_sppark_msm::reset_transfer_counters();
}

fn gpu_transfer_counters() -> TransferSnapshot {
    #[cfg(feature = "gpu-blitzar")]
    {
        let counters = midnight_sppark_msm::transfer_counters();
        let breakdown = midnight_sppark_msm::transfer_breakdown()
            .into_iter()
            .map(|bucket| TransferBucketSnapshot {
                name: bucket.name.to_string(),
                h2d_bytes: bucket.h2d_bytes,
                d2h_bytes: bucket.d2h_bytes,
            })
            .collect();
        return TransferSnapshot {
            h2d_bytes: counters.h2d_bytes,
            d2h_bytes: counters.d2h_bytes,
            breakdown,
        };
    }

    #[cfg(not(feature = "gpu-blitzar"))]
    TransferSnapshot::default()
}

#[cfg(feature = "cuda")]
fn adapter_digest(bytes: &[u8]) -> Option<String> {
    let digest = proof_server_cuda::proof_byte_digest(bytes);
    Some(format!(
        "len:{} xor:{} sum64:{} rolling:{}",
        digest.len, digest.xor, digest.sum64, digest.rolling
    ))
}

#[cfg(not(feature = "cuda"))]
fn adapter_digest(_bytes: &[u8]) -> Option<String> {
    None
}
