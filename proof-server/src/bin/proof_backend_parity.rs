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
use std::time::{Duration, Instant};

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
use midnight_proof_server::prover_backend::{ProofExecutionRoute, ProverBackend};
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
const METAL_QUOTIENT_DIAGNOSTIC_ENV: &str = "MIDNIGHT_METAL_QUOTIENT_DIAGNOSTIC";
const METAL_QUOTIENT_SUBSTITUTE_ENV: &str = "MIDNIGHT_METAL_QUOTIENT_SUBSTITUTE";
const RAW_BLS12_381_FR_BYTES: u64 = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchBackend {
    Cpu,
    Cuda,
    Metal,
    MetalHybrid,
    Both,
    CpuMetal,
    CpuMetalHybrid,
}

impl BenchBackend {
    fn selected(self) -> &'static [ProverBackend] {
        match self {
            Self::Cpu => &[ProverBackend::Cpu],
            Self::Cuda => &[ProverBackend::Cuda],
            Self::Metal => &[ProverBackend::Metal],
            Self::MetalHybrid => &[ProverBackend::MetalHybrid],
            Self::Both => &[ProverBackend::Cpu, ProverBackend::Cuda],
            Self::CpuMetal => &[ProverBackend::Cpu, ProverBackend::Metal],
            Self::CpuMetalHybrid => &[ProverBackend::Cpu, ProverBackend::MetalHybrid],
        }
    }
}

impl fmt::Display for BenchBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::MetalHybrid => "metal-hybrid",
            Self::Both => "both",
            Self::CpuMetal => "cpu-metal",
            Self::CpuMetalHybrid => "cpu-metal-hybrid",
        })
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
    /// Idle time after warmups, before the first measured pair.
    #[arg(long, default_value_t = 0)]
    settle_seconds: u64,
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
    ZswapTransaction(Box<TransactionTemplate>),
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
    execution_route: ProofExecutionRoute,
    iterations: usize,
    total_ms: f64,
    seeds: Vec<u64>,
    iteration_ms: Vec<f64>,
    transfer_h2d_bytes: Vec<u64>,
    transfer_d2h_bytes: Vec<u64>,
    transfer_breakdown: Vec<TransferBucketSnapshot>,
    native_calls: Vec<u64>,
    compute_calls: Vec<u64>,
    fallback_counts: Vec<u64>,
    metal_msm: Vec<MetalMsmSnapshot>,
    metal_quotient: Vec<MetalQuotientSnapshot>,
    proof_bytes: Vec<Vec<u8>>,
    skips: Vec<Vec<Option<usize>>>,
    adapter_digests: Vec<Option<String>>,
}

impl BackendReport {
    fn new(
        backend: ProverBackend,
        execution_route: ProofExecutionRoute,
        iterations: usize,
    ) -> Self {
        Self {
            backend,
            execution_route,
            iterations,
            total_ms: 0.0,
            seeds: Vec::with_capacity(iterations),
            iteration_ms: Vec::with_capacity(iterations),
            transfer_h2d_bytes: Vec::with_capacity(iterations),
            transfer_d2h_bytes: Vec::with_capacity(iterations),
            transfer_breakdown: Vec::new(),
            native_calls: Vec::with_capacity(iterations),
            compute_calls: Vec::with_capacity(iterations),
            fallback_counts: Vec::with_capacity(iterations),
            metal_msm: Vec::with_capacity(iterations),
            metal_quotient: Vec::with_capacity(iterations),
            proof_bytes: Vec::with_capacity(iterations),
            skips: Vec::with_capacity(iterations),
            adapter_digests: Vec::with_capacity(iterations),
        }
    }

    fn record(&mut self, seed: u64, sample: BackendSample) {
        self.total_ms += sample.elapsed_ms;
        self.seeds.push(seed);
        self.iteration_ms.push(sample.elapsed_ms);
        self.transfer_h2d_bytes.push(sample.transfer.h2d_bytes);
        self.transfer_d2h_bytes.push(sample.transfer.d2h_bytes);
        merge_transfer_breakdown(&mut self.transfer_breakdown, &sample.transfer.breakdown);
        self.native_calls.push(sample.accelerator.native_calls);
        self.compute_calls.push(sample.accelerator.compute_calls);
        self.fallback_counts.push(sample.accelerator.fallbacks);
        self.metal_msm.push(sample.accelerator.metal_msm);
        self.metal_quotient.push(sample.accelerator.metal_quotient);
        self.proof_bytes.push(sample.result_bytes);
        self.skips.push(sample.skips);
    }

    fn mean_ms(&self) -> f64 {
        self.total_ms / self.iterations as f64
    }

    fn median_ms(&self) -> f64 {
        let mut values = self.iteration_ms.clone();
        values.sort_by(f64::total_cmp);
        let midpoint = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[midpoint - 1] + values[midpoint]) / 2.0
        } else {
            values[midpoint]
        }
    }

    fn p95_ms(&self) -> f64 {
        nearest_rank_percentile(&self.iteration_ms, 95)
    }

    fn runs_per_second(&self) -> f64 {
        1_000.0 / self.mean_ms()
    }

    fn proofs_per_second(&self, proof_jobs: usize) -> f64 {
        self.runs_per_second() * proof_jobs as f64
    }

    fn total_h2d_bytes(&self) -> u64 {
        self.transfer_h2d_bytes.iter().sum()
    }

    fn total_d2h_bytes(&self) -> u64 {
        self.transfer_d2h_bytes.iter().sum()
    }

    fn total_native_calls(&self) -> u64 {
        self.native_calls.iter().sum()
    }

    fn total_compute_calls(&self) -> u64 {
        self.compute_calls.iter().sum()
    }

    fn total_fallbacks(&self) -> u64 {
        self.fallback_counts.iter().sum()
    }

    fn total_metal_msm(&self) -> MetalMsmSnapshot {
        self.metal_msm.iter().copied().fold(
            MetalMsmSnapshot::default(),
            MetalMsmSnapshot::saturating_add,
        )
    }

    fn median_quotient_ms(&self) -> Option<f64> {
        let mut values = self
            .metal_quotient
            .iter()
            .filter(|sample| sample.epochs != 0)
            .map(|sample| sample.elapsed_ns as f64 / 1e6)
            .collect::<Vec<_>>();
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let midpoint = values.len() / 2;
        Some(if values.len().is_multiple_of(2) {
            (values[midpoint - 1] + values[midpoint]) / 2.0
        } else {
            values[midpoint]
        })
    }
}

struct BackendSample {
    elapsed_ms: f64,
    transfer: TransferSnapshot,
    accelerator: AcceleratorSnapshot,
    result_bytes: Vec<u8>,
    skips: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AcceleratorSnapshot {
    native_calls: u64,
    compute_calls: u64,
    fallbacks: u64,
    metal_msm: MetalMsmSnapshot,
    metal_quotient: MetalQuotientSnapshot,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MetalMsmSnapshot {
    metal_submissions: u64,
    metal_outputs: u64,
    metal_logical_scalars: u64,
    metal_nonzero_scalars: u64,
    metal_failures: u64,
    metal_service_ns: u64,
    metal_coalesce_ns: u64,
    metal_canonicalization_ns: u64,
    policy_cpu_outputs: u64,
    policy_cpu_logical_scalars: u64,
    policy_cpu_nonzero_scalars: u64,
    policy_cpu_service_ns: u64,
    policy_cpu_lock_wait_ns: u64,
}

impl MetalMsmSnapshot {
    fn saturating_add(mut self, rhs: Self) -> Self {
        self.metal_submissions = self.metal_submissions.saturating_add(rhs.metal_submissions);
        self.metal_outputs = self.metal_outputs.saturating_add(rhs.metal_outputs);
        self.metal_logical_scalars = self
            .metal_logical_scalars
            .saturating_add(rhs.metal_logical_scalars);
        self.metal_nonzero_scalars = self
            .metal_nonzero_scalars
            .saturating_add(rhs.metal_nonzero_scalars);
        self.metal_failures = self.metal_failures.saturating_add(rhs.metal_failures);
        self.metal_service_ns = self.metal_service_ns.saturating_add(rhs.metal_service_ns);
        self.metal_coalesce_ns = self.metal_coalesce_ns.saturating_add(rhs.metal_coalesce_ns);
        self.metal_canonicalization_ns = self
            .metal_canonicalization_ns
            .saturating_add(rhs.metal_canonicalization_ns);
        self.policy_cpu_outputs = self
            .policy_cpu_outputs
            .saturating_add(rhs.policy_cpu_outputs);
        self.policy_cpu_logical_scalars = self
            .policy_cpu_logical_scalars
            .saturating_add(rhs.policy_cpu_logical_scalars);
        self.policy_cpu_nonzero_scalars = self
            .policy_cpu_nonzero_scalars
            .saturating_add(rhs.policy_cpu_nonzero_scalars);
        self.policy_cpu_service_ns = self
            .policy_cpu_service_ns
            .saturating_add(rhs.policy_cpu_service_ns);
        self.policy_cpu_lock_wait_ns = self
            .policy_cpu_lock_wait_ns
            .saturating_add(rhs.policy_cpu_lock_wait_ns);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MetalQuotientSnapshot {
    epochs: u64,
    elapsed_ns: u64,
    failures: u64,
    failure_elapsed_ns: u64,
    cpu_msm_leases: u64,
    cpu_msm_lease_bytes: u64,
    resident_msm_leases: u64,
    resident_msm_lease_bytes: u64,
    materializations: u64,
    whole_polynomial_materializations: u64,
    clear_failures: u64,
    poison_events: u64,
    quarantined_secret_bytes: u64,
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
    if args.settle_seconds > 600 {
        return Err("--settle-seconds must not exceed 600".into());
    }

    let selected_backends = args.backend.selected();
    let execution_routes = selected_backends
        .iter()
        .map(|backend| {
            backend
                .execution_route()
                .map_err(|error| format!("backend preflight failed for {backend}: {error}"))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let fixture = build_fixture(&args).await?;
    validate_metal_quotient_diagnostic_configuration(&args, selected_backends, &fixture)?;
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
    let reports = run_backends(selected_backends, &execution_routes, &fixture, &args).await?;
    for report in &reports {
        print_report(report, fixture.proof_jobs);
    }

    let mut byte_exact = None;
    if reports.len() == 2 {
        assert_byte_exact(&reports[0], &reports[1])?;
        byte_exact = Some(true);
        println!(
            "byte_exact=ok cpu_median_ms={:.3} accelerator_median_ms={:.3} speedup={:.3}x",
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

fn metal_quotient_diagnostic_requested() -> bool {
    matches!(
        std::env::var(METAL_QUOTIENT_DIAGNOSTIC_ENV).as_deref(),
        Ok("1")
    )
}

fn metal_quotient_substitute_requested() -> bool {
    match std::env::var(METAL_QUOTIENT_SUBSTITUTE_ENV) {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "metal" | "auto"
        ),
        Err(std::env::VarError::NotUnicode(_)) => false,
    }
}

fn validate_metal_quotient_diagnostic_configuration(
    args: &Args,
    backends: &[ProverBackend],
    fixture: &Fixture,
) -> Result<(), String> {
    let diagnostic = metal_quotient_diagnostic_requested();
    let metal_selected = backends.contains(&ProverBackend::MetalHybrid);
    if !metal_selected {
        if diagnostic {
            return Err(
                "the Metal quotient diagnostic requires the metal-hybrid backend".to_owned(),
            );
        }
        return Ok(());
    }
    if !cfg!(feature = "metal") {
        return Err(
            "the Metal route requires resident quotient support; enable the metal Cargo feature"
                .to_owned(),
        );
    }
    if backends
        .iter()
        .any(|backend| matches!(backend, ProverBackend::Cuda | ProverBackend::Metal))
    {
        return Err(
            "the Metal quotient diagnostic requires metal-hybrid, optionally paired with CPU"
                .to_owned(),
        );
    }
    if !diagnostic {
        return Ok(());
    }
    if fixture.kind != FixtureKind::ZswapTransaction || fixture.proof_jobs != 5 {
        return Err(format!(
            "the representative quotient gate requires the five-proof zswap transaction fixture, got fixture={} proof_jobs={}",
            fixture.kind, fixture.proof_jobs
        ));
    }
    if args.warmup < 2 || args.iterations < 5 {
        return Err(
            "the representative quotient gate requires at least 2 warmups and 5 timed iterations"
                .to_owned(),
        );
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
    if shape.spends != 2 || shape.outputs != 2 || shape.dust_spends != 1 {
        return Err(format!(
            "unexpected zswap transaction shape: spends={} outputs={} dust_spends={}",
            shape.spends, shape.outputs, shape.dust_spends
        ));
    }
    let payload = FixturePayload::ZswapTransaction(Box::new(template));
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
    let mut rng = StdRng::seed_from_u64(0x005a_5a42);
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
    slots: Option<Arc<tokio::sync::Semaphore>>,
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
            let permit = match self.slots.clone() {
                Some(slots) => Some(
                    slots
                        .acquire_owned()
                        .await
                        .map_err(|_| anyhow::anyhow!("accelerated proof slots were closed"))?,
                ),
                None => None,
            };
            let preimage = preimage.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                futures::executor::block_on(prove_current_transaction_job(
                    self.rng,
                    self.resolver,
                    preimage,
                    overwrite_binding_input,
                    use_current,
                ))
            })
            .await
            .map_err(|error| anyhow::anyhow!("accelerated proof worker failed: {error}"))?
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
            slots: self.slots.clone(),
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
    let execution = backend
        .enter_proof_execution()
        .map_err(|error| error.to_string())?;

    let mut root_rng = StdRng::seed_from_u64(seed);
    let mut transaction_rng = root_rng.split();
    let proof_rng = root_rng.split();
    let seal_rng = root_rng.split();
    let transaction =
        build_balanced_transaction(template, resolver.as_ref(), &mut transaction_rng).await?;
    let shape = transaction_shape(&transaction)?;
    if shape.spends != 2 || shape.outputs != 2 || shape.dust_spends != 1 {
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
        parallel: execution.route().supports_parallel_proof_jobs(),
        slots: metal_proof_slots(execution.route())?,
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

fn metal_proof_slots(
    route: ProofExecutionRoute,
) -> Result<Option<Arc<tokio::sync::Semaphore>>, String> {
    if route != ProofExecutionRoute::MetalHybrid {
        return Ok(None);
    }
    let slots = match std::env::var("MIDNIGHT_METAL_PARALLEL_PROOFS") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            format!("MIDNIGHT_METAL_PARALLEL_PROOFS must be an integer, got {value:?}")
        })?,
        Err(std::env::VarError::NotPresent) => 5,
        Err(error) => {
            return Err(format!(
                "could not read MIDNIGHT_METAL_PARALLEL_PROOFS: {error}"
            ));
        }
    };
    if !(1..=8).contains(&slots) {
        return Err(format!(
            "MIDNIGHT_METAL_PARALLEL_PROOFS must be in 1..=8, got {slots}"
        ));
    }
    Ok(Some(Arc::new(tokio::sync::Semaphore::new(slots))))
}

async fn run_backends(
    backends: &[ProverBackend],
    execution_routes: &[ProofExecutionRoute],
    fixture: &Fixture,
    args: &Args,
) -> Result<Vec<BackendReport>, String> {
    debug_assert!(!backends.is_empty() && backends.len() <= 2);
    debug_assert_eq!(backends.len(), execution_routes.len());
    let mut reports = backends
        .iter()
        .copied()
        .zip(execution_routes.iter().copied())
        .map(|(backend, route)| BackendReport::new(backend, route, args.iterations))
        .collect::<Vec<_>>();

    for round in 0..(args.warmup + args.iterations) {
        if round == args.warmup && args.settle_seconds != 0 {
            println!(
                "settle_seconds={} phase=between-warmup-and-measurement",
                args.settle_seconds
            );
            tokio::time::sleep(Duration::from_secs(args.settle_seconds)).await;
        }
        let seed = args.seed.wrapping_add(round as u64);
        for report_index in round_backend_order(backends.len(), round) {
            let backend = backends[report_index];
            let sample = run_backend_sample(backend, fixture, round, seed).await?;
            if round >= args.warmup {
                reports[report_index].record(seed, sample);
            }
        }
    }

    debug_assert!(
        reports
            .iter()
            .all(|report| report.iteration_ms.len() == args.iterations)
    );
    // Adapter diagnostics are deliberately deferred until every timed sample
    // completes so their CPU or GPU work cannot perturb the paired rounds.
    let diagnostic_backend = adapter_diagnostic_backend(backends);
    for report in &mut reports {
        report.adapter_digests = report
            .proof_bytes
            .iter()
            .map(|bytes| adapter_digest(diagnostic_backend, bytes))
            .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(reports)
}

fn adapter_diagnostic_backend(backends: &[ProverBackend]) -> Option<ProverBackend> {
    backends
        .iter()
        .copied()
        .find(|backend| *backend != ProverBackend::Cpu)
}

fn round_backend_order(backend_count: usize, round: usize) -> Vec<usize> {
    match backend_count {
        1 => vec![0],
        2 if round.is_multiple_of(2) => vec![0, 1],
        2 => vec![1, 0],
        _ => panic!("parity benchmark supports one backend or one CPU/accelerator pair"),
    }
}

async fn run_backend_sample(
    backend: ProverBackend,
    fixture: &Fixture,
    round: usize,
    seed: u64,
) -> Result<BackendSample, String> {
    reset_accelerator_activity_counters(backend)?;
    let start = Instant::now();
    let result = match &fixture.payload {
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
                    .await
                }
                FixturePipeline::CurrentProver => {
                    prove_current_ir_with_backend_and_rng(
                        backend,
                        ppi.clone(),
                        &fixture.resolver,
                        rng,
                    )
                    .await
                }
            }?;
            Ok((proof.0, skips))
        }
        FixturePayload::ZswapTransaction(template) => {
            prove_zswap_transaction(backend, template, &fixture.resolver, seed).await
        }
    }
    .map_err(|error: String| format!("backend={backend} round={round} seed={seed}: {error}"))?;

    let accelerator = accelerator_activity_counters(backend)?;
    match backend {
        ProverBackend::Cpu if accelerator != AcceleratorSnapshot::default() => {
            return Err(format!(
                "backend={backend} round={round} seed={seed}: CPU route observed native_calls={} compute_calls={} fallbacks={}",
                accelerator.native_calls, accelerator.compute_calls, accelerator.fallbacks
            ));
        }
        ProverBackend::Cuda | ProverBackend::MetalHybrid if accelerator.compute_calls == 0 => {
            return Err(format!(
                "backend={backend} round={round} seed={seed}: no native accelerator compute call was observed"
            ));
        }
        ProverBackend::Cuda | ProverBackend::MetalHybrid if accelerator.fallbacks != 0 => {
            return Err(format!(
                "backend={backend} round={round} seed={seed}: observed {} explicit accelerator fallback(s)",
                accelerator.fallbacks
            ));
        }
        ProverBackend::Cpu
        | ProverBackend::Cuda
        | ProverBackend::Metal
        | ProverBackend::MetalHybrid => {}
    }
    validate_metal_quotient_sample(backend, fixture, accelerator)
        .map_err(|error| format!("backend={backend} round={round} seed={seed}: {error}"))?;

    Ok(BackendSample {
        elapsed_ms: start.elapsed().as_secs_f64() * 1e3,
        transfer: accelerator_transfer_counters(backend)?,
        accelerator,
        result_bytes: result.0,
        skips: result.1,
    })
}

fn validate_metal_quotient_sample(
    backend: ProverBackend,
    fixture: &Fixture,
    accelerator: AcceleratorSnapshot,
) -> Result<(), String> {
    let substitute = metal_quotient_substitute_requested();
    let diagnostic = metal_quotient_diagnostic_requested();
    let quotient = accelerator.metal_quotient;
    if backend == ProverBackend::Cpu {
        if quotient != MetalQuotientSnapshot::default() {
            return Err("CPU route observed Metal quotient activity".to_owned());
        }
        return Ok(());
    }
    if !substitute && !diagnostic {
        if quotient != MetalQuotientSnapshot::default() {
            return Err("disabled Metal quotient path reported activity".to_owned());
        }
        return Ok(());
    }
    if backend != ProverBackend::MetalHybrid {
        return Err("resident Metal quotient ran on a non-hybrid accelerator route".to_owned());
    }
    let expected_epochs = u64::try_from(fixture.proof_jobs)
        .map_err(|_| "fixture proof-job count overflowed".to_owned())?;
    if quotient.epochs != expected_epochs
        || quotient.cpu_msm_leases != 0
        || quotient.resident_msm_leases != expected_epochs
    {
        return Err(format!(
            "expected quotient epochs={expected_epochs} CPU leases=0 resident leases={expected_epochs}, observed epochs={} CPU leases={} resident leases={}",
            quotient.epochs, quotient.cpu_msm_leases, quotient.resident_msm_leases,
        ));
    }
    if quotient.elapsed_ns == 0 || quotient.resident_msm_lease_bytes == 0 {
        return Err(
            "quotient diagnostic reported no elapsed time or resident lease bytes".to_owned(),
        );
    }
    if !quotient
        .cpu_msm_lease_bytes
        .is_multiple_of(RAW_BLS12_381_FR_BYTES)
    {
        return Err("quotient CPU-MSM lease bytes are not RawFr-aligned".to_owned());
    }
    if !quotient
        .resident_msm_lease_bytes
        .is_multiple_of(RAW_BLS12_381_FR_BYTES)
    {
        return Err("quotient resident-MSM lease bytes are not RawFr-aligned".to_owned());
    }
    if quotient.cpu_msm_lease_bytes != 0 {
        return Err(format!(
            "resident quotient unexpectedly exposed {} CPU-MSM scalar bytes",
            quotient.cpu_msm_lease_bytes
        ));
    }
    if quotient.failures != 0
        || quotient.failure_elapsed_ns != 0
        || quotient.clear_failures != 0
        || quotient.poison_events != 0
        || quotient.quarantined_secret_bytes != 0
    {
        return Err(format!(
            "quotient lifecycle failure: failures={} failure_elapsed_ns={} clear_failures={} poison_events={} quarantined_secret_bytes={}",
            quotient.failures,
            quotient.failure_elapsed_ns,
            quotient.clear_failures,
            quotient.poison_events,
            quotient.quarantined_secret_bytes
        ));
    }
    let expected_materializations = if substitute { expected_epochs } else { 0 };
    if quotient.materializations != expected_materializations
        || quotient.whole_polynomial_materializations != expected_materializations
    {
        return Err(format!(
            "resident quotient materialization mismatch: expected={expected_materializations} materializations={} whole={}",
            quotient.materializations, quotient.whole_polynomial_materializations,
        ));
    }
    Ok(())
}

fn print_report(report: &BackendReport, proof_jobs: usize) {
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
    let msm = report.total_metal_msm();
    println!(
        "backend={} execution_route={} native_calls={} compute_calls={} fallback_count={} msm_metal_submissions={} msm_metal_outputs={} msm_metal_nonzero_scalars={} msm_metal_service_ms={:.3} msm_metal_canonicalization_work_ms={:.3} msm_policy_cpu_outputs={} msm_policy_cpu_nonzero_scalars={} msm_policy_cpu_service_ms={:.3} msm_policy_cpu_lock_wait_work_ms={:.3} msm_metal_failures={} iterations={} total_ms={:.3} mean_ms={:.3} median_ms={:.3} p95_ms={:.3} runs_s={:.3} proofs_s={:.3} h2d_mb={:.3} d2h_mb={:.3} top_transfer={} result_bytes={} adapter_digest={}",
        report.backend,
        report.execution_route,
        report.total_native_calls(),
        report.total_compute_calls(),
        report.total_fallbacks(),
        msm.metal_submissions,
        msm.metal_outputs,
        msm.metal_nonzero_scalars,
        msm.metal_service_ns as f64 / 1e6,
        msm.metal_canonicalization_ns as f64 / 1e6,
        msm.policy_cpu_outputs,
        msm.policy_cpu_nonzero_scalars,
        msm.policy_cpu_service_ns as f64 / 1e6,
        msm.policy_cpu_lock_wait_ns as f64 / 1e6,
        msm.metal_failures,
        report.iterations,
        report.total_ms,
        report.mean_ms(),
        report.median_ms(),
        report.p95_ms(),
        report.runs_per_second(),
        report.proofs_per_second(proof_jobs),
        report.total_h2d_bytes() as f64 / (1024.0 * 1024.0),
        report.total_d2h_bytes() as f64 / (1024.0 * 1024.0),
        top_transfer,
        result_len,
        adapter_digest
    );
    if let Some(median_ms) = report.median_quotient_ms() {
        let mode = if metal_quotient_substitute_requested() {
            "resident-substitution-with-host-opening"
        } else {
            "discarded-shadow-prefix"
        };
        println!(
            "backend={} metal_quotient_mode={mode} metal_quotient_epochs={} median_aggregate_ms={median_ms:.3}",
            report.backend,
            report
                .metal_quotient
                .iter()
                .map(|sample| sample.epochs)
                .sum::<u64>()
        );
        if metal_quotient_diagnostic_requested() && !metal_quotient_substitute_requested() {
            println!(
                "backend={} metal_quotient_shadow_gate_ms=600.000 stage_gate_pass={}",
                report.backend,
                median_ms <= 600.0
            );
        }
    }
}

/// Returns the nearest-rank percentile used by the benchmark report.
///
/// Samples are never interpolated: p95 is the sample at `ceil(0.95 * n)` in
/// sorted order. This keeps reported tail latency tied to an observed proof.
fn nearest_rank_percentile(values: &[f64], percentile: usize) -> f64 {
    assert!(
        !values.is_empty(),
        "a percentile requires at least one sample"
    );
    assert!(
        (1..=100).contains(&percentile),
        "percentile must be in 1..=100"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile * sorted.len()).div_ceil(100);
    sorted[rank - 1]
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
    let p95_speedup = (reports.len() == 2).then(|| reports[0].p95_ms() / reports[1].p95_ms());
    let native_call_count = reports
        .iter()
        .map(BackendReport::total_native_calls)
        .sum::<u64>();
    let compute_call_count = reports
        .iter()
        .map(BackendReport::total_compute_calls)
        .sum::<u64>();
    let fallback_count = reports
        .iter()
        .map(BackendReport::total_fallbacks)
        .sum::<u64>();
    let execution_routes = reports
        .iter()
        .map(|report| report.execution_route.to_string())
        .collect::<Vec<_>>();
    let quotient_median_ms = reports
        .iter()
        .find(|report| report.backend == ProverBackend::MetalHybrid)
        .and_then(BackendReport::median_quotient_ms);
    let quotient_diagnostic = metal_quotient_diagnostic_requested().then(|| {
        serde_json::json!({
            "schema_version": 1,
            "mode": "discarded-shadow-prefix",
            "trusted_local_only": true,
            "aggregate_gate_ms": 600.0,
            "median_aggregate_ms": quotient_median_ms,
            "stage_gate_pass": quotient_median_ms.is_some_and(|value| value <= 600.0),
            "does_not_claim_end_to_end_speedup": true,
        })
    });
    let quotient_status = reports
        .iter()
        .any(|report| report.backend == ProverBackend::MetalHybrid)
        .then(|| {
            let substitute = metal_quotient_substitute_requested();
            serde_json::json!({
                "schema_version": 1,
                "mode": if substitute {
                    "resident-substitution-with-host-opening"
                } else if metal_quotient_diagnostic_requested() {
                    "discarded-shadow-prefix"
                } else {
                    "disabled"
                },
                "resident_commitments": substitute || metal_quotient_diagnostic_requested(),
                "combined_opening_materialized_to_host": substitute,
            })
        });
    let reports = reports
        .iter()
        .map(|report| {
            let metal_msm_total = report.total_metal_msm();
            serde_json::json!({
                "backend": report.backend.to_string(),
                "execution_route": report.execution_route.to_string(),
                "native_call_count": report.total_native_calls(),
                "native_calls": report.native_calls,
                "compute_call_count": report.total_compute_calls(),
                "compute_calls": report.compute_calls,
                "fallback_count": report.total_fallbacks(),
                "fallback_counts": report.fallback_counts,
                "metal_msm_total": {
                    "metal_submissions": metal_msm_total.metal_submissions,
                    "metal_outputs": metal_msm_total.metal_outputs,
                    "metal_logical_scalars": metal_msm_total.metal_logical_scalars,
                    "metal_nonzero_scalars": metal_msm_total.metal_nonzero_scalars,
                    "metal_failures": metal_msm_total.metal_failures,
                    "metal_service_ns": metal_msm_total.metal_service_ns,
                    "metal_coalesce_ns": metal_msm_total.metal_coalesce_ns,
                    "metal_canonicalization_ns": metal_msm_total.metal_canonicalization_ns,
                    "policy_cpu_outputs": metal_msm_total.policy_cpu_outputs,
                    "policy_cpu_logical_scalars": metal_msm_total.policy_cpu_logical_scalars,
                    "policy_cpu_nonzero_scalars": metal_msm_total.policy_cpu_nonzero_scalars,
                    "policy_cpu_service_ns": metal_msm_total.policy_cpu_service_ns,
                    "policy_cpu_lock_wait_ns": metal_msm_total.policy_cpu_lock_wait_ns,
                },
                "metal_msm": report.metal_msm.iter().map(|sample| {
                    serde_json::json!({
                        "metal_submissions": sample.metal_submissions,
                        "metal_outputs": sample.metal_outputs,
                        "metal_logical_scalars": sample.metal_logical_scalars,
                        "metal_nonzero_scalars": sample.metal_nonzero_scalars,
                        "metal_failures": sample.metal_failures,
                        "metal_service_ns": sample.metal_service_ns,
                        "metal_coalesce_ns": sample.metal_coalesce_ns,
                        "metal_canonicalization_ns": sample.metal_canonicalization_ns,
                        "policy_cpu_outputs": sample.policy_cpu_outputs,
                        "policy_cpu_logical_scalars": sample.policy_cpu_logical_scalars,
                        "policy_cpu_nonzero_scalars": sample.policy_cpu_nonzero_scalars,
                        "policy_cpu_service_ns": sample.policy_cpu_service_ns,
                        "policy_cpu_lock_wait_ns": sample.policy_cpu_lock_wait_ns,
                    })
                }).collect::<Vec<_>>(),
                "metal_quotient": report.metal_quotient.iter().map(|sample| {
                    serde_json::json!({
                        "epochs": sample.epochs,
                        "safe_elapsed_ns": sample.elapsed_ns,
                        "failures": sample.failures,
                        "failure_elapsed_ns": sample.failure_elapsed_ns,
                        "cpu_msm_leases": sample.cpu_msm_leases,
                        "cpu_msm_lease_bytes": sample.cpu_msm_lease_bytes,
                        "resident_msm_leases": sample.resident_msm_leases,
                        "resident_msm_lease_bytes": sample.resident_msm_lease_bytes,
                        "materializations": sample.materializations,
                        "whole_polynomial_materializations": sample.whole_polynomial_materializations,
                        "clear_failures": sample.clear_failures,
                        "poison_events": sample.poison_events,
                        "quarantined_secret_bytes": sample.quarantined_secret_bytes,
                    })
                }).collect::<Vec<_>>(),
                "iterations": report.iterations,
                "total_ms": report.total_ms,
                "mean_ms": report.mean_ms(),
                "median_ms": report.median_ms(),
                "p95_ms": report.p95_ms(),
                "runs_per_second": report.runs_per_second(),
                "proofs_per_second": report.proofs_per_second(fixture.proof_jobs),
                "seeds": report.seeds,
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
            "backend": args.backend.to_string(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "settle_seconds": args.settle_seconds,
            "seed": args.seed,
        },
        "byte_exact": byte_exact,
        "execution_route": execution_routes,
        "native_call_count": native_call_count,
        "compute_call_count": compute_call_count,
        "fallback_count": fallback_count,
        "metal_quotient": quotient_status,
        "metal_quotient_diagnostic": quotient_diagnostic,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "p95_speedup": p95_speedup,
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
    if cpu.backend != ProverBackend::Cpu || gpu.backend == ProverBackend::Cpu {
        return Err(
            "byte-exact comparison expects a CPU report followed by an accelerator report".into(),
        );
    }
    if cpu.proof_bytes.len() != gpu.proof_bytes.len() {
        return Err(format!(
            "iteration count mismatch: cpu={} gpu={}",
            cpu.proof_bytes.len(),
            gpu.proof_bytes.len()
        ));
    }
    if cpu.seeds != gpu.seeds {
        return Err(format!(
            "paired seed order mismatch: cpu={:?} accelerator={:?}",
            cpu.seeds, gpu.seeds
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

fn reset_accelerator_activity_counters(backend: ProverBackend) -> Result<(), String> {
    match backend {
        ProverBackend::Cpu | ProverBackend::Cuda => reset_cuda_activity_counters(backend),
        ProverBackend::Metal => Err(
            "pure Metal proof benchmarking is unavailable; use metal-hybrid until a native full-proof Metal route is connected"
                .to_owned(),
        ),
        ProverBackend::MetalHybrid => reset_metal_activity_counters(),
    }
}

fn reset_cuda_activity_counters(_backend: ProverBackend) -> Result<(), String> {
    #[cfg(feature = "gpu-blitzar")]
    {
        midnight_sppark_msm::reset_accelerator_activity_counters();
        return Ok(());
    }

    #[cfg(not(feature = "gpu-blitzar"))]
    match _backend {
        ProverBackend::Cpu => Ok(()),
        ProverBackend::Cuda => {
            Err("CUDA activity counters are unavailable without gpu-blitzar".to_owned())
        }
        ProverBackend::Metal | ProverBackend::MetalHybrid => {
            unreachable!("Metal routes are handled before the CUDA counter bridge")
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn reset_metal_activity_counters() -> Result<(), String> {
    midnight_gpu_prover::metal::reset_shared_metal_activity();
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn reset_metal_activity_counters() -> Result<(), String> {
    Err("Metal activity counters require the metal feature on Apple silicon macOS".to_owned())
}

fn accelerator_activity_counters(backend: ProverBackend) -> Result<AcceleratorSnapshot, String> {
    match backend {
        ProverBackend::Cpu | ProverBackend::Cuda => cuda_activity_counters(backend),
        ProverBackend::Metal => Err(
            "pure Metal proof benchmarking is unavailable; use metal-hybrid until a native full-proof Metal route is connected"
                .to_owned(),
        ),
        ProverBackend::MetalHybrid => metal_activity_counters(),
    }
}

fn cuda_activity_counters(_backend: ProverBackend) -> Result<AcceleratorSnapshot, String> {
    #[cfg(feature = "gpu-blitzar")]
    {
        let counters = midnight_sppark_msm::accelerator_activity_counters();
        return Ok(AcceleratorSnapshot {
            native_calls: counters.native_calls,
            compute_calls: counters.compute_calls,
            fallbacks: counters.fallbacks,
            ..AcceleratorSnapshot::default()
        });
    }

    #[cfg(not(feature = "gpu-blitzar"))]
    match _backend {
        ProverBackend::Cpu => Ok(AcceleratorSnapshot::default()),
        ProverBackend::Cuda => {
            Err("CUDA activity counters are unavailable without gpu-blitzar".to_owned())
        }
        ProverBackend::Metal | ProverBackend::MetalHybrid => {
            unreachable!("Metal routes are handled before the CUDA counter bridge")
        }
    }
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn metal_accelerator_snapshot(
    command_buffers: u64,
    dispatches: u64,
    execution_failures: u64,
) -> Result<AcceleratorSnapshot, String> {
    if execution_failures != 0 {
        return Err(format!(
            "Metal observed {execution_failures} failed command buffer completion(s)"
        ));
    }
    Ok(AcceleratorSnapshot {
        native_calls: command_buffers,
        compute_calls: dispatches,
        fallbacks: 0,
        ..AcceleratorSnapshot::default()
    })
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_activity_counters() -> Result<AcceleratorSnapshot, String> {
    let activity = midnight_gpu_prover::metal::shared_metal_activity();
    if activity.msm_metal_failures != 0 {
        return Err(format!(
            "Metal observed {} failed standalone MSM submission(s)",
            activity.msm_metal_failures
        ));
    }
    let mut snapshot = metal_accelerator_snapshot(
        activity.command_buffers,
        activity.dispatches,
        activity.execution_failures,
    )?;
    snapshot.metal_msm = MetalMsmSnapshot {
        metal_submissions: activity.msm_metal_submissions,
        metal_outputs: activity.msm_metal_outputs,
        metal_logical_scalars: activity.msm_metal_logical_scalars,
        metal_nonzero_scalars: activity.msm_metal_nonzero_scalars,
        metal_failures: activity.msm_metal_failures,
        metal_service_ns: activity.msm_metal_service_ns,
        metal_coalesce_ns: activity.msm_metal_coalesce_ns,
        metal_canonicalization_ns: activity.msm_metal_canonicalization_ns,
        policy_cpu_outputs: activity.msm_cpu_outputs,
        policy_cpu_logical_scalars: activity.msm_cpu_logical_scalars,
        policy_cpu_nonzero_scalars: activity.msm_cpu_nonzero_scalars,
        policy_cpu_service_ns: activity.msm_cpu_service_ns,
        policy_cpu_lock_wait_ns: activity.msm_cpu_lock_wait_ns,
    };
    snapshot.metal_quotient = MetalQuotientSnapshot {
        epochs: activity.quotient_epochs,
        elapsed_ns: activity.quotient_epoch_elapsed_ns,
        failures: activity.quotient_epoch_failures,
        failure_elapsed_ns: activity.quotient_epoch_failure_elapsed_ns,
        cpu_msm_leases: activity.cpu_msm_leases,
        cpu_msm_lease_bytes: activity.cpu_msm_lease_bytes,
        resident_msm_leases: activity.resident_msm_leases,
        resident_msm_lease_bytes: activity.resident_msm_lease_bytes,
        materializations: activity.quotient_materializations,
        whole_polynomial_materializations: activity.quotient_whole_polynomial_materializations,
        clear_failures: activity.clear_failures,
        poison_events: activity.poison_events,
        quarantined_secret_bytes: activity.quarantined_secret_bytes,
    };
    Ok(snapshot)
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn metal_activity_counters() -> Result<AcceleratorSnapshot, String> {
    Err("Metal activity counters require the metal feature on Apple silicon macOS".to_owned())
}

fn accelerator_transfer_counters(backend: ProverBackend) -> Result<TransferSnapshot, String> {
    match backend {
        ProverBackend::Cpu | ProverBackend::Cuda => cuda_transfer_counters(backend),
        ProverBackend::Metal => Err(
            "pure Metal proof benchmarking is unavailable; use metal-hybrid until a native full-proof Metal route is connected"
                .to_owned(),
        ),
        ProverBackend::MetalHybrid => metal_transfer_counters(),
    }
}

fn cuda_transfer_counters(_backend: ProverBackend) -> Result<TransferSnapshot, String> {
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
        return Ok(TransferSnapshot {
            h2d_bytes: counters.h2d_bytes,
            d2h_bytes: counters.d2h_bytes,
            breakdown,
        });
    }

    #[cfg(not(feature = "gpu-blitzar"))]
    match _backend {
        ProverBackend::Cpu => Ok(TransferSnapshot::default()),
        ProverBackend::Cuda => {
            Err("CUDA transfer counters are unavailable without gpu-blitzar".to_owned())
        }
        ProverBackend::Metal | ProverBackend::MetalHybrid => {
            unreachable!("Metal routes are handled before the CUDA counter bridge")
        }
    }
}

#[cfg(any(
    test,
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn metal_transfer_snapshot(host_write_bytes: u64, host_read_bytes: u64) -> TransferSnapshot {
    TransferSnapshot {
        h2d_bytes: host_write_bytes,
        d2h_bytes: host_read_bytes,
        breakdown: vec![TransferBucketSnapshot {
            name: "metal-unified-host-mappings".to_owned(),
            h2d_bytes: host_write_bytes,
            d2h_bytes: host_read_bytes,
        }],
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_transfer_counters() -> Result<TransferSnapshot, String> {
    let activity = midnight_gpu_prover::metal::shared_metal_activity();
    Ok(metal_transfer_snapshot(
        activity.host_write_bytes,
        activity.host_read_bytes,
    ))
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn metal_transfer_counters() -> Result<TransferSnapshot, String> {
    Err("Metal transfer counters require the metal feature on Apple silicon macOS".to_owned())
}

fn adapter_digest(backend: Option<ProverBackend>, bytes: &[u8]) -> Result<Option<String>, String> {
    match backend {
        None | Some(ProverBackend::Cpu) => Ok(None),
        Some(ProverBackend::Cuda) => cuda_adapter_digest(bytes).map(Some),
        Some(ProverBackend::Metal | ProverBackend::MetalHybrid) => {
            metal_adapter_digest(bytes).map(Some)
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_adapter_digest(bytes: &[u8]) -> Result<String, String> {
    let digest = proof_server_cuda::proof_byte_digest(bytes);
    Ok(format!(
        "len:{} xor:{} sum64:{} rolling:{}",
        digest.len, digest.xor, digest.sum64, digest.rolling
    ))
}

#[cfg(not(feature = "cuda"))]
fn cuda_adapter_digest(_bytes: &[u8]) -> Result<String, String> {
    Err("CUDA adapter digest is unavailable in this build".to_owned())
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_adapter_digest(bytes: &[u8]) -> Result<String, String> {
    let digest = midnight_gpu_prover::metal::proof_byte_digest_metal(bytes)
        .map_err(|error| format!("Metal adapter digest failed: {error}"))?;
    Ok(format!(
        "len:{} xor:{} sum64:{} rolling:{}",
        digest.len, digest.xor, digest.sum64, digest.rolling
    ))
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
fn metal_adapter_digest(_bytes: &[u8]) -> Result<String, String> {
    Err("Metal adapter digest is unavailable in this build".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_rounds_alternate_which_backend_runs_first() {
        assert_eq!(round_backend_order(2, 0), [0, 1]);
        assert_eq!(round_backend_order(2, 1), [1, 0]);
        assert_eq!(round_backend_order(2, 2), [0, 1]);
        assert_eq!(round_backend_order(2, 3), [1, 0]);
    }

    #[test]
    fn alternating_execution_preserves_each_report_in_seed_order() {
        let mut report_rounds = [Vec::new(), Vec::new()];
        for round in 0..6 {
            for backend_index in round_backend_order(2, round) {
                report_rounds[backend_index].push(round);
            }
        }

        assert_eq!(report_rounds[0], [0, 1, 2, 3, 4, 5]);
        assert_eq!(report_rounds[1], [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn p95_uses_an_observed_nearest_rank_sample() {
        let reversed = (1..=20).rev().map(f64::from).collect::<Vec<_>>();

        assert_eq!(nearest_rank_percentile(&[4.25], 95), 4.25);
        assert_eq!(nearest_rank_percentile(&reversed, 95), 19.0);
        assert_eq!(nearest_rank_percentile(&reversed, 100), 20.0);
    }

    #[test]
    fn cpu_adapter_diagnostic_does_not_invoke_an_accelerator() {
        assert_eq!(adapter_digest(None, b"proof"), Ok(None));
        assert_eq!(adapter_digest(Some(ProverBackend::Cpu), b"proof"), Ok(None));
        assert_eq!(adapter_diagnostic_backend(&[ProverBackend::Cpu]), None);
        assert_eq!(
            adapter_diagnostic_backend(&[ProverBackend::Cpu, ProverBackend::Metal]),
            Some(ProverBackend::Metal)
        );
        assert_eq!(
            adapter_diagnostic_backend(&[ProverBackend::Cpu, ProverBackend::Cuda]),
            Some(ProverBackend::Cuda)
        );
        assert_eq!(
            adapter_diagnostic_backend(&[ProverBackend::Cpu, ProverBackend::MetalHybrid]),
            Some(ProverBackend::MetalHybrid)
        );
    }

    #[test]
    fn pure_metal_sample_counters_remain_fail_closed() {
        let error = reset_accelerator_activity_counters(ProverBackend::Metal)
            .expect_err("pure Metal proofs must remain unavailable");
        assert!(error.contains("use metal-hybrid"));
    }

    #[test]
    fn hybrid_benchmark_modes_are_distinct_from_pure_metal() {
        assert_eq!(BenchBackend::MetalHybrid.to_string(), "metal-hybrid");
        assert_eq!(
            BenchBackend::MetalHybrid.selected(),
            &[ProverBackend::MetalHybrid]
        );
        assert_eq!(BenchBackend::CpuMetalHybrid.to_string(), "cpu-metal-hybrid");
        assert_eq!(
            BenchBackend::CpuMetalHybrid.selected(),
            &[ProverBackend::Cpu, ProverBackend::MetalHybrid]
        );
        assert_eq!(BenchBackend::Metal.selected(), &[ProverBackend::Metal]);
    }

    #[test]
    fn metal_activity_counts_native_commands_without_inventing_fallbacks() {
        assert_eq!(
            metal_accelerator_snapshot(7, 19, 0),
            Ok(AcceleratorSnapshot {
                native_calls: 7,
                compute_calls: 19,
                fallbacks: 0,
                ..AcceleratorSnapshot::default()
            })
        );
        let error = metal_accelerator_snapshot(7, 19, 1)
            .expect_err("failed Metal command buffers must fail the sample");
        assert!(error.contains("1 failed command buffer"));
    }

    #[test]
    fn metal_unified_memory_access_is_reported_as_a_named_transfer_bucket() {
        let transfer = metal_transfer_snapshot(1_024, 512);
        assert_eq!(transfer.h2d_bytes, 1_024);
        assert_eq!(transfer.d2h_bytes, 512);
        assert_eq!(transfer.breakdown.len(), 1);
        assert_eq!(transfer.breakdown[0].name, "metal-unified-host-mappings");
        assert_eq!(transfer.breakdown[0].h2d_bytes, 1_024);
        assert_eq!(transfer.breakdown[0].d2h_bytes, 512);
    }
}
