"""Versioned ownership map for proof-server CUDA/NVTX stage labels."""

TAXONOMY_VERSION = "proofscope-taxonomy-v3"

STAGE_ORDER = """
proof_server
proof_server.request proof_server.request.payload proof_server.request.decode
proof_server.request.resolve_ir proof_server.request.resolve_key
proof_server.prove proof_server.prove.dispatch proof_server.prove.ir_load
proof_server.prove.key_material proof_server.prove.provider
proof_server.prove.execute proof_server.prove.execute.witness
proof_server.prove.execute.commit proof_server.prove.execute.lookup
proof_server.prove.execute.sumcheck proof_server.prove.execute.open
proof_server.prove.execute.encode proof_server.prove.check
proof_server.prove.cuda proof_server.prove.cuda.params
proof_server.prove.cuda.transfer_h2d proof_server.prove.cuda.kernels
proof_server.prove.cuda.sync proof_server.prove.cuda.transfer_d2h
proof_server.prover proof_server.prover.stdlib
proof_server.prover.stdlib.format_instance
proof_server.prover.stdlib.format_committed_instances
proof_server.prover.stdlib.build_circuit
proof_server.prover.stdlib.plonk_prove proof_server.prover.stdlib.prove
proof_server.prover.plonk proof_server.prover.plonk.create_proof
proof_server.prover.plonk.compute_trace proof_server.prover.plonk.hash_vk
proof_server.prover.plonk.finalise_proof
proof_server.prover.plonk.compute_instances
proof_server.prover.plonk.parse_advices proof_server.prover.plonk.parse_advices.body
proof_server.prover.plonk.advice proof_server.prover.plonk.advice.synthesize
proof_server.prover.plonk.advice.batch_invert
proof_server.prover.plonk.advice.commit_lagrange_batch
proof_server.prover.plonk.advice.lagrange_to_coeff
proof_server.prover.plonk.lookup proof_server.prover.plonk.lookup.commit_permuted
proof_server.prover.plonk.lookup.commit_product
proof_server.prover.plonk.lookup.product_batch_invert
proof_server.prover.plonk.lookup.evaluate
proof_server.prover.plonk.permutation proof_server.prover.plonk.permutation.commit
proof_server.prover.plonk.permutation.numerator
proof_server.prover.plonk.permutation.denominator
proof_server.prover.plonk.permutation.batch_invert
proof_server.prover.plonk.permutation.prefix_product
proof_server.prover.plonk.permutation.blind
proof_server.prover.plonk.permutation.evaluate_common
proof_server.prover.plonk.permutation.evaluate
proof_server.prover.plonk.trash proof_server.prover.plonk.trash.commit
proof_server.prover.plonk.trash.evaluate
proof_server.prover.plonk.vanishing
proof_server.prover.plonk.vanishing.commit_random
proof_server.prover.plonk.vanishing.construct
proof_server.prover.plonk.vanishing.construct.divide_to_coeff.gpu_sppark
proof_server.prover.plonk.vanishing.construct.divide_by_vanishing
proof_server.prover.plonk.vanishing.construct.extended_to_coeff
proof_server.prover.plonk.vanishing.construct.truncate
proof_server.prover.plonk.vanishing.construct.split
proof_server.prover.plonk.vanishing.construct.blind_limbs
proof_server.prover.plonk.vanishing.construct.coeff_from_limbs
proof_server.prover.plonk.vanishing.construct.commit_batch
proof_server.prover.plonk.vanishing.construct.write_commitments
proof_server.prover.plonk.vanishing.evaluate
proof_server.prover.plonk.compute_nu_poly
proof_server.prover.plonk.compute_nu_poly.advice_cosets
proof_server.prover.plonk.compute_nu_poly.instance_cosets
proof_server.prover.plonk.compute_nu_poly.evaluate_numerator
proof_server.prover.plonk.evaluate_numerator
proof_server.prover.plonk.evaluate_numerator.custom_gates
proof_server.prover.plonk.evaluate_numerator.permutation
proof_server.prover.plonk.evaluate_numerator.lookup
proof_server.prover.plonk.evaluate_numerator.trash
proof_server.prover.plonk.write_evals
proof_server.prover.plonk.compute_queries
proof_server.prover.plonk.multi_open
proof_server.prover.kzg proof_server.prover.kzg.commit
proof_server.prover.kzg.commit.lagrange_msm
proof_server.prover.kzg.commit.lagrange_msm.gpu_blitzar
proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_blitzar
proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark
proof_server.prover.kzg.commit.coeff_msm
proof_server.prover.kzg.commit.coeff_msm.gpu_blitzar
proof_server.prover.kzg.commit.coeff_msm_batch.gpu_blitzar
proof_server.prover.kzg.commit.coeff_msm_batch.gpu_sppark
proof_server.prover.kzg.msm_specific
proof_server.prover.kzg.msm_specific.filtered
proof_server.prover.kzg.msm_specific.gpu_sppark
proof_server.prover.kzg.multi_open proof_server.prover.kzg.multi_open.body
proof_server.prover.kzg.multi_open.sort_point_sets
proof_server.prover.kzg.multi_open.q_polys
proof_server.prover.kzg.multi_open.q_eval_write
proof_server.prover.kzg.multi_open.f_poly
proof_server.prover.kzg.multi_open.f_commit
proof_server.prover.kzg.multi_open.f_commit_write
proof_server.prover.kzg.multi_open.final_poly
proof_server.prover.kzg.multi_open.final_eval
proof_server.prover.kzg.multi_open.pi_commit
proof_server.prover.kzg.multi_open.pi_write
proof_server.prover.kzg.multi_open.poly_lincomb.gpu_sppark
proof_server.prover.curves proof_server.prover.curves.best_fft
proof_server.prover.curves.msm_best proof_server.prover.curves.msm_parallel
proof_server.prover.curves.g1_msm_context
proof_server.prover.curves.g1_msm_context.create
proof_server.prover.curves.g1_msm_context.msm
proof_server.prover.curves.g1_msm_context.batch
proof_server.prover.curves.g1_msm_context.device_batch
proof_server.prover.curves.g1_msm_context.device_ragged
proof_server.prover.curves.g1_msm_context.device_scrambled_batch
proof_server.prover.kzg.commit.advice_resident.scrambled_batch
proof_server.prover.kzg.commit.advice_resident.upload
proof_server.prover.kzg.commit.coeff_msm_batch.gpu_sppark_device_scrambled
proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark_device
proof_server.prover.kzg.commit.lagrange_msm_batch.gpu_sppark_device_scrambled
proof_server.prover.kzg.commit.lagrange_msm_suffix_batch.cpu
proof_server.prover.kzg.commit.lagrange_msm_suffix_batch.resident_ragged
proof_server.prover.kzg.msm_specific.gpu_sppark_unfiltered
proof_server.prover.kzg.multi_open.f_poly.resident_gpu_sppark
proof_server.prover.kzg.multi_open.pi_commit.resident_gpu_sppark
proof_server.prover.kzg.multi_open.pi_poly.resident_gpu_sppark
proof_server.prover.kzg.multi_open.prepare.resident_pipeline_gpu_sppark
proof_server.prover.plonk.advice.configure
proof_server.prover.plonk.instance.resident_lagrange_to_coeff
proof_server.prover.plonk.instance.resident_upload
proof_server.prover.plonk.lookup.compress_permute.resident_context
proof_server.prover.plonk.lookup.permuted_inputs.stats_download
proof_server.prover.plonk.lookup.permuted_inputs.suffix_download
proof_server.prover.plonk.lookup.products.gpu_sppark_device
proof_server.prover.plonk.lookup.write_product_commitments
proof_server.prover.plonk.permutation.commit.resident_gpu_sppark
proof_server.prover.plonk.permutation.products.resident_context
proof_server.prover.plonk.permutation.write_commitments
proof_server.prover.plonk.resident.coefficients_to_cosets
proof_server.prover.plonk.resident.evaluate_numerator
proof_server.prover.plonk.resident.key_context
proof_server.prover.plonk.resident.quotient_to_coeff
proof_server.prover.plonk.resident.upload_coefficients
proof_server.prover.plonk.vanishing.construct.resident_commit_batch
proof_server.prover.plonk.vanishing.construct.resident_split_blind
proof_server.prover.plonk.vanishing.evaluate.resident_h_lincomb
proof_server.prover.stdlib.verify
proof_server.prover.stdlib.verify.format_instance
proof_server.prover.stdlib.verify.guard
proof_server.prover.stdlib.verify.plonk
proof_server.prover.stdlib.verify.prepare
proof_server.prover.stdlib.verify.transcript_empty
proof_server.prover.verify.final_exponentiation
proof_server.prover.verify.miller_loop
proof_server.prover.verify.msm_left
proof_server.prover.verify.msm_right
proof_server.prover.zkir.init_pk
proof_server.prover.zkir.load_ir
proof_server.prover.zkir.load_pk
proof_server.prover.zkir.load_vk
proof_server.prover.zkir.preprocess
proof_server.prover.zkir.prove.init_pk
proof_server.prover.zkir.prove.params
proof_server.transaction proof_server.transaction.decode
proof_server.transaction.zswap proof_server.transaction.dust
proof_server.transaction.seal
proof_server.response proof_server.response.encode
zkir zkir.v1 zkir.v2 zkir.v3 zkir.keygen zkir.prove
zkir.prove.load_keys zkir.prove.witness zkir.prove.commit
zkir.prove.lookup zkir.prove.sumcheck zkir.prove.open
zswap zswap.input zswap.output zswap.transient zswap.transient.input
zswap.transient.output
ledger ledger.transaction ledger.transaction.intent
ledger.transaction.zswap ledger.transaction.dust ledger.transaction.dust.spend
""".split()

KNOWN_STAGES = set(STAGE_ORDER)

SOURCE_ROWS = [
    ("proof_server.request", "proof-server/src/endpoints.rs", ""),
    ("proof_server.prove", "proof-server/src/endpoints.rs proof-server/src/versioned_ir.rs", ""),
    ("proof_server.prove.dispatch", "proof-server/src/versioned_ir.rs", ""),
    ("proof_server.prove.ir_load", "proof-server/src/versioned_ir.rs", ""),
    ("proof_server.prove.key_material", "proof-server/src/endpoints.rs ledger/src/prove.rs", ""),
    ("proof_server.prove.provider", "proof-server/src/versioned_ir.rs zkir/src/lib.rs", ""),
    ("proof_server.prove.execute", "proof-server/src/versioned_ir.rs zkir/src/lib.rs zkir/src/ir.rs", ""),
    ("proof_server.prove.check", "proof-server/src/versioned_ir.rs", ""),
    (
        "proof_server.prover",
        "vendor/midnight-zk-stdlib-2.3.1/src/lib.rs vendor/midnight-proofs-0.8.1/src/plonk/prover.rs",
        "",
    ),
    ("proof_server.prover.stdlib", "vendor/midnight-zk-stdlib-2.3.1/src/lib.rs", ""),
    (
        "proof_server.prover.plonk",
        "vendor/midnight-proofs-0.8.1/src/plonk/prover.rs vendor/midnight-proofs-0.8.1/src/plonk/vanishing/prover.rs",
        "",
    ),
    (
        "proof_server.prover.kzg",
        "vendor/midnight-proofs-0.8.1/src/poly/kzg/mod.rs vendor/midnight-proofs-0.8.1/src/poly/kzg/msm.rs",
        "",
    ),
    (
        "proof_server.prover.curves",
        "vendor/midnight-curves-0.3.1/src/fft.rs vendor/midnight-curves-0.3.1/src/msm.rs",
        "",
    ),
    ("proof_server.transaction", "proof-server/src/endpoints.rs ledger/src/prove.rs", ""),
    ("proof_server.response", "proof-server/src/endpoints.rs", ""),
    ("zkir", "zkir/src/lib.rs", ""),
    ("zkir.v1", "zkir/src/ir_v1.rs zkir/src/lib.rs", ""),
    ("zkir.v2", "zkir/src/ir.rs zkir/src/lib.rs", ""),
    ("zkir.prove", "zkir/src/lib.rs zkir/src/ir.rs zkir/src/ir_v1.rs", ""),
    ("zswap", "zswap/src/prove.rs", ""),
    ("zswap.input", "zswap/src/prove.rs", ""),
    ("zswap.output", "zswap/src/prove.rs", ""),
    ("zswap.transient", "zswap/src/prove.rs", ""),
    ("ledger.transaction", "ledger/src/prove.rs", ""),
    ("ledger.transaction.dust", "ledger/src/dust.rs", ""),
    ("ledger.transaction.dust.spend", "ledger/src/dust.rs", ""),
]

SOURCE_MAP = {
    stage: {
        "host": host.split(),
        "kernels": kernels.split(),
    }
    for stage, host, kernels in SOURCE_ROWS
}

STAGE_DEPENDENCIES = {
    "proof_server.request.payload": [],
    "proof_server.request.decode": ["proof_server.request.payload"],
    "proof_server.request.resolve_key": ["proof_server.request.decode"],
    "proof_server.prove.dispatch": ["proof_server.request.decode", "proof_server.request.resolve_key"],
    "proof_server.prove.ir_load": ["proof_server.prove.dispatch"],
    "proof_server.prove.key_material": ["proof_server.request.resolve_key"],
    "proof_server.prove.provider": ["proof_server.prove.key_material"],
    "proof_server.prove.execute": ["proof_server.prove.ir_load", "proof_server.prove.provider"],
    "proof_server.prove.check": ["proof_server.prove.execute", "proof_server.prove.ir_load"],
    "proof_server.prover.stdlib.prove": ["proof_server.prove.execute"],
    "proof_server.prover.stdlib.plonk_prove": ["proof_server.prover.stdlib.build_circuit"],
    "proof_server.prover.plonk.create_proof": ["proof_server.prover.stdlib.plonk_prove"],
    "proof_server.prover.plonk.compute_trace": ["proof_server.prover.plonk.create_proof"],
    "proof_server.prover.plonk.finalise_proof": ["proof_server.prover.plonk.compute_trace"],
    "proof_server.prover.plonk.parse_advices": ["proof_server.prover.plonk.compute_instances"],
    "proof_server.prover.plonk.lookup.commit_permuted": ["proof_server.prover.plonk.parse_advices"],
    "proof_server.prover.plonk.permutation.commit": ["proof_server.prover.plonk.lookup.commit_permuted"],
    "proof_server.prover.plonk.lookup.commit_product": ["proof_server.prover.plonk.lookup.commit_permuted"],
    "proof_server.prover.plonk.trash.commit": [
        "proof_server.prover.plonk.permutation.commit",
        "proof_server.prover.plonk.lookup.commit_product",
    ],
    "proof_server.prover.plonk.vanishing.commit_random": ["proof_server.prover.plonk.trash.commit"],
    "proof_server.prover.plonk.advice.lagrange_to_coeff": ["proof_server.prover.plonk.vanishing.commit_random"],
    "proof_server.prover.plonk.compute_nu_poly": ["proof_server.prover.plonk.finalise_proof"],
    "proof_server.prover.plonk.vanishing.construct": ["proof_server.prover.plonk.compute_nu_poly"],
    "proof_server.prover.plonk.write_evals": ["proof_server.prover.plonk.vanishing.construct"],
    "proof_server.prover.plonk.multi_open": ["proof_server.prover.plonk.compute_queries"],
    "proof_server.response.encode": ["proof_server.prove.execute"],
    "proof_server.transaction.decode": ["proof_server.request.payload"],
    "proof_server.transaction.zswap": ["proof_server.transaction.decode"],
    "proof_server.transaction.dust": ["proof_server.transaction.decode"],
    "proof_server.transaction.seal": ["proof_server.transaction.zswap", "proof_server.transaction.dust"],
    "zkir.prove.load_keys": ["proof_server.prove.key_material"],
    "zkir.prove.witness": ["zkir.prove.load_keys"],
    "zkir.prove.commit": ["zkir.prove.witness"],
    "zkir.prove.lookup": ["zkir.prove.commit"],
    "zkir.prove.sumcheck": ["zkir.prove.lookup"],
    "zkir.prove.open": ["zkir.prove.sumcheck"],
    "zswap.input": ["proof_server.transaction.zswap"],
    "zswap.output": ["proof_server.transaction.zswap"],
    "zswap.transient": ["proof_server.transaction.zswap"],
    "zswap.transient.input": ["zswap.transient"],
    "zswap.transient.output": ["zswap.transient"],
    "ledger.transaction.intent": ["proof_server.transaction.decode"],
    "ledger.transaction.zswap": ["proof_server.transaction.decode"],
    "ledger.transaction.dust": ["proof_server.transaction.decode"],
    "ledger.transaction.dust.spend": ["ledger.transaction.dust"],
}

PARALLEL_GROUPS = [
    {
        "group_id": "prove.request_setup",
        "after": ["proof_server.request.decode"],
        "stages": ["proof_server.request.resolve_key"],
        "note": "Binding-input rewrite and resolver construction are request-local work and can overlap key/IR fetch once instrumentation splits them.",
    },
    {
        "group_id": "prove.key_material_decode",
        "after": ["proof_server.prove.key_material"],
        "stages": ["proof_server.prove.ir_load", "zkir.prove.load_keys"],
        "note": "IR, prover-key, and verifier-key decoding are independent after key material is available; current code performs some of this serially and more than once.",
    },
    {
        "group_id": "plonk.advice_commitments",
        "after": ["proof_server.prover.plonk.advice.batch_invert"],
        "stages": ["proof_server.prover.plonk.advice.commit_lagrange_batch"],
        "note": "Advice polynomial commitments are independent until ordered transcript writes; the local prover computes this batch with order-preserving Rayon parallelism.",
    },
    {
        "group_id": "plonk.permutation_lookup_products",
        "after": ["proof_server.prover.plonk.lookup.commit_permuted"],
        "stages": [
            "proof_server.prover.plonk.permutation.commit",
            "proof_server.prover.plonk.lookup.commit_product",
        ],
        "note": "Permutation and lookup products are independent after beta/gamma and run concurrently; commitment bytes are written in protocol order.",
    },
    {
        "group_id": "plonk.vanishing_quotient_commitments",
        "after": ["proof_server.prover.plonk.vanishing.construct"],
        "stages": ["proof_server.prover.kzg.commit.coeff_msm"],
        "note": "Vanishing quotient limb commitments are independent after quotient blinding and use the resident device commitment batch before ordered transcript writes.",
    },
    {
        "group_id": "prove_tx.top_level",
        "after": ["proof_server.transaction.decode"],
        "stages": [
            "ledger.transaction.intent",
            "ledger.transaction.zswap",
            "ledger.transaction.dust",
        ],
        "note": "Standard transaction proving already joins intents, guaranteed coins, and fallible coins; this is the coarse-grained GPU batching boundary.",
    },
    {
        "group_id": "zswap.offer_parts",
        "after": ["proof_server.transaction.zswap"],
        "stages": ["zswap.input", "zswap.output", "zswap.transient"],
        "note": "Zswap offer inputs, outputs, and transient proofs are independent and already use futures::join/join_all.",
    },
    {
        "group_id": "zswap.transient_pair",
        "after": ["zswap.transient"],
        "stages": ["zswap.transient.input", "zswap.transient.output"],
        "note": "Each transient input proof and output proof are independent and already run through split provers.",
    },
    {
        "group_id": "dust.spends",
        "after": ["ledger.transaction.dust"],
        "stages": ["ledger.transaction.dust.spend"],
        "note": "Dust spends are independent across spends and already use join_all.",
    },
]


def ancestors(stage_id):
    parts = stage_id.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def ordered_stage_ids(stage_ids):
    seen = set(stage_ids)
    ordered = [stage_id for stage_id in STAGE_ORDER if stage_id in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def source_for(stage_id):
    parts = ancestors(stage_id)
    for candidate in reversed(parts):
        if candidate in SOURCE_MAP:
            return SOURCE_MAP[candidate]
    return {"host": [], "kernels": []}


def dependencies_for(stage_id):
    return STAGE_DEPENDENCIES.get(stage_id, [])


def stage_dependency_report():
    return {
        "stage_dependencies": STAGE_DEPENDENCIES,
        "parallel_groups": PARALLEL_GROUPS,
    }
