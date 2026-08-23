import os
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpuprof


def args(
    backend,
    fixture="zswap-transaction",
    cargo_features=None,
    metal_quotient_diagnostic=False,
):
    return SimpleNamespace(
        backend=backend,
        fixture=fixture,
        cargo_features=cargo_features,
        hash_rounds=64,
        repeat=5,
        warmup=1,
        seed="24301",
        json=None,
        metal_quotient_diagnostic=metal_quotient_diagnostic,
    )


class BackendSelectionTests(unittest.TestCase):
    def test_gate_parser_accepts_clear_cpu_vs_metal_hybrid_mode(self):
        parser = ArgumentParser()
        gpuprof.add_gate_args(parser, default_backend="cpu")
        parsed = parser.parse_args(
            ["proof-backend-parity", "--backend", "cpu-vs-metal-hybrid"]
        )
        self.assertEqual(parsed.backend, "cpu-vs-metal-hybrid")

    def test_clear_paired_aliases_map_to_existing_binary_spellings(self):
        self.assertEqual(gpuprof.normalized_backend("cpu-vs-cuda"), "both")
        self.assertEqual(gpuprof.normalized_backend("cpu-vs-metal"), "cpu-metal")
        self.assertEqual(
            gpuprof.normalized_backend("cpu-vs-metal-hybrid"),
            "cpu-metal-hybrid",
        )
        self.assertEqual(gpuprof.normalized_backend("metal-hybrid"), "metal-hybrid")

    def test_metal_hybrid_aliases_enable_only_the_metal_cargo_feature(self):
        for backend in ("metal-hybrid", "cpu-metal-hybrid", "cpu-vs-metal-hybrid"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    gpuprof.resolve_cargo_features(args(backend), collection=False),
                    "deterministic-test-rng,metal",
                )
                self.assertFalse(gpuprof.backend_uses_cuda(backend))
                self.assertTrue(gpuprof.backend_uses_metal(backend))

    def test_cpu_vs_cuda_alias_enables_only_the_cuda_cargo_feature(self):
        self.assertEqual(
            gpuprof.resolve_cargo_features(args("cpu-vs-cuda"), collection=False),
            "deterministic-test-rng,gpu-blitzar",
        )
        self.assertTrue(gpuprof.backend_uses_cuda("cpu-vs-cuda"))
        self.assertFalse(gpuprof.backend_uses_metal("cpu-vs-cuda"))

    def test_parity_arguments_pass_the_binary_spelling(self):
        command = gpuprof.parity_args(args("cpu-vs-metal-hybrid"))
        self.assertEqual(command[:2], ["--backend", "cpu-metal-hybrid"])

    def test_sppark_lock_is_added_only_for_cuda_routes(self):
        with patch.dict(os.environ, {}, clear=True):
            cuda_env = gpuprof.parity_environment(args("cpu-vs-cuda"))
            metal_env = gpuprof.parity_environment(args("cpu-vs-metal-hybrid"))
        self.assertEqual(cuda_env["MIDNIGHT_GPU_SPPARK_LOCK"], "on")
        self.assertNotIn("MIDNIGHT_GPU_SPPARK_LOCK", metal_env)

    def test_metal_route_drops_an_inherited_sppark_lock(self):
        with patch.dict(
            os.environ,
            {"MIDNIGHT_GPU_SPPARK_LOCK": "on"},
            clear=True,
        ):
            metal_env = gpuprof.parity_environment(args("metal-hybrid"))
        self.assertNotIn("MIDNIGHT_GPU_SPPARK_LOCK", metal_env)

    def test_quotient_diagnostic_forces_feature_and_runtime_acknowledgement(self):
        diagnostic = args(
            "cpu-vs-metal-hybrid",
            cargo_features="deterministic-test-rng,perf-timers",
            metal_quotient_diagnostic=True,
        )
        diagnostic.warmup = 2
        with patch.dict(os.environ, {}, clear=True):
            environment = gpuprof.parity_environment(diagnostic)
        self.assertEqual(
            gpuprof.resolve_cargo_features(diagnostic, collection=False),
            "deterministic-test-rng,perf-timers,metal,metal-quotient-diagnostic",
        )
        self.assertEqual(environment["MIDNIGHT_METAL_QUOTIENT_DIAGNOSTIC"], "1")
        self.assertIsNone(gpuprof.validate_metal_quotient_args(diagnostic))

    def test_quotient_diagnostic_rejects_nonrepresentative_selection(self):
        diagnostic = args("metal", metal_quotient_diagnostic=True)
        diagnostic.warmup = 2
        self.assertIn(
            "requires metal-hybrid",
            gpuprof.validate_metal_quotient_args(diagnostic),
        )


if __name__ == "__main__":
    unittest.main()
