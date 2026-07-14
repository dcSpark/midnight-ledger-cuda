import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
import levers
import parse
import taxonomy


N = 1_000_000


class AnalyzeTests(unittest.TestCase):
    def test_decompose_window_reconciles(self):
        intervals = {
            "kernel": analyze.interval_union([(0, 10 * N)]),
            "copy": analyze.interval_union([(10 * N, 15 * N)]),
            "memset": [],
            "device_busy": analyze.interval_union([(0, 10 * N)]),
            "device": analyze.interval_union([(0, 10 * N), (10 * N, 15 * N)]),
            "sync_api": analyze.interval_union([(15 * N, 18 * N)]),
            "other_api": analyze.interval_union([(18 * N, 20 * N)]),
        }

        row = analyze.decompose_window(0, 25 * N, intervals)
        bucket_sum = (
            row["gpu_busy_ms"]
            + row["transfer_wait_ms"]
            + row["sync_wait_ms"]
            + row["api_ms"]
            + row["host_gap_ms"]
            + row["unattributed_ms"]
        )

        self.assertAlmostEqual(row["wall_ms"], 25.0)
        self.assertAlmostEqual(row["gpu_busy_ms"], 10.0)
        self.assertAlmostEqual(row["transfer_wait_ms"], 5.0)
        self.assertAlmostEqual(row["sync_wait_ms"], 3.0)
        self.assertAlmostEqual(row["api_ms"], 2.0)
        self.assertAlmostEqual(row["host_gap_ms"], 5.0)
        self.assertAlmostEqual(row["wall_ms"], bucket_sum)

    def test_exclusive_segments_choose_innermost_range(self):
        phases = [
            {"label": "proof_server.prove", "start": 0, "end": 10 * N, "duration": 10 * N},
            {
                "label": "proof_server.prove.execute",
                "start": 2 * N,
                "end": 5 * N,
                "duration": 3 * N,
            },
        ]

        segments = analyze.exclusive_segments(phases)

        self.assertEqual(
            [(s["label"], s["start"] // N, s["end"] // N) for s in segments],
            [
                ("proof_server.prove", 0, 2),
                ("proof_server.prove.execute", 2, 5),
                ("proof_server.prove", 5, 10),
            ],
        )

    def test_stderr_phase_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "stderr.txt"
            stderr.write_text(
                "[proof-server] prove            2.00ms @3000000\n",
                encoding="utf-8",
            )

            phases, counts = parse.build_phases(stderr, 0, [])

        self.assertEqual(phases[0]["label"], "proof_server.prove")
        self.assertEqual(phases[0]["start"], 1 * N)
        self.assertEqual(phases[0]["end"], 3 * N)
        self.assertEqual(phases[0]["chain"], "gpu")
        self.assertEqual(counts["stderr_stamped_lines"], 1)

    def test_midnight_prover_timer_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "stderr.txt"
            stderr.write_text(
                "[midnight-prover] proof_server.prover.kzg.commit.lagrange_msm count=4096 6.25ms @10000000\n",
                encoding="utf-8",
            )

            phases, counts = parse.build_phases(stderr, 0, [])

        self.assertEqual(phases[0]["label"], "proof_server.prover.kzg.commit.lagrange_msm")
        self.assertEqual(phases[0]["start"], 3_750_000)
        self.assertEqual(phases[0]["end"], 10_000_000)
        self.assertEqual(counts["stderr_stamped_lines"], 1)

    def test_nvtx_aliases_are_normalized(self):
        nvtx = [{"text": "proof-server:prove", "start": 1 * N, "end": 4 * N}]

        phases, _ = parse.build_phases(None, None, nvtx)

        self.assertEqual(phases[0]["label"], "proof_server.prove")
        self.assertEqual(phases[0]["source"], "nvtx")

    def test_analyze_bundle_writes_stage_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            nsys = bundle / "nsys"
            nsys.mkdir()
            sqlite_path = nsys / "report.sqlite"
            db = sqlite3.connect(sqlite_path)
            db.execute(
                "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, "
                "text TEXT, rangeId INTEGER, globalTid INTEGER)"
            )
            db.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (0, 25 * N, 59, "proof_server.prove", 1, 1),
                    (5 * N, 15 * N, 59, "proof_server.prove.execute", 2, 1),
                ],
            )
            db.commit()
            db.close()

            output_path, report = analyze.analyze_bundle(bundle)
            self.assertTrue((output_path.parent / "dependencies.json").exists())

        stage_ids = {row["stage_id"] for row in report["stages"]}
        self.assertEqual(output_path.name, "stages.json")
        self.assertIn("proof_server", stage_ids)
        self.assertIn("proof_server.prove", stage_ids)
        self.assertIn("proof_server.prove.execute", stage_ids)
        self.assertEqual(report["taxonomy_version"], taxonomy.TAXONOMY_VERSION)
        by_stage = {row["stage_id"]: row for row in report["stages"]}
        self.assertIn("proof_server.prove.ir_load", by_stage["proof_server.prove.execute"]["depends_on"])

    def test_levers_rank_by_window_recoverable_time(self):
        stages = [
            {
                "stage_id": "proof_server.prove.fast",
                "window_wall_ms": 10.0,
                "wall_ms": 4.0,
                "gpu_busy_ms": 8.0,
                "launches": 1,
                "host_gap_ms": 1.0,
                "sync_wait_ms": 0.0,
                "api_ms": 0.0,
                "transfer_wait_ms": 0.0,
                "unattributed_ms": 0.0,
                "top_kernels": [],
            },
            {
                "stage_id": "proof_server.prove.slow",
                "window_wall_ms": 50.0,
                "wall_ms": 20.0,
                "gpu_busy_ms": 10.0,
                "launches": 2,
                "host_gap_ms": 30.0,
                "sync_wait_ms": 2.0,
                "api_ms": 1.0,
                "transfer_wait_ms": 0.0,
                "unattributed_ms": 0.0,
                "top_kernels": ["slow_kernel"],
            },
        ]

        report = levers.build_levers(stages, {"memcpys": []})

        self.assertEqual(report["levers"][0]["stage_id"], "proof_server.prove.slow")
        self.assertEqual(report["levers"][0]["fix_class"], "cpu_pipeline")

    def test_taxonomy_names_parallel_boundaries(self):
        groups = {row["group_id"]: row for row in taxonomy.PARALLEL_GROUPS}

        self.assertIn("prove.key_material_decode", groups)
        self.assertIn("zswap.offer_parts", groups)
        self.assertIn("proof_server.prove.ir_load", groups["prove.key_material_decode"]["stages"])
        self.assertIn("zswap.output", groups["zswap.offer_parts"]["stages"])
        self.assertIn("zswap.transient.input", groups["zswap.transient_pair"]["stages"])


if __name__ == "__main__":
    unittest.main()
