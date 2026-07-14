import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare


class LoadStageMapTests(unittest.TestCase):
    def test_rejects_parity_report(self):
        with TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "parity.json"
            report.write_text('{"byte_exact": true, "reports": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a proofscope stage analysis"):
                compare.load_stage_map(report)


def stage(stage_id="proof_server.prover.plonk.multi_open", wall=100.0, launches=10,
          h2d_mb=0.0, kernels=None, kernel_ms=None):
    return {
        "stage_id": stage_id,
        "window_wall_ms": wall,
        "gpu_busy_ms": wall * 0.4,
        "transfer_wait_ms": 0.0,
        "sync_wait_ms": 0.0,
        "host_gap_ms": wall * 0.6,
        "unattributed_ms": 0.0,
        "launches": launches,
        "h2d_copies": 0,
        "d2h_copies": 0,
        "dtod_copies": 0,
        "memset_count": 0,
        "syncs": 0,
        "h2d_mb": h2d_mb,
        "d2h_mb": 0.0,
        "dtod_mb": 0.0,
        "memset_mb": 0.0,
        "kernel_attributed_ms": kernel_ms if kernel_ms is not None else wall * 0.4,
        "kernels": kernels or {},
    }


class StructuralCausesTests(unittest.TestCase):
    def test_reports_launch_and_transfer_growth(self):
        a = stage(launches=10, h2d_mb=5.0)
        b = stage(launches=40, h2d_mb=120.0)
        causes = compare.structural_causes(a, b)
        self.assertTrue(any("launches 10 -> 40" in c for c in causes))
        self.assertTrue(any("h2d_mb 5.0 -> 120.0" in c for c in causes))

    def test_reports_new_kernel(self):
        a = stage(kernels={})
        b = stage(kernels={"poly_lincomb_kernel": 42.0})
        causes = compare.structural_causes(a, b)
        self.assertTrue(any("poly_lincomb_kernel" in c and "(new)" in c for c in causes))

    def test_quiet_when_nothing_changed(self):
        a = stage()
        self.assertEqual(compare.structural_causes(a, dict(a)), [])


class CheckTests(unittest.TestCase):
    def test_regression_beyond_tolerance_fails_with_causes(self):
        base = {"s": stage(stage_id="s", wall=100.0, launches=10)}
        cand = {"s": stage(stage_id="s", wall=130.0, launches=60)}
        checked, failures = compare.check_stages(base, cand, tolerance=0.05, abs_floor_ms=2.0)
        self.assertEqual(checked, 1)
        self.assertEqual(len(failures), 1)
        self.assertTrue(any("launches" in c for c in failures[0]["causes"]))

    def test_within_tolerance_passes(self):
        base = {"s": stage(stage_id="s", wall=100.0)}
        cand = {"s": stage(stage_id="s", wall=104.0)}
        checked, failures = compare.check_stages(base, cand, tolerance=0.05, abs_floor_ms=2.0)
        self.assertEqual(checked, 1)
        self.assertEqual(failures, [])

    def test_small_stages_are_ignored(self):
        base = {"s": stage(stage_id="s", wall=2.0)}
        cand = {"s": stage(stage_id="s", wall=50.0)}
        checked, failures = compare.check_stages(base, cand, min_stage_ms=5.0)
        self.assertEqual(checked, 0)
        self.assertEqual(failures, [])


class DiffTests(unittest.TestCase):
    def test_rows_sorted_by_absolute_delta(self):
        base = {
            "a": stage(stage_id="a", wall=100.0),
            "b": stage(stage_id="b", wall=50.0),
        }
        cand = {
            "a": stage(stage_id="a", wall=105.0),
            "b": stage(stage_id="b", wall=200.0),
        }
        rows = compare.diff_rows(base, cand)
        self.assertEqual([r["stage_id"] for r in rows], ["b", "a"])
        self.assertAlmostEqual(rows[0]["delta_ms"], 150.0)


if __name__ == "__main__":
    unittest.main()
