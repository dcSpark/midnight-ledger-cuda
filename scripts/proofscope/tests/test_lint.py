import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lint

N = 1_000_000  # 1ms in ns


def kernel(name="evaluate_graph", start=0, ms=2.0, grid=(2048, 1, 1), block=(256, 1, 1),
           regs=64, local_mem=0, smem=0):
    return {
        "name": name,
        "start": start,
        "end": start + int(ms * N),
        "grid": list(grid),
        "block": list(block),
        "registers_per_thread": regs,
        "local_memory_per_thread": local_mem,
        "static_shared_memory": smem,
        "dynamic_shared_memory": 0,
    }


class LintTests(unittest.TestCase):
    def test_register_spill_is_high_severity(self):
        findings = lint.lint_kernels([kernel(local_mem=8192)])
        spills = [f for f in findings if f["rule"] == "register-spill"]
        self.assertEqual(len(spills), 1)
        self.assertEqual(spills[0]["severity"], "high")
        self.assertIn("8192 B/thread", spills[0]["detail"])

    def test_clean_kernel_has_no_findings(self):
        findings = lint.lint_kernels([kernel()])
        self.assertEqual(findings, [])

    def test_block_not_warp_multiple(self):
        findings = lint.lint_kernels([kernel(block=(100, 1, 1), ms=0.1)])
        rules = {f["rule"] for f in findings}
        self.assertIn("block-not-warp-multiple", rules)

    def test_grid_underfill_flagged_for_long_kernels_only(self):
        small_grid = kernel(grid=(4, 1, 1), ms=2.0)
        self.assertIn("grid-underfill", {f["rule"] for f in lint.lint_kernels([small_grid])})
        short = kernel(grid=(4, 1, 1), ms=0.1)
        self.assertNotIn("grid-underfill", {f["rule"] for f in lint.lint_kernels([short])})

    def test_low_occupancy_reports_limiter(self):
        heavy_regs = kernel(regs=255, ms=2.0)
        findings = [f for f in lint.lint_kernels([heavy_regs]) if f["rule"] == "low-occupancy"]
        self.assertEqual(len(findings), 1)
        self.assertIn("registers", findings[0]["detail"])

    def test_kernel_stats_aggregates_by_name(self):
        rows = lint.kernel_stats([kernel(ms=1.0), kernel(ms=3.0, start=5 * N), kernel(name="ntt", ms=2.0)])
        self.assertEqual(rows[0]["kernel"], "evaluate_graph")
        self.assertEqual(rows[0]["launches"], 2)
        self.assertAlmostEqual(rows[0]["total_ms"], 4.0)
        self.assertAlmostEqual(rows[0]["mean_us"], 2000.0)


if __name__ == "__main__":
    unittest.main()
