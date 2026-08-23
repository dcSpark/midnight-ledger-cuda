#!/usr/bin/env python3

import copy
import unittest

import metal_msm_runs


def valid_call() -> dict:
    windows = [
        {
            "window": window,
            "records": 0,
            "nonempty_runs": 0,
            "max_run": 0,
            "run_histogram": [],
        }
        for window in range(32)
    ]
    windows[0] = {
        "window": 0,
        "records": 7,
        "nonempty_runs": 3,
        "max_run": 3,
        "run_histogram": [[1, 1], [3, 2]],
    }
    return {
        "schema": metal_msm_runs.INPUT_SCHEMA,
        "call": 7,
        "logical_len": 8_192,
        "nonzero_scalars": 2,
        "window_bits": 8,
        "window_count": 32,
        "windows": windows,
        "total_records": 7,
        "total_nonempty_runs": 3,
        "total_pair_reductions": 4,
    }


class MetalMsmRunsTests(unittest.TestCase):
    def test_valid_call_reconciles_and_summarizes(self):
        call = metal_msm_runs.validate_call(valid_call(), 1)
        summary = metal_msm_runs.summarize([call], "00" * 32)
        self.assertEqual(summary["total_records"], 7)
        self.assertEqual(summary["total_nonempty_runs"], 3)
        self.assertEqual(summary["total_pair_reductions"], 4)
        self.assertEqual(summary["classes"][0]["balanced_depth"], 2)

    def test_unknown_field_is_rejected(self):
        value = valid_call()
        value["scalar_hash"] = "forbidden"
        with self.assertRaisesRegex(metal_msm_runs.DiagnosticError, "unknown"):
            metal_msm_runs.validate_call(value, 1)

    def test_histogram_mismatch_is_rejected(self):
        value = copy.deepcopy(valid_call())
        value["windows"][0]["records"] = 8
        with self.assertRaisesRegex(metal_msm_runs.DiagnosticError, "histogram records"):
            metal_msm_runs.validate_call(value, 1)

    def test_seed_anchor_checks_shape_and_nonzero_totals(self):
        calls = []
        call_id = 0
        for logical_len, count in metal_msm_runs.SEED_24302_SHAPES.items():
            for _ in range(count):
                calls.append(
                    {
                        "call": call_id,
                        "logical_len": logical_len,
                        "nonzero_scalars": 0,
                    }
                )
                call_id += 1
        remaining = 1_988_928
        for call in calls:
            assigned = min(call["logical_len"], remaining)
            call["nonzero_scalars"] = assigned
            remaining -= assigned
        self.assertEqual(remaining, 0)
        metal_msm_runs.check_seed_24302(calls)


if __name__ == "__main__":
    unittest.main()
