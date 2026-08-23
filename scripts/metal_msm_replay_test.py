#!/usr/bin/env python3

import collections
import copy
import json
import stat
import tempfile
import unittest
from pathlib import Path

try:
    import metal_msm_replay
    import metal_msm_runs
except ModuleNotFoundError:  # Support invocation as a module from the repo root.
    from scripts import metal_msm_replay, metal_msm_runs


AUTHORITATIVE_CAPTURE = Path("/tmp/metal-msm-runs-seed24302.jsonl")


def valid_call(call_id: int, run_len: int, logical_len: int = 32_768) -> dict:
    windows = [
        {
            "window": window,
            "records": 0,
            "nonempty_runs": 0,
            "max_run": 0,
            "run_histogram": [],
        }
        for window in range(26)
    ]
    if run_len:
        windows[0] = {
            "window": 0,
            "records": run_len,
            "nonempty_runs": 1,
            "max_run": run_len,
            "run_histogram": [[run_len, 1]],
        }
    return {
        "schema": metal_msm_runs.INPUT_SCHEMA,
        "call": call_id,
        "logical_len": logical_len,
        "nonzero_scalars": call_id + 1,
        "window_bits": 10,
        "window_count": 26,
        "windows": windows,
        "total_records": run_len,
        "total_nonempty_runs": int(run_len != 0),
        "total_pair_reductions": max(0, run_len - 1),
    }


def synthetic_calls() -> list[dict]:
    raw_calls = [
        valid_call(call_id, call_id + 1, 32_767 if call_id < 2 else 32_768)
        for call_id in range(62)
    ]
    return [
        metal_msm_runs.validate_call(call, index + 1)
        for index, call in enumerate(raw_calls)
    ]


class MetalMsmReplayTests(unittest.TestCase):
    def test_call_hash_excludes_call_identifier_and_schema(self):
        left = metal_msm_runs.validate_call(valid_call(1, 7), 1)
        right_raw = valid_call(99, 7)
        right_raw["nonzero_scalars"] = left["nonzero_scalars"]
        right = metal_msm_runs.validate_call(right_raw, 2)
        self.assertEqual(
            metal_msm_replay.call_sha256(left),
            metal_msm_replay.call_sha256(right),
        )

    def test_manifest_is_deterministic_under_source_reordering(self):
        calls = synthetic_calls()
        forward = metal_msm_replay.build_manifest(calls, "11" * 32)
        reverse = metal_msm_replay.build_manifest(list(reversed(calls)), "11" * 32)
        self.assertEqual(forward, reverse)

    def test_lpt_bins_cover_every_call_with_bounded_capacity(self):
        profile_calls = metal_msm_replay.select_32k_calls(synthetic_calls())
        bins = metal_msm_replay.lpt_balance(profile_calls)
        self.assertEqual(len(bins), metal_msm_replay.LPT_BIN_COUNT)
        self.assertTrue(
            all(
                0 < len(bin_calls) <= metal_msm_replay.MAX_CALLS_PER_BIN
                for bin_calls in bins
            )
        )
        expected = collections.Counter(call.sha256 for call in profile_calls)
        observed = collections.Counter(
            call.sha256 for bin_calls in bins for call in bin_calls
        )
        self.assertEqual(observed, expected)

    def test_sensitive_diagnostic_schema_is_exact(self):
        manifest = metal_msm_replay.build_manifest(synthetic_calls(), "22" * 32)
        self.assertEqual(
            set(manifest),
            {
                "schema",
                "source_anchor",
                "selection",
                "chunks",
                "selected_totals",
                "manifest_sha256",
            },
        )
        self.assertEqual(
            set(manifest["source_anchor"]),
            {
                "source_workload_sha256",
                "call_count",
                "logical_shapes",
                "logical_positions",
                "total_records",
                "total_nonempty_runs",
                "total_pair_reductions",
                "max_run",
                "run_histogram_sha256",
            },
        )
        self.assertEqual(
            set(manifest["selection"]),
            {
                "logical_lengths",
                "lpt_bin_count",
                "max_calls_per_bin",
                "selected_load_ranks",
            },
        )
        expected_chunk_keys = {
            "label",
            "load_rank",
            "call_count",
            "call_hashes",
            "total_records",
            "total_nonempty_runs",
            "total_pair_reductions",
            "max_run",
            "run_histogram",
            "chunk_sha256",
        }
        for chunk in manifest["chunks"]:
            self.assertEqual(set(chunk), expected_chunk_keys)
            self.assertEqual(chunk["call_count"], len(chunk["call_hashes"]))
            self.assertIn("run_histogram", chunk)
            self.assertTrue(
                all(
                    len(call_hash) == 64 and set(call_hash) <= set("0123456789abcdef")
                    for call_hash in chunk["call_hashes"]
                )
            )

        encoded = json.dumps(manifest, sort_keys=True)
        for forbidden in (
            '"call"',
            '"schema":"midnight-msm-runs-v1"',
            '"windows"',
            '"window"',
            '"bucket"',
            '"sign"',
            '"point_index"',
            '"srs_index"',
            '"nonzero_scalars"',
        ):
            self.assertNotIn(forbidden, encoded)

        self.assertIn("sensitive", metal_msm_replay.__doc__)

    def test_manifest_and_chunk_checksums_are_recomputed(self):
        manifest = metal_msm_replay.build_manifest(synthetic_calls(), "23" * 32)
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        self.assertEqual(manifest["manifest_sha256"], metal_msm_replay._sha256(body))

        for chunk in manifest["chunks"]:
            replay_payload = {
                "call_hashes": chunk["call_hashes"],
                "total_records": chunk["total_records"],
                "total_nonempty_runs": chunk["total_nonempty_runs"],
                "total_pair_reductions": chunk["total_pair_reductions"],
                "max_run": chunk["max_run"],
                "run_histogram": chunk["run_histogram"],
            }
            self.assertEqual(
                chunk["chunk_sha256"], metal_msm_replay._sha256(replay_payload)
            )

        tampered = copy.deepcopy(manifest)
        tampered["chunks"][0]["run_histogram"][0][1] += 1
        tampered_body = {
            key: value for key, value in tampered.items() if key != "manifest_sha256"
        }
        self.assertNotEqual(
            tampered["manifest_sha256"], metal_msm_replay._sha256(tampered_body)
        )

    def test_histogram_arithmetic_and_aggregate_totals_reconcile(self):
        manifest = metal_msm_replay.build_manifest(synthetic_calls(), "33" * 32)
        for chunk in manifest["chunks"]:
            histogram = chunk["run_histogram"]
            self.assertEqual(
                chunk["total_records"],
                sum(run_len * count for run_len, count in histogram),
            )
            self.assertEqual(
                chunk["total_nonempty_runs"],
                sum(count for _, count in histogram),
            )
            self.assertEqual(
                chunk["total_pair_reductions"],
                sum((run_len - 1) * count for run_len, count in histogram),
            )
            self.assertEqual(chunk["max_run"], histogram[-1][0])

        selected = manifest["selected_totals"]
        self.assertEqual(
            selected["total_records"],
            sum(chunk["total_records"] for chunk in manifest["chunks"]),
        )
        self.assertEqual(
            selected["total_nonempty_runs"],
            sum(chunk["total_nonempty_runs"] for chunk in manifest["chunks"]),
        )
        self.assertEqual(
            selected["total_pair_reductions"],
            sum(chunk["total_pair_reductions"] for chunk in manifest["chunks"]),
        )
        self.assertEqual(manifest["source_anchor"]["call_count"], 62)
        self.assertEqual(
            manifest["source_anchor"]["logical_shapes"],
            [[32_767, 2], [32_768, 60]],
        )

    def test_more_than_bounded_capacity_is_rejected(self):
        calls = synthetic_calls()
        extra = []
        for offset in range(3):
            raw = valid_call(100 + offset, 100 + offset)
            extra.append(metal_msm_runs.validate_call(raw, 100 + offset))
        with self.assertRaisesRegex(
            metal_msm_replay.ReplayManifestError, "bounded capacity"
        ):
            metal_msm_replay.build_manifest(calls + extra, "44" * 32)

    def test_output_file_is_owner_only(self):
        if not AUTHORITATIVE_CAPTURE.exists():
            self.skipTest("authoritative capture is not present")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "manifest.json"
            output.write_text("preexisting wider-mode file", encoding="utf-8")
            output.chmod(0o644)
            result = metal_msm_replay.main(
                [
                    str(AUTHORITATIVE_CAPTURE),
                    "--expect-seed-24302",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_cli_defaults_to_the_private_tmp_workflow(self):
        arguments = metal_msm_replay.parse_args([str(AUTHORITATIVE_CAPTURE)])
        self.assertEqual(arguments.output, metal_msm_replay.DEFAULT_OUTPUT)

    @unittest.skipUnless(
        AUTHORITATIVE_CAPTURE.exists(), "authoritative capture is not present"
    )
    def test_authoritative_seed_24302_anchors(self):
        manifest = metal_msm_replay.build_from_path(
            AUTHORITATIVE_CAPTURE, expect_seed_24302=True
        )
        self.assertEqual(
            manifest["source_anchor"], metal_msm_replay.SEED_24302_SOURCE_ANCHOR
        )
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        self.assertEqual(manifest["manifest_sha256"], metal_msm_replay._sha256(body))
        self.assertEqual(
            [chunk["load_rank"] for chunk in manifest["chunks"]],
            list(metal_msm_replay.SELECTED_LOAD_RANKS),
        )


if __name__ == "__main__":
    unittest.main()
