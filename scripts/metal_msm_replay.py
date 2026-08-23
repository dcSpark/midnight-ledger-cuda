#!/usr/bin/env python3
"""Build a deterministic 32K Metal MSM replay manifest for local diagnostics.

Both the aggregate JSONL input and the generated manifest contain sensitive,
witness-derived diagnostic data.  Omitting direct call identifiers, bucket
identifiers, record order, scalar signs, and SRS indices does not declassify the
remaining call fingerprints or run-occupancy histograms.  The CLI therefore
writes an owner-only temporary artifact instead of emitting it to stdout.  Do
not publish or commit either file.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import metal_msm_runs
except ModuleNotFoundError:  # Support ``python -m unittest scripts/...``.
    from scripts import metal_msm_runs


OUTPUT_SCHEMA = "midnight-metal-msm-replay-manifest-v1"
DEFAULT_OUTPUT = Path("/tmp/metal-msm-replay-seed24302.json")
LOGICAL_LENGTHS = (32_767, 32_768)
LPT_BIN_COUNT = 16
MAX_CALLS_PER_BIN = 4
SELECTED_LOAD_RANKS = (0, 5, 10, 15)
SELECTED_LABELS = ("min", "low_mid", "high_mid", "max")

# This population anchor is already recorded in the seed-24302 diagnostic
# report.  Its presence here does not make the generated selection fingerprints
# or occupancy histograms safe to publish.
SEED_24302_SOURCE_ANCHOR = {
    "source_workload_sha256": "47a6510def79f770e1efe57c8efd0630d916e1f60037865a8f9a0a295d4e27f1",
    "call_count": 62,
    "logical_shapes": [[32_767, 2], [32_768, 60]],
    "logical_positions": 2_031_614,
    "total_records": 27_886_288,
    "total_nonempty_runs": 634_284,
    "total_pair_reductions": 27_252_004,
    "max_run": 16_797,
    "run_histogram_sha256": "ab3843f6d6ec98b223b2adaf2f3803b99439aa8312f2121df781726980af1ee0",
}


class ReplayManifestError(ValueError):
    """The source cannot produce the requested bounded replay manifest."""


@dataclass(frozen=True)
class ProfileCall:
    """A validated call coupled to its sensitive canonical aggregate hash."""

    sha256: str
    value: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def call_sha256(call: dict[str, Any]) -> str:
    """Hash aggregate call data; this fingerprint remains witness-derived."""

    hash_keys = metal_msm_runs.CALL_KEYS - {"schema", "call"}
    missing = hash_keys - set(call)
    if missing:
        raise ReplayManifestError(
            f"validated call is missing hash fields: {sorted(missing)}"
        )
    payload = {key: call[key] for key in hash_keys}
    return _sha256(payload)


def select_32k_calls(calls: Iterable[dict[str, Any]]) -> list[ProfileCall]:
    selected = [
        ProfileCall(call_sha256(call), call)
        for call in calls
        if call["logical_len"] in LOGICAL_LENGTHS
    ]
    if len(selected) < LPT_BIN_COUNT:
        raise ReplayManifestError(
            f"32K profile has {len(selected)} calls; at least {LPT_BIN_COUNT} required"
        )
    capacity = LPT_BIN_COUNT * MAX_CALLS_PER_BIN
    if len(selected) > capacity:
        raise ReplayManifestError(
            f"32K profile has {len(selected)} calls; bounded capacity is {capacity}"
        )
    return selected


def _run_histogram(calls: Iterable[ProfileCall]) -> list[list[int]]:
    histogram: collections.Counter[int] = collections.Counter()
    for profile_call in calls:
        for window in profile_call.value["windows"]:
            for run_len, count in window["run_histogram"]:
                histogram[run_len] += count
    return [[run_len, histogram[run_len]] for run_len in sorted(histogram)]


def _aggregate(calls: list[ProfileCall]) -> dict[str, Any]:
    histogram = _run_histogram(calls)
    records = sum(run_len * count for run_len, count in histogram)
    runs = sum(count for _, count in histogram)
    pairs = sum((run_len - 1) * count for run_len, count in histogram)

    reported_records = sum(call.value["total_records"] for call in calls)
    reported_runs = sum(call.value["total_nonempty_runs"] for call in calls)
    reported_pairs = sum(call.value["total_pair_reductions"] for call in calls)
    if (records, runs, pairs) != (
        reported_records,
        reported_runs,
        reported_pairs,
    ):
        raise ReplayManifestError(
            "merged histogram disagrees with validated call totals: "
            f"histogram={(records, runs, pairs)}, "
            f"reported={(reported_records, reported_runs, reported_pairs)}"
        )

    return {
        "total_records": records,
        "total_nonempty_runs": runs,
        "total_pair_reductions": pairs,
        "max_run": histogram[-1][0] if histogram else 0,
        "run_histogram": histogram,
    }


def lpt_balance(calls: list[ProfileCall]) -> list[list[ProfileCall]]:
    """Assign descending pair loads to the least-loaded bounded bin."""

    capacity = LPT_BIN_COUNT * MAX_CALLS_PER_BIN
    if len(calls) > capacity:
        raise ReplayManifestError(
            f"cannot place {len(calls)} calls in bounded capacity {capacity}"
        )
    ordered = sorted(
        calls,
        key=lambda call: (-call.value["total_pair_reductions"], call.sha256),
    )
    bins: list[list[ProfileCall]] = [[] for _ in range(LPT_BIN_COUNT)]
    loads = [0] * LPT_BIN_COUNT
    for call in ordered:
        candidates = [
            index
            for index, bin_calls in enumerate(bins)
            if len(bin_calls) < MAX_CALLS_PER_BIN
        ]
        if not candidates:
            raise ReplayManifestError("LPT placement exhausted bounded bins")
        destination = min(candidates, key=lambda index: (loads[index], index))
        bins[destination].append(call)
        loads[destination] += call.value["total_pair_reductions"]
    return bins


def _rank_bins(
    bins: list[list[ProfileCall]],
) -> list[list[ProfileCall]]:
    def rank_key(bin_calls: list[ProfileCall]) -> tuple[int, str]:
        load = sum(call.value["total_pair_reductions"] for call in bin_calls)
        fingerprint = _sha256(sorted(call.sha256 for call in bin_calls))
        return load, fingerprint

    return sorted(bins, key=rank_key)


def _chunk(label: str, load_rank: int, calls: list[ProfileCall]) -> dict[str, Any]:
    aggregate = _aggregate(calls)
    replay_payload = {
        "call_hashes": sorted(call.sha256 for call in calls),
        **aggregate,
    }
    return {
        "label": label,
        "load_rank": load_rank,
        "call_count": len(calls),
        **replay_payload,
        "chunk_sha256": _sha256(replay_payload),
    }


def _shape_anchor(calls: list[ProfileCall]) -> tuple[list[list[int]], int]:
    shapes = collections.Counter(call.value["logical_len"] for call in calls)
    normalized = [[logical_len, shapes[logical_len]] for logical_len in sorted(shapes)]
    positions = sum(logical_len * count for logical_len, count in normalized)
    return normalized, positions


def build_manifest(
    calls: list[dict[str, Any]], source_workload_sha256: str
) -> dict[str, Any]:
    if len(source_workload_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_workload_sha256
    ):
        raise ReplayManifestError("source workload hash must be lowercase SHA-256")

    profile_calls = select_32k_calls(calls)
    profile_aggregate = _aggregate(profile_calls)
    profile_shapes, profile_positions = _shape_anchor(profile_calls)
    ranked_bins = _rank_bins(lpt_balance(profile_calls))
    chunks = [
        _chunk(label, rank, ranked_bins[rank])
        for label, rank in zip(SELECTED_LABELS, SELECTED_LOAD_RANKS)
    ]

    selected_calls = [
        call for rank in SELECTED_LOAD_RANKS for call in ranked_bins[rank]
    ]
    selected_aggregate = _aggregate(selected_calls)
    selected_shapes, selected_positions = _shape_anchor(selected_calls)
    source_anchor = {
        "source_workload_sha256": source_workload_sha256,
        "call_count": len(profile_calls),
        "logical_shapes": profile_shapes,
        "logical_positions": profile_positions,
        "total_records": profile_aggregate["total_records"],
        "total_nonempty_runs": profile_aggregate["total_nonempty_runs"],
        "total_pair_reductions": profile_aggregate["total_pair_reductions"],
        "max_run": profile_aggregate["max_run"],
        "run_histogram_sha256": _sha256(profile_aggregate["run_histogram"]),
    }
    selected_totals = {
        "chunk_count": len(chunks),
        "call_count": len(selected_calls),
        "logical_shapes": selected_shapes,
        "logical_positions": selected_positions,
        "total_records": selected_aggregate["total_records"],
        "total_nonempty_runs": selected_aggregate["total_nonempty_runs"],
        "total_pair_reductions": selected_aggregate["total_pair_reductions"],
        "max_run": selected_aggregate["max_run"],
        "run_histogram_sha256": _sha256(selected_aggregate["run_histogram"]),
    }
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "source_anchor": source_anchor,
        "selection": {
            "logical_lengths": list(LOGICAL_LENGTHS),
            "lpt_bin_count": LPT_BIN_COUNT,
            "max_calls_per_bin": MAX_CALLS_PER_BIN,
            "selected_load_ranks": list(SELECTED_LOAD_RANKS),
        },
        "chunks": chunks,
        "selected_totals": selected_totals,
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def check_seed_24302(calls: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    metal_msm_runs.check_seed_24302(calls)
    if manifest["source_anchor"] != SEED_24302_SOURCE_ANCHOR:
        raise ReplayManifestError(
            "seed-24302 32K source anchor mismatch: "
            f"expected={SEED_24302_SOURCE_ANCHOR}, "
            f"observed={manifest['source_anchor']}"
        )


def build_from_path(path: Path, expect_seed_24302: bool = False) -> dict[str, Any]:
    calls, source_sha256 = metal_msm_runs.read_calls(path)
    summary = metal_msm_runs.summarize(calls, source_sha256)
    manifest = build_manifest(calls, summary["normalized_workload_sha256"])
    if expect_seed_24302:
        check_seed_24302(calls, manifest)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="aggregate JSONL emitted by the prover"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "write the sensitive owner-only replay manifest here "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--expect-seed-24302",
        action="store_true",
        help="require the authoritative 147-call seed-24302 anchors",
    )
    return parser.parse_args(argv)


def _write_owner_only(path: Path, encoded: str) -> None:
    """Atomically publish a mode-0600 diagnostic without a wider-mode window."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = build_from_path(args.input, args.expect_seed_24302)
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        _write_owner_only(args.output, encoded)
    except (OSError, ReplayManifestError, metal_msm_runs.DiagnosticError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
