#!/usr/bin/env python3
"""Validate and normalize aggregate-only Metal MSM workload diagnostics."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "midnight-msm-runs-v1"
OUTPUT_SCHEMA = "midnight-msm-run-summary-v1"
CALL_KEYS = {
    "schema",
    "call",
    "logical_len",
    "nonzero_scalars",
    "window_bits",
    "window_count",
    "windows",
    "total_records",
    "total_nonempty_runs",
    "total_pair_reductions",
}
WINDOW_KEYS = {
    "window",
    "records",
    "nonempty_runs",
    "max_run",
    "run_histogram",
}
PROFILES = {
    8_191: (8, 32, "8K"),
    8_192: (8, 32, "8K"),
    16_383: (9, 29, "16K"),
    16_384: (9, 29, "16K"),
    32_767: (10, 26, "32K"),
    32_768: (10, 26, "32K"),
}
SEED_24302_SHAPES = {
    8_191: 1,
    8_192: 22,
    16_383: 2,
    16_384: 60,
    32_767: 2,
    32_768: 60,
}


class DiagnosticError(ValueError):
    """The diagnostic is malformed, incomplete, or violates its minimized schema."""


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DiagnosticError(
            f"{context}: keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _uint(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DiagnosticError(f"{context}: expected a nonnegative integer")
    return value


def _validate_window(
    value: Any, expected_window: int, bucket_count: int, context: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosticError(f"{context}: expected an object")
    _exact_keys(value, WINDOW_KEYS, context)
    window = _uint(value["window"], f"{context}.window")
    if window != expected_window:
        raise DiagnosticError(
            f"{context}.window: expected {expected_window}, observed {window}"
        )

    records = _uint(value["records"], f"{context}.records")
    runs = _uint(value["nonempty_runs"], f"{context}.nonempty_runs")
    max_run = _uint(value["max_run"], f"{context}.max_run")
    histogram = value["run_histogram"]
    if not isinstance(histogram, list):
        raise DiagnosticError(f"{context}.run_histogram: expected an array")

    normalized: list[list[int]] = []
    prior_run_len = 0
    calculated_records = 0
    calculated_runs = 0
    for index, pair in enumerate(histogram):
        item_context = f"{context}.run_histogram[{index}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise DiagnosticError(f"{item_context}: expected [run_length, count]")
        run_len = _uint(pair[0], f"{item_context}[0]")
        count = _uint(pair[1], f"{item_context}[1]")
        if run_len == 0 or count == 0 or run_len <= prior_run_len:
            raise DiagnosticError(
                f"{item_context}: entries must be positive and strictly ordered"
            )
        prior_run_len = run_len
        calculated_records += run_len * count
        calculated_runs += count
        normalized.append([run_len, count])

    calculated_max = normalized[-1][0] if normalized else 0
    if calculated_records != records:
        raise DiagnosticError(
            f"{context}: histogram records={calculated_records}, reported={records}"
        )
    if calculated_runs != runs:
        raise DiagnosticError(
            f"{context}: histogram runs={calculated_runs}, reported={runs}"
        )
    if calculated_max != max_run:
        raise DiagnosticError(
            f"{context}: histogram max={calculated_max}, reported={max_run}"
        )
    if runs > bucket_count:
        raise DiagnosticError(
            f"{context}: {runs} nonempty runs exceed {bucket_count} buckets"
        )
    return {
        "window": window,
        "records": records,
        "nonempty_runs": runs,
        "max_run": max_run,
        "run_histogram": normalized,
    }


def validate_call(value: Any, line_number: int) -> dict[str, Any]:
    context = f"line {line_number}"
    if not isinstance(value, dict):
        raise DiagnosticError(f"{context}: expected an object")
    _exact_keys(value, CALL_KEYS, context)
    if value["schema"] != INPUT_SCHEMA:
        raise DiagnosticError(
            f"{context}.schema: expected {INPUT_SCHEMA!r}, observed {value['schema']!r}"
        )

    call = _uint(value["call"], f"{context}.call")
    logical_len = _uint(value["logical_len"], f"{context}.logical_len")
    nonzero_scalars = _uint(value["nonzero_scalars"], f"{context}.nonzero_scalars")
    if nonzero_scalars > logical_len:
        raise DiagnosticError(f"{context}: nonzero scalar count exceeds logical length")
    if logical_len not in PROFILES:
        raise DiagnosticError(f"{context}: unsupported logical length {logical_len}")
    expected_bits, expected_windows, profile_class = PROFILES[logical_len]
    window_bits = _uint(value["window_bits"], f"{context}.window_bits")
    window_count = _uint(value["window_count"], f"{context}.window_count")
    if (window_bits, window_count) != (expected_bits, expected_windows):
        raise DiagnosticError(
            f"{context}: profile mismatch: expected {expected_bits}/{expected_windows}, "
            f"observed {window_bits}/{window_count}"
        )

    windows = value["windows"]
    if not isinstance(windows, list) or len(windows) != window_count:
        raise DiagnosticError(f"{context}.windows: expected {window_count} entries")
    bucket_count = 1 << (window_bits - 1)
    windows = [
        _validate_window(window, index, bucket_count, f"{context}.windows[{index}]")
        for index, window in enumerate(windows)
    ]
    total_records = sum(window["records"] for window in windows)
    total_runs = sum(window["nonempty_runs"] for window in windows)
    total_pairs = total_records - total_runs
    for key, calculated in [
        ("total_records", total_records),
        ("total_nonempty_runs", total_runs),
        ("total_pair_reductions", total_pairs),
    ]:
        reported = _uint(value[key], f"{context}.{key}")
        if reported != calculated:
            raise DiagnosticError(
                f"{context}.{key}: calculated={calculated}, reported={reported}"
            )
    return {
        "call": call,
        "logical_len": logical_len,
        "nonzero_scalars": nonzero_scalars,
        "profile_class": profile_class,
        "window_bits": window_bits,
        "window_count": window_count,
        "windows": windows,
        "total_records": total_records,
        "total_nonempty_runs": total_runs,
        "total_pair_reductions": total_pairs,
    }


def read_calls(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    calls: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DiagnosticError(f"line {line_number}: invalid JSON: {error}") from error
        calls.append(validate_call(value, line_number))
    if not calls:
        raise DiagnosticError("diagnostic contains no calls")
    call_ids = [call["call"] for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise DiagnosticError("diagnostic contains duplicate call identifiers")
    return calls, hashlib.sha256(raw).hexdigest()


def _merge_histogram(target: collections.Counter[int], entries: list[list[int]]) -> None:
    for run_len, count in entries:
        target[run_len] += count


def summarize(calls: list[dict[str, Any]], source_sha256: str) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for call in calls:
        by_class[call["profile_class"]].append(call)

    classes = []
    for profile_class in ("8K", "16K", "32K"):
        class_calls = by_class.get(profile_class, [])
        if not class_calls:
            continue
        window_count = class_calls[0]["window_count"]
        aggregate_windows = []
        for window_index in range(window_count):
            histogram: collections.Counter[int] = collections.Counter()
            for call in class_calls:
                _merge_histogram(histogram, call["windows"][window_index]["run_histogram"])
            aggregate_windows.append(
                {
                    "window": window_index,
                    "records": sum(
                        call["windows"][window_index]["records"] for call in class_calls
                    ),
                    "nonempty_runs": sum(
                        call["windows"][window_index]["nonempty_runs"]
                        for call in class_calls
                    ),
                    "max_run": max(
                        call["windows"][window_index]["max_run"] for call in class_calls
                    ),
                    "run_histogram": [[key, histogram[key]] for key in sorted(histogram)],
                }
            )
        max_run = max(window["max_run"] for window in aggregate_windows)
        classes.append(
            {
                "class": profile_class,
                "calls": len(class_calls),
                "logical_positions": sum(call["logical_len"] for call in class_calls),
                "nonzero_scalars": sum(call["nonzero_scalars"] for call in class_calls),
                "window_bits": class_calls[0]["window_bits"],
                "window_count": window_count,
                "total_records": sum(call["total_records"] for call in class_calls),
                "total_nonempty_runs": sum(
                    call["total_nonempty_runs"] for call in class_calls
                ),
                "total_pair_reductions": sum(
                    call["total_pair_reductions"] for call in class_calls
                ),
                "max_run": max_run,
                "balanced_depth": (max_run - 1).bit_length() if max_run else 0,
                "windows": aggregate_windows,
            }
        )

    summary = {
        "schema": OUTPUT_SCHEMA,
        "source_sha256": source_sha256,
        "calls": len(calls),
        "logical_positions": sum(call["logical_len"] for call in calls),
        "nonzero_scalars": sum(call["nonzero_scalars"] for call in calls),
        "total_records": sum(call["total_records"] for call in calls),
        "total_nonempty_runs": sum(call["total_nonempty_runs"] for call in calls),
        "total_pair_reductions": sum(call["total_pair_reductions"] for call in calls),
        "shapes": dict(sorted(collections.Counter(call["logical_len"] for call in calls).items())),
        "classes": classes,
    }
    normalized = dict(summary)
    normalized.pop("source_sha256")
    normalized_bytes = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    summary["normalized_workload_sha256"] = hashlib.sha256(normalized_bytes).hexdigest()
    return summary


def check_seed_24302(calls: list[dict[str, Any]]) -> None:
    shapes = collections.Counter(call["logical_len"] for call in calls)
    if dict(shapes) != SEED_24302_SHAPES:
        raise DiagnosticError(
            f"seed-24302 shape mismatch: expected={SEED_24302_SHAPES}, observed={dict(shapes)}"
        )
    logical_positions = sum(call["logical_len"] for call in calls)
    nonzero_scalars = sum(call["nonzero_scalars"] for call in calls)
    if logical_positions != 3_235_835:
        raise DiagnosticError(
            f"seed-24302 logical count mismatch: expected=3235835, observed={logical_positions}"
        )
    if nonzero_scalars != 1_988_928:
        raise DiagnosticError(
            f"seed-24302 nonzero count mismatch: expected=1988928, observed={nonzero_scalars}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="aggregate JSONL emitted by the prover")
    parser.add_argument("--output", type=Path, help="write normalized aggregate JSON here")
    parser.add_argument(
        "--expect-seed-24302",
        action="store_true",
        help="require the exact 147-call balanced-zswap anchor",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        calls, source_sha256 = read_calls(args.input)
        if args.expect_seed_24302:
            check_seed_24302(calls)
        summary = summarize(calls, source_sha256)
    except (OSError, DiagnosticError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        args.output.chmod(0o600)
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
