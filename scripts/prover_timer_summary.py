#!/usr/bin/env python3
"""Summarize overlapping Midnight prover timer intervals from standard input."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass


TIMER = re.compile(
    r"^\[midnight-prover\] (?P<label>\S+?)"
    r"(?: count=(?P<count>\d+))? "
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)ms @(?P<end>\d+)$"
)
CREATE_PROOF_LABEL = "proof_server.prover.plonk.create_proof"


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int
    elapsed_ms: float
    count: int | None


def interval_union_ns(intervals: list[Interval]) -> int:
    ordered = sorted(intervals, key=lambda interval: (interval.start_ns, interval.end_ns))
    if not ordered:
        return 0
    union = 0
    start = ordered[0].start_ns
    end = ordered[0].end_ns
    for interval in ordered[1:]:
        if interval.start_ns <= end:
            end = max(end, interval.end_ns)
        else:
            union += end - start
            start = interval.start_ns
            end = interval.end_ns
    return union + end - start


def parse(lines: list[str]) -> dict[str, list[Interval]]:
    grouped: dict[str, list[Interval]] = defaultdict(list)
    for line in lines:
        match = TIMER.match(line.rstrip("\n"))
        if match is None:
            continue
        elapsed_ms = float(match.group("elapsed"))
        end_ns = int(match.group("end"))
        elapsed_ns = round(elapsed_ms * 1_000_000)
        count = match.group("count")
        grouped[match.group("label")].append(
            Interval(
                start_ns=end_ns - elapsed_ns,
                end_ns=end_ns,
                elapsed_ms=elapsed_ms,
                count=None if count is None else int(count),
            )
        )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=40, help="maximum rows to print")
    parser.add_argument("--prefix", default="", help="retain only labels with this prefix")
    parser.add_argument(
        "--last-root-batch",
        type=int,
        default=0,
        help="retain timers nested in the last N create-proof root intervals",
    )
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.last_root_batch < 0:
        parser.error("--last-root-batch must be non-negative")

    grouped = parse(list(sys.stdin))
    if args.last_root_batch:
        roots = sorted(grouped.get(CREATE_PROOF_LABEL, []), key=lambda interval: interval.end_ns)
        if len(roots) < args.last_root_batch:
            parser.error(
                f"requested {args.last_root_batch} roots but parsed only {len(roots)}"
            )
        roots = roots[-args.last_root_batch :]
        batch_start = min(interval.start_ns for interval in roots)
        batch_end = max(interval.end_ns for interval in roots)
        grouped = {
            label: [
                interval
                for interval in intervals
                if interval.start_ns >= batch_start and interval.end_ns <= batch_end
            ]
            for label, intervals in grouped.items()
        }
    rows = []
    for label, intervals in grouped.items():
        if not intervals or not label.startswith(args.prefix):
            continue
        rows.append(
            (
                interval_union_ns(intervals) / 1_000_000,
                sum(interval.elapsed_ms for interval in intervals),
                max(interval.elapsed_ms for interval in intervals),
                len(intervals),
                sum(interval.count or 0 for interval in intervals),
                label,
            )
        )
    rows.sort(key=lambda row: (-row[0], -row[1], row[5]))

    print("union_ms      sum_ms      max_ms  calls        count  label")
    for union_ms, sum_ms, max_ms, calls, count, label in rows[: args.top]:
        print(
            f"{union_ms:9.3f}  {sum_ms:10.3f}  {max_ms:10.3f}  "
            f"{calls:5d}  {count:11d}  {label}"
        )
    if not rows:
        print("no matching timer records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
