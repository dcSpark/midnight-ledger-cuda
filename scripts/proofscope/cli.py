"""Command-line entry point for proofscope."""

import argparse
import json
import sys
from pathlib import Path

try:
    from .analyze import analyze_bundle
    from .compare import check_stages, diff_rows, load_stage_map, print_check, print_diff
    from .report import render_summary
except ImportError:  # Direct `python scripts/proofscope/cli.py ...` execution.
    from analyze import analyze_bundle
    from compare import check_stages, diff_rows, load_stage_map, print_check, print_diff
    from report import render_summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="proofscope")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="derive analysis artifacts from a proof-server Nsight bundle"
    )
    analyze.add_argument(
        "bundle_dir",
        help="bundle directory containing nsys/report.sqlite, report.sqlite, or proofscope.sqlite",
    )
    analyze.add_argument(
        "--quiet", action="store_true", help="suppress the terminal report; JSON only"
    )

    report = sub.add_parser("report", help="re-render the terminal report from an analyzed bundle")
    report.add_argument("bundle_dir", help="bundle or analysis directory with stages.json")

    diff = sub.add_parser("diff", help="exploratory per-stage delta between two analyses")
    diff.add_argument("baseline", help="bundle dir or stages.json for the baseline")
    diff.add_argument("candidate", help="bundle dir or stages.json for the candidate")
    diff.add_argument("--min-stage-ms", type=float, default=1.0)

    check = sub.add_parser("check", help="exit-coded regression gate between two analyses")
    check.add_argument("baseline")
    check.add_argument("candidate")
    check.add_argument("--tolerance", type=float, default=0.05, help="relative limit (default 5%%)")
    check.add_argument("--abs-floor-ms", type=float, default=2.0)
    check.add_argument("--min-stage-ms", type=float, default=5.0)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        output_path, out = analyze_bundle(args.bundle_dir)
        reconciliation = out["reconciliation"]
        print(f"wrote {output_path}")
        if not args.quiet:
            render_summary(out)
        else:
            print(
                f"stages={out['counts']['stages']} "
                f"max_reconciliation_error_ms={reconciliation['max_abs_error_ms']:.6f} "
                f"idle_overlap_clamp_ms={reconciliation['idle_overlap_clamp_ms']:.6f} "
                f"unknown_labels={len(out['unknown_nvtx'])}"
            )
        return 0 if reconciliation["ok"] else 2

    if args.command == "report":
        report_path = find_stages_json(args.bundle_dir)
        render_summary(json.loads(report_path.read_text(encoding="utf-8")))
        return 0

    if args.command == "diff":
        base_map, _ = load_stage_map(args.baseline)
        cand_map, _ = load_stage_map(args.candidate)
        print_diff(diff_rows(base_map, cand_map, args.min_stage_ms))
        return 0

    if args.command == "check":
        base_map, _ = load_stage_map(args.baseline)
        cand_map, _ = load_stage_map(args.candidate)
        checked, failures = check_stages(
            base_map,
            cand_map,
            tolerance=args.tolerance,
            abs_floor_ms=args.abs_floor_ms,
            min_stage_ms=args.min_stage_ms,
        )
        print_check(checked, failures)
        return 1 if failures else 0

    parser.error(f"unknown command: {args.command}")
    return 2


def find_stages_json(path):
    p = Path(path)
    for candidate in (p, p / "analysis" / "stages.json", p / "stages.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no stages.json under {path}")


if __name__ == "__main__":
    sys.exit(main())
