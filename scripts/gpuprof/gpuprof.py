#!/usr/bin/env python3
"""Nightstream-style gpuprof compatibility entry point for midnight-ledger."""

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from proofscope.analyze import analyze_bundle
from proofscope.compare import check_stages, diff_rows, load_stage_map, print_check, print_diff
from proofscope.parse import TS_RE, parse_phase_line
from proofscope.report import render_summary


def add_gate_args(parser, default_backend):
    parser.add_argument("gate")
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "both"), default=default_backend
    )
    parser.add_argument(
        "--fixture",
        choices=(
            "zswap-transaction",
            "minimal-v2",
            "hash-chain-v2",
            "zswap-output",
            "zswap-spend",
        ),
        default="zswap-transaction",
    )
    parser.add_argument("--hash-rounds", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", default="24301")
    parser.add_argument("--json", help="write a machine-readable parity report")
    parser.add_argument(
        "--cargo-features",
        default=None,
        help=(
            "comma-separated features used to build the parity binary "
            "(default: deterministic-test-rng, plus gpu-blitzar for CUDA and perf-timers for collection)"
        ),
    )


def resolve_cargo_features(args, collection):
    if args.cargo_features:
        return args.cargo_features
    features = ["deterministic-test-rng"]
    if collection:
        # Collection is useless without stamped stage ranges, and a CUDA
        # backend request needs the GPU feature compiled in.
        features.append("perf-timers")
    if args.backend != "cpu":
        features.append("gpu-blitzar")
    return ",".join(features)


def parity_environment(args):
    env = os.environ.copy()
    # TestState's fee-token setup constructs a proof-free rewards transaction
    # through the ledger test resolver. No fixture keys are read from this
    # directory, but the helper requires a valid root path.
    env.setdefault("MIDNIGHT_LEDGER_TEST_STATIC_DIR", str(SCRIPT_DIR.parent))
    # The current zswap transaction relations still use CUDA leaf paths. Those
    # paths share sppark scratch state, so serialize calls until every proof
    # type has a resident context with owned scratch.
    if args.fixture.startswith("zswap-"):
        env.setdefault("MIDNIGHT_GPU_SPPARK_LOCK", "on")
    return env


def run_command_capture(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_provenance(repo_root):
    """Environment facts stamped into every collection bundle."""
    git_head = run_command_capture(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_dirty = run_command_capture(["git", "-C", str(repo_root), "status", "--porcelain"])
    smi = run_command_capture(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    gpu_name = driver = gpu_memory = None
    if smi:
        parts = [part.strip() for part in smi.splitlines()[0].split(",")]
        if len(parts) >= 2:
            gpu_name, driver = parts[0], parts[1]
        if len(parts) >= 3:
            gpu_memory = parts[2]
    return {
        "git_head": git_head,
        "git_dirty": bool(git_dirty),
        "gpu_name": gpu_name,
        "driver": driver,
        "gpu_memory": gpu_memory,
        "midnight_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("MIDNIGHT_GPU_", "MIDNIGHT_PROVER_", "MIDNIGHT_PROOF_SERVER_"))
        },
    }


def stamped_timer_lines(stderr_path):
    """Count perf-timer lines the analyzer can align. Zero means a stale or
    perf-timer-less binary and the bundle is unattributable."""
    count = 0
    try:
        for line in Path(stderr_path).read_text(encoding="utf-8", errors="replace").splitlines():
            if TS_RE.search(line) and parse_phase_line(line):
                count += 1
    except OSError:
        return 0
    return count


def append_history(history_path, entry):
    """One JSONL line per collection: the campaign trend log. Never fails a run."""
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        print(f"history: appended to {history_path}")
    except Exception as exc:
        print(f"history: skipped ({exc})")


def require_parity_gate(gate):
    if gate == "proof-backend-parity":
        return None
    return (
        "error: only the proof-backend-parity gate is wired; add a "
        "proof-server benchmark/parity gate before using other names"
    )


def parity_args(args, json_path=None):
    cmd = [
        "--backend",
        args.backend,
        "--fixture",
        args.fixture,
        "--hash-rounds",
        str(args.hash_rounds),
        "--iterations",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--seed",
        args.seed,
    ]
    if json_path:
        cmd.extend(["--json", str(json_path)])
    elif args.json:
        cmd.extend(["--json", args.json])
    return cmd


def build_parity_binary(repo_root, cargo_features):
    return subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "midnight-proof-server",
            "--features",
            cargo_features,
            "--bin",
            "proof-backend-parity",
        ],
        cwd=repo_root,
    ).returncode


def tool_path(name):
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/usr/local/cuda-13.0", "/usr/local/cuda", "/opt/cuda"):
        candidate = Path(prefix) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


def runtime_library_path(repo_root):
    paths = []
    for so_path in (repo_root / "target" / "release" / "build").glob("*/out/*.so"):
        parent = str(so_path.parent)
        if parent not in paths:
            paths.append(parent)
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpuprof")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze an existing proof-server Nsight bundle")
    analyze.add_argument("bundle_dir")

    run = sub.add_parser("run", help="run the deterministic CPU/CUDA parity gate")
    add_gate_args(run, default_backend="both")

    nsys = sub.add_parser("nsys", help="collect an Nsight Systems bundle for a wired gate")
    add_gate_args(nsys, default_backend="cuda")
    nsys.add_argument("--out-dir", help="artifact bundle directory")
    nsys.add_argument("--trace", default="cuda,nvtx,osrt")
    nsys.add_argument("--sample", default="none")
    nsys.add_argument("--force-overwrite", action="store_true", default=True)

    diff = sub.add_parser("diff", help="exploratory per-stage delta between two analyzed bundles")
    diff.add_argument("baseline")
    diff.add_argument("candidate")
    diff.add_argument("--min-stage-ms", type=float, default=1.0)

    check = sub.add_parser("check", help="exit-coded regression gate between two analyzed bundles")
    check.add_argument("baseline")
    check.add_argument("candidate")
    check.add_argument("--tolerance", type=float, default=0.05)
    check.add_argument("--abs-floor-ms", type=float, default=2.0)
    check.add_argument("--min-stage-ms", type=float, default=5.0)

    trend = sub.add_parser("trend", help="collection history from the JSONL trend log")
    trend.add_argument("--last", type=int, default=20)

    metadata = sub.add_parser("metadata", help="print the environment/provenance snapshot")

    args = parser.parse_args(argv)
    repo_root = SCRIPT_DIR.parent
    history_path = repo_root / "benchmark-results" / "gpuprof-history.jsonl"

    if args.command == "analyze":
        try:
            output_path, report = analyze_bundle(args.bundle_dir)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        reconciliation = report["reconciliation"]
        print(f"wrote {output_path}")
        render_summary(report)
        return 0 if reconciliation["ok"] else 2

    if args.command == "diff":
        try:
            base_map, _ = load_stage_map(args.baseline)
            cand_map, _ = load_stage_map(args.candidate)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print_diff(diff_rows(base_map, cand_map, args.min_stage_ms))
        return 0

    if args.command == "check":
        try:
            base_map, _ = load_stage_map(args.baseline)
            cand_map, _ = load_stage_map(args.candidate)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        checked, failures = check_stages(
            base_map,
            cand_map,
            tolerance=args.tolerance,
            abs_floor_ms=args.abs_floor_ms,
            min_stage_ms=args.min_stage_ms,
        )
        print_check(checked, failures)
        return 1 if failures else 0

    if args.command == "trend":
        if not history_path.exists():
            print(f"error: no history at {history_path} - collect some bundles first", file=sys.stderr)
            return 2
        lines = history_path.read_text(encoding="utf-8").splitlines()[-args.last :]
        print(f"{'timestamp':<22} {'backend':<8} {'fixture':<14} {'rounds':>6} {'headline_ms':>12} bundle")
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            stages = entry.get("stages") or {}
            headline = stages.get("proof_server.prover.plonk.create_proof") or max(
                stages.values(), default=0.0
            )
            print(
                f"{entry.get('ts', '?'):<22} {entry.get('backend', '?'):<8} "
                f"{entry.get('fixture', '?'):<14} {entry.get('hash_rounds', 0):>6} "
                f"{headline:>12.1f} {entry.get('bundle', '?')}"
            )
        return 0

    if args.command == "metadata":
        print(json.dumps(collect_provenance(repo_root), indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        gate_error = require_parity_gate(args.gate)
        if gate_error:
            print(gate_error, file=sys.stderr)
            return 2
        cmd = [
            "cargo",
            "run",
            "--release",
            "-p",
            "midnight-proof-server",
            "--features",
            resolve_cargo_features(args, collection=False),
            "--bin",
            "proof-backend-parity",
            "--",
            *parity_args(args),
        ]
        return subprocess.run(cmd, cwd=repo_root, env=parity_environment(args)).returncode

    if args.command == "nsys":
        gate_error = require_parity_gate(args.gate)
        if gate_error:
            print(gate_error, file=sys.stderr)
            return 2
        nsys_path = tool_path("nsys")
        if not nsys_path:
            print("error: nsys not found; run scripts/gpuprof.sh doctor", file=sys.stderr)
            return 2

        cargo_features = resolve_cargo_features(args, collection=True)
        build_rc = build_parity_binary(repo_root, cargo_features)
        if build_rc != 0:
            return build_rc

        stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fixture_label = args.fixture
        if args.fixture == "hash-chain-v2":
            fixture_label = f"{fixture_label}-rounds{args.hash_rounds}"
        out_dir = Path(args.out_dir) if args.out_dir else (
            repo_root / "benchmark-results" / f"gpuprof-nsys-{fixture_label}-{stamp}"
        )
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
        nsys_dir = out_dir / "nsys"
        nsys_dir.mkdir(parents=True, exist_ok=True)
        report_prefix = nsys_dir / "report"
        report_rep = report_prefix.with_suffix(".nsys-rep")
        report_sqlite = report_prefix.with_suffix(".sqlite")
        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        json_path = Path(args.json) if args.json else out_dir / "gpuprof.json"
        if not json_path.is_absolute():
            json_path = repo_root / json_path

        binary = repo_root / "target" / "release" / "proof-backend-parity"
        profile_cmd = [
            nsys_path,
            "profile",
            f"--trace={args.trace}",
            f"--sample={args.sample}",
            "--force-overwrite=true",
            "--output",
            str(report_prefix),
            str(binary),
            *parity_args(args, json_path=json_path),
        ]
        metadata = {
            "created_at": stamp,
            "gate": args.gate,
            "backend": args.backend,
            "fixture": args.fixture,
            "hash_rounds": args.hash_rounds,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "seed": args.seed,
            "cargo_features": cargo_features,
            "nsys": nsys_path,
            "profile_command": profile_cmd,
            "json": str(json_path),
            "nsys_report": str(report_rep),
            "nsys_sqlite": str(report_sqlite),
            "ld_library_path": runtime_library_path(repo_root),
            **collect_provenance(repo_root),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        env = parity_environment(args)
        # Prover-internal stage ranges are runtime-gated; collection without
        # them cannot attribute anything below proof_server.prove.execute.
        env.setdefault("MIDNIGHT_PROVER_TIMERS", "1")
        ld_library_path = runtime_library_path(repo_root)
        if ld_library_path:
            env["LD_LIBRARY_PATH"] = ld_library_path
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            rc = subprocess.run(
                profile_cmd,
                cwd=repo_root,
                stdout=stdout,
                stderr=stderr,
                env=env,
            ).returncode
        if rc != 0:
            print(f"nsys profile failed; see {stderr_path}", file=sys.stderr)
            return rc

        export_cmd = [
            nsys_path,
            "export",
            "--type",
            "sqlite",
            "--output",
            str(report_sqlite),
            str(report_rep),
        ]
        export_rc = subprocess.run(export_cmd, cwd=repo_root).returncode
        if export_rc != 0:
            print("warning: nsys sqlite export failed", file=sys.stderr)

        print(f"bundle={out_dir}")
        print(f"json={json_path}")
        print(f"nsys_report={report_rep}")
        print(f"nsys_sqlite={report_sqlite}")

        stamped = stamped_timer_lines(stderr_path)
        if stamped == 0:
            print(
                "error: no stamped perf-timer lines in stderr - the profiled binary was "
                "built without perf-timers (or is stale); the bundle cannot be attributed. "
                f"Rebuild with --cargo-features {cargo_features}",
                file=sys.stderr,
            )
            return 3
        print(f"stamped_timer_lines={stamped}")

        try:
            output_path, report = analyze_bundle(out_dir)
        except Exception as exc:
            print(f"warning: bundle collected but analysis failed: {exc}", file=sys.stderr)
            return 0
        print(f"wrote {output_path}")
        render_summary(report)

        stage_walls = {
            row["stage_id"]: round(row.get("window_wall_ms", 0.0), 1)
            for row in report.get("stages", [])
            if row.get("window_wall_ms", 0.0) >= 5.0
        }
        append_history(
            history_path,
            {
                "ts": stamp,
                "gate": args.gate,
                "backend": args.backend,
                "fixture": args.fixture,
                "hash_rounds": args.hash_rounds,
                "seed": args.seed,
                "git_head": metadata.get("git_head"),
                "git_dirty": metadata.get("git_dirty"),
                "bundle": str(out_dir),
                "reconciliation_ok": report["reconciliation"]["ok"],
                "stages": stage_walls,
            },
        )
        return 0 if report["reconciliation"]["ok"] else 2

    print(
        f"error: gpuprof {args.command} is reserved for the accepted-baseline workflow "
        "but is not implemented yet",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
