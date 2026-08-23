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


# The parity binary retains its original compact spellings. The clearer
# `cpu-vs-*` names are profiler-facing aliases: they request two separate,
# alternating proof runs and never combine accelerator implementations.
BACKEND_ALIASES = {
    "cpu-vs-cuda": "both",
    "cpu-vs-metal": "cpu-metal",
    "cpu-vs-metal-hybrid": "cpu-metal-hybrid",
}
BACKEND_CHOICES = (
    "cpu",
    "cuda",
    "metal",
    "metal-hybrid",
    "both",
    "cpu-metal",
    "cpu-metal-hybrid",
    *BACKEND_ALIASES,
)


def normalized_backend(backend):
    """Return the spelling accepted by the proof-backend-parity binary."""
    return BACKEND_ALIASES.get(backend, backend)


def backend_uses_cuda(backend):
    """Whether this selection executes a CUDA route in at least one sample."""
    return normalized_backend(backend) in ("cuda", "both")


def backend_uses_metal(backend):
    """Whether this selection executes a Metal route in at least one sample."""
    return normalized_backend(backend) in (
        "metal",
        "metal-hybrid",
        "cpu-metal",
        "cpu-metal-hybrid",
    )


def add_gate_args(parser, default_backend):
    parser.add_argument("gate")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=default_backend,
        help=(
            "proof route or paired comparison; cpu-vs-* runs canonical CPU and "
            "the named accelerator as separate alternating samples"
        ),
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
        "--metal-quotient-diagnostic",
        action="store_true",
        help=(
            "enable the trusted-local resident quotient shadow on the five-proof "
            "Metal-hybrid transaction gate"
        ),
    )
    parser.add_argument(
        "--cargo-features",
        default=None,
        help=(
            "comma-separated features used to build the parity binary "
            "(default: deterministic-test-rng, plus the requested accelerator and perf-timers for collection)"
        ),
    )


def resolve_cargo_features(args, collection):
    if args.cargo_features:
        features = [feature for feature in args.cargo_features.split(",") if feature]
    else:
        features = ["deterministic-test-rng"]
        if collection:
            # Collection is useless without stamped stage ranges, and a CUDA
            # backend request needs the GPU feature compiled in.
            features.append("perf-timers")
        if backend_uses_cuda(args.backend):
            features.append("gpu-blitzar")
        elif backend_uses_metal(args.backend):
            features.append("metal")
    if getattr(args, "metal_quotient_diagnostic", False):
        features.extend(("metal", "metal-quotient-diagnostic"))
    features = list(dict.fromkeys(features))
    return ",".join(features)


def parity_environment(args):
    env = os.environ.copy()
    # TestState's fee-token setup constructs a proof-free rewards transaction
    # through the ledger test resolver. No fixture keys are read from this
    # directory, but the helper requires a valid root path.
    env.setdefault("MIDNIGHT_LEDGER_TEST_STATIC_DIR", str(SCRIPT_DIR.parent))
    # CUDA zswap paths share sppark scratch state, so serialize those calls until
    # every proof type has a resident context with owned scratch. Metal and CPU
    # lanes are separate builds and must not inherit a CUDA-only control.
    if args.fixture.startswith("zswap-") and backend_uses_cuda(args.backend):
        env.setdefault("MIDNIGHT_GPU_SPPARK_LOCK", "on")
    else:
        env.pop("MIDNIGHT_GPU_SPPARK_LOCK", None)
    if getattr(args, "metal_quotient_diagnostic", False):
        env["MIDNIGHT_METAL_QUOTIENT_DIAGNOSTIC"] = "1"
    else:
        env.pop("MIDNIGHT_METAL_QUOTIENT_DIAGNOSTIC", None)
    return env


def validate_metal_quotient_args(args):
    """Reject profiler selections that could mislabel the quotient diagnostic."""
    if not getattr(args, "metal_quotient_diagnostic", False):
        return None
    if normalized_backend(args.backend) not in ("metal-hybrid", "cpu-metal-hybrid"):
        return "--metal-quotient-diagnostic requires metal-hybrid or cpu-vs-metal-hybrid"
    if args.fixture != "zswap-transaction":
        return "--metal-quotient-diagnostic requires --fixture zswap-transaction"
    if args.warmup < 2 or args.repeat < 5:
        return "--metal-quotient-diagnostic requires --warmup >= 2 and --repeat >= 5"
    return None


def run_command_capture(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_provenance(repo_root, effective_env=None):
    """Environment facts stamped into every collection bundle."""
    if effective_env is None:
        effective_env = os.environ
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
    apple_gpu = None
    displays = run_command_capture(["system_profiler", "SPDisplaysDataType", "-json"])
    if displays:
        try:
            adapters = json.loads(displays).get("SPDisplaysDataType", [])
            if adapters:
                adapter = adapters[0]
                apple_gpu = {
                    "name": adapter.get("sppci_model") or adapter.get("_name"),
                    "cores": adapter.get("sppci_cores"),
                    "metal": adapter.get("spdisplays_mtlgpufamilysupport"),
                }
        except (TypeError, ValueError):
            apple_gpu = {"raw": displays.splitlines()[0]}
    return {
        "git_head": git_head,
        "git_dirty": bool(git_dirty),
        "architecture": run_command_capture(["uname", "-m"]),
        "macos_version": run_command_capture(["sw_vers", "-productVersion"]),
        "xcode": run_command_capture(["xcodebuild", "-version"]),
        "metal_compiler": run_command_capture(
            ["xcrun", "--sdk", "macosx", "metal", "--version"]
        ),
        "thermal_state": run_command_capture(["pmset", "-g", "therm"]),
        "apple_gpu": apple_gpu,
        "gpu_name": gpu_name,
        "driver": driver,
        "gpu_memory": gpu_memory,
        "midnight_env": {
            key: value
            for key, value in sorted(effective_env.items())
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


def stamp_parity_provenance(json_path, repo_root, effective_env=None):
    """Attach the host/toolchain snapshot beside raw parity samples."""
    path = Path(json_path)
    if not path.is_absolute():
        path = repo_root / path
    body = json.loads(path.read_text(encoding="utf-8"))
    body["provenance"] = collect_provenance(repo_root, effective_env)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


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
        normalized_backend(args.backend),
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


def build_parity_binary(repo_root, cargo_features, profile="release"):
    return subprocess.run(
        [
            "cargo",
            "build",
            "--profile",
            profile,
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


def runtime_library_path(repo_root, profile="release"):
    paths = []
    for so_path in (repo_root / "target" / profile / "build").glob("*/out/*.so"):
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

    run = sub.add_parser("run", help="run the deterministic CPU/accelerator parity gate")
    add_gate_args(run, default_backend="both")

    nsys = sub.add_parser("nsys", help="collect an Nsight Systems bundle for a wired gate")
    add_gate_args(nsys, default_backend="cuda")
    nsys.add_argument("--out-dir", help="artifact bundle directory")
    nsys.add_argument("--trace", default="cuda,nvtx,osrt")
    nsys.add_argument("--sample", default="none")
    nsys.add_argument("--force-overwrite", action="store_true", default=True)

    metal = sub.add_parser(
        "metal-trace", help="collect an Apple Metal System Trace bundle for a wired gate"
    )
    add_gate_args(metal, default_backend="metal-hybrid")
    metal.add_argument("--out-dir", help="artifact bundle directory")
    metal.add_argument("--template", default="Metal System Trace")
    metal.add_argument("--time-limit", help="optional xctrace limit such as 30s or 2m")
    metal.add_argument("--force-overwrite", action="store_true")

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
    if hasattr(args, "metal_quotient_diagnostic"):
        quotient_error = validate_metal_quotient_args(args)
        if quotient_error:
            print(f"error: {quotient_error}", file=sys.stderr)
            return 2

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
        env = parity_environment(args)
        rc = subprocess.run(cmd, cwd=repo_root, env=env).returncode
        if rc == 0 and args.json:
            try:
                stamp_parity_provenance(args.json, repo_root, env)
            except (OSError, ValueError) as exc:
                print(f"error: could not stamp parity provenance: {exc}", file=sys.stderr)
                return 3
        return rc

    if args.command == "nsys":
        gate_error = require_parity_gate(args.gate)
        if gate_error:
            print(gate_error, file=sys.stderr)
            return 2
        if backend_uses_metal(args.backend):
            print(
                "error: Nsight Systems cannot profile Metal; use the Metal System Trace workflow",
                file=sys.stderr,
            )
            return 2
        nsys_path = tool_path("nsys")
        if not nsys_path:
            print("error: nsys not found; run scripts/gpuprof.sh doctor", file=sys.stderr)
            return 2

        cargo_features = resolve_cargo_features(args, collection=True)
        cargo_profile = "profiling"
        build_rc = build_parity_binary(repo_root, cargo_features, cargo_profile)
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

        binary = repo_root / "target" / cargo_profile / "proof-backend-parity"
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
        env = parity_environment(args)
        # Prover-internal stage ranges are runtime-gated; collection without
        # them cannot attribute anything below proof_server.prove.execute.
        env.setdefault("MIDNIGHT_PROVER_TIMERS", "1")
        ld_library_path = runtime_library_path(repo_root, cargo_profile)
        if ld_library_path:
            env["LD_LIBRARY_PATH"] = ld_library_path

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
            "cargo_profile": cargo_profile,
            "nsys": nsys_path,
            "profile_command": profile_cmd,
            "json": str(json_path),
            "nsys_report": str(report_rep),
            "nsys_sqlite": str(report_sqlite),
            "ld_library_path": ld_library_path,
            **collect_provenance(repo_root, env),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

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
        try:
            stamp_parity_provenance(json_path, repo_root, env)
        except (OSError, ValueError) as exc:
            print(f"error: could not stamp parity provenance: {exc}", file=sys.stderr)
            return 3

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

    if args.command == "metal-trace":
        gate_error = require_parity_gate(args.gate)
        if gate_error:
            print(gate_error, file=sys.stderr)
            return 2
        if not backend_uses_metal(args.backend):
            print(
                "error: metal-trace requires a Metal backend or CPU-vs-Metal comparison",
                file=sys.stderr,
            )
            return 2
        xctrace_path = run_command_capture(["xcrun", "--find", "xctrace"])
        if not xctrace_path:
            print("error: xctrace not found in the selected Xcode", file=sys.stderr)
            return 2

        cargo_features = resolve_cargo_features(args, collection=True)
        cargo_profile = "profiling"
        build_rc = build_parity_binary(repo_root, cargo_features, cargo_profile)
        if build_rc != 0:
            return build_rc

        stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fixture_label = args.fixture
        if args.fixture == "hash-chain-v2":
            fixture_label = f"{fixture_label}-rounds{args.hash_rounds}"
        out_dir = Path(args.out_dir) if args.out_dir else (
            repo_root / "benchmark-results" / f"gpuprof-metal-{fixture_label}-{stamp}"
        )
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
        trace_path = out_dir / "metal-system.trace"
        if trace_path.exists():
            if not args.force_overwrite:
                print(
                    f"error: trace already exists at {trace_path}; pass --force-overwrite",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(trace_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = out_dir / "stdout.txt"
        stderr_path = out_dir / "stderr.txt"
        xctrace_stdout_path = out_dir / "xctrace-stdout.txt"
        json_path = Path(args.json) if args.json else out_dir / "gpuprof.json"
        if not json_path.is_absolute():
            json_path = repo_root / json_path
        binary = repo_root / "target" / cargo_profile / "proof-backend-parity"
        profile_cmd = [
            xctrace_path,
            "record",
            "--template",
            args.template,
            "--output",
            str(trace_path),
            "--no-prompt",
            "--target-stdout",
            str(stdout_path),
        ]
        if args.time_limit:
            profile_cmd.extend(["--time-limit", args.time_limit])
        profile_cmd.extend(
            ["--launch", "--", str(binary), *parity_args(args, json_path=json_path)]
        )

        env = parity_environment(args)
        env.setdefault("MIDNIGHT_PROVER_TIMERS", "1")

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
            "cargo_profile": cargo_profile,
            "xctrace": xctrace_path,
            "template": args.template,
            "profile_command": profile_cmd,
            "json": str(json_path),
            "trace": str(trace_path),
            **collect_provenance(repo_root, env),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        with xctrace_stdout_path.open("w") as xctrace_stdout, stderr_path.open("w") as stderr:
            rc = subprocess.run(
                profile_cmd,
                cwd=repo_root,
                stdout=xctrace_stdout,
                stderr=stderr,
                env=env,
            ).returncode
        if rc != 0:
            print(f"xctrace recording failed; see {stderr_path}", file=sys.stderr)
            return rc
        if not trace_path.exists():
            print("error: xctrace completed without producing a trace", file=sys.stderr)
            return 3
        stamped = stamped_timer_lines(stderr_path)
        if stamped == 0:
            print(
                "error: no stamped perf-timer lines in stderr - the profiled binary was "
                "built without perf-timers (or is stale); the Metal trace cannot be attributed. "
                f"Rebuild with --cargo-features {cargo_features}",
                file=sys.stderr,
            )
            return 3
        try:
            stamp_parity_provenance(json_path, repo_root, env)
        except (OSError, ValueError) as exc:
            print(f"error: could not stamp parity provenance: {exc}", file=sys.stderr)
            return 3

        print(f"bundle={out_dir}")
        print(f"json={json_path}")
        print(f"trace={trace_path}")
        print(f"stamped_timer_lines={stamped}")
        return 0

    print(
        f"error: gpuprof {args.command} is reserved for the accepted-baseline workflow "
        "but is not implemented yet",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
