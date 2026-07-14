"""Read raw Nsight Systems SQLite exports for proof-server profiling bundles."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

ONLINE_RE = re.compile(r"(?:proofscope|proof-server|prove).*(?:gpu|cuda)=([\d.]+)ms")
TS_RE = re.compile(r"@(\d+)\s*$")
DUR_RE = re.compile(r"([\d][\d_,.]*)\s*(ms|µs|us|ns|s)\b")

LINE_PATTERNS = [
    (re.compile(r"^\[proofscope\]\s+(.+?)\s{2,}"), "proofscope"),
    (re.compile(r"^\[proof-server\]\s+(.+?)\s{2,}"), "proof_server"),
    (
        re.compile(r"^\[midnight-prover\]\s+(.+?)(?:\s+count=\d+)?\s+[\d_,.]+\s*(?:ms|µs|us|ns|s)\b"),
        "proof_server",
    ),
    (re.compile(r"^\[zkir\]\s+(.+?)\s{2,}"), "zkir"),
    (re.compile(r"^\[zswap\]\s+(.+?)\s{2,}"), "zswap"),
    (re.compile(r"^\[ledger\]\s+(.+?)\s{2,}"), "ledger"),
]

ALIASES = {
    "proof_server:payload": "proof_server.request.payload",
    "proof_server:decode": "proof_server.request.decode",
    "proof_server:resolve_ir": "proof_server.request.resolve_ir",
    "proof_server:resolve_key": "proof_server.request.resolve_key",
    "proof_server:prove": "proof_server.prove",
    "proof_server:check": "proof_server.prove.check",
    "proofscope:payload": "proof_server.request.payload",
    "proofscope:decode": "proof_server.request.decode",
    "proofscope:prove": "proof_server.prove",
    "proofscope:check": "proof_server.prove.check",
    "zkir:prove": "zkir.prove",
    "zswap:input": "zswap.input",
    "zswap:output": "zswap.output",
    "ledger:transaction": "ledger.transaction",
}

UNIT_NS = {"s": 1e9, "ms": 1e6, "µs": 1e3, "us": 1e3, "ns": 1.0}


@dataclass(frozen=True)
class InputBundle:
    requested_dir: Path
    analysis_dir: Path
    run_dir: Path
    sqlite_path: Path
    stdout_path: Path | None
    stderr_path: Path | None
    selection: str


def table_exists(db, name):
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _query(db, table, sql):
    return db.execute(sql).fetchall() if table_exists(db, table) else []


def load_timeline(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    kernels = [
        {
            "start": row[0],
            "end": row[1],
            "name": row[2],
            "stream_id": row[3],
            "correlation_id": row[4],
            "registers_per_thread": row[5],
            "grid": [row[6], row[7], row[8]],
            "block": [row[9], row[10], row[11]],
            "static_shared_memory": row[12],
            "dynamic_shared_memory": row[13],
            "local_memory_per_thread": row[14],
        }
        for row in _query(
            db,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "SELECT k.start, k.end, s.value, k.streamId, k.correlationId, "
            "k.registersPerThread, k.gridX, k.gridY, k.gridZ, k.blockX, "
            "k.blockY, k.blockZ, k.staticSharedMemory, k.dynamicSharedMemory, "
            "k.localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id",
        )
    ]
    memcpys = [
        {
            "start": row[0],
            "end": row[1],
            "bytes": row[2],
            "copy_kind": row[3],
            "stream_id": row[4],
            "correlation_id": row[5],
            "src_kind": row[6],
            "dst_kind": row[7],
            "copy_count": row[8],
        }
        for row in _query(
            db,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "SELECT start, end, bytes, copyKind, streamId, correlationId, "
            "srcKind, dstKind, copyCount FROM CUPTI_ACTIVITY_KIND_MEMCPY",
        )
    ]
    memsets = [
        {
            "start": row[0],
            "end": row[1],
            "bytes": row[2],
            "stream_id": row[3],
            "correlation_id": row[4],
            "mem_kind": row[5],
        }
        for row in _query(
            db,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "SELECT start, end, bytes, streamId, correlationId, memKind "
            "FROM CUPTI_ACTIVITY_KIND_MEMSET",
        )
    ]
    syncs = [
        {
            "start": row[0],
            "end": row[1],
            "stream_id": row[2],
            "correlation_id": row[3],
            "sync_type": row[4],
            "sync_label": row[5],
        }
        for row in _query(
            db,
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            "SELECT s.start, s.end, s.streamId, s.correlationId, s.syncType, "
            "coalesce(e.label, e.name, '') FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s "
            "LEFT JOIN ENUM_CUPTI_SYNC_TYPE e ON s.syncType = e.id",
        )
    ]
    api_calls = [
        {
            "start": row[0],
            "end": row[1],
            "name": row[2],
            "correlation_id": row[3],
            "thread_id": row[4],
            "return_value": row[5],
        }
        for row in _query(
            db,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "SELECT r.start, r.end, s.value, r.correlationId, r.globalTid, "
            "r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId = s.id",
        )
    ]
    nvtx = [
        {
            "start": row[0],
            "end": row[1],
            "event_type": row[2],
            "text": row[3],
            "range_id": row[4],
            "thread_id": row[5],
        }
        for row in _query(
            db,
            "NVTX_EVENTS",
            "SELECT start, end, eventType, text, rangeId, globalTid "
            "FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start",
        )
    ]
    session_start = None
    if table_exists(db, "TARGET_INFO_SESSION_START_TIME"):
        row = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        session_start = row[0] if row else None
    db.close()
    return {
        "kernels": kernels,
        "memcpys": memcpys,
        "memsets": memsets,
        "syncs": syncs,
        "api_calls": api_calls,
        "nvtx": nvtx,
        "session_start": session_start,
    }


def parse_phase_line(line):
    duration = DUR_RE.search(line)
    if not duration:
        return None
    elapsed_ns = (
        float(duration.group(1).replace(",", "").replace("_", ""))
        * UNIT_NS[duration.group(2)]
    )
    for pattern, family in LINE_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        raw_label = " ".join(group for group in match.groups() if group).strip()
        label = normalize_label(raw_label)
        prefixed = normalize_label(f"{family}:{label}")
        resolved = ALIASES.get(prefixed) or ALIASES.get(label)
        if resolved is None:
            resolved = label if "." in label else prefixed
        return {
            "label": resolved,
            "family": family,
            "elapsed_ns": elapsed_ns,
        }
    return None


def build_phases(stderr_path, session_start_epoch_ns, nvtx_ranges):
    phases = []
    stamped = 0
    parsed_total = 0
    if stderr_path and session_start_epoch_ns is not None:
        for line in Path(stderr_path).read_text(encoding="utf-8", errors="replace").splitlines():
            timestamp = TS_RE.search(line)
            parsed = parse_phase_line(line)
            if parsed:
                parsed_total += 1
            if not (timestamp and parsed):
                continue
            stamped += 1
            end = int(timestamp.group(1)) - session_start_epoch_ns
            start = int(end - parsed["elapsed_ns"])
            phases.append(
                {
                    "label": parsed["label"],
                    "family": parsed["family"],
                    "source": "stderr",
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )

    nvtx_phases = []
    for row in nvtx_ranges:
        label = normalize_label(row.get("text") or "")
        if not label:
            continue
        nvtx_phases.append(
            {
                "label": ALIASES.get(label, label),
                "family": "cuda",
                "source": "nvtx",
                "start": int(row["start"]),
                "end": int(row["end"]),
                "duration": int(row["end"]) - int(row["start"]),
            }
        )
    if nvtx_phases:
        # NVTX ranges carry the native nsys clock, so they win for any label
        # they cover — but the prover-internal stages are stderr-only today,
        # so stderr phases with uncovered labels must survive the merge.
        nvtx_labels = {phase["label"] for phase in nvtx_phases}
        phases = nvtx_phases + [phase for phase in phases if phase["label"] not in nvtx_labels]

    for phase in phases:
        phase["chain"] = "gpu"

    phases.sort(key=lambda phase: (phase["start"], phase["end"], phase["label"]))
    return phases, {
        "stderr_timer_lines": parsed_total,
        "stderr_stamped_lines": stamped,
        "stderr_stamp_coverage": stamped / parsed_total if parsed_total else None,
        "nvtx_ranges": len(nvtx_phases),
        "cpu_phases": 0,
        "gpu_phases": sum(1 for phase in phases if phase["chain"] == "gpu"),
    }


def normalize_label(label):
    label = label.strip()
    if not label:
        return ""
    label = re.sub(r"^(?:nvtx|stage|proofscope|proof-server)::", "", label)
    label = label.replace("proof-server.", "proof_server.")
    label = label.replace("proof-server:", "proof_server:")
    label = label.replace("/", ".")
    label = re.sub(r"\s+", "_", label)
    return label


def select_input_bundle(bundle_dir):
    requested = Path(bundle_dir).resolve()
    if not requested.exists():
        raise FileNotFoundError(f"bundle directory does not exist: {requested}")

    direct = [
        requested / "nsys" / "report.sqlite",
        requested / "report.sqlite",
        requested / "proofscope.sqlite",
        requested / "nsys.sqlite",
    ]
    for sqlite_path in direct:
        if sqlite_path.exists():
            return InputBundle(
                requested_dir=requested,
                analysis_dir=requested / "analysis",
                run_dir=requested,
                sqlite_path=sqlite_path,
                stdout_path=_existing(requested / "stdout.txt"),
                stderr_path=_existing(requested / "stderr.txt"),
                selection="direct sqlite",
            )

    run_candidates = []
    for path in requested.glob("run-*"):
        for sqlite_name in (
            "nsys/report.sqlite",
            "report.sqlite",
            "proofscope.sqlite",
            "nsys.sqlite",
        ):
            sqlite_path = path / sqlite_name
            if sqlite_path.exists():
                run_candidates.append((path, sqlite_path))
                break
    if not run_candidates:
        raise FileNotFoundError(
            f"no nsys/report.sqlite, report.sqlite, proofscope.sqlite, or run-* bundle under {requested}"
        )

    runs = [path for path, _ in run_candidates]
    ranked = sorted((_run_online_cuda_ms(run), run) for run in runs)
    if all(value is None for value, _ in ranked):
        chosen = runs[len(runs) // 2]
        selection = "middle run by name"
    else:
        ranked = sorted((value if value is not None else float("inf"), run) for value, run in ranked)
        chosen = ranked[len(ranked) // 2][1]
        selection = "median run by stdout gpu_ms"

    return InputBundle(
        requested_dir=requested,
        analysis_dir=requested / "analysis",
        run_dir=chosen,
        sqlite_path=dict(run_candidates)[chosen],
        stdout_path=_existing(chosen / "stdout.txt"),
        stderr_path=_existing(chosen / "stderr.txt"),
        selection=selection,
    )


def _existing(path):
    return path if path.exists() else None


def _run_online_cuda_ms(run_dir):
    stdout = run_dir / "stdout.txt"
    if not stdout.exists():
        return None
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ONLINE_RE.search(line)
        if match:
            return float(match.group(1))
    return None
