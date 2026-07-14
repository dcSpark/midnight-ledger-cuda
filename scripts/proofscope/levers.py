"""Rank optimization levers from analyzed proof-server stage facts."""

LAUNCH_OVERHEAD_US = 4.0
DEFAULT_BANDWIDTH_MB_PER_MS = 20.0
LEVER_ROOTS = ("proof_server", "zkir", "zswap", "ledger")

FIX_CLASS_FIELDS = (
    ("host_gap_ms", "cpu_pipeline"),
    ("sync_wait_ms", "sync_fence"),
    ("api_ms", "launch_overhead"),
    ("transfer_wait_ms", "transfer_elimination"),
    ("unattributed_ms", "unknown"),
)


def build_levers(stages, timeline=None):
    stage_map = {row["stage_id"]: row for row in stages}
    leaves = leaf_stage_ids(stage_map)
    bandwidth = measured_bandwidths(timeline or {})
    levers = []
    for stage_id in leaves:
        if not stage_id.startswith(LEVER_ROOTS):
            continue
        stage = stage_map[stage_id]
        # Rank on the exclusive decomposition (innermost-stage ownership):
        # scoped windows double-count nested rollups when the label hierarchy
        # has parallel roots, which yields the same lever five times.
        wall_ms = stage.get("wall_ms", 0.0) or stage.get("window_wall_ms", 0.0)
        if wall_ms < 1.0:
            continue
        transfer_floor_ms = (
            stage.get("h2d_mb", 0.0) / bandwidth["h2d_mb_per_ms"]
            + stage.get("d2h_mb", 0.0) / bandwidth["d2h_mb_per_ms"]
            + stage.get("dtod_mb", 0.0) / bandwidth["dtod_mb_per_ms"]
        )
        launch_floor_ms = stage.get("launches", 0) * LAUNCH_OVERHEAD_US / 1000.0
        busy_ms = stage.get("gpu_busy_ms", 0.0)
        floor_ms = busy_ms + transfer_floor_ms + launch_floor_ms
        recoverable_ms = max(0.0, wall_ms - floor_ms)
        cause, cause_ms, fix_class = dominant_cause(stage)
        levers.append(
            {
                "stage_id": stage_id,
                "fix_class": fix_class,
                "cause": cause,
                "cause_ms": rounded(cause_ms),
                "window_wall_ms": rounded(stage.get("window_wall_ms", 0.0)),
                "exclusive_wall_ms": rounded(wall_ms),
                "gpu_busy_ms": rounded(busy_ms),
                "transfer_floor_ms": rounded(transfer_floor_ms),
                "launch_floor_ms": rounded(launch_floor_ms),
                "floor_ms": rounded(floor_ms),
                "recoverable_ms": rounded(recoverable_ms),
                "recoverable_pct_of_stage": rounded(100.0 * recoverable_ms / wall_ms),
                "launches": int(stage.get("launches", 0)),
                "top_kernels": stage.get("top_kernels", []),
                "buckets": cause_buckets(stage),
            }
        )
    levers.sort(key=lambda row: (-row["recoverable_ms"], -row["exclusive_wall_ms"], row["stage_id"]))
    return {
        "schema_version": 2,
        "model": {
            "wall_field": "wall_ms (exclusive ownership)",
            "floor": "gpu_busy_ms + transfer_floor_ms + launch_floor_ms",
            "launch_overhead_us": LAUNCH_OVERHEAD_US,
            "bandwidth": bandwidth,
        },
        "levers": levers,
    }


def leaf_stage_ids(stage_map):
    out = []
    for stage_id in stage_map:
        prefix = stage_id + "."
        if not any(other.startswith(prefix) for other in stage_map):
            out.append(stage_id)
    return sorted(out)


def measured_bandwidths(timeline):
    memcpys = timeline.get("memcpys") or []
    return {
        "h2d_mb_per_ms": measured_bandwidth_mb_per_ms(memcpys, 1),
        "d2h_mb_per_ms": measured_bandwidth_mb_per_ms(memcpys, 2),
        "dtod_mb_per_ms": measured_bandwidth_mb_per_ms(memcpys, 3),
    }


def measured_bandwidth_mb_per_ms(memcpys, copy_kind):
    rates = sorted(
        copy["bytes"] / max(copy["end"] - copy["start"], 1)
        for copy in memcpys
        if copy.get("copy_kind") == copy_kind and copy.get("bytes", 0) >= (1 << 20)
    )
    if not rates:
        return DEFAULT_BANDWIDTH_MB_PER_MS
    return rates[max(0, int(len(rates) * 0.9) - 1)]


def dominant_cause(stage):
    buckets = cause_buckets(stage)
    candidates = [(buckets[field], field, fix_class) for field, fix_class in FIX_CLASS_FIELDS]
    value, field, fix_class = max(candidates, key=lambda item: (item[0], item[1]))
    return field, value, fix_class


def cause_buckets(stage):
    return {
        "host_gap_ms": rounded(stage.get("host_gap_ms", stage.get("window_host_gap_ms", 0.0))),
        "sync_wait_ms": rounded(stage.get("sync_wait_ms", stage.get("window_sync_wait_ms", 0.0))),
        "api_ms": rounded(stage.get("api_ms", stage.get("window_api_ms", 0.0))),
        "transfer_wait_ms": rounded(stage.get("transfer_wait_ms", stage.get("window_transfer_wait_ms", 0.0))),
        "unattributed_ms": rounded(stage.get("unattributed_ms", stage.get("window_unattributed_ms", 0.0))),
    }


def rounded(value):
    return round(float(value), 6)
