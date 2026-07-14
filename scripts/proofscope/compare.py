"""Diff and regression-gate two proofscope stage analyses.

Ported from Nightstream's gpuprof decision layer. `diff` is exploratory;
`check` is the exit-coded gate. Every flagged regression is explained with
structural causes (launch counts, copy counts/MB, per-kernel ms deltas) so a
FAIL is never just a red number.
"""

import json
from pathlib import Path

try:
    from .lint import short_kernel_name
except ImportError:  # Direct script execution.
    from lint import short_kernel_name

DIFF_FIELDS = (
    "window_wall_ms",
    "gpu_busy_ms",
    "transfer_wait_ms",
    "sync_wait_ms",
    "host_gap_ms",
    "unattributed_ms",
    "h2d_mb",
    "d2h_mb",
    "launches",
)


def load_stage_map(path):
    """Accept a stages.json path, an analysis dir, or a bundle dir."""
    p = Path(path)
    if p.is_dir():
        for candidate in (p / "analysis" / "stages.json", p / "stages.json"):
            if candidate.exists():
                p = candidate
                break
        else:
            raise FileNotFoundError(f"no stages.json under {path}")
    report = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(report, dict):
        if "stages" not in report:
            raise ValueError(
                f"{p} is not a proofscope stage analysis; expected stages.json or an analyzed bundle"
            )
        stages = report["stages"]
    else:
        stages = report
    return {row["stage_id"]: row for row in stages}, report if isinstance(report, dict) else {}


def structural_causes(a, b):
    """Explain a stage's wall change via structure: counts, sizes, kernels."""
    causes = []

    def count_delta(key):
        av, bv = a.get(key, 0) or 0, b.get(key, 0) or 0
        if abs(bv - av) > 0.5:
            causes.append(f"{key} {av:.0f} -> {bv:.0f}")

    def mb_delta(key):
        av, bv = a.get(key, 0.0) or 0.0, b.get(key, 0.0) or 0.0
        if abs(bv - av) > 1.0:
            causes.append(f"{key} {av:.1f} -> {bv:.1f}")

    for key in ("launches", "h2d_copies", "d2h_copies", "dtod_copies", "memset_count", "syncs"):
        count_delta(key)
    for key in ("h2d_mb", "d2h_mb", "dtod_mb", "memset_mb"):
        mb_delta(key)

    def avg_kernel_us(row):
        launches = row.get("launches", 0) or 0
        return 1000.0 * (row.get("kernel_attributed_ms", 0.0) or 0.0) / launches if launches else 0.0

    if (a.get("launches") or b.get("launches")) and abs(avg_kernel_us(b) - avg_kernel_us(a)) > 20.0:
        causes.append(f"avg kernel {avg_kernel_us(a):.0f}us -> {avg_kernel_us(b):.0f}us")

    ak = a.get("kernels") or {}
    bk = b.get("kernels") or {}
    changed = sorted(set(ak) | set(bk), key=lambda n: -abs(bk.get(n, 0.0) - ak.get(n, 0.0)))
    for name in changed[:4]:
        av, bv = ak.get(name, 0.0), bk.get(name, 0.0)
        if abs(bv - av) > 0.5:
            tag = " (new)" if av == 0.0 else (" (gone)" if bv == 0.0 else "")
            causes.append(f"kernel {short_kernel_name(name)} {av:.1f} -> {bv:.1f}ms{tag}")
    return causes


def diff_rows(base_map, cand_map, min_stage_ms=1.0):
    """Per-stage deltas sorted by absolute wall movement, with causes."""
    rows = []
    for stage_id in sorted(set(base_map) | set(cand_map)):
        a = base_map.get(stage_id, {})
        b = cand_map.get(stage_id, {})
        wall_a = a.get("window_wall_ms", 0.0) or 0.0
        wall_b = b.get("window_wall_ms", 0.0) or 0.0
        if max(wall_a, wall_b) < min_stage_ms:
            continue
        row = {"stage_id": stage_id, "delta_ms": round(wall_b - wall_a, 3)}
        for field in DIFF_FIELDS:
            row[field] = (round(a.get(field, 0.0) or 0.0, 3), round(b.get(field, 0.0) or 0.0, 3))
        row["causes"] = structural_causes(a, b) if abs(row["delta_ms"]) >= 1.0 else []
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["delta_ms"]))
    return rows


def check_stages(base_map, cand_map, tolerance=0.05, abs_floor_ms=2.0, min_stage_ms=5.0):
    """Regression gate. Returns (checked, failures) where each failure is a
    dict with the stage, walls, limit, and structural causes."""
    checked = 0
    failures = []
    for stage_id in sorted(set(base_map) & set(cand_map)):
        a, b = base_map[stage_id], cand_map[stage_id]
        wall_a = a.get("window_wall_ms", 0.0) or 0.0
        wall_b = b.get("window_wall_ms", 0.0) or 0.0
        if wall_a < min_stage_ms:
            continue
        checked += 1
        limit = wall_a * (1.0 + tolerance) + abs_floor_ms
        if wall_b > limit:
            failures.append(
                {
                    "stage_id": stage_id,
                    "baseline_ms": round(wall_a, 3),
                    "candidate_ms": round(wall_b, 3),
                    "limit_ms": round(limit, 3),
                    "causes": structural_causes(a, b),
                }
            )
    return checked, failures


def print_diff(rows, limit=20):
    print("STAGE DIFF (candidate - baseline, by |delta window wall|)")
    print("---------------------------------------------------------")
    if not rows:
        print("no stages above threshold in either run")
        return
    header = f"{'stage':<58} {'base':>9} {'cand':>9} {'delta':>8}"
    print(header)
    for row in rows[:limit]:
        base_ms, cand_ms = row["window_wall_ms"]
        print(f"{row['stage_id']:<58} {base_ms:>9.1f} {cand_ms:>9.1f} {row['delta_ms']:>+8.1f}")
        for cause in row["causes"][:4]:
            print(f"{'':<58} {cause}")
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more stages")


def print_check(checked, failures):
    for failure in failures:
        print(
            f"[FAIL] {failure['stage_id']} window_wall_ms "
            f"{failure['baseline_ms']:.1f} -> {failure['candidate_ms']:.1f} "
            f"(limit {failure['limit_ms']:.1f})"
        )
        for cause in failure["causes"]:
            print(f"       {cause}")
    verdict = "REGRESSED" if failures else "clean"
    print(f"check: {checked} comparisons, {len(failures)} regressions - {verdict}")
