"""Static kernel lint from launch configs alone — no NCU counters needed.

Ported from Nightstream's gpuprof analysis layer. Every finding is computed
from data the nsys capture already collects (grid/block, registers, local and
shared memory per launch), so this works even when GPU counter permissions
(ERR_NVGPUCTRPERM) block Nsight Compute.
"""

import re

# RTX 4090 (sm_89) static limits for theoretical-occupancy math. If profiling
# moves to a different device, add its table and select on the metadata GPU
# name; the lint findings state which table was used.
SM89 = {
    "name": "sm_89",
    "sms": 128,
    "warp": 32,
    "threads_per_sm": 1536,
    "regs_per_sm": 65536,
    "smem_per_sm": 102400,
    "blocks_per_sm": 24,
}

MIN_LINT_MS = 0.5


def short_kernel_name(name):
    name = re.sub(r"<.*>", "", name or "")
    name = name.split("(")[0]
    return name.rsplit("::", 1)[-1][:48]


def kernel_configs(kernels):
    """Aggregate launches by (name, grid, block, regs, local, smem)."""
    agg = {}
    for k in kernels:
        grid = tuple(v or 1 for v in (k.get("grid") or [1, 1, 1]))
        block = tuple(v or 1 for v in (k.get("block") or [1, 1, 1]))
        key = (
            k["name"],
            grid,
            block,
            k.get("registers_per_thread") or 0,
            k.get("local_memory_per_thread") or 0,
            (k.get("static_shared_memory") or 0) + (k.get("dynamic_shared_memory") or 0),
        )
        a = agg.setdefault(key, {"count": 0, "ms": 0.0})
        a["count"] += 1
        a["ms"] += (k["end"] - k["start"]) / 1e6
    return agg


def theoretical_occupancy(threads, regs, smem, limits=SM89):
    if threads <= 0:
        return 1.0, "none"
    bounds = {
        "threads": limits["threads_per_sm"] // threads,
        "blocks": limits["blocks_per_sm"],
    }
    if regs:
        bounds["registers"] = limits["regs_per_sm"] // max(regs * threads, 1)
    if smem:
        bounds["shared_memory"] = limits["smem_per_sm"] // smem
    limiter = min(bounds, key=bounds.get)
    occ = min(1.0, max(bounds[limiter], 0) * threads / limits["threads_per_sm"])
    return occ, (limiter if occ < 1.0 else "none")


def lint_kernels(kernels, limits=SM89):
    """Static red flags: spills, non-warp blocks, occupancy caps, underfill."""
    findings = []
    for key, a in sorted(kernel_configs(kernels).items(), key=lambda kv: -kv[1]["ms"]):
        name, grid, block, regs, local_mem, smem = key
        threads = block[0] * block[1] * block[2]
        blocks = grid[0] * grid[1] * grid[2]
        base = {
            "kernel": short_kernel_name(name),
            "kernel_full": name,
            "grid": list(grid),
            "block": list(block),
            "registers": regs,
            "launches": a["count"],
            "total_ms": round(a["ms"], 3),
            "limits": limits["name"],
        }
        if local_mem > 0:
            findings.append(
                {
                    **base,
                    "severity": "high",
                    "rule": "register-spill",
                    "detail": f"{local_mem} B/thread spilled to local memory",
                }
            )
        if threads % limits["warp"] != 0:
            findings.append(
                {
                    **base,
                    "severity": "warn",
                    "rule": "block-not-warp-multiple",
                    "detail": f"block of {threads} threads is not a multiple of {limits['warp']}",
                }
            )
        occ, limiter = theoretical_occupancy(threads, regs, smem, limits)
        if occ < 0.5 and a["ms"] >= MIN_LINT_MS:
            findings.append(
                {
                    **base,
                    "severity": "warn",
                    "rule": "low-occupancy",
                    "detail": f"theoretical occupancy {occ:.0%}, limited by {limiter}",
                }
            )
        if a["ms"] >= MIN_LINT_MS and blocks * threads < limits["sms"] * limits["threads_per_sm"] // 2:
            findings.append(
                {
                    **base,
                    "severity": "warn",
                    "rule": "grid-underfill",
                    "detail": f"{blocks} blocks x {threads} threads cannot fill {limits['sms']} SMs",
                }
            )
    return findings


def kernel_stats(kernels, limit=20):
    """Per-kernel-name aggregates for the analysis JSON and report table."""
    agg = {}
    for k in kernels:
        row = agg.setdefault(
            k["name"],
            {
                "kernel": short_kernel_name(k["name"]),
                "kernel_full": k["name"],
                "launches": 0,
                "total_ms": 0.0,
                "max_registers": 0,
                "max_local_mem_per_thread": 0,
            },
        )
        row["launches"] += 1
        row["total_ms"] += (k["end"] - k["start"]) / 1e6
        row["max_registers"] = max(row["max_registers"], k.get("registers_per_thread") or 0)
        row["max_local_mem_per_thread"] = max(
            row["max_local_mem_per_thread"], k.get("local_memory_per_thread") or 0
        )
    rows = sorted(agg.values(), key=lambda r: -r["total_ms"])[:limit]
    for row in rows:
        row["total_ms"] = round(row["total_ms"], 3)
        row["mean_us"] = round(1000.0 * row["total_ms"] / row["launches"], 1) if row["launches"] else 0.0
    return rows
