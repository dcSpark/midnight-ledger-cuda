"""Terminal rendering of a proofscope analysis. The JSON stays authoritative;
this is the courtesy layer so an engineer never has to read raw stages.json."""


def render_summary(report, stage_limit=18, lint_limit=10, lever_limit=5):
    recon = report.get("reconciliation", {})
    counts = report.get("counts", {})
    provenance = report.get("provenance") or {}

    print()
    print("PROOFSCOPE ANALYSIS")
    print("===================")
    source = report.get("source", {})
    print(f"bundle: {source.get('run_dir', '?')}")
    if provenance:
        bits = []
        for key in ("gpu_name", "driver", "git_head", "backend", "fixture", "hash_rounds"):
            value = provenance.get(key)
            if value:
                bits.append(f"{key}={value}")
        if bits:
            print("run:    " + " ".join(str(b) for b in bits))
    ok = "ok" if recon.get("ok") else "FAILED"
    print(
        f"reconciliation: {ok} "
        f"(max_error={recon.get('max_abs_error_ms', 0.0):.6f}ms, "
        f"clamp={recon.get('idle_overlap_clamp_ms', 0.0):.6f}ms, "
        f"unknown_labels={len(report.get('unknown_nvtx', []))})"
    )

    totals = report.get("totals") or {}
    if totals:
        print(
            f"device: kernels {totals.get('kernel_busy_ms', 0.0):.1f}ms/"
            f"{totals.get('launches', 0)} launches, "
            f"h2d {totals.get('h2d_mb', 0.0):.1f}MB/{totals.get('h2d_copies', 0)}, "
            f"d2h {totals.get('d2h_mb', 0.0):.1f}MB/{totals.get('d2h_copies', 0)}, "
            f"syncs {totals.get('syncs', 0)}"
        )
    print()

    render_stage_table(report.get("stages", []), stage_limit)
    render_lint(report.get("lint", []), lint_limit)
    render_levers(report.get("levers_detail") or [], lever_limit)


def render_stage_table(stages, limit):
    # Exclusive decomposition: each time slice is charged to the innermost
    # active stage, so rows never double-count and rollup labels only show
    # their own uncovered remainder. `scoped` is the window wall for context.
    rows = sorted(stages, key=lambda r: -(r.get("wall_ms") or 0.0))
    rows = [row for row in rows if (row.get("wall_ms") or 0.0) >= 1.0][:limit]
    print("STAGES (exclusive ownership; ms)")
    print("--------------------------------")
    if not rows:
        print("no stages above 1ms of exclusive time")
        print()
        return
    print(
        f"{'stage':<52} {'excl':>8} {'busy':>7} {'xfer':>6} {'sync':>6} "
        f"{'host':>7} {'unatt':>6} {'scoped':>8} {'h2d MB':>7} {'d2h MB':>7} {'launch':>6}"
    )
    for row in rows:
        print(
            f"{row['stage_id']:<52} "
            f"{row.get('wall_ms', 0.0):>8.1f} "
            f"{row.get('gpu_busy_ms', 0.0):>7.1f} "
            f"{row.get('transfer_wait_ms', 0.0):>6.1f} "
            f"{row.get('sync_wait_ms', 0.0):>6.1f} "
            f"{row.get('host_gap_ms', 0.0):>7.1f} "
            f"{row.get('unattributed_ms', 0.0):>6.1f} "
            f"{row.get('window_wall_ms', 0.0):>8.1f} "
            f"{row.get('h2d_mb', 0.0):>7.1f} "
            f"{row.get('d2h_mb', 0.0):>7.1f} "
            f"{row.get('launches', 0):>6d}"
        )
    print()


def render_lint(findings, limit):
    print("KERNEL LINT (static launch-config checks)")
    print("-----------------------------------------")
    if not findings:
        print("clean: no spills, occupancy caps, or underfilled grids detected")
        print()
        return
    for f in findings[:limit]:
        print(
            f"[{f['severity']:>4}] {f['rule']:<24} {f['kernel']:<40} "
            f"{f['total_ms']:>8.1f}ms/{f['launches']:<5} {f['detail']}"
        )
    if len(findings) > limit:
        print(f"... {len(findings) - limit} more findings (see 'lint' in stages.json)")
    print()


def render_levers(levers, limit):
    print("TOP LEVERS (recoverable ms vs physics floor)")
    print("--------------------------------------------")
    if not levers:
        print("no levers above threshold")
        print()
        return
    for lever in levers[:limit]:
        print(
            f"{lever['stage_id']:<52} recoverable={lever['recoverable_ms']:>8.1f}ms "
            f"({lever['recoverable_pct_of_stage']:.0f}% of stage) "
            f"fix={lever['fix_class']} cause={lever['cause']}"
        )
    print()
