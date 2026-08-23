#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
gpuprof_py="${repo_root}/scripts/gpuprof/gpuprof.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/gpuprof.sh analyze <bundle-dir>
  scripts/gpuprof.sh parity [--backend cpu|cuda|metal|metal-hybrid|cpu-vs-cuda|cpu-vs-metal|cpu-vs-metal-hybrid] [--fixture zswap-transaction|minimal-v2|hash-chain-v2|zswap-output|zswap-spend] [--repeat N] [--metal-quotient-diagnostic]
  scripts/gpuprof.sh cuda-adapter
  scripts/gpuprof.sh run proof-backend-parity [options]
  scripts/gpuprof.sh doctor
  scripts/gpuprof.sh nsys <gate> [options]
  scripts/gpuprof.sh metal-trace <gate> [options]
  scripts/gpuprof.sh diff <baseline-bundle> <candidate-bundle>
  scripts/gpuprof.sh check <baseline-bundle> <candidate-bundle> [--tolerance F]
  scripts/gpuprof.sh trend [--last N]
  scripts/gpuprof.sh metadata
  scripts/gpuprof.sh quick <gate> [options]
  scripts/gpuprof.sh ncu <gate> --top N [options]
  scripts/gpuprof.sh sanitize <gate> --tool TOOL [options]
  scripts/gpuprof.sh --all <gate> [options]

Current status:
  analyze  Offline proof-server stage analysis from an existing Nsight SQLite
           bundle: reconciled decomposition, kernel lint, transfer totals,
           levers, terminal report.
  parity   Deterministic CPU/GPU proof backend timing and byte-exactness gate.
           Defaults to the main zswap-transaction gate: wallet and fee
           balancing, transaction proving/sealing, and byte-exact comparison.
           Other fixtures are retained only as focused diagnostics.
           cpu-vs-* modes are separate alternating runs, not hybrid runtimes;
           legacy both, cpu-metal, and cpu-metal-hybrid aliases remain accepted.
           --metal-quotient-diagnostic is a trusted-local, five-proof shadow
           gate; it does not substitute Metal output into the proof.
  nsys     Collect a proof-backend-parity Nsight bundle: builds with
           perf-timers, profiles, exports SQLite, analyzes, appends history.
  metal-trace  Collect a proof-backend-parity Apple Metal System Trace bundle
           with xctrace, parity JSON, and host/toolchain provenance.
  diff     Exploratory per-stage delta between two analyzed bundles.
  check    Exit-coded regression gate with structural causes on every FAIL.
  trend    Collection history from benchmark-results/gpuprof-history.jsonl.
  metadata Environment/provenance snapshot (git, GPU, MIDNIGHT_GPU_* env).
  cuda-adapter  Build/run the cuda-oxide proof-server-cuda adapter smoke gate.
  doctor   CUDA/NVIDIA tool availability probe.

  quick/ncu/sanitize/all remain reserved to match the Nightstream workflow.
EOF
}

tool_path() {
  local tool="$1"
  if command -v "${tool}" >/dev/null 2>&1; then
    command -v "${tool}"
    return 0
  fi
  for prefix in /usr/local/cuda-13.0 /usr/local/cuda /opt/cuda; do
    if [[ -x "${prefix}/bin/${tool}" ]]; then
      printf '%s\n' "${prefix}/bin/${tool}"
      return 0
    fi
  done
  return 1
}

doctor() {
  local missing=0
  for tool in nsys ncu compute-sanitizer nvcc; do
    if path="$(tool_path "${tool}")"; then
      printf '%-18s %s\n' "${tool}" "${path}"
      "${path}" --version 2>&1 | sed 's/^/  /' | head -n 2 || true
    else
      printf '%-18s missing\n' "${tool}"
      missing=1
    fi
  done
  return "${missing}"
}

collection_not_ready() {
  local mode="$1"
  local gate="${2:-}"
  {
    printf 'error: gpuprof %s collection is not wired for midnight-ledger yet\n' "${mode}"
    if [[ -n "${gate}" ]]; then
      printf 'requested gate: %s\n' "${gate}"
    fi
    printf 'next steps: emit stable CUDA stage ranges for this mode, then route it through the matching ncu/sanitizer/baseline workflow.\n'
  } >&2
  return 2
}

cuda_adapter() {
  (
    cd "${repo_root}/proof-server-cuda"
    cargo +nightly-2026-04-03 oxide build --features cuda-oxide
  )
  exec "${repo_root}/target/release/proof-cuda-adapter"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

case "$1" in
  -h|--help|help)
    usage
    ;;
  analyze)
    shift
    exec python3 "${gpuprof_py}" analyze "$@"
    ;;
  parity)
    shift
    exec python3 "${gpuprof_py}" run proof-backend-parity "$@"
    ;;
  cuda-adapter|adapter)
    shift
    cuda_adapter "$@"
    ;;
  run)
    shift
    exec python3 "${gpuprof_py}" run "$@"
    ;;
  nsys)
    shift
    exec python3 "${gpuprof_py}" nsys "$@"
    ;;
  metal-trace|xctrace)
    shift
    exec python3 "${gpuprof_py}" metal-trace "$@"
    ;;
  doctor)
    doctor
    ;;
  --all|all|quick|ncu|sanitize|cpu)
    mode="$1"
    shift
    collection_not_ready "${mode}" "${1:-}"
    ;;
  diff|check|trend|metadata)
    exec python3 "${gpuprof_py}" "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
