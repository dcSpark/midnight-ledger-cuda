#!/bin/sh
# Copyright 2026 Midnight Foundation
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=${1:-${TMPDIR:-/tmp}/bls12-381-metal-msm-metal-mulhi-probe}

mkdir -p "$build_dir"

xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=13.0 \
  -fno-fast-math -O3 \
  -c "$script_dir/constant_rhs.metal" \
  -o "$build_dir/constant_rhs.air"
xcrun -sdk macosx metallib \
  "$build_dir/constant_rhs.air" \
  -o "$build_dir/constant_rhs.metallib"
swiftc -O "$script_dir/constant_rhs.swift" -framework Metal \
  -o "$build_dir/constant-rhs-mulhi-probe"

xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=13.0 \
  -fno-fast-math -O3 \
  -c "$script_dir/rotating_two_state.metal" \
  -o "$build_dir/rotating_two_state.air"
xcrun -sdk macosx metallib \
  "$build_dir/rotating_two_state.air" \
  -o "$build_dir/rotating_two_state.metallib"
swiftc -O "$script_dir/rotating_two_state.swift" -framework Metal \
  -o "$build_dir/rotating-two-state-mulhi-probe"

"$build_dir/constant-rhs-mulhi-probe" \
  "$build_dir/constant_rhs.metallib"
"$build_dir/rotating-two-state-mulhi-probe" \
  "$build_dir/rotating_two_state.metallib"

shasum -a 256 \
  "$script_dir/constant_rhs.metal" \
  "$script_dir/constant_rhs.swift" \
  "$build_dir/constant_rhs.air" \
  "$build_dir/constant_rhs.metallib" \
  "$script_dir/rotating_two_state.metal" \
  "$script_dir/rotating_two_state.swift" \
  "$build_dir/rotating_two_state.air" \
  "$build_dir/rotating_two_state.metallib"
