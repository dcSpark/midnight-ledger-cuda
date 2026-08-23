// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

// Production library root. Research and feasibility kernels remain in
// root.metal and are compiled only by the explicit metal-diagnostics feature.
#include "msm_pipeline.metal"
#include "msm_parallel_bucket_v1.metal"
#include "msm_parallel_jacobian_v2.metal"
#include "msm_gpu_route_dense_v3.metal"
#include "msm_gpu_route_xyzz_v4.metal"
#include "msm_gpu_route_xyzz_v5.metal"
#include "msm_gpu_route_xyzz_v6.metal"
#include "msm_gpu_route_xyzz_v7.metal"
#include "msm_gpu_route_xyzz_v8.metal"
#include "msm_batch_affine.metal"
#include "msm_gpu_route_batch_affine_v9.metal"
#include "msm_fixed_base_v10.metal"
