// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

#define BLS12_381_METAL_MSM_DIAGNOSTICS 1

#include "product_issue.metal"
#include "u32_field.metal"
#include "bls12_381_amns8.metal"
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
