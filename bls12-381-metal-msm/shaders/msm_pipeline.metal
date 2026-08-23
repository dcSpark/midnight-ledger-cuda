// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Correctness-first standalone MSM pipeline. CPU routing supplies grouped
// signed records and dense bucket-to-run maps; every G1 operation, bucket
// integration, window collection, and output conversion executes on Metal.

#ifndef BLS12_381_METAL_MSM_PIPELINE
#define BLS12_381_METAL_MSM_PIPELINE

#include "msm_g1.metal"

// This is the private GPU ABI, not the compact public router record. The host
// expands the public high-bit sign encoding into this explicit flags word;
// v1 does not claim zero-copy routing.
struct Bls12381MetalMsmSignedRecord {
    uint point_index;
    uint flags;
};

struct Bls12381MetalMsmBucketRun {
    uint record_offset;
    uint record_count;
    uint reserved_0;
    uint reserved_1;
};

struct Bls12381MetalMsmReduceParameters {
    uint base_count;
    uint record_count;
    uint run_count;
    uint reserved;
};

struct Bls12381MetalMsmWindowDescriptor {
    uint dense_map_offset;
    uint bucket_count;
    uint window_point_index;
    uint reserved;
};

struct Bls12381MetalMsmIntegrationParameters {
    uint dense_map_count;
    uint run_count;
    uint window_count;
    uint window_point_count;
};

struct Bls12381MetalMsmCollectDescriptor {
    uint window_point_offset;
    uint window_count;
    uint window_bits;
    uint output_index;
};

struct Bls12381MetalMsmCollectParameters {
    uint window_point_count;
    uint descriptor_count;
    uint output_count;
    uint reserved;
};

struct Bls12381MetalMsmStatus {
    atomic_uint flags;
    atomic_uint record_errors;
    atomic_uint descriptor_errors;
    atomic_uint reserved;
};

static_assert(sizeof(Bls12381MetalMsmFp) == 48, "Fp ABI");
static_assert(sizeof(Bls12381MetalMsmAffine) == 96, "affine ABI");
static_assert(sizeof(Bls12381MetalMsmHomogeneous) == 144, "homogeneous ABI");
static_assert(sizeof(Bls12381MetalMsmJacobian) == 144, "Jacobian ABI");
static_assert(sizeof(Bls12381MetalMsmSignedRecord) == 8, "record ABI");
static_assert(sizeof(Bls12381MetalMsmBucketRun) == 16, "run ABI");
static_assert(sizeof(Bls12381MetalMsmReduceParameters) == 16, "reduce ABI");
static_assert(sizeof(Bls12381MetalMsmWindowDescriptor) == 16, "window ABI");
static_assert(sizeof(Bls12381MetalMsmIntegrationParameters) == 16, "integration ABI");
static_assert(sizeof(Bls12381MetalMsmCollectDescriptor) == 16, "collect ABI");
static_assert(sizeof(Bls12381MetalMsmCollectParameters) == 16, "collect parameters ABI");
static_assert(sizeof(Bls12381MetalMsmStatus) == 16, "status ABI");

constant uint BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE = 1U;
constant uint BLS12_381_METAL_MSM_DENSE_MAP_IDENTITY = 0xffffffffU;

constant uint BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS = 1U << 0;
constant uint BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE = 1U << 1;
constant uint BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE = 1U << 2;
constant uint BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS = 1U << 3;
constant uint BLS12_381_METAL_MSM_STATUS_INVALID_RUN = 1U << 4;
constant uint BLS12_381_METAL_MSM_STATUS_INVALID_DENSE_MAP = 1U << 5;
constant uint BLS12_381_METAL_MSM_STATUS_INVALID_WINDOW = 1U << 6;
constant uint BLS12_381_METAL_MSM_STATUS_INVALID_COLLECT = 1U << 7;
constant uint BLS12_381_METAL_MSM_STATUS_KNOWN_MASK = 0xffU;

inline __attribute__((always_inline))
void bls12_381_msm_status_record_error(
    device Bls12381MetalMsmStatus* status,
    uint flag)
{
    atomic_fetch_or_explicit(&status->flags, flag, memory_order_relaxed);
    atomic_fetch_add_explicit(&status->record_errors, 1U, memory_order_relaxed);
}

inline __attribute__((always_inline))
void bls12_381_msm_status_descriptor_error(
    device Bls12381MetalMsmStatus* status,
    uint flag)
{
    atomic_fetch_or_explicit(&status->flags, flag, memory_order_relaxed);
    atomic_fetch_add_explicit(&status->descriptor_errors, 1U, memory_order_relaxed);
}

inline __attribute__((always_inline))
bool bls12_381_msm_range_valid(uint offset, uint count, uint total)
{
    return offset <= total && count <= total - offset;
}

// Dispatch one thread per run. Runs must be nonempty, pairwise disjoint spans
// into records, and already grouped by one logical (MSM, window, bucket).
kernel void bls12_381_metal_msm_bucket_reduce_serial(
    device const Bls12381MetalMsmAffine* bases [[buffer(0)]],
    device const Bls12381MetalMsmSignedRecord* records [[buffer(1)]],
    device const Bls12381MetalMsmBucketRun* runs [[buffer(2)]],
    device Bls12381MetalMsmHomogeneous* run_roots [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmReduceParameters& parameters [[buffer(5)]],
    uint run_index [[thread_position_in_grid]])
{
    if (run_index >= parameters.run_count) {
        return;
    }

    const Bls12381MetalMsmBucketRun run = runs[run_index];
    Bls12381MetalMsmHomogeneous accumulator = bls12_381_msm_g1_identity();
    run_roots[run_index] = accumulator;
    const bool reserved_zero = (run.reserved_0 | run.reserved_1) == 0U;
    if (parameters.reserved != 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        return;
    }
    if (!reserved_zero || run.record_count == 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_RUN);
        return;
    }
    if (!bls12_381_msm_range_valid(
            run.record_offset, run.record_count, parameters.record_count)) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_RECORD_OUT_OF_RANGE);
        return;
    }

    for (uint local_index = 0; local_index < run.record_count; ++local_index) {
        const Bls12381MetalMsmSignedRecord record =
            records[run.record_offset + local_index];
        if ((record.flags & ~BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE) != 0U) {
            bls12_381_msm_status_record_error(
                status, BLS12_381_METAL_MSM_STATUS_INVALID_RECORD_FLAGS);
            continue;
        }
        if (record.point_index >= parameters.base_count) {
            bls12_381_msm_status_record_error(
                status, BLS12_381_METAL_MSM_STATUS_POINT_OUT_OF_RANGE);
            continue;
        }
        const Bls12381MetalMsmAffine point = bls12_381_msm_g1_signed_affine(
            bases[record.point_index], record.flags);
        accumulator = bls12_381_msm_g1_mixed_add(accumulator, point);
    }
    run_roots[run_index] = accumulator;
}

// Dispatch one thread per window descriptor. Each dense-map segment stores
// bucket magnitudes 1..bucket_count in ascending order. The map value is a run
// index or DENSE_MAP_IDENTITY for an empty bucket.
kernel void bls12_381_metal_msm_integrate_windows_serial(
    device const Bls12381MetalMsmHomogeneous* run_roots [[buffer(0)]],
    device const uint* dense_run_map [[buffer(1)]],
    device const Bls12381MetalMsmWindowDescriptor* windows [[buffer(2)]],
    device Bls12381MetalMsmHomogeneous* window_points [[buffer(3)]],
    device Bls12381MetalMsmStatus* status [[buffer(4)]],
    constant Bls12381MetalMsmIntegrationParameters& parameters [[buffer(5)]],
    uint window_index [[thread_position_in_grid]])
{
    if (window_index >= parameters.window_count) {
        return;
    }
    const Bls12381MetalMsmWindowDescriptor descriptor = windows[window_index];
    const bool span_valid = descriptor.bucket_count != 0U &&
        bls12_381_msm_range_valid(
            descriptor.dense_map_offset,
            descriptor.bucket_count,
            parameters.dense_map_count);
    const bool output_valid =
        descriptor.window_point_index < parameters.window_point_count;
    if (output_valid) {
        window_points[descriptor.window_point_index] = bls12_381_msm_g1_identity();
    }
    if (!span_valid || !output_valid || descriptor.reserved != 0U) {
        bls12_381_msm_status_descriptor_error(
            status, BLS12_381_METAL_MSM_STATUS_INVALID_WINDOW);
        return;
    }

    Bls12381MetalMsmHomogeneous running = bls12_381_msm_g1_identity();
    Bls12381MetalMsmHomogeneous weighted = bls12_381_msm_g1_identity();
    for (uint remaining = descriptor.bucket_count; remaining != 0U; --remaining) {
        const uint map_index = descriptor.dense_map_offset + remaining - 1U;
        const uint run_index = dense_run_map[map_index];
        Bls12381MetalMsmHomogeneous bucket = bls12_381_msm_g1_identity();
        if (run_index != BLS12_381_METAL_MSM_DENSE_MAP_IDENTITY) {
            if (run_index >= parameters.run_count) {
                bls12_381_msm_status_descriptor_error(
                    status, BLS12_381_METAL_MSM_STATUS_INVALID_DENSE_MAP);
            } else {
                bucket = run_roots[run_index];
            }
        }
        running = bls12_381_msm_g1_add(running, bucket);
        weighted = bls12_381_msm_g1_add(weighted, running);
    }
    window_points[descriptor.window_point_index] = weighted;
}

// Dispatch one thread per MSM descriptor. A referenced window-point span must
// be contiguous and ordered low-to-high by scalar significance.
kernel void bls12_381_metal_msm_collect_serial(
    device const Bls12381MetalMsmHomogeneous* window_points [[buffer(0)]],
    device const Bls12381MetalMsmCollectDescriptor* descriptors [[buffer(1)]],
    device Bls12381MetalMsmJacobian* outputs [[buffer(2)]],
    device Bls12381MetalMsmStatus* status [[buffer(3)]],
    constant Bls12381MetalMsmCollectParameters& parameters [[buffer(4)]],
    uint descriptor_index [[thread_position_in_grid]])
{
    if (descriptor_index >= parameters.descriptor_count) {
        return;
    }
    const Bls12381MetalMsmCollectDescriptor descriptor =
        descriptors[descriptor_index];
    const bool span_valid = descriptor.window_count != 0U &&
        bls12_381_msm_range_valid(
            descriptor.window_point_offset,
            descriptor.window_count,
            parameters.window_point_count);
    const bool bits_valid = descriptor.window_bits != 0U &&
        descriptor.window_bits <= 16U;
    const bool output_valid = descriptor.output_index < parameters.output_count;
    if (output_valid) {
        outputs[descriptor.output_index] =
            bls12_381_msm_g1_to_jacobian(bls12_381_msm_g1_identity());
    }
    if (!span_valid || !bits_valid || !output_valid || parameters.reserved != 0U) {
        bls12_381_msm_status_descriptor_error(
            status,
            parameters.reserved == 0U
                ? BLS12_381_METAL_MSM_STATUS_INVALID_COLLECT
                : BLS12_381_METAL_MSM_STATUS_INVALID_PARAMETERS);
        return;
    }

    Bls12381MetalMsmHomogeneous accumulator = bls12_381_msm_g1_identity();
    for (uint remaining = descriptor.window_count; remaining != 0U; --remaining) {
        if (remaining != descriptor.window_count) {
            for (uint bit = 0; bit < descriptor.window_bits; ++bit) {
                accumulator = bls12_381_msm_g1_double(accumulator);
            }
        }
        accumulator = bls12_381_msm_g1_add(
            accumulator,
            window_points[descriptor.window_point_offset + remaining - 1U]);
    }
    outputs[descriptor.output_index] = bls12_381_msm_g1_to_jacobian(accumulator);
}

// Dispatch one thread per 32-bit word. The host must wait for completion
// before releasing or reusing the allocation.
kernel void bls12_381_metal_msm_clear_words(
    device uint* words [[buffer(0)]],
    constant uint& word_count [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < word_count) {
        words[index] = 0U;
    }
}

inline __attribute__((always_inline))
ulong bls12_381_msm_abi_mix(ulong fingerprint, ulong value)
{
    return (fingerprint ^ value) * 0x00000100000001b3UL;
}

// One thread writes 68 ulong words. Words 0..63 describe the versioned POD
// layout, word 64 is their FNV-1a fingerprint, and words 65..67 pin sentinels.
kernel void bls12_381_metal_msm_msm_abi(
    device ulong* output [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]])
{
    if (thread_id != 0U) {
        return;
    }
    output[0] = BLS12_381_METAL_MSM_ABI_VERSION;
    output[1] = BLS12_381_METAL_MSM_CONSTANTS_FINGERPRINT;
    output[2] = sizeof(Bls12381MetalMsmFp);
    output[3] = alignof(Bls12381MetalMsmFp);
    output[4] = sizeof(Bls12381MetalMsmAffine);
    output[5] = alignof(Bls12381MetalMsmAffine);
    output[6] = __builtin_offsetof(Bls12381MetalMsmAffine, x);
    output[7] = __builtin_offsetof(Bls12381MetalMsmAffine, y);
    output[8] = sizeof(Bls12381MetalMsmHomogeneous);
    output[9] = alignof(Bls12381MetalMsmHomogeneous);
    output[10] = __builtin_offsetof(Bls12381MetalMsmHomogeneous, x);
    output[11] = __builtin_offsetof(Bls12381MetalMsmHomogeneous, y);
    output[12] = __builtin_offsetof(Bls12381MetalMsmHomogeneous, z);
    output[13] = sizeof(Bls12381MetalMsmJacobian);
    output[14] = alignof(Bls12381MetalMsmJacobian);
    output[15] = __builtin_offsetof(Bls12381MetalMsmJacobian, x);
    output[16] = __builtin_offsetof(Bls12381MetalMsmJacobian, y);
    output[17] = __builtin_offsetof(Bls12381MetalMsmJacobian, z);
    output[18] = sizeof(Bls12381MetalMsmSignedRecord);
    output[19] = alignof(Bls12381MetalMsmSignedRecord);
    output[20] = __builtin_offsetof(Bls12381MetalMsmSignedRecord, point_index);
    output[21] = __builtin_offsetof(Bls12381MetalMsmSignedRecord, flags);
    output[22] = sizeof(Bls12381MetalMsmBucketRun);
    output[23] = alignof(Bls12381MetalMsmBucketRun);
    output[24] = __builtin_offsetof(Bls12381MetalMsmBucketRun, record_offset);
    output[25] = __builtin_offsetof(Bls12381MetalMsmBucketRun, record_count);
    output[26] = __builtin_offsetof(Bls12381MetalMsmBucketRun, reserved_0);
    output[27] = __builtin_offsetof(Bls12381MetalMsmBucketRun, reserved_1);
    output[28] = sizeof(Bls12381MetalMsmReduceParameters);
    output[29] = alignof(Bls12381MetalMsmReduceParameters);
    output[30] = __builtin_offsetof(Bls12381MetalMsmReduceParameters, base_count);
    output[31] = __builtin_offsetof(Bls12381MetalMsmReduceParameters, record_count);
    output[32] = __builtin_offsetof(Bls12381MetalMsmReduceParameters, run_count);
    output[33] = __builtin_offsetof(Bls12381MetalMsmReduceParameters, reserved);
    output[34] = sizeof(Bls12381MetalMsmWindowDescriptor);
    output[35] = alignof(Bls12381MetalMsmWindowDescriptor);
    output[36] = __builtin_offsetof(Bls12381MetalMsmWindowDescriptor, dense_map_offset);
    output[37] = __builtin_offsetof(Bls12381MetalMsmWindowDescriptor, bucket_count);
    output[38] = __builtin_offsetof(Bls12381MetalMsmWindowDescriptor, window_point_index);
    output[39] = __builtin_offsetof(Bls12381MetalMsmWindowDescriptor, reserved);
    output[40] = sizeof(Bls12381MetalMsmIntegrationParameters);
    output[41] = alignof(Bls12381MetalMsmIntegrationParameters);
    output[42] = __builtin_offsetof(Bls12381MetalMsmIntegrationParameters, dense_map_count);
    output[43] = __builtin_offsetof(Bls12381MetalMsmIntegrationParameters, run_count);
    output[44] = __builtin_offsetof(Bls12381MetalMsmIntegrationParameters, window_count);
    output[45] = __builtin_offsetof(Bls12381MetalMsmIntegrationParameters, window_point_count);
    output[46] = sizeof(Bls12381MetalMsmCollectDescriptor);
    output[47] = alignof(Bls12381MetalMsmCollectDescriptor);
    output[48] = __builtin_offsetof(Bls12381MetalMsmCollectDescriptor, window_point_offset);
    output[49] = __builtin_offsetof(Bls12381MetalMsmCollectDescriptor, window_count);
    output[50] = __builtin_offsetof(Bls12381MetalMsmCollectDescriptor, window_bits);
    output[51] = __builtin_offsetof(Bls12381MetalMsmCollectDescriptor, output_index);
    output[52] = sizeof(Bls12381MetalMsmCollectParameters);
    output[53] = alignof(Bls12381MetalMsmCollectParameters);
    output[54] = __builtin_offsetof(Bls12381MetalMsmCollectParameters, window_point_count);
    output[55] = __builtin_offsetof(Bls12381MetalMsmCollectParameters, descriptor_count);
    output[56] = __builtin_offsetof(Bls12381MetalMsmCollectParameters, output_count);
    output[57] = __builtin_offsetof(Bls12381MetalMsmCollectParameters, reserved);
    output[58] = sizeof(Bls12381MetalMsmStatus);
    output[59] = alignof(Bls12381MetalMsmStatus);
    output[60] = __builtin_offsetof(Bls12381MetalMsmStatus, flags);
    output[61] = __builtin_offsetof(Bls12381MetalMsmStatus, record_errors);
    output[62] = __builtin_offsetof(Bls12381MetalMsmStatus, descriptor_errors);
    output[63] = __builtin_offsetof(Bls12381MetalMsmStatus, reserved);

    ulong layout_fingerprint = 0xcbf29ce484222325UL;
    for (uint index = 0; index < 64U; ++index) {
        layout_fingerprint = bls12_381_msm_abi_mix(layout_fingerprint, output[index]);
    }
    output[64] = layout_fingerprint;
    output[65] = BLS12_381_METAL_MSM_DENSE_MAP_IDENTITY;
    output[66] = BLS12_381_METAL_MSM_SIGNED_RECORD_NEGATE;
    output[67] = BLS12_381_METAL_MSM_STATUS_KNOWN_MASK;
}

#endif
