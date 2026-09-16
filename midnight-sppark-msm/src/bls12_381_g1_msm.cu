#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/std/tuple>
#include <gmp.h>
#include <cgbn/cgbn.h>

#define FEATURE_BLS12_381
#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

using field_t = bls12_381::fp_t;
using scalar_t = bls12_381::fr_t;
using affine_t = Affine_t<field_t>;
using bucket_t = xyzz_t<field_t>;
using point_t = jacobian_t<field_t>;

// batch_addition.cuh keeps one device work cursor per stream id. Proof lanes
// use sid = proof_lane * 32 + local_lane * 2, so sppark's default eight slots
// alias every concurrent proof onto the same cursor. Reserve the complete lane
// namespace before including pippenger.cuh, which instantiates that array.
#define BATCH_ADD_NSTREAMS 288
#include <msm/pippenger.cuh>
#include <ntt/ntt.cuh>
#include <polynomial/prefix_op.cuh>

using msm_engine_t = msm_t<bucket_t, point_t, affine_t, scalar_t>;

static constexpr uint32_t MAX_PROOF_EXECUTION_LANES = 8;
static constexpr uint32_t MSM_EXECUTION_SID_STRIDE = 32;
static thread_local uint32_t proof_execution_lane = 0;

struct RawMsmConfig {
    uint32_t domain_k;
    uint32_t min_msm_scalars;
    uint32_t unfiltered_min_scalars;
    uint32_t window_bits;
    uint32_t sort_grid_divisor;
    uint32_t sort_window_lanes;
    uint32_t integration_blocks;
    uint32_t batch_lanes;
    uint32_t reuse_event;
    uint32_t device_reduce;
};

struct RawMsmSlice {
    size_t scalar_offset;
    size_t point_offset;
    size_t len;
    size_t nonzero_len;
};

static RawMsmConfig default_msm_config(size_t npoints)
{
    uint32_t domain_k = 0;
    size_t capacity = npoints > 1 ? npoints - 1 : 0;
    while (capacity != 0) {
        ++domain_k;
        capacity >>= 1;
    }
    uint32_t window_bits = domain_k == 0 ? 0 : 10;
    if (domain_k >= 16 && domain_k <= 18)
        window_bits = 12;
    if (domain_k > 18)
        window_bits = 0;
    uint32_t device_reduce = domain_k >= 14 && domain_k <= 18 ? 1 : 0;
    uint32_t integration_blocks = 1;
    return RawMsmConfig{
        domain_k, 100000, 32768, window_bits, 3, 2,
        integration_blocks, 4, 0, device_reduce};
}

static bool valid_msm_config(const RawMsmConfig& config)
{
    return config.domain_k <= 31
        && config.min_msm_scalars >= 1
        && config.unfiltered_min_scalars >= 1
        && (config.window_bits == 0
            || (config.window_bits >= 10 && config.window_bits <= 18))
        && config.sort_grid_divisor >= 1
        && config.sort_grid_divisor <= 8
        && config.sort_window_lanes >= 1
        && config.sort_window_lanes <= 4
        && (config.integration_blocks == 1
            || config.integration_blocks == 2
            || config.integration_blocks == 4
            || config.integration_blocks == 8
            || config.integration_blocks == 16)
        && config.batch_lanes >= 1
        && config.batch_lanes <= 16
        && config.reuse_event <= 1
        && config.device_reduce <= 3;
}

static uint32_t active_execution_lane()
{
    return proof_execution_lane;
}

static const gpu_t& active_gpu()
{
    if (proof_execution_lane == 0)
        return select_gpu( -1 );

    static std::array<std::unique_ptr<gpu_t>, MAX_PROOF_EXECUTION_LANES + 1> lanes;
    static std::mutex lanes_mutex;
    std::lock_guard<std::mutex> lock(lanes_mutex);
    auto& lane = lanes[proof_execution_lane];
    if (!lane) {
        const gpu_t& shared = select_gpu( -1 );
        shared.select();
        lane = std::make_unique<gpu_t>(shared.id(), shared.cid(), shared.props());
    }
    return *lane;
}

extern "C" RustError midnight_sppark_execution_lane_set(uint32_t lane)
{
    if (lane > MAX_PROOF_EXECUTION_LANES)
        return RustError{cudaErrorInvalidValue, "invalid sppark proof execution lane"};
    try {
        proof_execution_lane = lane;
        active_gpu().select();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        proof_execution_lane = 0;
        return RustError{e.code(), e.what()};
    }
}

static std::mutex& msm_dispatch_mutex()
{
    static std::mutex mutex;
    return mutex;
}

class msm_context_t {
    uint32_t execution_lane;
    RawMsmConfig config;
    size_t point_count;
    const gpu_t *shared_gpu;
    std::vector<std::unique_ptr<gpu_t>> private_gpus;
    std::vector<std::unique_ptr<msm_engine_t>> engines;

    static size_t configured_lanes(const RawMsmConfig& config)
    {
        return static_cast<size_t>(config.batch_lanes);
    }

    const gpu_t& lane_gpu(size_t lane) const
    {
        return lane == 0 ? *shared_gpu : *private_gpus[lane - 1];
    }

    template<class Work>
    RustError invoke_parallel(size_t nmsms, Work work, const event_t *ready = nullptr)
    {
        if (nmsms == 0)
            return RustError{cudaSuccess};

        const size_t workers = std::min(nmsms, engines.size());
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::vector<RustError> statuses(workers);
        auto run = [&](size_t lane) {
            lane_gpu(lane).select();
            if (ready != nullptr)
                lane_gpu(lane)[2].wait(*ready);
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= nmsms)
                    break;
                RustError status = work(*engines[lane], index);
                if (status.code != 0) {
                    statuses[lane] = status;
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        };

        if (workers == 1) {
            run(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (size_t lane = 0; lane < workers; ++lane)
                threads.emplace_back(run, lane);
            for (auto& thread : threads)
                thread.join();
        }

        for (const auto& status : statuses) {
            if (status.code != 0)
                return status;
        }
        return RustError{cudaSuccess};
    }

public:
    msm_context_t(const affine_t *points, size_t npoints, RawMsmConfig config_)
        : execution_lane(active_execution_lane()), config(config_),
          point_count(npoints),
          shared_gpu(&active_gpu())
    {
        shared_gpu->select();
        const size_t lanes = std::min(
            configured_lanes(config),
            static_cast<size_t>(MSM_NSTREAMS / 2));
        engines.reserve(lanes);
        private_gpus.reserve(lanes - 1);
        engines.emplace_back(std::make_unique<msm_engine_t>(
            points,
            npoints,
            sizeof(affine_t),
            -1,
            shared_gpu,
            execution_lane * MSM_EXECUTION_SID_STRIDE,
            config.window_bits,
            config.sort_grid_divisor,
            config.sort_window_lanes,
            config.integration_blocks,
            config.reuse_event,
            config.device_reduce));
        for (size_t lane = 1; lane < lanes; ++lane) {
            private_gpus.emplace_back(std::make_unique<gpu_t>(
                shared_gpu->id(),
                shared_gpu->cid(),
                shared_gpu->props()));
            engines.emplace_back(std::make_unique<msm_engine_t>(
                points,
                npoints,
                sizeof(affine_t),
                -1,
                private_gpus.back().get(),
                execution_lane * MSM_EXECUTION_SID_STRIDE
                    + static_cast<uint32_t>(lane * 2),
                config.window_bits,
                config.sort_grid_divisor,
                config.sort_window_lanes,
                config.integration_blocks,
                config.reuse_event,
                config.device_reduce));
        }
    }

    RustError invoke_host_batch(
        point_t *out,
        const scalar_t *scalars,
        size_t npoints,
        size_t nmsms)
    {
        return invoke_parallel(nmsms, [&](msm_engine_t& engine, size_t index) {
            return engine.invoke(
                out[index],
                static_cast<const affine_t *>(nullptr),
                npoints,
                scalars + (index * npoints),
                true);
        });
    }

    RustError invoke_device_batch(
        point_t *out,
        const scalar_t *scalars,
        size_t npoints,
        size_t nmsms,
        const event_t *ready = nullptr)
    {
        return invoke_parallel(nmsms, [&](msm_engine_t& engine, size_t index) {
            dev_ptr_t<scalar_t> batch(
                const_cast<scalar_t *>(scalars + (index * npoints)),
                npoints);
            return engine.invoke(
                out[index],
                static_cast<const affine_t *>(nullptr),
                npoints,
                batch,
                true);
        }, ready);
    }

    RustError invoke_device_ragged(
        point_t *out,
        const scalar_t *scalars,
        size_t scalar_count,
        const RawMsmSlice *slices,
        size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            const auto& slice = slices[i];
            if (slice.point_offset > point_count
                || slice.len > point_count - slice.point_offset
                || slice.scalar_offset > scalar_count
                || slice.len > scalar_count - slice.scalar_offset
                || slice.nonzero_len > slice.len) {
                return RustError{cudaErrorInvalidValue,
                                 "ragged resident MSM slice exceeds its source"};
            }
        }
        return invoke_parallel(count, [&](msm_engine_t& engine, size_t index) {
            const auto& slice = slices[index];
            return engine.invoke_sparse_device(
                out[index],
                slice.point_offset,
                scalars + slice.scalar_offset,
                slice.len,
                slice.nonzero_len);
        });
    }
};

class NTTAccess : public NTT {
public:
    using NTT::NTT_internal;
};

struct RawFr {
    uint64_t limbs[4];
};

struct RawG1Affine {
    uint64_t x[6];
    uint64_t y[6];
};

struct RawG1Projective {
    uint64_t x[6];
    uint64_t y[6];
    uint64_t z[6];
};

struct RawValueSource {
    uint32_t kind;
    uint32_t a;
    uint32_t b;
};

struct RawCalculation {
    uint32_t op;
    uint32_t target;
    uint32_t parts_offset;
    uint32_t parts_len;
    RawValueSource lhs;
    RawValueSource rhs;
};

struct RawGraph {
    const RawFr *constants;
    size_t constants_len;
    const int32_t *rotations;
    size_t rotations_len;
    const RawCalculation *calculations;
    size_t calculations_len;
    const RawValueSource *horner_parts;
    size_t horner_parts_len;
    uint32_t num_intermediates;
};

struct RawColumnRef {
    uint32_t kind;
    uint32_t index;
};

struct RawDevicePolyRef {
    uint32_t source;
    uint32_t index;
};

static_assert(sizeof(RawFr) == sizeof(scalar_t),
              "RawFr must match sppark BLS12-381 scalar layout");
static_assert(sizeof(RawG1Affine) == sizeof(affine_t),
              "RawG1Affine must match sppark BLS12-381 affine layout");
static_assert(sizeof(RawG1Projective) == sizeof(point_t),
              "RawG1Projective must match sppark BLS12-381 projective layout");

static scalar_t load_scalar(const RawFr& raw)
{
    scalar_t scalar;
    auto *dst = reinterpret_cast<unsigned char *>(&scalar);
    const auto *src = reinterpret_cast<const unsigned char *>(&raw);
    for (size_t i = 0; i < sizeof(scalar); ++i) {
        dst[i] = src[i];
    }
    return scalar;
}

__host__ __device__ static scalar_t fr_zero()
{
    scalar_t out;
    out.zero();
    return out;
}

__device__ static size_t rotated_index(size_t idx, int32_t rot, int32_t rot_scale, size_t len)
{
    int64_t shifted = static_cast<int64_t>(idx) + (static_cast<int64_t>(rot) * rot_scale);
    int64_t modulus = static_cast<int64_t>(len);
    shifted %= modulus;
    if (shifted < 0) {
        shifted += modulus;
    }
    return static_cast<size_t>(shifted);
}

__device__ static const scalar_t *select_column_base(
    uint32_t kind,
    uint32_t index,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    size_t len)
{
    if (kind == 0 && index < fixed_cols) {
        return fixed + (static_cast<size_t>(index) * len);
    }
    if (kind == 1 && index < advice_cols) {
        return advice + (static_cast<size_t>(index) * len);
    }
    if (kind == 2 && index < instance_cols) {
        return instance + (static_cast<size_t>(index) * len);
    }
    return nullptr;
}

__device__ static scalar_t source_value(
    RawValueSource source,
    size_t idx,
    size_t len,
    int32_t rot_scale,
    const RawGraph& graph,
    const scalar_t *intermediates,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    scalar_t previous_value)
{
    switch (source.kind) {
    case 0:
        if (source.a < graph.constants_len) {
            return reinterpret_cast<const scalar_t *>(graph.constants)[source.a];
        }
        return fr_zero();
    case 1:
        return intermediates[source.a];
    case 2:
    case 3:
    case 4: {
        uint32_t column_kind = source.kind - 2;
        const scalar_t *base = select_column_base(
            column_kind,
            source.a,
            fixed,
            fixed_cols,
            advice,
            advice_cols,
            instance,
            instance_cols,
            len);
        if (base == nullptr || source.b >= graph.rotations_len) {
            return fr_zero();
        }
        return base[rotated_index(idx, graph.rotations[source.b], rot_scale, len)];
    }
    case 5:
        if (source.a < challenge_len) {
            return challenges[source.a];
        }
        return fr_zero();
    case 6:
        return beta;
    case 7:
        return gamma;
    case 8:
        return theta;
    case 9:
        return trash_challenge;
    case 10:
        return y;
    case 11:
        return previous_value;
    default:
        return fr_zero();
    }
}

template <size_t MAX_INTERMEDIATES>
__device__ static scalar_t evaluate_graph(
    const RawGraph& graph,
    size_t idx,
    size_t len,
    int32_t rot_scale,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    scalar_t previous_value)
{
    scalar_t intermediates[MAX_INTERMEDIATES];
    for (uint32_t i = 0; i < graph.num_intermediates && i < MAX_INTERMEDIATES; ++i) {
        intermediates[i].zero();
    }

    scalar_t last;
    last.zero();
    for (size_t calc_idx = 0; calc_idx < graph.calculations_len; ++calc_idx) {
        RawCalculation calc = graph.calculations[calc_idx];
        scalar_t value;
        value.zero();

        auto get = [&](RawValueSource source) {
            return source_value(
                source,
                idx,
                len,
                rot_scale,
                graph,
                intermediates,
                fixed,
                fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                challenges,
                challenge_len,
                beta,
                gamma,
                theta,
                trash_challenge,
                y,
                previous_value);
        };

        switch (calc.op) {
        case 0:
            value = get(calc.lhs) + get(calc.rhs);
            break;
        case 1:
            value = get(calc.lhs) - get(calc.rhs);
            break;
        case 2:
            value = get(calc.lhs) * get(calc.rhs);
            break;
        case 3:
            value = get(calc.lhs);
            value = value * value;
            break;
        case 4:
            value = get(calc.lhs);
            value += value;
            break;
        case 5:
            value = -get(calc.lhs);
            break;
        case 6: {
            scalar_t factor = get(calc.rhs);
            value = get(calc.lhs);
            for (uint32_t part_idx = 0; part_idx < calc.parts_len; ++part_idx) {
                value = value * factor + get(graph.horner_parts[calc.parts_offset + part_idx]);
            }
            break;
        }
        case 7:
            value = get(calc.lhs);
            break;
        default:
            value.zero();
            break;
        }

        if (calc.target < MAX_INTERMEDIATES) {
            intermediates[calc.target] = value;
        }
        last = value;
    }
    return last;
}

template <size_t MAX_INTERMEDIATES>
__global__ void numerator_custom_kernel(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = evaluate_graph<MAX_INTERMEDIATES>(
            graph,
            idx,
            len,
            rot_scale,
            fixed,
            fixed_cols,
            advice,
            advice_cols,
            instance,
            instance_cols,
            challenges,
            challenge_len,
            beta,
            gamma,
            theta,
            trash_challenge,
            y,
            fr_zero());
    }
}

__global__ void numerator_permutation_kernel(
    scalar_t *out,
    size_t len,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *l0,
    const scalar_t *l_last,
    const scalar_t *l_active,
    const scalar_t *permutation_products,
    size_t permutation_sets,
    const RawColumnRef *columns,
    size_t column_count,
    const scalar_t *permutation_pk_cosets,
    scalar_t beta,
    scalar_t gamma,
    scalar_t y,
    const scalar_t *omega_powers,
    scalar_t delta_start,
    scalar_t delta_const,
    int32_t rot_scale,
    int32_t last_rotation,
    uint32_t chunk_len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len || permutation_sets == 0 || chunk_len == 0) {
        return;
    }

    size_t r_next = rotated_index(idx, 1, rot_scale, len);
    size_t r_last = rotated_index(idx, last_rotation, rot_scale, len);
    scalar_t value = out[idx];
    const scalar_t one = scalar_t::one();
    const scalar_t *first_product = permutation_products;
    const scalar_t *last_product = permutation_products + ((permutation_sets - 1) * len);

    value = value * y + ((one - first_product[idx]) * l0[idx]);
    value = value * y + ((last_product[idx] * last_product[idx] - last_product[idx]) * l_last[idx]);

    for (size_t set_idx = 1; set_idx < permutation_sets; ++set_idx) {
        const scalar_t *current = permutation_products + (set_idx * len);
        const scalar_t *previous = permutation_products + ((set_idx - 1) * len);
        value = value * y + ((current[idx] - previous[r_last]) * l0[idx]);
    }

    scalar_t current_delta = delta_start * omega_powers[idx];

    for (size_t set_idx = 0; set_idx < permutation_sets; ++set_idx) {
        const scalar_t *product = permutation_products + (set_idx * len);
        scalar_t left = product[r_next];
        size_t base = set_idx * chunk_len;
        for (uint32_t j = 0; j < chunk_len && base + j < column_count; ++j) {
            const scalar_t *column = select_column_base(
                columns[base + j].kind,
                columns[base + j].index,
                fixed,
                fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                len);
            if (column != nullptr) {
                const scalar_t *permutation = permutation_pk_cosets + ((base + j) * len);
                left *= column[idx] + beta * permutation[idx] + gamma;
            }
        }

        scalar_t right = product[idx];
        for (uint32_t j = 0; j < chunk_len && base + j < column_count; ++j) {
            const scalar_t *column = select_column_base(
                columns[base + j].kind,
                columns[base + j].index,
                fixed,
                fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                len);
            if (column != nullptr) {
                right *= column[idx] + current_delta + gamma;
                current_delta *= delta_const;
            }
        }

        value = value * y + ((left - right) * l_active[idx]);
    }

    out[idx] = value;
}

template <size_t MAX_INTERMEDIATES>
__global__ void numerator_lookup_kernel(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    const scalar_t *l0,
    const scalar_t *l_last,
    const scalar_t *l_active,
    const scalar_t *product,
    const scalar_t *permuted_input,
    const scalar_t *permuted_table,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    size_t r_next = rotated_index(idx, 1, rot_scale, len);
    size_t r_prev = rotated_index(idx, -1, rot_scale, len);
    const scalar_t one = scalar_t::one();
    scalar_t value = out[idx];
    scalar_t table_value = evaluate_graph<MAX_INTERMEDIATES>(
        graph,
        idx,
        len,
        rot_scale,
        fixed,
        fixed_cols,
        advice,
        advice_cols,
        instance,
        instance_cols,
        challenges,
        challenge_len,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
        fr_zero());

    scalar_t a_minus_s = permuted_input[idx] - permuted_table[idx];
    value = value * y + ((one - product[idx]) * l0[idx]);
    value = value * y + ((product[idx] * product[idx] - product[idx]) * l_last[idx]);
    value = value * y
        + ((product[r_next] * (permuted_input[idx] + beta) * (permuted_table[idx] + gamma)
            - product[idx] * table_value)
            * l_active[idx]);
    value = value * y + (a_minus_s * l0[idx]);
    value = value * y
        + (a_minus_s * (permuted_input[idx] - permuted_input[r_prev]) * l_active[idx]);
    out[idx] = value;
}

template <size_t MAX_INTERMEDIATES>
__global__ void numerator_trash_kernel(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    const scalar_t *trash_poly,
    uint32_t selector_fixed_index,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len || selector_fixed_index >= fixed_cols) {
        return;
    }
    scalar_t value = out[idx];
    scalar_t compressed = evaluate_graph<MAX_INTERMEDIATES>(
        graph,
        idx,
        len,
        rot_scale,
        fixed,
        fixed_cols,
        advice,
        advice_cols,
        instance,
        instance_cols,
        challenges,
        challenge_len,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
        fr_zero());
    const scalar_t one = scalar_t::one();
    scalar_t q = fixed[(static_cast<size_t>(selector_fixed_index) * len) + idx];
    out[idx] = value * y + (compressed - (one - q) * trash_poly[idx]);
}

__global__ void omega_power_factors_kernel(
    scalar_t *values,
    scalar_t omega,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] = idx == 0 ? scalar_t::one() : omega;
    }
}

static uint32_t intermediate_capacity(uint32_t num_intermediates)
{
    if (num_intermediates <= 8) {
        return 8;
    }
    if (num_intermediates <= 16) {
        return 16;
    }
    if (num_intermediates <= 32) {
        return 32;
    }
    if (num_intermediates <= 64) {
        return 64;
    }
    if (num_intermediates <= 128) {
        return 128;
    }
    return 256;
}

static void launch_numerator_custom(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale,
    size_t grid,
    size_t block,
    const gpu_t& gpu)
{
#define LAUNCH_CUSTOM(MAX) \
    numerator_custom_kernel<MAX><<<grid, block, 0, gpu>>>( \
        out, len, graph, fixed, fixed_cols, advice, advice_cols, instance, instance_cols, \
        challenges, challenge_len, beta, gamma, theta, trash_challenge, y, rot_scale)

    switch (intermediate_capacity(graph.num_intermediates)) {
    case 8:
        LAUNCH_CUSTOM(8);
        break;
    case 16:
        LAUNCH_CUSTOM(16);
        break;
    case 32:
        LAUNCH_CUSTOM(32);
        break;
    case 64:
        LAUNCH_CUSTOM(64);
        break;
    case 128:
        LAUNCH_CUSTOM(128);
        break;
    default:
        LAUNCH_CUSTOM(256);
        break;
    }
#undef LAUNCH_CUSTOM
    CUDA_OK(cudaGetLastError());
}

static void launch_numerator_lookup(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    const scalar_t *l0,
    const scalar_t *l_last,
    const scalar_t *l_active,
    const scalar_t *product,
    const scalar_t *permuted_input,
    const scalar_t *permuted_table,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale,
    size_t grid,
    size_t block,
    const gpu_t& gpu)
{
#define LAUNCH_LOOKUP(MAX) \
    numerator_lookup_kernel<MAX><<<grid, block, 0, gpu>>>( \
        out, len, graph, fixed, fixed_cols, advice, advice_cols, instance, instance_cols, \
        challenges, challenge_len, l0, l_last, l_active, product, permuted_input, \
        permuted_table, beta, gamma, theta, trash_challenge, y, rot_scale)

    switch (intermediate_capacity(graph.num_intermediates)) {
    case 8:
        LAUNCH_LOOKUP(8);
        break;
    case 16:
        LAUNCH_LOOKUP(16);
        break;
    case 32:
        LAUNCH_LOOKUP(32);
        break;
    case 64:
        LAUNCH_LOOKUP(64);
        break;
    case 128:
        LAUNCH_LOOKUP(128);
        break;
    default:
        LAUNCH_LOOKUP(256);
        break;
    }
#undef LAUNCH_LOOKUP
    CUDA_OK(cudaGetLastError());
}

static void launch_numerator_trash(
    scalar_t *out,
    size_t len,
    RawGraph graph,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const scalar_t *challenges,
    size_t challenge_len,
    const scalar_t *trash_poly,
    uint32_t selector_fixed_index,
    scalar_t beta,
    scalar_t gamma,
    scalar_t theta,
    scalar_t trash_challenge,
    scalar_t y,
    int32_t rot_scale,
    size_t grid,
    size_t block,
    const gpu_t& gpu)
{
#define LAUNCH_TRASH(MAX) \
    numerator_trash_kernel<MAX><<<grid, block, 0, gpu>>>( \
        out, len, graph, fixed, fixed_cols, advice, advice_cols, instance, instance_cols, \
        challenges, challenge_len, trash_poly, selector_fixed_index, beta, gamma, theta, \
        trash_challenge, y, rot_scale)

    switch (intermediate_capacity(graph.num_intermediates)) {
    case 8:
        LAUNCH_TRASH(8);
        break;
    case 16:
        LAUNCH_TRASH(16);
        break;
    case 32:
        LAUNCH_TRASH(32);
        break;
    case 64:
        LAUNCH_TRASH(64);
        break;
    case 128:
        LAUNCH_TRASH(128);
        break;
    default:
        LAUNCH_TRASH(256);
        break;
    }
#undef LAUNCH_TRASH
    CUDA_OK(cudaGetLastError());
}

template <typename T>
static std::unique_ptr<dev_ptr_t<T>> upload_array(const T *src, size_t len, const gpu_t& gpu)
{
    if (len == 0) {
        return nullptr;
    }
    auto dst = std::make_unique<dev_ptr_t<T>>(len, gpu);
    gpu.HtoD(&(*dst)[0], src, len);
    return dst;
}

static std::unique_ptr<dev_ptr_t<scalar_t>> upload_scalars(
    const RawFr *src,
    size_t len,
    const gpu_t& gpu)
{
    return upload_array(reinterpret_cast<const scalar_t *>(src), len, gpu);
}

template <typename T>
static T *device_ptr(std::unique_ptr<dev_ptr_t<T>>& ptr)
{
    return ptr ? &(*ptr)[0] : nullptr;
}

template <typename T>
static const T *device_ptr(const std::unique_ptr<dev_ptr_t<T>>& ptr)
{
    return ptr ? &(*ptr)[0] : nullptr;
}

struct UploadedGraph {
    std::unique_ptr<dev_ptr_t<RawFr>> constants;
    std::unique_ptr<dev_ptr_t<int32_t>> rotations;
    std::unique_ptr<dev_ptr_t<RawCalculation>> calculations;
    std::unique_ptr<dev_ptr_t<RawValueSource>> horner_parts;
    RawGraph graph;
};

struct scalar_device_vec_t {
    uint32_t execution_lane = active_execution_lane();
    size_t len;
    std::unique_ptr<dev_ptr_t<scalar_t>> values;
};

struct numerator_context_t {
    uint32_t execution_lane = active_execution_lane();
    size_t len;
    size_t input_len;
    size_t fixed_cols;
    size_t permutation_columns_len;
    size_t permutation_pk_cosets_len;
    size_t lookup_count;
    size_t trash_count;
    int32_t rot_scale;
    int32_t last_rotation;
    uint32_t chunk_len;
    scalar_t omega;
    scalar_t g_coset;
    scalar_t delta_const;
    std::unique_ptr<dev_ptr_t<scalar_t>> fixed;
    std::unique_ptr<dev_ptr_t<scalar_t>> fixed_lagrange;
    std::unique_ptr<dev_ptr_t<scalar_t>> l0;
    std::unique_ptr<dev_ptr_t<scalar_t>> l_last;
    std::unique_ptr<dev_ptr_t<scalar_t>> l_active;
    std::unique_ptr<dev_ptr_t<RawColumnRef>> permutation_columns;
    std::unique_ptr<dev_ptr_t<scalar_t>> permutation_pk_cosets;
    std::unique_ptr<dev_ptr_t<scalar_t>> permutation_pk_lagrange;
    std::vector<uint32_t> trash_selector_fixed_indices;
    std::unique_ptr<dev_ptr_t<scalar_t>> omega_powers;
    std::unique_ptr<dev_ptr_t<scalar_t>> trace_omega_powers;
    UploadedGraph custom_graph;
    std::vector<UploadedGraph> lookup_graphs;
    std::vector<UploadedGraph> lookup_input_graphs;
    std::vector<UploadedGraph> lookup_table_graphs;
    std::vector<UploadedGraph> trash_graphs;
};

static UploadedGraph upload_graph(const RawGraph& graph, const gpu_t& gpu)
{
    UploadedGraph uploaded;
    uploaded.constants = upload_array(graph.constants, graph.constants_len, gpu);
    uploaded.rotations = upload_array(graph.rotations, graph.rotations_len, gpu);
    uploaded.calculations = upload_array(graph.calculations, graph.calculations_len, gpu);
    uploaded.horner_parts = upload_array(graph.horner_parts, graph.horner_parts_len, gpu);
    uploaded.graph = RawGraph{
        device_ptr(uploaded.constants),
        graph.constants_len,
        device_ptr(uploaded.rotations),
        graph.rotations_len,
        device_ptr(uploaded.calculations),
        graph.calculations_len,
        device_ptr(uploaded.horner_parts),
        graph.horner_parts_len,
        graph.num_intermediates,
    };
    return uploaded;
}

static scalar_t *device_values(scalar_device_vec_t *device)
{
    return device == nullptr || device->len == 0 ? nullptr : &(*device->values)[0];
}

static const scalar_t *device_values(const scalar_device_vec_t *device)
{
    return device == nullptr || device->len == 0 ? nullptr : &(*device->values)[0];
}

static bool device_vec_has_len(const void *raw, size_t expected)
{
    if (expected == 0) {
        return raw == nullptr || reinterpret_cast<const scalar_device_vec_t *>(raw)->len == 0;
    }
    return raw != nullptr && reinterpret_cast<const scalar_device_vec_t *>(raw)->len == expected;
}

__global__ void fr_mul_assign_kernel(scalar_t *values, scalar_t scalar, size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] *= scalar;
    }
}

__global__ void fr_mul_assign_zeta_kernel(
    scalar_t *values,
    scalar_t first,
    scalar_t second,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t class_idx = idx % 3;
        if (class_idx == 1) {
            values[idx] *= first;
        } else if (class_idx == 2) {
            values[idx] *= second;
        }
    }
}

__global__ void fr_mul_assign_repeating_kernel(
    scalar_t *values,
    const scalar_t *factors,
    size_t factor_len,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] *= factors[idx % factor_len];
    }
}

__global__ void fr_invert_assign_kernel(scalar_t *values, size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] = 1 / values[idx];
    }
}

struct RawFrCanonicalDecomposer {
    __host__ __device__ auto operator()(RawFr& key) const
    {
        return cuda::std::tie(
            key.limbs[3], key.limbs[2], key.limbs[1], key.limbs[0]);
    }
};

struct RawFrSegmentedKey {
    RawFr value;
    uint32_t segment;
};

struct RawFrSegmentedDecomposer {
    __host__ __device__ auto operator()(RawFrSegmentedKey& key) const
    {
        return cuda::std::tie(
            key.segment,
            key.value.limbs[3],
            key.value.limbs[2],
            key.value.limbs[1],
            key.value.limbs[0]);
    }
};

__global__ void fr_canonical_sort_keys_kernel(
    RawFr *keys,
    const scalar_t *values,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        scalar_t canonical = values[idx];
        canonical.from();
        *reinterpret_cast<scalar_t *>(&keys[idx]) = canonical;
    }
}

__global__ void fr_segmented_canonical_sort_keys_kernel(
    RawFrSegmentedKey *keys,
    const scalar_t *values,
    size_t segment_len,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        scalar_t canonical = values[idx];
        canonical.from();
        RawFrSegmentedKey key;
        *reinterpret_cast<scalar_t *>(&key.value) = canonical;
        key.segment = static_cast<uint32_t>(idx / segment_len);
        keys[idx] = key;
    }
}

__global__ void fr_extract_segmented_sort_keys_kernel(
    RawFr *keys,
    const RawFrSegmentedKey *segmented_keys,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        keys[idx] = segmented_keys[idx].value;
    }
}

__device__ static int raw_fr_canonical_compare(const RawFr& left, const RawFr& right)
{
    for (int limb = 3; limb >= 0; --limb) {
        if (left.limbs[limb] < right.limbs[limb]) {
            return -1;
        }
        if (left.limbs[limb] > right.limbs[limb]) {
            return 1;
        }
    }
    return 0;
}

__global__ void fr_lookup_mark_input_rows_kernel(
    const RawFr *sorted_keys,
    const scalar_t *sorted_values,
    uint8_t *unique_flags,
    uint8_t *duplicate_flags,
    uint32_t *indices,
    scalar_t *permuted_table,
    size_t len)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const bool unique = idx == 0
        || raw_fr_canonical_compare(sorted_keys[idx], sorted_keys[idx - 1]) != 0;
    unique_flags[idx] = unique ? 1 : 0;
    duplicate_flags[idx] = unique ? 0 : 1;
    indices[idx] = static_cast<uint32_t>(idx);
    if (unique) {
        permuted_table[idx] = sorted_values[idx];
    }
}

__global__ void fr_lookup_mark_table_removals_kernel(
    const RawFr *unique_keys,
    const uint32_t *unique_count,
    const RawFr *table_keys,
    uint8_t *removed,
    uint32_t *invalid,
    size_t len)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = *unique_count;
    if (idx >= count) {
        return;
    }
    const RawFr needle = unique_keys[idx];
    size_t low = 0;
    size_t high = len;
    while (low < high) {
        size_t mid = low + ((high - low) / 2);
        if (raw_fr_canonical_compare(table_keys[mid], needle) < 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low >= len || raw_fr_canonical_compare(table_keys[low], needle) != 0) {
        atomicExch(invalid, 1u);
        return;
    }
    removed[low] = 1;
}

__global__ void fr_lookup_invert_flags_kernel(
    uint8_t *out,
    const uint8_t *removed,
    size_t len)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = removed[idx] ? 0 : 1;
    }
}

__global__ void fr_lookup_assign_leftovers_kernel(
    scalar_t *permuted_table,
    const scalar_t *leftovers,
    const uint32_t *leftover_count,
    const uint32_t *duplicate_rows,
    const uint32_t *duplicate_count,
    uint32_t *invalid)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t leftovers_len = *leftover_count;
    const size_t duplicates_len = *duplicate_count;
    if (leftovers_len != duplicates_len) {
        if (idx == 0) {
            atomicExch(invalid, 1u);
        }
        return;
    }
    if (idx < leftovers_len) {
        const size_t row = duplicate_rows[duplicates_len - 1 - idx];
        permuted_table[row] = leftovers[idx];
    }
}

__global__ void fr_nonzero_suffix_stats_kernel(
    size_t *starts,
    size_t *nonzero_counts,
    const scalar_t *values,
    size_t poly_len,
    size_t count)
{
    const size_t poly_idx = blockIdx.x;
    if (poly_idx >= count) {
        return;
    }

    __shared__ size_t first[256];
    __shared__ size_t live[256];
    const size_t lane = threadIdx.x;
    const scalar_t *poly = values + (poly_idx * poly_len);
    size_t local_first = poly_len;
    size_t local_live = 0;
    for (size_t idx = lane; idx < poly_len; idx += blockDim.x) {
        if (!poly[idx].is_zero()) {
            local_first = min(local_first, idx);
            ++local_live;
        }
    }
    first[lane] = local_first;
    live[lane] = local_live;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (lane < stride) {
            first[lane] = min(first[lane], first[lane + stride]);
            live[lane] += live[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        starts[poly_idx] = first[0];
        nonzero_counts[poly_idx] = live[0];
    }
}

__global__ void fr_set_one_kernel(scalar_t *value)
{
    *value = scalar_t::one();
}

__global__ void fr_shifted_prefix_product_kernel(
    scalar_t *values,
    const scalar_t *prefix,
    scalar_t initial,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        if (idx == 0) {
            values[idx] = initial;
        } else {
            values[idx] = initial * prefix[idx - 1];
        }
    }
}

__global__ void fr_permutation_fraction_kernel(
    scalar_t *out,
    const scalar_t *columns,
    const scalar_t *permutations,
    const scalar_t *delta_powers,
    const scalar_t *omega_powers,
    scalar_t beta,
    scalar_t gamma,
    size_t len,
    size_t column_offset,
    size_t column_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    scalar_t numerator = scalar_t::one();
    scalar_t denominator = scalar_t::one();
    for (size_t local = 0; local < column_count; ++local) {
        size_t column = column_offset + local;
        scalar_t value = columns[(column * len) + idx];
        numerator *= value + (delta_powers[column] * omega_powers[idx] * beta) + gamma;
        denominator *= value + (beta * permutations[(column * len) + idx]) + gamma;
    }
    out[idx] = numerator / denominator;
}

__global__ void fr_permutation_fraction_sources_kernel(
    scalar_t *out,
    const scalar_t *fixed,
    size_t fixed_cols,
    const scalar_t *advice,
    size_t advice_cols,
    const scalar_t *instance,
    size_t instance_cols,
    const RawColumnRef *columns,
    const scalar_t *permutations,
    const scalar_t *delta_powers,
    const scalar_t *omega_powers,
    scalar_t beta,
    scalar_t gamma,
    size_t len,
    size_t column_offset,
    size_t column_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    scalar_t numerator = scalar_t::one();
    scalar_t denominator = scalar_t::one();
    for (size_t local = 0; local < column_count; ++local) {
        size_t column = column_offset + local;
        RawColumnRef reference = columns[column];
        const scalar_t *base = select_column_base(
            reference.kind,
            reference.index,
            fixed,
            fixed_cols,
            advice,
            advice_cols,
            instance,
            instance_cols,
            len);
        if (base == nullptr) {
            out[idx] = fr_zero();
            return;
        }
        scalar_t value = base[idx];
        numerator *= value + (delta_powers[column] * omega_powers[idx] * beta) + gamma;
        denominator *= value + (beta * permutations[(column * len) + idx]) + gamma;
    }
    out[idx] = numerator / denominator;
}

__global__ void fr_lookup_fraction_kernel(
    scalar_t *out,
    const scalar_t *compressed_inputs,
    const scalar_t *compressed_tables,
    const scalar_t *permuted_inputs,
    const scalar_t *permuted_tables,
    scalar_t beta,
    scalar_t gamma,
    size_t len,
    size_t lookup_count)
{
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = len * lookup_count;
    if (flat_idx >= total) {
        return;
    }

    scalar_t numerator = (compressed_inputs[flat_idx] + beta)
                       * (compressed_tables[flat_idx] + gamma);
    scalar_t denominator = (permuted_inputs[flat_idx] + beta)
                         * (permuted_tables[flat_idx] + gamma);
    out[flat_idx] = numerator / denominator;
}

__global__ void fr_shifted_prefix_product_device_initial_kernel(
    scalar_t *values,
    const scalar_t *prefix,
    const scalar_t *initial,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        scalar_t initial_value = *initial;
        values[idx] = idx == 0 ? initial_value : initial_value * prefix[idx - 1];
    }
}

__global__ void fr_blind_tail_kernel(
    scalar_t *values,
    const scalar_t *blind_values,
    size_t len,
    size_t blinding_factors)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < blinding_factors) {
        values[len - blinding_factors + idx] = blind_values[idx];
    }
}

__global__ void fr_copy_scalar_kernel(scalar_t *out, const scalar_t *value)
{
    *out = *value;
}

__global__ void fr_poly_lincomb_kernel(
    scalar_t *out,
    const scalar_t *polys,
    const scalar_t *scalars,
    size_t poly_len,
    size_t poly_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= poly_len) {
        return;
    }

    scalar_t acc;
    acc.zero();
    for (size_t poly_idx = 0; poly_idx < poly_count; ++poly_idx) {
        acc += polys[(poly_idx * poly_len) + idx] * scalars[poly_idx];
    }
    out[idx] = acc;
}

__global__ void fr_quotient_pieces_kernel(
    scalar_t *out,
    const scalar_t *quotient,
    size_t poly_len,
    size_t piece_count)
{
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = poly_len * piece_count;
    if (flat_idx >= total) {
        return;
    }
    size_t piece = flat_idx / poly_len;
    size_t index = flat_idx % poly_len;
    if (index == poly_len - 1) {
        out[flat_idx].zero();
    } else {
        out[flat_idx] = quotient[piece * (poly_len - 1) + index];
    }
}

__global__ void fr_blind_quotient_pieces_kernel(
    scalar_t *pieces,
    const scalar_t *blinds,
    size_t poly_len,
    size_t piece_count)
{
    size_t piece = blockIdx.x * blockDim.x + threadIdx.x;
    if (piece + 1 >= piece_count) {
        return;
    }
    scalar_t blind = blinds[piece];
    pieces[piece * poly_len + (poly_len - 1)] = blind;
    pieces[(piece + 1) * poly_len] -= blind;
}

__global__ void fr_coeff_to_extended_prepare_kernel(
    scalar_t *out,
    const scalar_t *inputs,
    scalar_t first,
    scalar_t second,
    size_t input_len,
    size_t output_len,
    size_t count)
{
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = output_len * count;
    if (flat_idx >= total) {
        return;
    }

    size_t local_idx = flat_idx % output_len;
    size_t poly_idx = flat_idx / output_len;
    if (local_idx >= input_len) {
        out[flat_idx].zero();
        return;
    }

    scalar_t value = inputs[(poly_idx * input_len) + local_idx];
    size_t zeta_idx = local_idx % 3;
    if (zeta_idx == 1) {
        value *= first;
    } else if (zeta_idx == 2) {
        value *= second;
    }
    out[flat_idx] = value;
}

__global__ void fr_ntt_zeta_postprocess_kernel(
    scalar_t *values,
    scalar_t scale,
    scalar_t first,
    scalar_t second,
    size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    scalar_t value = values[idx] * scale;
    size_t zeta_idx = idx % 3;
    if (zeta_idx == 1) {
        value *= first;
    } else if (zeta_idx == 2) {
        value *= second;
    }
    values[idx] = value;
}

template <typename Launch>
static RustError fr_unary_vector_op(
    RawFr *values,
    size_t len,
    Launch launch)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }

    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values{len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);
        launch(&d_values[0], len, gpu);
        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

__global__ void fr_multiopen_q_polys_kernel(
    scalar_t *out,
    const scalar_t *polys,
    const uint32_t *q_offsets,
    const uint32_t *poly_indices,
    const scalar_t *coefficients,
    size_t poly_len,
    size_t q_count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = poly_len * q_count;
    if (idx >= total) {
        return;
    }

    const size_t q_index = idx / poly_len;
    const size_t coeff_index = idx - (q_index * poly_len);
    scalar_t acc = fr_zero();
    for (uint32_t item = q_offsets[q_index]; item < q_offsets[q_index + 1]; ++item) {
        acc += polys[(static_cast<size_t>(poly_indices[item]) * poly_len) + coeff_index]
            * coefficients[item];
    }
    out[idx] = acc;
}

__global__ void fr_fill_kernel(scalar_t *out, scalar_t value, size_t len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = value;
    }
}

__global__ void fr_add_mask_copy_kernel(
    scalar_t *out,
    const scalar_t *input,
    const scalar_t *mask,
    size_t mask_len,
    size_t len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = input[idx] + mask[idx % mask_len];
    }
}

__global__ void fr_multiopen_eval_kernel(
    scalar_t *out,
    const scalar_t *polys,
    const scalar_t *point_powers,
    const uint32_t *query_poly_indices,
    const uint32_t *query_point_indices,
    size_t poly_len,
    size_t query_count)
{
    const size_t query = blockIdx.x;
    if (query >= query_count) {
        return;
    }

    __shared__ RawFr raw_partial[256];
    auto *partial = reinterpret_cast<scalar_t *>(raw_partial);
    const size_t poly_offset = static_cast<size_t>(query_poly_indices[query]) * poly_len;
    const size_t power_offset = static_cast<size_t>(query_point_indices[query]) * poly_len;
    scalar_t acc = fr_zero();
    for (size_t index = threadIdx.x; index < poly_len; index += blockDim.x) {
        scalar_t power = index == 0
            ? scalar_t::one()
            : point_powers[power_offset + index - 1];
        acc += polys[poly_offset + index] * power;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[query] = partial[0];
    }
}

__device__ static const scalar_t *fr_device_poly(
    RawDevicePolyRef reference,
    const scalar_t *dynamic_polys,
    const scalar_t *fixed_polys,
    const scalar_t *extra_polys,
    size_t poly_len)
{
    const scalar_t *source = reference.source == 0
        ? dynamic_polys
        : (reference.source == 1 ? fixed_polys : extra_polys);
    return source + (static_cast<size_t>(reference.index) * poly_len);
}

__global__ void fr_multiopen_device_q_polys_kernel(
    scalar_t *out,
    const scalar_t *dynamic_polys,
    const scalar_t *fixed_polys,
    const scalar_t *extra_polys,
    const RawDevicePolyRef *poly_refs,
    const uint32_t *q_offsets,
    const uint32_t *poly_indices,
    const scalar_t *coefficients,
    size_t poly_len,
    size_t q_count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = poly_len * q_count;
    if (idx >= total) {
        return;
    }

    const size_t q_index = idx / poly_len;
    const size_t coeff_index = idx - (q_index * poly_len);
    scalar_t acc = fr_zero();
    for (uint32_t item = q_offsets[q_index]; item < q_offsets[q_index + 1]; ++item) {
        const scalar_t *poly = fr_device_poly(
            poly_refs[poly_indices[item]],
            dynamic_polys,
            fixed_polys,
            extra_polys,
            poly_len);
        acc += poly[coeff_index] * coefficients[item];
    }
    out[idx] = acc;
}

__global__ void fr_multiopen_device_eval_kernel(
    scalar_t *out,
    const scalar_t *dynamic_polys,
    const scalar_t *fixed_polys,
    const scalar_t *extra_polys,
    const RawDevicePolyRef *poly_refs,
    const scalar_t *point_powers,
    const uint32_t *query_poly_indices,
    const uint32_t *query_point_indices,
    size_t poly_len,
    size_t query_count)
{
    const size_t query = blockIdx.x;
    if (query >= query_count) {
        return;
    }

    __shared__ RawFr raw_partial[256];
    auto *partial = reinterpret_cast<scalar_t *>(raw_partial);
    const scalar_t *poly = fr_device_poly(
        poly_refs[query_poly_indices[query]],
        dynamic_polys,
        fixed_polys,
        extra_polys,
        poly_len);
    const size_t power_offset = static_cast<size_t>(query_point_indices[query]) * poly_len;
    scalar_t acc = fr_zero();
    for (size_t index = threadIdx.x; index < poly_len; index += blockDim.x) {
        scalar_t power = index == 0
            ? scalar_t::one()
            : point_powers[power_offset + index - 1];
        acc += poly[index] * power;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[query] = partial[0];
    }
}

struct FrAffineTransform {
    scalar_t mul;
    scalar_t add;
};

struct FrReverseCompose {
    __device__ __forceinline__ FrAffineTransform operator()(
        const FrAffineTransform& suffix,
        const FrAffineTransform& next) const
    {
        return FrAffineTransform{
            next.mul * suffix.mul,
            next.mul * suffix.add + next.add,
        };
    }
};

__global__ void fr_kate_scan_input_kernel(
    FrAffineTransform *transforms,
    const scalar_t *poly,
    scalar_t point,
    size_t poly_len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx + 1 < poly_len) {
        transforms[idx] = FrAffineTransform{point, poly[poly_len - 1 - idx]};
    }
}

__global__ void fr_kate_scan_output_kernel(
    scalar_t *out,
    const FrAffineTransform *scanned,
    size_t poly_len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx + 1 < poly_len) {
        out[poly_len - 2 - idx] = scanned[idx].add;
    }
    if (idx == 0) {
        out[poly_len - 1] = fr_zero();
    }
}

__global__ void fr_kate_scan_compact_output_kernel(
    scalar_t *out,
    const FrAffineTransform *scanned,
    size_t poly_len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx + 1 < poly_len) {
        out[poly_len - 2 - idx] = scanned[idx].add;
    }
}

__global__ void fr_eval_contiguous_polys_kernel(
    scalar_t *out,
    const scalar_t *polys,
    const scalar_t *point_powers,
    size_t poly_len,
    size_t poly_count)
{
    const size_t poly_index = blockIdx.x;
    if (poly_index >= poly_count) {
        return;
    }
    __shared__ RawFr raw_partial[256];
    auto *partial = reinterpret_cast<scalar_t *>(raw_partial);
    scalar_t acc = fr_zero();
    const scalar_t *poly = polys + (poly_index * poly_len);
    for (size_t index = threadIdx.x; index < poly_len; index += blockDim.x) {
        const scalar_t power = index == 0 ? scalar_t::one() : point_powers[index - 1];
        acc += poly[index] * power;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[poly_index] = partial[0];
    }
}

__global__ void fr_multiopen_final_poly_kernel(
    scalar_t *out,
    const scalar_t *q_polys,
    const scalar_t *f_poly,
    const scalar_t *scalars,
    size_t poly_len,
    size_t q_count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= poly_len) {
        return;
    }
    scalar_t acc = f_poly[idx] * scalars[q_count];
    for (size_t q_index = 0; q_index < q_count; ++q_index) {
        acc += q_polys[(q_index * poly_len) + idx] * scalars[q_index];
    }
    out[idx] = acc;
}

__global__ void fr_subtract_constant_kernel(scalar_t *poly, const scalar_t *value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        poly[0] -= value[0];
    }
}

__global__ void fr_poseidon_advice_from_witness_kernel(
    scalar_t *out,
    size_t domain_len,
    const scalar_t *witness,
    size_t hash_count,
    const uint16_t *layout,
    const scalar_t *metadata)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t witness_count = hash_count * 256;
    if (idx < witness_count) {
        const size_t hash_index = idx / 256;
        const size_t local = idx % 256;
        const size_t permutation = local / 128;
        const uint16_t packed = layout[local % 128];
        const size_t column = packed >> 8;
        const size_t row = packed & 0xff;
        const size_t row_start = 2 + 41 * hash_index + 21 * permutation;
        out[column * domain_len + row_start + row] = witness[idx];
    }

    if (idx < hash_count) {
        const size_t row_start = 2 + 41 * idx;
        const scalar_t previous = idx == 0 ? metadata[4] : witness[(idx - 1) * 256 + 253];
        const scalar_t first_inputs[3] = {previous, metadata[5], metadata[7]};
        for (size_t column = 0; column < 3; ++column) {
            out[column * domain_len + row_start] = first_inputs[column];
            out[column * domain_len + row_start + 1] = first_inputs[column] + metadata[8 + column];
        }

        const scalar_t first_output[3] = {
            witness[idx * 256 + 125],
            witness[idx * 256 + 126],
            witness[idx * 256 + 127],
        };
        const scalar_t second_input0 = first_output[0] + metadata[6];
        out[row_start + 20] = second_input0;
        out[domain_len + row_start + 20] = first_output[0];
        out[2 * domain_len + row_start + 20] = metadata[6];
        const scalar_t second_inputs[3] = {
            second_input0,
            first_output[1],
            first_output[2],
        };
        for (size_t column = 0; column < 3; ++column) {
            out[column * domain_len + row_start + 21] = second_inputs[column];
            out[column * domain_len + row_start + 22] =
                second_inputs[column] + metadata[8 + column];
        }
    }

    if (idx == 0) {
        out[0] = metadata[0];
        out[domain_len] = metadata[1];
        out[2 * domain_len] = metadata[2];
        out[1] = metadata[3];
    }
}

__global__ void fr_device_scatter_kernel(
    scalar_t *out,
    const uint32_t *indices,
    const scalar_t *values,
    size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[indices[idx]] = values[idx];
    }
}

__device__ static void poseidon_permutation_raw(
    scalar_t *state,
    scalar_t *sboxed,
    scalar_t *next,
    const scalar_t *mds,
    const scalar_t *round_constants,
    scalar_t *circuit_witness)
{
    const unsigned lane = threadIdx.x;
    size_t witness_offset = 0;
    if (lane < 3) {
        state[lane] += round_constants[lane];
    }
    __syncwarp();

    for (unsigned round = 0; round < 68; ++round) {
        if (lane < 3) {
            scalar_t value = state[lane];
            if (round < 4 || round >= 64 || lane == 2) {
                scalar_t square = value * value;
                if (circuit_witness != nullptr && (round < 4 || round >= 64)) {
                    circuit_witness[witness_offset + lane] = square * value;
                }
                value = square * square * value;
            }
            sboxed[lane] = value;
        }
        __syncwarp();

        if (lane < 3) {
            scalar_t acc = fr_zero();
            if (round + 1 < 68) {
                acc = round_constants[((round + 1) * 3) + lane];
            }
            for (unsigned column = 0; column < 3; ++column) {
                acc += mds[(lane * 3) + column] * sboxed[column];
            }
            next[lane] = acc;
        }
        __syncwarp();

        if (circuit_witness != nullptr) {
            if (round < 4 || round >= 64) {
                if (lane < 3) {
                    circuit_witness[witness_offset + 3 + lane] = next[lane];
                }
                witness_offset += 6;
            } else {
                const unsigned partial_in_batch = (round - 4) % 6;
                if (partial_in_batch < 5) {
                    if (lane == 2) {
                        circuit_witness[witness_offset + partial_in_batch] = next[2];
                    }
                } else {
                    if (lane < 3) {
                        circuit_witness[witness_offset + 5 + lane] = next[lane];
                    }
                    witness_offset += 8;
                }
            }
        }
        __syncwarp();

        if (lane < 3) {
            state[lane] = next[lane];
        }
        __syncwarp();
    }
}

__global__ void fr_poseidon_hash_chain_kernel(
    scalar_t *out,
    scalar_t *circuit_witness,
    scalar_t initial,
    scalar_t tail0,
    scalar_t tail1,
    scalar_t input_len,
    const scalar_t *mds,
    const scalar_t *round_constants,
    size_t len)
{
    __shared__ RawFr raw_state[3];
    __shared__ RawFr raw_sboxed[3];
    __shared__ RawFr raw_next[3];
    auto *state = reinterpret_cast<scalar_t *>(raw_state);
    auto *sboxed = reinterpret_cast<scalar_t *>(raw_sboxed);
    auto *next = reinterpret_cast<scalar_t *>(raw_next);

    const unsigned lane = threadIdx.x;
    scalar_t previous = initial;
    for (size_t index = 0; index < len; ++index) {
        if (lane < 3) {
            state[lane] = lane == 2 ? input_len : fr_zero();
        }
        __syncwarp();

        if (lane == 0) {
            state[0] += previous;
        } else if (lane == 1) {
            state[1] += tail0;
        }
        __syncwarp();
        scalar_t *hash_witness = circuit_witness == nullptr
            ? nullptr
            : circuit_witness + (index * 256);
        poseidon_permutation_raw(
            state, sboxed, next, mds, round_constants, hash_witness);

        if (lane == 0) {
            state[0] += tail1;
        }
        __syncwarp();
        poseidon_permutation_raw(
            state,
            sboxed,
            next,
            mds,
            round_constants,
            hash_witness == nullptr ? nullptr : hash_witness + 128);

        if (lane == 0) {
            previous = state[0];
            if (out != nullptr) {
                out[index] = previous;
            }
        }
    }
}

using poseidon_cgbn_context_t = cgbn_context_t<8>;
using poseidon_cgbn_env_t = cgbn_env_t<poseidon_cgbn_context_t, 256>;
using poseidon_cgbn_mem_t = cgbn_mem_t<256>;

__device__ __forceinline__ poseidon_cgbn_mem_t *poseidon_cgbn_mem(RawFr *value)
{
    return reinterpret_cast<poseidon_cgbn_mem_t *>(value);
}

__device__ __forceinline__ poseidon_cgbn_mem_t *poseidon_cgbn_mem(const RawFr *value)
{
    // CGBN's load API is missing const qualification even though it only reads.
    return reinterpret_cast<poseidon_cgbn_mem_t *>(const_cast<RawFr *>(value));
}

__device__ __forceinline__ void poseidon_cgbn_add_mod(
    poseidon_cgbn_env_t& env,
    poseidon_cgbn_env_t::cgbn_t& out,
    const poseidon_cgbn_env_t::cgbn_t& lhs,
    const poseidon_cgbn_env_t::cgbn_t& rhs,
    const poseidon_cgbn_env_t::cgbn_t& modulus)
{
    const int32_t carry = cgbn_add(env, out, lhs, rhs);
    if (carry != 0 || cgbn_compare(env, out, modulus) >= 0) {
        cgbn_sub(env, out, out, modulus);
    }
}

__device__ __forceinline__ void poseidon_cgbn_canonicalize(
    poseidon_cgbn_env_t& env,
    poseidon_cgbn_env_t::cgbn_t& value,
    const poseidon_cgbn_env_t::cgbn_t& modulus)
{
    if (cgbn_compare(env, value, modulus) >= 0) {
        cgbn_sub(env, value, value, modulus);
    }
}

__device__ __forceinline__ void poseidon_cgbn_square3(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *out,
    const RawFr *input)
{
    const unsigned group = threadIdx.x / 8;
    if (group < 3) {
        poseidon_cgbn_env_t::cgbn_t value;
        poseidon_cgbn_env_t::cgbn_t squared;
        cgbn_load(env, value, poseidon_cgbn_mem(&input[group]));
        cgbn_mont_sqr(env, squared, value, modulus, 0xffffffffu);
        poseidon_cgbn_canonicalize(env, squared, modulus);
        cgbn_store(env, poseidon_cgbn_mem(&out[group]), squared);
    }
    __syncthreads();
}

__device__ __forceinline__ void poseidon_cgbn_cube_and_fourth3(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *cube,
    RawFr *fourth,
    const RawFr *square,
    const RawFr *input)
{
    const unsigned group = threadIdx.x / 8;
    if (group < 6) {
        const unsigned field = group % 3;
        poseidon_cgbn_env_t::cgbn_t lhs;
        poseidon_cgbn_env_t::cgbn_t rhs;
        poseidon_cgbn_env_t::cgbn_t product;
        cgbn_load(env, lhs, poseidon_cgbn_mem(&square[field]));
        cgbn_load(
            env,
            rhs,
            poseidon_cgbn_mem(group < 3 ? &input[field] : &square[field]));
        cgbn_mont_mul(env, product, lhs, rhs, modulus, 0xffffffffu);
        poseidon_cgbn_canonicalize(env, product, modulus);
        cgbn_store(
            env,
            poseidon_cgbn_mem(group < 3 ? &cube[field] : &fourth[field]),
            product);
    }
    __syncthreads();
}

__device__ __forceinline__ void poseidon_cgbn_mul3(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *out,
    const RawFr *lhs_values,
    const RawFr *rhs_values)
{
    const unsigned group = threadIdx.x / 8;
    if (group < 3) {
        poseidon_cgbn_env_t::cgbn_t lhs;
        poseidon_cgbn_env_t::cgbn_t rhs;
        poseidon_cgbn_env_t::cgbn_t product;
        cgbn_load(env, lhs, poseidon_cgbn_mem(&lhs_values[group]));
        cgbn_load(env, rhs, poseidon_cgbn_mem(&rhs_values[group]));
        cgbn_mont_mul(env, product, lhs, rhs, modulus, 0xffffffffu);
        poseidon_cgbn_canonicalize(env, product, modulus);
        cgbn_store(env, poseidon_cgbn_mem(&out[group]), product);
    }
    __syncthreads();
}

__device__ __forceinline__ void poseidon_cgbn_mds(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *state,
    RawFr *products,
    const RawFr *sboxed,
    const RawFr *mds,
    const RawFr *round_constants)
{
    const unsigned group = threadIdx.x / 8;
    if (group < 9) {
        poseidon_cgbn_env_t::cgbn_t lhs;
        poseidon_cgbn_env_t::cgbn_t rhs;
        poseidon_cgbn_env_t::cgbn_t product;
        cgbn_load(env, lhs, poseidon_cgbn_mem(&sboxed[group % 3]));
        cgbn_load(env, rhs, poseidon_cgbn_mem(&mds[group]));
        cgbn_mont_mul(env, product, lhs, rhs, modulus, 0xffffffffu);
        poseidon_cgbn_canonicalize(env, product, modulus);
        cgbn_store(env, poseidon_cgbn_mem(&products[group]), product);
    }
    __syncthreads();

    if (group < 3) {
        poseidon_cgbn_env_t::cgbn_t acc;
        poseidon_cgbn_env_t::cgbn_t value;
        poseidon_cgbn_env_t::cgbn_t sum;
        cgbn_load(env, acc, poseidon_cgbn_mem(&products[group * 3]));
        cgbn_load(env, value, poseidon_cgbn_mem(&products[(group * 3) + 1]));
        poseidon_cgbn_add_mod(env, sum, acc, value, modulus);
        cgbn_load(env, value, poseidon_cgbn_mem(&products[(group * 3) + 2]));
        poseidon_cgbn_add_mod(env, acc, sum, value, modulus);
        if (round_constants != nullptr) {
            cgbn_load(env, value, poseidon_cgbn_mem(&round_constants[group]));
            poseidon_cgbn_add_mod(env, sum, acc, value, modulus);
            cgbn_set(env, acc, sum);
        }
        cgbn_store(env, poseidon_cgbn_mem(&state[group]), acc);
    }
    __syncthreads();
}

__device__ __forceinline__ void poseidon_cgbn_add_constants(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *state,
    const RawFr *constants)
{
    const unsigned group = threadIdx.x / 8;
    if (group < 3) {
        poseidon_cgbn_env_t::cgbn_t value;
        poseidon_cgbn_env_t::cgbn_t constant;
        poseidon_cgbn_env_t::cgbn_t sum;
        cgbn_load(env, value, poseidon_cgbn_mem(&state[group]));
        cgbn_load(env, constant, poseidon_cgbn_mem(&constants[group]));
        poseidon_cgbn_add_mod(env, sum, value, constant, modulus);
        cgbn_store(env, poseidon_cgbn_mem(&state[group]), sum);
    }
    __syncthreads();
}

__device__ __forceinline__ void poseidon_cgbn_copy_fields(
    scalar_t *out,
    size_t offset,
    const RawFr *values,
    size_t count)
{
    if (out != nullptr) {
        auto *out_words = reinterpret_cast<uint32_t *>(out + offset);
        auto *value_words = reinterpret_cast<const uint32_t *>(values);
        for (size_t index = threadIdx.x; index < count * 8; index += blockDim.x) {
            out_words[index] = value_words[index];
        }
    }
    __syncthreads();
}

__device__ static void poseidon_permutation_cgbn(
    poseidon_cgbn_env_t& env,
    const poseidon_cgbn_env_t::cgbn_t& modulus,
    RawFr *state,
    RawFr *square,
    RawFr *cube,
    RawFr *fourth,
    RawFr *sboxed,
    RawFr *products,
    const RawFr *mds,
    const RawFr *round_constants,
    scalar_t *circuit_witness)
{
    size_t witness_offset = 0;
    poseidon_cgbn_add_constants(env, modulus, state, round_constants);

    for (unsigned round = 0; round < 68; ++round) {
        const bool full_round = round < 4 || round >= 64;
        poseidon_cgbn_square3(env, modulus, square, state);
        if (full_round) {
            poseidon_cgbn_cube_and_fourth3(
                env, modulus, cube, fourth, square, state);
            poseidon_cgbn_copy_fields(circuit_witness, witness_offset, cube, 3);
        } else {
            poseidon_cgbn_square3(env, modulus, fourth, square);
        }
        poseidon_cgbn_mul3(env, modulus, sboxed, fourth, state);
        if (!full_round) {
            auto *sboxed_words = reinterpret_cast<uint32_t *>(sboxed);
            auto *state_words = reinterpret_cast<const uint32_t *>(state);
            for (size_t index = threadIdx.x; index < 16; index += blockDim.x) {
                sboxed_words[index] = state_words[index];
            }
            __syncthreads();
        }

        const RawFr *next_constants = round + 1 < 68
            ? round_constants + ((round + 1) * 3)
            : nullptr;
        poseidon_cgbn_mds(
            env, modulus, state, products, sboxed, mds, next_constants);

        if (circuit_witness != nullptr) {
            if (full_round) {
                poseidon_cgbn_copy_fields(circuit_witness, witness_offset + 3, state, 3);
                witness_offset += 6;
            } else {
                const unsigned partial_in_batch = (round - 4) % 6;
                if (partial_in_batch < 5) {
                    poseidon_cgbn_copy_fields(
                        circuit_witness, witness_offset + partial_in_batch, &state[2], 1);
                } else {
                    poseidon_cgbn_copy_fields(circuit_witness, witness_offset + 5, state, 3);
                    witness_offset += 8;
                }
            }
        }
    }
}

__global__ void fr_poseidon_hash_chain_cgbn_kernel(
    scalar_t *out,
    scalar_t *circuit_witness,
    scalar_t initial,
    scalar_t tail0,
    scalar_t tail1,
    scalar_t input_len,
    const scalar_t *mds,
    const scalar_t *round_constants,
    size_t len)
{
    __shared__ RawFr state[3];
    __shared__ RawFr square[3];
    __shared__ RawFr cube[3];
    __shared__ RawFr fourth[3];
    __shared__ RawFr sboxed[3];
    __shared__ RawFr products[9];

    poseidon_cgbn_context_t context;
    poseidon_cgbn_env_t env(context.env<poseidon_cgbn_env_t>());
    poseidon_cgbn_env_t::cgbn_t modulus;
    cgbn_load(
        env,
        modulus,
        reinterpret_cast<poseidon_cgbn_mem_t *>(
            const_cast<uint32_t *>(device::BLS12_381_r)));

    auto *typed_state = reinterpret_cast<scalar_t *>(state);
    scalar_t previous = initial;
    for (size_t index = 0; index < len; ++index) {
        if (threadIdx.x == 0) {
            typed_state[0] = previous;
        } else if (threadIdx.x == 1) {
            typed_state[1] = tail0;
        } else if (threadIdx.x == 2) {
            typed_state[2] = input_len;
        }
        __syncthreads();

        scalar_t *hash_witness = circuit_witness == nullptr
            ? nullptr
            : circuit_witness + (index * 256);
        poseidon_permutation_cgbn(
            env,
            modulus,
            state,
            square,
            cube,
            fourth,
            sboxed,
            products,
            reinterpret_cast<const RawFr *>(mds),
            reinterpret_cast<const RawFr *>(round_constants),
            hash_witness);

        if (threadIdx.x == 0) {
            typed_state[0] += tail1;
        }
        __syncthreads();
        poseidon_permutation_cgbn(
            env,
            modulus,
            state,
            square,
            cube,
            fourth,
            sboxed,
            products,
            reinterpret_cast<const RawFr *>(mds),
            reinterpret_cast<const RawFr *>(round_constants),
            hash_witness == nullptr ? nullptr : hash_witness + 128);

        if (threadIdx.x == 0) {
            previous = typed_state[0];
            if (out != nullptr) {
                out[index] = previous;
            }
        }
        __syncthreads();
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_coeff_to_extended_batch(
    RawFr *out,
    const RawFr *inputs,
    size_t input_len,
    size_t count,
    uint32_t extended_log_n,
    RawFr first,
    RawFr second)
{
    if (input_len == 0 || count == 0) {
        return RustError{cudaSuccess};
    }

    size_t output_len = size_t{1} << extended_log_n;
    if (input_len > output_len) {
        return RustError{-1, "input length exceeds extended domain length"};
    }

    auto *typed_out = reinterpret_cast<scalar_t *>(out);
    auto *typed_inputs = reinterpret_cast<const scalar_t *>(inputs);
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();

        dev_ptr_t<scalar_t> d_inputs{input_len * count, gpu};
        dev_ptr_t<scalar_t> d_out{output_len * count, gpu};
        gpu.HtoD(&d_inputs[0], typed_inputs, input_len * count);

        const size_t block = 256;
        const size_t total = output_len * count;
        const size_t grid = (total + block - 1) / block;
        fr_coeff_to_extended_prepare_kernel<<<grid, block, 0, gpu>>>(
            &d_out[0],
            &d_inputs[0],
            typed_first,
            typed_second,
            input_len,
            output_len,
            count);
        CUDA_OK(cudaGetLastError());

        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            NTTAccess::NTT_internal(
                &d_out[poly_idx * output_len],
                extended_log_n,
                NTT::InputOutputOrder::NN,
                NTT::Direction::forward,
                NTT::Type::standard,
                const_cast<gpu_t&>(gpu));
        }

        gpu.DtoH(typed_out, &d_out[0], output_len * count);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_coeff_to_extended_batch(
    void **out,
    const RawFr *inputs,
    size_t input_len,
    size_t count,
    uint32_t extended_log_n,
    RawFr first,
    RawFr second)
{
    if (out == nullptr) {
        return RustError{cudaErrorInvalidValue, "out is null"};
    }
    *out = nullptr;
    if (input_len == 0 || count == 0 || inputs == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid coefficient batch"};
    }

    size_t output_len = size_t{1} << extended_log_n;
    if (input_len > output_len) {
        return RustError{cudaErrorInvalidValue, "input length exceeds extended domain length"};
    }

    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_inputs{input_len * count, gpu};
        gpu.HtoD(
            &d_inputs[0],
            reinterpret_cast<const scalar_t *>(inputs),
            input_len * count);

        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = output_len * count;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);

        const size_t block = 256;
        const size_t grid = (device->len + block - 1) / block;
        fr_coeff_to_extended_prepare_kernel<<<grid, block, 0, gpu>>>(
            device_values(device.get()),
            &d_inputs[0],
            typed_first,
            typed_second,
            input_len,
            output_len,
            count);
        CUDA_OK(cudaGetLastError());

        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            NTTAccess::NTT_internal(
                device_values(device.get()) + (poly_idx * output_len),
                extended_log_n,
                NTT::InputOutputOrder::NN,
                NTT::Direction::forward,
                NTT::Type::standard,
                const_cast<gpu_t&>(gpu));
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_coeff_to_extended_from_device_batch(
    void **out,
    const void *source,
    size_t input_len,
    size_t count,
    uint32_t extended_log_n,
    RawFr first,
    RawFr second)
{
    if (out == nullptr || source == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid resident coefficient source"};
    }
    *out = nullptr;
    const auto *input = reinterpret_cast<const scalar_device_vec_t *>(source);
    if (input_len == 0 || count == 0 || input->len != input_len * count) {
        return RustError{cudaErrorInvalidValue, "resident coefficient source length mismatch"};
    }

    size_t output_len = size_t{1} << extended_log_n;
    if (input_len > output_len || count > SIZE_MAX / output_len) {
        return RustError{cudaErrorInvalidValue, "invalid resident extended-domain size"};
    }

    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = output_len * count;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);

        const size_t block = 256;
        const size_t grid = (device->len + block - 1) / block;
        fr_coeff_to_extended_prepare_kernel<<<grid, block, 0, gpu>>>(
            device_values(device.get()),
            device_values(input),
            typed_first,
            typed_second,
            input_len,
            output_len,
            count);
        CUDA_OK(cudaGetLastError());

        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            NTTAccess::NTT_internal(
                device_values(device.get()) + (poly_idx * output_len),
                extended_log_n,
                NTT::InputOutputOrder::NN,
                NTT::Direction::forward,
                NTT::Type::standard,
                const_cast<gpu_t&>(gpu));
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_upload(
    void **out,
    const RawFr *values,
    size_t len)
{
    if (out == nullptr) {
        return RustError{cudaErrorInvalidValue, "out is null"};
    }
    *out = nullptr;

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(len, gpu);
        if (len != 0) {
            if (values == nullptr) {
                return RustError{cudaErrorInvalidValue, "values is null"};
            }
            gpu.HtoD(&(*device->values)[0], reinterpret_cast<const scalar_t *>(values), len);
        }
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_upload_many(
    void **out,
    const RawFr *const *values,
    const size_t *lengths,
    size_t count)
{
    if (out == nullptr || (count != 0 && (values == nullptr || lengths == nullptr))) {
        return RustError{cudaErrorInvalidValue, "invalid multi-slice upload arguments"};
    }
    *out = nullptr;
    size_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        if ((lengths[index] != 0 && values[index] == nullptr)
            || lengths[index] > SIZE_MAX - total) {
            return RustError{cudaErrorInvalidValue, "invalid multi-slice upload source"};
        }
        total += lengths[index];
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = total;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(total, gpu);
        size_t offset = 0;
        for (size_t index = 0; index < count; ++index) {
            if (lengths[index] != 0) {
                gpu.HtoD(
                    &(*device->values)[offset],
                    reinterpret_cast<const scalar_t *>(values[index]),
                    lengths[index]);
                offset += lengths[index];
            }
        }
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_poseidon_advice_from_witness(
    void **out,
    const RawFr *witness,
    size_t hash_count,
    size_t domain_len,
    const RawFr *metadata,
    size_t metadata_len,
    const uint16_t *layout,
    size_t layout_len)
{
    constexpr size_t advice_columns = 8;
    constexpr size_t witness_per_hash = 256;
    if (out == nullptr || hash_count == 0 || domain_len == 0 || witness == nullptr
        || metadata == nullptr || metadata_len != 11 || layout == nullptr || layout_len != 128
        || hash_count > SIZE_MAX / witness_per_hash
        || domain_len > SIZE_MAX / advice_columns
        || 2 + 41 * hash_count >= domain_len) {
        return RustError{cudaErrorInvalidValue, "invalid Poseidon advice arguments"};
    }
    *out = nullptr;

    const size_t witness_len = hash_count * witness_per_hash;
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = advice_columns * domain_len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        dev_ptr_t<scalar_t> d_witness{witness_len, gpu};
        dev_ptr_t<scalar_t> d_metadata{metadata_len, gpu};
        dev_ptr_t<uint16_t> d_layout{layout_len, gpu};
        gpu.bzero(device_values(device.get()), device->len);
        gpu.HtoD(&d_witness[0], reinterpret_cast<const scalar_t *>(witness), witness_len);
        gpu.HtoD(&d_metadata[0], reinterpret_cast<const scalar_t *>(metadata), metadata_len);
        gpu.HtoD(&d_layout[0], layout, layout_len);

        const size_t block = 256;
        const size_t grid = (witness_len + block - 1) / block;
        fr_poseidon_advice_from_witness_kernel<<<grid, block, 0, gpu>>>(
            device_values(device.get()),
            domain_len,
            &d_witness[0],
            hash_count,
            &d_layout[0],
            &d_metadata[0]);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_scatter_host(
    void *device_vec,
    const uint32_t *indices,
    const RawFr *values,
    size_t count)
{
    if (device_vec == nullptr || (count != 0 && (indices == nullptr || values == nullptr))) {
        return RustError{cudaErrorInvalidValue, "invalid device scatter arguments"};
    }
    auto *device = reinterpret_cast<scalar_device_vec_t *>(device_vec);
    for (size_t index = 0; index < count; ++index) {
        if (indices[index] >= device->len) {
            return RustError{cudaErrorInvalidValue, "device scatter index out of range"};
        }
    }
    if (count == 0) {
        return RustError{cudaSuccess};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<uint32_t> d_indices{count, gpu};
        dev_ptr_t<scalar_t> d_values{count, gpu};
        gpu.HtoD(&d_indices[0], indices, count);
        gpu.HtoD(&d_values[0], reinterpret_cast<const scalar_t *>(values), count);
        const size_t block = 256;
        fr_device_scatter_kernel<<<(count + block - 1) / block, block, 0, gpu>>>(
            device_values(device), &d_indices[0], &d_values[0], count);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_concat_host(
    void **out,
    const void *prefix_device_vec,
    const RawFr *suffix,
    size_t suffix_len)
{
    if (out == nullptr || prefix_device_vec == nullptr
        || (suffix_len != 0 && suffix == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid device vector concat arguments"};
    }
    *out = nullptr;
    const auto *prefix = reinterpret_cast<const scalar_device_vec_t *>(prefix_device_vec);
    if (suffix_len > SIZE_MAX - prefix->len) {
        return RustError{cudaErrorInvalidValue, "device vector concat length overflow"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = prefix->len + suffix_len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        if (prefix->len != 0) {
            CUDA_OK(cudaMemcpyAsync(
                &(*device->values)[0],
                device_values(prefix),
                prefix->len * sizeof(scalar_t),
                cudaMemcpyDeviceToDevice,
                gpu));
        }
        if (suffix_len != 0) {
            gpu.HtoD(
                &(*device->values)[prefix->len],
                reinterpret_cast<const scalar_t *>(suffix),
                suffix_len);
        }
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_concat_many_host(
    void **out,
    const void *const *source_device_vecs,
    size_t source_count,
    const RawFr *suffix,
    size_t suffix_len)
{
    if (out == nullptr || source_count == 0 || source_device_vecs == nullptr
        || (suffix_len != 0 && suffix == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid multi-device vector concat arguments"};
    }
    *out = nullptr;
    size_t total_len = suffix_len;
    for (size_t i = 0; i < source_count; ++i) {
        if (source_device_vecs[i] == nullptr) {
            return RustError{cudaErrorInvalidValue, "null device vector concat source"};
        }
        const auto *source = reinterpret_cast<const scalar_device_vec_t *>(source_device_vecs[i]);
        if (source->len > SIZE_MAX - total_len) {
            return RustError{cudaErrorInvalidValue, "multi-device vector concat length overflow"};
        }
        total_len += source->len;
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = total_len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(total_len, gpu);
        size_t offset = 0;
        for (size_t i = 0; i < source_count; ++i) {
            const auto *source = reinterpret_cast<const scalar_device_vec_t *>(source_device_vecs[i]);
            if (source->len != 0) {
                CUDA_OK(cudaMemcpyAsync(
                    &(*device->values)[offset],
                    device_values(source),
                    source->len * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice,
                    gpu));
                offset += source->len;
            }
        }
        if (suffix_len != 0) {
            gpu.HtoD(
                &(*device->values)[offset],
                reinterpret_cast<const scalar_t *>(suffix),
                suffix_len);
        }
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_gather_polys(
    void **out,
    const void *source_device_vec,
    size_t poly_len,
    const uint32_t *indices,
    size_t index_count)
{
    if (out == nullptr || source_device_vec == nullptr || poly_len == 0
        || indices == nullptr || index_count == 0 || index_count > SIZE_MAX / poly_len) {
        return RustError{cudaErrorInvalidValue, "invalid device polynomial gather arguments"};
    }
    *out = nullptr;
    const auto *source = reinterpret_cast<const scalar_device_vec_t *>(source_device_vec);
    if (source->len % poly_len != 0) {
        return RustError{cudaErrorInvalidValue, "device polynomial gather source shape mismatch"};
    }
    const size_t source_count = source->len / poly_len;
    for (size_t i = 0; i < index_count; ++i) {
        if (indices[i] >= source_count) {
            return RustError{cudaErrorInvalidValue, "device polynomial gather index out of range"};
        }
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = index_count * poly_len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        for (size_t i = 0; i < index_count; ++i) {
            CUDA_OK(cudaMemcpyAsync(
                device_values(device.get()) + (i * poly_len),
                device_values(source) + (static_cast<size_t>(indices[i]) * poly_len),
                poly_len * sizeof(scalar_t),
                cudaMemcpyDeviceToDevice,
                gpu));
        }
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_download(
    RawFr *out,
    const void *device_vec,
    size_t len)
{
    if (device_vec == nullptr || (len != 0 && out == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid device vector download arguments"};
    }
    auto *device = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    if (device->len != len) {
        return RustError{cudaErrorInvalidValue, "device vector length mismatch"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        if (len != 0) {
            gpu.DtoH(reinterpret_cast<scalar_t *>(out), &(*device->values)[0], len);
        }
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_download_nonzero_suffixes(
    RawFr *out,
    size_t out_capacity,
    size_t *out_len,
    size_t *starts,
    size_t *nonzero_counts,
    const void *device_vec,
    size_t scalar_offset,
    size_t poly_len,
    size_t count)
{
    if (out_len == nullptr || starts == nullptr || device_vec == nullptr
        || poly_len == 0 || count == 0 || count > SIZE_MAX / poly_len
        || (out_capacity != 0 && out == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid nonzero suffix download arguments"};
    }
    *out_len = 0;
    const auto *device = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    const size_t required = poly_len * count;
    if (scalar_offset > device->len || required > device->len - scalar_offset) {
        return RustError{cudaErrorInvalidValue, "nonzero suffix range exceeds device vector"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<size_t> d_starts{count, gpu};
        dev_ptr_t<size_t> d_nonzero_counts{count, gpu};
        fr_nonzero_suffix_stats_kernel<<<count, 256, 0, gpu>>>(
            &d_starts[0],
            &d_nonzero_counts[0],
            device_values(device) + scalar_offset,
            poly_len,
            count);
        CUDA_OK(cudaGetLastError());
        gpu.DtoH(starts, &d_starts[0], count);
        if (nonzero_counts != nullptr) {
            gpu.DtoH(nonzero_counts, &d_nonzero_counts[0], count);
        }
        gpu.sync();

        size_t total = 0;
        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            if (starts[poly_idx] > poly_len
                || poly_len - starts[poly_idx] > SIZE_MAX - total) {
                return RustError{cudaErrorInvalidValue, "invalid nonzero suffix start"};
            }
            total += poly_len - starts[poly_idx];
        }
        if (out == nullptr && out_capacity == 0) {
            *out_len = total;
            return RustError{cudaSuccess};
        }
        if (total > out_capacity) {
            return RustError{cudaErrorInvalidValue, "nonzero suffix output is too small"};
        }

        size_t cursor = 0;
        auto *typed_out = reinterpret_cast<scalar_t *>(out);
        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            const size_t len = poly_len - starts[poly_idx];
            if (len == 0) {
                continue;
            }
            gpu.DtoH(
                typed_out + cursor,
                device_values(device) + scalar_offset + (poly_idx * poly_len)
                    + starts[poly_idx],
                len);
            cursor += len;
        }
        gpu.sync();
        *out_len = total;
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_quotient_pieces(
    void **out,
    const void *quotient_vec,
    size_t poly_len,
    size_t piece_count,
    const RawFr *blinds,
    size_t blind_count)
{
    if (out == nullptr || quotient_vec == nullptr || poly_len < 2 || piece_count < 2
        || blind_count != piece_count - 1 || blinds == nullptr
        || piece_count > SIZE_MAX / poly_len) {
        return RustError{cudaErrorInvalidValue, "invalid resident quotient piece arguments"};
    }
    *out = nullptr;
    const auto *quotient = reinterpret_cast<const scalar_device_vec_t *>(quotient_vec);
    const size_t required = (poly_len - 1) * piece_count;
    if (quotient->len < required) {
        return RustError{cudaErrorInvalidValue, "resident quotient is too short"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_blinds = upload_scalars(blinds, blind_count, gpu);
        auto pieces = std::make_unique<scalar_device_vec_t>();
        pieces->len = poly_len * piece_count;
        pieces->values = std::make_unique<dev_ptr_t<scalar_t>>(pieces->len, gpu);
        const size_t block = 256;
        const size_t grid = (pieces->len + block - 1) / block;
        fr_quotient_pieces_kernel<<<grid, block, 0, gpu>>>(
            device_values(pieces.get()),
            device_values(quotient),
            poly_len,
            piece_count);
        CUDA_OK(cudaGetLastError());
        fr_blind_quotient_pieces_kernel<<<1, block, 0, gpu>>>(
            device_values(pieces.get()),
            device_ptr(d_blinds),
            poly_len,
            piece_count);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        *out = pieces.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_poly_lincomb(
    void **out,
    const void *polys_vec,
    size_t poly_len,
    size_t poly_count,
    const RawFr *scalars,
    size_t scalar_count)
{
    if (out == nullptr || polys_vec == nullptr || poly_len == 0 || poly_count == 0
        || scalar_count != poly_count || scalars == nullptr
        || poly_count > SIZE_MAX / poly_len) {
        return RustError{cudaErrorInvalidValue, "invalid resident polynomial combination"};
    }
    *out = nullptr;
    const auto *polys = reinterpret_cast<const scalar_device_vec_t *>(polys_vec);
    if (polys->len != poly_len * poly_count) {
        return RustError{cudaErrorInvalidValue, "resident polynomial combination shape mismatch"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_scalars = upload_scalars(scalars, scalar_count, gpu);
        auto combined = std::make_unique<scalar_device_vec_t>();
        combined->len = poly_len;
        combined->values = std::make_unique<dev_ptr_t<scalar_t>>(poly_len, gpu);
        const size_t block = 256;
        const size_t grid = (poly_len + block - 1) / block;
        fr_poly_lincomb_kernel<<<grid, block, 0, gpu>>>(
            device_values(combined.get()),
            device_values(polys),
            device_ptr(d_scalars),
            poly_len,
            poly_count);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        *out = combined.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_ntt_zeta(
    void *device_vec,
    uint32_t log_n,
    bool inverse,
    RawFr scale,
    RawFr first,
    RawFr second)
{
    if (device_vec == nullptr) {
        return RustError{cudaErrorInvalidValue, "device vector is null"};
    }
    auto *device = reinterpret_cast<scalar_device_vec_t *>(device_vec);
    size_t len = size_t{1} << log_n;
    if (device->len != len) {
        return RustError{cudaErrorInvalidValue, "device vector length mismatch"};
    }
    if (log_n == 0) {
        return RustError{cudaSuccess};
    }

    auto typed_scale = load_scalar(scale);
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        NTTAccess::NTT_internal(
            &(*device->values)[0],
            log_n,
            NTT::InputOutputOrder::NN,
            inverse ? NTT::Direction::inverse : NTT::Direction::forward,
            NTT::Type::standard,
            const_cast<gpu_t&>(gpu));

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_ntt_zeta_postprocess_kernel<<<grid, block, 0, gpu>>>(
            &(*device->values)[0],
            typed_scale,
            typed_first,
            typed_second,
            len);
        CUDA_OK(cudaGetLastError());
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_ntt_zeta_batch(
    void *device_vec,
    uint32_t log_n,
    size_t count,
    bool inverse,
    RawFr scale,
    RawFr first,
    RawFr second)
{
    if (device_vec == nullptr || count == 0) {
        return RustError{cudaErrorInvalidValue, "invalid resident NTT batch arguments"};
    }
    auto *device = reinterpret_cast<scalar_device_vec_t *>(device_vec);
    const size_t len = size_t{1} << log_n;
    if (len != 0 && count > SIZE_MAX / len) {
        return RustError{cudaErrorInvalidValue, "resident NTT batch size overflow"};
    }
    if (device->len != len * count) {
        return RustError{cudaErrorInvalidValue, "resident NTT batch length mismatch"};
    }

    const auto typed_scale = load_scalar(scale);
    const auto typed_first = load_scalar(first);
    const auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
            scalar_t *poly = device_values(device) + (poly_idx * len);
            NTTAccess::NTT_internal(
                poly,
                log_n,
                NTT::InputOutputOrder::NN,
                inverse ? NTT::Direction::inverse : NTT::Direction::forward,
                NTT::Type::standard,
                const_cast<gpu_t&>(gpu));
            fr_ntt_zeta_postprocess_kernel<<<grid, block, 0, gpu>>>(
                poly,
                typed_scale,
                typed_first,
                typed_second,
                len);
            CUDA_OK(cudaGetLastError());
        }
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_vec_divide_ntt_zeta(
    void *device_vec,
    uint32_t log_n,
    const RawFr *factors,
    size_t factor_len,
    RawFr scale,
    RawFr first,
    RawFr second)
{
    if (device_vec == nullptr || factors == nullptr || factor_len == 0) {
        return RustError{cudaErrorInvalidValue, "invalid resident quotient arguments"};
    }
    auto *device = reinterpret_cast<scalar_device_vec_t *>(device_vec);
    size_t len = size_t{1} << log_n;
    if (device->len != len) {
        return RustError{cudaErrorInvalidValue, "device vector length mismatch"};
    }

    auto typed_scale = load_scalar(scale);
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_factors = upload_scalars(factors, factor_len, gpu);
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_mul_assign_repeating_kernel<<<grid, block, 0, gpu>>>(
            device_values(device),
            device_ptr(d_factors),
            factor_len,
            len);
        CUDA_OK(cudaGetLastError());

        NTTAccess::NTT_internal(
            device_values(device),
            log_n,
            NTT::InputOutputOrder::NN,
            NTT::Direction::inverse,
            NTT::Type::standard,
            const_cast<gpu_t&>(gpu));

        fr_ntt_zeta_postprocess_kernel<<<grid, block, 0, gpu>>>(
            device_values(device),
            typed_scale,
            typed_first,
            typed_second,
            len);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" void midnight_sppark_bls12_381_fr_device_vec_free(void *device_vec)
{
    delete reinterpret_cast<scalar_device_vec_t *>(device_vec);
}

extern "C" RustError midnight_sppark_bls12_381_fr_ntt_zeta(
    RawFr *values,
    uint32_t log_n,
    bool inverse,
    RawFr scale,
    RawFr first,
    RawFr second)
{
    if (log_n == 0) {
        return RustError{cudaSuccess};
    }

    size_t len = size_t{1} << log_n;
    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    auto typed_scale = load_scalar(scale);
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values{len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);

        NTTAccess::NTT_internal(
            &d_values[0],
            log_n,
            NTT::InputOutputOrder::NN,
            inverse ? NTT::Direction::inverse : NTT::Direction::forward,
            NTT::Type::standard,
            const_cast<gpu_t&>(gpu));

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_ntt_zeta_postprocess_kernel<<<grid, block, 0, gpu>>>(
            &d_values[0],
            typed_scale,
            typed_first,
            typed_second,
            len);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_divide_ntt_zeta(
    RawFr *values,
    uint32_t log_n,
    const RawFr *factors,
    size_t factor_len,
    RawFr scale,
    RawFr first,
    RawFr second)
{
    if (log_n == 0 || factor_len == 0) {
        return RustError{cudaSuccess};
    }

    size_t len = size_t{1} << log_n;
    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    auto *typed_factors = reinterpret_cast<const scalar_t *>(factors);
    auto typed_scale = load_scalar(scale);
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values{len, gpu};
        dev_ptr_t<scalar_t> d_factors{factor_len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);
        gpu.HtoD(&d_factors[0], typed_factors, factor_len);

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_mul_assign_repeating_kernel<<<grid, block, 0, gpu>>>(
            &d_values[0],
            &d_factors[0],
            factor_len,
            len);
        CUDA_OK(cudaGetLastError());

        NTTAccess::NTT_internal(
            &d_values[0],
            log_n,
            NTT::InputOutputOrder::NN,
            NTT::Direction::inverse,
            NTT::Type::standard,
            const_cast<gpu_t&>(gpu));

        fr_ntt_zeta_postprocess_kernel<<<grid, block, 0, gpu>>>(
            &d_values[0],
            typed_scale,
            typed_first,
            typed_second,
            len);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_poly_lincomb(
    RawFr *out,
    const RawFr *polys,
    size_t poly_len,
    size_t poly_count,
    const RawFr *scalars)
{
    if (poly_len == 0 || poly_count == 0) {
        return RustError{cudaSuccess};
    }

    auto *typed_out = reinterpret_cast<scalar_t *>(out);
    auto *typed_polys = reinterpret_cast<const scalar_t *>(polys);
    auto *typed_scalars = reinterpret_cast<const scalar_t *>(scalars);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_out{poly_len, gpu};
        dev_ptr_t<scalar_t> d_polys{poly_len * poly_count, gpu};
        dev_ptr_t<scalar_t> d_scalars{poly_count, gpu};
        gpu.HtoD(&d_polys[0], typed_polys, poly_len * poly_count);
        gpu.HtoD(&d_scalars[0], typed_scalars, poly_count);

        const size_t block = 256;
        const size_t grid = (poly_len + block - 1) / block;
        fr_poly_lincomb_kernel<<<grid, block, 0, gpu>>>(
            &d_out[0],
            &d_polys[0],
            &d_scalars[0],
            poly_len,
            poly_count);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_out, &d_out[0], poly_len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_multiopen_prepare(
    RawFr *q_out,
    RawFr *eval_out,
    const RawFr *polys,
    size_t poly_len,
    size_t poly_count,
    const uint32_t *q_offsets,
    size_t q_offsets_len,
    const uint32_t *q_poly_indices,
    const RawFr *q_coefficients,
    size_t q_items_len,
    const RawFr *points,
    size_t point_count,
    const uint32_t *query_poly_indices,
    const uint32_t *query_point_indices,
    size_t query_count)
{
    if (poly_len == 0 || poly_count == 0 || q_offsets_len < 2
        || q_out == nullptr || eval_out == nullptr || polys == nullptr
        || q_offsets == nullptr || q_poly_indices == nullptr || q_coefficients == nullptr
        || points == nullptr || point_count == 0 || query_poly_indices == nullptr
        || query_point_indices == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid multi-open preparation arguments"};
    }
    const size_t q_count = q_offsets_len - 1;
    if (q_offsets[0] != 0 || q_offsets[q_count] != q_items_len) {
        return RustError{cudaErrorInvalidValue, "invalid multi-open q offsets"};
    }
    for (size_t index = 0; index < q_count; ++index) {
        if (q_offsets[index] > q_offsets[index + 1]) {
            return RustError{cudaErrorInvalidValue, "non-monotonic multi-open q offsets"};
        }
    }
    for (size_t index = 0; index < q_items_len; ++index) {
        if (q_poly_indices[index] >= poly_count) {
            return RustError{cudaErrorInvalidValue, "multi-open q polynomial index out of range"};
        }
    }
    for (size_t index = 0; index < query_count; ++index) {
        if (query_poly_indices[index] >= poly_count
            || query_point_indices[index] >= point_count) {
            return RustError{cudaErrorInvalidValue, "multi-open query index out of range"};
        }
    }
    if (poly_count > SIZE_MAX / poly_len || q_count > SIZE_MAX / poly_len
        || point_count > SIZE_MAX / poly_len) {
        return RustError{cudaErrorInvalidValue, "multi-open preparation size overflow"};
    }

    auto *typed_q_out = reinterpret_cast<scalar_t *>(q_out);
    auto *typed_eval_out = reinterpret_cast<scalar_t *>(eval_out);
    auto *typed_polys = reinterpret_cast<const scalar_t *>(polys);
    auto *typed_q_coefficients = reinterpret_cast<const scalar_t *>(q_coefficients);
    auto *typed_points = reinterpret_cast<const scalar_t *>(points);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_polys{poly_len * poly_count, gpu};
        dev_ptr_t<scalar_t> d_q_out{poly_len * q_count, gpu};
        dev_ptr_t<uint32_t> d_q_offsets{q_offsets_len, gpu};
        dev_ptr_t<uint32_t> d_q_poly_indices{q_items_len, gpu};
        dev_ptr_t<scalar_t> d_q_coefficients{q_items_len, gpu};
        dev_ptr_t<scalar_t> d_points{point_count, gpu};
        dev_ptr_t<scalar_t> d_point_powers{poly_len * point_count, gpu};
        dev_ptr_t<scalar_t> d_power_factors{poly_len, gpu};
        dev_ptr_t<uint32_t> d_query_poly_indices{query_count, gpu};
        dev_ptr_t<uint32_t> d_query_point_indices{query_count, gpu};
        dev_ptr_t<scalar_t> d_eval_out{query_count, gpu};

        gpu.HtoD(&d_polys[0], typed_polys, poly_len * poly_count);
        gpu.HtoD(&d_q_offsets[0], q_offsets, q_offsets_len);
        gpu.HtoD(&d_q_poly_indices[0], q_poly_indices, q_items_len);
        gpu.HtoD(&d_q_coefficients[0], typed_q_coefficients, q_items_len);
        gpu.HtoD(&d_points[0], typed_points, point_count);
        gpu.HtoD(&d_query_poly_indices[0], query_poly_indices, query_count);
        gpu.HtoD(&d_query_point_indices[0], query_point_indices, query_count);

        const size_t block = 256;
        const size_t q_total = poly_len * q_count;
        fr_multiopen_q_polys_kernel<<<(q_total + block - 1) / block, block, 0, gpu>>>(
            &d_q_out[0],
            &d_polys[0],
            &d_q_offsets[0],
            &d_q_poly_indices[0],
            &d_q_coefficients[0],
            poly_len,
            q_count);
        CUDA_OK(cudaGetLastError());

        for (size_t point = 0; point < point_count; ++point) {
            fr_fill_kernel<<<(poly_len + block - 1) / block, block, 0, gpu>>>(
                &d_power_factors[0],
                typed_points[point],
                poly_len);
            CUDA_OK(cudaGetLastError());
            prefix_op<Multiply<scalar_t>>(
                &d_point_powers[point * poly_len],
                &d_power_factors[0],
                poly_len,
                gpu);
            CUDA_OK(cudaGetLastError());
        }

        fr_multiopen_eval_kernel<<<query_count, block, 0, gpu>>>(
            &d_eval_out[0],
            &d_polys[0],
            &d_point_powers[0],
            &d_query_poly_indices[0],
            &d_query_point_indices[0],
            poly_len,
            query_count);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_q_out, &d_q_out[0], poly_len * q_count);
        gpu.DtoH(typed_eval_out, &d_eval_out[0], query_count);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_multiopen_prepare_device(
    RawFr *q_out,
    void **q_device_out,
    RawFr *eval_out,
    const void *dynamic_source,
    const void *fixed_source,
    const void *extra_source,
    size_t poly_len,
    const RawDevicePolyRef *poly_refs,
    size_t poly_count,
    const uint32_t *q_offsets,
    size_t q_offsets_len,
    const uint32_t *q_poly_indices,
    const RawFr *q_coefficients,
    size_t q_items_len,
    const RawFr *points,
    size_t point_count,
    const uint32_t *query_poly_indices,
    const uint32_t *query_point_indices,
    size_t query_count)
{
    if (poly_len == 0 || poly_count == 0 || q_offsets_len < 2
        || (q_out == nullptr && q_device_out == nullptr) || eval_out == nullptr
        || dynamic_source == nullptr
        || fixed_source == nullptr || extra_source == nullptr || poly_refs == nullptr
        || q_offsets == nullptr || q_poly_indices == nullptr || q_coefficients == nullptr
        || points == nullptr || point_count == 0 || query_poly_indices == nullptr
        || query_point_indices == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid resident multi-open arguments"};
    }
    if (q_device_out != nullptr) {
        *q_device_out = nullptr;
    }
    const auto *dynamic = reinterpret_cast<const scalar_device_vec_t *>(dynamic_source);
    const auto *fixed = reinterpret_cast<const scalar_device_vec_t *>(fixed_source);
    const auto *extra = reinterpret_cast<const scalar_device_vec_t *>(extra_source);
    const scalar_device_vec_t *sources[3] = {dynamic, fixed, extra};
    for (size_t index = 0; index < poly_count; ++index) {
        const auto reference = poly_refs[index];
        if (reference.source > 2
            || static_cast<size_t>(reference.index) >= sources[reference.source]->len / poly_len) {
            return RustError{cudaErrorInvalidValue, "resident polynomial reference out of range"};
        }
    }

    const size_t q_count = q_offsets_len - 1;
    if (q_offsets[0] != 0 || q_offsets[q_count] != q_items_len) {
        return RustError{cudaErrorInvalidValue, "invalid resident multi-open q offsets"};
    }
    for (size_t index = 0; index < q_count; ++index) {
        if (q_offsets[index] > q_offsets[index + 1]) {
            return RustError{cudaErrorInvalidValue, "non-monotonic resident q offsets"};
        }
    }
    for (size_t index = 0; index < q_items_len; ++index) {
        if (q_poly_indices[index] >= poly_count) {
            return RustError{cudaErrorInvalidValue, "resident q polynomial index out of range"};
        }
    }
    for (size_t index = 0; index < query_count; ++index) {
        if (query_poly_indices[index] >= poly_count
            || query_point_indices[index] >= point_count) {
            return RustError{cudaErrorInvalidValue, "resident query index out of range"};
        }
    }

    auto *typed_q_out = reinterpret_cast<scalar_t *>(q_out);
    auto *typed_eval_out = reinterpret_cast<scalar_t *>(eval_out);
    auto *typed_q_coefficients = reinterpret_cast<const scalar_t *>(q_coefficients);
    auto *typed_points = reinterpret_cast<const scalar_t *>(points);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto q_device = std::make_unique<scalar_device_vec_t>();
        q_device->len = poly_len * q_count;
        q_device->values = std::make_unique<dev_ptr_t<scalar_t>>(q_device->len, gpu);
        dev_ptr_t<RawDevicePolyRef> d_poly_refs{poly_count, gpu};
        dev_ptr_t<uint32_t> d_q_offsets{q_offsets_len, gpu};
        dev_ptr_t<uint32_t> d_q_poly_indices{q_items_len, gpu};
        dev_ptr_t<scalar_t> d_q_coefficients{q_items_len, gpu};
        dev_ptr_t<scalar_t> d_point_powers{poly_len * point_count, gpu};
        dev_ptr_t<scalar_t> d_power_factors{poly_len, gpu};
        dev_ptr_t<uint32_t> d_query_poly_indices{query_count, gpu};
        dev_ptr_t<uint32_t> d_query_point_indices{query_count, gpu};
        dev_ptr_t<scalar_t> d_eval_out{query_count, gpu};

        gpu.HtoD(&d_poly_refs[0], poly_refs, poly_count);
        gpu.HtoD(&d_q_offsets[0], q_offsets, q_offsets_len);
        gpu.HtoD(&d_q_poly_indices[0], q_poly_indices, q_items_len);
        gpu.HtoD(&d_q_coefficients[0], typed_q_coefficients, q_items_len);
        gpu.HtoD(&d_query_poly_indices[0], query_poly_indices, query_count);
        gpu.HtoD(&d_query_point_indices[0], query_point_indices, query_count);

        const size_t block = 256;
        const size_t q_total = poly_len * q_count;
        fr_multiopen_device_q_polys_kernel<<<(q_total + block - 1) / block, block, 0, gpu>>>(
            device_values(q_device.get()),
            device_values(dynamic),
            device_values(fixed),
            device_values(extra),
            &d_poly_refs[0],
            &d_q_offsets[0],
            &d_q_poly_indices[0],
            &d_q_coefficients[0],
            poly_len,
            q_count);
        CUDA_OK(cudaGetLastError());

        for (size_t point = 0; point < point_count; ++point) {
            fr_fill_kernel<<<(poly_len + block - 1) / block, block, 0, gpu>>>(
                &d_power_factors[0],
                typed_points[point],
                poly_len);
            CUDA_OK(cudaGetLastError());
            prefix_op<Multiply<scalar_t>>(
                &d_point_powers[point * poly_len],
                &d_power_factors[0],
                poly_len,
                gpu);
            CUDA_OK(cudaGetLastError());
        }

        fr_multiopen_device_eval_kernel<<<query_count, block, 0, gpu>>>(
            &d_eval_out[0],
            device_values(dynamic),
            device_values(fixed),
            device_values(extra),
            &d_poly_refs[0],
            &d_point_powers[0],
            &d_query_poly_indices[0],
            &d_query_point_indices[0],
            poly_len,
            query_count);
        CUDA_OK(cudaGetLastError());

        if (q_out != nullptr) {
            gpu.DtoH(typed_q_out, device_values(q_device.get()), poly_len * q_count);
        }
        gpu.DtoH(typed_eval_out, &d_eval_out[0], query_count);
        gpu.sync();
        if (q_device_out != nullptr) {
            *q_device_out = q_device.release();
        }
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_multiopen_f_poly_device(
    void **out,
    const void *q_device_vec,
    size_t poly_len,
    const uint32_t *point_offsets,
    size_t point_offsets_len,
    const RawFr *points,
    size_t point_count,
    const RawFr *combination_scalars,
    size_t scalar_count)
{
    if (out == nullptr || q_device_vec == nullptr || poly_len < 2
        || point_offsets == nullptr || point_offsets_len < 2 || points == nullptr
        || combination_scalars == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid resident f-polynomial arguments"};
    }
    *out = nullptr;
    const size_t q_count = point_offsets_len - 1;
    const auto *q_polys = reinterpret_cast<const scalar_device_vec_t *>(q_device_vec);
    if (q_count != scalar_count || q_count > SIZE_MAX / poly_len
        || q_polys->len != q_count * poly_len || point_offsets[0] != 0
        || point_offsets[q_count] != point_count) {
        return RustError{cudaErrorInvalidValue, "resident f-polynomial shape mismatch"};
    }
    for (size_t i = 0; i < q_count; ++i) {
        if (point_offsets[i] > point_offsets[i + 1]) {
            return RustError{cudaErrorInvalidValue, "non-monotonic f-polynomial point offsets"};
        }
    }

    const auto *typed_points = reinterpret_cast<const scalar_t *>(points);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = poly_len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(poly_len, gpu);
        dev_ptr_t<scalar_t> divided{q_count * poly_len, gpu};
        dev_ptr_t<scalar_t> work_a{poly_len, gpu};
        dev_ptr_t<scalar_t> work_b{poly_len, gpu};
        dev_ptr_t<FrAffineTransform> scan_input{poly_len - 1, gpu};
        dev_ptr_t<FrAffineTransform> scan_output{poly_len - 1, gpu};
        size_t scan_temp_bytes = 0;
        FrReverseCompose compose;
        CUDA_OK(cub::DeviceScan::InclusiveScan(
            nullptr,
            scan_temp_bytes,
            &scan_input[0],
            &scan_output[0],
            compose,
            poly_len - 1,
            gpu));
        dev_ptr_t<uint8_t> scan_temp{scan_temp_bytes, gpu};

        const size_t block = 256;
        const size_t grid = (poly_len + block - 1) / block;
        for (size_t q_index = 0; q_index < q_count; ++q_index) {
            CUDA_OK(cudaMemcpyAsync(
                &work_a[0],
                device_values(q_polys) + (q_index * poly_len),
                poly_len * sizeof(scalar_t),
                cudaMemcpyDeviceToDevice,
                gpu));
            scalar_t *current = &work_a[0];
            scalar_t *next = &work_b[0];
            for (uint32_t point_index = point_offsets[q_index];
                 point_index < point_offsets[q_index + 1];
                 ++point_index) {
                fr_kate_scan_input_kernel<<<grid, block, 0, gpu>>>(
                    &scan_input[0], current, typed_points[point_index], poly_len);
                CUDA_OK(cudaGetLastError());
                CUDA_OK(cub::DeviceScan::InclusiveScan(
                    &scan_temp[0],
                    scan_temp_bytes,
                    &scan_input[0],
                    &scan_output[0],
                    compose,
                    poly_len - 1,
                    gpu));
                fr_kate_scan_output_kernel<<<grid, block, 0, gpu>>>(
                    next, &scan_output[0], poly_len);
                CUDA_OK(cudaGetLastError());
                std::swap(current, next);
            }
            CUDA_OK(cudaMemcpyAsync(
                &divided[q_index * poly_len],
                current,
                poly_len * sizeof(scalar_t),
                cudaMemcpyDeviceToDevice,
                gpu));
        }

        auto d_scalars = upload_scalars(combination_scalars, scalar_count, gpu);
        fr_poly_lincomb_kernel<<<grid, block, 0, gpu>>>(
            device_values(device.get()),
            &divided[0],
            device_ptr(d_scalars),
            poly_len,
            q_count);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_device_eval_polys(
    RawFr *out,
    const void *device_vec,
    size_t poly_len,
    size_t poly_count,
    RawFr point)
{
    if (out == nullptr || device_vec == nullptr || poly_len == 0 || poly_count == 0
        || poly_count > SIZE_MAX / poly_len) {
        return RustError{cudaErrorInvalidValue, "invalid resident polynomial evaluation arguments"};
    }
    const auto *polys = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    if (polys->len != poly_len * poly_count) {
        return RustError{cudaErrorInvalidValue, "resident polynomial evaluation shape mismatch"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> factors{poly_len, gpu};
        dev_ptr_t<scalar_t> point_powers{poly_len, gpu};
        dev_ptr_t<scalar_t> evals{poly_count, gpu};
        const size_t block = 256;
        fr_fill_kernel<<<(poly_len + block - 1) / block, block, 0, gpu>>>(
            &factors[0], load_scalar(point), poly_len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(&point_powers[0], &factors[0], poly_len, gpu);
        CUDA_OK(cudaGetLastError());
        fr_eval_contiguous_polys_kernel<<<poly_count, block, 0, gpu>>>(
            &evals[0], device_values(polys), &point_powers[0], poly_len, poly_count);
        CUDA_OK(cudaGetLastError());
        gpu.DtoH(reinterpret_cast<scalar_t *>(out), &evals[0], poly_count);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_multiopen_pi_device(
    void **out,
    const void *q_device_vec,
    const void *f_device_vec,
    size_t poly_len,
    const RawFr *combination_scalars,
    size_t scalar_count,
    RawFr point)
{
    if (out == nullptr || q_device_vec == nullptr || f_device_vec == nullptr
        || poly_len < 2 || combination_scalars == nullptr || scalar_count < 2) {
        return RustError{cudaErrorInvalidValue, "invalid resident pi-polynomial arguments"};
    }
    *out = nullptr;
    const size_t q_count = scalar_count - 1;
    const auto *q_polys = reinterpret_cast<const scalar_device_vec_t *>(q_device_vec);
    const auto *f_poly = reinterpret_cast<const scalar_device_vec_t *>(f_device_vec);
    if (q_count > SIZE_MAX / poly_len || q_polys->len != q_count * poly_len
        || f_poly->len != poly_len) {
        return RustError{cudaErrorInvalidValue, "resident pi-polynomial shape mismatch"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> final_poly{poly_len, gpu};
        auto d_scalars = upload_scalars(combination_scalars, scalar_count, gpu);
        const size_t block = 256;
        const size_t grid = (poly_len + block - 1) / block;
        fr_multiopen_final_poly_kernel<<<grid, block, 0, gpu>>>(
            &final_poly[0],
            device_values(q_polys),
            device_values(f_poly),
            device_ptr(d_scalars),
            poly_len,
            q_count);
        CUDA_OK(cudaGetLastError());

        dev_ptr_t<scalar_t> factors{poly_len, gpu};
        dev_ptr_t<scalar_t> point_powers{poly_len, gpu};
        dev_ptr_t<scalar_t> eval{1, gpu};
        fr_fill_kernel<<<grid, block, 0, gpu>>>(
            &factors[0], load_scalar(point), poly_len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(&point_powers[0], &factors[0], poly_len, gpu);
        CUDA_OK(cudaGetLastError());
        fr_eval_contiguous_polys_kernel<<<1, block, 0, gpu>>>(
            &eval[0], &final_poly[0], &point_powers[0], poly_len, 1);
        CUDA_OK(cudaGetLastError());
        fr_subtract_constant_kernel<<<1, 1, 0, gpu>>>(&final_poly[0], &eval[0]);
        CUDA_OK(cudaGetLastError());

        dev_ptr_t<FrAffineTransform> scan_input{poly_len - 1, gpu};
        dev_ptr_t<FrAffineTransform> scan_output{poly_len - 1, gpu};
        const size_t scan_grid = (poly_len - 1 + block - 1) / block;
        fr_kate_scan_input_kernel<<<scan_grid, block, 0, gpu>>>(
            &scan_input[0], &final_poly[0], load_scalar(point), poly_len);
        CUDA_OK(cudaGetLastError());
        size_t scan_temp_bytes = 0;
        FrReverseCompose compose;
        CUDA_OK(cub::DeviceScan::InclusiveScan(
            nullptr,
            scan_temp_bytes,
            &scan_input[0],
            &scan_output[0],
            compose,
            poly_len - 1,
            gpu));
        dev_ptr_t<uint8_t> scan_temp{scan_temp_bytes, gpu};
        CUDA_OK(cub::DeviceScan::InclusiveScan(
            &scan_temp[0],
            scan_temp_bytes,
            &scan_input[0],
            &scan_output[0],
            compose,
            poly_len - 1,
            gpu));

        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = poly_len - 1;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        fr_kate_scan_compact_output_kernel<<<scan_grid, block, 0, gpu>>>(
            device_values(device.get()), &scan_output[0], poly_len);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_poseidon_hash_chain(
    RawFr *out,
    size_t len,
    RawFr initial,
    RawFr tail0,
    RawFr tail1,
    RawFr input_len,
    const RawFr *mds,
    size_t mds_len,
    const RawFr *round_constants,
    size_t round_constants_len)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }
    if (out == nullptr || mds == nullptr || mds_len != 9
        || round_constants == nullptr || round_constants_len != 68 * 3) {
        return RustError{cudaErrorInvalidValue, "invalid Poseidon hash-chain arguments"};
    }

    auto *typed_out = reinterpret_cast<scalar_t *>(out);
    auto *typed_mds = reinterpret_cast<const scalar_t *>(mds);
    auto *typed_round_constants = reinterpret_cast<const scalar_t *>(round_constants);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_out{len, gpu};
        dev_ptr_t<scalar_t> d_mds{mds_len, gpu};
        dev_ptr_t<scalar_t> d_round_constants{round_constants_len, gpu};
        gpu.HtoD(&d_mds[0], typed_mds, mds_len);
        gpu.HtoD(&d_round_constants[0], typed_round_constants, round_constants_len);

        const char *implementation = std::getenv("MIDNIGHT_SPPARK_POSEIDON_IMPL");
        const bool use_cgbn = implementation != nullptr
            && (std::strcmp(implementation, "cgbn") == 0
                || std::strcmp(implementation, "cooperative") == 0
                || std::strcmp(implementation, "warp") == 0);
        if (use_cgbn) {
            fr_poseidon_hash_chain_cgbn_kernel<<<1, 96, 0, gpu>>>(
                &d_out[0],
                nullptr,
                load_scalar(initial),
                load_scalar(tail0),
                load_scalar(tail1),
                load_scalar(input_len),
                &d_mds[0],
                &d_round_constants[0],
                len);
        } else {
            fr_poseidon_hash_chain_kernel<<<1, 32, 0, gpu>>>(
                &d_out[0],
                nullptr,
                load_scalar(initial),
                load_scalar(tail0),
                load_scalar(tail1),
                load_scalar(input_len),
                &d_mds[0],
                &d_round_constants[0],
                len);
        }
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_out, &d_out[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_poseidon_hash_chain_witness(
    RawFr *out,
    size_t len,
    RawFr initial,
    RawFr tail0,
    RawFr tail1,
    RawFr input_len,
    const RawFr *mds,
    size_t mds_len,
    const RawFr *round_constants,
    size_t round_constants_len)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }
    if (out == nullptr || mds == nullptr || mds_len != 9
        || round_constants == nullptr || round_constants_len != 68 * 3) {
        return RustError{cudaErrorInvalidValue, "invalid Poseidon witness arguments"};
    }

    auto *typed_out = reinterpret_cast<scalar_t *>(out);
    auto *typed_mds = reinterpret_cast<const scalar_t *>(mds);
    auto *typed_round_constants = reinterpret_cast<const scalar_t *>(round_constants);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_out{len * 256, gpu};
        dev_ptr_t<scalar_t> d_mds{mds_len, gpu};
        dev_ptr_t<scalar_t> d_round_constants{round_constants_len, gpu};
        gpu.HtoD(&d_mds[0], typed_mds, mds_len);
        gpu.HtoD(&d_round_constants[0], typed_round_constants, round_constants_len);

        const char *implementation = std::getenv("MIDNIGHT_SPPARK_POSEIDON_IMPL");
        const bool use_cgbn = implementation != nullptr
            && (std::strcmp(implementation, "cgbn") == 0
                || std::strcmp(implementation, "cooperative") == 0
                || std::strcmp(implementation, "warp") == 0);
        if (use_cgbn) {
            fr_poseidon_hash_chain_cgbn_kernel<<<1, 96, 0, gpu>>>(
                nullptr,
                &d_out[0],
                load_scalar(initial),
                load_scalar(tail0),
                load_scalar(tail1),
                load_scalar(input_len),
                &d_mds[0],
                &d_round_constants[0],
                len);
        } else {
            fr_poseidon_hash_chain_kernel<<<1, 32, 0, gpu>>>(
                nullptr,
                &d_out[0],
                load_scalar(initial),
                load_scalar(tail0),
                load_scalar(tail1),
                load_scalar(input_len),
                &d_mds[0],
                &d_round_constants[0],
                len);
        }
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_out, &d_out[0], len * 256);
        const char *validate = std::getenv("MIDNIGHT_SPPARK_POSEIDON_VALIDATE_CGBN");
        if (use_cgbn && validate != nullptr && std::strcmp(validate, "0") != 0) {
            dev_ptr_t<scalar_t> d_reference{len * 256, gpu};
            fr_poseidon_hash_chain_kernel<<<1, 32, 0, gpu>>>(
                nullptr,
                &d_reference[0],
                load_scalar(initial),
                load_scalar(tail0),
                load_scalar(tail1),
                load_scalar(input_len),
                &d_mds[0],
                &d_round_constants[0],
                len);
            CUDA_OK(cudaGetLastError());
            std::vector<scalar_t> reference(len * 256);
            gpu.DtoH(reference.data(), &d_reference[0], reference.size());
            gpu.sync();
            for (size_t index = 0; index < reference.size(); ++index) {
                if (std::memcmp(&typed_out[index], &reference[index], sizeof(scalar_t)) == 0) {
                    continue;
                }
                const auto *actual = reinterpret_cast<const RawFr *>(&typed_out[index]);
                const auto *expected = reinterpret_cast<const RawFr *>(&reference[index]);
                std::fprintf(
                    stderr,
                    "CGBN Poseidon witness mismatch at field %zu: "
                    "actual=%016llx:%016llx:%016llx:%016llx "
                    "expected=%016llx:%016llx:%016llx:%016llx\n",
                    index,
                    static_cast<unsigned long long>(actual->limbs[3]),
                    static_cast<unsigned long long>(actual->limbs[2]),
                    static_cast<unsigned long long>(actual->limbs[1]),
                    static_cast<unsigned long long>(actual->limbs[0]),
                    static_cast<unsigned long long>(expected->limbs[3]),
                    static_cast<unsigned long long>(expected->limbs[2]),
                    static_cast<unsigned long long>(expected->limbs[1]),
                    static_cast<unsigned long long>(expected->limbs[0]));
                return RustError{-1, "CGBN Poseidon witness differs from sppark reference"};
            }
        }
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_shifted_prefix_product(
    RawFr *values,
    RawFr initial,
    size_t len)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }

    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    auto typed_initial = load_scalar(initial);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values{len, gpu};
        dev_ptr_t<scalar_t> d_prefix{len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);

        prefix_op<Multiply<scalar_t>>(&d_prefix[0], &d_values[0], len, gpu);
        CUDA_OK(cudaGetLastError());

        const size_t block = 256;
        const size_t num_blocks = (len + block - 1) / block;
        fr_shifted_prefix_product_kernel<<<num_blocks, block, 0, gpu>>>(
            &d_values[0],
            &d_prefix[0],
            typed_initial,
            len);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_permutation_products(
    void **out,
    const RawFr *columns,
    const RawFr *permutations,
    size_t len,
    size_t column_count,
    size_t chunk_len,
    RawFr beta,
    RawFr gamma,
    RawFr omega,
    const RawFr *delta_powers,
    const RawFr *blind_values,
    size_t blinding_factors)
{
    if (out == nullptr || len == 0 || column_count == 0 || chunk_len == 0
        || columns == nullptr || permutations == nullptr || delta_powers == nullptr
        || blinding_factors >= len
        || (blinding_factors != 0 && blind_values == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid permutation product arguments"};
    }
    *out = nullptr;
    const size_t set_count = (column_count + chunk_len - 1) / chunk_len;
    if (column_count > SIZE_MAX / len || set_count > SIZE_MAX / len) {
        return RustError{cudaErrorInvalidValue, "permutation product size overflow"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_columns = upload_scalars(columns, column_count * len, gpu);
        auto d_permutations = upload_scalars(permutations, column_count * len, gpu);
        auto d_delta_powers = upload_scalars(delta_powers, column_count, gpu);
        auto d_blinds = upload_scalars(
            blind_values,
            set_count * blinding_factors,
            gpu);

        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = set_count * len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        dev_ptr_t<scalar_t> modified{len, gpu};
        dev_ptr_t<scalar_t> prefix{len, gpu};
        dev_ptr_t<scalar_t> omega_factors{len, gpu};
        dev_ptr_t<scalar_t> omega_powers{len, gpu};
        dev_ptr_t<scalar_t> initial{1, gpu};

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        omega_power_factors_kernel<<<grid, block, 0, gpu>>>(
            &omega_factors[0],
            load_scalar(omega),
            len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(&omega_powers[0], &omega_factors[0], len, gpu);
        CUDA_OK(cudaGetLastError());
        fr_set_one_kernel<<<1, 1, 0, gpu>>>(&initial[0]);
        CUDA_OK(cudaGetLastError());

        for (size_t set_idx = 0; set_idx < set_count; ++set_idx) {
            const size_t column_offset = set_idx * chunk_len;
            const size_t set_columns = std::min(chunk_len, column_count - column_offset);
            scalar_t *set_out = device_values(device.get()) + (set_idx * len);
            fr_permutation_fraction_kernel<<<grid, block, 0, gpu>>>(
                &modified[0],
                device_ptr(d_columns),
                device_ptr(d_permutations),
                device_ptr(d_delta_powers),
                &omega_powers[0],
                load_scalar(beta),
                load_scalar(gamma),
                len,
                column_offset,
                set_columns);
            CUDA_OK(cudaGetLastError());
            prefix_op<Multiply<scalar_t>>(&prefix[0], &modified[0], len, gpu);
            CUDA_OK(cudaGetLastError());
            fr_shifted_prefix_product_device_initial_kernel<<<grid, block, 0, gpu>>>(
                set_out,
                &prefix[0],
                &initial[0],
                len);
            CUDA_OK(cudaGetLastError());
            fr_copy_scalar_kernel<<<1, 1, 0, gpu>>>(
                &initial[0],
                set_out + (len - (blinding_factors + 1)));
            CUDA_OK(cudaGetLastError());
            if (blinding_factors != 0) {
                const size_t blind_grid = (blinding_factors + block - 1) / block;
                fr_blind_tail_kernel<<<blind_grid, block, 0, gpu>>>(
                    set_out,
                    device_ptr(d_blinds) + (set_idx * blinding_factors),
                    len,
                    blinding_factors);
                CUDA_OK(cudaGetLastError());
            }
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_lookup_products(
    void **out,
    const RawFr *compressed_inputs,
    const RawFr *compressed_tables,
    const RawFr *permuted_inputs,
    const RawFr *permuted_tables,
    size_t len,
    size_t lookup_count,
    RawFr beta,
    RawFr gamma,
    const RawFr *blind_values,
    size_t blinding_factors)
{
    if (out == nullptr || len == 0 || lookup_count == 0
        || compressed_inputs == nullptr || compressed_tables == nullptr
        || permuted_inputs == nullptr || permuted_tables == nullptr
        || blinding_factors >= len
        || (blinding_factors != 0 && blind_values == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid lookup product arguments"};
    }
    *out = nullptr;
    if (lookup_count > SIZE_MAX / len) {
        return RustError{cudaErrorInvalidValue, "lookup product size overflow"};
    }

    const size_t total = lookup_count * len;
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_compressed_inputs = upload_scalars(compressed_inputs, total, gpu);
        auto d_compressed_tables = upload_scalars(compressed_tables, total, gpu);
        auto d_permuted_inputs = upload_scalars(permuted_inputs, total, gpu);
        auto d_permuted_tables = upload_scalars(permuted_tables, total, gpu);
        auto d_blinds = upload_scalars(
            blind_values,
            lookup_count * blinding_factors,
            gpu);

        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = total;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(total, gpu);
        dev_ptr_t<scalar_t> fractions{total, gpu};
        dev_ptr_t<scalar_t> prefix{len, gpu};

        const size_t block = 256;
        const size_t grid = (total + block - 1) / block;
        fr_lookup_fraction_kernel<<<grid, block, 0, gpu>>>(
            &fractions[0],
            device_ptr(d_compressed_inputs),
            device_ptr(d_compressed_tables),
            device_ptr(d_permuted_inputs),
            device_ptr(d_permuted_tables),
            load_scalar(beta),
            load_scalar(gamma),
            len,
            lookup_count);
        CUDA_OK(cudaGetLastError());

        const size_t poly_grid = (len + block - 1) / block;
        const scalar_t one = scalar_t::one();
        for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
            scalar_t *product = device_values(device.get()) + (lookup_idx * len);
            const scalar_t *lookup_fractions = &fractions[lookup_idx * len];
            prefix_op<Multiply<scalar_t>>(&prefix[0], lookup_fractions, len, gpu);
            CUDA_OK(cudaGetLastError());
            fr_shifted_prefix_product_kernel<<<poly_grid, block, 0, gpu>>>(
                product,
                &prefix[0],
                one,
                len);
            CUDA_OK(cudaGetLastError());
            if (blinding_factors != 0) {
                const size_t blind_grid = (blinding_factors + block - 1) / block;
                fr_blind_tail_kernel<<<blind_grid, block, 0, gpu>>>(
                    product,
                    device_ptr(d_blinds) + (lookup_idx * blinding_factors),
                    len,
                    blinding_factors);
                CUDA_OK(cudaGetLastError());
            }
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_lookup_products_device(
    void **out,
    const void *raw_trace_lagrange,
    size_t len,
    size_t lookup_count,
    RawFr beta,
    RawFr gamma,
    const RawFr *blind_values,
    size_t blinding_factors)
{
    if (out == nullptr || len == 0 || lookup_count == 0
        || raw_trace_lagrange == nullptr || blinding_factors >= len
        || (blinding_factors != 0 && blind_values == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid resident lookup product arguments"};
    }
    *out = nullptr;
    if (lookup_count > SIZE_MAX / len) {
        return RustError{cudaErrorInvalidValue, "resident lookup product size overflow"};
    }
    const size_t total = lookup_count * len;
    const auto *trace = reinterpret_cast<const scalar_device_vec_t *>(raw_trace_lagrange);
    if (trace->len != total * 4) {
        return RustError{cudaErrorInvalidValue, "resident lookup trace length mismatch"};
    }

    const scalar_t *base = device_values(trace);
    const scalar_t *compressed_inputs = base;
    const scalar_t *compressed_tables = base + total;
    const scalar_t *permuted_inputs = base + (2 * total);
    const scalar_t *permuted_tables = base + (3 * total);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_blinds = upload_scalars(
            blind_values,
            lookup_count * blinding_factors,
            gpu);
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = total;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(total, gpu);
        dev_ptr_t<scalar_t> fractions{total, gpu};
        dev_ptr_t<scalar_t> prefix{len, gpu};
        const size_t block = 256;
        const size_t grid = (total + block - 1) / block;
        fr_lookup_fraction_kernel<<<grid, block, 0, gpu>>>(
            &fractions[0],
            compressed_inputs,
            compressed_tables,
            permuted_inputs,
            permuted_tables,
            load_scalar(beta),
            load_scalar(gamma),
            len,
            lookup_count);
        CUDA_OK(cudaGetLastError());

        const size_t poly_grid = (len + block - 1) / block;
        const scalar_t one = scalar_t::one();
        for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
            scalar_t *product = device_values(device.get()) + (lookup_idx * len);
            const scalar_t *lookup_fractions = &fractions[lookup_idx * len];
            prefix_op<Multiply<scalar_t>>(&prefix[0], lookup_fractions, len, gpu);
            CUDA_OK(cudaGetLastError());
            fr_shifted_prefix_product_kernel<<<poly_grid, block, 0, gpu>>>(
                product,
                &prefix[0],
                one,
                len);
            CUDA_OK(cudaGetLastError());
            if (blinding_factors != 0) {
                const size_t blind_grid = (blinding_factors + block - 1) / block;
                fr_blind_tail_kernel<<<blind_grid, block, 0, gpu>>>(
                    product,
                    device_ptr(d_blinds) + (lookup_idx * blinding_factors),
                    len,
                    blinding_factors);
                CUDA_OK(cudaGetLastError());
            }
        }
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

static RustError populate_lookup_permutation_serial(
    scalar_device_vec_t *trace,
    size_t len,
    size_t lookup_count,
    size_t usable_rows,
    const scalar_t *input_blinds,
    const scalar_t *table_blinds,
    size_t blind_count,
    const gpu_t& gpu)
{
    const size_t group_len = lookup_count * len;
    scalar_t *trace_values = device_values(trace);
    dev_ptr_t<RawFr> keys_in{usable_rows, gpu};
    dev_ptr_t<RawFr> keys_out{usable_rows, gpu};
    dev_ptr_t<scalar_t> values_in{usable_rows, gpu};
    dev_ptr_t<scalar_t> values_out{usable_rows, gpu};
    dev_ptr_t<uint8_t> unique_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> duplicate_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> removed_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> leftover_flags{usable_rows, gpu};
    dev_ptr_t<uint32_t> indices{usable_rows, gpu};
    dev_ptr_t<RawFr> unique_keys{usable_rows, gpu};
    dev_ptr_t<uint32_t> duplicate_rows{usable_rows, gpu};
    dev_ptr_t<scalar_t> leftovers{usable_rows, gpu};
    dev_ptr_t<uint32_t> counts{3, gpu};
    dev_ptr_t<uint32_t> invalid{1, gpu};

    size_t sort_temp_bytes = 0;
    CUDA_OK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_temp_bytes,
        &keys_in[0],
        &keys_out[0],
        &values_in[0],
        &values_out[0],
        usable_rows,
        RawFrCanonicalDecomposer{},
        0,
        256,
        gpu));
    size_t unique_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        unique_temp_bytes,
        &keys_out[0],
        &unique_flags[0],
        &unique_keys[0],
        &counts[0],
        usable_rows,
        gpu));
    size_t duplicate_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        duplicate_temp_bytes,
        &indices[0],
        &duplicate_flags[0],
        &duplicate_rows[0],
        &counts[1],
        usable_rows,
        gpu));
    size_t leftover_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        leftover_temp_bytes,
        &values_out[0],
        &leftover_flags[0],
        &leftovers[0],
        &counts[2],
        usable_rows,
        gpu));
    size_t select_temp_bytes = std::max(
        unique_temp_bytes,
        std::max(duplicate_temp_bytes, leftover_temp_bytes));
    dev_ptr_t<uint8_t> sort_temp{sort_temp_bytes, gpu};
    dev_ptr_t<uint8_t> select_temp{select_temp_bytes, gpu};
    gpu.bzero(&invalid[0], 1);

    const size_t block = 256;
    const size_t grid = (usable_rows + block - 1) / block;
    for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
        const scalar_t *input = trace_values + (lookup_idx * len);
        const scalar_t *table = trace_values + group_len + (lookup_idx * len);
        scalar_t *permuted_input =
            trace_values + (2 * group_len) + (lookup_idx * len);
        scalar_t *permuted_table =
            trace_values + (3 * group_len) + (lookup_idx * len);

        CUDA_OK(cudaMemcpyAsync(
            &values_in[0],
            input,
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
        fr_canonical_sort_keys_kernel<<<grid, block, 0, gpu>>>(
            &keys_in[0],
            &values_in[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceRadixSort::SortPairs(
            &sort_temp[0],
            sort_temp_bytes,
            &keys_in[0],
            &keys_out[0],
            &values_in[0],
            &values_out[0],
            usable_rows,
            RawFrCanonicalDecomposer{},
            0,
            256,
            gpu));
        CUDA_OK(cudaMemcpyAsync(
            permuted_input,
            &values_out[0],
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
        fr_lookup_mark_input_rows_kernel<<<grid, block, 0, gpu>>>(
            &keys_out[0],
            &values_out[0],
            &unique_flags[0],
            &duplicate_flags[0],
            &indices[0],
            permuted_table,
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &keys_out[0],
            &unique_flags[0],
            &unique_keys[0],
            &counts[0],
            usable_rows,
            gpu));
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &indices[0],
            &duplicate_flags[0],
            &duplicate_rows[0],
            &counts[1],
            usable_rows,
            gpu));

        CUDA_OK(cudaMemcpyAsync(
            &values_in[0],
            table,
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
        fr_canonical_sort_keys_kernel<<<grid, block, 0, gpu>>>(
            &keys_in[0],
            &values_in[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceRadixSort::SortPairs(
            &sort_temp[0],
            sort_temp_bytes,
            &keys_in[0],
            &keys_out[0],
            &values_in[0],
            &values_out[0],
            usable_rows,
            RawFrCanonicalDecomposer{},
            0,
            256,
            gpu));
        gpu.bzero(&removed_flags[0], usable_rows);
        fr_lookup_mark_table_removals_kernel<<<grid, block, 0, gpu>>>(
            &unique_keys[0],
            &counts[0],
            &keys_out[0],
            &removed_flags[0],
            &invalid[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        fr_lookup_invert_flags_kernel<<<grid, block, 0, gpu>>>(
            &leftover_flags[0],
            &removed_flags[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &values_out[0],
            &leftover_flags[0],
            &leftovers[0],
            &counts[2],
            usable_rows,
            gpu));
        fr_lookup_assign_leftovers_kernel<<<grid, block, 0, gpu>>>(
            permuted_table,
            &leftovers[0],
            &counts[2],
            &duplicate_rows[0],
            &counts[1],
            &invalid[0]);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(
            permuted_input + usable_rows,
            input_blinds + (lookup_idx * blind_count),
            blind_count);
        gpu.HtoD(
            permuted_table + usable_rows,
            table_blinds + (lookup_idx * blind_count),
            blind_count);
    }

    uint32_t invalid_host = 0;
    gpu.DtoH(&invalid_host, &invalid[0], 1);
    gpu.sync();
    if (invalid_host != 0) {
        return RustError{-1, "lookup input is not contained in the lookup table"};
    }
    return RustError{cudaSuccess};
}

static RustError populate_lookup_permutation_batched(
    scalar_device_vec_t *trace,
    size_t len,
    size_t lookup_count,
    size_t usable_rows,
    const scalar_t *input_blinds,
    const scalar_t *table_blinds,
    size_t blind_count,
    const gpu_t& gpu)
{
    if (usable_rows == 0 || lookup_count > SIZE_MAX / usable_rows) {
        return RustError{cudaErrorInvalidValue, "lookup batch size overflow"};
    }
    const size_t total_items = lookup_count * usable_rows;
    if (lookup_count > UINT32_MAX) {
        return RustError{cudaErrorInvalidValue, "lookup batch has too many segments"};
    }

    const size_t group_len = lookup_count * len;
    scalar_t *trace_values = device_values(trace);
    dev_ptr_t<RawFrSegmentedKey> keys_in{total_items, gpu};
    dev_ptr_t<RawFrSegmentedKey> keys_out{total_items, gpu};
    dev_ptr_t<RawFr> sorted_keys{total_items, gpu};
    dev_ptr_t<scalar_t> values_in{total_items, gpu};
    dev_ptr_t<scalar_t> values_out{total_items, gpu};
    dev_ptr_t<uint8_t> unique_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> duplicate_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> removed_flags{usable_rows, gpu};
    dev_ptr_t<uint8_t> leftover_flags{usable_rows, gpu};
    dev_ptr_t<uint32_t> indices{usable_rows, gpu};
    dev_ptr_t<RawFr> unique_keys{total_items, gpu};
    dev_ptr_t<uint32_t> duplicate_rows{total_items, gpu};
    dev_ptr_t<scalar_t> leftovers{usable_rows, gpu};
    dev_ptr_t<uint32_t> counts{lookup_count * 3, gpu};
    dev_ptr_t<uint32_t> invalid{1, gpu};

    size_t sort_temp_bytes = 0;
    CUDA_OK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_temp_bytes,
        &keys_in[0],
        &keys_out[0],
        &values_in[0],
        &values_out[0],
        total_items,
        RawFrSegmentedDecomposer{},
        0,
        288,
        gpu));
    size_t unique_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        unique_temp_bytes,
        &sorted_keys[0],
        &unique_flags[0],
        &unique_keys[0],
        &counts[0],
        usable_rows,
        gpu));
    size_t duplicate_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        duplicate_temp_bytes,
        &indices[0],
        &duplicate_flags[0],
        &duplicate_rows[0],
        &counts[1],
        usable_rows,
        gpu));
    size_t leftover_temp_bytes = 0;
    CUDA_OK(cub::DeviceSelect::Flagged(
        nullptr,
        leftover_temp_bytes,
        &values_out[0],
        &leftover_flags[0],
        &leftovers[0],
        &counts[2],
        usable_rows,
        gpu));
    size_t select_temp_bytes = std::max(
        unique_temp_bytes,
        std::max(duplicate_temp_bytes, leftover_temp_bytes));
    dev_ptr_t<uint8_t> sort_temp{sort_temp_bytes, gpu};
    dev_ptr_t<uint8_t> select_temp{select_temp_bytes, gpu};
    gpu.bzero(&invalid[0], 1);

    const size_t block = 256;
    const size_t row_grid = (usable_rows + block - 1) / block;
    const size_t total_grid = (total_items + block - 1) / block;

    for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
        CUDA_OK(cudaMemcpyAsync(
            &values_in[lookup_idx * usable_rows],
            trace_values + (lookup_idx * len),
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
    }
    fr_segmented_canonical_sort_keys_kernel<<<total_grid, block, 0, gpu>>>(
        &keys_in[0],
        &values_in[0],
        usable_rows,
        total_items);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cub::DeviceRadixSort::SortPairs(
        &sort_temp[0],
        sort_temp_bytes,
        &keys_in[0],
        &keys_out[0],
        &values_in[0],
        &values_out[0],
        total_items,
        RawFrSegmentedDecomposer{},
        0,
        288,
        gpu));
    fr_extract_segmented_sort_keys_kernel<<<total_grid, block, 0, gpu>>>(
        &sorted_keys[0],
        &keys_out[0],
        total_items);
    CUDA_OK(cudaGetLastError());

    for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
        const size_t batch_offset = lookup_idx * usable_rows;
        scalar_t *permuted_input =
            trace_values + (2 * group_len) + (lookup_idx * len);
        scalar_t *permuted_table =
            trace_values + (3 * group_len) + (lookup_idx * len);
        CUDA_OK(cudaMemcpyAsync(
            permuted_input,
            &values_out[batch_offset],
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
        fr_lookup_mark_input_rows_kernel<<<row_grid, block, 0, gpu>>>(
            &sorted_keys[batch_offset],
            &values_out[batch_offset],
            &unique_flags[0],
            &duplicate_flags[0],
            &indices[0],
            permuted_table,
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &sorted_keys[batch_offset],
            &unique_flags[0],
            &unique_keys[batch_offset],
            &counts[lookup_idx * 3],
            usable_rows,
            gpu));
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &indices[0],
            &duplicate_flags[0],
            &duplicate_rows[batch_offset],
            &counts[(lookup_idx * 3) + 1],
            usable_rows,
            gpu));
        gpu.HtoD(
            permuted_input + usable_rows,
            input_blinds + (lookup_idx * blind_count),
            blind_count);
    }

    for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
        CUDA_OK(cudaMemcpyAsync(
            &values_in[lookup_idx * usable_rows],
            trace_values + group_len + (lookup_idx * len),
            usable_rows * sizeof(scalar_t),
            cudaMemcpyDeviceToDevice,
            gpu));
    }
    fr_segmented_canonical_sort_keys_kernel<<<total_grid, block, 0, gpu>>>(
        &keys_in[0],
        &values_in[0],
        usable_rows,
        total_items);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cub::DeviceRadixSort::SortPairs(
        &sort_temp[0],
        sort_temp_bytes,
        &keys_in[0],
        &keys_out[0],
        &values_in[0],
        &values_out[0],
        total_items,
        RawFrSegmentedDecomposer{},
        0,
        288,
        gpu));
    fr_extract_segmented_sort_keys_kernel<<<total_grid, block, 0, gpu>>>(
        &sorted_keys[0],
        &keys_out[0],
        total_items);
    CUDA_OK(cudaGetLastError());

    for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
        const size_t batch_offset = lookup_idx * usable_rows;
        scalar_t *permuted_table =
            trace_values + (3 * group_len) + (lookup_idx * len);
        gpu.bzero(&removed_flags[0], usable_rows);
        fr_lookup_mark_table_removals_kernel<<<row_grid, block, 0, gpu>>>(
            &unique_keys[batch_offset],
            &counts[lookup_idx * 3],
            &sorted_keys[batch_offset],
            &removed_flags[0],
            &invalid[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        fr_lookup_invert_flags_kernel<<<row_grid, block, 0, gpu>>>(
            &leftover_flags[0],
            &removed_flags[0],
            usable_rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cub::DeviceSelect::Flagged(
            &select_temp[0],
            select_temp_bytes,
            &values_out[batch_offset],
            &leftover_flags[0],
            &leftovers[0],
            &counts[(lookup_idx * 3) + 2],
            usable_rows,
            gpu));
        fr_lookup_assign_leftovers_kernel<<<row_grid, block, 0, gpu>>>(
            permuted_table,
            &leftovers[0],
            &counts[(lookup_idx * 3) + 2],
            &duplicate_rows[batch_offset],
            &counts[(lookup_idx * 3) + 1],
            &invalid[0]);
        CUDA_OK(cudaGetLastError());
        gpu.HtoD(
            permuted_table + usable_rows,
            table_blinds + (lookup_idx * blind_count),
            blind_count);
    }

    uint32_t invalid_host = 0;
    gpu.DtoH(&invalid_host, &invalid[0], 1);
    gpu.sync();
    if (invalid_host != 0) {
        return RustError{-1, "lookup input is not contained in the lookup table"};
    }
    return RustError{cudaSuccess};
}

static RustError populate_lookup_permutation(
    scalar_device_vec_t *trace,
    size_t len,
    size_t lookup_count,
    size_t usable_rows,
    const scalar_t *input_blinds,
    const scalar_t *table_blinds,
    size_t blind_count,
    const gpu_t& gpu)
{
    const char *batched = std::getenv("MIDNIGHT_GPU_LOOKUP_BATCHED_RADIX");
    if (lookup_count > 1 && batched != nullptr
        && (std::strcmp(batched, "1") == 0
            || std::strcmp(batched, "true") == 0
            || std::strcmp(batched, "on") == 0)) {
        return populate_lookup_permutation_batched(
            trace,
            len,
            lookup_count,
            usable_rows,
            input_blinds,
            table_blinds,
            blind_count,
            gpu);
    }
    return populate_lookup_permutation_serial(
        trace,
        len,
        lookup_count,
        usable_rows,
        input_blinds,
        table_blinds,
        blind_count,
        gpu);
}

extern "C" RustError midnight_sppark_bls12_381_fr_lookup_permute_batch(
    void **out,
    const RawFr *compressed_inputs,
    const RawFr *compressed_tables,
    size_t len,
    size_t lookup_count,
    size_t usable_rows,
    const RawFr *input_blinds,
    const RawFr *table_blinds,
    size_t blind_count)
{
    if (out == nullptr || compressed_inputs == nullptr || compressed_tables == nullptr
        || input_blinds == nullptr || table_blinds == nullptr
        || len == 0 || lookup_count == 0 || usable_rows == 0
        || usable_rows > len || blind_count != len - usable_rows
        || usable_rows > UINT32_MAX || lookup_count > SIZE_MAX / len) {
        return RustError{cudaErrorInvalidValue, "invalid resident lookup permutation arguments"};
    }
    *out = nullptr;
    const size_t group_len = lookup_count * len;
    if (group_len > SIZE_MAX / 4) {
        return RustError{cudaErrorInvalidValue, "resident lookup permutation size overflow"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto trace = std::make_unique<scalar_device_vec_t>();
        trace->len = group_len * 4;
        trace->values = std::make_unique<dev_ptr_t<scalar_t>>(trace->len, gpu);
        scalar_t *trace_values = device_values(trace.get());
        gpu.HtoD(trace_values, compressed_inputs, group_len);
        gpu.HtoD(trace_values + group_len, compressed_tables, group_len);

        RustError status = populate_lookup_permutation(
            trace.get(),
            len,
            lookup_count,
            usable_rows,
            reinterpret_cast<const scalar_t *>(input_blinds),
            reinterpret_cast<const scalar_t *>(table_blinds),
            blind_count,
            gpu);
        if (status.code != 0) {
            return status;
        }
        *out = trace.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_eval_numerator(
    RawFr *out,
    size_t len,
    const RawFr *initial_out,
    uint32_t run_custom,
    const RawFr *fixed,
    size_t fixed_cols,
    const RawFr *advice,
    size_t advice_cols,
    const RawFr *instance,
    size_t instance_cols,
    const RawFr *challenges,
    size_t challenge_len,
    RawFr beta,
    RawFr gamma,
    RawFr theta,
    RawFr trash_challenge,
    RawFr y,
    RawFr omega,
    RawFr delta_start,
    RawFr delta_const,
    int32_t rot_scale,
    int32_t last_rotation,
    uint32_t chunk_len,
    const RawFr *l0,
    const RawFr *l_last,
    const RawFr *l_active,
    const RawFr *permutation_products,
    size_t permutation_sets,
    const RawColumnRef *permutation_columns,
    size_t permutation_columns_len,
    const RawFr *permutation_pk_cosets,
    size_t permutation_pk_cosets_len,
    const RawFr *lookup_product_cosets,
    const RawFr *lookup_input_cosets,
    const RawFr *lookup_table_cosets,
    size_t lookup_count,
    const RawFr *trash_cosets,
    const uint32_t *trash_selector_fixed_indices,
    size_t trash_count,
    RawGraph custom_graph,
    const RawGraph *lookup_graphs,
    const RawGraph *trash_graphs)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }
    if (!run_custom && initial_out == nullptr) {
        return RustError{cudaErrorInvalidValue, "initial_out is required when custom evaluation is skipped"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();

        auto d_out = run_custom
            ? std::make_unique<dev_ptr_t<scalar_t>>(len, gpu)
            : upload_scalars(initial_out, len, gpu);
        auto d_fixed = upload_scalars(fixed, fixed_cols * len, gpu);
        auto d_advice = upload_scalars(advice, advice_cols * len, gpu);
        auto d_instance = upload_scalars(instance, instance_cols * len, gpu);
        auto d_challenges = upload_scalars(challenges, challenge_len, gpu);
        const size_t lagrange_len = (permutation_sets != 0 || lookup_count != 0) ? len : 0;
        auto d_l0 = upload_scalars(l0, lagrange_len, gpu);
        auto d_l_last = upload_scalars(l_last, lagrange_len, gpu);
        auto d_l_active = upload_scalars(l_active, lagrange_len, gpu);
        auto d_permutation_products =
            upload_scalars(permutation_products, permutation_sets * len, gpu);
        auto d_permutation_columns =
            upload_array(permutation_columns, permutation_columns_len, gpu);
        auto d_permutation_pk_cosets =
            upload_scalars(permutation_pk_cosets, permutation_pk_cosets_len * len, gpu);
        auto d_lookup_product_cosets =
            upload_scalars(lookup_product_cosets, lookup_count * len, gpu);
        auto d_lookup_input_cosets =
            upload_scalars(lookup_input_cosets, lookup_count * len, gpu);
        auto d_lookup_table_cosets =
            upload_scalars(lookup_table_cosets, lookup_count * len, gpu);
        auto d_trash_cosets = upload_scalars(trash_cosets, trash_count * len, gpu);
        auto d_trash_selector_fixed_indices =
            upload_array(trash_selector_fixed_indices, trash_count, gpu);

        auto typed_beta = load_scalar(beta);
        auto typed_gamma = load_scalar(gamma);
        auto typed_theta = load_scalar(theta);
        auto typed_trash_challenge = load_scalar(trash_challenge);
        auto typed_y = load_scalar(y);
        auto typed_omega = load_scalar(omega);
        auto typed_delta_start = load_scalar(delta_start);
        auto typed_delta_const = load_scalar(delta_const);

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;

        dev_ptr_t<scalar_t> d_omega_factors{len, gpu};
        dev_ptr_t<scalar_t> d_omega_powers{len, gpu};
        omega_power_factors_kernel<<<grid, block, 0, gpu>>>(
            &d_omega_factors[0],
            typed_omega,
            len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(&d_omega_powers[0], &d_omega_factors[0], len, gpu);
        CUDA_OK(cudaGetLastError());

        UploadedGraph custom_uploaded;
        if (run_custom) {
            custom_uploaded = upload_graph(custom_graph, gpu);
        }
        std::vector<UploadedGraph> lookup_uploaded;
        lookup_uploaded.reserve(lookup_count);
        for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
            lookup_uploaded.emplace_back(upload_graph(lookup_graphs[lookup_idx], gpu));
        }
        std::vector<UploadedGraph> trash_uploaded;
        trash_uploaded.reserve(trash_count);
        for (size_t trash_idx = 0; trash_idx < trash_count; ++trash_idx) {
            trash_uploaded.emplace_back(upload_graph(trash_graphs[trash_idx], gpu));
        }

        if (run_custom) {
            launch_numerator_custom(
                &(*d_out)[0],
                len,
                custom_uploaded.graph,
                device_ptr(d_fixed),
                fixed_cols,
                device_ptr(d_advice),
                advice_cols,
                device_ptr(d_instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                typed_beta,
                typed_gamma,
                typed_theta,
                typed_trash_challenge,
                typed_y,
                rot_scale,
                grid,
                block,
                gpu);
        }

        if (permutation_sets != 0) {
            numerator_permutation_kernel<<<grid, block, 0, gpu>>>(
                &(*d_out)[0],
                len,
                device_ptr(d_fixed),
                fixed_cols,
                device_ptr(d_advice),
                advice_cols,
                device_ptr(d_instance),
                instance_cols,
                device_ptr(d_l0),
                device_ptr(d_l_last),
                device_ptr(d_l_active),
                device_ptr(d_permutation_products),
                permutation_sets,
                device_ptr(d_permutation_columns),
                permutation_columns_len,
                device_ptr(d_permutation_pk_cosets),
                typed_beta,
                typed_gamma,
                typed_y,
                &d_omega_powers[0],
                typed_delta_start,
                typed_delta_const,
                rot_scale,
                last_rotation,
                chunk_len);
            CUDA_OK(cudaGetLastError());
        }

        for (size_t lookup_idx = 0; lookup_idx < lookup_count; ++lookup_idx) {
            launch_numerator_lookup(
                &(*d_out)[0],
                len,
                lookup_uploaded[lookup_idx].graph,
                device_ptr(d_fixed),
                fixed_cols,
                device_ptr(d_advice),
                advice_cols,
                device_ptr(d_instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                device_ptr(d_l0),
                device_ptr(d_l_last),
                device_ptr(d_l_active),
                device_ptr(d_lookup_product_cosets) + (lookup_idx * len),
                device_ptr(d_lookup_input_cosets) + (lookup_idx * len),
                device_ptr(d_lookup_table_cosets) + (lookup_idx * len),
                typed_beta,
                typed_gamma,
                typed_theta,
                typed_trash_challenge,
                typed_y,
                rot_scale,
                grid,
                block,
                gpu);
        }

        for (size_t trash_idx = 0; trash_idx < trash_count; ++trash_idx) {
            launch_numerator_trash(
                &(*d_out)[0],
                len,
                trash_uploaded[trash_idx].graph,
                device_ptr(d_fixed),
                fixed_cols,
                device_ptr(d_advice),
                advice_cols,
                device_ptr(d_instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                device_ptr(d_trash_cosets) + (trash_idx * len),
                trash_selector_fixed_indices[trash_idx],
                typed_beta,
                typed_gamma,
                typed_theta,
                typed_trash_challenge,
                typed_y,
                rot_scale,
                grid,
                block,
                gpu);
        }

        gpu.DtoH(reinterpret_cast<scalar_t *>(out), &(*d_out)[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_eval_graph_batch(
    RawFr *out,
    size_t len,
    const RawFr *fixed,
    size_t fixed_cols,
    const RawFr *advice,
    size_t advice_cols,
    const RawFr *instance,
    size_t instance_cols,
    const RawFr *challenges,
    size_t challenge_len,
    RawFr beta,
    RawFr gamma,
    RawFr theta,
    RawFr trash_challenge,
    RawFr y,
    int32_t rot_scale,
    const RawGraph *graphs,
    size_t graph_count)
{
    if (len == 0 || graph_count == 0) {
        return RustError{cudaSuccess};
    }
    if (out == nullptr || graphs == nullptr
        || (fixed_cols != 0 && fixed == nullptr)
        || (advice_cols != 0 && advice == nullptr)
        || (instance_cols != 0 && instance == nullptr)
        || (challenge_len != 0 && challenges == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid graph batch arguments"};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_out{len * graph_count, gpu};
        auto d_fixed = upload_scalars(fixed, fixed_cols * len, gpu);
        auto d_advice = upload_scalars(advice, advice_cols * len, gpu);
        auto d_instance = upload_scalars(instance, instance_cols * len, gpu);
        auto d_challenges = upload_scalars(challenges, challenge_len, gpu);
        std::vector<UploadedGraph> uploaded;
        uploaded.reserve(graph_count);
        for (size_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
            uploaded.emplace_back(upload_graph(graphs[graph_idx], gpu));
        }

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        for (size_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
            launch_numerator_custom(
                &d_out[graph_idx * len],
                len,
                uploaded[graph_idx].graph,
                device_ptr(d_fixed),
                fixed_cols,
                device_ptr(d_advice),
                advice_cols,
                device_ptr(d_instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                load_scalar(beta),
                load_scalar(gamma),
                load_scalar(theta),
                load_scalar(trash_challenge),
                load_scalar(y),
                rot_scale,
                grid,
                block,
                gpu);
        }

        gpu.DtoH(reinterpret_cast<scalar_t *>(out), &d_out[0], len * graph_count);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_numerator_context_create(
    void **out,
    size_t len,
    size_t input_len,
    const RawFr *fixed,
    const RawFr *fixed_lagrange,
    size_t fixed_cols,
    const RawFr *l0,
    const RawFr *l_last,
    const RawFr *l_active,
    const RawColumnRef *permutation_columns,
    size_t permutation_columns_len,
    const RawFr *permutation_pk_cosets,
    const RawFr *permutation_pk_lagrange,
    size_t permutation_pk_cosets_len,
    const uint32_t *trash_selector_fixed_indices,
    size_t trash_count,
    RawGraph custom_graph,
    const RawGraph *lookup_graphs,
    const RawGraph *lookup_input_graphs,
    const RawGraph *lookup_table_graphs,
    size_t lookup_count,
    const RawGraph *trash_graphs,
    RawFr omega,
    RawFr trace_omega,
    RawFr g_coset,
    RawFr delta_const,
    int32_t rot_scale,
    int32_t last_rotation,
    uint32_t chunk_len)
{
    if (out == nullptr || len == 0 || input_len == 0
        || l0 == nullptr || l_last == nullptr || l_active == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid numerator context arguments"};
    }
    if ((fixed_cols != 0 && (fixed == nullptr || fixed_lagrange == nullptr))
        || (permutation_columns_len != 0 && permutation_columns == nullptr)
        || (permutation_pk_cosets_len != 0
            && (permutation_pk_cosets == nullptr || permutation_pk_lagrange == nullptr))
        || (lookup_count != 0
            && (lookup_graphs == nullptr || lookup_input_graphs == nullptr
                || lookup_table_graphs == nullptr))
        || (trash_count != 0
            && (trash_graphs == nullptr || trash_selector_fixed_indices == nullptr))) {
        return RustError{cudaErrorInvalidValue, "missing numerator context data"};
    }
    *out = nullptr;

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto context = std::make_unique<numerator_context_t>();
        context->len = len;
        context->input_len = input_len;
        context->fixed_cols = fixed_cols;
        context->permutation_columns_len = permutation_columns_len;
        context->permutation_pk_cosets_len = permutation_pk_cosets_len;
        context->lookup_count = lookup_count;
        context->trash_count = trash_count;
        context->rot_scale = rot_scale;
        context->last_rotation = last_rotation;
        context->chunk_len = chunk_len;
        context->omega = load_scalar(omega);
        context->g_coset = load_scalar(g_coset);
        context->delta_const = load_scalar(delta_const);
        context->fixed = upload_scalars(fixed, fixed_cols * len, gpu);
        context->fixed_lagrange = upload_scalars(
            fixed_lagrange,
            fixed_cols * input_len,
            gpu);
        context->l0 = upload_scalars(l0, len, gpu);
        context->l_last = upload_scalars(l_last, len, gpu);
        context->l_active = upload_scalars(l_active, len, gpu);
        context->permutation_columns =
            upload_array(permutation_columns, permutation_columns_len, gpu);
        context->permutation_pk_cosets =
            upload_scalars(permutation_pk_cosets, permutation_pk_cosets_len * len, gpu);
        context->permutation_pk_lagrange = upload_scalars(
            permutation_pk_lagrange,
            permutation_pk_cosets_len * input_len,
            gpu);
        if (trash_count != 0) {
            context->trash_selector_fixed_indices.assign(
                trash_selector_fixed_indices,
                trash_selector_fixed_indices + trash_count);
        }
        context->custom_graph = upload_graph(custom_graph, gpu);

        context->lookup_graphs.reserve(lookup_count);
        context->lookup_input_graphs.reserve(lookup_count);
        context->lookup_table_graphs.reserve(lookup_count);
        for (size_t i = 0; i < lookup_count; ++i) {
            context->lookup_graphs.emplace_back(upload_graph(lookup_graphs[i], gpu));
            context->lookup_input_graphs.emplace_back(upload_graph(lookup_input_graphs[i], gpu));
            context->lookup_table_graphs.emplace_back(upload_graph(lookup_table_graphs[i], gpu));
        }
        context->trash_graphs.reserve(trash_count);
        for (size_t i = 0; i < trash_count; ++i) {
            context->trash_graphs.emplace_back(upload_graph(trash_graphs[i], gpu));
        }

        dev_ptr_t<scalar_t> omega_factors{len, gpu};
        context->omega_powers = std::make_unique<dev_ptr_t<scalar_t>>(len, gpu);
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        omega_power_factors_kernel<<<grid, block, 0, gpu>>>(
            &omega_factors[0],
            context->omega,
            len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(
            &(*context->omega_powers)[0],
            &omega_factors[0],
            len,
            gpu);
        CUDA_OK(cudaGetLastError());

        dev_ptr_t<scalar_t> trace_omega_factors{input_len, gpu};
        context->trace_omega_powers =
            std::make_unique<dev_ptr_t<scalar_t>>(input_len, gpu);
        const size_t trace_grid = (input_len + block - 1) / block;
        omega_power_factors_kernel<<<trace_grid, block, 0, gpu>>>(
            &trace_omega_factors[0],
            load_scalar(trace_omega),
            input_len);
        CUDA_OK(cudaGetLastError());
        prefix_op<Multiply<scalar_t>>(
            &(*context->trace_omega_powers)[0],
            &trace_omega_factors[0],
            input_len,
            gpu);
        CUDA_OK(cudaGetLastError());

        gpu.sync();
        *out = context.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" void midnight_sppark_bls12_381_numerator_context_free(void *context)
{
    delete reinterpret_cast<numerator_context_t *>(context);
}

extern "C" RustError midnight_sppark_bls12_381_trace_context_trash(
    void **out,
    const void *raw_context,
    const void *raw_advice,
    size_t advice_cols,
    const void *raw_instance,
    size_t instance_cols,
    const RawFr *challenges,
    size_t challenge_len,
    RawFr trash_challenge)
{
    if (out == nullptr || raw_context == nullptr
        || (challenge_len != 0 && challenges == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid resident trash arguments"};
    }
    *out = nullptr;
    const auto *context = reinterpret_cast<const numerator_context_t *>(raw_context);
    const size_t len = context->input_len;
    if (context->trash_count == 0
        || !device_vec_has_len(raw_advice, advice_cols * len)
        || !device_vec_has_len(raw_instance, instance_cols * len)) {
        return RustError{cudaErrorInvalidValue, "resident trash vector length mismatch"};
    }

    const auto *advice = reinterpret_cast<const scalar_device_vec_t *>(raw_advice);
    const auto *instance = reinterpret_cast<const scalar_device_vec_t *>(raw_instance);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_challenges = upload_scalars(challenges, challenge_len, gpu);
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = context->trash_count * len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        const scalar_t zero = fr_zero();
        for (size_t index = 0; index < context->trash_count; ++index) {
            launch_numerator_custom(
                device_values(device.get()) + (index * len),
                len,
                context->trash_graphs[index].graph,
                device_ptr(context->fixed_lagrange),
                context->fixed_cols,
                device_values(advice),
                advice_cols,
                device_values(instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                zero,
                zero,
                zero,
                load_scalar(trash_challenge),
                zero,
                1,
                grid,
                block,
                gpu);
        }
        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_trace_context_permutation(
    void **out,
    const void *raw_context,
    const void *raw_advice,
    size_t advice_cols,
    const void *raw_instance,
    size_t instance_cols,
    RawFr beta,
    RawFr gamma,
    const RawFr *delta_powers,
    size_t delta_powers_len,
    const RawFr *blind_values,
    size_t blind_values_len,
    size_t blinding_factors)
{
    if (out == nullptr || raw_context == nullptr || delta_powers == nullptr) {
        return RustError{cudaErrorInvalidValue, "invalid resident permutation arguments"};
    }
    *out = nullptr;
    const auto *context = reinterpret_cast<const numerator_context_t *>(raw_context);
    const size_t len = context->input_len;
    const size_t column_count = context->permutation_columns_len;
    const size_t chunk_len = context->chunk_len;
    if (column_count == 0 || chunk_len == 0
        || delta_powers_len != column_count
        || !device_vec_has_len(raw_advice, advice_cols * len)
        || !device_vec_has_len(raw_instance, instance_cols * len)) {
        return RustError{cudaErrorInvalidValue, "resident permutation vector length mismatch"};
    }
    const size_t set_count = (column_count + chunk_len - 1) / chunk_len;
    if (blinding_factors >= len
        || blind_values_len != set_count * blinding_factors
        || (blind_values_len != 0 && blind_values == nullptr)) {
        return RustError{cudaErrorInvalidValue, "resident permutation blind mismatch"};
    }

    const auto *advice = reinterpret_cast<const scalar_device_vec_t *>(raw_advice);
    const auto *instance = reinterpret_cast<const scalar_device_vec_t *>(raw_instance);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_delta_powers = upload_scalars(delta_powers, delta_powers_len, gpu);
        auto d_blinds = upload_scalars(blind_values, blind_values_len, gpu);
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = set_count * len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(device->len, gpu);
        dev_ptr_t<scalar_t> modified{len, gpu};
        dev_ptr_t<scalar_t> prefix{len, gpu};
        dev_ptr_t<scalar_t> initial{1, gpu};
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_set_one_kernel<<<1, 1, 0, gpu>>>(&initial[0]);
        CUDA_OK(cudaGetLastError());

        for (size_t set_idx = 0; set_idx < set_count; ++set_idx) {
            const size_t column_offset = set_idx * chunk_len;
            const size_t set_columns = std::min(chunk_len, column_count - column_offset);
            scalar_t *set_out = device_values(device.get()) + (set_idx * len);
            fr_permutation_fraction_sources_kernel<<<grid, block, 0, gpu>>>(
                &modified[0],
                device_ptr(context->fixed_lagrange),
                context->fixed_cols,
                device_values(advice),
                advice_cols,
                device_values(instance),
                instance_cols,
                device_ptr(context->permutation_columns),
                device_ptr(context->permutation_pk_lagrange),
                device_ptr(d_delta_powers),
                device_ptr(context->trace_omega_powers),
                load_scalar(beta),
                load_scalar(gamma),
                len,
                column_offset,
                set_columns);
            CUDA_OK(cudaGetLastError());
            prefix_op<Multiply<scalar_t>>(&prefix[0], &modified[0], len, gpu);
            CUDA_OK(cudaGetLastError());
            fr_shifted_prefix_product_device_initial_kernel<<<grid, block, 0, gpu>>>(
                set_out,
                &prefix[0],
                &initial[0],
                len);
            CUDA_OK(cudaGetLastError());
            fr_copy_scalar_kernel<<<1, 1, 0, gpu>>>(
                &initial[0],
                set_out + (len - (blinding_factors + 1)));
            CUDA_OK(cudaGetLastError());
            if (blinding_factors != 0) {
                const size_t blind_grid = (blinding_factors + block - 1) / block;
                fr_blind_tail_kernel<<<blind_grid, block, 0, gpu>>>(
                    set_out,
                    device_ptr(d_blinds) + (set_idx * blinding_factors),
                    len,
                    blinding_factors);
                CUDA_OK(cudaGetLastError());
            }
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_trace_context_lookup_permute(
    void **out,
    const void *raw_context,
    const void *raw_advice,
    size_t advice_cols,
    const void *raw_instance,
    size_t instance_cols,
    const RawFr *challenges,
    size_t challenge_len,
    RawFr theta,
    size_t usable_rows,
    const RawFr *input_blinds,
    const RawFr *table_blinds,
    size_t blind_count)
{
    if (out == nullptr || raw_context == nullptr
        || (challenge_len != 0 && challenges == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid resident lookup permutation arguments"};
    }
    *out = nullptr;
    const auto *context = reinterpret_cast<const numerator_context_t *>(raw_context);
    const size_t len = context->input_len;
    const size_t lookup_count = context->lookup_count;
    if (lookup_count == 0 || usable_rows == 0 || usable_rows > len
        || blind_count != len - usable_rows
        || context->lookup_input_graphs.size() != lookup_count
        || context->lookup_table_graphs.size() != lookup_count
        || !device_vec_has_len(raw_advice, advice_cols * len)
        || !device_vec_has_len(raw_instance, instance_cols * len)
        || (blind_count != 0 && (input_blinds == nullptr || table_blinds == nullptr))) {
        return RustError{cudaErrorInvalidValue, "resident lookup permutation shape mismatch"};
    }
    if (lookup_count > SIZE_MAX / len || lookup_count * len > SIZE_MAX / 4) {
        return RustError{cudaErrorInvalidValue, "resident lookup permutation size overflow"};
    }

    const auto *advice = reinterpret_cast<const scalar_device_vec_t *>(raw_advice);
    const auto *instance = reinterpret_cast<const scalar_device_vec_t *>(raw_instance);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto d_challenges = upload_scalars(challenges, challenge_len, gpu);
        const size_t group_len = lookup_count * len;
        auto trace = std::make_unique<scalar_device_vec_t>();
        trace->len = group_len * 4;
        trace->values = std::make_unique<dev_ptr_t<scalar_t>>(trace->len, gpu);
        scalar_t *trace_values = device_values(trace.get());
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        const scalar_t zero = fr_zero();
        const scalar_t typed_theta = load_scalar(theta);
        for (size_t index = 0; index < lookup_count; ++index) {
            launch_numerator_custom(
                trace_values + (index * len),
                len,
                context->lookup_input_graphs[index].graph,
                device_ptr(context->fixed_lagrange),
                context->fixed_cols,
                device_values(advice),
                advice_cols,
                device_values(instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                zero,
                zero,
                typed_theta,
                zero,
                zero,
                1,
                grid,
                block,
                gpu);
            launch_numerator_custom(
                trace_values + group_len + (index * len),
                len,
                context->lookup_table_graphs[index].graph,
                device_ptr(context->fixed_lagrange),
                context->fixed_cols,
                device_values(advice),
                advice_cols,
                device_values(instance),
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                zero,
                zero,
                typed_theta,
                zero,
                zero,
                1,
                grid,
                block,
                gpu);
        }

        RustError status = populate_lookup_permutation(
            trace.get(),
            len,
            lookup_count,
            usable_rows,
            reinterpret_cast<const scalar_t *>(input_blinds),
            reinterpret_cast<const scalar_t *>(table_blinds),
            blind_count,
            gpu);
        if (status.code != 0) {
            return status;
        }
        *out = trace.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_numerator_context_eval(
    void **out,
    const void *raw_context,
    const void *raw_columns,
    size_t total_columns,
    size_t advice_offset,
    size_t advice_cols,
    size_t instance_offset,
    size_t instance_cols,
    const RawFr *challenges,
    size_t challenge_len,
    RawFr beta,
    RawFr gamma,
    RawFr theta,
    RawFr trash_challenge,
    RawFr y,
    size_t permutation_offset,
    size_t permutation_sets,
    size_t lookup_product_offset,
    size_t lookup_input_offset,
    size_t lookup_table_offset,
    size_t lookup_count,
    size_t trash_offset,
    size_t trash_count)
{
    if (out == nullptr || raw_context == nullptr || (challenge_len != 0 && challenges == nullptr)) {
        return RustError{cudaErrorInvalidValue, "invalid numerator evaluation arguments"};
    }
    *out = nullptr;
    const auto *context = reinterpret_cast<const numerator_context_t *>(raw_context);
    const size_t len = context->len;
    const auto in_bounds = [total_columns](size_t offset, size_t count) {
        return offset <= total_columns && count <= total_columns - offset;
    };
    if (lookup_count != context->lookup_count || trash_count != context->trash_count
        || total_columns > SIZE_MAX / len
        || !device_vec_has_len(raw_columns, total_columns * len)
        || !in_bounds(advice_offset, advice_cols)
        || !in_bounds(instance_offset, instance_cols)
        || !in_bounds(permutation_offset, permutation_sets)
        || !in_bounds(lookup_product_offset, lookup_count)
        || !in_bounds(lookup_input_offset, lookup_count)
        || !in_bounds(lookup_table_offset, lookup_count)
        || !in_bounds(trash_offset, trash_count)) {
        return RustError{cudaErrorInvalidValue, "resident numerator vector length mismatch"};
    }

    const auto *columns = reinterpret_cast<const scalar_device_vec_t *>(raw_columns);
    const auto *base = device_values(columns);
    const auto column_view = [base, len](size_t offset, size_t count) {
        return count == 0 ? static_cast<const scalar_t *>(nullptr) : base + (offset * len);
    };
    const auto *advice = column_view(advice_offset, advice_cols);
    const auto *instance = column_view(instance_offset, instance_cols);
    const auto *permutation_products = column_view(permutation_offset, permutation_sets);
    const auto *lookup_products = column_view(lookup_product_offset, lookup_count);
    const auto *lookup_inputs = column_view(lookup_input_offset, lookup_count);
    const auto *lookup_tables = column_view(lookup_table_offset, lookup_count);
    const auto *trash = column_view(trash_offset, trash_count);

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        auto device = std::make_unique<scalar_device_vec_t>();
        device->len = len;
        device->values = std::make_unique<dev_ptr_t<scalar_t>>(len, gpu);
        auto d_challenges = upload_scalars(challenges, challenge_len, gpu);

        auto typed_beta = load_scalar(beta);
        auto typed_gamma = load_scalar(gamma);
        auto typed_theta = load_scalar(theta);
        auto typed_trash_challenge = load_scalar(trash_challenge);
        auto typed_y = load_scalar(y);
        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;

        launch_numerator_custom(
            device_values(device.get()),
            len,
            context->custom_graph.graph,
            device_ptr(context->fixed),
            context->fixed_cols,
            advice,
            advice_cols,
            instance,
            instance_cols,
            device_ptr(d_challenges),
            challenge_len,
            typed_beta,
            typed_gamma,
            typed_theta,
            typed_trash_challenge,
            typed_y,
            context->rot_scale,
            grid,
            block,
            gpu);

        if (permutation_sets != 0) {
            numerator_permutation_kernel<<<grid, block, 0, gpu>>>(
                device_values(device.get()),
                len,
                device_ptr(context->fixed),
                context->fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                device_ptr(context->l0),
                device_ptr(context->l_last),
                device_ptr(context->l_active),
                permutation_products,
                permutation_sets,
                device_ptr(context->permutation_columns),
                context->permutation_columns_len,
                device_ptr(context->permutation_pk_cosets),
                typed_beta,
                typed_gamma,
                typed_y,
                device_ptr(context->omega_powers),
                typed_beta * context->g_coset,
                context->delta_const,
                context->rot_scale,
                context->last_rotation,
                context->chunk_len);
            CUDA_OK(cudaGetLastError());
        }

        for (size_t i = 0; i < lookup_count; ++i) {
            launch_numerator_lookup(
                device_values(device.get()),
                len,
                context->lookup_graphs[i].graph,
                device_ptr(context->fixed),
                context->fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                device_ptr(context->l0),
                device_ptr(context->l_last),
                device_ptr(context->l_active),
                lookup_products + (i * len),
                lookup_inputs + (i * len),
                lookup_tables + (i * len),
                typed_beta,
                typed_gamma,
                typed_theta,
                typed_trash_challenge,
                typed_y,
                context->rot_scale,
                grid,
                block,
                gpu);
        }

        for (size_t i = 0; i < trash_count; ++i) {
            launch_numerator_trash(
                device_values(device.get()),
                len,
                context->trash_graphs[i].graph,
                device_ptr(context->fixed),
                context->fixed_cols,
                advice,
                advice_cols,
                instance,
                instance_cols,
                device_ptr(d_challenges),
                challenge_len,
                trash + (i * len),
                context->trash_selector_fixed_indices[i],
                typed_beta,
                typed_gamma,
                typed_theta,
                typed_trash_challenge,
                typed_y,
                context->rot_scale,
                grid,
                block,
                gpu);
        }

        gpu.sync();
        *out = device.release();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc&) {
        gpu.sync();
        return RustError{cudaErrorMemoryAllocation, "host allocation failed"};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm(
    RawG1Projective *out,
    const RawG1Affine *points,
    const RawFr *scalars,
    size_t len)
{
    std::lock_guard<std::mutex> lock(msm_dispatch_mutex());
    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *typed_points = reinterpret_cast<const affine_t *>(points);
    auto *typed_scalars = reinterpret_cast<const scalar_t *>(scalars);

    try {
        const gpu_t& gpu = active_gpu();
        const RawMsmConfig config = default_msm_config(len);
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{
            typed_points,
            len,
            sizeof(affine_t),
            -1,
            &gpu,
            active_execution_lane() * MSM_EXECUTION_SID_STRIDE,
            config.window_bits,
            config.sort_grid_divisor,
            config.sort_window_lanes,
            config.integration_blocks,
            config.reuse_event,
            config.device_reduce};
        return msm.invoke(
            *typed_out,
            static_cast<const affine_t *>(nullptr),
            len,
            typed_scalars,
            true);
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_batch(
    RawG1Projective *out,
    const RawG1Affine *points,
    const RawFr *scalars,
    size_t npoints,
    size_t nmsms)
{
    std::lock_guard<std::mutex> lock(msm_dispatch_mutex());
    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *typed_points = reinterpret_cast<const affine_t *>(points);
    auto *typed_scalars = reinterpret_cast<const scalar_t *>(scalars);

    try {
        const gpu_t& gpu = active_gpu();
        const RawMsmConfig config = default_msm_config(npoints);
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{
            typed_points,
            npoints,
            sizeof(affine_t),
            -1,
            &gpu,
            active_execution_lane() * MSM_EXECUTION_SID_STRIDE,
            config.window_bits,
            config.sort_grid_divisor,
            config.sort_window_lanes,
            config.integration_blocks,
            config.reuse_event,
            config.device_reduce};
        for (size_t i = 0; i < nmsms; ++i) {
            RustError status = msm.invoke(
                typed_out[i],
                static_cast<const affine_t *>(nullptr),
                npoints,
                typed_scalars + (i * npoints),
                true);
            if (status.code != 0) {
                return status;
            }
        }
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_create_with_config(
    void **out,
    const RawG1Affine *points,
    size_t npoints,
    RawMsmConfig config)
{
    auto *typed_points = reinterpret_cast<const affine_t *>(points);
    if (!valid_msm_config(config)) {
        *out = nullptr;
        return RustError{cudaErrorInvalidValue, "invalid BLS12-381 MSM config"};
    }

    try {
        *out = new msm_context_t{typed_points, npoints, config};
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        *out = nullptr;
        return RustError{e.code(), e.what()};
    } catch (const std::bad_alloc& e) {
        *out = nullptr;
        return RustError{-1, e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_create(
    void **out,
    const RawG1Affine *points,
    size_t npoints)
{
    return midnight_sppark_bls12_381_g1_msm_context_create_with_config(
        out,
        points,
        npoints,
        default_msm_config(npoints));
}

extern "C" void midnight_sppark_bls12_381_g1_msm_context_free(void *context)
{
    delete reinterpret_cast<msm_context_t *>(context);
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_batch(
    RawG1Projective *out,
    void *context,
    const RawFr *scalars,
    size_t npoints,
    size_t nmsms)
{
    if (context == nullptr) {
        return RustError{-1, "null BLS12-381 G1 MSM context"};
    }

    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *typed_scalars = reinterpret_cast<const scalar_t *>(scalars);
    auto *msm = reinterpret_cast<msm_context_t *>(context);

    try {
        return msm->invoke_host_batch(typed_out, typed_scalars, npoints, nmsms);
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_device_batch(
    RawG1Projective *out,
    void *context,
    const void *device_vec,
    size_t scalar_offset,
    size_t npoints,
    size_t nmsms)
{
    if (context == nullptr || device_vec == nullptr || (nmsms != 0 && out == nullptr)) {
        return RustError{-1, "invalid resident BLS12-381 G1 MSM arguments"};
    }

    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *msm = reinterpret_cast<msm_context_t *>(context);
    const auto *device = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    if (npoints != 0 && nmsms > (SIZE_MAX - scalar_offset) / npoints) {
        return RustError{cudaErrorInvalidValue, "resident MSM range overflow"};
    }
    const size_t required = scalar_offset + (npoints * nmsms);
    if (required > device->len) {
        return RustError{cudaErrorInvalidValue, "resident MSM range exceeds device vector"};
    }

    try {
        const scalar_t *base = device_values(device) + scalar_offset;
        return msm->invoke_device_batch(typed_out, base, npoints, nmsms);
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_device_ragged(
    RawG1Projective *out,
    void *context,
    const void *device_vec,
    const RawMsmSlice *slices,
    size_t count)
{
    if (context == nullptr || device_vec == nullptr
        || (count != 0 && (out == nullptr || slices == nullptr))) {
        return RustError{-1, "invalid ragged resident BLS12-381 G1 MSM arguments"};
    }
    if (count == 0)
        return RustError{cudaSuccess};

    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *msm = reinterpret_cast<msm_context_t *>(context);
    const auto *device = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    try {
        return msm->invoke_device_ragged(
            typed_out,
            device_values(device),
            device->len,
            slices,
            count);
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_g1_msm_context_device_scrambled_batch(
    RawG1Projective *out,
    void *context,
    const void *device_vec,
    size_t scalar_offset,
    size_t npoints,
    size_t nmsms,
    const void *mask_device_vec)
{
    if (context == nullptr || device_vec == nullptr || mask_device_vec == nullptr
        || (nmsms != 0 && out == nullptr)) {
        return RustError{-1, "invalid scrambled resident BLS12-381 G1 MSM arguments"};
    }

    auto *typed_out = reinterpret_cast<point_t *>(out);
    auto *msm = reinterpret_cast<msm_context_t *>(context);
    const auto *device = reinterpret_cast<const scalar_device_vec_t *>(device_vec);
    const auto *mask_device = reinterpret_cast<const scalar_device_vec_t *>(mask_device_vec);
    if (mask_device->len != npoints) {
        return RustError{cudaErrorInvalidValue, "resident MSM scramble mask length mismatch"};
    }
    if (npoints != 0 && nmsms > (SIZE_MAX - scalar_offset) / npoints) {
        return RustError{cudaErrorInvalidValue, "scrambled resident MSM range overflow"};
    }
    const size_t scalar_count = npoints * nmsms;
    const size_t required = scalar_offset + scalar_count;
    if (required > device->len) {
        return RustError{cudaErrorInvalidValue, "scrambled resident MSM range exceeds device vector"};
    }
    if (scalar_count == 0) {
        return RustError{cudaSuccess};
    }

    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> masked{scalar_count, gpu};
        const size_t block = 256;
        fr_add_mask_copy_kernel<<<(scalar_count + block - 1) / block, block, 0, gpu>>>(
            &masked[0],
            device_values(device) + scalar_offset,
            device_values(mask_device),
            npoints,
            scalar_count);
        CUDA_OK(cudaGetLastError());
        event_t masked_ready;
        gpu[0].record(masked_ready);
        return msm->invoke_device_batch(
            typed_out,
            &masked[0],
            npoints,
            nmsms,
            &masked_ready);
    } catch (const cuda_error& e) {
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_ntt(
    RawFr *values,
    uint32_t log_n,
    bool inverse)
{
    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    const gpu_t& gpu = active_gpu();
    return NTT::Base(
        gpu,
        typed_values,
        log_n,
        NTT::InputOutputOrder::NN,
        inverse ? NTT::Direction::inverse : NTT::Direction::forward,
        NTT::Type::standard);
}

extern "C" RustError midnight_sppark_bls12_381_fr_mul_assign(
    RawFr *values,
    RawFr scalar,
    size_t len)
{
    auto typed_scalar = load_scalar(scalar);
    return fr_unary_vector_op(
        values,
        len,
        [typed_scalar](scalar_t *d_values, size_t len, const gpu_t& gpu) {
            const size_t block = 256;
            const size_t grid = (len + block - 1) / block;
            fr_mul_assign_kernel<<<grid, block, 0, gpu>>>(d_values, typed_scalar, len);
            CUDA_OK(cudaGetLastError());
        });
}

extern "C" RustError midnight_sppark_bls12_381_fr_mul_assign_zeta(
    RawFr *values,
    RawFr first,
    RawFr second,
    size_t len)
{
    auto typed_first = load_scalar(first);
    auto typed_second = load_scalar(second);
    return fr_unary_vector_op(
        values,
        len,
        [typed_first, typed_second](scalar_t *d_values, size_t len, const gpu_t& gpu) {
            const size_t block = 256;
            const size_t grid = (len + block - 1) / block;
            fr_mul_assign_zeta_kernel<<<grid, block, 0, gpu>>>(
                d_values,
                typed_first,
                typed_second,
                len);
            CUDA_OK(cudaGetLastError());
        });
}

extern "C" RustError midnight_sppark_bls12_381_fr_mul_assign_repeating(
    RawFr *values,
    const RawFr *factors,
    size_t factor_len,
    size_t len)
{
    if (len == 0 || factor_len == 0) {
        return RustError{cudaSuccess};
    }

    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    auto *typed_factors = reinterpret_cast<const scalar_t *>(factors);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values{len, gpu};
        dev_ptr_t<scalar_t> d_factors{factor_len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);
        gpu.HtoD(&d_factors[0], typed_factors, factor_len);

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_mul_assign_repeating_kernel<<<grid, block, 0, gpu>>>(
            &d_values[0],
            &d_factors[0],
            factor_len,
            len);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_invert_assign(
    RawFr *values,
    size_t len)
{
    if (len == 0) {
        return RustError{cudaSuccess};
    }

    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        const size_t device_len = len + (len & 1);
        dev_ptr_t<scalar_t> d_values{device_len, gpu};
        gpu.HtoD(&d_values[0], typed_values, len);

        if (device_len != len) {
            fr_set_one_kernel<<<1, 1, 0, gpu>>>(&d_values[len]);
            CUDA_OK(cudaGetLastError());
        }

        const size_t block = 256;
        const size_t grid = (device_len + block - 1) / block;
        fr_invert_assign_kernel<<<grid, block, 0, gpu>>>(&d_values[0], device_len);
        CUDA_OK(cudaGetLastError());

        gpu.DtoH(typed_values, &d_values[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}

extern "C" RustError midnight_sppark_bls12_381_fr_sort_canonical(
    RawFr *values,
    size_t len)
{
    if (len < 2) {
        return RustError{cudaSuccess};
    }

    auto *typed_values = reinterpret_cast<scalar_t *>(values);
    const gpu_t& gpu = active_gpu();
    try {
        gpu.select();
        dev_ptr_t<scalar_t> d_values_in{len, gpu};
        dev_ptr_t<scalar_t> d_values_out{len, gpu};
        dev_ptr_t<RawFr> d_keys_in{len, gpu};
        dev_ptr_t<RawFr> d_keys_out{len, gpu};
        gpu.HtoD(&d_values_in[0], typed_values, len);

        const size_t block = 256;
        const size_t grid = (len + block - 1) / block;
        fr_canonical_sort_keys_kernel<<<grid, block, 0, gpu>>>(
            &d_keys_in[0], &d_values_in[0], len);
        CUDA_OK(cudaGetLastError());

        size_t temp_storage_bytes = 0;
        CUDA_OK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            &d_keys_in[0],
            &d_keys_out[0],
            &d_values_in[0],
            &d_values_out[0],
            len,
            RawFrCanonicalDecomposer{},
            0,
            256,
            gpu));
        dev_ptr_t<uint8_t> d_temp_storage{temp_storage_bytes, gpu};
        CUDA_OK(cub::DeviceRadixSort::SortPairs(
            &d_temp_storage[0],
            temp_storage_bytes,
            &d_keys_in[0],
            &d_keys_out[0],
            &d_values_in[0],
            &d_values_out[0],
            len,
            RawFrCanonicalDecomposer{},
            0,
            256,
            gpu));

        gpu.DtoH(typed_values, &d_values_out[0], len);
        gpu.sync();
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }
}
