// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cassert>
#include <cstring>

#include <util/vec2d_t.hpp>
#include <util/slice_t.hpp>

#include "sort.cuh"
#include "batch_addition.cuh"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#ifdef __CUDA_ARCH__
// Transposed scalar_t
template<class scalar_t>
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator()(uint32_t laneid)
    {   return *reinterpret_cast<scalar_T*>(&val[0][laneid]);   }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template<class scalar_t>
__device__ __forceinline__
static uint32_t get_wval(const scalar_T<scalar_t>& scalar, uint32_t off,
                         uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i+1] << 32;

    return ret >> (off%32);
}

__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0-wval : wval;
}
#endif

template<class scalar_t>
__launch_bounds__(1024) __global__
void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
               uint32_t nwins, uint32_t wbits, bool mont = true)
{
    assert(len <= (1U<<31) && wbits < 32);

#ifdef __CUDA_ARCH__
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x*blockDim.x;

    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31-wbits); // (1U << (wbits+1)) - 1;

    auto& scalar = xchange[tid/WARP_SZ](tid%WARP_SZ);

    #pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x*blockDim.x) {
        auto s = scalars[i];

#if 0
        s.from();
        if (!mont) s.to();
#else
        if (mont) s.from();
#endif

        // clear the most significant bit
        uint32_t msb = s.abs();
        msb <<= 31;

        scalar = s;

        #pragma unroll 1
        for (uint32_t bit0 = nwins*wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval) wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval) wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
# if __CUDA_ARCH__-0 >= 800
#  define LARGE_L1_CODE_CACHE 1
#  define ACCUMULATE_NTHREADS 384
# else
#  define LARGE_L1_CODE_CACHE 0
#  define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
# endif
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0
# error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8
#elif MSM_NSTREAMS<2
# error "invalid MSM_NSTREAMS"
#endif

template<class bucket_t,
         class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                const vec2d_t<uint32_t> histogram, uint32_t *current_ptr)
{
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits};
    const affine_h* points = points_;

    uint32_t& current = *current_ptr;
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
#if 1
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x/degree);
    __syncthreads();
    x = xchg + threadIdx.x/degree;
#else
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
#endif

    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t* h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            const uint32_t* digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31);
                else
                    bucket.uadd(p, digit >> 31);
            }

            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
{
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    assert((blockDim.x & (blockDim.x-1)) == 0 && wbits-1 > Nthrbits);

    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};
    extern __shared__ uint4 scratch_[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    auto* row = &buckets[bid][0];
    uint32_t i = 1U << (wbits-1-Nthrbits);
    row += tid * i;

    uint32_t mask = 0;
    if ((bid+1)*wbits > nbits) {
        uint32_t lsbits = nbits - bid*wbits;
        mask = (1U << (wbits-lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }

    bucket_t p;

    #pragma unroll 1
    while (i--) {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
        #pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2*tid] = p;
    buckets[bid][2*tid+1] = acc;
}

/* A window is the weighted sum of its Pippenger buckets. Compute that sum
 * directly with one block per window instead of serially combining the 256
 * partial pairs emitted by integrate(). Top-window buckets are spaced by the
 * number of unused scalar bits, hence the right shift in |weight|. */
template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(MSM_NTHREADS) __global__
void integrate_weighted_device(bucket_h rows_[], const bucket_h buckets_[],
                               uint32_t nwins, uint32_t wbits,
                               uint32_t nbits)
{
    extern __shared__ uint4 scratch_[];
    bucket_h* partials = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x;
    const uint32_t row_index = blockIdx.x;
    const uint32_t row_stride = 1U << (wbits - 1);
    const uint32_t used_bits = row_index + 1 == nwins
                             ? nbits - row_index * wbits
                             : wbits;
    const uint32_t shift = wbits - used_bits;
    const bucket_h* row = buckets_ + row_index * row_stride;

    bucket_t sum;
    sum.inf();
    for (uint32_t index = tid; index < row_stride; index += blockDim.x) {
        uint32_t weight = (index >> shift) + 1;
        bucket_t point = row[index];
        while (weight != 0) {
            if (weight & 1)
                sum.add(point);
            weight >>= 1;
            if (weight != 0)
                point.add(point);
        }
    }
    partials[tid] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (tid < stride) {
            bucket_t left = partials[tid];
            left.add((bucket_t)partials[tid + stride]);
            partials[tid] = left;
        }
        __syncthreads();
    }
    if (tid == 0)
        rows_[row_index] = partials[0];
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(128) __global__
void integrate_weighted_partials_device(bucket_h partials_[],
                                        const bucket_h buckets_[],
                                        uint32_t nwins, uint32_t wbits,
                                        uint32_t nbits,
                                        uint32_t producer_blocks)
{
    assert(blockDim.x * producer_blocks == MSM_NTHREADS);
    const uint32_t row_index = blockIdx.x;
    const uint32_t logical_tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t row_stride = 1U << (wbits - 1);
    const uint32_t used_bits = row_index + 1 == nwins
                             ? nbits - row_index * wbits
                             : wbits;
    const uint32_t shift = wbits - used_bits;
    const bucket_h* row = buckets_ + row_index * row_stride;

    bucket_t sum;
    sum.inf();
    for (uint32_t index = logical_tid;
         index < row_stride;
         index += MSM_NTHREADS) {
        uint32_t weight = (index >> shift) + 1;
        bucket_t point = row[index];
        while (weight != 0) {
            if (weight & 1)
                sum.add(point);
            weight >>= 1;
            if (weight != 0)
                point.add(point);
        }
    }
    partials_[row_index * MSM_NTHREADS + logical_tid] = sum;
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(MSM_NTHREADS) __global__
void integrate_weighted_rows_device(bucket_h rows_[],
                                    const bucket_h partials_[],
                                    uint32_t nwins)
{
    extern __shared__ uint4 scratch_[];
    bucket_h* shared = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x;
    const uint32_t row_index = blockIdx.x;

    shared[tid] = partials_[row_index * MSM_NTHREADS + tid];
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (tid < stride) {
            bucket_t left = shared[tid];
            left.add((bucket_t)shared[tid + stride]);
            shared[tid] = left;
        }
        __syncthreads();
    }
    if (tid == 0)
        rows_[row_index] = shared[0];
}

/* Fold completed windows and the batch-addition partials into one XYZZ point.
 * Only this point crosses back to the host. */
template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(WARP_SZ) __global__
void collect_device(bucket_h out_[], const bucket_h rows_[],
                    const bucket_h ones_[], uint32_t nwins,
                    uint32_t wbits, uint32_t ones_count)
{
    extern __shared__ uint4 scratch_[];
    bucket_h* one_partials = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x;

    bucket_t one;
    one.inf();
    for (uint32_t i = tid; i < ones_count; i += blockDim.x)
        one.add((bucket_t)ones_[i]);
    one_partials[tid] = one;
    __syncthreads();

    if (tid == 0) {
        bucket_t out;
        out.inf();
        for (uint32_t y = nwins; y-- > 0;) {
            out.add((bucket_t)rows_[y]);
            if (y != 0) {
                #pragma unroll 1
                for (uint32_t bit = 0; bit < wbits; ++bit)
                    out.add(out);
            }
        }
        for (uint32_t i = 0; i < blockDim.x; ++i)
            out.add((bucket_t)one_partials[i]);
        out_[0] = out;
    }
}

#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__
void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                           uint32_t nwins, uint32_t wbits,
                                           /*const*/ affine_t::mem_t points_[],
                                           const vec2d_t<uint32_t> digits,
                                           const vec2d_t<uint32_t> histogram,
                                           uint32_t *current_ptr);
template __global__
void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                              const affine_t::mem_t points[], size_t npoints,
                              const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                         uint32_t wbits, uint32_t nbits);
template __global__
void integrate_weighted_device<bucket_t>(bucket_t::mem_t rows_[],
                                         const bucket_t::mem_t buckets_[],
                                         uint32_t nwins, uint32_t wbits,
                                         uint32_t nbits);
template __global__
void integrate_weighted_partials_device<bucket_t>(
    bucket_t::mem_t partials_[], const bucket_t::mem_t buckets_[],
    uint32_t nwins, uint32_t wbits, uint32_t nbits,
    uint32_t producer_blocks);
template __global__
void integrate_weighted_rows_device<bucket_t>(
    bucket_t::mem_t rows_[], const bucket_t::mem_t partials_[],
    uint32_t nwins);
template __global__
void collect_device<bucket_t>(bucket_t::mem_t out_[],
                              const bucket_t::mem_t rows_[],
                              const bucket_t::mem_t ones_[],
                              uint32_t nwins, uint32_t wbits,
                              uint32_t ones_count);
template __global__
void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                         size_t len, uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>
#include <memory>
#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template<class scalar_t, class affine_h>
__launch_bounds__(256) __global__ void compact_nonzero_msm_inputs_stable(
    scalar_t compact_scalars[], affine_h compact_points[],
    const scalar_t scalars[], const affine_h points[], size_t len,
    size_t capacity)
{
    __shared__ uint32_t scan[256];
    __shared__ uint32_t output_base;
    const uint32_t lane = threadIdx.x;
    if (lane == 0)
        output_base = 0;
    __syncthreads();

    for (size_t tile = 0; tile < len; tile += blockDim.x) {
        const size_t idx = tile + lane;
        const bool selected = idx < len && !scalars[idx].is_zero();
        scan[lane] = selected ? 1 : 0;
        __syncthreads();

        for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
            const uint32_t add = lane >= offset ? scan[lane - offset] : 0;
            __syncthreads();
            if (lane >= offset)
                scan[lane] += add;
            __syncthreads();
        }

        if (selected) {
            const size_t output = output_base + scan[lane] - 1;
            if (output < capacity) {
                compact_scalars[output] = scalars[idx];
                compact_points[output] = points[idx];
            }
        }
        __syncthreads();
        if (lane == 0)
            output_base += scan[blockDim.x - 1];
        __syncthreads();
    }
}

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = class affine_t::mem_t,
         class bucket_h = class bucket_t::mem_t>
class msm_t {
    const gpu_t& gpu;
    size_t npoints;
    uint32_t wbits, nwins, sort_grid_divisor, sort_window_lanes,
             integration_blocks, reuse_event, device_reduce;
    bucket_h *d_buckets;
    bucket_h *d_rows;
    bucket_h *d_result;
    bucket_h *d_weighted_partials;
    affine_h *d_points;
    scalar_t *d_scalars;
    uint32_t *d_accumulate_cursors;
    vec2d_t<uint32_t> d_hist;

    template<typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

    event_t persistent_event;

    constexpr static int lg2(size_t n)
    {   int ret=0; while (n>>=1) ret++; return ret;   }

public:
    msm_t(const affine_t points[], size_t np,
          size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1,
          const gpu_t* gpu_override = nullptr,
          uint32_t accumulate_sid_base_ = 0,
          uint32_t configured_window_bits = 0,
          uint32_t configured_sort_grid_divisor = 3,
          uint32_t configured_sort_window_lanes = 2,
          uint32_t configured_integration_blocks = 1,
          uint32_t configured_reuse_event = 0,
          uint32_t configured_device_reduce = 0)
        : gpu(gpu_override ? *gpu_override : select_gpu(device_id)),
          sort_grid_divisor(configured_sort_grid_divisor),
          sort_window_lanes(configured_sort_window_lanes),
          integration_blocks(configured_integration_blocks),
          reuse_event(configured_reuse_event),
          device_reduce(configured_device_reduce), d_rows(nullptr),
          d_result(nullptr), d_weighted_partials(nullptr), d_points(nullptr),
          d_scalars(nullptr), d_accumulate_cursors(nullptr)
    {
        (void)accumulate_sid_base_;
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
        if (npoints >= (1U << 16) && npoints <= (1U << 18))
            wbits = 12;
        if (configured_window_bits != 0)
            wbits = configured_window_bits;
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits-1);

        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t))
                         + (points ? npoints * sizeof(d_points[0]) : 0)
                         + (nwins + 1) * sizeof(d_result[0]);

        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_rows = reinterpret_cast<decltype(d_rows)>(
            reinterpret_cast<uint8_t*>(d_buckets) + d_blob_sz
            - (nwins + 1) * sizeof(d_rows[0]));
        d_result = d_rows + nwins;
        if (device_reduce != 0 && integration_blocks > 1) {
            d_weighted_partials = reinterpret_cast<bucket_h *>(gpu.Dmalloc(
                static_cast<size_t>(nwins) * MSM_NTHREADS
                    * sizeof(bucket_h)));
        }
        d_accumulate_cursors = reinterpret_cast<decltype(d_accumulate_cursors)>(
            gpu.Dmalloc(2 * sizeof(d_accumulate_cursors[0])));
        gpu.bzero(d_accumulate_cursors, 2);
        gpu.sync();
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points) {
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            gpu.HtoD(d_points, points, np, ffi_affine_sz);
            npoints = np;
        } else {
            npoints = 0;
        }

    }
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points, points.size(), ffi_affine_sz, device_id) {};
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, 0, 0, device_id) {};
    ~msm_t()
    {
        gpu.sync();
        if (d_accumulate_cursors) gpu.Dfree(d_accumulate_cursors);
        if (d_weighted_partials) gpu.Dfree(d_weighted_partials);
        if (d_buckets) gpu.Dfree(d_buckets);
    }

private:
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>&d_temps, bool mont)
    {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        uint32_t grid_size = gpu.sm_count() / sort_grid_divisor;
        if (grid_size > WARP_SZ)
            grid_size = WARP_SZ;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2*grid_size, 1024, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
#if 0
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u);
        }
        uint32_t top = scalar_t::bit_length() - wbits * win;
        gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
#else
        // Independent windows share a cooperative launch. The row count is
        // capped to the number of blocks that can occupy the device together.
        uint32_t top = scalar_t::bit_length() - wbits * (nwins-1);
        uint32_t max_rows = std::max(1u, gpu.sm_count() / grid_size);
        uint32_t configured_rows = std::min(sort_window_lanes, max_rows);
        for (uint32_t win = 0; win < nwins;) {
            uint32_t rows = std::min(configured_rows, nwins - win);
            gpu[2].launch_coop(sort, {{grid_size, rows}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, nwins-1, top-1);
            win += rows;
        }
#endif
    }

public:
    RustError invoke(point_t& out, const affine_t* points_, size_t npoints,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t),
                                   size_t point_offset = 0)
    {
        assert(this->npoints == 0
               || (point_offset <= this->npoints
                   && npoints <= this->npoints - point_offset));
        assert(points_ == nullptr || point_offset == 0);

        uint32_t lg_npoints = lg2(npoints + npoints/2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        const uint32_t ones_count = gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ;
        const bool upstream_reduce = device_reduce == 0 || device_reduce == 2;
        std::vector<result_t> res(upstream_reduce ? nwins : 0);
        std::vector<bucket_t> ones(
            upstream_reduce || device_reduce == 1 ? ones_count : 0);
        std::vector<bucket_t> device_rows(
            device_reduce == 1 || device_reduce == 2 ? nwins : 0);
        static_assert(sizeof(bucket_t) == sizeof(bucket_h),
                      "device and host bucket layouts must match");

        out.inf();
        point_t p;

        try {
            // |scalars| being nullptr means the scalars are pre-loaded to
            // |d_scalars|, otherwise allocate stride.
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
            temp_sz = stride * std::max(
                static_cast<size_t>(sort_window_lanes) * sizeof(uint2),
                temp_sz);

            // |points| being nullptr means the points are pre-loaded to
            // |d_points|, otherwise allocate double-stride.
            const char* points = reinterpret_cast<const char*>(points_);
            size_t d_point_sz = points ? (batch > 1 ? 2*stride : stride) : 0;
            d_point_sz *= sizeof(affine_h);

            size_t digits_sz = nwins * stride * sizeof(uint32_t);

            dev_ptr_t<uint8_t> d_temp{temp_sz + digits_sz + d_point_sz, gpu[2]};

            vec2d_t<uint2> d_temps{&d_temp[0], stride};
            vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};

            scalar_t* d_scalars = scalars ? (scalar_t*)&d_temp[0]
                                          : this->d_scalars;
            affine_h* d_points = points ? (affine_h*)&d_temp[temp_sz + digits_sz]
                                        : this->d_points;

            size_t d_off = points ? 0 : point_offset; // device offset
            size_t h_off = 0;   // host offset
            size_t num = stride > npoints ? npoints : stride;
            std::unique_ptr<event_t> local_event;
            event_t *event_ptr = &persistent_event;
            if (!reuse_event) {
                local_event = std::make_unique<event_t>();
                event_ptr = local_event.get();
            } else {
                CUDA_OK(cudaEventSynchronize(*event_ptr));
            }
            event_t& event = *event_ptr;

            if (scalars)
                gpu[2].HtoD(&d_scalars[d_off], &scalars[h_off], num);
            digits(&d_scalars[0], num, d_digits, d_temps, mont);
            gpu[2].record(event);

            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num,              ffi_affine_sz);

            for (uint32_t i = 0; i < batch; i++) {
                gpu[i&1].wait(event);
                if (device_reduce != 0) {
                    const size_t bucket_count =
                        (static_cast<size_t>(nwins) << (wbits - 1))
                        + ones_count;
                    gpu[i&1].bzero(d_buckets, bucket_count);
                    gpu[i&1].bzero(d_rows, nwins + 1);
                    gpu[i&1].bzero(&d_accumulate_cursors[i&1], 1);
                }

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                           0, gpu[i&1]>>>(
                    &d_buckets[nwins << (wbits-1)], &d_points[d_off], num,
                    &d_digits[0][0], d_hist[0][0]
                );
                CUDA_OK(cudaGetLastError());

                gpu[i&1].launch_coop(accumulate<bucket_t, affine_h>,
                    {gpu.sm_count(), 0},
                    d_buckets, nwins, wbits, &d_points[d_off], d_digits, d_hist,
                    &d_accumulate_cursors[i&1]
                );
                gpu[i&1].record(event);

                if (device_reduce != 0) {
                    if (integration_blocks == 1) {
                        integrate_weighted_device<bucket_t><<<
                            nwins, MSM_NTHREADS,
                            sizeof(bucket_h) * MSM_NTHREADS,
                            gpu[i&1]>>>(
                                d_rows, d_buckets, nwins, wbits,
                                scalar_t::bit_length());
                    } else {
                        const uint32_t integration_threads =
                            MSM_NTHREADS / integration_blocks;
                        integrate_weighted_partials_device<bucket_t><<<
                            dim3(nwins, integration_blocks),
                            integration_threads, 0, gpu[i&1]>>>(
                                d_weighted_partials, d_buckets, nwins, wbits,
                                scalar_t::bit_length(), integration_blocks);
                        CUDA_OK(cudaGetLastError());
                        integrate_weighted_rows_device<bucket_t><<<
                            nwins, MSM_NTHREADS,
                            sizeof(bucket_h) * MSM_NTHREADS,
                            gpu[i&1]>>>(
                                d_rows, d_weighted_partials, nwins);
                    }
                    CUDA_OK(cudaGetLastError());
                }
                if (upstream_reduce) {
                    integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                          sizeof(bucket_t)*MSM_NTHREADS/bucket_t::degree,
                                          gpu[i&1]>>>(
                        d_buckets, nwins, wbits, scalar_t::bit_length()
                    );
                    CUDA_OK(cudaGetLastError());
                }

                if (i < batch-1) {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;

                    if (scalars)
                        gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
                    gpu[2].wait(event);
                    digits(&d_scalars[scalars ? 0 : h_off], num,
                           d_digits, d_temps, mont);
                    gpu[2].record(event);

                    if (points) {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;
                        gpu[j].HtoD(&d_points[d_off], &points[h_off*ffi_affine_sz],
                                    num,              ffi_affine_sz);
                    } else {
                        d_off = point_offset + h_off;
                    }
                }

                if (device_reduce != 0) {
                    if (device_reduce == 1) {
                        gpu[i&1].DtoH(device_rows, d_rows);
                        gpu[i&1].DtoH(
                            ones, d_buckets + (nwins << (wbits-1)));
                        gpu[i&1].sync();
                        collect_device_rows(p, device_rows, sum_up(ones));
                        out.add(p);
                        continue;
                    }

                    const size_t shared_sz = WARP_SZ * sizeof(bucket_h);
                    collect_device<bucket_t><<<
                        1, WARP_SZ, shared_sz, gpu[i&1]>>>(
                            d_result, d_rows,
                            d_buckets + (nwins << (wbits-1)),
                            nwins, wbits, ones_count);
                    CUDA_OK(cudaGetLastError());

                    bucket_t device_bucket;
                    gpu[i&1].DtoH(&device_bucket, d_result, 1);
                    if (device_reduce == 2) {
                        gpu[i&1].DtoH(device_rows, d_rows);
                        gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                        gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                    }
                    gpu[i&1].sync();

                    point_t device_point = device_bucket;
                    if (device_reduce == 2) {
                        for (uint32_t y = 0; y < nwins; ++y) {
                            const uint32_t lsbits = y + 1 == nwins
                                                  ? scalar_t::bit_length() - y * wbits
                                                  : wbits;
                            point_t device_row = device_rows[y];
                            point_t host_row = integrate_row(res[y], lsbits);
                            affine_t device_row_affine = device_row;
                            affine_t host_row_affine = host_row;
                            if (std::memcmp(&device_row_affine, &host_row_affine,
                                            sizeof(affine_t)) != 0)
                                return RustError{-1, std::string("device MSM row reduction mismatch at window ") + std::to_string(y)};
                        }
                        collect(p, res, ones);
                        point_t hybrid_point;
                        collect_device_rows(hybrid_point, device_rows, sum_up(ones));
                        affine_t device_affine = device_point;
                        affine_t host_affine = p;
                        if (std::memcmp(&device_affine, &host_affine,
                                        sizeof(affine_t)) != 0)
                            return RustError{-1, "device MSM reduction mismatch"};
                        affine_t hybrid_affine = hybrid_point;
                        if (std::memcmp(&hybrid_affine, &host_affine,
                                        sizeof(affine_t)) != 0)
                            return RustError{-1, "hybrid MSM reduction mismatch"};
                    }
                    out.add(device_point);
                } else {
                    if (i > 0) {
                        collect(p, res, ones);
                        out.add(p);
                    }

                    gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                    gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                    gpu[i&1].sync();
                }
            }
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        if (device_reduce == 0) {
            collect(p, res, ones);
            out.add(p);
        }

        return RustError{cudaSuccess};
    }

#if 0
    RustError invoke(point_t& out, const affine_t* points, size_t npoints,
                                   gpu_ptr_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        d_scalars = scalars;
        return invoke(out, points, npoints, nullptr, mont, ffi_affine_sz);
    }
#else
    template<typename affine_ptr_t = const affine_h*,
             typename scalar_ptr_t = const scalar_t*>
    RustError invoke(point_t& out, affine_ptr_t points, size_t npoints,
                                   scalar_ptr_t scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t),
                                   size_t point_offset = 0)
    {
        const auto* p_ptr = &points[0];
        if (is_device_ptr<affine_ptr_t>::value) {
            d_points = (decltype(d_points))p_ptr;
            p_ptr = nullptr;
        }

        const auto* s_ptr = &scalars[0];
        if (is_device_ptr<scalar_ptr_t>::value) {
            d_scalars = const_cast<decltype(d_scalars)>(s_ptr);
            s_ptr = nullptr;
        }

        return invoke(out, reinterpret_cast<const affine_t*>(p_ptr), npoints,
                      reinterpret_cast<const scalar_t*>(s_ptr), mont,
                      ffi_affine_sz, point_offset);
    }
#endif

    RustError invoke(point_t& out, vec_t<scalar_t> scalars, bool mont = true)
    {   return invoke(out, nullptr, scalars.size(), scalars, mont);   }

    RustError invoke_sparse_device(point_t& out, size_t point_offset,
                                   const scalar_t* scalars, size_t len,
                                   size_t nonzero_len)
    {
        if (point_offset > npoints || len > npoints - point_offset
            || nonzero_len > len)
            return RustError{cudaErrorInvalidValue, "invalid sparse resident MSM range"};
        if (nonzero_len == 0) {
            out.inf();
            return RustError{cudaSuccess};
        }
        if (nonzero_len == len) {
            dev_ptr_t<scalar_t> scalar_view(const_cast<scalar_t*>(scalars), len);
            return invoke(out, static_cast<const affine_t*>(nullptr), len,
                          scalar_view, true, sizeof(affine_t), point_offset);
        }

        dev_ptr_t<scalar_t> compact_scalars{nonzero_len, gpu[2]};
        dev_ptr_t<affine_h> compact_points{nonzero_len, gpu[2]};
        compact_nonzero_msm_inputs_stable<<<1, 256, 0, gpu[2]>>>(
            &compact_scalars[0], &compact_points[0], scalars,
            d_points + point_offset, len, nonzero_len);
        CUDA_OK(cudaGetLastError());

        affine_h* saved_points = d_points;
        scalar_t* saved_scalars = d_scalars;
        RustError status = invoke(out, compact_points, nonzero_len,
                                  compact_scalars, true);
        d_points = saved_points;
        d_scalars = saved_scalars;
        if (status.code != 0 || device_reduce != 2)
            return status;

        point_t compact_result = out;
        dev_ptr_t<scalar_t> scalar_view(const_cast<scalar_t*>(scalars), len);
        status = invoke(out, static_cast<const affine_t*>(nullptr), len,
                        scalar_view, true, sizeof(affine_t), point_offset);
        if (status.code != 0)
            return status;
        affine_t compact_affine = compact_result;
        affine_t direct_affine = out;
        if (std::memcmp(&compact_affine, &direct_affine, sizeof(affine_t)) != 0)
            return RustError{-1, "sparse resident MSM compaction mismatch"};
        return status;
    }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   vec_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, const std::vector<affine_t>& points,
                                   const std::vector<scalar_t>& scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points.data(),
                           std::min(points.size(), scalars.size()),
                           scalars.data(), mont, ffi_affine_sz);
    }

private:
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);

        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    void collect_device_rows(point_t& out, const std::vector<bucket_t>& rows,
                             const bucket_t& one)
    {
        out.inf();
        for (uint32_t y = nwins; y-- > 0;) {
            out.add(point_t{rows[y]});
            if (y != 0) {
                for (uint32_t bit = 0; bit < wbits; ++bit)
                    out.dbl();
            }
        }
        out.add(point_t{one});
    }

    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = scalar_t::bit_length() - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                                       size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                                scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}
#endif
