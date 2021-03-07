#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"
#include "reduce.cuh"

using namespace clustering;

__inline__ __device__ float maha_dist(
    const float* __restrict__ point, const float* __restrict__ matrix, float mf, csize_t size, unsigned int lane_id)
{
    float tmp_point = 0;

    auto icov_size = (size + 1) * size / 2;

    for (csize_t i = lane_id; i < icov_size; i += warpSize)
    {
        auto coords = compute_coordinates(size, i);

        tmp_point = fmaf(matrix[i], point[coords.x] * point[coords.y], tmp_point);
    }

    reduce_sum_warp(&tmp_point, 1);

    if (lane_id == 0)
    {
        if (tmp_point < 0)
            return euclidean_norm(point, size);
        return sqrtf(tmp_point / mf);
    }
    return 0;
}

template<csize_t N>
__inline__ __device__ void point_neighbors_mat_warp(neighbor_t* __restrict__ neighbors,
    const float* __restrict__ curr_centroid,
    const float* __restrict__ curr_icov,
    float curr_mf,
    const float* __restrict__ this_centroid,
    const float* __restrict__ this_icov,
    float this_mf,
    float* __restrict__ work_centroid,
    csize_t dim,
    csize_t idx,
    bool new_idx = false)
{
    float dist = 0;

    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    for (csize_t i = lane_id; i < dim; i += warpSize)
        work_centroid[warp_id * dim + i] = curr_centroid[i] - this_centroid[i];

    __syncwarp();

    dist += maha_dist(work_centroid + warp_id * dim, curr_icov, curr_mf, dim, lane_id);

    dist += maha_dist(work_centroid + warp_id * dim, this_icov, this_mf, dim, lane_id);

    if (lane_id == 0)
    {
        dist = (isinf(dist) || isnan(dist)) ? FLT_MAX : dist / 2;

        if (new_idx)
            add_neighbor_disruptive<N>(neighbors, neighbor_t { dist, idx });
        else
            add_neighbor<N>(neighbors, neighbor_t { dist, idx });
    }
}

template<csize_t N>
__global__ void point_neighbors_mat(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    const float* __restrict__ mfactors,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t dim,
    csize_t count,
    csize_t x,
    csize_t new_idx)
{
    extern __shared__ float shared_mem[];

    float* this_centroid = shared_mem;
    float* this_icov = shared_mem + dim;
    float this_mf = mfactors ? mfactors[x] : 1;
    float* work_centroid = shared_mem + dim + dim * dim;

    neighbor_t local_neighbors[N];

    for (csize_t i = 0; i < N; ++i)
        local_neighbors[i].distance = FLT_INF;

    auto icov_size = (dim + 1) * dim / 2;

    for (csize_t i = threadIdx.x; i < dim + icov_size; i += blockDim.x)
    {
        if (i < dim)
            shared_mem[i] = centroids[x * dim + i];
        else
            shared_mem[i] = inverses[x * icov_size + i - dim];
    }

    __syncthreads();

    csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (x == new_idx)
    {
        for (; idx < x * warpSize; idx += blockDim.x * gridDim.x)
        {
            auto y = idx / warpSize;
            auto curr_mf = mfactors ? mfactors[y] : 1;
            point_neighbors_mat_warp<N>(neighbors + y * N,
                centroids + y * dim,
                inverses + y * icov_size,
                curr_mf,
                this_centroid,
                this_icov,
                this_mf,
                work_centroid,
                dim,
                x,
                true && !(x == count - 1 && y == count - 2));
        }
        idx += warpSize;
    }
    else
        idx += (x + 1) * warpSize;

    for (; idx < count * warpSize; idx += blockDim.x * gridDim.x)
    {
        auto y = idx / warpSize;
        if (y == new_idx)
            continue;

        auto curr_mf = mfactors ? mfactors[y] : 1;
        point_neighbors_mat_warp<N>(local_neighbors,
            centroids + y * dim,
            inverses + y * icov_size,
            curr_mf,
            this_centroid,
            this_icov,
            this_mf,
            work_centroid,
            dim,
            y);
    }

    __syncthreads();

    reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem), false);

    __syncthreads();

    if (threadIdx.x == 0)
        memcpy(work_neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, sizeof(neighbor_t) * N);
}

template<csize_t N>
__global__ void neighbors_mat(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    const float* __restrict__ mfactors,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t dim,
    csize_t count,
    csize_t max_threads,
    csize_t max_blocks)
{
    for (csize_t x = 0; x < count; ++x)
    {
        auto work = count - x;

        auto threads = work * warpSize;
        auto blocks = (threads + max_threads - 1) / max_threads;
        threads = threads > max_threads ? max_threads : threads;
        blocks = blocks > max_blocks ? max_blocks : blocks;

        auto warps = threads / warpSize;
        csize_t shared_new = (dim + warps + 1) * dim * sizeof(float);
        auto shared_size = max(shared_new, warps * (csize_t)sizeof(neighbor_t) * N);

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        point_neighbors_mat<N><<<blocks, threads, shared_size, s>>>(
            centroids, inverses, mfactors, nullptr, work_neighbors, dim, count, x, 0);
        reduce_u<N><<<1, 32, 0, s>>>(work_neighbors, neighbors, x, blocks);

        cudaStreamDestroy(s);
    }
}

template<csize_t N>
__global__ void neighbors_mat_u(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    const float* __restrict__ mfactors,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_count,
    csize_t new_idx,
    csize_t dim,
    csize_t count,
    csize_t max_threads,
    csize_t max_blocks)
{
    auto update_count = *upd_count;

    for (csize_t i = 0; i < update_count; ++i)
    {
        auto idx = updated[i];

        auto work = count - idx;

        auto threads = work * warpSize;
        auto blocks = (threads + max_threads - 1) / max_threads;
        threads = threads > max_threads ? max_threads : threads;
        blocks = blocks > max_blocks ? max_blocks : blocks;

        auto warps = threads / warpSize;
        csize_t shared_new = (dim + warps + 1) * dim * sizeof(float);
        auto shared_size = max(shared_new, warps * (csize_t)sizeof(neighbor_t) * N);

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        point_neighbors_mat<N><<<blocks, threads, shared_size, s>>>(
            centroids, inverses, mfactors, neighbors, work_neighbors, dim, count, idx, new_idx);
        reduce_u<N><<<1, 32, 0, s>>>(work_neighbors, neighbors, idx, blocks);

        cudaStreamDestroy(s);
    }
    // printf("finished\n");
}
