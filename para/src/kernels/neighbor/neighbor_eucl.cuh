#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

__inline__ __device__ float euclidean_distance(
    const cluster_representants_t representants, const float* __restrict__ centroid, csize_t dim)
{
    float dist = 0;
    for (csize_t i = 0; i < representants.size; i++)
        dist += euclidean_norm(centroid, representants.cu_points + i * dim, dim);

    dist /= representants.size;

    return dist;
}

template<csize_t N>
__inline__ __device__ void point_neighbors_thread(neighbor_t* __restrict__ neighbors,
    const float* __restrict__ this_centroid,
    const cluster_representants_t this_representants,
    const float* __restrict__ curr_centroid,
    const cluster_representants_t curr_representants,
    csize_t dim,
    csize_t idx,
    bool new_idx = false)
{
    float dist = euclidean_distance(this_representants, curr_centroid, dim);
    dist += euclidean_distance(curr_representants, this_centroid, dim);
    
    if ((isinf(dist) || isnan(dist)))
        dist = FLT_MAX;
    else
        dist /= 2;

    if (new_idx)
        add_neighbor_disruptive<N>(neighbors, neighbor_t { dist, idx });
    else
        add_neighbor<N>(neighbors, neighbor_t { dist, idx });
}

template<csize_t N>
__device__ void point_neighbor(const float* __restrict__ centroids,
    const cluster_representants_t* __restrict__ representants,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    float* __restrict__ shared_mem,
    csize_t dim,
    csize_t count,
    csize_t x,
    csize_t new_idx)
{
    neighbor_t local_neighbors[N];

    for (csize_t i = 0; i < N; ++i)
        local_neighbors[i].distance = FLT_INF;

    for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
        shared_mem[i] = centroids[x * dim + i];

    __syncthreads();

    csize_t y = threadIdx.x + blockIdx.x * blockDim.x;

    if (x == new_idx)
    {
        for (; y < x; y += blockDim.x * gridDim.x)
        {
            bool special_case = x == count - 1 && y == count - 2;
            point_neighbors_thread<N>(neighbors + y * N,
                shared_mem,
                representants[x],
                centroids + y * dim,
                representants[y],
                dim,
                x,
                true && !special_case);
        }

        y += 1;
    }
    else
        y += x + 1;

    for (; y < count; y += blockDim.x * gridDim.x)
    {
        if (y == new_idx)
            continue;
        point_neighbors_thread<N>(
            local_neighbors, shared_mem, representants[x], centroids + y * dim, representants[y], dim, y);
    }

    reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + dim));

    if (threadIdx.x == 0)
        memcpy(work_neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, N * sizeof(neighbor_t));
}

template<csize_t N>
__global__ void neighbors(const float* __restrict__ centroids,
    const cluster_representants_t* __restrict__ representants,
    neighbor_t* __restrict__ neighbors,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    for (csize_t x = 0; x < count; ++x)
        point_neighbor<N>(centroids, representants, nullptr, neighbors, shared_mem, dim, count, x, 0);
}

template<csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids,
    const cluster_representants_t* __restrict__ representants,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    const csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_count,
    csize_t new_idx,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    auto update_count = *upd_count;

    for (csize_t i = 0; i < update_count; ++i)
        point_neighbor<N>(
            centroids, representants, neighbors, work_neighbors, shared_mem, dim, count, updated[i], new_idx);
}

template<csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids,
    const cluster_representants_t* __restrict__ representants,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t new_idx,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    point_neighbor<N>(centroids, representants, neighbors, work_neighbors, shared_mem, dim, count, new_idx, new_idx);
}
