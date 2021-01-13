#include <algorithm>
#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

template<csize_t N>
__global__ void neighbor_min(const neighbor_t* __restrict__ neighbors, csize_t count, chunk_t* __restrict__ result)
{
    static __shared__ chunk_t shared_mem[32];

    chunk_t tmp;
    tmp.min_dist = FLT_INF;

    for (csize_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        if (tmp.min_dist > neighbors[idx * N].distance)
        {
            tmp.min_dist = neighbors[idx * N].distance;
            tmp.min_j = neighbors[idx * N].idx;
            tmp.min_i = idx;
        }
    }

    tmp = reduce_min_block(tmp, shared_mem);

    if (threadIdx.x == 0)
        *result = tmp;
}

#include "neighbor_eucl.cuh"
#include "neighbor_maha.cuh"
#include "reduce.cuh"
#include "update.cuh"

template<csize_t N>
void run_update_neighbors(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    csize_t size,
    update_data_t upd_data,
    bool use_eucl,
    kernel_info info)
{
    csize_t shared_new = (data.dim + 33) * data.dim * sizeof(float);
    csize_t shared_mat = std::max(shared_new, 32 * (csize_t)sizeof(neighbor_t) * N);

    CUCH(cudaMemsetAsync(upd_data.update_size, 0, sizeof(csize_t), info.stream));

    update<N><<<info.grid_dim, info.block_dim, 0, info.stream>>>(
        act_neighbors, upd_data.to_update, upd_data.update_size, size, upd_data.old_a, upd_data.old_b);

    if (use_eucl)
        neighbors_u<N><<<info.grid_dim, info.block_dim, shared_mat, info.stream>>>(data.centroids,
            act_neighbors,
            tmp_neighbors,
            upd_data.to_update,
            upd_data.update_size,
            upd_data.old_a,
            data.dim,
            size);
    else
        neighbors_mat_u<N><<<info.grid_dim, info.block_dim, shared_mat, info.stream>>>(data.centroids,
            data.inverses,
            data.mfactors,
            act_neighbors,
            tmp_neighbors,
            upd_data.to_update,
            upd_data.update_size,
            upd_data.old_a,
            data.dim,
            size);

    reduce_u<N><<<info.grid_dim, info.block_dim, 0, info.stream>>>(
        tmp_neighbors, act_neighbors, upd_data.to_update, upd_data.update_size, info.grid_dim);
}

template<csize_t N>
void run_update_neighbors_new(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    csize_t size,
    csize_t new_idx,
    bool use_eucl,
    kernel_info info)
{
    csize_t shared_new = (data.dim + 33) * data.dim * sizeof(float);
    csize_t shared_mat = std::max(shared_new, 32 * (csize_t)sizeof(neighbor_t) * N);

    if (use_eucl)
        neighbors_u<N><<<info.grid_dim, info.block_dim, shared_mat, info.stream>>>(
            data.centroids, act_neighbors, tmp_neighbors, new_idx, data.dim, size);
    else
        neighbors_mat_u<N><<<info.grid_dim, info.block_dim, shared_mat, info.stream>>>(
            data.centroids, data.inverses, data.mfactors, act_neighbors, tmp_neighbors, new_idx, data.dim, size);

    reduce_u<N>
        <<<info.grid_dim, info.block_dim, 0, info.stream>>>(tmp_neighbors, act_neighbors, new_idx, info.grid_dim);
}

template<csize_t N>
void run_neighbors(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    csize_t size,
    bool use_eucl,
    kernel_info info)
{
    csize_t shared_new = (data.dim + 33) * data.dim * sizeof(float);
    csize_t shared_mat = std::max(shared_new, 32 * (csize_t)sizeof(neighbor_t) * N);

    if (use_eucl)
        neighbors<N><<<info.grid_dim, info.block_dim, shared_mat>>>(data.centroids, tmp_neighbors, data.dim, size);
    else
        neighbors_mat<N><<<info.grid_dim, info.block_dim, shared_mat>>>(
            data.centroids, data.inverses, data.mfactors, tmp_neighbors, data.dim, size);

    reduce<N><<<info.grid_dim, info.block_dim>>>(tmp_neighbors, act_neighbors, size, info.grid_dim);
}

template<csize_t N>
chunk_t run_neighbors_min(const neighbor_t* neighbors, csize_t size, chunk_t* result)
{
    neighbor_min<N><<<1, 1024>>>(neighbors, size, result);

    CUCH(cudaDeviceSynchronize());

    chunk_t res;
    CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (res.min_i > res.min_j)
        std::swap(res.min_i, res.min_j);

    return res;
}

#define INIT_TEMPLATES(N)                                                                                              \
    template void run_neighbors<N>(centroid_data_t data,                                                               \
        neighbor_t * tmp_neighbors,                                                                                    \
        neighbor_t * act_neighbors,                                                                                    \
        csize_t size,                                                                                                  \
        bool use_eucl,                                                                                                 \
        kernel_info info);                                                                                             \
    template void run_update_neighbors<N>(centroid_data_t data,                                                        \
        neighbor_t * tmp_neighbors,                                                                                    \
        neighbor_t * act_neighbors,                                                                                    \
        csize_t size,                                                                                                  \
        update_data_t upd_data,                                                                                        \
        bool use_eucl,                                                                                                 \
        kernel_info info);                                                                                             \
    template void run_update_neighbors_new<N>(centroid_data_t data,                                                    \
        neighbor_t * tmp_neighbors,                                                                                    \
        neighbor_t * act_neighbors,                                                                                    \
        csize_t size,                                                                                                  \
        csize_t new_idx,                                                                                               \
        bool use_eucl,                                                                                                 \
        kernel_info info);                                                                                             \
    template chunk_t run_neighbors_min<N>(const neighbor_t* neighbors, csize_t size, chunk_t* result);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)