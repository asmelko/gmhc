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
#include "update.cuh"

template <csize_t N>
void tune_info(kernel_info& info, size_t size, bool use_eucl, size_t dim)
{
    auto multiplier = use_eucl ? 1 : 32;
    auto block_dim = (unsigned int)((size + 31) / 32) * 32 * multiplier;
    auto grid_dim = (unsigned int)(block_dim + 1023) / 1024;
    info.block_dim = block_dim > info.block_dim ? info.block_dim : block_dim;
    info.grid_dim = grid_dim > info.grid_dim ? info.grid_dim : grid_dim;
    
    auto warps = info.block_dim / 32;
    auto icov_size = use_eucl ? 0 : (dim + 1) * dim / 2;
    csize_t shared_new = (csize_t)(icov_size + dim) * sizeof(float);
    info.shared_size = std::max(shared_new, warps * (csize_t)sizeof(neighbor_t) * N);
}

template<csize_t N>
void run_update_neighbors(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    csize_t size,
    update_data_t upd_data,
    bool use_eucl,
    kernel_info info)
{
    CUCH(cudaMemsetAsync(upd_data.update_size, 0, sizeof(csize_t), info.stream));

    update<N><<<info.grid_dim, info.block_dim, 0, info.stream>>>(
        act_neighbors, upd_data.to_update, upd_data.update_size, size, upd_data.old_a, upd_data.old_b);

    if (use_eucl)
    {
        tune_info<N>(info, size, use_eucl, data.dim);
        neighbors_u<N><<<info.grid_dim, info.block_dim, info.shared_size, info.stream>>>(data.centroids,
            data.representants, 
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
    else
        neighbors_mat_u<N><<<1, 1, 0, info.stream>>>(data.centroids,
            data.representants, 
            data.inverses,
            data.mfactors,
            act_neighbors,
            tmp_neighbors,
            upd_data.to_update,
            upd_data.update_size,
            upd_data.old_a,
            data.dim,
            size,
            info.block_dim,
            info.grid_dim);
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
    tune_info<N>(info, size, use_eucl, data.dim);

    if (use_eucl)
        neighbors_u<N><<<info.grid_dim, info.block_dim, info.shared_size, info.stream>>>(
            data.centroids, data.representants, act_neighbors, tmp_neighbors, new_idx, data.dim, size);
    else
        point_neighbors_mat<N><<<info.grid_dim, info.block_dim, info.shared_size, info.stream>>>(data.centroids,
            data.representants, 
            data.inverses,
            data.mfactors,
            act_neighbors,
            tmp_neighbors,
            data.dim,
            size,
            new_idx,
            new_idx);

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
    if (use_eucl)
    {
        tune_info<N>(info, size, use_eucl, data.dim);
        neighbors<N><<<info.grid_dim, info.block_dim, info.shared_size>>>(data.centroids, data.representants, tmp_neighbors, data.dim, size);
        reduce<N><<<info.grid_dim, info.block_dim>>>(tmp_neighbors, act_neighbors, size, info.grid_dim);
    }
    else
        neighbors_mat<N><<<1, 1>>>(data.centroids,
            data.representants, 
            data.inverses,
            data.mfactors,
            act_neighbors,
            tmp_neighbors,
            data.dim,
            size,
            info.block_dim,
            info.grid_dim);
}

template<csize_t N>
chunk_t run_neighbors_min(const neighbor_t* neighbors, csize_t size, chunk_t* result, kernel_info info)
{
    neighbor_min<N><<<1, 1024, 0, info.stream>>>(neighbors, size, result);

    chunk_t res;
    CUCH(cudaMemcpyAsync(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost, info.stream));

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
    template chunk_t run_neighbors_min<N>(const neighbor_t* neighbors, csize_t size, chunk_t* result, kernel_info info);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(4)
INIT_TEMPLATES(5)
