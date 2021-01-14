#include <device_launch_parameters.h>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

__global__ void centroid(const float* __restrict__ points,
    const clustering::csize_t* idxs,
    float* __restrict__ work_centroid,
    csize_t count,
    csize_t dim)
{
    extern __shared__ float shared_mem[];

    float tmp[MAX_DIM];

    memset(tmp, 0, dim * sizeof(float));

    for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
    {
        auto aidx = idxs[idx];
        for (csize_t i = 0; i < dim; ++i)
            tmp[i] += points[aidx * dim + i];
    }

    reduce_sum_block(tmp, dim, shared_mem);

    if (threadIdx.x == 0)
    {
        for (csize_t i = 0; i < dim; ++i)
            work_centroid[blockIdx.x * dim + i] = tmp[i];
    }
}

__global__ void reduce_centroid(const float* __restrict__ grid_centroid,
    float* __restrict__ out_centroid,
    csize_t grid_size,
    csize_t divisor,
    csize_t dim)
{
    for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float sum = 0;
        for (size_t j = 0; j < grid_size; j++)
        {
            sum += grid_centroid[j * dim + i];
        }
        out_centroid[i] = sum / divisor;
    }
}

void run_centroid(const float* points,
    const clustering::csize_t* idxs,
    float* work_centroid,
    float* out_centroid,
    csize_t cluster_size,
    csize_t dim,
    kernel_info info)
{
    auto block_dim = ((cluster_size + 31) / 32) * 32;
    auto grid_dim = (block_dim + 1023) / 1024;
    block_dim = block_dim > info.block_dim ? info.block_dim : block_dim;
    grid_dim = grid_dim > info.grid_dim ? info.grid_dim : grid_dim;

    centroid<<<grid_dim, block_dim, 32 * (dim * sizeof(float)), info.stream>>>(
        points, idxs, work_centroid, cluster_size, dim);
    reduce_centroid<<<1, 32, 0, info.stream>>>(work_centroid, out_centroid, grid_dim, cluster_size, dim);
}
