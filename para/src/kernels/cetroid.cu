#include <device_launch_parameters.h>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

__global__ void centroid(
    const input_t in, const asgn_t* __restrict__ assignments, float* __restrict__ out, asgn_t cid, csize_t cluster_size)
{
    extern __shared__ float shared_mem[];

    float tmp[MAX_DIM];

    memset(tmp, 0, in.dim * sizeof(float));

    for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < in.count; idx += gridDim.x * blockDim.x)
    {
        if (assignments[idx] == cid)
        {
            for (csize_t i = 0; i < in.dim; ++i)
                tmp[i] += in.data[idx * in.dim + i];
        }
    }

    reduce_sum_block(tmp, in.dim, shared_mem);

    if (threadIdx.x == 0)
    {
        for (csize_t i = 0; i < in.dim; ++i)
            out[blockIdx.x * in.dim + i] = tmp[i];
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

void run_centroid(
    const input_t in, const asgn_t* assignments, float* work_centroid, float* out, asgn_t cetroid_id, csize_t cluster_size, kernel_info info)
{
    centroid<<<info.grid_dim, info.block_dim, 32 * (in.dim * sizeof(float))>>>(
        in, assignments, work_centroid, cetroid_id, cluster_size);
    reduce_centroid<<<1, 32>>>(work_centroid, out, info.grid_dim, cluster_size, in.dim);
}
