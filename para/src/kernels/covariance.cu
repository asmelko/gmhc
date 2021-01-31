#include <device_launch_parameters.h>

#include <cub/block/block_reduce.cuh>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

__constant__ float expected_point[MAX_DIM];

void assign_constant_storage(const float* value, csize_t size, cudaMemcpyKind kind, cudaStream_t stream)
{
    CUCH(cudaMemcpyToSymbolAsync(expected_point, value, size, (size_t)0, kind, stream));
}

#define BUFF_SIZE 32

template<size_t DIM_X>
__global__ void covariance(const float* __restrict__ points,
    float* __restrict__ cov_matrix,
    csize_t count,
    csize_t dim)
{
    typedef cub::BlockReduce<float, DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float cov_point[BUFF_SIZE];

    csize_t cov_idx = 0;
    csize_t cov_size = ((dim + 1) * dim) / 2;

    while (cov_idx < cov_size)
    {
        auto need = cov_size - cov_idx;
        need = need > BUFF_SIZE ? BUFF_SIZE : need;
        auto end = cov_idx + need;

        memset(cov_point, 0, need * sizeof(float));

        for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
        {
            for (csize_t point_idx = cov_idx; point_idx < end; point_idx++)
            {
                auto coords = compute_coordinates(dim, point_idx);
                cov_point[point_idx - cov_idx] += (points[idx * dim + coords.x] - expected_point[coords.x])
                    * (points[idx * dim + coords.y] - expected_point[coords.y]);
            }
        }

        for (csize_t i = 0; i < need; i++)
        {
            float aggregate = BlockReduce(temp_storage).Sum(cov_point[i]);

            if (threadIdx.x == 0)
                cov_matrix[blockIdx.x * cov_size + cov_idx + i] = aggregate;
        }

        cov_idx += need;
    }
}

__global__ void finish_covariance(const float* __restrict__ in_cov_matrix,
    float* __restrict__ out_cov_matrix,
    csize_t grid_size,
    csize_t divisor,
    csize_t dim)
{
    csize_t cov_size = ((dim + 1) * dim) / 2;

    for (csize_t i = threadIdx.x; i < cov_size; i += blockDim.x)
    {
        float sum = 0;
        for (size_t j = 0; j < grid_size; j++)
        {
            sum += in_cov_matrix[j * cov_size + i];
        }
        sum /= divisor;

        auto coords = compute_coordinates(dim, i);
        out_cov_matrix[coords.x + coords.y * dim] = sum;
        out_cov_matrix[coords.x * dim + coords.y] = sum;
    }
}

__global__ void store_icov_data(float* __restrict__ icov_dest,
    float* __restrict__ mf_dest,
    const float* __restrict__ icov_src,
    const float mf_src,
    clustering::csize_t dim)
{
    csize_t cov_size = ((dim + 1) * dim) / 2;

    for (csize_t idx = threadIdx.x; idx < cov_size; idx += blockDim.x)
    {
        auto coords = compute_coordinates(dim, idx);

        if (coords.x == coords.y)
            icov_dest[idx] = icov_src[coords.x + coords.y * dim];
        else
            icov_dest[idx] = 2 * icov_src[coords.x + coords.y * dim];
    }

    if (threadIdx.x == 0 && mf_dest)
        *mf_dest = mf_src;
}

__device__ void reduce_mul_warp(float* __restrict__ point)
{
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
        *point *= __shfl_down_sync(0xFFFFFFFF, *point, offset);
}

__device__ void reduce_mul_block(float* __restrict__ point, float* __restrict__ shared_mem)
{
    reduce_mul_warp(point);

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        memcpy(shared_mem + warp_id, point, sizeof(float));

    __syncthreads();

    *point = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : 1;

    reduce_mul_warp(point);
}

__global__ void transform_cov(float* __restrict__ matrix,
    csize_t dim,
    float weight_factor,
    bool use_cholesky,
    const float* __restrict__ cholesky_decomp,
    const int* __restrict__ cholesky_success)
{
    __shared__ float shared[1];

    float mf = 1.f;

    if (use_cholesky && *cholesky_success == 0)
    {
        for (csize_t idx = threadIdx.x; idx < dim; idx += blockDim.x)
            mf *= powf(cholesky_decomp[idx * (dim + 1)], 2.f / dim);

        __syncthreads();

        reduce_mul_warp(&mf);

        if (threadIdx.x == 0)
            shared[0] = mf;

        __syncthreads();

        mf = shared[0];
    }

    csize_t cov_size = ((dim + 1) * dim) / 2;

    for (csize_t idx = threadIdx.x; idx < cov_size; idx += blockDim.x)
    {
        auto coords = compute_coordinates(dim, idx);

        matrix[coords.x + coords.y * dim] =
            weight_factor * matrix[coords.x + coords.y * dim] + (1 - weight_factor) * mf * (coords.x == coords.y);
    }
}

__global__ void compute_store_icov_mf(float* __restrict__ dest, csize_t dim, const float* __restrict__ cholesky_decomp)
{
    float icmf = 1.f;

    for (csize_t idx = threadIdx.x; idx < dim; idx += blockDim.x)
        icmf *= powf(cholesky_decomp[idx * (dim + 1)], -2.f / dim);

    __syncthreads();

    reduce_mul_warp(&icmf);

    if (threadIdx.x == 0)
        *dest = icmf;
}


void run_covariance(const float* points,
    float* work_covariance,
    float* out_covariance,
    csize_t cluster_size,
    csize_t dim,
    kernel_info info)
{
    auto block_dim = ((cluster_size + 31) / 32) * 32;
    auto grid_dim = (block_dim + 1023) / 1024;
    block_dim = block_dim > info.block_dim ? info.block_dim : block_dim;
    grid_dim = grid_dim > info.grid_dim ? info.grid_dim : grid_dim;

    if (block_dim == 32)
        covariance<32><<<grid_dim, 32, 0, info.stream>>>(points, work_covariance, cluster_size, dim);
    else if (block_dim <= 64)
        covariance<64><<<grid_dim, 64, 0, info.stream>>>(points, work_covariance, cluster_size, dim);
    else if (block_dim <= 128)
        covariance<128><<<grid_dim, 128, 0, info.stream>>>(points, work_covariance, cluster_size, dim);
    else if (block_dim <= 256)
        covariance<256><<<grid_dim, 256, 0, info.stream>>>(points, work_covariance, cluster_size, dim);
    else if (block_dim <= 512)
        covariance<512><<<grid_dim, 512, 0, info.stream>>>(points, work_covariance, cluster_size, dim);
    else
        covariance<1024><<<grid_dim, 1024, 0, info.stream>>>(points, work_covariance, cluster_size, dim);

    finish_covariance<<<1, 32, 0, info.stream>>>(work_covariance, out_covariance, grid_dim, cluster_size, dim);
}

void run_store_icovariance_data(float* icov_dest,
    float* mf_dest,
    const float* icov_src,
    const float mf_src,
    clustering::csize_t dim,
    cudaStream_t stream)
{
    store_icov_data<<<1, 32, 0, stream>>>(icov_dest, mf_dest, icov_src, mf_src, dim);
}

void run_transform_cov(float* matrix,
    csize_t dim,
    float weight_factor,
    bool use_cholesky,
    const float* cholesky_decomp,
    const int* cholesky_success,
    cudaStream_t stream)
{
    transform_cov<<<1, 32, 0, stream>>>(matrix, dim, weight_factor, use_cholesky, cholesky_decomp, cholesky_success);
}

void run_compute_store_icov_mf(float* dest, csize_t dim, const float* cholesky_decomp, cudaStream_t stream)
{
    compute_store_icov_mf<<<1, 32, 0, stream>>>(dest, dim, cholesky_decomp);
}