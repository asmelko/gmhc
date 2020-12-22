#include <device_launch_parameters.h>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

__constant__ float expected_point[MAX_DIM];

void assign_constant_storage(const float* value, csize_t size, cudaMemcpyKind kind)
{
    CUCH(cudaMemcpyToSymbol(expected_point, value, size, (size_t)0, kind));
}

__inline__ __device__ void reduce_sum_warp(float* __restrict__ cov, csize_t size, unsigned mask)
{
    for (csize_t i = 0; i < size; ++i)
    {
        float tmp = cov[i];
        for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
            tmp += __shfl_down_sync(mask, tmp, offset);
        if (threadIdx.x % warpSize == 0)
            cov[i] = tmp;
    }
}

__inline__ __device__ void reduce_sum_block(float* __restrict__ shared_mem, csize_t shared_chunks, csize_t cov_size)
{
    float* tmp_cov;
    unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < shared_chunks);
    if (threadIdx.x < shared_chunks)
    {
        tmp_cov = shared_mem + cov_size * threadIdx.x;
        reduce_sum_warp(tmp_cov, cov_size, mask);
    }

    __syncthreads();

    auto ciel_div = (shared_chunks + warpSize - 1) / warpSize;

    mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < ciel_div);
    if (threadIdx.x < ciel_div)
    {
        tmp_cov = shared_mem + cov_size * threadIdx.x * warpSize;
        reduce_sum_warp(tmp_cov, cov_size, mask);
    }
}

__inline__ __device__ void point_covariance(
    const float* __restrict__ point, csize_t dim, float* __restrict__ shared_mem)
{
    csize_t idx = 0;
    for (csize_t i = 0; i < dim; ++i)
        for (csize_t j = i; j < dim; ++j)
            atomicAdd(shared_mem + idx++, point[i] * point[j]);
}

__global__ void covariance(const float* __restrict__ points,
    csize_t dim,
    csize_t count,
    const asgn_t* __restrict__ assignments,
    float* __restrict__ cov_matrix,
    asgn_t cid,
    csize_t shared_chunks)
{
    csize_t cov_size = ((dim + 1) * dim) / 2;
    extern __shared__ float shared_mem[];
    float tmp_point[MAX_DIM];

    for (csize_t idx = threadIdx.x; idx < cov_size * shared_chunks; idx += blockDim.x)
        shared_mem[idx] = 0;

    __syncthreads();

    float* tmp_cov = shared_mem + cov_size * (threadIdx.x % shared_chunks);

    for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
        if (assignments[idx] == cid)
        {
            for (csize_t i = 0; i < dim; ++i)
                tmp_point[i] = points[idx * dim + i] - expected_point[i];

            point_covariance(tmp_point, dim, tmp_cov);
        }

    __syncthreads();

    reduce_sum_block(shared_mem, shared_chunks, cov_size);

    if (threadIdx.x == 0)
        for (csize_t i = 0; i < cov_size; ++i)
            atomicAdd(cov_matrix + i, tmp_cov[i]);
}

__global__ void finish_covariance(
    const float* __restrict__ in_cov_matrix, csize_t divisor, csize_t dim, float* __restrict__ out_cov_matrix)
{
    csize_t cov_size = ((dim + 1) * dim) / 2;

    for (csize_t idx = threadIdx.x; idx < cov_size; idx += blockDim.x)
    {
        auto coords = compute_coordinates(dim, idx);
        auto tmp = in_cov_matrix[idx] / divisor;
        out_cov_matrix[coords.x + coords.y * dim] = tmp;
        out_cov_matrix[coords.x * dim + coords.y] = tmp;
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
    __shared__ float shared[32];

    float mf = 1.f;

    if (use_cholesky && *cholesky_success == 0)
    {
        for (csize_t idx = threadIdx.x; idx < dim; idx += blockDim.x)
            mf *= powf(cholesky_decomp[idx * (dim + 1)], 2.f / dim);

        __syncthreads();

        reduce_mul_block(&mf, shared);

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
    __shared__ float shared[32];

    float icmf = 1.f;

    for (csize_t idx = threadIdx.x; idx < dim; idx += blockDim.x)
        icmf *= powf(cholesky_decomp[idx * (dim + 1)], -2.f / dim);

    __syncthreads();

    reduce_mul_block(&icmf, shared);

    if (threadIdx.x == 0)
        *dest = icmf;
}


void run_covariance(const input_t in, const asgn_t* assignments, float* out, asgn_t centroid_id, kernel_info info)
{
    csize_t cov_size = ((in.dim + 1) * in.dim) / 2;
    csize_t shared_chunks = 10000 / cov_size;

    CUCH(cudaMemset(out, 0, cov_size * sizeof(float)));
    CUCH(cudaDeviceSynchronize());
    covariance<<<info.grid_dim, info.block_dim, shared_chunks * cov_size * sizeof(float)>>>(
        in.data, in.dim, in.count, assignments, out, centroid_id, shared_chunks);
}

void run_finish_covariance(const float* in_cov_matrix, csize_t divisor, csize_t dim, float* out_cov_matrix)
{
    finish_covariance<<<1, 32>>>(in_cov_matrix, divisor, dim, out_cov_matrix);
}

void run_store_icovariance_data(
    float* icov_dest, float* mf_dest, const float* icov_src, const float mf_src, clustering::csize_t dim)
{
    store_icov_data<<<1, 32>>>(icov_dest, mf_dest, icov_src, mf_src, dim);
}

void run_transform_cov(float* matrix,
    csize_t dim,
    float weight_factor,
    bool use_cholesky,
    const float* cholesky_decomp,
    const int* cholesky_success)
{
    transform_cov<<<1, 32>>>(matrix, dim, weight_factor, use_cholesky, cholesky_decomp, cholesky_success);
}

void run_compute_store_icov_mf(float* dest, csize_t dim, const float* cholesky_decomp)
{
    compute_store_icov_mf<<<1, 32>>>(dest, dim, cholesky_decomp);
}