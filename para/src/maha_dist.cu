#include <kernels.cuh>
#include <clustering.hpp>

#include <device_launch_parameters.h>



__inline__ __device__ void reduce_sum_warp(float* point, size_t N)
{
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
        for (size_t i = 0; i < N; ++i)
            point[i] += __shfl_down_sync(0xFFFFFFFF, point[i], offset);
}

__inline__ __device__ void reduce_sum_block(float* point, size_t N, float* shared_mem)
{
    reduce_sum_warp(point, N);

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        memcpy(shared_mem + warp_id * N * sizeof(float), point, N * sizeof(float));

    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize)
        memcpy(point, shared_mem + threadIdx.x * N * sizeof(float), N * sizeof(float));
    else
        memset(point, 0, N * sizeof(float));

    reduce_sum_warp(point,N);
}

__inline__ __device__ size2 compute_coordinates(size_t count_in_line, size_t plain_index)
{
    size_t y = 0;
    while (plain_index >= count_in_line)
    {
        y++;
        plain_index -= count_in_line--;
    }
    return { plain_index + y, y };
}

__global__ void maha_dist(const float* __restrict__ matrix, const float* __restrict__ vector, size_t N, float* result)
{
    extern __shared__ float shared_mem[];

    for (size_t i = threadIdx.x; i < N; i+=blockDim.x)
        shared_mem[i] = vector[i];

    float tmp_point[MAX_DIM];

    memset(tmp_point, 0, N * sizeof(float));

    __syncthreads();

    size_t count = ((N + 1) * N) / 2;

    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        auto coords = compute_coordinates(N, i);

        tmp_point[coords.y] = fmaf(matrix[i], shared_mem[coords.x], tmp_point[coords.y]);
    }

    reduce_sum_block(tmp_point, N, shared_mem + N * sizeof(float));

    if (threadIdx.x == 0)
    {
        float res = 0;
        for (size_t i = 0; i < N; ++i)
            res = fmaf(tmp_point[i], shared_mem[i], res);

        *result = sqrtf(res);
    }
}
