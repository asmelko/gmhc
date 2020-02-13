#include <kernels.cuh>
#include <clustering.hpp>
#include <iostream>
#include <cstdio>

#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
void __syncthreads() {}
template <typename T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize) {}
#endif

void cuda_check(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << " at " << file << ":" << line;
        exit(code);
    }
}

__device__ void print_point(const float* data, size_t x, size_t dim)
{
    for (size_t i = 0; i < dim; i++)
    {
        printf("%f ", data[x * dim + i]);
    }
    printf("\n");
}

__device__ void print_min(const output_t* output)
{
    printf("%f %d %d\n", output->distance, output->pair[0],output->pair[1]);
}

__inline__ __device__ float euclidean_norm(const float* mem, size_t x, size_t y, size_t dim)
{
    float tmp_sum = 0;
    for (size_t i = 0; i < dim; ++i)
    {
        auto tmp = mem[x * dim + i] - mem[y * dim + i];
        tmp_sum += tmp * tmp;
    }
    return sqrtf(tmp_sum);
}

__inline__ __device__ void load_data(const float* points, size_t dim, size_t count, float* dest, size_t hsize)
{
    auto up_ptr = points + hsize * dim * blockIdx.x;
    auto left_ptr = points + hsize * dim * blockIdx.y; //possible optimalization PO1: points + hsize * dim * (blockIdx.y - 1) 

    for (size_t i = threadIdx.x; i < hsize * 2; i += blockDim.x)
    {
        if (i < hsize)
            memcpy(dest + i * dim, up_ptr + i * dim, dim * sizeof(float));
        else
            memcpy(dest + i * dim, left_ptr + (i - hsize) * dim, dim * sizeof(float)); //possible optimalization PO1: left_ptr + i * dim
    }
}

__inline__ __device__ void reduce_min_warp(output_t& data)
{
    std::uint64_t shfl_pair = *reinterpret_cast<std::uint64_t*>(data.pair);

    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        auto tmp_dist = __shfl_down_sync(0xFFFFFFFF, data.distance, offset);
        auto tmp_pair = __shfl_down_sync(0xFFFFFFFF, shfl_pair, offset);
        if (tmp_dist < data.distance)
        {
            data.distance = tmp_dist;
            shfl_pair = tmp_pair;
        }
    }

    *reinterpret_cast<std::uint64_t*>(data.pair) = shfl_pair;
}

__global__ void euclidean_min(const float* points, size_t dim, size_t count, size_t hshsize, output_t* res)
{
    extern __shared__ float shared_points[];

    load_data(points, dim, count, shared_points, hshsize);

    __syncthreads();

    output_t min;
    min.distance = FLT_MAX;

    for (size_t i = threadIdx.x; i < hshsize * hshsize; i += blockDim.x)
    {
        auto x = i % hshsize;
        auto y = hshsize + i / hshsize;

        float dist = euclidean_norm(shared_points, x, y, dim);

        if (min.distance > dist)
        {
            min.distance = dist;
            min.pair[0] = blockIdx.x * hshsize + x;
            min.pair[1] = blockIdx.y * hshsize + y - hshsize;
        }
    }

    reduce_min_warp(min);

    output_t* tmp_res = reinterpret_cast<output_t*>(shared_points); //requirement for the shared mem to have at least #warps in block * sizeof(output_t) bytes 

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        tmp_res[warp_id] = min;

    __syncthreads();

    min = (threadIdx.x < blockDim.x / warpSize) ? tmp_res[threadIdx.x] : tmp_res[0];

    reduce_min_warp(min);

    if (threadIdx.x == 0)
    {
        res[blockIdx.y * gridDim.x + blockIdx.x] = min;
    }
}
