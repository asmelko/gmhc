#include <kernels.cuh>
#include <clustering.hpp>
#include <iostream>

#include <device_launch_parameters.h>
#include <cuda.h>

#define TWO_INTS_TO_LONG(X,Y) ((long)(X) << 32) | ((long)(Y) & 0xFFFFFFFFL)

void cuda_check(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << "at " << file << ":" << line;
        exit(code);
    }
}

__device__ void reduce_min_warp(output_t& data)
{
    float tmp_dist;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        tmp_dist = __shfl_down_sync(0xffffffff, data.distance, offset);
        if (tmp_dist < data.distance)
            data.pair = __shfl_down_sync(0xffffffff, data.pair, offset);
    }
}

__device__ void load_data(const float* points, size_t dim, size_t count, float* dest, size_t hsize)
{
    auto up_ptr = points + hsize * dim * blockIdx.x;
    auto left_ptr = points + hsize * dim * blockIdx.y; //possible optimalization PO1: points + hsize * dim * (blockIdx.y - 1) 

    for (size_t i = threadIdx.x; i < hsize * 2; i += blockDim.x)
    {
        if (i < hsize)
        {
            std::memcpy(dest + i * dim, up_ptr + i * dim, dim);
        }
        else
        {
            std::memcpy(dest + i * dim, left_ptr + i * dim - hsize * dim, dim); //possible optimalization PO1: left_ptr + i * dim
        }
    }
}

__global__ void euclid_max(const float* points, size_t dim, size_t count, size_t hshsize, output_t* res)
{
    extern __shared__ float shared_points[];

    load_data(points, dim, count, shared_points, hshsize);

    __syncthreads();

    output_t min;
    min.distance = -1;

    for (size_t i = threadIdx.x; i < hshsize * hshsize; i += blockDim.x)
    {
        auto x = i % hshsize;
        auto y = hshsize + i / hshsize;
        float tmp_sum = 0;
        for (size_t j = 0; j < dim; ++j)
        {
            auto tmp = shared_points[x + j] - shared_points[y + j];
            tmp_sum += tmp * tmp;
        }
        tmp_sum = sqrtf(tmp_sum);
        
        if (min.distance > tmp_sum)
        {
            min.distance = tmp_sum;
            min.pair = TWO_INTS_TO_LONG(blockIdx.x + x, blockIdx.y + y - hshsize);
        }
    }

    __syncthreads();

    reduce_min_warp(min);

    output_t* tmp_res = reinterpret_cast<output_t*>(shared_points);

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

__global__ void kernel()
{

}


void run_euclid_max()
{
    dim3 d(5, 5);
    kernel<<<d, 32>>>();
}