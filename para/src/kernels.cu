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

struct size2
{
    size_t x, y;
};

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

__inline__ __device__ size2 load_data(const float* points, size_t count, size_t dim, float* dest, size_t hsize, uint2 coords)
{
    size_t up_point_offset = hsize * coords.x;
    size_t left_point_offset = hsize * coords.y;

    auto up_ptr = points + up_point_offset * dim;
    auto left_ptr = points + left_point_offset * dim; //possible optimalization PO1: points + hsize * dim * (blockIdx.y - 1) 

    auto up_size = up_point_offset + hsize > count ? count - up_point_offset : hsize;
    auto left_size = left_point_offset + hsize > count ? count - left_point_offset : hsize;

    for (size_t i = threadIdx.x; i < up_size + left_size; i += blockDim.x)
    {
        if (i < up_size)
            memcpy(dest + i * dim, up_ptr + i * dim, dim * sizeof(float));
        else
            memcpy(dest + i * dim, left_ptr + (i - up_size) * dim, dim * sizeof(float)); //possible optimalization PO1: left_ptr + i * dim
    }

    return { up_size, left_size };
}

__inline__ __device__ void reduce_min_warp(output_t& data)
{
    std::uint64_t shfl_pair = data.pair[0];
    shfl_pair <<= 32;
    shfl_pair += data.pair[1];

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

    data.pair[1] = (std::uint32_t)shfl_pair;
    shfl_pair >>= 32;
    data.pair[0] = (std::uint32_t)shfl_pair;
}

__inline__ __device__ void reduce_min_block(output_t& data, output_t* shared_mem)
{
    reduce_min_warp(data);

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        shared_mem[warp_id] = data;

    __syncthreads();

    data = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : shared_mem[0];

    reduce_min_warp(data);
}

__global__ void reduce_min(output_t* output, size_t output_size)
{
    static __shared__ output_t shared_mem[32];

    output_t min;
    min.distance = FLT_MAX;

    for (size_t i = threadIdx.x; i < output_size; i += blockDim.x)
    {
        auto tmp = output[i];
        if (tmp.distance < min.distance)
            min = tmp;
    }

    reduce_min_block(min, shared_mem);

    if (threadIdx.x == 0)
        output[0] = min;
}

__inline__ __device__ uint2 compute_coordinates(unsigned int count_in_line, unsigned int plain_index)
{  
    unsigned int y = 0;
    while (plain_index >= count_in_line)
    {
        y++;
        plain_index -= count_in_line--;
    }
    return make_uint2(plain_index + y, y);
}

__inline__ __device__ void diagonal_loop(size_t up_size, size_t left_size, size_t dim, float* shared_mem, output_t& min)
{
    for (size_t i = threadIdx.x; i < (((up_size + 1) * left_size) / 2) - up_size; i += blockDim.x)
    {
        auto coords = compute_coordinates(up_size - 1, i);
        coords.x++;

        float dist = euclidean_norm(shared_mem, coords.x, coords.y + up_size, dim);

        if (min.distance > dist)
        {
            min.distance = dist;
            min.pair[0] = coords.x;
            min.pair[1] = coords.y;
        }
    }
}

__inline__ __device__ void non_diagonal_loop(size_t up_size, size_t left_size, size_t dim, float* shared_mem, output_t& min) 
{
    for (size_t i = threadIdx.x; i < up_size * left_size; i += blockDim.x)
    {
        auto x = i % up_size;
        auto y = i / up_size;

        float dist = euclidean_norm(shared_mem, x, y + up_size, dim);

        if (min.distance > dist)
        {
            min.distance = dist;
            min.pair[0] = x;
            min.pair[1] = y;
        }
    }
}

__inline__ __device__ output_t block_euclidean_min(const float* points, size_t count, size_t dim, float* shared_mem, size_t hshsize, uint2 coords)
{
    auto sh_sizes = load_data(points, count, dim, shared_mem, hshsize, coords);


    __syncthreads();

    output_t min;
    min.distance = FLT_MAX;

    if (coords.x == coords.y)
        diagonal_loop(sh_sizes.x, sh_sizes.y, dim, shared_mem, min);
    else
        non_diagonal_loop(sh_sizes.x, sh_sizes.y, dim, shared_mem, min);


  

    min.pair[0] += coords.x * hshsize;
    min.pair[1] += coords.y * hshsize;

   // output_t* tmp_res = reinterpret_cast<output_t*>(shared_mem); //requirement for the shared mem to have at least #warps in block * sizeof(output_t) bytes 

    static __shared__ output_t tmp_res[32];

    reduce_min_block(min, tmp_res);

    return min;
}

__global__ void euclidean_min(const float* points, size_t point_count, size_t point_dim, size_t shared_size, output_t* res)
{
    extern __shared__ float shared_mem[];

    size_t half_shared_size = shared_size / 2;

    auto blocks_in_line = (point_count + half_shared_size - 1) / half_shared_size;
    auto block_count = ((blocks_in_line + 1) * blocks_in_line) / 2;

    for (size_t i = blockIdx.x; i < block_count; i += gridDim.x)
    {
        auto coords = compute_coordinates(blocks_in_line, i);

        auto block_min = block_euclidean_min(points, point_count, point_dim, shared_mem, half_shared_size, coords);

        if (threadIdx.x == 0)
        {
            printf("reduced %d %d\n", coords.x, coords.y);
        }

        if (threadIdx.x == 0)
            res[i] = block_min;
    }
}
