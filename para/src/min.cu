#include <kernels.cuh>
#include <clustering.hpp>
#include <iostream>
#include <cstdio>

#include <device_launch_parameters.h>

void cuda_check(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << " at " << file << ":" << line;
        exit(code);
    }
}

void cuBLAS_check(cublasStatus_t code, const char* file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        switch (code)
        {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_ALLOC_FAILED:
            std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_INVALID_VALUE:
            std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_MAPPING_ERROR:
            std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_EXECUTION_FAILED:
            std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << " at " << file << ":" << line;
            return;

        case CUBLAS_STATUS_INTERNAL_ERROR:
            std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << " at " << file << ":" << line;
            return;
        }
        std::cerr << "Unknown cuBLAS error" << " at " << file << ":" << line;
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

__device__ void print_min(const clustering::chunk_t* output)
{
    printf("%f %d %d\n", output->min_dist, output->min_i,output->min_j);
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

__inline__ __device__ size2 load_data(const float* points, float const * const * inverses, size_t count, size_t dim, float* dest, size_t hsize, size2 coords)
{
    size_t up_point_offset = hsize * coords.x;
    size_t left_point_offset = hsize * coords.y;

    auto up_ptr = points + up_point_offset * dim;
    auto left_ptr = points + left_point_offset * dim;

    size_t up_size = up_point_offset + hsize > count ? count - up_point_offset : hsize;
    size_t left_size = left_point_offset + hsize > count ? count - left_point_offset : hsize;

    for (size_t i = threadIdx.x; i < up_size + left_size; i += blockDim.x)
    {
        if (i < up_size)
            memcpy(dest + i * dim, up_ptr + i * dim, dim * sizeof(float));
        else
            memcpy(dest + i * dim, left_ptr + (i - up_size) * dim, dim * sizeof(float));
    }

    const float** inv_dest = reinterpret_cast<const float**>(dest + (up_size + left_size) * dim);

    for (size_t i = threadIdx.x; i < up_size + left_size; i += blockDim.x)
    {
        if (i < up_size)
            inv_dest[i] = inverses[hsize * coords.x + i];
        else
            inv_dest[i] = inverses[hsize * coords.y + (i - up_size)];
    }

    return { up_size, left_size };
}

__inline__ __device__ clustering::chunk_t reduce_min_warp(clustering::chunk_t data)
{
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        auto tmp_dist = __shfl_down_sync(0xFFFFFFFF, data.min_dist, offset);
        auto tmp_i = __shfl_down_sync(0xFFFFFFFF, data.min_i, offset);
        auto tmp_j = __shfl_down_sync(0xFFFFFFFF, data.min_j, offset);
        if (tmp_dist < data.min_dist)
        {
            data.min_dist = tmp_dist;
            data.min_i = tmp_i;
            data.min_j = tmp_j;
        }
    }
    return data;
}

__inline__ __device__ clustering::chunk_t reduce_min_block(clustering::chunk_t data, clustering::chunk_t* shared_mem)
{
    data = reduce_min_warp(data);

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        shared_mem[warp_id] = data;

    __syncthreads();

    data = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : shared_mem[0];

    data = reduce_min_warp(data);
    return data;
}

__global__ void reduce_min(const clustering::chunk_t* input, clustering::chunk_t* output, size_t input_size)
{
    static __shared__ clustering::chunk_t shared_mem[32];

    clustering::chunk_t min;
    min.min_dist = FLT_MAX;

    for (size_t i = threadIdx.x; i < input_size; i += blockDim.x)
    {
        auto tmp = input[i];
        if (tmp.min_dist < min.min_dist)
            min = tmp;
    }

    min = reduce_min_block(min, shared_mem);

    if (threadIdx.x == 0)
        *output = min;
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

__inline__ __device__ clustering::chunk_t diagonal_loop(size_t block_size, size_t dim, float* shared_mem)
{
    clustering::chunk_t min;
    min.min_dist = FLT_MAX;

    for (size_t i = threadIdx.x; i < (((block_size + 1) * block_size) / 2) - block_size; i += blockDim.x)
    {
        auto coords = compute_coordinates(block_size - 1, i);
        coords.x++;

        float dist;
        float** inv_mem = reinterpret_cast<float**>(shared_mem + (block_size * 2)*dim);
        if (!inv_mem[coords.x] && !inv_mem[coords.y])
            dist = euclidean_norm(shared_mem, coords.x, coords.y + block_size, dim);

        if (min.min_dist > dist)
        {
            min.min_dist = dist;
            min.min_i = coords.y;
            min.min_j = coords.x;
        }
    }
    return min;
}

__inline__ __device__ clustering::chunk_t non_diagonal_loop(size2 chunk_dim, size_t dim, float* shared_mem)
{
    clustering::chunk_t min;
    min.min_dist = FLT_MAX;

    for (size_t i = threadIdx.x; i < chunk_dim.x * chunk_dim.y; i += blockDim.x)
    {
        auto x = i % chunk_dim.x;
        auto y = i / chunk_dim.x;

        float dist;
        float** inv_mem = reinterpret_cast<float**>(shared_mem + (chunk_dim.x + chunk_dim.y)*dim);
        if (!inv_mem[x] && !inv_mem[y])
            dist = euclidean_norm(shared_mem, x, y + chunk_dim.x, dim);

        if (min.min_dist > dist)
        {
            min.min_dist = dist;
            min.min_i = y;
            min.min_j = x;
        }
    }
    return min;
}

__inline__ __device__ clustering::chunk_t block_euclidean_min(const float* points, size_t count, size_t dim, float* shared_mem, size_t hshsize, size2 coords, const float* const* inverses)
{
    auto sh_sizes = load_data(points, inverses, count, dim, shared_mem, hshsize, coords);

    __syncthreads();

    clustering::chunk_t min;

    if (coords.x == coords.y)
        min = diagonal_loop(sh_sizes.x, dim, shared_mem);
    else
        min = non_diagonal_loop(sh_sizes, dim, shared_mem);

    min.min_i += coords.y * hshsize;
    min.min_j += coords.x * hshsize;

    clustering::chunk_t* tmp_res = reinterpret_cast<clustering::chunk_t*>(shared_mem);

    min = reduce_min_block(min, tmp_res);

    return min;
}

__global__ void euclidean_min(const float* points, size_t point_count, size_t point_dim, size_t half_shared_size, clustering::chunk_t* res, size_t chunks_in_line, size_t chunk_count, const float* const* inverses)
{
    extern __shared__ float shared_mem[];

    for (size_t i = blockIdx.x; i < chunk_count; i += gridDim.x)
    {
        auto coords = compute_coordinates(chunks_in_line, i);

        auto block_min = block_euclidean_min(points, point_count, point_dim, shared_mem, half_shared_size, coords, inverses);

        if (threadIdx.x == 0)
            res[i] = block_min;
    }
}

void run_euclidean_min(const input_t in, clustering::chunk_t* out, const float * const * inverses, kernel_info info)
{
    auto half_shared_size = info.shared_size / 2;
    auto chunks_in_line = (in.count + half_shared_size - 1) / half_shared_size;
    auto chunk_count =  ((chunks_in_line + 1) * chunks_in_line) / 2;

    euclidean_min << <info.grid_dim, info.block_dim, info.shared_size * in.dim * sizeof(float) + info.shared_size * sizeof(float*) >> > (in.data, in.count, in.dim, half_shared_size, out, chunks_in_line, chunk_count, inverses);
    reduce_min << <1, 1024 >> > (out, out, chunk_count);
}

void run_min(const input_t in, clustering::chunk_t* out, const float* const* inverses, kernel_info info)
{
    auto half_shared_size = info.shared_size / 2;
    auto chunks_in_line = (in.count + half_shared_size - 1) / half_shared_size;
    auto chunk_count = ((chunks_in_line + 1) * chunks_in_line) / 2;

    euclidean_min << <info.grid_dim, info.block_dim, info.shared_size* in.dim * sizeof(float) >> > (in.data, in.count, in.dim, half_shared_size, out, chunks_in_line, chunk_count, inverses);
}

clustering::chunk_t run_reduce(const clustering::chunk_t* chunks, clustering::chunk_t* out, size_t chunk_count, kernel_info info)
{
    reduce_min << <1, 1024 >> > (chunks, out, chunk_count);
    CUCH(cudaDeviceSynchronize());
    clustering::chunk_t res;
    CUCH(cudaMemcpy(&res, out, sizeof(clustering::chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}