#include <kernels.cuh>

#include <device_launch_parameters.h>
#include <cfloat>

#include "common_kernels.cuh"

using namespace clustering;

__inline__ __device__ csize2 load_data(const float* points, float const * const * inverses, csize_t count, csize_t dim, float* dest, csize_t hsize, csize2 coords)
{
    csize_t up_point_offset = hsize * coords.x;
    csize_t left_point_offset = hsize * coords.y;

    auto up_ptr = points + up_point_offset * dim;
    auto left_ptr = points + left_point_offset * dim;

    csize_t up_size = up_point_offset + hsize > count ? count - up_point_offset : hsize;
    csize_t left_size = left_point_offset + hsize > count ? count - left_point_offset : hsize;

    for (csize_t i = threadIdx.x; i < up_size + left_size; i += blockDim.x)
    {
        if (i < up_size)
            memcpy(dest + i * dim, up_ptr + i * dim, dim * sizeof(float));
        else
            memcpy(dest + i * dim, left_ptr + (i - up_size) * dim, dim * sizeof(float));
    }
    
    const float** inv_dest = reinterpret_cast<const float**>(dest + (up_size + left_size) * dim);

    for (csize_t i = threadIdx.x; i < up_size + left_size; i += blockDim.x)
    {
        if (i < up_size)
            inv_dest[i] = inverses[hsize * coords.x + i];
        else
            inv_dest[i] = inverses[hsize * coords.y + (i - up_size)];
    }

    return { up_size, left_size };
}

__global__ void reduce_min(const chunk_t* input, chunk_t* output, csize_t input_size)
{
    static __shared__ chunk_t shared_mem[32];

    chunk_t min;
    min.min_dist = FLT_MAX;

    for (csize_t i = threadIdx.x; i < input_size; i += blockDim.x)
    {
        auto tmp = input[i];
        if (tmp.min_dist < min.min_dist)
            min = tmp;
    }

    min = reduce_min_block(min, shared_mem);

    if (threadIdx.x == 0)
        *output = min;
}

__inline__ __device__ chunk_t diagonal_loop(csize_t block_size, csize_t dim, float* shared_mem)
{
    chunk_t min;
    min.min_dist = FLT_MAX;

    for (csize_t i = threadIdx.x; i < (((block_size + 1) * block_size) / 2) - block_size; i += blockDim.x)
    {
        auto coords = compute_coordinates(block_size - 1, i);
        coords.x++;

        float dist;
        float** inv_mem = reinterpret_cast<float**>(shared_mem + (block_size * 2)*dim);
        if (!inv_mem[coords.x] && !inv_mem[coords.y])
            dist = euclidean_norm(shared_mem + coords.x * dim, shared_mem + (coords.y + block_size) * dim, dim);

        if (min.min_dist > dist)
        {
            min.min_dist = dist;
            min.min_i = coords.y;
            min.min_j = coords.x;
        }
    }
    return min;
}

__inline__ __device__ chunk_t non_diagonal_loop(csize2 chunk_dim, csize_t dim, float* shared_mem)
{
    chunk_t min;
    min.min_dist = FLT_MAX;

    for (csize_t i = threadIdx.x; i < chunk_dim.x * chunk_dim.y; i += blockDim.x)
    {
        auto x = i % chunk_dim.x;
        auto y = i / chunk_dim.x;

        float dist;
        float** inv_mem = reinterpret_cast<float**>(shared_mem + (chunk_dim.x + chunk_dim.y)*dim);
        if (!inv_mem[x] && !inv_mem[y])
            dist = euclidean_norm(shared_mem + x * dim, shared_mem + (y + chunk_dim.x) * dim, dim);

        if (min.min_dist > dist)
        {
            min.min_dist = dist;
            min.min_i = y;
            min.min_j = x;
        }
    }
    return min;
}

__inline__ __device__ chunk_t block_euclidean_min(const float* points, csize_t count, csize_t dim, float* shared_mem, csize_t hshsize, csize2 coords, const float* const* inverses)
{
    auto sh_sizes = load_data(points, inverses, count, dim, shared_mem, hshsize, coords);

    __syncthreads();

    chunk_t min;

    if (coords.x == coords.y)
        min = diagonal_loop(sh_sizes.x, dim, shared_mem);
    else
        min = non_diagonal_loop(sh_sizes, dim, shared_mem);

    min.min_i += coords.y * hshsize;
    min.min_j += coords.x * hshsize;

    chunk_t* tmp_res = reinterpret_cast<chunk_t*>(shared_mem);

    min = reduce_min_block(min, tmp_res);

    return min;
}

__global__ void euclidean_min(const float* points, csize_t point_count, csize_t point_dim, csize_t half_shared_size, chunk_t* res, csize_t chunks_in_line, csize_t chunk_count, const float* const* inverses)
{
    extern __shared__ float shared_mem[];

    for (csize_t i = blockIdx.x; i < chunk_count; i += gridDim.x)
    {
        auto coords = compute_coordinates(chunks_in_line, i);

        auto block_min = block_euclidean_min(points, point_count, point_dim, shared_mem, half_shared_size, coords, inverses);

        if (threadIdx.x == 0)
            res[i] = block_min;
    }
}

chunk_t run_euclidean_min(const input_t in, chunk_t* out, const float * const * inverses, kernel_info info)
{
    auto half_shared_size = info.shared_size / 2;
    auto chunks_in_line = (in.count + half_shared_size - 1) / half_shared_size;
    auto chunk_count =  ((chunks_in_line + 1) * chunks_in_line) / 2;

    euclidean_min << <info.grid_dim, info.block_dim, info.shared_size * in.dim * sizeof(float) + info.shared_size * sizeof(float*) >> > (in.data, in.count, in.dim, half_shared_size, out, chunks_in_line, chunk_count, inverses);
    reduce_min << <1, 1024 >> > (out, out, chunk_count);

    CUCH(cudaDeviceSynchronize());
    chunk_t res;
    CUCH(cudaMemcpy(&res, out, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (res.min_i > res.min_j)
        std::swap(res.min_i, res.min_j);

    return res;
}

void run_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info)
{
    auto half_shared_size = info.shared_size / 2;
    auto chunks_in_line = (in.count + half_shared_size - 1) / half_shared_size;
    auto chunk_count = ((chunks_in_line + 1) * chunks_in_line) / 2;

    euclidean_min << <info.grid_dim, info.block_dim, info.shared_size* in.dim * sizeof(float) >> > (in.data, in.count, in.dim, half_shared_size, out, chunks_in_line, chunk_count, inverses);
}

chunk_t run_reduce(const chunk_t* chunks, chunk_t* out, csize_t chunk_count, kernel_info info)
{
    reduce_min << <1, 1024 >> > (chunks, out, chunk_count);
    CUCH(cudaDeviceSynchronize());
    chunk_t res;
    CUCH(cudaMemcpy(&res, out, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    return res;
}