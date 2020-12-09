#include <cfloat>
#include <device_launch_parameters.h>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

__global__ void set_default_asgn(asgn_t* __restrict__ asgns, csize_t size)
{
    for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        asgns[i] = i;
}

__global__ void set_default_icovs(float* __restrict__ icovs, csize_t size, csize_t point_dim)
{
    auto icov_size = ((point_dim + 1) * point_dim) / 2;
    for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size * icov_size; i += gridDim.x * blockDim.x)
    {
        auto cell_idx = i % icov_size;
        auto coords = compute_coordinates(point_dim, cell_idx);
        if (coords.x == coords.y)
            icovs[i] = 1;
    }
}

__global__ void set_unit_matrix(float* __restrict__ matrix, csize_t size)
{
    for (csize_t i = threadIdx.x; i < size * size; i += blockDim.x)
        matrix[i] = i / size == i % size ? 1 : 0;
}

__global__ void set_default_neigh(neighbor_t* neighbors, csize_t count)
{
    for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        neighbors[i].distance = FLT_INF;
}

__global__ void set_default_icov_mfs(float* __restrict__ mfs, csize_t size)
{
    for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        mfs[i] = 1;
}


void run_set_default_asgn(asgn_t* asgns, csize_t N) { set_default_asgn<<<50, 1024>>>(asgns, N); }

void run_set_unit_matrix(float* matrix, csize_t size) { set_unit_matrix<<<1, size * size>>>(matrix, size); }

void run_set_default_neigh(neighbor_t* neighbors, csize_t count, kernel_info info)
{
    set_default_neigh<<<info.grid_dim, info.block_dim>>>(neighbors, count);
}

void run_set_default_icovs(float* __restrict__ icovs, csize_t size, csize_t point_dim, kernel_info info)
{
    set_default_icovs<<<info.grid_dim, info.block_dim>>>(icovs, size, point_dim);
}

void run_set_default_icov_mfs(float* __restrict__ mfs, csize_t size, kernel_info info)
{
    set_default_icov_mfs<<<info.grid_dim, info.block_dim>>>(mfs, size);
}
