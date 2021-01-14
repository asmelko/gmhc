#include "device_launch_parameters.h"
#include "kernels.cuh"

using namespace clustering;

__global__ void merge_clusters(asgn_t* __restrict__ assignments,
    csize_t* __restrict__ asgn_idxs,
    csize_t* __restrict__ idxs_size,
    csize_t point_size,
    asgn_t old_A,
    asgn_t old_B,
    asgn_t new_C)
{
    for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_size; i += gridDim.x * blockDim.x)
        if (assignments[i] == old_A || assignments[i] == old_B)
        {
            auto idx = atomicAdd(idxs_size, 1U);
            assignments[i] = new_C;
            asgn_idxs[idx] = i;
        }
}

void run_merge_clusters(asgn_t* assignments,
    csize_t* assignment_idxs,
    csize_t* idxs_size,
    csize_t point_size,
    asgn_t old_A,
    asgn_t old_B,
    asgn_t new_C,
    kernel_info info)
{
    CUCH(cudaMemsetAsync(idxs_size, 0, sizeof(csize_t), info.stream));
    merge_clusters<<<info.grid_dim, info.block_dim, 0, info.stream>>>(
        assignments, assignment_idxs, idxs_size, point_size, old_A, old_B, new_C);
}
