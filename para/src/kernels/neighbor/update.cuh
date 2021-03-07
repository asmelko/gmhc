#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

template<csize_t N, csize_t Storage>
__global__ void update(neighbor_t* __restrict__ neighbors_a,
    csize_t* __restrict__ updated,
    csize_t* __restrict__ work_idx,
    csize_t size,
    csize_t old_a,
    csize_t old_b)
{
    for (csize_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x)
    {
        if (idx == old_a)
            continue;

        if (idx == old_b)
        {
            csize_t store_idx = atomicAdd(work_idx, 1);

            updated[store_idx] = idx;
            continue;
        }

        neighbor_t tmp_nei[N];
        memcpy(tmp_nei, neighbors_a + idx * Storage, sizeof(neighbor_t) * N);

        csize_t last_empty = 0;

        for (csize_t i = 0; i < N; i++)
        {
            if (tmp_nei[i].distance == FLT_INF)
                break;

            if (tmp_nei[i].idx == old_a || tmp_nei[i].idx == old_b)
                tmp_nei[i].distance = FLT_INF;
            else
            {
                if (tmp_nei[i].idx == size)
                    tmp_nei[i].idx = old_b;

                tmp_nei[last_empty++] = tmp_nei[i];
            }
        }

        for (csize_t i = last_empty; i < N; i++)
            tmp_nei[i].distance = FLT_INF;

        if (tmp_nei[0].distance == FLT_INF)
        {
            csize_t store_idx = atomicAdd(work_idx, 1);

            updated[store_idx] = idx;
        }

        memcpy(neighbors_a + idx * Storage, tmp_nei, sizeof(neighbor_t) * N);
    }
}