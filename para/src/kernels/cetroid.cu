#include "kernels.cuh"

#include "device_launch_parameters.h"

#include "common_kernels.cuh"

using namespace clustering;

__global__ void centroid(const input_t in, const asgn_t* __restrict__ assignments, float* __restrict__ out, asgn_t cid, csize_t cluster_size)
{
	extern __shared__ float shared_mem[];

	float tmp[MAX_DIM];

	memset(tmp, 0, in.dim * sizeof(float));

	for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < in.count; idx += gridDim.x * blockDim.x)
	{
		if (assignments[idx] == cid)
		{
			for (csize_t i = 0; i < in.dim; ++i)
				tmp[i] += in.data[idx * in.dim + i];
		}
	}

	reduce_sum_block(tmp, in.dim, shared_mem);

	if (threadIdx.x == 0)
		for (csize_t i = 0; i < in.dim; ++i)
			atomicAdd(out + i, tmp[i] / cluster_size);
}

void run_centroid(const input_t in, const asgn_t* assignments, float* out, asgn_t cetroid_id, csize_t cluster_size, kernel_info info)
{
	CUCH(cudaMemset(out, 0, sizeof(float) * in.dim));
	cudaDeviceSynchronize();
	centroid << <info.grid_dim, info.block_dim, 32 * (in.dim * sizeof(float)) >> > (in, assignments, out, cetroid_id, cluster_size);
}
