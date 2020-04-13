#include <kernels.cuh>

#include <device_launch_parameters.h>
#include <algorithm>

#include "common_kernels.cuh"

using namespace clustering;

__global__ void centroid(const float* __restrict__ points, csize_t dim, csize_t point_count, const asgn_t* __restrict__ assignment_idxs, csize_t cluster_size, float* __restrict__ out)
{
	extern __shared__ float shared_mem[];

	float tmp[MAX_DIM];

	memset(tmp, 0, dim * sizeof(float));

	for (csize_t i = blockDim.x * blockIdx.x + threadIdx.x; i < cluster_size; i += gridDim.x * blockDim.x)
	{
		auto idx = assignment_idxs[i];

		for (csize_t i = 0; i < dim; ++i)
			tmp[i] += points[idx * dim + i];
	}

	reduce_sum_block(tmp, dim, shared_mem);

	if (threadIdx.x == 0)
		for (csize_t i = 0; i < dim; ++i)
			atomicAdd(out + i, tmp[i] / cluster_size);
}

void run_centroid(const input_t in, const asgn_t* assignment_idxs, csize_t cluster_size, float* out, kernel_info info)
{
	constexpr unsigned int warpSize = 32;
	CUCH(cudaMemset(out, 0, sizeof(float) * in.dim));

	auto warps = (cluster_size + warpSize - 1) / warpSize;
	if (warps < info.grid_dim)
	{
		info.grid_dim = warps;
		info.block_dim = warpSize;
	}
	else
		info.block_dim = std::min(warpSize * warps, info.block_dim);

	centroid<<<info.grid_dim, info.block_dim, 32 * (in.dim * sizeof(float))>>>(in.data, in.dim, in.count, assignment_idxs, cluster_size, out);
}
