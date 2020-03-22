#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

extern __shared__ float shared_mem[];

__inline__ __device__ void reduce_sum_warp(float* point, size_t dim)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
		for (size_t i = 0; i < dim; ++i)
			point[i] += __shfl_down_sync(0xFFFFFFFF, point[i], offset);
}

__inline__ __device__ void reduce_sum_block(float* point, size_t dim)
{
	reduce_sum_warp(point, dim);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		memcpy(shared_mem + warp_id * dim, point, dim * sizeof(float));

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(point, shared_mem + threadIdx.x * dim, dim * sizeof(float));
	else
		memset(point, 0, dim * sizeof(float));

	reduce_sum_warp(point, dim);
}

__global__ void centroid(const input_t in, const asgn_t* __restrict__ assignments, float* __restrict__ out, asgn_t cid, size_t cluster_size)
{

	float tmp[MAX_DIM];

	memset(tmp, 0, in.dim * sizeof(float));

	for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < in.count; idx += gridDim.x * blockDim.x)
	{
		if (assignments[idx] == cid)
		{
			for (size_t i = 0; i < in.dim; ++i)
				tmp[i] += in.data[idx * in.dim + i];
		}
	}

	reduce_sum_block(tmp, in.dim);

	if (threadIdx.x == 0)
		for (size_t i = 0; i < in.dim; ++i)
			atomicAdd(out + i, tmp[i] / cluster_size);
}

__global__ void print_centroid(const float* centroid, size_t dim, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("%d. ", (int)i);
		for (size_t j = 0; j < dim; j++)
		{
			printf("%f ", centroid[i * dim + j]);
		}
		printf("\n");
	}
}

void run_print_centroid(const float* centroid, size_t dim, size_t count)
{
	print_centroid << <1, 1 >> > (centroid, dim, count);
}

void run_centroid(const input_t in, const asgn_t* assignments, float* out, asgn_t cetroid_id, size_t cluster_size, kernel_info info)
{
	CUCH(cudaMemset(out, 0, sizeof(float) * in.dim));
	cudaDeviceSynchronize();
	centroid << <info.grid_dim, info.block_dim, 32 * (in.dim * sizeof(float)) >> > (in, assignments, out, cetroid_id, cluster_size);
}
