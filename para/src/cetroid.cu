#include <kernels.cuh>

#include <device_launch_parameters.h>

#define FLT2INTP(x) reinterpret_cast<uint32_t*>(x)

using namespace clustering;

__inline__ __device__ void reduce_sum_warp(float* point, size_t dim)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		for (size_t i = 0; i < dim; ++i)
			point[i] += __shfl_down_sync(0xFFFFFFFF, point[i], offset);
		*(uint32_t*)(point + dim) += __shfl_down_sync(0xFFFFFFFF, *(uint32_t*)(point + dim), offset);
	}
}

__inline__ __device__ void reduce_sum_block(float* point, size_t dim, char* shared_mem)
{
	reduce_sum_warp(point, dim);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		memcpy(shared_mem + warp_id * (dim * sizeof(float) + sizeof(uint32_t)), point, dim * sizeof(float) + sizeof(uint32_t));

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(point, shared_mem + threadIdx.x * (dim * sizeof(float) + sizeof(uint32_t)), dim * sizeof(float) + sizeof(uint32_t));
	else
		memset(point, 0, dim * sizeof(float) + sizeof(uint32_t));

	reduce_sum_warp(point, dim);
}

__global__ void compute_centroid(const input_t in, float* out, asgn_t cid)
{
	extern __shared__ float shared_mem[];

	uint32_t count = 0;

	float* tmp = (float*)malloc(in.dim * sizeof(float) + sizeof(uint32_t));

	memset(tmp, 0, in.dim * sizeof(float));

	for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < in.count; idx += gridDim.x * blockDim.x)
	{
		if (*FLT2INTP(in.data + idx * (in.dim + 1) + in.dim) == cid)
		{
			for (size_t i = 0; i < in.dim; ++i)
				tmp[i] += in.data[idx * (in.dim + 1) + i];
			++count;
		}
	}

	*(uint32_t*)(tmp + in.dim) = count;

	reduce_sum_block(tmp, in.dim, (char*)shared_mem);

	if (threadIdx.x == 0)
	{
		for (size_t i = 0; i < in.dim; ++i)
			atomicAdd(out + i, tmp[i]);
		atomicAdd((uint32_t*)(out + in.dim), *FLT2INTP(tmp + in.dim));
	}

	free(tmp);
}

void run_compute_centroid(const input_t in, float* out, clustering::asgn_t cetroid_id, kernel_info info)
{
	compute_centroid << <info.grid_dim, info.block_dim, 32 * (in.dim * sizeof(float) + sizeof(uint32_t)) >> > (in, out, cetroid_id);
}
