#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;


__inline__ __device__ void reduce_sum_warp(float* point, size_t dim)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		for (size_t i = 0; i < dim; ++i)
		{
			point[i] += __shfl_down_sync(__activemask(), point[i], offset);
		}
	}
}

__device__ void matvec(const float* mat, const float* vec, float* result, size_t dim, size_t offset, size_t threads)
{
	for (size_t idx = threadIdx.x - offset; idx < dim * dim; idx += threads)
	{
		size_t x = idx / dim;
		size_t y = idx % dim;

		result[x] += mat[x + y * dim] * vec[x];
	}
}

__inline__ __device__ float maha(const float* icov, float* shared_mem, size_t dim)
{
	float tmp_point[MAX_DIM];
	memset(tmp_point, 0, MAX_DIM * sizeof(float));

	matvec(icov, shared_mem, tmp_point, dim, 0, blockDim.x);
	reduce_sum_warp(tmp_point, dim);

	if (threadIdx.x % warpSize == 0)
		for (size_t i = 0; i < dim; i++)
			atomicAdd(shared_mem + dim + i, tmp_point[i]);

	float res = 0;
	if (threadIdx.x == 0)
	{
		for (size_t i = 0; i < dim; i++)
		{
			res += shared_mem[i] * shared_mem[dim + i];
		}
		res = sqrtf(res);
	}
	return res;
}

__global__ void maha_dist(const float* l_point, const float* r_point, const float* l_icov, const float* r_icov, size_t dim, float* result)
{
	extern __shared__ float shared_mem[];

	for (size_t idx = threadIdx.x; idx < 2 * dim; idx += blockDim.x)
	{
		if (idx < dim)
			shared_mem[idx] = l_point[idx] - r_point[idx];
		else
			shared_mem[idx] = 0;
	}

	__syncthreads();

	float res = 0;

	if (l_icov)
		res = maha(l_icov, shared_mem, dim);
	if (r_icov)
		res += maha(r_icov, shared_mem, dim);

	if (!l_icov && !r_icov && threadIdx.x == 0)
	{
		float tmp = 0;
		for (size_t i = 0; i < dim; i++)
		{
			float sum = l_point[i] - r_point[i];
			tmp += sum * sum;
		}
		res += sqrtf(tmp);
	}

	if (threadIdx.x == 0)
		*result = res / 2;

}