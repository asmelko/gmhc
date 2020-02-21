#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

__constant__ float expected_point[MAX_DIM];

void assign_constant_storage(const float* value, size_t size, cudaMemcpyKind kind)
{
	CUCH(cudaMemcpyToSymbol(expected_point, value, size, (size_t)0, kind));
}

__inline__ __device__ void reduce_sum_warp(float* cov, size_t size, unsigned mask)
{
	for (size_t i = 0; i < size; ++i)
	{
		float tmp = cov[i];
		for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
			tmp += __shfl_down_sync(mask, tmp, offset);
		if (threadIdx.x % warpSize == 0)
			cov[i] = tmp;
	}
}

__inline__ __device__ void reduce_sum_block(float* shared_mem, size_t shared_chunks, size_t cov_size)
{
	float* tmp_cov;
	if (threadIdx.x < shared_chunks)
	{
		tmp_cov = shared_mem + cov_size * threadIdx.x;
		reduce_sum_warp(tmp_cov, cov_size, __activemask());
	}

	auto ciel_div = (shared_chunks + warpSize - 1) / warpSize;

	if (threadIdx.x < ciel_div)
	{
		tmp_cov = shared_mem + cov_size * threadIdx.x * warpSize;
		reduce_sum_warp(tmp_cov, cov_size, __activemask());
	}
}

__inline__ __device__ void point_covariance(const float* __restrict__ point, size_t dim, float* __restrict__ shared_mem)
{
	size_t idx = 0;
	for (size_t i = 0; i < dim; ++i)
		for (size_t j = i; j < dim; ++j)
			atomicAdd(shared_mem + idx++, point[i] * point[j]);
}

__global__ void covariance(const float* __restrict__ points, size_t dim, size_t count, const asgn_t* __restrict__ assignments, float* __restrict__ cov_matrix, asgn_t cid, size_t shared_chunks)
{
	size_t cov_size = ((dim + 1) * dim) / 2;
	extern __shared__ float shared_mem[];
	float tmp_point[MAX_DIM];

	for (size_t idx = threadIdx.x; idx < cov_size * shared_chunks; idx += blockDim.x)
		shared_mem[idx] = 0;
	
	__syncthreads();

	float* tmp_cov = shared_mem + cov_size * (threadIdx.x % shared_chunks);

	for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
		if (assignments[idx] == cid)
		{
			for (size_t i = 0; i < dim; ++i)
				tmp_point[i] = points[idx * dim + i] - expected_point[i];

			point_covariance(tmp_point, dim, tmp_cov);
		}

	reduce_sum_block(shared_mem, shared_chunks, cov_size);

	if (threadIdx.x == 0)
		for (size_t i = 0; i < cov_size; ++i)
			atomicAdd(cov_matrix + i, tmp_cov[i]);
}

void run_covariance(const input_t in, const asgn_t* assignments, float* out, asgn_t centroid_id, kernel_info info)
{
	size_t cov_size = ((in.dim + 1) * in.dim) / 2;
	size_t shared_chunks = info.shared_size;

	covariance << <info.grid_dim, info.block_dim, shared_chunks* cov_size * sizeof(float) >> > (in.data, in.dim, in.count, assignments, out, centroid_id, shared_chunks);
}