#include "common_kernels.cuh"

__device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim)
{
	float tmp_sum = 0;
	for (size_t i = 0; i < dim; ++i)
	{
		auto tmp = l_point[i] - r_point[i];
		tmp_sum += tmp * tmp;
	}
	return sqrtf(tmp_sum);
}

__device__ void reduce_sum_warp(float* point, size_t dim)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
		for (size_t i = 0; i < dim; ++i)
			point[i] += __shfl_down_sync(0xFFFFFFFF, point[i], offset);
}

__device__ void reduce_sum_block(float* point, size_t dim, float* shared_mem)
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

__device__ size2 compute_coordinates(size_t count_in_line, size_t plain_index)
{
	size_t y = 0;
	while (plain_index >= count_in_line)
	{
		y++;
		plain_index -= count_in_line--;
	}
	return { plain_index + y, y };
}

__device__ chunk_t reduce_min_warp(chunk_t data)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		auto tmp_dist = __shfl_down_sync(0xFFFFFFFF, data.min_dist, offset);
		auto tmp_i = __shfl_down_sync(0xFFFFFFFF, data.min_i, offset);
		auto tmp_j = __shfl_down_sync(0xFFFFFFFF, data.min_j, offset);
		if (tmp_dist < data.min_dist)
		{
			data.min_dist = tmp_dist;
			data.min_i = tmp_i;
			data.min_j = tmp_j;
		}
	}
	return data;
}

__device__ chunk_t reduce_min_block(chunk_t data, chunk_t* shared_mem)
{
	data = reduce_min_warp(data);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		shared_mem[warp_id] = data;

	__syncthreads();

	data = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : shared_mem[0];

	data = reduce_min_warp(data);
	return data;
}