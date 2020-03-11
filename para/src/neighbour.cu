#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

template <size_t N>
__device__ void add_neighbour(neighbour_array_t<N>* neighbours, neighbour_t neighbour)
{
	neighbour_t prev_min;
	size_t i = 0;
	for (; i < N; ++i)
	{
		if (neighbours->neighbours[i].distance > neighbour.distance)
		{
			prev_min = neighbours->neighbours[i];
			neighbours->neighbours[i++] = neighbour;
			break;
		}
	}

	for (; i < N; i++)
	{
		if (prev_min.distance == FLT_MAX)
			return;

		neighbour_t tmp = neighbours->neighbours[i];
		neighbours->neighbours[i] = prev_min;
		prev_min = tmp;
	}
}

template <size_t N>
__device__ neighbour_array_t<N> merge_neighbours(const neighbour_array_t<N>* l_neighbours, const neighbour_array_t<N>* r_neighbours)
{
	neighbour_array_t<N> tmp;

	size_t l_idx = 0, r_idx = 0;

	for (size_t i = 0; i < N; ++i)
	{
		if (l_neighbours->neighbours[l_idx].distance < r_neighbours->neighbours[r_idx].distance)
			tmp.neighbours[i] = l_neighbours->neighbours[l_idx++];
		else
			tmp.neighbours[i] = r_neighbours->neighbours[r_idx++];
	}

	return tmp;
}

__inline__ __device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim)
{
	float tmp_sum = 0;
	for (size_t i = 0; i < dim; ++i)
	{
		auto tmp = l_point[i] - r_point[i];
		tmp_sum += tmp * tmp;
	}
	return sqrtf(tmp_sum);
}


template <size_t N>
__inline__ __device__ void reduce_min_warp(neighbour_array_t<N>* neighbours)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		neighbour_array_t<N> tmp;
		for (size_t i = 0; i < N; ++i)
		{
			tmp.neighbours[i].distance = __shfl_down_sync(0xFFFFFFFF, neighbours->neighbours[i].distance, offset);
			tmp.neighbours[i].idx = __shfl_down_sync(0xFFFFFFFF, neighbours->neighbours[i].idx, offset);
		}

		*neighbours = merge_neighbours(neighbours, &tmp);
	}
}

template <size_t N>
__inline__ __device__ void reduce_min_block(neighbour_array_t<N>* neighbours, neighbour_array_t<N>* shared_mem)
{
	reduce_min_warp(neighbours);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		shared_mem[warp_id] = *neighbours;

	__syncthreads();

	*neighbours = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : shared_mem[0];

	reduce_min_warp(neighbours);
}

template <size_t N>
__inline__ __device__ void reduce_min(neighbour_array_t<N>* neighbours, neighbour_array_t<N>* shared_mem, neighbour_array_t<N>* global_mem)
{
	reduce_min_block(neighbours, shared_mem);

	if (threadIdx.x == 0)
	{

	}
}

template <size_t N>
__global__ void neighbours(const float* centroids, const float *const * inverses, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours)
{
	extern __shared__ float shared_mem[];

	neighbour_array_t<N> local_neighbours;
	for (size_t i = 0; i < N; ++i)
		local_neighbours.neighbours[i].distance = FLT_MAX;

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (inverses[x])
			continue;

		for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[x * dim + i];

		for (asgn_t y = x + threadIdx.x; y < centroid_count; y += blockDim.x * gridDim.x)
		{
			if (inverses[y])
				continue;

			float dist = euclidean_norm(shared_mem, centroids + y * dim, dim);

			add_neighbour(&local_neighbours, neighbour_t{ dist, y });
		}
	}
}