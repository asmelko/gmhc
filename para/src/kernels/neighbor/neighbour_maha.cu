#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

__inline__ __device__ float maha_dist(const float* point, const float* matrix, size_t size, size_t lane_id)
{
	float tmp_point = 0;

	auto icov_size = (size + 1) * size / 2;

	for (size_t i = lane_id; i < icov_size; i += warpSize)
	{
		auto coords = compute_coordinates(size, i);

		tmp_point = fmaf(matrix[i], point[coords.x] * point[coords.y], tmp_point);
	}

	reduce_sum_warp(&tmp_point, 1);

	if (lane_id == 0)
	{
		return sqrtf(tmp_point);
	}
	return 0;
}

template <size_t N>
__inline__ __device__ void point_neighbours_mat_warp
(const float* centroids, const float* inverses, neighbour_t* neighbours, 
	float* curr_centroid, const float* this_centroid, const float* this_icov, 
	asgn_t dim, asgn_t big_begin, asgn_t idx, bool eucl = false)
{
	float dist = 0;

	size_t warp_id = threadIdx.x / warpSize;
	size_t lane_id = threadIdx.x % warpSize;

	for (size_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = centroids[idx * dim + i];

	__syncwarp();

	if (eucl || idx < big_begin)
		dist += euclidean_norm(this_centroid, curr_centroid + warp_id * dim, dim);

	for (size_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = curr_centroid[warp_id * dim + i] - this_centroid[i];

	__syncwarp();

	if (!eucl)
		dist += maha_dist(curr_centroid + warp_id * dim, this_icov, dim, lane_id);

	if (eucl || idx >= big_begin)
	{
		auto icov_size = (dim + 1) * dim / 2;
		dist += maha_dist(curr_centroid + warp_id * dim, inverses + idx * icov_size, dim, lane_id);
	}

	if (lane_id == 0)
		add_neighbour<N>(neighbours, neighbour_t{ dist / 2, idx });
}

template <size_t N>
__inline__ __device__ void point_neighbours_mat
(const float* centroids, const float* inverses, neighbour_t* neighbours, float* shared_mem,
	asgn_t dim, asgn_t small_count, asgn_t big_begin, asgn_t big_count, size_t x, bool from_start)
{
	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = shared_mem + dim + dim * dim;

	neighbour_t local_neighbours[N];

	for (size_t i = 0; i < N; ++i)
		local_neighbours[i].distance = FLT_MAX;

	bool needs_merge = false;

	if (x < big_begin) // new eucl in mahas
	{
		needs_merge = true;
		for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[x * dim + i];

		__syncthreads();

		for (asgn_t idx = threadIdx.x + blockIdx.x * blockDim.x + big_begin * warpSize;
			idx < (big_begin + big_count) * warpSize;
			idx += blockDim.x * gridDim.x)
		{
			point_neighbours_mat_warp<N>
				(centroids, inverses, local_neighbours, curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize, true);
		}
	}
	else
	{
		{
			auto icov_size = (dim + 1) * dim / 2;

			for (size_t i = threadIdx.x; i < dim + icov_size; i += blockDim.x)
				if (i < dim)
					shared_mem[i] = centroids[x * dim + i];
				else
				{
					shared_mem[i] = inverses[x * icov_size + i - dim];
				}
		}
		

		__syncthreads();

		asgn_t idx = threadIdx.x + blockIdx.x * blockDim.x;

		for (; idx < small_count * warpSize; idx += blockDim.x * gridDim.x)
		{
			point_neighbours_mat_warp<N>
				(centroids, inverses, local_neighbours, curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize);
		}

		idx += (big_begin - small_count) * warpSize;

		if (from_start)
			for (; idx < x * warpSize; idx += blockDim.x * gridDim.x)
			{
				point_neighbours_mat_warp<N>
					(centroids, inverses, local_neighbours, curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize);
			}
		else
			idx += (x - big_begin) * warpSize;

		for (; idx < (big_begin + big_count - 1) * warpSize; idx += blockDim.x * gridDim.x)
		{
			point_neighbours_mat_warp<N>
				(centroids, inverses, local_neighbours, curr_centroid, this_centroid, this_icov, dim, big_begin, (idx / warpSize) + 1);
		}
	}

	reduce_min_block<N>(local_neighbours, reinterpret_cast<neighbour_t*>(shared_mem + (33 + dim) * dim), false);

	if (threadIdx.x == 0)
	{
		if (needs_merge)
		{
			neighbour_t tmp_nei[N];
			merge_neighbours<N>(neighbours + (gridDim.x * x + blockIdx.x) * N, local_neighbours, tmp_nei);
			memcpy(local_neighbours, tmp_nei, sizeof(neighbour_t) * N);
		}

		memcpy(neighbours + (gridDim.x * x + blockIdx.x) * N, local_neighbours, sizeof(neighbour_t) * N);
	}
}


template <size_t N>
__global__ void neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* neighbours, kernel_info info)
{
	size_t shared_mat = (dim + 33) * dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		point_neighbours_mat<N> << <info.grid_dim, info.block_dim, shared_mat >> > (centroids, inverses, dim, centroid_count, neighbours, x);
	}
}

template <size_t N>
__global__ void neighbours_mat_u
(const float* centroids, const float* inverses, neighbour_t* neighbours, uint8_t* updated,
	asgn_t dim, asgn_t small_count, asgn_t big_begin, asgn_t big_count, size_t new_idx)
{
	extern __shared__ float shared_mem[];

	if (new_idx < big_begin)
	{
		point_neighbours_mat<N>(centroids, inverses, neighbours, shared_mem, dim, small_count, big_begin, big_count, new_idx, true);
	}

	for (asgn_t x = big_begin; x < big_begin + big_count; ++x)
	{
		if (updated[x])
		{
			point_neighbours_mat<N>(centroids, inverses, neighbours, shared_mem, dim, small_count, big_begin, big_count, x, x == new_idx);
		}
	}
}