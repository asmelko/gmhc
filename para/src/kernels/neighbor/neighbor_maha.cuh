#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

__inline__ __device__ float maha_dist(const float* __restrict__ point, const float* __restrict__ matrix, csize_t size, unsigned int lane_id)
{
	float tmp_point = 0;

	auto icov_size = (size + 1) * size / 2;

	for (csize_t i = lane_id; i < icov_size; i += warpSize)
	{
		auto coords = compute_coordinates(size, i);

		tmp_point = fmaf(matrix[i], point[coords.x] * point[coords.y], tmp_point);
	}

	reduce_sum_warp(&tmp_point, 1);

	if (lane_id == 0)
	{
		if (tmp_point < 0)
		{
			tmp_point = 0;
			for (size_t i = 0; i < size; ++i)
				tmp_point += point[i] * point[i];
		}
		return sqrtf(tmp_point);
	}
	return 0;
}

template <csize_t N>
__inline__ __device__ void point_neighbors_mat_warp
(const float* __restrict__ centroids, const float* __restrict__ inverses, neighbor_t* __restrict__ neighbors,
	float* __restrict__ curr_centroid, const float* __restrict__ this_centroid, const float* __restrict__ this_icov,
	csize_t dim, csize_t big_begin, csize_t idx, csize_t orig_idx, bool eucl = false)
{
	float dist = 0;

	auto warp_id = threadIdx.x / warpSize;
	auto lane_id = threadIdx.x % warpSize;

	for (csize_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = centroids[idx * dim + i];

	__syncwarp();

	if (eucl || idx < big_begin)
		dist += euclidean_norm(this_centroid, curr_centroid + warp_id * dim, dim);

	for (csize_t i = lane_id; i < dim; i += warpSize)
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
	{
		idx = orig_idx == (csize_t)-1 ? idx : orig_idx;
		add_neighbor<N>(neighbors, neighbor_t{ dist / 2, idx });
	}
}

template <csize_t N>
__inline__ __device__ void point_neighbors_mat
(const float* __restrict__ centroids, const float* __restrict__ inverses, 
	neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ neighbors_act, float* __restrict__ shared_mem,
	csize_t dim, csize_t small_count, csize_t big_begin, csize_t big_count, csize_t x, bool from_start)
{
	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = shared_mem + dim + dim * dim;

	neighbor_t local_neighbors[N];

	for (csize_t i = 0; i < N; ++i)
		local_neighbors[i].distance = FLT_MAX;

	bool needs_merge = false;

	if (x < big_begin) // new eucl in mahas
	{
		needs_merge = true;
		for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[x * dim + i];

		__syncthreads();

		for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x + big_begin * warpSize;
			idx < (big_begin + big_count) * warpSize;
			idx += blockDim.x * gridDim.x)
		{
			point_neighbors_mat_warp<N>
				(centroids, inverses, local_neighbors, curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize, (csize_t)-1, true);
		}
	}
	else
	{
		{
			auto icov_size = (dim + 1) * dim / 2;

			for (csize_t i = threadIdx.x; i < dim + icov_size; i += blockDim.x)
				if (i < dim)
					shared_mem[i] = centroids[x * dim + i];
				else
				{
					shared_mem[i] = inverses[x * icov_size + i - dim];
				}
		}
		
		__syncthreads();

		csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;

		for (; idx < small_count * warpSize; idx += blockDim.x * gridDim.x)
		{
			point_neighbors_mat_warp<N>
				(centroids, inverses, local_neighbors, curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize, (csize_t)-1);
		}

		idx += (big_begin - small_count) * warpSize;

		if (from_start)
		{
			for (; idx < x * warpSize; idx += blockDim.x * gridDim.x)
			{
				point_neighbors_mat_warp<N>
					(centroids, inverses, neighbors_act + (idx / warpSize) * N, 
						curr_centroid, this_centroid, this_icov, dim, big_begin, idx / warpSize, x);
			}
		}
		else
			idx += (x - big_begin) * warpSize;

		for (; idx < (big_begin + big_count - 1) * warpSize; idx += blockDim.x * gridDim.x)
		{
			point_neighbors_mat_warp<N>
				(centroids, inverses, local_neighbors, curr_centroid, this_centroid, this_icov, dim, big_begin, (idx / warpSize) + 1, (csize_t)-1);
		}
	}

	reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + (33 + dim) * dim), false);

	if (threadIdx.x == 0)
	{
		if (needs_merge)
		{
			neighbor_t tmp_nei[N];
			merge_neighbors<N>(neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, tmp_nei);
			memcpy(local_neighbors, tmp_nei, sizeof(neighbor_t) * N);
		}

		memcpy(neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, sizeof(neighbor_t) * N);
	}
}

template <csize_t N>
__global__ void neighbors_mat_u
(const float* __restrict__ centroids, const float* __restrict__ inverses, 
	neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ neighbors_act,
	csize_t* __restrict__ updated, const csize_t* __restrict__ upd_count,
	csize_t dim, csize_t small_count, csize_t big_begin, csize_t big_count, csize_t new_idx)
{
	extern __shared__ float shared_mem[];

	if (new_idx < big_begin)
	{
		point_neighbors_mat<N>(centroids, inverses, neighbors, neighbors_act, shared_mem, dim, small_count, big_begin, big_count, new_idx, true);
	}

	auto count = *upd_count - big_begin;

	for (csize_t i = 0; i < count; ++i)
	{
		auto x = updated[i + big_begin];
		point_neighbors_mat<N>(centroids, inverses, neighbors, neighbors_act, shared_mem, dim, small_count, big_begin, big_count, x, x == new_idx);
	}
}
