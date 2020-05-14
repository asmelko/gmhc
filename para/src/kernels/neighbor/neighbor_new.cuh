#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

#include "neighbor_maha.cuh"

using namespace clustering;

template <csize_t N>
__inline__ __device__ void point_neighbors_new_warp
(const float* __restrict__ centroids, const float* __restrict__ inverses, neighbor_t* __restrict__ neighbors,
	float* __restrict__ curr_centroid, const float* __restrict__ this_centroid, const float* __restrict__ this_icov,
	csize_t dim, csize_t idx, csize_t orig_idx, bool eucl)
{
	float dist = 0;

	auto warp_id = threadIdx.x / warpSize;
	auto lane_id = threadIdx.x % warpSize;

	for (csize_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = centroids[idx * dim + i] - this_centroid[i];

	__syncwarp();

	if (eucl)
		dist += euclidean_norm(curr_centroid + warp_id * dim, dim);
	else
		dist += maha_dist(curr_centroid + warp_id * dim, this_icov, dim, lane_id);

	auto icov_size = (dim + 1) * dim / 2;
	dist += maha_dist(curr_centroid + warp_id * dim, inverses + idx * icov_size, dim, lane_id);

	if (lane_id == 0)
	{
		dist = (isnan(dist) || isinf(dist)) ? FLT_MAX : dist / 2;

		add_neighbor<N>(neighbors, neighbor_t{ dist, orig_idx });
	}
}

template <csize_t N>
__global__ void neighbors_new_u
(const float* __restrict__ centroids, const float* __restrict__ inverses, neighbor_t* __restrict__ neighbors,
	csize_t dim, csize_t small_count, csize_t big_begin, csize_t big_count, csize_t new_idx)
{
	extern __shared__ float shared_mem[];

	if (new_idx < big_begin)
	{
		for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[new_idx * dim + i];

		__syncthreads();

		for (csize_t x = threadIdx.x + blockIdx.x * blockDim.x; x < new_idx; x += blockDim.x * gridDim.x)
		{
			float dist = euclidean_norm(shared_mem, centroids + x * dim, dim);
			if (isinf(dist) || isnan(dist))
				dist = FLT_MAX;

			add_neighbor<N>(neighbors + x * N, neighbor_t{ dist, new_idx });
		}
	}

	__syncthreads();

	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = shared_mem + dim + dim * dim;

	if (new_idx < big_begin) // new eucl in mahas
	{
		for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[new_idx * dim + i];

		__syncthreads();

		for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x + big_begin * warpSize;
			idx < (big_begin + big_count) * warpSize;
			idx += blockDim.x * gridDim.x)
		{
			point_neighbors_new_warp<N>
				(centroids, inverses, neighbors + (idx / warpSize) * N,
					curr_centroid, this_centroid, this_icov, dim, idx / warpSize, new_idx, true);
		}
	}
	else
	{
		auto icov_size = (dim + 1) * dim / 2;

		for (csize_t i = threadIdx.x; i < dim + icov_size; i += blockDim.x)
		{
			shared_mem[i] = (i < dim) ?
				centroids[new_idx * dim + i] :
				inverses[new_idx * icov_size + i - dim];
		}

		__syncthreads();

		for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x + big_begin * warpSize;
			idx < new_idx * warpSize; idx += blockDim.x * gridDim.x)
		{
			point_neighbors_new_warp<N>
				(centroids, inverses, neighbors + (idx / warpSize) * N,
					curr_centroid, this_centroid, this_icov, dim, idx / warpSize, new_idx, false);
		}
	}
}
