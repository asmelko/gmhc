#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__device__ void point_neighbor(const float* __restrict__ centroids, csize_t dim, csize_t centroid_count, 
	neighbor_t* __restrict__ neighbors_a, neighbor_t* __restrict__ neighbors_act,
	float* __restrict__ shared_mem, csize_t idx, bool from_start)
{
	neighbor_t local_neighbors[N];

	for (csize_t i = 0; i < N; ++i)
		local_neighbors[i].distance = FLT_MAX;

	for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
		shared_mem[i] = centroids[idx * dim + i];

	__syncthreads();

	csize_t y = threadIdx.x + blockIdx.x * blockDim.x;

	if (from_start)
		for (; y < idx; y += blockDim.x * gridDim.x)
		{
			float dist = euclidean_norm(shared_mem, centroids + y * dim, dim);

			//add_neighbor<N>(local_neighbors, neighbor_t{ dist, y });
			add_neighbor<N>(neighbors_act + y * N, neighbor_t{ dist, idx });
		}
	else
		y += idx;

	for (; y < centroid_count - 1; y += blockDim.x * gridDim.x)
	{
		float dist = euclidean_norm(shared_mem, centroids + (y + 1) * dim, dim);

		add_neighbor<N>(local_neighbors, neighbor_t{ dist, y + 1 });
	}

	reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + dim));

	if (threadIdx.x == 0)
		memcpy(neighbors_a + (gridDim.x * idx + blockIdx.x) * N, local_neighbors, N * sizeof(neighbor_t));
}

template <csize_t N>
__global__ void neighbors(const float* __restrict__ centroids, csize_t dim, csize_t centroid_count, neighbor_t* __restrict__ neighbors_a)
{
	extern __shared__ float shared_mem[];

	for (csize_t x = 0; x < centroid_count; ++x)
	{
		point_neighbor<N>(centroids, dim, centroid_count, neighbors_a, nullptr, shared_mem, x, false);
	}
}

template <csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids, 
	neighbor_t* __restrict__ neighbors_a, neighbor_t* __restrict__ neighbors_act,
	const csize_t* __restrict__ updated, const csize_t* __restrict__ upd_count,
	csize_t dim, csize_t centroid_count, csize_t new_idx)
{
	extern __shared__ float shared_mem[];

	csize_t count = *upd_count;

	for (csize_t i = 0; i < count; ++i)
	{
		auto x = updated[i];
		point_neighbor<N>(centroids, dim, centroid_count, neighbors_a, neighbors_act, shared_mem, x, x == new_idx);
	}
}
