#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__device__ void point_neighbor(const float* __restrict__ centroids, csize_t dim, csize_t centroid_count, 
	neighbor_t* __restrict__ neighbors, float* __restrict__ shared_mem, csize_t idx)
{
	neighbor_t local_neighbors[N];

	for (csize_t i = 0; i < N; ++i)
		local_neighbors[i].distance = FLT_INF;

	for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
		shared_mem[i] = centroids[idx * dim + i];

	__syncthreads();

	for (csize_t y = threadIdx.x + blockIdx.x * blockDim.x + idx + 1; 
		y < centroid_count; y += blockDim.x * gridDim.x)
	{
		float dist = euclidean_norm(shared_mem, centroids + y * dim, dim);

		if (isinf(dist))
			dist = FLT_MAX;

		add_neighbor<N>(local_neighbors, neighbor_t{ dist, y });
	}

	reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + dim));

	__syncthreads();

	if (threadIdx.x == 0)
		memcpy(neighbors + (gridDim.x * idx + blockIdx.x) * N, local_neighbors, N * sizeof(neighbor_t));
}

template <csize_t N>
__global__ void neighbors(const float* __restrict__ centroids, csize_t dim, csize_t centroid_count, neighbor_t* __restrict__ neighbors)
{
	extern __shared__ float shared_mem[];

	for (csize_t x = 0; x < centroid_count; ++x)
	{
		point_neighbor<N>(centroids, dim, centroid_count, neighbors, shared_mem, x);
	}
}

template <csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids, 
	neighbor_t* __restrict__ neighbors,
	const csize_t* __restrict__ updated, const csize_t* __restrict__ upd_count,
	csize_t dim, csize_t centroid_count)
{
	extern __shared__ float shared_mem[];

	csize_t count = *upd_count;

	for (csize_t i = 0; i < count; ++i)
	{
		auto x = updated[i];
		point_neighbor<N>(centroids, dim, centroid_count, neighbors, shared_mem, x);
	}
}
