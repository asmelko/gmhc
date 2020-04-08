#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "common_kernels.cuh"

using namespace clustering;

template <size_t N>
__device__ void point_neighbour(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* neighbours_a, float* shared_mem, cluster_kind* cluster_kinds, size_t idx, bool from_start)
{
	neighbour_t local_neighbours[N];

	for (size_t i = 0; i < N; ++i)
		local_neighbours[i].distance = FLT_MAX;

	for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
		shared_mem[i] = centroids[idx * dim + i];

	__syncthreads();

	asgn_t y = threadIdx.x + blockIdx.x * blockDim.x;

	if (from_start)
		for (; y < idx; y += blockDim.x * gridDim.x)
		{
			if (cluster_kinds[y] != cluster_kind::EUCL)
				continue;

			//printf("from start %d\n", (int)y);

			float dist = euclidean_norm(shared_mem, centroids + y * dim, dim);

			add_neighbour<N>(local_neighbours, neighbour_t{ dist, y });
		}
	else
		y += idx;

	for (; y < centroid_count - 1; y += blockDim.x * gridDim.x)
	{
		if (cluster_kinds[y + 1] != cluster_kind::EUCL)
			continue;

		//printf("cycle %d %d\n", (int)y, (int)new_y);


		float dist = euclidean_norm(shared_mem, centroids + (y + 1) * dim, dim);

		add_neighbour<N>(local_neighbours, neighbour_t{ dist, y + 1 });
	}

	reduce_min_block<N>(local_neighbours, reinterpret_cast<neighbour_t*>(shared_mem + dim));

	if (threadIdx.x == 0)
	{
		memcpy(neighbours_a + (gridDim.x * idx + blockIdx.x) * N, local_neighbours, N * sizeof(neighbour_t));
	}
}

template <size_t N>
__global__ void neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* neighbours_a, cluster_kind* cluster_kinds)
{
	extern __shared__ float shared_mem[];

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (cluster_kinds[x] != cluster_kind::EUCL)
			continue;

		point_neighbour<N>(centroids, dim, centroid_count, neighbours_a, shared_mem, cluster_kinds, x, false);
	}
}

template <size_t N>
__global__ void neighbours_u(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* neighbours_a, cluster_kind* cluster_kinds, uint8_t* updated, size_t new_idx)
{
	extern __shared__ float shared_mem[];

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (!updated[x] || cluster_kinds[x] != cluster_kind::EUCL)
			continue;

		point_neighbour<N>(centroids, dim, centroid_count, neighbours_a, shared_mem, cluster_kinds, x, x == new_idx);
	}
}