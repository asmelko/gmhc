#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

__inline__ __device__ float maha_dist(const float* point, const float* matrix, size_t size, size_t lane_id)
{
	float tmp_point[MAX_DIM];
	for (size_t i = 0; i < size; i++)
		tmp_point[i] = 0;

	for (size_t i = lane_id; i < size * size; i += warpSize)
	{
		size_t mat_x = i / size;
		size_t mat_y = i % size;

		tmp_point[mat_x] += matrix[mat_x * size + mat_y] * point[mat_y];
	}

	reduce_sum_warp(tmp_point, size);

	/*
	if (lane_id == 0)
	{
		printf("matvec ");
		for (size_t i = 0; i < size; i++)
		{
			printf("%f ", tmp_point[i]);
		}
		printf("\n");
	}*/

	if (lane_id == 0)
	{
		float tmp_res = 0;
		for (size_t i = 0; i < size; ++i)
			tmp_res += tmp_point[i] * point[i];

		//printf("dot %f\n", tmp_res);

		return sqrtf(tmp_res);
	}
	return 0;
}

template <size_t N>
__inline__ __device__ void point_neighbours_mat_warp
(const float* centroids, const float* const* inverses, size_t dim, neighbour_t* neighbours, float* curr_centroid,
	const float* this_centroid, const float* this_icov, const cluster_kind* cluster_kinds, asgn_t idx, bool eucl = false)
{
	float dist = 0;

	size_t warp_id = threadIdx.x / warpSize;
	size_t lane_id = threadIdx.x % warpSize;

	for (size_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = centroids[idx * dim + i];

	__syncwarp();

	if (eucl || cluster_kinds[idx] == cluster_kind::EUCL)
		dist += euclidean_norm(this_centroid, curr_centroid + warp_id * dim, dim);

	//if (lane_id == 0)
	//	printf("dist eucl %d %f\n", (int)idx, dist);

	for (size_t i = lane_id; i < dim; i += warpSize)
		curr_centroid[warp_id * dim + i] = curr_centroid[warp_id * dim + i] - this_centroid[i];

	__syncwarp();

	/*
	if (lane_id == 0)
	{
		printf("centr ");
		for (size_t i = 0; i < dim; i++)
		{
			printf("%f ", curr_centroid[warp_id * dim + i]);
		}
		printf("\n");
	}*/

	if (!eucl)
		dist += maha_dist(curr_centroid + warp_id * dim, this_icov, dim, lane_id);

	//if (lane_id == 0)
	//	printf("dist maha %d %f\n", (int)idx, dist);

	if (eucl || cluster_kinds[idx] == cluster_kind::MAHA)
		dist += maha_dist(curr_centroid + warp_id * dim, inverses[idx], dim, lane_id);


	//if (lane_id == 0)
	//	printf("dist maha2 %d %f\n", (int)idx, dist);

	//if (lane_id == 0)
	//	printf("dist fin %d %f\n", (int)idx, dist);

	if (lane_id == 0)
		add_neighbour<N>(neighbours, neighbour_t{ dist / 2, idx });
}

template <size_t N>
__inline__ __device__ void point_neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* neighbours, cluster_kind* cluster_kinds, float* shared_mem, asgn_t x, bool from_start)
{
	//extern __shared__ float shared_mem[];

	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = shared_mem + dim + dim * dim;

	neighbour_t local_neighbours[N];

	for (size_t i = 0; i < N; ++i)
		local_neighbours[i].distance = FLT_MAX;



	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	bool needs_merge = false;

	if (from_start && cluster_kinds[x] == cluster_kind::EUCL)
	{
		needs_merge = true;
		for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[x * dim + i];

		__syncthreads();

		for (; idx < centroid_count * warpSize; idx += blockDim.x * gridDim.x)
		{
			size_t y = idx / warpSize;

			if (y == x || cluster_kinds[y] != cluster_kind::MAHA)
				continue;

			point_neighbours_mat_warp<N>(centroids, inverses, dim, local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y, true);
		}
	}
	else
	{


		if (threadIdx.x == 0 && x == 8)
			printf("was here\n");

		for (size_t i = threadIdx.x; i < dim + dim * dim; i += blockDim.x)
			if (i < dim)
				shared_mem[i] = centroids[x * dim + i];
			else
				shared_mem[i] = inverses[x][i - dim];

		__syncthreads();

		if (from_start)
			for (; idx < x * warpSize; idx += blockDim.x * gridDim.x)
			{

				if (threadIdx.x % warpSize == 0 && x == 8)
					printf("was from start\n");

				size_t y = idx / warpSize;

				point_neighbours_mat_warp<N>(centroids, inverses, dim, local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y);
			}
		else
			idx += x * warpSize;

		for (; idx < (centroid_count - 1) * warpSize; idx += blockDim.x * gridDim.x)
		{
			size_t y = (idx / warpSize) + 1;


			if (threadIdx.x % warpSize == 0 && x == 8)
				printf("was here %d\n", (int)y);

			point_neighbours_mat_warp<N>(centroids, inverses, dim, local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y);
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
		memcpy(neighbours + (gridDim.x * x + blockIdx.x) * N, local_neighbours, sizeof(neighbour_t )*N);
	}
}


template <size_t N>
__global__ void neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* neighbours, cluster_kind* cluster_kinds, kernel_info info)
{
	size_t shared_mat = (dim + 33) * dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (cluster_kinds[x] != cluster_kind::MAHA)
			continue;

		point_neighbours_mat<N> << <info.grid_dim, info.block_dim, shared_mat >> > (centroids, inverses, dim, centroid_count, neighbours, cluster_kinds, x);
	}
}

template <size_t N>
__global__ void neighbours_mat_u(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t new_idx)
{
	extern __shared__ float shared_mem[];
	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (!updated[x] || (cluster_kinds[x] != cluster_kind::MAHA && x != new_idx))
			continue;

		point_neighbours_mat<N>(centroids, inverses, dim, centroid_count, neighbours, cluster_kinds, shared_mem, x, x == new_idx);
	}
}
