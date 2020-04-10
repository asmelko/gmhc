#include <kernels.cuh>

#include <device_launch_parameters.h>

#include "neighbor_common.cuh"

using namespace clustering;

template <size_t N>
__inline__ __device__ void point_reduce(const neighbour_t* neighbours, size_t to_reduce, size_t count, neighbour_t* reduced, size_t idx)
{
	size_t block = idx / warpSize;
	neighbour_t local[N];

	size_t nei = idx % warpSize;

	if (nei < to_reduce)
		memcpy(local, neighbours + (block * to_reduce + nei) * N, sizeof(neighbour_t) * N);
	else
		for (size_t i = 0; i < N; i++)
			local[i].distance = FLT_MAX;


	for (nei += warpSize; nei < to_reduce; nei += warpSize)
	{
		neighbour_t tmp[N];
		merge_neighbours<N>(local, neighbours + (block * to_reduce + nei)*N, tmp);
		memcpy(local, tmp, sizeof(neighbour_t) * N);
	}


	reduce_min_warp<N>(local);


	if (threadIdx.x % warpSize == 0)
	{
		memcpy(reduced + block*N, local, sizeof(neighbour_t) * N);
	}
}


template <size_t N>
__global__ void reduce(const neighbour_t* neighbours, size_t to_reduce, size_t count, neighbour_t* reduced)
{
	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbours, to_reduce, count, reduced, idx);
	}
}

template <size_t N>
__global__ void update_neighbours(size_t centroid_count, neighbour_t* neighbours_a, uint8_t* updated, size_t old_i, size_t old_j)
{
	extern __shared__ float shared_mem[];

	auto idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (asgn_t x = idx; x < centroid_count; x += blockDim.x * gridDim.x)
	{
		if (x == old_i || x == old_j)
		{
			updated[x] = 1;
			continue;
		}

		neighbour_t tmp_nei[N];
		memcpy(tmp_nei, neighbours_a + x * N, sizeof(neighbour_t) * N);

		size_t last_empty = 0;

		for (size_t i = 0; i < N; i++)
		{
			if (tmp_nei[i].distance == FLT_MAX)
				break;

			if (tmp_nei[i].idx == old_i || tmp_nei[i].idx == old_j)
				tmp_nei[i].distance = FLT_MAX;
			else
			{
				if (tmp_nei[i].idx == centroid_count)
					tmp_nei[i].idx = old_j;

				tmp_nei[last_empty++] = tmp_nei[i];
			}
		}

		updated[x] = tmp_nei[0].distance == FLT_MAX ? 1 : 0;

		memcpy(neighbours_a + x * N,tmp_nei, sizeof(neighbour_t) * N);
	}
}

template <size_t N>
__global__ void reduce_u(const neighbour_t* neighbours, size_t to_reduce, size_t count, neighbour_t* reduced, uint8_t* updated)
{
	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		if (updated[idx / warpSize])
			point_reduce<N>(neighbours, to_reduce, count, reduced, idx);
	}
}

template <size_t N>
__global__ void min(const neighbour_t* neighbours, size_t count, chunk_t* result)
{
	static __shared__ chunk_t shared_mem[32];

	chunk_t tmp;
	tmp.min_dist = FLT_MAX;
	for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x)
	{
		if (tmp.min_dist > neighbours[idx*N].distance)
		{
			tmp.min_dist = neighbours[idx * N].distance;
			tmp.min_j = neighbours[idx * N].idx;
			tmp.min_i = idx;
		}
	}

	tmp = reduce_min_block(tmp, shared_mem);

	if (threadIdx.x == 0)
		*result = tmp;
}

#include "neighbour_eucl.cu"
#include "neighbour_maha.cu"

template <size_t N>
void run_update_neighbours(const float* centroids, const float* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info)
{
	size_t shared = dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	size_t shared_mat = (dim + 33) * dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	update_neighbours<N><<<info.grid_dim, info.block_dim >>>(centroid_count, act_neighbours, updated, old_i, old_j);

	neighbours_u<N> << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbours, cluster_kinds, updated, old_i);
	neighbours_mat_u<N> << <info.grid_dim, info.block_dim, shared_mat >> > (centroids, inverses, dim, centroid_count, tmp_neighbours, cluster_kinds, updated, old_i);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	reduce_u<N><<<info.grid_dim, info.block_dim>>>(tmp_neighbours, info.grid_dim, centroid_count, act_neighbours, updated);
}

template <size_t N>
void run_neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, kernel_info info)
{
	size_t shared = dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	neighbours<N> << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbours, cluster_kinds);
	reduce<N> << <info.grid_dim, info.block_dim >> > (tmp_neighbours, info.grid_dim, centroid_count, act_neighbours);
}

template <size_t N>
chunk_t run_neighbours_min(const neighbour_t* neighbours, size_t count, chunk_t* result)
{
	min<N> << <1, 1024 >> > (neighbours, count, result);

	CUCH(cudaDeviceSynchronize());

	chunk_t res;
	CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}

#define INIT_TEMPLATES(N) \
template void run_neighbours<N>(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, cluster_kind* cluster_kinds, kernel_info info);\
template chunk_t run_neighbours_min<N>(const neighbour_t* neighbours, size_t count, chunk_t* result);\
template void run_update_neighbours<N>(const float* centroids, const float* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)

