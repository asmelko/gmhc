#include <kernels.cuh>

#include <device_launch_parameters.h>

#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__inline__ __device__ void point_reduce
(const neighbour_t* __restrict__ neighbours, csize_t to_reduce, neighbour_t* __restrict__ reduced, csize_t idx)
{
	csize_t block = idx / warpSize;
	neighbour_t local[N];

	csize_t nei = idx % warpSize;

	if (nei < to_reduce)
		memcpy(local, neighbours + (block * to_reduce + nei) * N, sizeof(neighbour_t) * N);
	else
		for (csize_t i = 0; i < N; i++)
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


template <csize_t N>
__global__ void reduce(const neighbour_t* __restrict__ neighbours, csize_t to_reduce, csize_t count, neighbour_t* __restrict__  reduced)
{
	for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbours, to_reduce, reduced, idx);
	}
}

template <csize_t N>
__global__ void update_neighbours(neighbour_t* __restrict__  neighbours_a, flag_t* __restrict__ updated, 
	csize_t small_size, csize_t big_begin, csize_t big_size, pasgn_t move_a, pasgn_t move_b, csize_t new_idx)
{
	for (csize_t idx = threadIdx.x + blockDim.x * blockIdx.x;
		idx < small_size + big_size;
		idx += blockDim.x * gridDim.x)
	{
		if (idx >= small_size)
			idx += big_begin - small_size;

		if (idx == move_a.first || idx == move_b.first || idx == new_idx)
		{
			updated[idx] = 1;
			continue;
		}

		neighbour_t tmp_nei[N];
		memcpy(tmp_nei, neighbours_a + idx * N, sizeof(neighbour_t) * N);

		csize_t last_empty = 0;

		for (csize_t i = 0; i < N; i++)
		{
			if (tmp_nei[i].distance == FLT_MAX)
				break;

			if (tmp_nei[i].idx == move_a.first || tmp_nei[i].idx == move_b.first)
				tmp_nei[i].distance = FLT_MAX;
			else
			{
				if (tmp_nei[i].idx == move_a.second)
					tmp_nei[i].idx = move_a.first;
				else if (tmp_nei[i].idx == move_b.second)
					tmp_nei[i].idx = move_b.first;

				tmp_nei[last_empty++] = tmp_nei[i];
			}
		}

		for (csize_t i = last_empty; i < N; i++)
			tmp_nei[i].distance = FLT_MAX;

		updated[idx] = tmp_nei[0].distance == FLT_MAX ? 1 : 0;

		memcpy(neighbours_a + idx * N, tmp_nei, sizeof(neighbour_t) * N);
	}
}

template <csize_t N>
__global__ void reduce_u(const neighbour_t* __restrict__ neighbours, neighbour_t* __restrict__ reduced, flag_t* __restrict__ updated,
	csize_t to_reduce, csize_t small_count, csize_t big_begin, csize_t big_count)
{
	csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (; idx < small_count * warpSize; idx += blockDim.x * gridDim.x)
	{
		if (updated[idx / warpSize])
			point_reduce<N>(neighbours, to_reduce, reduced, idx);
	}

	idx += (big_begin - small_count) * warpSize;

	for (; idx < (big_begin + big_count) * warpSize; idx += blockDim.x * gridDim.x)
	{
		if (updated[idx / warpSize])
			point_reduce<N>(neighbours, to_reduce, reduced, idx);
	}
}

template <csize_t N>
__global__ void min(const neighbour_t* __restrict__ neighbours, csize_t small_count, csize_t big_begin, csize_t big_count, chunk_t* __restrict__ result)
{
	static __shared__ chunk_t shared_mem[32];

	chunk_t tmp;
	tmp.min_dist = FLT_MAX;
	for (csize_t idx = threadIdx.x; idx < small_count + big_count; idx += blockDim.x)
	{
		if (idx >= small_count)
			idx += big_begin - small_count;

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

template <csize_t N>
void run_update_neighbours(centroid_data_t data, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, 
	cluster_bound_t sizes, update_data_t upd_data, kernel_info info)
{
	csize_t shared = data.dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	csize_t shared_mat = (data.dim + 33) * data.dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;

	update_neighbours<N><<<info.grid_dim, info.block_dim >>>
		(act_neighbours, upd_data.to_update, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.move_a, upd_data.move_b, upd_data.new_idx);

	if (sizes.eucl_size)
		neighbours_u<N><<<info.grid_dim, info.block_dim, shared>>>
			(data.centroids, data.dim, sizes.eucl_size, tmp_neighbours, upd_data.to_update, upd_data.new_idx);

	if (sizes.maha_size)
		neighbours_mat_u<N><<<info.grid_dim, info.block_dim, shared_mat>>> 
			(data.centroids, data.inverses, tmp_neighbours, upd_data.to_update, data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.new_idx);

	reduce_u<N><<<info.grid_dim, info.block_dim>>>
		(tmp_neighbours, act_neighbours, upd_data.to_update, info.grid_dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size);
		
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

template <csize_t N>
void run_neighbours(const float* centroids, csize_t dim, csize_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, kernel_info info)
{
	csize_t shared = dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	neighbours<N> << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbours);
	reduce<N> << <info.grid_dim, info.block_dim >> > (tmp_neighbours, info.grid_dim, centroid_count, act_neighbours);
}

template <csize_t N>
chunk_t run_neighbours_min(const neighbour_t* neighbours, cluster_bound_t sizes, chunk_t* result)
{
	min<N><<<1, 1024>>>(neighbours, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, result);

	CUCH(cudaDeviceSynchronize());

	chunk_t res;
	CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}

#define INIT_TEMPLATES(N) \
template void run_neighbours<N>(const float* centroids, csize_t dim, csize_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, kernel_info info);\
template chunk_t run_neighbours_min<N>(const neighbour_t* neighbours, cluster_bound_t sizes, chunk_t* result);\
template void run_update_neighbours<N>(centroid_data_t data, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_bound_t sizes, update_data_t upd_data, kernel_info info);


INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)

