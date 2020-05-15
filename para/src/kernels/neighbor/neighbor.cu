#include "kernels.cuh"

#include "device_launch_parameters.h"

#include "neighbor_common.cuh"
#include <algorithm>

using namespace clustering;

template <csize_t N>
__global__ void neighbor_min(const neighbor_t* __restrict__ neighbors, csize_t small_count, csize_t big_begin, csize_t big_count, chunk_t* __restrict__ result)
{
	static __shared__ chunk_t shared_mem[32];

	chunk_t tmp;
	tmp.min_dist = FLT_INF;
	for (csize_t idx = threadIdx.x; idx < small_count + big_count; idx += blockDim.x)
	{
		if (idx >= small_count)
			idx += big_begin - small_count;

		if (tmp.min_dist > neighbors[idx * N].distance)
		{
			tmp.min_dist = neighbors[idx * N].distance;
			tmp.min_j = neighbors[idx * N].idx;
			tmp.min_i = idx;
		}
	}

	tmp = reduce_min_block(tmp, shared_mem);

	if (threadIdx.x == 0)
		*result = tmp;
}

#include "update.cuh"
#include "reduce.cuh"
#include "neighbor_eucl.cuh"
#include "neighbor_new.cuh"

template <csize_t N>
void run_update_neighbors(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors,
	cluster_bound_t sizes, update_data_t upd_data, kernel_info info)
{
	csize_t shared = data.dim * sizeof(float) + 32 * sizeof(neighbor_t) * N;
	csize_t shared_new = (data.dim + 33) * data.dim * sizeof(float);
	csize_t shared_mat = std::max(shared_new, 32 * (csize_t)sizeof(neighbor_t) * N);

	CUCH(cudaMemset(upd_data.eucl_update_size, 0, sizeof(csize_t)));
	CUCH(cudaMemcpy(upd_data.maha_update_size, &sizes.maha_begin, sizeof(csize_t), cudaMemcpyKind::cudaMemcpyHostToDevice));

	update<N> << <info.grid_dim, info.block_dim >> >
		(act_neighbors, upd_data.to_update, upd_data.eucl_update_size, upd_data.maha_update_size,
			sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.move_a, upd_data.move_b, upd_data.new_idx);

	if (sizes.eucl_size)
		neighbors_u<N> << <info.grid_dim, info.block_dim, shared >> >
		(data.centroids, tmp_neighbors, 
			upd_data.to_update, upd_data.eucl_update_size, data.dim, sizes.eucl_size);

	if (sizes.maha_size)
		neighbors_mat_u<N> << <info.grid_dim, info.block_dim, shared_mat >> >
		(data.centroids, data.inverses, tmp_neighbors, upd_data.to_update, upd_data.maha_update_size,
			data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size);

	reduce_u<N> << <info.grid_dim, info.block_dim >> >
		(tmp_neighbors, act_neighbors, upd_data.to_update, upd_data.eucl_update_size,
			sizes.maha_begin, upd_data.maha_update_size, info.grid_dim);

	neighbors_new_u<N><<<info.grid_dim, info.block_dim, shared_new >>> (data.centroids, data.inverses, act_neighbors,
		data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.new_idx);
}

template <csize_t N>
void run_neighbors(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors,
	cluster_bound_t sizes, kernel_info info)
{
	csize_t shared = data.dim * sizeof(float) + 32 * sizeof(neighbor_t) * N;
	csize_t shared_new = (data.dim + 33) * data.dim * sizeof(float);
	csize_t shared_mat = std::max(shared_new, 32 * (csize_t)sizeof(neighbor_t) * N);

	if (sizes.eucl_size)
		neighbors<N> << <info.grid_dim, info.block_dim, shared >> > (data.centroids, data.dim, sizes.eucl_size, tmp_neighbors);

	if (sizes.maha_size)
		neighbors_mat<N> << <info.grid_dim, info.block_dim, shared_mat >> > (data.centroids, data.inverses, tmp_neighbors,
			data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size);

	reduce<N> << <info.grid_dim, info.block_dim >> > (tmp_neighbors, act_neighbors, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, info.grid_dim);
}

template <csize_t N>
chunk_t run_neighbors_min(const neighbor_t* neighbors, cluster_bound_t sizes, chunk_t* result)
{
	neighbor_min<N> << <1, 1024 >> > (neighbors, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, result);

	CUCH(cudaDeviceSynchronize());

	chunk_t res;
	CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}

#define INIT_TEMPLATES(N) \
template void run_neighbors<N>(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors, cluster_bound_t sizes, kernel_info info);\
template chunk_t run_neighbors_min<N>(const neighbor_t* neighbors, cluster_bound_t sizes, chunk_t* result);\
template void run_update_neighbors<N>(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors, cluster_bound_t sizes, update_data_t upd_data, kernel_info info);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)