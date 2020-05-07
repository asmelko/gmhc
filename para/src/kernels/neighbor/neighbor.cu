#include <kernels.cuh>

#include <device_launch_parameters.h>

#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__global__ void neighbor_min(const neighbor_t* __restrict__ neighbors, csize_t small_count, csize_t big_begin, csize_t big_count, chunk_t* __restrict__ result)
{
	static __shared__ chunk_t shared_mem[32];

	chunk_t tmp;
	tmp.min_dist = FLT_MAX;
	for (csize_t idx = threadIdx.x; idx < small_count + big_count; idx += blockDim.x)
	{
		if (idx >= small_count)
			idx += big_begin - small_count;

		if (tmp.min_dist > neighbors[idx*N].distance)
		{
			tmp.min_dist = neighbors[idx * N].distance;
			tmp.min_j = neighbors[idx * N].idx;
			tmp.min_i = idx;
		}

		if (idx >= small_count)
			idx -= big_begin - small_count;
	}

	tmp = reduce_min_block(tmp, shared_mem);

	if (threadIdx.x == 0)
		*result = tmp;
}

#include "update.cuh"
#include "reduce.cuh"
#include "neighbor_eucl.cuh"
#include "neighbor_maha.cuh"

template <csize_t N>
void run_update_neighbors(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors, 
	cluster_bound_t sizes, update_data_t upd_data, kernel_info info)
{
	csize_t shared = data.dim * sizeof(float) + 32 * sizeof(neighbor_t) * N;
	csize_t shared_mat = (data.dim + 33) * data.dim * sizeof(float) + 32 * sizeof(neighbor_t) * N;

	CUCH(cudaMemset(upd_data.eucl_update_size, 0, sizeof(csize_t)));
	CUCH(cudaMemcpy(upd_data.maha_update_size, &sizes.maha_begin, sizeof(csize_t), cudaMemcpyKind::cudaMemcpyHostToDevice));

	update<N><<<info.grid_dim, info.block_dim >>>
		(act_neighbors, upd_data.to_update, upd_data.eucl_update_size, upd_data.maha_update_size,
			sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.move_a, upd_data.move_b, upd_data.new_idx);

	if (sizes.eucl_size)
	{
		neighbors_u<N> << <info.grid_dim, info.block_dim, shared >> >
			(data.centroids, act_neighbors, upd_data.to_update, upd_data.eucl_update_size, data.dim, sizes.eucl_size, upd_data.new_idx);

		neighbors<N> << <info.grid_dim, info.block_dim, shared >> >
			(data.centroids, data.dim, sizes.eucl_size, tmp_neighbors);

		cudaDeviceSynchronize();

		run_compare_nei(act_neighbors, tmp_neighbors,
			sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.new_idx);

		cudaDeviceSynchronize();
	}

	if (sizes.maha_size)
		neighbors_mat_u<N><<<info.grid_dim, info.block_dim, shared_mat>>> 
			(data.centroids, data.inverses, act_neighbors, upd_data.to_update, upd_data.maha_update_size,
				data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.new_idx);

	//reduce<N> << <info.grid_dim, info.block_dim >> >(tmp_neighbors, info.grid_dim, sizes.eucl_size, act_neighbors);

	//reduce_u<N><<<info.grid_dim, info.block_dim>>>
	//	(tmp_neighbors, act_neighbors, upd_data.to_update, upd_data.eucl_update_size, 
	//		sizes.maha_begin, upd_data.maha_update_size, info.grid_dim);
		
	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());
}

template <csize_t N>
void run_neighbors(const float* centroids, csize_t dim, csize_t centroid_count, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors, kernel_info info)
{
	csize_t shared = dim * sizeof(float) + 32 * sizeof(neighbor_t) * N;
	neighbors<N> << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbors);
	reduce<N> << <info.grid_dim, info.block_dim >> > (tmp_neighbors, info.grid_dim, centroid_count, act_neighbors);
}

template <csize_t N>
chunk_t run_neighbors_min(const neighbor_t* neighbors, cluster_bound_t sizes, chunk_t* result)
{
	neighbor_min<N><<<1, 1024>>>(neighbors, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, result);

	CUCH(cudaDeviceSynchronize());

	chunk_t res;
	CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}

#define INIT_TEMPLATES(N) \
template void run_neighbors<N>(const float* centroids, csize_t dim, csize_t centroid_count, neighbor_t* tmp_neighbors, neighbor_t* neighbors, kernel_info info);\
template chunk_t run_neighbors_min<N>(const neighbor_t* neighbors, cluster_bound_t sizes, chunk_t* result);\
template void run_update_neighbors<N>(centroid_data_t data, neighbor_t* tmp_neighbors, neighbor_t* act_neighbors, cluster_bound_t sizes, update_data_t upd_data, kernel_info info);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)