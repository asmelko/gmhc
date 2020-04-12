#include <kernels.cuh>

#include <device_launch_parameters.h>

#include "neighbor_common.cuh"

using namespace clustering;

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

#include "update.cuh"
#include "reduce.cuh"
#include "neighbour_eucl.cuh"
#include "neighbour_maha.cuh"

template <csize_t N>
void run_update_neighbours(centroid_data_t data, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, 
	cluster_bound_t sizes, update_data_t upd_data, kernel_info info, cudaStream_t* streams)
{
	csize_t shared = data.dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;
	csize_t shared_mat = (data.dim + 33) * data.dim * sizeof(float) + 32 * sizeof(neighbour_t) * N;

	CUCH(cudaMemsetAsync(upd_data.eucl_update_size, 0, sizeof(csize_t), streams[0]));
	CUCH(cudaMemcpyAsync(upd_data.maha_update_size, &sizes.maha_begin, sizeof(csize_t), cudaMemcpyKind::cudaMemcpyHostToDevice, streams[1]));

	CUCH(cudaStreamSynchronize(streams[0]));
	CUCH(cudaStreamSynchronize(streams[1]));

	update_neighbours<N><<<info.grid_dim, info.block_dim >>>
		(act_neighbours, upd_data.to_update, upd_data.eucl_update_size, upd_data.maha_update_size,
			sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.move_a, upd_data.move_b, upd_data.new_idx);

	CUCH(cudaStreamSynchronize(0));

	if (sizes.eucl_size)
		neighbours_u<N><<<info.grid_dim, info.block_dim, shared, streams[0]>>>
			(data.centroids, tmp_neighbours, upd_data.to_update, upd_data.eucl_update_size, data.dim, sizes.eucl_size, upd_data.new_idx);

	if (sizes.maha_size)
		neighbours_mat_u<N><<<info.grid_dim, info.block_dim, shared_mat, streams[1]>>>
			(data.centroids, data.inverses, tmp_neighbours, upd_data.to_update, upd_data.maha_update_size,
				data.dim, sizes.eucl_size, sizes.maha_begin, sizes.maha_size, upd_data.new_idx);

	CUCH(cudaStreamSynchronize(streams[0]));
	CUCH(cudaStreamSynchronize(streams[1]));

	reduce_u<N><<<info.grid_dim, info.block_dim>>>
		(tmp_neighbours, act_neighbours, upd_data.to_update, upd_data.eucl_update_size, 
			sizes.maha_begin, upd_data.maha_update_size, info.grid_dim);
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
template void run_update_neighbours<N>(centroid_data_t data, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_bound_t sizes, update_data_t upd_data, kernel_info info, cudaStream_t* streams);

INIT_TEMPLATES(1)
INIT_TEMPLATES(2)
INIT_TEMPLATES(3)
INIT_TEMPLATES(5)
INIT_TEMPLATES(10)