#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

template <size_t N>
__device__ void add_neighbour(neighbour_t* neighbours, neighbour_t neighbour)
{
	neighbour_t prev_min;
	size_t i = 0;
	for (; i < N; ++i)
	{
		if (neighbours[i].distance > neighbour.distance)
		{
			prev_min = neighbours[i];
			neighbours[i] = neighbour;
			break;
		}
	}

	for (++i; i < N; i++)
	{
		if (prev_min.distance == FLT_MAX)
			return;

		neighbour_t tmp = neighbours[i];
		neighbours[i] = prev_min;
		prev_min = tmp;
	}
}

template <size_t N>
__device__ void merge_neighbours(const neighbour_t* l_neighbours, const neighbour_t* r_neighbours, neighbour_t* res)
{
	size_t l_idx = 0, r_idx = 0;

	for (size_t i = 0; i < N; ++i)
	{
		if (l_neighbours[l_idx].distance < r_neighbours[r_idx].distance)
			res[i] = l_neighbours[l_idx++];
		else
			res[i] = r_neighbours[r_idx++];
	}
}


__inline__ __device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim)
{
	float tmp_sum = 0;
	for (size_t i = 0; i < dim; ++i)
	{
		auto tmp = l_point[i] - r_point[i];
		tmp_sum += tmp * tmp;
	}
	return sqrtf(tmp_sum);
}

__inline__ __device__ void reduce_sum_warp(float* point, size_t dim)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		for (size_t i = 0; i < dim; ++i)
		{
			point[i] += __shfl_down_sync(0xFFFFFFFF, point[i], offset);
		}
	}
}


template <size_t N>
__inline__ __device__ void reduce_min_warp(neighbour_t* neighbours)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		neighbour_t tmp[N];
		for (size_t i = 0; i < N; ++i)
		{
			tmp[i].distance = __shfl_down_sync(0xFFFFFFFF, neighbours[i].distance, offset);
			tmp[i].idx = __shfl_down_sync(0xFFFFFFFF, neighbours[i].idx, offset);
		}

		neighbour_t tmp_cpy[N];
		merge_neighbours<N>(neighbours, tmp, tmp_cpy);
		memcpy(neighbours, tmp_cpy, sizeof(neighbour_t) * N);
	}
}

template <size_t N>
__inline__ __device__ void reduce_min_block(neighbour_t* neighbours, neighbour_t* shared_mem, bool reduce_warp = true)
{
	if (reduce_warp)
		reduce_min_warp<N>(neighbours);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		memcpy(shared_mem + warp_id * N, neighbours, sizeof(neighbour_t) * N);

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(neighbours, shared_mem + threadIdx.x * N, sizeof(neighbour_t) * N);
	else
		for (size_t i = 0; i < N; i++)
			neighbours[i].distance = FLT_MAX;

	reduce_min_warp<N>(neighbours);
}

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

__inline__ __device__ chunk_t reduce_min_warp(chunk_t data)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		auto tmp_dist = __shfl_down_sync(0xFFFFFFFF, data.min_dist, offset);
		auto tmp_i = __shfl_down_sync(0xFFFFFFFF, data.min_i, offset);
		auto tmp_j = __shfl_down_sync(0xFFFFFFFF, data.min_j, offset);
		if (tmp_dist < data.min_dist)
		{
			data.min_dist = tmp_dist;
			data.min_i = tmp_i;
			data.min_j = tmp_j;
		}
	}
	return data;
}

__inline__ __device__ chunk_t reduce_min_block(chunk_t data, chunk_t* shared_mem)
{
	data = reduce_min_warp(data);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		shared_mem[warp_id] = data;

	__syncthreads();

	data = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[threadIdx.x] : shared_mem[0];

	data = reduce_min_warp(data);
	return data;
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

__global__ void print_up(uint8_t* updated, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("%d. %d\n", (int)i, updated[i]);
	}
}

__global__ void print_ne(neighbour_t* neighbours, size_t nei_number, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("%d. %f %d\n", (int)i, neighbours[i * nei_number].distance, (int)neighbours[i* nei_number].idx);
	}
}

void print_nei(neighbour_t* neighbours, size_t nei_number, size_t count)
{
	print_ne << <1, 1 >> > (neighbours, nei_number, count);
}

#include "neighbour_eucl.cu"
#include "neighbour_maha.cu"

template <size_t N>
void run_update_neighbours(const float* centroids, const float*const* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info)
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

template void run_neighbours<1>(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template void run_neighbours<2>(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template void run_neighbours<5>(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template chunk_t run_neighbours_min<1>(const neighbour_t* neighbours, size_t count, chunk_t* result);
template chunk_t run_neighbours_min<2>(const neighbour_t* neighbours, size_t count, chunk_t* result);
template chunk_t run_neighbours_min<5>(const neighbour_t* neighbours, size_t count, chunk_t* result);
template void run_update_neighbours<1>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
template void run_update_neighbours<2>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
template void run_update_neighbours<5>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
