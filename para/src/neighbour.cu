#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

template <size_t N>
__device__ void add_neighbour(neighbour_array_t<N>* neighbours, neighbour_t neighbour)
{
	neighbour_t prev_min;
	size_t i = 0;
	for (; i < N; ++i)
	{
		if (neighbours->neighbours[i].distance > neighbour.distance)
		{
			prev_min = neighbours->neighbours[i];
			neighbours->neighbours[i] = neighbour;
			break;
		}
	}

	for (++i; i < N; i++)
	{
		if (prev_min.distance == FLT_MAX)
			return;

		neighbour_t tmp = neighbours->neighbours[i];
		neighbours->neighbours[i] = prev_min;
		prev_min = tmp;
	}
}

template <size_t N>
__device__ neighbour_array_t<N> merge_neighbours(const neighbour_array_t<N>* l_neighbours, const neighbour_array_t<N>* r_neighbours)
{
	neighbour_array_t<N> tmp;

	size_t l_idx = 0, r_idx = 0;

	for (size_t i = 0; i < N; ++i)
	{
		if (l_neighbours->neighbours[l_idx].distance < r_neighbours->neighbours[r_idx].distance)
			tmp.neighbours[i] = l_neighbours->neighbours[l_idx++];
		else
			tmp.neighbours[i] = r_neighbours->neighbours[r_idx++];
	}

	return tmp;
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
__inline__ __device__ void reduce_min_warp(neighbour_array_t<N>* neighbours)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		neighbour_array_t<N> tmp;
		for (size_t i = 0; i < N; ++i)
		{
			tmp.neighbours[i].distance = __shfl_down_sync(0xFFFFFFFF, neighbours->neighbours[i].distance, offset);
			tmp.neighbours[i].idx = __shfl_down_sync(0xFFFFFFFF, neighbours->neighbours[i].idx, offset);
		}

		*neighbours = merge_neighbours(neighbours, &tmp);
	}
}

template <size_t N>
__inline__ __device__ void reduce_min_block(neighbour_array_t<N>* neighbours, neighbour_array_t<N>* shared_mem, bool reduce_warp = true)
{
	if (reduce_warp)
		reduce_min_warp(neighbours);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		shared_mem[warp_id] = *neighbours;

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		*neighbours = shared_mem[threadIdx.x];
	else
		for (size_t i = 0; i < N; i++)
			neighbours->neighbours[i].distance = FLT_MAX;

	reduce_min_warp(neighbours);
}

template <size_t N>
__inline__ __device__ void point_reduce(const neighbour_array_t<N>* neighbours, size_t to_reduce, size_t count, neighbour_array_t<N>* reduced, size_t idx)
{
	size_t block = idx / warpSize;
	neighbour_array_t<N> local;

	size_t nei = idx % warpSize;

	if (nei < to_reduce)
		local = neighbours[block * to_reduce + nei];
	else
		for (size_t i = 0; i < N; i++)
			local.neighbours[i].distance = FLT_MAX;


	for (nei += warpSize; nei < to_reduce; nei += warpSize)
		local = merge_neighbours(&local, neighbours + block * to_reduce + nei);


	reduce_min_warp(&local);


	if (threadIdx.x % warpSize == 0)
	{
		reduced[block] = local;
	}
}


template <size_t N>
__global__ void reduce(const neighbour_array_t<N>* neighbours, size_t to_reduce, size_t count, neighbour_array_t<N>* reduced)
{
	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce(neighbours, to_reduce, count, reduced, idx);
	}
}

template <size_t N>
__device__ void point_neighbour(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours_a, float* shared_mem, cluster_kind* cluster_kinds, size_t idx, bool from_start)
{
	neighbour_array_t<N> local_neighbours;

	for (size_t i = 0; i < N; ++i)
		local_neighbours.neighbours[i].distance = FLT_MAX;

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

			add_neighbour(&local_neighbours, neighbour_t{ dist, y });
		}
	else
		y += idx;

	for (; y < centroid_count - 1; y += blockDim.x * gridDim.x)
	{
		if (cluster_kinds[y+1] != cluster_kind::EUCL)
			continue;

		//printf("cycle %d %d\n", (int)y, (int)new_y);


		float dist = euclidean_norm(shared_mem, centroids + (y + 1) * dim, dim);

		add_neighbour(&local_neighbours, neighbour_t{ dist, y + 1 });
	}

	reduce_min_block(&local_neighbours, reinterpret_cast<neighbour_array_t<N>*>(shared_mem + dim));

	if (threadIdx.x == 0)
	{
		neighbours_a[gridDim.x * idx + blockIdx.x] = local_neighbours;
	}
}

template <size_t N>
__global__ void neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours_a, cluster_kind* cluster_kinds)
{
	extern __shared__ float shared_mem[];

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (cluster_kinds[x] != cluster_kind::EUCL)
			continue;

		point_neighbour(centroids, dim, centroid_count, neighbours_a, shared_mem, cluster_kinds, x, false);
	}
}

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
(const float* centroids, const float* const* inverses, size_t dim, neighbour_array_t<N>* neighbours, float* curr_centroid, 
	const float* this_centroid,const float* this_icov,const cluster_kind* cluster_kinds, asgn_t idx, bool eucl = false)
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
		add_neighbour(neighbours, neighbour_t{ dist / 2, idx });
}

template <size_t N>
__inline__ __device__ void point_neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours, cluster_kind* cluster_kinds, float* shared_mem, asgn_t x, bool from_start)
{
	//extern __shared__ float shared_mem[];

	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = shared_mem + dim + dim * dim;

	neighbour_array_t<N> local_neighbours;

	for (size_t i = 0; i < N; ++i)
		local_neighbours.neighbours[i].distance = FLT_MAX;

	

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

			point_neighbours_mat_warp(centroids, inverses, dim, &local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y, true);
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

				if (threadIdx.x  % warpSize == 0 && x == 8)
					printf("was from start\n");

				size_t y = idx / warpSize;

				point_neighbours_mat_warp(centroids, inverses, dim, &local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y);
			}
		else
			idx += x * warpSize;

		for (; idx < (centroid_count - 1) * warpSize; idx += blockDim.x * gridDim.x)
		{
			size_t y = (idx / warpSize) + 1;


			if (threadIdx.x % warpSize == 0 && x == 8)
				printf("was here %d\n",(int)y);

			point_neighbours_mat_warp(centroids, inverses, dim, &local_neighbours, curr_centroid, this_centroid, this_icov, cluster_kinds, (asgn_t)y);
		}
	}

	reduce_min_block(&local_neighbours, reinterpret_cast<neighbour_array_t<N>*>(shared_mem + (33 + dim) * dim), false);

	if (threadIdx.x == 0)
	{
		if (needs_merge)
		{
			auto tmp = neighbours[gridDim.x * x + blockIdx.x];
			local_neighbours = merge_neighbours(&tmp, &local_neighbours);
		}
		neighbours[gridDim.x * x + blockIdx.x] = local_neighbours;
	}
}


template <size_t N>
__global__ void neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours, cluster_kind* cluster_kinds, kernel_info info)
{
	size_t shared_mat = (dim + 33) * dim * sizeof(float) + 32 * sizeof(neighbour_array_t<N>);
	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (cluster_kinds[x] != cluster_kind::MAHA)
			continue;

		point_neighbours_mat<<<info.grid_dim, info.block_dim, shared_mat >>>(centroids, inverses, dim, centroid_count, neighbours, cluster_kinds, x);
	}
}

template <size_t N>
__global__ void neighbours_mat_u(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t new_idx)
{
	extern __shared__ float shared_mem[];
	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (!updated[x] || (cluster_kinds[x] != cluster_kind::MAHA && x != new_idx))
			continue;

		point_neighbours_mat(centroids, inverses, dim, centroid_count, neighbours, cluster_kinds,shared_mem, x, x == new_idx);
	}
}

template <size_t N>
__global__ void update_neighbours(size_t centroid_count, neighbour_array_t<N>* neighbours_a, uint8_t* updated, size_t old_i, size_t old_j)
{
	extern __shared__ float shared_mem[];

	auto idx = threadIdx.x + blockDim.x * blockIdx.x;

	for (asgn_t x = idx; x < centroid_count; x += blockDim.x * gridDim.x)
	{
		if (x == old_i)
		{
			updated[x] = 1;
			continue;
		}

		auto tmp_nei = neighbours_a[x];

		size_t last_empty = 0;

		for (size_t i = 0; i < N; i++)
		{
			if (tmp_nei.neighbours[i].distance == FLT_MAX)
				break;

			if (tmp_nei.neighbours[i].idx == old_i || tmp_nei.neighbours[i].idx == old_j)
				tmp_nei.neighbours[i].distance = FLT_MAX;
			else
			{
				if (tmp_nei.neighbours[i].idx == centroid_count)
					tmp_nei.neighbours[i].idx = old_j;

				tmp_nei.neighbours[last_empty++] = tmp_nei.neighbours[i];
			}
		}

		updated[x] = tmp_nei.neighbours[0].distance == FLT_MAX ? 1 : 0;

		neighbours_a[x] = tmp_nei;
	}
}

template <size_t N>
__global__ void neighbours_u(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours_a, cluster_kind* cluster_kinds, uint8_t* updated, size_t new_idx)
{
	extern __shared__ float shared_mem[];

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		if (!updated[x] || cluster_kinds[x] != cluster_kind::EUCL)
			continue;

		point_neighbour(centroids, dim, centroid_count, neighbours_a, shared_mem, cluster_kinds, x, x == new_idx);
	}
}

template <size_t N>
__global__ void reduce_u(const neighbour_array_t<N>* neighbours, size_t to_reduce, size_t count, neighbour_array_t<N>* reduced, uint8_t* updated)
{
	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		if (updated[idx / warpSize])
			point_reduce(neighbours, to_reduce, count, reduced, idx);
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
__global__ void min(const neighbour_array_t<N>* neighbours, size_t count, chunk_t* result)
{
	static __shared__ chunk_t shared_mem[32];

	chunk_t tmp;
	tmp.min_dist = FLT_MAX;
	for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x)
	{
		if (tmp.min_dist > neighbours[idx].neighbours[0].distance)
		{
			tmp.min_dist = neighbours[idx].neighbours[0].distance;
			tmp.min_j = neighbours[idx].neighbours[0].idx;
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

template <size_t N>
__global__ void print_ne(neighbour_array_t<N>* neighbours, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("%d. %f %d\n", (int)i, neighbours[i].neighbours[0].distance, (int)neighbours[i].neighbours[0].idx);
	}
}

void print_nei(neighbour_array_t<1>* neighbours, size_t count)
{
	print_ne << <1, 1 >> > (neighbours, count);
}

template <size_t N>
void run_update_neighbours(const float* centroids, const float*const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<N>* tmp_neighbours, neighbour_array_t<N>* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info)
{
	size_t shared = dim * sizeof(float) + 32 * sizeof(neighbour_array_t<N>);
	size_t shared_mat = (dim + 33) * dim * sizeof(float) + 32 * sizeof(neighbour_array_t<N>);
	update_neighbours<<<info.grid_dim, info.block_dim >>>(centroid_count, act_neighbours, updated, old_i, old_j);


	if (old_i == 18 && old_j == 23)
	{
		cudaDeviceSynchronize();
		print_up<<<1,1>>>(updated, centroid_count);
		cudaDeviceSynchronize();
	}

	neighbours_u << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbours, cluster_kinds, updated, old_i);


	neighbours_mat_u << <info.grid_dim, info.block_dim, shared_mat >> > (centroids, inverses, dim, centroid_count, tmp_neighbours, cluster_kinds, updated, old_i);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	reduce_u<<<info.grid_dim, info.block_dim>>>(tmp_neighbours, info.grid_dim, centroid_count, act_neighbours, updated);
}

template <size_t N>
void run_neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* tmp_neighbours, neighbour_array_t<N>* act_neighbours, cluster_kind* cluster_kinds, kernel_info info)
{
	size_t shared = dim * sizeof(float) + 32 * sizeof(neighbour_array_t<N>);
	neighbours << <info.grid_dim, info.block_dim, shared >> > (centroids, dim, centroid_count, tmp_neighbours, cluster_kinds);
	reduce << <info.grid_dim, info.block_dim >> > (tmp_neighbours, info.grid_dim, centroid_count, act_neighbours);
}

template <size_t N>
chunk_t run_neighbours_min(const neighbour_array_t<N>* neighbours, size_t count, chunk_t* result)
{
	min << <1, 1024 >> > (neighbours, count, result);

	CUCH(cudaDeviceSynchronize());

	chunk_t res;
	CUCH(cudaMemcpy(&res, result, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}

template void run_neighbours<1>(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<1>* tmp_neighbours, neighbour_array_t<1>* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template void run_neighbours<2>(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<2>* tmp_neighbours, neighbour_array_t<2>* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template void run_neighbours<5>(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<5>* tmp_neighbours, neighbour_array_t<5>* neighbours, cluster_kind* cluster_kinds, kernel_info info);
template chunk_t run_neighbours_min<1>(const neighbour_array_t<1>* neighbours, size_t count, chunk_t* result);
template chunk_t run_neighbours_min<2>(const neighbour_array_t<2>* neighbours, size_t count, chunk_t* result);
template chunk_t run_neighbours_min<5>(const neighbour_array_t<5>* neighbours, size_t count, chunk_t* result);
template void run_update_neighbours<1>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<1>* tmp_neighbours, neighbour_array_t<1>* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
template void run_update_neighbours<2>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<2>* tmp_neighbours, neighbour_array_t<2>* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
template void run_update_neighbours<5>(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_array_t<5>* tmp_neighbours, neighbour_array_t<5>* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);
