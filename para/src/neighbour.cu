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
			point[i] += __shfl_down_sync(__activemask(), point[i], offset);
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
__inline__ __device__ void reduce_min_block(neighbour_array_t<N>* neighbours, neighbour_array_t<N>* shared_mem)
{
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
__global__ void reduce(const neighbour_array_t<N>* neighbours, size_t to_reduce, size_t count, neighbour_array_t<N>* reduced)
{
	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		size_t block = idx / warpSize;
		neighbour_array_t<N> local;

		size_t nei = idx % warpSize;

		if (nei < to_reduce)
			local = neighbours[block * to_reduce + nei];
		else
			for (size_t i = 0; i < N; i++)
				local.neighbours[i].distance = FLT_MAX;


		for (nei += warpSize; nei < to_reduce; nei+=warpSize)
			local = merge_neighbours(&local, neighbours + block * to_reduce + nei);


		reduce_min_warp(&local);


		if (threadIdx.x % warpSize == 0)
		{
			reduced[block] = local;
		}
	}
}

template <size_t N>
__global__ void neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* neighbours_a)
{
	extern __shared__ float shared_mem[];

	for (asgn_t x = 0; x < centroid_count; ++x)
	{
		neighbour_array_t<N> local_neighbours;

		for (size_t i = 0; i < N; ++i)
			local_neighbours.neighbours[i].distance = FLT_MAX;

		for (size_t i = threadIdx.x; i < dim; i += blockDim.x)
			shared_mem[i] = centroids[x * dim + i];

		__syncthreads();

		for (asgn_t y = x + 1 + threadIdx.x + blockIdx.x * blockDim.x; y < centroid_count; y += blockDim.x * gridDim.x)
		{
			float dist = euclidean_norm(shared_mem, centroids + y * dim, dim);

			add_neighbour(&local_neighbours, neighbour_t{ dist, y });
		}

		reduce_min_block(&local_neighbours, reinterpret_cast<neighbour_array_t<N>*>(shared_mem + dim));

		if (threadIdx.x == 0)
		{
			neighbours_a[gridDim.x * x + blockIdx.x] = local_neighbours;
		}
	}
}



template <size_t N>
__global__ void neighbours_mat(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, size_t start, neighbour_array_t<N>* neighbours)
{
	extern __shared__ float shared_mem[];

	float* this_centroid = shared_mem;
	float* this_icov = shared_mem + dim;
	float* curr_centroid = this_icov + dim * dim;

	for (asgn_t x = start; x < centroid_count; ++x)
	{
		neighbour_array_t<N> local_neighbours;

		for (size_t i = 0; i < N; ++i)
			local_neighbours.neighbours[i].distance = FLT_MAX;

		for (size_t i = threadIdx.x; i < dim + dim * dim; i += blockDim.x)
			if (i < dim)
				shared_mem[i] = centroids[x * dim + i];
			else
				shared_mem[i] = inverses[x][i - dim];

		__syncthreads();

		for (asgn_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < (centroid_count - start) * warpSize; idx += blockDim.x * gridDim.x)
		{
			size_t y = idx / warpSize;

			size_t warp_in_block = threadIdx.x / warpSize;
			size_t thread_in_warp = threadIdx.x % warpSize;

			for (size_t i = thread_in_warp; i < dim; i += warpSize)
				curr_centroid[warp_in_block * dim + i] = centroids[y * dim + i];

			float dist = euclidean_norm(this_centroid, curr_centroid + warp_in_block * dim, dim);

			for (size_t i = thread_in_warp; i < dim; i += warpSize)
				curr_centroid[warp_in_block * dim + i] -= this_centroid[i];

			float tmp_point[MAX_DIM];
			for (size_t i = 0; i < MAX_DIM; i++)
				tmp_point[i] = 0;

			for (size_t i = thread_in_warp; i < dim * dim; i += warpSize)
			{
				size_t mat_x = i / dim;
				size_t mat_y = i % dim;

				tmp_point[x] += this_icov[x + y * dim] * curr_centroid[warp_in_block * dim + x];
			}

			reduce_sum_warp(tmp_point, dim);

			if (threadIdx.x == 0)
			{
				float tmp_res = 0;
				for (size_t i = 0; i < dim; ++i)
					tmp_res += tmp_point[i] * curr_centroid[warp_in_block * dim + i];

				tmp_res = sqrtf(tmp_res);

				dist = (dist + tmp_res) / 2;

				add_neighbour(&local_neighbours, neighbour_t{ dist, y });
			}
		}

		for (asgn_t idx = threadIdx.x + blockIdx.x * blockDim.x + (x + 1) * warpSize; idx < centroid_count * warpSize; idx += blockDim.x * gridDim.x)
		{
			size_t y = idx / warpSize;

			size_t warp_in_block = threadIdx.x / warpSize;
			size_t thread_in_warp = threadIdx.x % warpSize;

			float* curr_icov = this_icov + dim * dim;
			curr_centroid = curr_icov + dim * dim;

			for (size_t i = threadIdx.x; i < dim*dim; i+=blockDim.x)
				curr_icov[i] = inverses[y][i];

			for (size_t i = thread_in_warp; i < dim; i += warpSize)
				curr_centroid[warp_in_block * dim + i] = centroids[y * dim + i] - this_centroid[i];

			float* icov = this_icov;
			float distance = 0;
			while (true)
			{
				float tmp_point[MAX_DIM];
				for (size_t i = 0; i < MAX_DIM; i++)
					tmp_point[i] = 0;

				for (size_t i = thread_in_warp; i < dim * dim; i += warpSize)
				{
					size_t mat_x = i / dim;
					size_t mat_y = i % dim;

					tmp_point[x] += icov[x + y * dim] * curr_centroid[warp_in_block * dim + x];
				}

				reduce_sum_warp(tmp_point, dim);

				if (threadIdx.x == 0)
				{
					float tmp_res = 0;
					for (size_t i = 0; i < dim; ++i)
						tmp_res += tmp_point[i] * curr_centroid[warp_in_block * dim + i];

					distance += sqrtf(tmp_res);
				}

				if (icov == curr_icov)
					break;
				else
					icov = curr_icov;
			}

			if (threadIdx.x == 0)
				add_neighbour(&local_neighbours, neighbour_t{ distance / 2, y });
		}

		reduce_min_block(&local_neighbours, reinterpret_cast<neighbour_array_t<N>*>(shared_mem));

		if (threadIdx.x == 0)
			neighbours[gridDim.x * x + blockIdx.x] = local_neighbours;
	}
}

__inline__ __device__ clustering::chunk_t reduce_min_warp(clustering::chunk_t data)
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

__inline__ __device__ clustering::chunk_t reduce_min_block(clustering::chunk_t data, clustering::chunk_t* shared_mem)
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
	for (size_t idx = threadIdx.x; idx < count; idx+=blockDim.x)
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

template <size_t N>
void run_neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<N>* tmp_neighbours, neighbour_array_t<N>* act_neighbours, chunk_t* result, kernel_info info)
{
	size_t shared = dim * sizeof(float) + info.grid_dim * sizeof(neighbour_array_t<N>);
	neighbours << <info.grid_dim, info.block_dim, shared>> > (centroids, dim, centroid_count, tmp_neighbours);
	reduce << <info.grid_dim, info.block_dim >> > (tmp_neighbours, info.grid_dim, centroid_count, act_neighbours);
	min<<<1,1024>>>(act_neighbours, centroid_count, result);
}

template void run_neighbours<1>(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<1>* tmp_neighbours, neighbour_array_t<1>* neighbours, chunk_t* result, kernel_info info);
template void run_neighbours<5>(const float* centroids, size_t dim, size_t centroid_count, neighbour_array_t<5>* tmp_neighbours, neighbour_array_t<5>* neighbours, chunk_t* result, kernel_info info);