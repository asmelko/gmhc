#include <kernels.cuh>

#include <device_launch_parameters.h>
#include <iostream>
#include <cfloat>

#include "common_kernels.cuh"

using namespace clustering;

void cuda_check(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(code) << " at " << file << ":" << line;
		exit(code);
	}
}

void cuBLAS_check(cublasStatus_t code, const char* file, int line)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		switch (code)
		{
		case CUBLAS_STATUS_NOT_INITIALIZED:
			std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_ALLOC_FAILED:
			std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_INVALID_VALUE:
			std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_MAPPING_ERROR:
			std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << " at " << file << ":" << line;
			return;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << " at " << file << ":" << line;
			return;
		}
		std::cerr << "Unknown cuBLAS error" << " at " << file << ":" << line;
	}
}

__global__ void print_a(asgn_t* assignments, csize_t point_size)
{
	for (csize_t i = 0; i < point_size; i++)
		printf("%d: %d\n", (int)i, (int)assignments[i]);
}

void run_print_assg(asgn_t* assignments, csize_t point_size)
{
	print_a << <1, 1 >> > (assignments, point_size);
}

__global__ void print_centroid(const float* centroid, csize_t dim, csize_t count)
{
	for (csize_t i = 0; i < count; i++)
	{
		printf("%d. ", (int)i);
		for (csize_t j = 0; j < dim; j++)
		{
			printf("%f ", centroid[i * dim + j]);
		}
		printf("\n");
	}
}

void run_print_centroid(const float* centroid, csize_t dim, csize_t count)
{
	print_centroid << <1, 1 >> > (centroid, dim, count);
}


__device__ void print_point(const float* data, csize_t x, csize_t dim)
{
	for (csize_t i = 0; i < dim; i++)
	{
		printf("%f ", data[x * dim + i]);
	}
	printf("\n");
}

__device__ void print_min(const chunk_t* output)
{
	printf("%f %d %d\n", output->min_dist, output->min_i, output->min_j);
}



__global__ void print_up(csize_t* updated, csize_t* eucl_count, csize_t maha_begin, csize_t* maha_count)
{
	printf("update:\n");
	for (csize_t i = 0; i < *eucl_count; i++)
		printf("eucl %d\n", updated[i]);

	for (csize_t i = maha_begin; i < *maha_count; i++)
		printf("maha %d\n", updated[i]);
}

void run_print_up(csize_t* updated, csize_t* eucl_count, csize_t maha_begin, csize_t* maha_count)
{
	print_up << <1, 1 >> > (updated, eucl_count, maha_begin, maha_count);
}

__global__ void print_ne(neighbor_t* neighbors, csize_t nei_number, csize_t count)
{
	for (csize_t i = 0; i < count; i++)
	{
		for (csize_t j = 0; j < nei_number; j++)
		{
			printf("%d. %f %d\n", (int)i, neighbors[i * nei_number + j].distance, (int)neighbors[i * nei_number + j].idx);
		}
	}
}

void print_nei(neighbor_t* neighbors, csize_t nei_number, csize_t count)
{
	print_ne << <1, 1 >> > (neighbors, nei_number, count);
}


__global__ void simple_min(const float* clusters, csize_t dim, csize_t count, chunk_t* out)
{
	__shared__ chunk_t shared_mem[32];

	auto new_count = ((count + 1) * (count)) / 2;

	chunk_t tmp;
	tmp.min_dist = FLT_MAX;

	for (csize_t idx = threadIdx.x; idx < new_count; idx += blockDim.x)
	{
		auto coords = compute_coordinates(count, idx);

		if (coords.x == coords.y)
			continue;

		float dist = euclidean_norm(clusters + coords.x * dim, clusters + coords.y * dim, dim);

		if (dist < tmp.min_dist)
		{
			tmp.min_i = (clustering::asgn_t)coords.x;
			tmp.min_j = (clustering::asgn_t)coords.y;
			tmp.min_dist = dist;
		}
	}

	tmp = reduce_min_block(tmp, shared_mem);

	if (threadIdx.x == 0)
		*out = tmp;
}

chunk_t run_simple_min(const float* clusters, csize_t dim, csize_t count, chunk_t* out)
{
	simple_min << <1, 1024 >> > (clusters, dim, count, out);

	CUCH(cudaDeviceSynchronize());
	chunk_t res;
	CUCH(cudaMemcpy(&res, out, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (res.min_i > res.min_j)
		std::swap(res.min_i, res.min_j);

	return res;
}