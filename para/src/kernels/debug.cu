#include <cfloat>
#include <device_launch_parameters.h>
#include <iostream>

#include "common_kernels.cuh"
#include "kernels.cuh"

using namespace clustering;

void cuda_check(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << " at " << file << ":" << line;
        exit(code);
    }
}

void cuSOLVER_check(cusolverStatus_t code, const char* file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuSOLVER error"
                  << " at " << file << ":" << line;
        exit(1);
    }
}

__global__ void print_a(asgn_t* assignments, csize_t point_size)
{
    for (csize_t i = 0; i < point_size; i++)
        printf("%d: %d\n", (int)i, (int)assignments[i]);
}

void run_print_assg(asgn_t* assignments, csize_t point_size) { print_a<<<1, 1>>>(assignments, point_size); }

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
    print_centroid<<<1, 1>>>(centroid, dim, count);
    CUCH(cudaDeviceSynchronize());
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



__global__ void print_up(csize_t* updated, csize_t* count)
{
    printf("update:\n");
    for (csize_t i = 0; i < *count; i++)
        printf("upd %d\n", updated[i]);
}

void run_print_up(csize_t* updated, csize_t* count)
{
    print_up<<<1, 1>>>(updated, count);
    CUCH(cudaDeviceSynchronize());
}

__global__ void print_ne(neighbor_t* neighbors, csize_t nei_number, csize_t count)
{
    for (csize_t i = 0; i < count; i++)
    {
        for (csize_t j = 0; j < nei_number; j++)
        {
            printf(
                "%d. %f %d\n", (int)i, neighbors[i * nei_number + j].distance, (int)neighbors[i * nei_number + j].idx);
        }
    }
}

void run_print_nei(neighbor_t* neighbors, csize_t nei_number, csize_t count)
{
    print_ne<<<1, 1>>>(neighbors, nei_number, count);
    CUCH(cudaDeviceSynchronize());
}


__global__ void simple_min(const float* clusters, csize_t dim, csize_t count, chunk_t* out)
{
    __shared__ chunk_t shared_mem[32];

    auto new_count = ((count + 1) * (count)) / 2;

    chunk_t tmp;
    tmp.min_dist = FLT_INF;

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
    simple_min<<<1, 1024>>>(clusters, dim, count, out);

    CUCH(cudaDeviceSynchronize());
    chunk_t res;
    CUCH(cudaMemcpy(&res, out, sizeof(chunk_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (res.min_i > res.min_j)
        std::swap(res.min_i, res.min_j);

    return res;
}

__global__ void point_eucl(const float* lhs_centroid, const float* rhs_centroid, csize_t dim, float* res)
{
    *res = euclidean_norm(lhs_centroid, rhs_centroid, dim);
}

#include "neighbor/neighbor_maha.cuh"

__global__ void point_maha(const float* lhs_centroid,
    const float* rhs_centroid,
    csize_t dim,
    const float* lhs_icov,
    const float* rhs_icov,
    const float* lhs_mf,
    const float* rhs_mf,
    float* ret)
{
    extern __shared__ float shared_mem[];

    float dist = 0;

    auto lane_id = threadIdx.x % warpSize;

    if (!rhs_icov)
        dist += euclidean_norm(lhs_centroid, rhs_centroid, dim);

    for (csize_t i = lane_id; i < dim; i += warpSize)
        shared_mem[i] = lhs_centroid[i] - rhs_centroid[i];

    __syncwarp();

    if (rhs_icov)
        dist += maha_dist(shared_mem, rhs_icov, *rhs_mf, dim, lane_id);

    dist += maha_dist(shared_mem, lhs_icov, *lhs_mf, dim, lane_id);

    if (lane_id == 0)
    {
        *ret = dist / 2;
    }
}

float run_point_eucl(const float* lhs_centroid, const float* rhs_centroid, csize_t dim)
{
    float* cu_res;
    CUCH(cudaMalloc(&cu_res, sizeof(float)));

    point_eucl<<<1, 1>>>(lhs_centroid, rhs_centroid, dim, cu_res);

    CUCH(cudaDeviceSynchronize());
    float res;
    CUCH(cudaMemcpy(&res, cu_res, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUCH(cudaFree(cu_res));

    return res;
}

float run_point_maha(const float* lhs_centroid,
    const float* rhs_centroid,
    csize_t dim,
    const float* lhs_icov,
    const float* rhs_icov,
    const float* lhs_mf,
    const float* rhs_mf)
{
    float* cu_res;
    CUCH(cudaMalloc(&cu_res, sizeof(float)));
    auto icov_size = (dim + 1) * dim / 2;

    point_maha<<<1, 32, icov_size * sizeof(float)>>>(
        lhs_centroid, rhs_centroid, dim, lhs_icov, rhs_icov, lhs_mf, rhs_mf, cu_res);

    CUCH(cudaDeviceSynchronize());
    float res;
    CUCH(cudaMemcpy(&res, cu_res, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUCH(cudaFree(cu_res));

    return res;
}

__global__ void compare_nei_u(
    const neighbor_t* lhs, const neighbor_t* rhs, const csize_t* update, const csize_t* size, csize_t new_idx)
{
    printf("update size %d\n", *size);
    printf("to update:\n");
    for (size_t i = 0; i < *size; i++)
    {
        printf("%d ", update[i]);
    }
    printf("\n");
    for (size_t i = 0; i < *size; i++)
    {
        printf("e %f %f ", lhs[update[i]].distance, rhs[update[i]].distance);

        if (lhs[update[i]].distance != rhs[update[i]].distance)
            printf("d");

        if (update[i] == new_idx)
            printf(" new");
        printf("\n");
    }
}

__global__ void compare_nei(const neighbor_t* lhs,
    const neighbor_t* rhs,
    const csize_t small_size,
    csize_t big_begin,
    csize_t big_size,
    csize_t new_idx)
{
    printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    for (csize_t i = 0; i < small_size; i++)
    {
        printf("e %d: %f %d, %f %d ", i, lhs[i].distance, lhs[i].idx, rhs[i].distance, rhs[i].idx);

        if (lhs[i].distance != rhs[i].distance)
            printf("d");

        if (i == new_idx)
            printf(" new");
        printf("\n");
    }

    for (csize_t i = big_begin; i < big_begin + big_size; i++)
    {
        printf("e %d: %f %d, %f %d ", i, lhs[i].distance, lhs[i].idx, rhs[i].distance, rhs[i].idx);

        if (lhs[i].distance != rhs[i].distance)
            printf("d");

        if (lhs[i].distance > rhs[i].distance)
            printf("xxxx");

        if (i == new_idx)
            printf(" new");
        printf("\n");
    }
}

void run_compare_nei_u(const neighbor_t* lhs,
    const neighbor_t* rhs,
    const clustering::csize_t* update,
    const clustering::csize_t* size,
    clustering::csize_t new_idx)
{
    compare_nei_u<<<1, 1>>>(lhs, rhs, update, size, new_idx);
}

void run_compare_nei(const neighbor_t* lhs,
    const neighbor_t* rhs,
    const csize_t small_size,
    csize_t big_begin,
    csize_t big_size,
    csize_t new_idx)
{
    compare_nei<<<1, 1>>>(lhs, rhs, small_size, big_begin, big_size, new_idx);
}