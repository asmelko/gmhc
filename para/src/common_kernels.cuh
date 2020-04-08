#pragma once
__inline__ __device__ void reduce_sum_warp(float* point, size_t dim);
__inline__ __device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim);


template <size_t N>
__device__ void add_neighbour(neighbour_t* neighbours, neighbour_t neighbour);

template <size_t N>
__device__ void merge_neighbours(const neighbour_t* l_neighbours, const neighbour_t* r_neighbours, neighbour_t* res);

template <size_t N>
__inline__ __device__ void reduce_min_block(neighbour_t* neighbours, neighbour_t* shared_mem, bool reduce_warp);