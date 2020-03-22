#pragma once
__inline__ __device__ void reduce_sum_warp(float* point, size_t dim);
__inline__ __device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim);


template <size_t N>
__device__ void add_neighbour(neighbour_array_t<N>* neighbours, neighbour_t neighbour);

template <size_t N>
__device__ neighbour_array_t<N> merge_neighbours(const neighbour_array_t<N>* l_neighbours, const neighbour_array_t<N>* r_neighbours);


template <size_t N>
__inline__ __device__ void reduce_min_block(neighbour_array_t<N>* neighbours, neighbour_array_t<N>* shared_mem, bool reduce_warp);

