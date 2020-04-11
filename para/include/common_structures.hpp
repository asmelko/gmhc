#ifndef COMMON_STRUCTURES_CUH
#define COMMON_STRUCTURES_CUH

#include <clustering.hpp>

struct chunk_t
{
	float min_dist;
	clustering::asgn_t min_i, min_j;
};

struct neighbour_t
{
	float distance;
	clustering::asgn_t idx;
};

struct input_t
{
	float* data;
	size_t count;
	size_t dim;
};

using flag_t = std::uint8_t;

struct update_data_t
{
	clustering::asgn_t new_idx;
	clustering::pasgn_t move_a;
	clustering::pasgn_t move_b;
	
	flag_t* to_update;
};

struct cluster_bound_t
{
	clustering::asgn_t eucl_size;
	clustering::asgn_t maha_begin;
	clustering::asgn_t maha_size;
};

struct centroid_data_t
{
	float* centroids;
	float* inverses;
	clustering::asgn_t dim;
};

struct kernel_info
{
	unsigned int grid_dim;
	unsigned int block_dim;
	size_t shared_size;
};

#endif