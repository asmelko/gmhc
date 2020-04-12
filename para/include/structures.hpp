#ifndef COMMON_STRUCTURES_CUH
#define COMMON_STRUCTURES_CUH

#include <clustering.hpp>

struct chunk_t
{
	float min_dist;
	clustering::csize_t min_i, min_j;
};

struct neighbour_t
{
	float distance;
	clustering::csize_t idx;
};

struct input_t
{
	float* data;
	clustering::csize_t count;
	clustering::csize_t dim;
};

struct update_data_t
{
	clustering::csize_t new_idx;
	clustering::pasgn_t move_a;
	clustering::pasgn_t move_b;
	
	clustering::csize_t* to_update;
	clustering::csize_t* eucl_update_size;
	clustering::csize_t* maha_update_size;
};

struct cluster_bound_t
{
	clustering::csize_t eucl_size;
	clustering::csize_t maha_begin;
	clustering::csize_t maha_size;
};

struct centroid_data_t
{
	float* centroids;
	float* inverses;
	clustering::csize_t dim;
};

struct kernel_info
{
	unsigned int grid_dim;
	unsigned int block_dim;
	clustering::csize_t shared_size;

	kernel_info(unsigned int grid_dim, unsigned int block_dim, clustering::csize_t shared_size = 0)
		: grid_dim(grid_dim), block_dim(block_dim), shared_size(shared_size) {}

	kernel_info() = default;
};

#endif