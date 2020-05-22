#ifndef COMMON_STRUCTURES_CUH
#define COMMON_STRUCTURES_CUH

#include "clustering.hpp"

//this file contains structures used in kernels or to pass parameters to kernels

//structure that holds indices of cluster pair and distance between them 
struct chunk_t
{
	float min_dist;
	clustering::csize_t min_i, min_j;
};

//represents 2D coordinates of csize_t type
struct csize2
{
	clustering::csize_t x, y;
};


//structure describing a clusters id and size
struct cluster_data_t
{
	clustering::asgn_t id;
	clustering::csize_t size;
};

//strucutre that represents a neighbor of a cluster
//the neighbor is represented by its index and distance to the cluster
struct neighbor_t
{
	float distance;
	clustering::csize_t idx;
};

//input structure for kernels
struct input_t
{
	float* data;
	clustering::csize_t count;
	clustering::csize_t dim;
};

//input structure for update kernel
struct update_data_t
{
	//new cluster
	clustering::csize_t new_idx;
	//first and second place for index that have to be changed due to a cluster move
	clustering::pasgn_t move_a;
	clustering::pasgn_t move_b;
	
	//update array
	clustering::csize_t* to_update;
	//variable describing top of euclidean part
	clustering::csize_t* eucl_update_size;
	//variable describing top of mahalanobis part
	clustering::csize_t* maha_update_size;
};

//indexing strucutre
struct cluster_bound_t
{
	//size of euclidean part (start is at 0)
	clustering::csize_t eucl_size;
	//start of mahalanobis part
	clustering::csize_t maha_begin;
	//size of mahalanobis part
	clustering::csize_t maha_size;
};

//input data for kernels
struct centroid_data_t
{
	float* centroids;
	float* inverses;
	clustering::csize_t dim;
};

//structure that holds parameters for kernel start
struct kernel_info
{
	//1D dimension of grid
	unsigned int grid_dim;
	//1D dimension of block
	unsigned int block_dim;
	//size of shared memory
	clustering::csize_t shared_size;

	kernel_info(unsigned int grid_dim, unsigned int block_dim, clustering::csize_t shared_size = 0)
		: grid_dim(grid_dim), block_dim(block_dim), shared_size(shared_size) {}

	kernel_info() = default;
};

#endif