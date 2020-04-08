#ifndef COMMON_STRUCTURES_CUH
#define COMMON_STRUCTURES_CUH

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

enum class cluster_kind : uint8_t
{
	EMPTY = 0, EUCL = 1, MAHA = 2
};


struct input_t
{
	float* data;
	size_t count;
	size_t dim;
};

struct kernel_info
{
	unsigned int grid_dim;
	unsigned int block_dim;
	size_t shared_size;
};

#endif