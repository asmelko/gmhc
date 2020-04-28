#ifndef MHCA_H
#define MHCA_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>

namespace clustering {

using csize_t = uint32_t;
using asgn_t = csize_t;
using pasgn_t = std::pair<asgn_t, asgn_t>;

template <typename T>
class hierarchical_clustering
{
protected:
	const T* points;
	csize_t points_size;
	csize_t point_dim;

public:
	virtual void initialize(const T* data_points, csize_t data_points_size, csize_t data_point_dim)
	{
		this->points = data_points;
		this->points_size = data_points_size;
		this->point_dim = data_point_dim;
	}

	virtual std::vector<pasgn_t> run() = 0;

	virtual void free() = 0;
};

}

#endif