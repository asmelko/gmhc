#include "smhc.hpp"

#include <cblas.h>
#include <lapacke.h>

using namespace clustering;

smhc::smhc()
	: kmeans_(1024, 1000) {}

void smhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);
	kmeans_.initialize(points, points_size, point_dim);
}

const asgn_t* smhc::iterate()
{
	auto [assignments, cetroids] = kmeans_.run();

	


	return nullptr;
}