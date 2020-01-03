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

std::vector<pasgn_t> smhc::run()
{
	auto [assignments, cetroids] = kmeans_.run();

	


	return {};
}

void smhc::free()
{

}

smhc::matrix_t smhc::create_covariance_matrix(const asgn_t* assignments, const asgn_t cluster)
{
	return matrix_t();
}
