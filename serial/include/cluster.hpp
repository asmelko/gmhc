#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "clustering.hpp"

namespace clustering {

struct cluster_t
{
	static size_t point_dim;

	asgn_t id;
	std::vector<float> points;
	std::vector<float> centroid;
	std::vector<float> icov;

	void add_points(const float* points, size_t count);
	void compute_centroid();
	void compute_inverse_covariance_matrix();
	float covariance(size_t X, size_t Y) const;

	float euclidean_distance(const float* point) const;
	float mahalanobis_distance(const float* point) const;
};

}

#endif