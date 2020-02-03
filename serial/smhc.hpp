#ifndef SMHC_H
#define SMHC_H

#include "kmeans.hpp"

#include <array>

namespace clustering {

class smhc : public hierarchical_clustering<float>
{
	static constexpr size_t initial_cluster_count_ = 100;
	kmeans kmeans_;
	std::vector<float> inverses_;
	std::vector<float> centroids_;
	std::vector<asgn_t> assignments_;
	std::array<bool, initial_cluster_count_> merged_;
	size_t cluster_count_;
public:
	smhc();

	virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

	virtual std::vector<pasgn_t> run() override;

	virtual void free() override;
private:
	float point_mean(const float* point) const;
	void point_subtract(const float* point, float scalar, float* dest) const;
	float covariance(size_t i, size_t j, asgn_t cluster) const;

	void create_inverse_covariance_matrix(const asgn_t cluster, float* dest) const;
	void create_cluster_cetroid(const asgn_t cluster, float* dest) const;

	float mahalanobis_distance(const pasgn_t cluster_pair) const;
	void merge_clusters(const pasgn_t cluster_pair);
};

}

#endif