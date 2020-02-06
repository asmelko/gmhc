#ifndef SMHC_H
#define SMHC_H

#include <cluster.hpp>
#include <array>

namespace clustering {

class smhc : public hierarchical_clustering<float>
{
	size_t id_;
	static constexpr size_t maha_threshold_ = 5;
	std::vector<cluster_t> clusters_;
public:
	smhc();
	
	virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

	virtual std::vector<pasgn_t> run() override;

	virtual void free() override;
private:
	float distance(const std::array<size_t, 2>& cluster_pair) const;

	void merge_clusters(const std::array<size_t, 2>& cluster_pair);
};

}

#endif