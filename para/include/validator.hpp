#ifndef VALIDATOR_HPP
#define VALIDATOR_HPP

#include <clustering.hpp>
#include <tuple>

namespace clustering
{

class validator
{
	struct cluster
	{
		std::vector<float> centroid;
		std::vector<float> icov;
		asgn_t id;
	};

	const float* points_;
	csize_t point_count_;
	csize_t point_dim_;
	csize_t maha_threshold_;
	bool error_;

	asgn_t id_;
	csize_t cluster_count_;
	std::vector<asgn_t> point_asgns_;
	std::vector<cluster> clusters_;

	csize_t iteration_;
public:
	void initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t maha_threshold);

	std::vector<float> cov_v, icov_v;
	bool verify(pasgn_t pair_v, float dist_v, const float* centroid_v);

	bool has_error() const;

	static bool float_diff(float a, float b, float d = 0.05f);
	static bool float_diff(const float* a, const float* b, csize_t size, float d = 0.05f);
private:
	std::tuple<pasgn_t, csize_t, float, csize_t> iterate(const pasgn_t& expected);
};

}

#endif