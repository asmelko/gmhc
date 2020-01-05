#ifndef SMHC_H
#define SMHC_H

#include "kmeans.hpp"

namespace clustering {

class smhc : public hierarchical_clustering<float>
{
	using matrix_t = std::vector<float>;
	kmeans kmeans_;
public:
	smhc();

	virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

	virtual std::vector<pasgn_t> run() override;

	virtual void free() override;

private:
	float point_mean(const float* point);
	void point_subtract(const float* point, float scalar, float* dest);
	matrix_t create_inverse_covariance_matrix(const asgn_t* assignments, const asgn_t cluster);
};

}

#endif