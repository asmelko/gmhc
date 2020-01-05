#include "smhc.hpp"

#include <cblas.h>
#include <lapacke.h>
#include <stdexcept>

#define LACH(x) if(x) throw std::runtime_error("LAPACK error")

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

float smhc::point_mean(const float* point)
{
	float res = 0;
	for (size_t i = 0; i < point_dim; res += point[i++]);
	return res / point_dim;
}

void smhc::point_subtract(const float* point, float scalar, float* dest)
{
	cblas_scopy(1, point, 1, dest, 1);
	for (size_t i = 0; i < point_dim; dest[i++] -= scalar);
}

smhc::matrix_t smhc::create_inverse_covariance_matrix(const asgn_t* assignments, const asgn_t cluster)
{
	int dim = (int)point_dim; //to get rid of warnings

	matrix_t icov;
	icov.resize(point_dim * point_dim);
	cblas_sscal(dim * dim, 0, icov.data(), 1); //zero out

	for (size_t i = 0; i < points_size; ++i)
	{
		if (assignments[i] != cluster) //ignore points in other clusters
			continue;

		std::vector<float> tmp;
		tmp.resize(point_dim);

		point_subtract(points + i, point_mean(points + i), tmp.data());

		cblas_ssyr(CblasRowMajor, CblasUpper, dim, 1, tmp.data(), 1, icov.data(), dim);
	}

	cblas_sscal(dim * dim, 1 / (float)dim, icov.data(), 1); //scale

	std::vector<int> piv;
	piv.resize(point_dim);

	LACH(LAPACKE_ssytrf(LAPACK_ROW_MAJOR, 'U', dim, icov.data(), dim, piv.data())); //factorization
	LACH(LAPACKE_ssytri(LAPACK_ROW_MAJOR, 'U', dim, icov.data(), dim, piv.data())); //inversion

	return icov;
}
