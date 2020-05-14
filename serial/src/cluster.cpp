#include "cluster.hpp"
#include "blas.hh"
#include "lapack.hh"
#include <stdexcept>

#define LACH(x) if (x) throw std::runtime_error("LAPACK error")

using namespace clustering;

size_t cluster_t::point_dim;

void cluster_t::add_points(const float* data, size_t count)
{
	auto tmp_size = points.size();
	points.resize(points.size() + count * point_dim);
	float* start = points.data() + tmp_size;
	blas::copy(count * point_dim, data, 1, start, 1);
}

void cluster_t::compute_centroid()
{
	centroid.resize(point_dim);
	blas::copy(point_dim, points.data(), 1, centroid.data(), 1);

	for (size_t i = point_dim; i < points.size(); i += point_dim)
		blas::axpy(point_dim, 1, points.data() + i, 1, centroid.data(), 1);

	blas::scal(point_dim, 1 / (float)(points.size() / point_dim), centroid.data(), 1);
}

void cluster_t::compute_inverse_covariance_matrix()
{
	icov.resize(point_dim * point_dim);
	blas::scal(point_dim * point_dim, 0, icov.data(), 1); //zero out

	for (size_t i = 0; i < point_dim; ++i)
		for (size_t j = i; j < point_dim; ++j)
			icov[i + j * point_dim] = covariance(i, j);

	std::vector<int64_t> piv;
	piv.resize(point_dim);

	LACH(lapack::sytrf(lapack::Uplo::Upper, point_dim, icov.data(), point_dim, piv.data())); //factorization
	LACH(lapack::sytri(lapack::Uplo::Upper, point_dim, icov.data(), point_dim, piv.data())); //inversion
}

float cluster_t::covariance(size_t X, size_t Y) const
{
	float res = 0;
	for (size_t i = 0; i < points.size(); i += point_dim)
	{
		for (size_t j = i + point_dim; j < points.size(); j += point_dim)
		{
			res += ((points.data() + i)[X] - (points.data() + j)[X]) *
				((points.data() + i)[Y] - (points.data() + j)[Y]);
		}
	}
	return res / ((points.size() / point_dim) * (points.size() / point_dim));
}

float cluster_t::euclidean_distance(const float* point) const
{
	std::vector<float> diff;
	diff.resize(point_dim);

	blas::copy(point_dim, point, 1, diff.data(), 1);
	blas::axpy(point_dim, -1, centroid.data(), 1, diff.data(), 1);
	return blas::nrm2(point_dim, diff.data(), 1);
}

float cluster_t::mahalanobis_distance(const float* point) const
{
	std::vector<float> diff;
	diff.resize(point_dim);

	blas::copy(point_dim, point, 1, diff.data(), 1);
	blas::axpy(point_dim, -1, centroid.data(), 1, diff.data(), 1);

	std::vector<float> tmp_res;
	tmp_res.resize(point_dim);

	blas::symv(blas::Layout::ColMajor, blas::Uplo::Upper, point_dim, 1, icov.data(), point_dim, diff.data(), 1, 0, tmp_res.data(), 1);
	float distance = blas::dot(point_dim, diff.data(), 1, tmp_res.data(), 1);
	return std::sqrt(distance);
}