#include "smhc.hpp"

#include <blas.hh>
#include <lapack.hh>
#include <stdexcept>

#define LACH(x) if (x) throw std::runtime_error("LAPACK error")

using namespace clustering;

smhc::smhc()
	: kmeans_(initial_cluster_count_, 1000), cluster_count_(initial_cluster_count_) {}

void smhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);
	kmeans_.initialize(points, points_size, point_dim);
	inverses_.resize(initial_cluster_count_ * point_dim * point_dim);
}

std::vector<pasgn_t> smhc::run()
{
	std::vector<pasgn_t> res;
	auto [assignments, cetroids] = kmeans_.run();
	centroids_ = std::move(cetroids);
	assignments_ = std::move(assignments);

	for (asgn_t i = 0; i < cluster_count_; ++i)
		create_inverse_covariance_matrix(i, inverses_.data() + point_dim * point_dim * i);

	float min_dist = -1.0f;
	pasgn_t min_pair;

	while (cluster_count_ > 2)
	{
		for (asgn_t i = 0; i < cluster_count_; ++i)
		{
			for (asgn_t j = i; j < cluster_count_; ++j)
			{
				float tmp = mahalanobis_distance(std::make_pair(i, j));

				if (min_dist == -1 || tmp < min_dist)
				{
					min_dist = tmp;
					min_pair = std::make_pair(i, j);
				};
			}
		}

		merge_clusters(min_pair);
		res.push_back(min_pair);
	}

	return res;
}

void smhc::free() {}

float smhc::point_mean(const float* point) const
{
	float res = 0;
	for (size_t i = 0; i < point_dim; res += point[i++]);
	return res / point_dim;
}

void smhc::point_subtract(const float* point, float scalar, float* dest) const
{
	blas::copy(1, point, 1, dest, 1);
	for (size_t i = 0; i < point_dim; dest[i++] -= scalar);
}

void smhc::create_inverse_covariance_matrix(const asgn_t cluster, float* dest) const
{
	int dim = (int)point_dim; //to get rid of warnings

	float* icov = dest;
	blas::scal(dim * dim, 0, icov, 1); //zero out

	for (size_t i = 0; i < points_size; ++i)
	{
		if (assignments_[i] != cluster) //ignore points in other clusters
			continue;

		std::vector<float> tmp;
		tmp.resize(point_dim);

		point_subtract(points + i, point_mean(points + i), tmp.data());

		blas::syr(blas::Layout::RowMajor, blas::Uplo::Upper, dim, 1, tmp.data(), 1, icov, dim);
	}

	blas::scal(dim * dim, 1 / (float)dim, icov, 1); //scale

	std::vector<int64_t> piv;
	piv.resize(point_dim);

	LACH(lapack::sytrf(lapack::Uplo::Upper, dim, icov, dim, piv.data())); //factorization
	LACH(lapack::sytri(lapack::Uplo::Upper, dim, icov, dim, piv.data())); //inversion
}

void smhc::create_cluster_cetroid(const asgn_t cluster, float* dest) const
{
	int dim = (int)point_dim;

	float* cetroid = dest;
	blas::scal(dim, 0, cetroid, 1); //zero out
	size_t count = 0;

	for (size_t i = 0; i < points_size; ++i)
	{
		if (assignments_[i] == cluster)
		{
			blas::axpy(dim, 1, points + point_dim * i, 1, cetroid, 1);
			count++;
		}
	}
	blas::scal(dim, 1 / (float)count, cetroid, 1);
}

float smhc::mahalanobis_distance(const pasgn_t cluster_pair) const
{
	int dim = (int)point_dim;
	std::vector<float> diff;
	diff.resize(point_dim);
	const float* l_data = centroids_.data() + cluster_pair.first * point_dim;
	const float* r_data = centroids_.data() + cluster_pair.second * point_dim;

	blas::copy(dim, r_data, 1, diff.data(), 1);
	blas::axpy(dim, -1, l_data, 1, diff.data(), 1);

	std::vector<float> tmp_res;
	tmp_res.resize(point_dim);

	blas::symv(blas::Layout::RowMajor, blas::Uplo::Upper, dim, 1, inverses_.data() + point_dim * point_dim * cluster_pair.first, dim, diff.data(), 1, 0, tmp_res.data(), 1);
	float distance = blas::dot(dim, diff.data(), 1, tmp_res.data(), 1);

	blas::symv(blas::Layout::RowMajor, blas::Uplo::Upper, dim, 1, inverses_.data() + point_dim * point_dim * cluster_pair.second, dim, diff.data(), 1, 0, tmp_res.data(), 1);
	distance += blas::dot(dim, diff.data(), 1, tmp_res.data(), 1);

	return distance / 2;
}

void smhc::merge_clusters(const pasgn_t cluster_pair)
{
	for (size_t i = 0; i < points_size; ++i)
	{
		if (assignments_[i] == cluster_pair.second)
			assignments_[i] = cluster_pair.first;
	}

	create_inverse_covariance_matrix(cluster_pair.first, inverses_.data() + point_dim * point_dim * cluster_pair.first);
	create_cluster_cetroid(cluster_pair.first, centroids_.data() + point_dim * cluster_pair.first);
}
