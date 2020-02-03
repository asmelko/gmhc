#include "kmeans.hpp"

#include <blas.hh>

using namespace clustering;

kmeans::kmeans(size_t clusters, size_t iterations)
	: clusters_(clusters), iters_(iterations) {}

float kmeans::distance(const float* x, const float* y) const
{
	float res = 0;
	for (size_t i = 0; i < point_dim; ++i)
	{
		auto tmp = x[i] - y[i];
		res += tmp * tmp;
	}
	return res;
}

asgn_t kmeans::nearest_cluster(const float* point, const float* centroids) const
{
	auto min_dist = distance(centroids, point);
	asgn_t nearest = 0;
	for (asgn_t i = 1; i < clusters_; ++i)
	{
		auto dist = distance(centroids + i * point_dim, point);
		if (dist < min_dist)
		{
			min_dist = dist;
			nearest = i;
		}
	}

	return nearest;
}

std::pair<std::vector<asgn_t>, std::vector<float>> kmeans::run()
{
	// Prepare for the first iteration
	std::vector<float> centroids;
	std::vector<asgn_t> assignments;

	std::vector<float> sums;
	std::vector<size_t> counts;

	centroids.resize(clusters_ * point_dim);
	assignments.resize(points_size);

	sums.resize(clusters_ * point_dim);
	counts.resize(clusters_);

	for (size_t i = 0; i < clusters_ * point_dim; i += point_dim)
		blas::copy((int)point_dim, points + i, 1, centroids.data() + i, 1);

	// Run the k-means refinements
	for (size_t k = 0; k < iters_; ++k)
	{
		// Prepare empty tmp fields.
		for (size_t i = 0; i < clusters_; ++i)
		{
			blas::scal((int)point_dim, 0, sums.data() + i * point_dim, 1);
			counts[i] = 0;
		}

		for (std::size_t i = 0; i < points_size; ++i)
		{
			auto nearest = nearest_cluster(points + i * point_dim, centroids.data());
			assignments[i] = nearest;
			blas::axpy((int)point_dim, 1, points + i * point_dim, 1, sums.data() + nearest * point_dim, 1);
			++counts[nearest];
		}

		for (std::size_t i = 0; i < clusters_; ++i)
		{
			if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
			blas::scal((int)point_dim, 1 / (float)counts[i], sums.data() + i * point_dim, 1);
			blas::copy((int)point_dim, sums.data() + i * point_dim, 1, centroids.data() + i * point_dim, 1);
		}
	}

	return std::make_pair(std::move(assignments), std::move(centroids));
}
