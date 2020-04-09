#include <smhc.hpp>

using namespace clustering;

smhc::smhc(size_t maha_threshold)
	: id_(0), maha_threshold_(maha_threshold) {}

void smhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	clusters_.resize(data_points_size);
	cluster_t::point_dim = data_point_dim;

	for (auto& cluster : clusters_)
	{
		cluster.add_points(data_points + id_ * data_point_dim, 1);
		cluster.compute_centroid();
		cluster.id = id_++;
	}
}

std::vector<pasgn_t> smhc::run()
{
	std::vector<pasgn_t> res;
	float min_dist;
	std::array<size_t, 2> min_pair;

	while (clusters_.size() > 1)
	{
		min_dist = -1.0f;
		for (asgn_t i = 0; i < clusters_.size(); ++i)
		{
			for (asgn_t j = i + 1; j < clusters_.size(); ++j)
			{
				float tmp = distance({ i, j });

				if (min_dist == -1 || tmp < min_dist)
				{
					min_dist = tmp;
					min_pair = { i, j };
				};
			}
		}

		res.push_back(std::make_pair(clusters_[min_pair[0]].id, clusters_[min_pair[1]].id));
		merge_clusters(min_pair);
	}

	return res;
}

void smhc::free() {}

float smhc::distance(const std::array<size_t, 2>& cluster_pair) const
{
	float distance = 0;
	for (size_t i = 0; i < 2; ++i)
		distance += clusters_[cluster_pair[i]].points.size() / point_dim >= maha_threshold_ ?
			clusters_[cluster_pair[i]].mahalanobis_distance(clusters_[cluster_pair[(i + 1) % 2]].centroid.data()) :
			clusters_[cluster_pair[i]].euclidean_distance(clusters_[cluster_pair[(i + 1) % 2]].centroid.data());

	return distance / 2;
}

void smhc::merge_clusters(const std::array<size_t, 2>& cluster_pair)
{
	clusters_[cluster_pair[0]].add_points(clusters_[cluster_pair[1]].points.data(), clusters_[cluster_pair[1]].points.size() / point_dim);

	clusters_[cluster_pair[0]].compute_centroid();

	clusters_[cluster_pair[0]].id = id_++;

	if (clusters_[cluster_pair[0]].points.size() / point_dim >= maha_threshold_)
		clusters_[cluster_pair[0]].compute_inverse_covariance_matrix();

	clusters_[cluster_pair[1]] = std::move(clusters_.back());
	clusters_.pop_back();
}
