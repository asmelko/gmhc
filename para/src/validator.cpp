#include <validator.hpp>
#include <algorithm>
#include "../tests/serial_impl.hpp"
#include <iostream>
#include <cfloat>

using namespace clustering;

void validator::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim, size_t maha_threshold)
{
	points_ = data_points;
	point_count_ = data_points_size;
	point_dim_ = data_point_dim;
	maha_threshold_ = maha_threshold;
	error_ = false;

	id_ = (asgn_t)point_count_;
	cluster_count_ = point_count_;

	for (size_t i = 0; i < point_count_; i++)
	{
		point_asgns_.push_back((asgn_t)i);

		cluster c;
		c.id = (asgn_t)i;

		for (size_t j = 0; j < point_dim_; j++)
			c.centroid.push_back(data_points[i * point_dim_ + j]);

		clusters_.emplace_back(std::move(c));
	}

	iteration_ = 0;
}

float eucl_dist(const float* lhs, const float* rhs, size_t size)
{
	float tmp_dist = 0;
	for (size_t k = 0; k < size; k++)
	{
		auto tmp = (lhs[k] - rhs[k]);
		tmp_dist += tmp * tmp;
	}
	return std::sqrt(tmp_dist);
}

std::vector<float> mat_vec(const float* mat, const float* vec, size_t size)
{
	std::vector<float> res;
	for (size_t i = 0; i < size; i++)
	{
		res.push_back(0);
		for (size_t j = 0; j < size; j++)
			res.back() += mat[i * size + j] * vec[j];
	}
	return res;
}

float dot(const float* lhs, const float* rhs, size_t size)
{
	float res = 0;
	for (size_t i = 0; i < size; i++)
		res += lhs[i] * rhs[i];
	return res;
}

std::vector<float> minus(const float* lhs, const float* rhs, size_t size)
{
	std::vector<float> res;
	for (size_t i = 0; i < size; i++)
		res.emplace_back(lhs[i] - rhs[i]);
	return res;
}

float compute_distance(const float* lhs_v, const float* lhs_m, const float* rhs_v, const float* rhs_m, size_t size)
{
	float dist = 0;
	if (lhs_m)
	{
		auto diff = minus(lhs_v, rhs_v, size);
		auto tmp = mat_vec(lhs_m, diff.data(), size);
		dist += std::sqrt(dot(tmp.data(), diff.data(), size));
	}
	else
		dist += eucl_dist(lhs_v, rhs_v, size);

	if (rhs_m)
	{
		auto diff = minus(lhs_v, rhs_v, size);
		auto tmp = mat_vec(rhs_m, diff.data(), size);
		dist += std::sqrt(dot(tmp.data(), diff.data(), size));
	}
	else
		dist += eucl_dist(lhs_v, rhs_v, size);

	return dist / 2;
}

size_t update_asgns(asgn_t* asgns, size_t count, pasgn_t old_pair, asgn_t id)
{
	size_t tmp_count = 0;
	for (size_t i = 0; i < count; i++)
	{
		if (asgns[i] == old_pair.first || asgns[i] == old_pair.second)
		{
			asgns[i] = id;
			tmp_count++;
		}
	}
	return tmp_count;
}

std::vector<float> compute_centroid(const float* points, size_t dim, size_t size, const asgn_t* assignments, asgn_t cid)
{
	size_t count = 0;
	std::vector<float> tmp_sum;
	tmp_sum.resize(dim);

	for (size_t i = 0; i < dim; i++)
		tmp_sum[i] = 0;

	for (size_t i = 0; i < size; i++)
	{
		if (assignments[i] == cid)
		{
			for (size_t j = 0; j < dim; j++)
				tmp_sum[j] += points[i * dim + j];
			count++;
		}
	}

	for (size_t i = 0; i < dim; i++)
		tmp_sum[i] /= count;

	return tmp_sum;
}

std::vector<float> compute_covariance(const float* points, size_t dim, size_t size, const asgn_t* assignments, const float* centroid, asgn_t cid)
{
	std::vector<float> cov;

	for (size_t i = 0; i < dim; ++i)
		for (size_t j = i; j < dim; ++j)
		{
			float res = 0;
			size_t count = 0;
			for (size_t k = 0; k < size; ++k)
			{
				if (assignments[k] != cid) continue;

				++count;

				res += (points[k * dim + i] - centroid[i]) *
					(points[k * dim + j] - centroid[j]);
			}
			cov.push_back(res / count);
		}
	return cov;
}

void print_pairs(size_t iteration, const pasgn_t& lhs, const pasgn_t& rhs)
{
	std::cerr << "Iteration " << iteration << ": pairs do not match: "
		<< lhs.first << ", " << lhs.second << " =/= "
		<< rhs.first << ", " << rhs.second << std::endl;
}

void print_arrays(size_t iteration, const std::string& msg, size_t size, const float* lhs, const float* rhs)
{
	std::cerr << "Iteration " << iteration << ": " << msg << std::endl;

	for (size_t j = 0; j < size; j++)
		std::cerr << lhs[j] << " ";

	std::cerr << std::endl;

	for (size_t j = 0; j < size; j++)
		std::cerr << rhs[j] << " ";

	std::cerr << std::endl;
}

bool validator::verify(pasgn_t pair_v, float dist_v, const float* centroid_v)
{
	auto [min_pair, new_clust, min_dist, cluster_count] = iterate(pair_v);

	if (min_pair != pair_v)
	{
		print_pairs(iteration_, min_pair, pair_v);

		error_ = true;
		return false;
	}

	if (float_diff(min_dist, dist_v))
	{
		std::cerr << "Iteration " << iteration_ << ": distances do not match: "
			<< min_dist << " =/= "
			<< dist_v << std::endl;

		error_ = true;
		return false;
	}

	if (float_diff(clusters_[new_clust].centroid.data(), centroid_v, point_dim_))
	{
		print_arrays(iteration_, "centroids do not match", point_dim_, clusters_[new_clust].centroid.data(), centroid_v);

		error_ = true;
		return false;
	}

	for (size_t i = 0; i < point_dim_; i++)
		clusters_[new_clust].centroid[i] = centroid_v[i];

	if (cluster_count >= maha_threshold_)
	{
		if (!cov_v.size() || !icov_v.size())
		{
			std::cerr << "Iteration " << iteration_ << ": expected matrices" << std::endl;
			error_ = true;
			return false;
		}

		size_t cov_size = (point_dim_ + 1) * point_dim_ / 2;
		for (size_t i = 0; i < cov_size; i++)
			cov_v[i] /= cluster_count;

		auto this_cov = compute_covariance(points_, point_dim_, point_count_, point_asgns_.data(), clusters_[new_clust].centroid.data(), clusters_[new_clust].id);

		if (float_diff(this_cov.data(), cov_v.data(), cov_size))
		{
			print_arrays(iteration_, "covariances do not match", cov_size, this_cov.data(), cov_v.data());

			error_ = true;
			return false;
		}
	}

	cov_v.resize(0);
	icov_v.resize(0);
	return true;
}

bool validator::has_error() const
{
	return error_;
}

std::tuple<pasgn_t, size_t, float, size_t> validator::iterate(const pasgn_t& expected)
{
	pasgn_t min_pair;
	std::pair<size_t, size_t> min_idx, expected_idx;
	float min_dist = FLT_MAX;
	float expected_dist = FLT_MAX;

	for (size_t i = 0; i < cluster_count_; i++)
	{
		for (size_t j = i + 1; j < cluster_count_; j++)
		{
			auto tmp_dist = compute_distance(clusters_[i].centroid.data(), clusters_[i].icov.data(), clusters_[j].centroid.data(), clusters_[j].icov.data(), point_dim_);

			pasgn_t curr_ids = clusters_[i].id > clusters_[j].id ?
				std::make_pair(clusters_[j].id, clusters_[i].id) : std::make_pair(clusters_[i].id, clusters_[j].id);

			if (curr_ids == expected)
			{
				expected_idx = std::make_pair(i, j);
				expected_dist = tmp_dist;
			}

			if (tmp_dist < min_dist)
			{
				min_pair = curr_ids;
				min_idx = std::make_pair(i, j);
				min_dist = tmp_dist;
			}
		}
	}

	if (expected != min_pair && !float_diff(expected_dist, min_dist, 0.001f))
	{
		std::cout << "cluster branching" << std::endl;
		min_pair = expected;
		min_idx = expected_idx;
	}

	size_t cluster_count = update_asgns(point_asgns_.data(), point_count_, min_pair, id_);

	cluster c;
	c.id = id_;
	c.centroid = compute_centroid(points_, point_dim_, point_count_, point_asgns_.data(), id_);

	if (icov_v.size())
		for (size_t i = 0; i < point_dim_ * point_dim_; i++)
			c.icov.push_back(icov_v[i]);

	if (min_idx.second != clusters_.size() - 1)
		clusters_[min_idx.second] = std::move(clusters_.back());

	clusters_.pop_back();
	clusters_[min_idx.first] = std::move(c);

	++id_;
	++iteration_;
	--cluster_count_;

	return std::tie(min_pair, min_idx.first, min_dist, cluster_count);
}


bool validator::float_diff(float a, float b, float d)
{
	return float_diff(&a, &b, 1, d);
}

bool validator::float_diff(const float* a, const float* b, size_t size, float d)
{
	float fr = 0;
	for (size_t i = 0; i < size; i++)
	{
		auto diff = std::abs(a[i] - b[i]);
		auto tmp = (diff / a[i] + diff / b[i]) / 2;

		if (tmp >= d * 4)
			return true;
	}
	return fr / size >= d;
}