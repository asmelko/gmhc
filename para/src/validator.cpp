#include <validator.hpp>
#include <algorithm>
#include "../tests/serial_impl.hpp"
#include <iostream>

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

bool float_diff(float a, float b)
{
	auto avg = (a + b) / 2;
	auto sdev = (a - avg) * (a - avg) * 2;
	if (sdev >= 0.0009f)
		return true;
	else

	return false;
}

bool validator::verify(pasgn_t pair_v, float dist_v, const float* centroid_v)
{
	pasgn_t min_pair;
	std::pair<size_t, size_t> min_idx;
	float min_dist = FLT_MAX;

	if (pair_v == std::make_pair<asgn_t, asgn_t>(47, 49))
	{
		int i;
		i = 5;
	}

	for (size_t i = 0; i < cluster_count_; i++)
	{
		for (size_t j = i+1; j < cluster_count_; j++)
		{
			auto tmp_dist = compute_distance(clusters_[i].centroid.data(), clusters_[i].icov.data(), clusters_[j].centroid.data(), clusters_[j].icov.data(), point_dim_);

			if (tmp_dist < min_dist)
			{
				min_pair = std::make_pair(clusters_[i].id, clusters_[j].id);
				min_idx = std::make_pair(i,j);
				min_dist = tmp_dist;
			}
		}
	}

	size_t cluster_count = update_asgns(point_asgns_.data(), point_count_, min_pair, id_);

	cluster c;
	c.id = id_;
	c.centroid = compute_centroid(points_, point_dim_, point_count_, point_asgns_.data(), id_);

	if (icov_v.size())
		for (size_t i = 0; i < point_dim_ * point_dim_; i++)
			c.icov.push_back(icov_v[i]);

	clusters_.erase(clusters_.begin() + min_idx.second);
	clusters_.erase(clusters_.begin() + min_idx.first);
	clusters_.emplace_back(std::move(c));

	++id_;
	++iteration_;
	--cluster_count_;

	//check

	if (min_pair != pair_v)
	{
		std::cerr << "Iteration " << iteration_ << ": pairs does not match: "
			<< min_pair.first << ", " << min_pair.second << " =/= "
			<< pair_v.first << ", " << pair_v.second << std::endl;

		error_ = true;
		return false;
	}

	if (float_diff(min_dist, dist_v))
	{
		std::cerr << "Iteration " << iteration_ << ": distances does not match: "
			<< min_dist << " =/= "
			<< dist_v << std::endl;

		error_ = true;
		return false;
	}

	for (size_t i = 0; i < point_dim_; i++)
	{
		if (float_diff(clusters_.back().centroid[i], centroid_v[i]))
		{
			std::cerr << "Iteration " << iteration_ << ": centroids does not match: ";

			for (size_t j = 0; j < point_dim_; j++)
				std::cerr << clusters_.back().centroid[j] << " ";

			std::cerr << std::endl;

			for (size_t j = 0; j < point_dim_; j++)
				std::cerr << centroid_v[j] << " ";

			std::cerr << std::endl;

			error_ = true;
			return false;
		}
	}

	for (size_t i = 0; i < point_dim_; i++)
		clusters_.back().centroid[i] =centroid_v[i];

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

		auto this_cov = compute_covariance(points_, point_dim_, point_count_, point_asgns_.data(), clusters_.back().centroid.data(), clusters_.back().id);
		for (size_t i = 0; i < cov_size; i++)
		{
			if (float_diff(this_cov[i],cov_v[i]))
			{
				std::cerr << "Iteration " << iteration_ << ": covariance matrices does not match: " << std::endl;
				std::cerr << this_cov[i] << " =/= " << cov_v[i] << std::endl;

				const float* to_print = this_cov.data();
				while (true)
				{
					for (size_t j = 0; j < cov_size; j++)
						std::cerr << to_print[j] << " ";

					if (to_print == cov_v.data())
						break;

					std::cerr << std::endl << "=/=" << std::endl;
					to_print = cov_v.data();
				}
				error_ = true;
				return false;
			}
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