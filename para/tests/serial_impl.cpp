#include "serial_impl.hpp"

#include <cmath>

using namespace clustering;

output_t serial_euclidean_min(const reader::data_t<float>& data)
{
	output_t res;
	res.distance = FLT_MAX;
	for (asgn_t i = 0; i < data.points; i++)
	{
		for (asgn_t j = i + 1; j < data.points; j++)
		{
			float tmp_dist = 0;
			for (size_t k = 0; k < data.dim; k++)
			{
				auto tmp = (data.data[i * data.dim + k] - data.data[j * data.dim + k]);
				tmp_dist += tmp * tmp;
			}
			tmp_dist = std::sqrt(tmp_dist);

			if (tmp_dist < res.distance)
			{
				res.distance = tmp_dist;
				res.i = i;
				res.j = j;
			}
		}
	}
	return res;
}

std::vector<clustering::asgn_t> create_assignments(size_t count, bool unique)
{
	clustering::asgn_t id = 0;
	std::vector<clustering::asgn_t> tmp;
	tmp.reserve(count);

	for (size_t i = 0; i < count; ++i)
	{
		tmp.push_back(id);
		if (unique) ++id;
	}
	return tmp;
}

std::vector<float> serial_centroid(const reader::data_t<float>& data, const asgn_t* assignments, asgn_t cid)
{
	size_t count = 0;
	std::vector<float> tmp_sum;
	tmp_sum.resize(data.dim);

	for (size_t i = 0; i < data.dim; i++)
		tmp_sum[i] = 0;

	for (size_t i = 0; i < data.points; i++)
	{
		if (assignments[i] == cid)
		{
			for (size_t j = 0; j < data.dim; j++)
				tmp_sum[j] += data.data[i * data.dim + j];
			count++;
		}
	}

	for (size_t i = 0; i < data.dim; i++)
		tmp_sum[i] /= count;

	return tmp_sum;
}

std::vector<float> serial_covariance(const reader::data_t<float>& data, const asgn_t* assignments, asgn_t cid)
{
	std::vector<float> cov;

	for (size_t i = 0; i < data.dim; ++i)
		for (size_t j = i; j < data.dim; ++j)
		{
			float res = 0;
			size_t count = 0;
			for (size_t k = 0; k < data.points; ++k)
			{
				if (assignments[k] != cid) continue;
				++count;

				for (size_t l = 0; l < data.points; ++l)
				{
					if (assignments[l] != cid) continue;

					res += ((data.data.data() + k * data.dim)[i] - (data.data.data() + l * data.dim)[i]) *
						((data.data.data() + k * data.dim)[j] - (data.data.data() + l * data.dim)[j]);
				}
			}
			cov.push_back(res / (count * count * 2));
		}
	return cov;
}

std::vector<float> serial_covariance_by_centroid(const reader::data_t<float>& data, const asgn_t* assignments, const float* centroid, asgn_t cid)
{
	std::vector<float> cov;

	for (size_t i = 0; i < data.dim; ++i)
		for (size_t j = i; j < data.dim; ++j)
		{
			float res = 0;
			size_t count = 0;
			for (size_t k = 0; k < data.points; ++k)
			{
				if (assignments[k] != cid) continue;

				++count;

				res += (data.data[k * data.dim + i] - centroid[i]) *
					(data.data[k * data.dim + j] - centroid[j]);
			}
			cov.push_back(res / count);
		}
	return cov;
}