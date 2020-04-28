#include "..\include\gmhc.hpp"
#include "..\include\gmhc.hpp"
#include <gmhc.hpp>
#include <kernels.cuh>
#include <cassert>
#include <iostream>
#include <map>

using namespace clustering;

void gmhc::initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t mahalanobis_threshold, const asgn_t* apriori_assignments, validator* vld)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	common_.id = (asgn_t)data_points_size;
	common_.cluster_count = data_points_size;
	csize_t icov_size = (point_dim + 1) * point_dim / 2;

	maha_threshold_ = mahalanobis_threshold;
	starting_info_ = kernel_info(6, 512);

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
	CUCH(cudaMalloc(&cu_neighs_, sizeof(neighbor_t) * common_.neighbors_size * data_points_size));
	CUCH(cudaMalloc(&cu_tmp_neighs_, sizeof(neighbor_t) * common_.neighbors_size * data_points_size * starting_info_.grid_dim));
	CUCH(cudaMalloc(&cu_icov_, sizeof(float) * icov_size * data_points_size));
	CUCH(cudaMalloc(&cu_update_, common_.cluster_count * sizeof(csize_t)));
	cluster_data_ = new cluster_data_t[common_.cluster_count];

	CUCH(cudaMalloc(&common_.cu_min, sizeof(chunk_t)));
	CUCH(cudaMalloc(&common_.cu_tmp_icov, 2 * data_point_dim * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&common_.cu_eucl_upd_size, sizeof(csize_t)));
	CUCH(cudaMalloc(&common_.cu_maha_upd_size, sizeof(csize_t)));
	CUCH(cudaMalloc(&common_.cu_read_icov, sizeof(float*)));
	CUCH(cudaMalloc(&common_.cu_write_icov, sizeof(float*)));
	CUCH(cudaMalloc(&common_.cu_info, sizeof(int)));
	CUCH(cudaMalloc(&common_.cu_pivot, sizeof(int) * data_point_dim));
	BUCH(cublasCreate(&common_.cublas_handle));

	if (apriori_assignments)
		initialize_apriori(apriori_assignments, vld);
	else
	{
		CUCH(cudaMemcpy(cu_points_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_centroids_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
		run_set_default_asgn(cu_point_asgns_, data_points_size);

		for (asgn_t i = 0; i < common_.cluster_count; ++i)
			cluster_data_[i] = cluster_data_t{ i, 1 };

		apr_ctxs_.emplace_back(common_);

		set_apriori(apr_ctxs_.front(), 0, points_size, vld);
		apriori_count_ = 0;
		return;
	}
}

void gmhc::set_apriori(clustering_context_t& cluster, csize_t offset, csize_t size, validator* vld)
{
	csize_t icov_size = (point_dim + 1) * point_dim / 2;

	cluster.point_size = size;
	cluster.point_dim = point_dim;
	cluster.icov_size = icov_size;

	cluster.cluster_count = size;
	cluster.maha_threshold = maha_threshold_;

	cluster.starting_info = starting_info_;

	cluster.cu_neighbors = cu_neighs_ + offset * common_.neighbors_size;
	cluster.cu_tmp_neighbors = cu_tmp_neighs_ + offset * common_.neighbors_size * starting_info_.grid_dim;

	cluster.cu_points = cu_points_ + offset * point_dim;
	cluster.cu_centroids = cu_centroids_ + offset * point_dim;
	cluster.cu_inverses = cu_icov_ + offset * icov_size;

	cluster.cu_point_asgns = cu_point_asgns_ + offset;

	cluster.cu_updates = cu_update_ + offset;
	cluster.cluster_data = cluster_data_ + offset;

	cluster.vld = vld;

	cluster.initialize();
}

void gmhc::initialize_apriori(const asgn_t* apriori_assignments, validator* vld)
{
	//get apriori sizes
	std::map<asgn_t, csize_t> counts;
	for (csize_t i = 0; i < points_size; ++i)
	{
		auto count = counts.find(apriori_assignments[i]);

		if (count != counts.end())
			count->second++;
		else
			counts.emplace(apriori_assignments[i], 1);
	}

	//create indexing structures
	std::vector<csize_t> indices, sizes;
	std::map<asgn_t, csize_t> order;
	csize_t idx = 0;
	csize_t tmp_sum = 0;
	for (auto& count : counts)
	{
		order.emplace(count.first, idx++);
		indices.push_back(tmp_sum);
		sizes.push_back(count.second);
		tmp_sum += count.second;
	}

	//sorting data for apriori clusters
	for (csize_t i = 0; i < points_size; ++i)
	{
		asgn_t apriori_asgn = apriori_assignments[i];
		auto next = indices[order[apriori_asgn]]++;
		CUCH(cudaMemcpy(cu_points_ + next * point_dim, points + i * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_point_asgns_ + next, &i, sizeof(asgn_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		cluster_data_[next] = cluster_data_t{ i, 1 };
	}
	CUCH(cudaMemcpy(cu_centroids_, cu_points_, points_size * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	//initialize apriori
	for (size_t i = 0; i < sizes.size(); ++i)
	{
		clustering_context_t cluster(common_);
		auto offset = i == 0 ? 0 : indices[i - 1];

		set_apriori(cluster, offset, sizes[i], vld);

		apr_ctxs_.emplace_back(std::move(cluster));
	}
	apriori_count_ = (csize_t)sizes.size();

	//initialize apriori merging structures
	csize_t icov_size = (point_dim + 1) * point_dim / 2;
	apriori_cluster_data_ = new cluster_data_t[sizes.size()];
	CUCH(cudaMalloc(&cu_apriori_centroids_, sizes.size() * point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_apriori_icov_, sizes.size() * icov_size * sizeof(float)));
}

void gmhc::move_apriori(csize_t eucl_size, csize_t maha_size)
{
	assert(common_.cluster_count == eucl_size + maha_size);
	csize_t icov_size = (point_dim + 1) * point_dim / 2;

	auto eucl_idx = 0;
	auto maha_idx = eucl_size;

	for (size_t i = 0; i < apriori_count_; ++i)
	{
		auto ctx = apr_ctxs_[i];
		csize_t to, from;

		if (ctx.bounds.eucl_size)
		{
			to = eucl_idx++;
			from = 0;
		}
		else
		{
			to = maha_idx++;
			from = ctx.bounds.maha_begin;
		}

		apriori_cluster_data_[to] = ctx.cluster_data[from];
		CUCH(cudaMemcpy(cu_apriori_centroids_ + to * point_dim, ctx.cu_centroids + from * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
		if (ctx.bounds.maha_size)
			CUCH(cudaMemcpy(cu_apriori_icov_ + to * icov_size, ctx.cu_inverses + from * icov_size, icov_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
	}

	apr_ctxs_.front().cluster_data = apriori_cluster_data_;
	apr_ctxs_.front().cu_centroids = cu_apriori_centroids_;
	apr_ctxs_.front().cu_inverses = cu_apriori_icov_;
	apr_ctxs_.front().point_size = points_size;
	apr_ctxs_.front().cluster_count = common_.cluster_count;
	apr_ctxs_.front().initialize();

	apr_ctxs_.front().bounds.eucl_size = eucl_size;
	apr_ctxs_.front().bounds.maha_begin = eucl_size;
	apr_ctxs_.front().bounds.maha_size = maha_size;
}

std::vector<pasgn_t> gmhc::run()
{
	std::vector<pasgn_t> ret;
	clustering_context_t& last = apr_ctxs_.front();

	//compute apriori
	if (apriori_count_)
	{
		csize_t eucl_size = 0;
		csize_t maha_size = 0;

		for (size_t i = 0; i < apriori_count_; ++i)
		{
			auto& ctx = apr_ctxs_[i];

			run_neighbors<shared_apriori_data_t::neighbors_size>(ctx.cu_centroids, point_dim, ctx.bounds.eucl_size, ctx.cu_tmp_neighbors, ctx.cu_neighbors, ctx.starting_info);

			while (ctx.cluster_count > 1)
			{
				auto tmp = ctx.iterate();
				ret.push_back(tmp);
			}

			if (ctx.bounds.eucl_size)
				++eucl_size;
			else
				++maha_size;
		}

		move_apriori(eucl_size, maha_size);

		run_set_default_neigh(last.cu_neighbors, common_.cluster_count * common_.neighbors_size, starting_info_);
		run_update_neighbors<shared_apriori_data_t::neighbors_size>(last.compute_data, last.cu_tmp_neighbors, last.cu_neighbors, last.bounds, last.update_data, last.starting_info);
	}
	else
		run_neighbors<shared_apriori_data_t::neighbors_size>(last.cu_centroids, point_dim, last.bounds.eucl_size, last.cu_tmp_neighbors, last.cu_neighbors, last.starting_info);

	//compute rest
	while (last.cluster_count > 1)
	{
		auto tmp = last.iterate();
		ret.push_back(tmp);
	}

	return ret;
}

void gmhc::free() {}