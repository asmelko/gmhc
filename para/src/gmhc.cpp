#include "..\include\gmhc.hpp"
#include "..\include\gmhc.hpp"
#include <gmhc.hpp>
#include <kernels.cuh>
#include <cassert>
#include <iostream>
#include <map>

using namespace clustering;

void gmhc::initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	id_ = (asgn_t)data_points_size;
	cluster_count_ = data_points_size;
	bounds_.eucl_size = (asgn_t)data_points_size;
	bounds_.maha_size = 0;
	bounds_.maha_begin = (asgn_t)data_points_size;

	icov_size = (point_dim + 1) * point_dim / 2;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
	CUCH(cudaMalloc(&cu_min_, sizeof(chunk_t)));
	CUCH(cudaMalloc(&cu_neighs_, sizeof(neighbour_t) * neigh_number_ * data_points_size));
	CUCH(cudaMalloc(&cu_icov_, sizeof(float) * icov_size * data_points_size));

	CUCH(cudaMemcpy(cu_points_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_centroids_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

	run_set_default_asgn(cu_point_asgns_, data_points_size);

	starting_info_ = kernel_info(6, 512);
	CUCH(cudaMalloc(&tmp_neigh, sizeof(neighbour_t) * neigh_number_ * points_size * starting_info_.grid_dim));

	CUCH(cudaMalloc(&tmp_icov, 2 * point_dim * point_dim * sizeof(float)));

	CUCH(cudaMalloc(&cu_update_, cluster_count_ * sizeof(csize_t)));
	CUCH(cudaMalloc(&cu_eucl_upd_size_, sizeof(csize_t)));
	CUCH(cudaMalloc(&cu_maha_upd_size_, sizeof(csize_t)));

	CUCH(cudaMalloc(&cu_read_icov, sizeof(float*)));
	CUCH(cudaMalloc(&cu_write_icov, sizeof(float*)));
	CUCH(cudaMalloc(&cu_info, sizeof(int)));
	CUCH(cudaMalloc(&cu_pivot, sizeof(int) * data_point_dim));

	cluster_data_ = new cluster_data_t[cluster_count_];
	compute_data_ = centroid_data_t{ cu_centroids_, cu_icov_, (asgn_t)point_dim };
	upd_data_.to_update = cu_update_;
	upd_data_.eucl_update_size = cu_eucl_upd_size_;
	upd_data_.maha_update_size = cu_maha_upd_size_;

	default_apr_.cu_centroids = cu_centroids_;
	default_apr_.cu_inverses = cu_icov_;
	default_apr_.cu_neighbours = cu_neighs_;
	default_apr_.cu_tmp_neighbours = tmp_neigh;
	default_apr_.cu_updates = cu_update_;
	default_apr_.clusters = cluster_data_;
	default_apr_.bounds = bounds_;
	apriori_count_ = 0;

	for (asgn_t i = 0; i < cluster_count_; ++i)
	{
		cluster_data_[i].id = i;
		cluster_data_[i].size = 1;
	}

	BUCH(cublasCreate(&handle_));
	vld_ = nullptr;
}

void gmhc::initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t mahalanobis_threshold, const asgn_t* apriori_assignments, validator* vld)
{
	initialize(data_points, data_points_size, data_point_dim);
	maha_threshold_ = mahalanobis_threshold;
	vld_ = vld;

	if (apriori_assignments)
		initialize_apriori(apriori_assignments);
}

void gmhc::move_apriori()
{
	bounds_.maha_begin = bounds_.eucl_size;

	auto size = bounds_.maha_size + bounds_.eucl_size;

	auto eucl_idx = 0;
	auto maha_idx = bounds_.maha_begin;

	float* cu_centroids_tmp, * cu_icov_tmp;
	cluster_data_t* cluster_data_tmp = new cluster_data_t[size];
	CUCH(cudaMalloc(&cu_centroids_tmp, size * point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_icov_tmp, size * icov_size * sizeof(float)));

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

		CUCH(cudaMemcpy(cu_centroids_tmp + to * point_dim, ctx.cu_centroids + from * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
		memcpy(cluster_data_tmp + to, cluster_data_ + from, sizeof(cluster_data_t));
		if (ctx.bounds.maha_size)
			CUCH(cudaMemcpy(cu_icov_tmp + to * icov_size, ctx.cu_inverses + from * icov_size, icov_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
	}
	cu_centroids_ = cu_centroids_tmp;
	cluster_data_ = cluster_data_tmp;
	cu_icov_ = cu_icov_tmp;
}


std::vector<pasgn_t> gmhc::run()
{
	std::vector<pasgn_t> ret;
	apriori_context_t final_ctx;

	if (apriori_count_)
	{
		for (size_t i = 0; i < apriori_count_; ++i)
		{
			auto& ctx = apr_ctxs_[i];

			compute_data_.centroids = ctx.cu_centroids;
			compute_data_.inverses = ctx.cu_inverses;
			upd_data_.to_update = ctx.cu_updates;

			run_neighbours<neigh_number_>(ctx.cu_centroids, point_dim, ctx.bounds.eucl_size, ctx.cu_tmp_neighbours, ctx.cu_neighbours, starting_info_);

			while (ctx.bounds.eucl_size + ctx.bounds.maha_size > 1)
			{
				auto tmp = run(ctx);
				ret.push_back(tmp);
			}

			if (ctx.bounds.eucl_size)
				bounds_.eucl_size++;
			else
				bounds_.maha_size++;
		}

		move_apriori();

		final_ctx = apr_ctxs_[0];

		final_ctx.bounds = bounds_;

		compute_data_.centroids = final_ctx.cu_centroids;
		compute_data_.inverses = final_ctx.cu_inverses;
		upd_data_.to_update = final_ctx.cu_updates;
	}
	else
		final_ctx = default_apr_;

	run_neighbours<neigh_number_>(final_ctx.cu_centroids, point_dim, final_ctx.bounds.eucl_size, final_ctx.cu_tmp_neighbours, final_ctx.cu_neighbours, starting_info_);

	while (final_ctx.bounds.eucl_size + final_ctx.bounds.maha_size > 1)
	{
		auto tmp = run(final_ctx);
		ret.push_back(tmp);
	}

	return ret;
}

pasgn_t gmhc::run(apriori_context_t& ctx)
{
	cluster_data_t data[2];

	auto min = run_neighbours_min<neigh_number_>(ctx.cu_neighbours, ctx.bounds, cu_min_);

	data[0] = ctx.clusters[min.min_i];
	data[1] = ctx.clusters[min.min_j];

	move_clusters(min.min_i, min.min_j, data[0].size + data[1].size >= maha_threshold_, ctx);

	update_iteration(data, ctx);

	++id_;
	--cluster_count_;

	if (data[0].id > data[1].id)
		std::swap(data[0].id, data[1].id);

	pasgn_t ret(data[0].id, data[1].id);

	if (vld_)
		verify(ret, min.min_dist, ctx);

	run_update_neighbours<neigh_number_>(compute_data_, ctx.cu_tmp_neighbours, ctx.cu_neighbours, ctx.bounds, upd_data_, starting_info_);

	return ret;
}


void gmhc::update_iteration(const cluster_data_t* merged, apriori_context_t& ctx)
{
	auto new_idx = upd_data_.new_idx;

	//update cluster data
	ctx.clusters[new_idx].id = id_;
	ctx.clusters[new_idx].size = merged[0].size + merged[1].size;

	//updating point asgns
	run_merge_clusters(cu_point_asgns_, points_size, merged[0].id, merged[1].id, id_, kernel_info(6, 1024));

	//compute new centroid
	run_centroid(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, ctx.cu_centroids + new_idx * point_dim, id_, ctx.clusters[new_idx].size, starting_info_);

	if (ctx.clusters[new_idx].size >= maha_threshold_)
	{
		CUCH(cudaDeviceSynchronize());
		compute_icov(new_idx, ctx);
	}
}

bool gmhc::remove(csize_t idx, apriori_context_t& ctx)
{
	csize_t end_idx;
	if (idx < ctx.bounds.maha_begin)
		end_idx = --ctx.bounds.eucl_size;
	else
		end_idx = ctx.bounds.maha_begin + --ctx.bounds.maha_size;

	if (idx == end_idx)
		return false;

	CUCH(cudaMemcpy(ctx.cu_centroids + idx * point_dim, ctx.cu_centroids + end_idx * point_dim,
		sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	ctx.clusters[idx] = ctx.clusters[end_idx];

	CUCH(cudaMemcpy(ctx.cu_neighbours + idx * neigh_number_, ctx.cu_neighbours + end_idx * neigh_number_,
		sizeof(neighbour_t) * neigh_number_, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	if (ctx.clusters[idx].size >= maha_threshold_)
		CUCH(cudaMemcpy(ctx.cu_inverses + idx * icov_size, ctx.cu_inverses + end_idx * icov_size,
			sizeof(float) * icov_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	return true;
}

void gmhc::move_clusters(csize_t i, csize_t j, bool maha, apriori_context_t& ctx)
{
	std::vector<pasgn_t> ret;
	if (j < ctx.bounds.maha_begin && !maha) //c+c=c
	{
		bool move = remove(j, ctx);
		upd_data_.move_a = { j, move ? ctx.bounds.eucl_size : (asgn_t)-1 };

		upd_data_.move_b = { i, (asgn_t)-1 };

		upd_data_.new_idx = i;
	}
	else if (j < ctx.bounds.maha_begin && maha) //c+c=m
	{
		{
			bool move = remove(j, ctx);
			upd_data_.move_a = { j, move ? ctx.bounds.eucl_size : (asgn_t)-1 };
		}

		{
			bool move = remove(i, ctx);
			upd_data_.move_b = { i, move ? ctx.bounds.eucl_size : (asgn_t)-1 };
		}

		++ctx.bounds.maha_size;
		--ctx.bounds.maha_begin;
		upd_data_.new_idx = ctx.bounds.maha_begin;
	}
	else if (i < ctx.bounds.maha_begin) //c+m=m
	{
		bool move = remove(i, ctx);
		upd_data_.move_a = { i, move ? ctx.bounds.eucl_size : (asgn_t)-1 };

		upd_data_.move_b = { j, (asgn_t)-1 };

		upd_data_.new_idx = j;
	}
	else //m+m=m
	{
		bool move = remove(j, ctx);
		upd_data_.move_a = { j, move ? ctx.bounds.maha_size : (asgn_t)-1 };

		upd_data_.move_b = { i, (asgn_t)-1 };

		upd_data_.new_idx = i;
	}
}		

void gmhc::compute_icov(csize_t pos, apriori_context_t& ctx)
{
	float* icov = tmp_icov + point_dim * point_dim;

	assign_constant_storage(ctx.cu_centroids + pos * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	run_covariance(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, icov, id_, kernel_info(6, 1024, 100));

	run_finish_covariance(icov, ctx.clusters[pos].size, point_dim, tmp_icov);

	CUCH(cudaDeviceSynchronize());

	if (vld_)
	{
		vld_->cov_v.resize((point_dim + 1) * point_dim / 2);
		CUCH(cudaMemcpy(vld_->cov_v.data(), icov, ((point_dim + 1) * point_dim / 2) * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}


	CUCH(cudaMemcpy(cu_read_icov, &tmp_icov, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_write_icov, &icov, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice));

	if (point_dim <= 16)
		BUCH(cublasSmatinvBatched(handle_, (int)point_dim, cu_read_icov, (int)point_dim, cu_write_icov, (int)point_dim, cu_info, 1));
	else
	{
		BUCH(cublasSgetrfBatched(handle_, (int)point_dim, cu_read_icov, (int)point_dim, cu_pivot, cu_info, 1));
		BUCH(cublasSgetriBatched(handle_, (int)point_dim, cu_read_icov, (int)point_dim, cu_pivot, cu_write_icov, (int)point_dim, cu_info, 1));
	}

	CUCH(cudaDeviceSynchronize());

	int info;
	CUCH(cudaMemcpy(&info, cu_info, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	if (info != 0)
		run_set_default_inverse(icov, point_dim * point_dim);

	run_store_icovariance(ctx.cu_inverses + pos * icov_size, icov, point_dim);

	if (vld_)
	{
		vld_->icov_v.resize(point_dim * point_dim);
		CUCH(cudaMemcpy(vld_->icov_v.data(), icov, point_dim * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
}

void gmhc::initialize_apriori(const asgn_t* apriori_assignments)
{
	std::map<asgn_t, csize_t> counts;
	for (csize_t i = 0; i < points_size; ++i)
	{
		auto count = counts.find(apriori_assignments[i]);

		if (count != counts.end())
			count->second++;
		else
			counts.emplace(i, 1);
	}

	std::vector<csize_t> indices, sizes;
	std::map<asgn_t, csize_t> order;

	csize_t idx = 0;
	csize_t tmp_sum = 0;
	for (auto& count : counts)
	{
		order.emplace(count.first, idx++);
		indices.push_back(tmp_sum);
		tmp_sum += count.second;
		sizes.push_back(count.second);
	}

	for (csize_t i = 0; i < points_size; ++i)
	{
		asgn_t apriori_asgn = apriori_assignments[i];
		auto next = indices[order[apriori_asgn]]++;
		CUCH(cudaMemcpy(cu_centroids_ + next * point_dim, points + i * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
		cluster_data_[next] = cluster_data_t{ i, 1 };
	}


	for (size_t i = 0; i < sizes.size(); ++i)
	{
		apriori_context_t ctx;
		ctx.bounds.eucl_size = sizes[i];
		ctx.bounds.maha_begin = sizes[i];
		ctx.bounds.maha_size = 0;

		auto offset = i == 0 ? 0 : indices[i - 1];

		ctx.cu_centroids = cu_centroids_ + offset * point_dim;
		ctx.cu_inverses = cu_icov_ + offset * icov_size;
		ctx.cu_neighbours = cu_neighs_ + offset * neigh_number_;
		ctx.cu_tmp_neighbours = tmp_neigh + offset * neigh_number_ * starting_info_.grid_dim;
		ctx.cu_updates = cu_update_ + offset;
		ctx.clusters = cluster_data_ + offset;

		apr_ctxs_.emplace_back(ctx);
	}

	apriori_count_ = sizes.size();
}

void gmhc::free() {}

void gmhc::verify(pasgn_t id_pair, float dist, apriori_context_t& ctx)
{
	printf("\rIteration %d", (int)points_size - (int)cluster_count_);
	fflush(stderr);

	std::vector<float> tmp_centr;
	tmp_centr.resize(point_dim);
	CUCH(cudaMemcpy(tmp_centr.data(), ctx.cu_centroids + upd_data_.new_idx * point_dim, sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToHost));

	vld_->verify(id_pair, dist, tmp_centr.data());

	if (vld_->has_error())
	{
		CUCH(cudaDeviceSynchronize());
		cluster_count_ = 0;
	}
}