#include <gmhc.hpp>
#include <kernels.cuh>
#include <cassert>

using namespace clustering;

void gmhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	id_ = (asgn_t)data_points_size;
	cluster_count_ = data_points_size;
	bounds_.eucl_size = data_points_size;
	bounds_.maha_size = 0;
	bounds_.maha_begin = data_points_size;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
	CUCH(cudaMalloc(&cu_min_, sizeof(chunk_t)));
	CUCH(cudaMalloc(&cu_neighs_, sizeof(neighbour_t) * neigh_number_ * data_points_size));
	CUCH(cudaMalloc(&cu_icov_, sizeof(float) * point_dim * point_dim * data_points_size));

	CUCH(cudaMemcpy(cu_points_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_centroids_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

	run_set_default_asgn(cu_point_asgns_, data_points_size);

	starting_info_ = kernel_info{ 5,  512 };
	CUCH(cudaMalloc(&tmp_neigh, sizeof(neighbour_t) * neigh_number_ * points_size * starting_info_.grid_dim));

	CUCH(cudaMalloc(&tmp_icov, point_dim * point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_update_, cluster_count_ * sizeof(uint8_t)));

	CUCH(cudaMalloc(&cu_read_icov, sizeof(float*)));
	CUCH(cudaMalloc(&cu_write_icov, sizeof(float*)));
	CUCH(cudaMalloc(&cu_info, sizeof(int)));
	CUCH(cudaMalloc(&cu_pivot, sizeof(int) * data_point_dim));

	cluster_data_ = new cluster_data_t[cluster_count_];
	compute_data_ = centroid_data_t{ cu_centroids_, cu_icov_, (asgn_t)point_dim };
	upd_data_.to_update = cu_update_;

	for (asgn_t i = 0; i < cluster_count_; ++i)
	{
		cluster_data_[i].id = i;
		cluster_data_[i].size = 1;
	}

	BUCH(cublasCreate(&handle_));
	vld_ = nullptr;
}

void gmhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim, size_t mahalanobis_threshold, validator* vld)
{
	initialize(data_points, data_points_size, data_point_dim);
	maha_threshold_ = mahalanobis_threshold;
	vld_ = vld;
}


std::vector<pasgn_t> gmhc::run()
{
	std::vector<pasgn_t> ret;

	run_neighbours<neigh_number_>(cu_centroids_, point_dim, cluster_count_, tmp_neigh, cu_neighs_, starting_info_);

	while (cluster_count_ > 1)
	{
		chunk_t min = run_neighbours_min<neigh_number_>(cu_neighs_, bounds_, cu_min_);

		cluster_data_t data[2];

		data[0] = cluster_data_[min.min_i];
		data[1] = cluster_data_[min.min_j];

		move_clusters(min.min_i, min.min_j, data[0].size + data[1].size >= maha_threshold_);

		update_iteration(data);

		++id_;
		--cluster_count_;

		if (data[0].id > data[1].id)
			std::swap(data[0].id, data[1].id);

		ret.emplace_back(data[0].id, data[1].id);

		if (vld_)
			verify(ret.back(), min.min_dist);

		run_update_neighbours<neigh_number_>(compute_data_, tmp_neigh, cu_neighs_, bounds_, upd_data_, starting_info_);
	}

	return ret;
}

void gmhc::update_iteration(const cluster_data_t* merged)
{
	auto new_idx = upd_data_.new_idx;

	//update cluster data
	cluster_data_[new_idx].id = id_;
	cluster_data_[new_idx].size = merged[0].size + merged[1].size;

	//updating point asgns
	run_merge_clusters(cu_point_asgns_, points_size, merged[0].id, merged[1].id, id_, kernel_info{ 6, 1024 });

	//compute new centroid
	run_centroid(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, cu_centroids_ + new_idx * point_dim, id_, cluster_data_[new_idx].size, starting_info_);

	if (cluster_data_[new_idx].size >= maha_threshold_)
	{
		CUCH(cudaDeviceSynchronize());
		compute_icov(new_idx);
	}
}

bool gmhc::hole(size_t idx)
{
	size_t end_idx;
	if (idx < bounds_.maha_begin)
		end_idx = --bounds_.eucl_size;
	else
		end_idx = bounds_.maha_begin + --bounds_.maha_size;

	if (idx == end_idx)
		return false;

	CUCH(cudaMemcpy(cu_centroids_ + idx * point_dim, cu_centroids_ + end_idx * point_dim,
		sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	cluster_data_[idx] = cluster_data_[end_idx];

	CUCH(cudaMemcpy(cu_neighs_ + idx * neigh_number_, cu_neighs_ + end_idx * neigh_number_,
		sizeof(neighbour_t) * neigh_number_, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	if (cluster_data_[idx].size >= maha_threshold_)
		CUCH(cudaMemcpy(cu_icov_ + idx * point_dim * point_dim, cu_icov_ + end_idx * point_dim * point_dim,
			sizeof(float) * point_dim * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	return true;
}

void gmhc::move_clusters(size_t i, size_t j, bool maha)
{
	std::vector<pasgn_t> ret;
	if (j < bounds_.maha_begin && !maha) //c+c=c
	{
		bool move = hole(j);
		upd_data_.move_a = { j, move ? bounds_.eucl_size : (asgn_t)-1 };

		upd_data_.move_b = { i, (asgn_t)-1 };

		upd_data_.new_idx = i;
	}
	else if (j < bounds_.maha_begin && maha) //c+c=m
	{
		{
			bool move = hole(j);
			upd_data_.move_a = { j, move ? bounds_.eucl_size : (asgn_t)-1 };
		}

		{
			bool move = hole(i);
			upd_data_.move_b = { i, move ? bounds_.eucl_size : (asgn_t)-1 };
		}

		++bounds_.maha_size;
		--bounds_.maha_begin;
		upd_data_.new_idx = bounds_.maha_begin;
	}
	else if (i < bounds_.maha_begin) //c+m=m
	{
		bool move = hole(i);
		upd_data_.move_a = { i, move ? bounds_.eucl_size : (asgn_t)-1 };

		upd_data_.move_b = { j, (asgn_t)-1 };

		upd_data_.new_idx = j;
	}
	else //m+m=m
	{
		bool move = hole(j);
		upd_data_.move_a = { j, move ? bounds_.maha_size : (asgn_t)-1 };

		upd_data_.move_b = { i, (asgn_t)-1 };

		upd_data_.new_idx = i;
	}
}		

void gmhc::compute_icov(size_t pos)
{
	float* icov = cu_icov_ + pos * point_dim * point_dim;

	assign_constant_storage(cu_centroids_ + pos * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	run_covariance(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, icov, id_, kernel_info{ 6, 1024, 100 });

	run_finish_covariance(icov, cluster_data_[pos].size, point_dim, tmp_icov);

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

	if (vld_)
	{
		vld_->icov_v.resize(point_dim * point_dim);
		CUCH(cudaMemcpy(vld_->icov_v.data(), icov, point_dim * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
}

void gmhc::free() {}

void gmhc::verify(pasgn_t id_pair, float dist)
{
	printf("\rIteration %d", (int)points_size - (int)cluster_count_);
	fflush(stderr);

	std::vector<float> tmp_centr;
	tmp_centr.resize(point_dim);
	CUCH(cudaMemcpy(tmp_centr.data(), cu_centroids_ + upd_data_.new_idx * point_dim, sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToHost));

	vld_->verify(id_pair, dist, tmp_centr.data());

	if (vld_->has_error())
	{
		CUCH(cudaDeviceSynchronize());
		cluster_count_ = 0;
	}
}