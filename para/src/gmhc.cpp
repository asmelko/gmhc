#include <gmhc.hpp>
#include <kernels.cuh>
#include <cublas_v2.h>
#include <cassert>

using namespace clustering;

#define SH 800
#define HSH SH/2

void gmhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	id_ = (asgn_t)data_points_size;
	cluster_count_ = data_points_size;
	small_cluster_count_ = data_points_size;
	big_cluster_count_ = 0;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
	//CUCH(cudaMalloc(&cu_centroid_asgns_, data_points_size * sizeof(cluster_data_t)));
	CUCH(cudaMalloc(&cu_chunks_, chunk_count_ * sizeof(chunk_t)));
	CUCH(cudaMalloc(&cu_min_, sizeof(chunk_t)));
	CUCH(cudaMalloc(&cu_neighs_, sizeof(neighbour_array_t<1>)* data_points_size));

	CUCH(cudaMemcpy(cu_points_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_centroids_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

	run_set_default_asgn(cu_point_asgns_, data_points_size);
	//run_set_default_asgn(cu_centroid_asgns_, data_points_size);

	cluster_data_ = new cluster_data_t[cluster_count_];
	cluster_data_tmp_ = new cluster_data_t[cluster_count_];

	for (asgn_t i = 0; i < points_size; ++i)
		cluster_data_[i] = cluster_data_t{ i,1 };

	BUCH(cublasCreate(&handle_));
}

std::vector<pasgn_t> gmhc::run()
{
	std::vector<pasgn_t> ret;

	neighbour_array_t<1>* tmp_neigh;
	kernel_info info{ 50,1024 };
	CUCH(cudaMalloc(&tmp_neigh, sizeof(neighbour_array_t<1>) * points_size * info.grid_dim));

	float* tmp_icov;
	CUCH(cudaMalloc(&tmp_icov, point_dim * point_dim * sizeof(float)));

	run_neighbours(cu_centroids_, point_dim, cluster_count_, tmp_neigh, cu_neighs_, info);

	while (chunk_count_)
	{
		chunk_t min = run_neighbours_min(cu_neighs_, cluster_count_, cu_min_);

		cluster_data_t data[2];

		data[0] = cluster_data_[min.min_j];
		data[1] = cluster_data_[min.min_i];

		auto new_centr_pos = move_clusters(min.min_i, min.min_j);

		cluster_data_[new_centr_pos].id = id_;
		cluster_data_[new_centr_pos].size = data[0].size + data[1].size;

		//updating point asgns
		run_merge_clusters(cu_point_asgns_, point_dim, data[0].id, data[1].id, id_, kernel_info{ 20,512 });

		//compute new centroid
		run_centroid(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, cu_centroids_ + new_centr_pos, id_, cluster_data_[new_centr_pos].size, kernel_info{ 50, 512 });

		if (new_centr_pos > small_cluster_count_)
			compute_icov(new_centr_pos, tmp_icov);

		++id_;
		--cluster_count_;
		ret.emplace_back(data[0].id, data[1].id);

		//update neighbours
	}

	return ret;
}

void gmhc::move_(size_t from, size_t to, int where)
{
	CUCH(cudaMemcpy(cu_centroids_tmp_ + (to + where) * point_dim, cu_centroids_ + to * point_dim, 
		sizeof(float) * point_dim * (to - from), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	std::memcpy(cluster_data_tmp_ + to + where, cluster_data_ + to,
		sizeof(cluster_data_t) * point_dim * (to - from));

	CUCH(cudaMemcpy(cu_neighs_tmp_ + to + where, cu_neighs_ + to,
		sizeof(neighbour_type) * (to - from), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	if (from > small_cluster_count_)
	{
		from -= small_cluster_count_ + 1;
		to -= small_cluster_count_ + 1;
		where = from = small_cluster_count_ + 1 && where == -1 ? 1 : where;

		if (to != cluster_count_)
		{
			float* tmp;
			CUCH(cudaMemcpy(tmp, cu_icov_ + to, sizeof(float*), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			CUCH(cudaFree(tmp));
		}

		CUCH(cudaMemcpy(cu_icov_tmp_ + to + where, cu_icov_ + to,
			sizeof(float*) * (to - from), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
	}
}


size_t gmhc::move_clusters(size_t i, size_t j)
{
	bool no_swap = false;
	size_t new_idx;

	size_t size = cluster_data_[i].size + cluster_data_[j].size;

	if (j < small_cluster_count_)
	{
		if (size >= maha_threshold_)
		{
			move_(0, i, 0);
			move_(i + 1, j, -1);
			move_(j + 1, small_cluster_count_, -2);
			move_(small_cluster_count_  + 1, cluster_count_, -1);

			small_cluster_count_ -= 2;
			big_cluster_count_++;
		}
		else
		{
			move_(0, i, 1);
			move_(i + 1, j, 0);
			move_(j + 1, cluster_count_, -1);
			no_swap = true;

			small_cluster_count_--;
		}
	}
	else if (i >= small_cluster_count_)
	{
		move_(0, small_cluster_count_, 0);
		move_(small_cluster_count_ + 1, i, 1);
		move_(i + 1, j, 0);
		move_(j + 1, cluster_count_, -1);

		big_cluster_count_--;
	}
	else
	{
		move_(0, i, 0);
		move_(i + 1, small_cluster_count_, -1);
		move_(small_cluster_count_ + 1, j, 0);
		move_(j + 1 , cluster_count_, -1);

		small_cluster_count_--;
	}

	std::swap(cu_centroids_, cu_centroids_tmp_);
	std::swap(cu_neighs_, cu_neighs_tmp_);
	std::swap(cluster_data_, cluster_data_tmp_);
	if (!no_swap)
		std::swap(cu_icov_, cu_icov_tmp_);

	if (size >= maha_threshold_)
		return small_cluster_count_;
	else
		return 0;
}

void gmhc::compute_icov(size_t pos, float* cu_tmp_icov)
{
	assert(cluster_data_[pos].size >= maha_threshold_);

	float* icov;
	CUCH(cudaMalloc(&icov, point_dim * point_dim * sizeof(float)));

	assign_constant_storage(cu_centroids_ + pos, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	run_covariance(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, cu_tmp_icov, id_, kernel_info{ 50, 1024, 100 });


	run_finish_covariance(cu_tmp_icov, cluster_data_[pos].size, point_dim, icov);

	int info;
	BUCH(cublasSmatinvBatched(handle_, (int)point_dim, &cu_tmp_icov, (int)point_dim, &icov, (int)point_dim, &info, 1));
	BUCH((cublasStatus_t)info);

	cudaMemcpy(cu_icov_ + pos - small_cluster_count_, &icov, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void gmhc::free() {}