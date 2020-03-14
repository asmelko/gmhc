#include <gmhc.hpp>
#include <kernels.cuh>
#include <cublas_v2.h>

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

	bool first = false;
	while (chunk_count_)
	{
		if (first)
		{
			kernel_info info{ 50,1024 };
			neighbour_array_t<1>* tmp;
			CUCH(cudaMalloc(&tmp, sizeof(neighbour_array_t<1>) * points_size * info.grid_dim));
			run_neighbours(cu_centroids_, point_dim, cluster_count_, tmp, cu_neighs_, cu_min_, info);
		}
		auto min = *cu_min_;

		//spocitat ci sme maha


		run_min(input_t{ cu_centroids_, points_size, point_dim }, cu_chunks_, nullptr, kernel_info{ 50, 1024, SH });
//		auto min = run_reduce(cu_chunks_, cu_min_, chunk_count_, {});

		CUCH(cudaDeviceSynchronize());

		//retrieving merging centroid data
		cluster_data_t a, b;
		//CUCH(cudaMemcpy(&a, cu_centroid_asgns_ + min.min_i, sizeof(cluster_data_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		//CUCH(cudaMemcpy(&b, cu_centroid_asgns_ + min.min_j, sizeof(cluster_data_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		//centroid_sizes_[min.min_i] += centroid_sizes_[min.min_j];

		cluster_data_t new_data;
		new_data.id = id_;

		/*
		if (centroid_sizes_[min.min_i] > maha_threshold_)
		{
			if (a.icov)
			{
				new_data.icov = a.icov;
				if (b.icov)
					CUCH(cudaFree(new_data.icov));
			}
			else if (b.icov)
				new_data.icov = b.icov;
			else
				CUCH(cudaMalloc(&new_data.icov, point_dim * point_dim * sizeof(float)));
		}*/

		//updating centoid data
		//CUCH(cudaMemcpy(cu_centroid_asgns_ + min.min_i, &new_data, sizeof(asgn_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		//CUCH(cudaMemcpy(cu_centroid_asgns_ + min.min_j, cu_centroid_asgns_ + cluster_count_ - 1, sizeof(cluster_data_t), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

		//updating point asgns
		run_merge_clusters(cu_point_asgns_, point_dim, a.id, b.id, id_, kernel_info{ 20,512 });

		//compute new centroid
		//run_centroid(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, cu_centroids_ + min.min_i, id_, centroid_sizes_[min.min_i], kernel_info{ 50, 512 });

		//move last centroid
		CUCH(cudaMemcpy(cu_centroids_ + min.min_j, cu_centroids_ + cluster_count_ - 1, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

		//move last centroid size
		//centroid_sizes_[min.min_j] = centroid_sizes_.back();
		//centroid_sizes_.pop_back();

		/*
		if (centroid_sizes_[min.min_i] > maha_threshold_)
		{
			assign_constant_storage(cu_centroids_ + min.min_i, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			run_covariance(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, new_data.icov, id_, kernel_info{ 50, 1024, 100 });

			float* tmp;
			CUCH(cudaMalloc(&tmp, point_dim * point_dim * sizeof(float)));

			run_finish_covariance(new_data.icov, centroid_sizes_[min.min_i], point_dim, tmp);

			int info;
			BUCH(cublasSmatinvBatched(handle_, (int)point_dim, &tmp, (int)point_dim, &new_data.icov, (int)point_dim, &info, 1));
			BUCH((cublasStatus_t)info);

			CUCH(cudaFree(tmp));
		}*/

		++id_;
		--cluster_count_;
		ret.emplace_back(a.id, b.id);
	}

	return ret;
}

void gmhc::move_(size_t from, size_t to, int where)
{
	CUCH(cudaMemcpy(cu_centroids_tmp_ + (to + where) * point_dim, cu_centroids_ + to * point_dim, 
		sizeof(float) * point_dim * (to - from), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	std::memcpy(cluster_data_tmp_ + to + where, cluster_data_ + to,
		sizeof(cluster_data_t) * point_dim * (to - from));

	if (from > small_cluster_count_)
	{
		from -= small_cluster_count_ + 1;
		to -= small_cluster_count_ + 1;

		CUCH(cudaMemcpy(cu_icov_tmp_ + to + where, cu_icov_ + to,
			sizeof(float*) * (to - from), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
	}
}


void gmhc::move_clusters(size_t i, size_t j)
{
	if (j < small_cluster_count_)
	{
		move_(0, i, 1);
		move_(i + 1, j, 0);
		move_(j + 1, cluster_count_, -1);
	}
	else if (i >= small_cluster_count_)
	{
		move_(0, small_cluster_count_, 0);
		move_(small_cluster_count_ + 1, i, 1);
		move_(i + 1, j, 0);
		move_(j + 1, cluster_count_, -1);

		std::swap(cu_icov_, cu_icov_tmp_);
	}
	else
	{
		move_(0, i, 0);
		move_(i + 1, small_cluster_count_, -1);
		move_(small_cluster_count_ + 1, j, 0);
		move_(j + 1 , cluster_count_, -1);

		std::swap(cu_icov_, cu_icov_tmp_);
	}
	std::swap(cu_centroids_, cu_centroids_tmp_);
	std::swap(cluster_data_, cluster_data_tmp_);
}

void gmhc::free() {}