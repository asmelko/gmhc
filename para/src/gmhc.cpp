#include <gmhc.hpp>
#include <kernels.cuh>
#include <cublas_v2.h>

using namespace clustering;

#define SH 800
#define HSH SH/2

void gmhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	size_t chunks_line = (points_size + HSH - 1) / HSH;
	chunk_count_ = (chunks_line + 1) * chunks_line / 2;
	id_ = (asgn_t)data_points_size;
	cluster_count_ = data_points_size;
	icov_size_ = (data_point_dim + 1) * data_point_dim / 2;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
	CUCH(cudaMalloc(&cu_centroid_asgns_, data_points_size * sizeof(centroid_data_t)));
	CUCH(cudaMalloc(&cu_chunks_, chunk_count_ * sizeof(chunk_t)));
	CUCH(cudaMalloc(&cu_min_, sizeof(chunk_t)));

	CUCH(cudaMemcpy(cu_points_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_centroids_, data_points, data_points_size * data_point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

	run_set_default_asgn(cu_point_asgns_, data_points_size);
	run_set_default_asgn(cu_centroid_asgns_, data_points_size);

	for (size_t i = 0; i < points_size; ++i)
		centroid_sizes_.push_back(1);

	BUCH(cublasCreate(&handle_));
}

std::vector<pasgn_t> gmhc::run()
{
	std::vector<pasgn_t> ret;

	while (chunk_count_)
	{
		run_min(input_t{ cu_centroids_, points_size, point_dim }, cu_chunks_, kernel_info{ 50, 1024, SH });
		auto min = run_reduce(cu_chunks_, cu_min_, chunk_count_, {});

		CUCH(cudaDeviceSynchronize());

		//retrieving merging centroid data
		centroid_data_t a, b;
		CUCH(cudaMemcpy(&a, cu_centroid_asgns_ + min.min_i, sizeof(centroid_data_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		CUCH(cudaMemcpy(&b, cu_centroid_asgns_ + min.min_j, sizeof(centroid_data_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		centroid_sizes_[min.min_i] += centroid_sizes_[min.min_j];

		centroid_data_t new_data;
		new_data.id = id_;

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
		}

		//updating centoid data
		CUCH(cudaMemcpy(cu_centroid_asgns_ + min.min_i, &new_data, sizeof(asgn_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_centroid_asgns_ + min.min_j, cu_centroid_asgns_ + cluster_count_ - 1, sizeof(centroid_data_t), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

		//updating point asgns
		run_merge_clusters(cu_point_asgns_, point_dim, a.id, b.id, id_, kernel_info{ 20,512 });

		//compute new centroid
		run_centroid(input_t{ cu_points_, points_size, point_dim }, cu_point_asgns_, cu_centroids_ + min.min_i, id_, centroid_sizes_[min.min_i], kernel_info{ 50, 512 });

		//move last centroid
		CUCH(cudaMemcpy(cu_centroids_ + min.min_j, cu_centroids_ + cluster_count_ - 1, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

		//move last centroid size
		centroid_sizes_[min.min_j] = centroid_sizes_.back();
		centroid_sizes_.pop_back();

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
		}

		++id_;
		--cluster_count_;
		ret.emplace_back(a.id, b.id);
	}

	return ret;
}

void gmhc::free() {}