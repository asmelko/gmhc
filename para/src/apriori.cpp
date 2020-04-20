#include <apriori.hpp>

#include <kernels.cuh>

#include <gmhc.hpp>

using namespace clustering;

apriori_cluster_t::apriori_cluster_t(shared_apriori_data_t& shared_data)
	: shared(shared_data) {}

void apriori_cluster_t::initialize()
{
	bounds.eucl_size = point_size;
	bounds.maha_begin = point_size;
	bounds.maha_size = 0;

	compute_data.centroids = cu_centroids;
	compute_data.inverses = cu_inverses;
	compute_data.dim = point_dim;

	update_data.eucl_update_size = shared.cu_eucl_upd_size;
	update_data.maha_update_size = shared.cu_maha_upd_size;
	update_data.to_update = cu_updates;
}

pasgn_t apriori_cluster_t::iterate()
{
	cluster_data_t data[2];

	auto min = run_neighbours_min<shared_apriori_data_t::neighbours_size>(cu_neighbours, bounds, shared.cu_min);

	data[0] = clusters[min.min_i];
	data[1] = clusters[min.min_j];

	move_clusters(min.min_i, min.min_j, data[0].size + data[1].size >= maha_threshold);

	update_iteration(data);

	++shared.id;
	--shared.cluster_count;
	--cluster_count;

	if (data[0].id > data[1].id)
		std::swap(data[0].id, data[1].id);

	pasgn_t ret(data[0].id, data[1].id);

	if (vld)
		verify(ret, min.min_dist);

	run_update_neighbours<shared_apriori_data_t::neighbours_size>(compute_data, cu_tmp_neighbours, cu_neighbours, bounds, update_data, starting_info);

	return ret;
}

bool apriori_cluster_t::remove(csize_t idx)
{
	csize_t end_idx;
	if (idx < bounds.maha_begin)
		end_idx = --bounds.eucl_size;
	else
		end_idx = bounds.maha_begin + --bounds.maha_size;

	if (idx == end_idx)
		return false;

	CUCH(cudaMemcpy(cu_centroids + idx * point_dim, cu_centroids + end_idx * point_dim,
		sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	clusters[idx] = clusters[end_idx];

	CUCH(cudaMemcpy(cu_neighbours + idx * shared.neighbours_size, cu_neighbours + end_idx * shared.neighbours_size,
		sizeof(neighbour_t) * shared.neighbours_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	if (clusters[idx].size >= maha_threshold)
		CUCH(cudaMemcpy(cu_inverses + idx * icov_size, cu_inverses + end_idx * icov_size,
			sizeof(float) * icov_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

	return true;
}

void apriori_cluster_t::move_clusters(csize_t i, csize_t j, bool maha)
{
	if (j < bounds.maha_begin && !maha) //c+c=c
	{
		bool move = remove(j);
		update_data.move_a = { j, move ? bounds.eucl_size : (asgn_t)-1 };

		update_data.move_b = { i, (asgn_t)-1 };

		update_data.new_idx = i;
	}
	else if (j < bounds.maha_begin && maha) //c+c=m
	{
		{
			bool move = remove(j);
			update_data.move_a = { j, move ? bounds.eucl_size : (asgn_t)-1 };
		}

		{
			bool move = remove(i);
			update_data.move_b = { i, move ? bounds.eucl_size : (asgn_t)-1 };
		}

		++bounds.maha_size;
		--bounds.maha_begin;
		update_data.new_idx = bounds.maha_begin;
	}
	else if (i < bounds.maha_begin) //c+m=m
	{
		bool move = remove(i);
		update_data.move_a = { i, move ? bounds.eucl_size : (asgn_t)-1 };

		update_data.move_b = { j, (asgn_t)-1 };

		update_data.new_idx = j;
	}
	else //m+m=m
	{
		bool move = remove(j);
		update_data.move_a = { j, move ? bounds.maha_begin + bounds.maha_size : (asgn_t)-1 };

		update_data.move_b = { i, (asgn_t)-1 };

		update_data.new_idx = i;
	}
}

void apriori_cluster_t::update_iteration(const cluster_data_t* merged)
{
	auto new_idx = update_data.new_idx;

	//update cluster data
	clusters[new_idx].id = shared.id;
	clusters[new_idx].size = merged[0].size + merged[1].size;

	//updating point asgns
	run_merge_clusters(cu_point_asgns, point_size, merged[0].id, merged[1].id, shared.id, kernel_info(6, 1024));

	//compute new centroid
	run_centroid(input_t{ cu_points, point_size, point_dim }, cu_point_asgns, cu_centroids + new_idx * point_dim, shared.id, clusters[new_idx].size, kernel_info(6, 1024));

	//compute new inverse of covariance matrix
	if (clusters[new_idx].size >= maha_threshold)
	{
		CUCH(cudaDeviceSynchronize());
		compute_icov(new_idx);
	}
}

void apriori_cluster_t::compute_icov(csize_t pos)
{
	float* icov = shared.cu_tmp_icov + point_dim * point_dim;

	//compute covariance
	assign_constant_storage(cu_centroids + pos * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	run_covariance(input_t{ cu_points, point_size, point_dim }, cu_point_asgns, icov, shared.id, kernel_info(6, 1024, 100));
	run_finish_covariance(icov, clusters[pos].size, point_dim, shared.cu_tmp_icov);

	CUCH(cudaDeviceSynchronize());

	//test covariance
	if (vld)
	{
		vld->cov_v.resize((point_dim + 1) * point_dim / 2);
		CUCH(cudaMemcpy(vld->cov_v.data(), icov, ((point_dim + 1) * point_dim / 2) * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}

	//compute inverse
	{
		CUCH(cudaMemcpy(shared.cu_read_icov, &shared.cu_tmp_icov, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(shared.cu_write_icov, &icov, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice));

		if (point_dim <= 16)
			BUCH(cublasSmatinvBatched(shared.cublas_handle, (int)point_dim, shared.cu_read_icov, (int)point_dim, shared.cu_write_icov, (int)point_dim, shared.cu_info, 1));
		else
		{
			BUCH(cublasSgetrfBatched(shared.cublas_handle, (int)point_dim, shared.cu_read_icov, (int)point_dim, shared.cu_pivot, shared.cu_info, 1));
			BUCH(cublasSgetriBatched(shared.cublas_handle, (int)point_dim, shared.cu_read_icov, (int)point_dim, shared.cu_pivot, shared.cu_write_icov, (int)point_dim, shared.cu_info, 1));
		}

		CUCH(cudaDeviceSynchronize());

		int info;
		CUCH(cudaMemcpy(&info, shared.cu_info, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		if (info != 0)
			run_set_default_inverse(icov, point_dim * point_dim);

		run_store_icovariance(cu_inverses + pos * icov_size, icov, point_dim);
	}

	//test inverse
	if (vld)
	{
		vld->icov_v.resize(point_dim * point_dim);
		CUCH(cudaMemcpy(vld->icov_v.data(), icov, point_dim * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
}

void apriori_cluster_t::verify(pasgn_t id_pair, float dist)
{
	//copy centroid
	std::vector<float> tmp_centr;
	tmp_centr.resize(point_dim);
	CUCH(cudaMemcpy(tmp_centr.data(), cu_centroids + update_data.new_idx * point_dim, sizeof(float) * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToHost));

	vld->verify(id_pair, dist, tmp_centr.data());

	if (vld->has_error())
	{
		CUCH(cudaDeviceSynchronize());
		shared.cluster_count = 0;
		cluster_count = 0;
	}
}
