#include "apriori.hpp"

#include <cfloat>

#include "gmhc.hpp"
#include "kernels.cuh"
#include "validator.hpp"

#define KERNEL_INFO kernel_info(6, 1024)

using namespace clustering;

clustering_context_t::clustering_context_t(shared_apriori_data_t& shared_data)
    : shared(shared_data)
{}

void clustering_context_t::initialize(bool is_final_context)
{
    compute_data.centroids = cu_centroids;
    compute_data.inverses = cu_inverses;
    compute_data.mfactors = cu_mfactors;
    compute_data.dim = point_dim;

    update_data.to_update = cu_updates;
    update_data.update_size = shared.cu_upd_size;

    maha_cluster_count = 0;
    switched_to_full_maha = false;
    initialize_neighbors = true;
    is_final = is_final_context;
}

gmhc::res_t clustering_context_t::iterate()
{
    cluster_data_t data[2];

    compute_neighbors();

    auto min = run_neighbors_min<shared_apriori_data_t::neighbors_size>(cu_neighbors, cluster_count, shared.cu_min);

    data[0] = cluster_data[min.min_i];
    data[1] = cluster_data[min.min_j];

    move_clusters(min.min_i, min.min_j);

    update_iteration(data);

    if (data[0].id > data[1].id)
        std::swap(data[0].id, data[1].id);

    pasgn_t ret(data[0].id, data[1].id);

    if (vld)
        verify(ret, min.min_dist);

    return std::make_pair(ret, min.min_dist);
}

void clustering_context_t::compute_neighbors()
{
    if (initialize_neighbors || (is_final && !switched_to_full_maha && maha_cluster_count == cluster_count))
    {
        initialize_neighbors = false;
        switched_to_full_maha = maha_cluster_count == cluster_count;

        if (switched_to_full_maha)
            compute_data.mfactors = nullptr;

        run_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, cluster_count == point_size, starting_info);
    }
    else
    {
        bool use_eucl = !switched_to_full_maha && subthreshold_kind == subthreshold_handling_kind::EUCLID;

        run_update_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, update_data, use_eucl, starting_info);

        run_update_neighbors_new<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, update_data.old_a, use_eucl, starting_info);
    }
}

void clustering_context_t::move_clusters(csize_t i, csize_t j)
{
    update_data.old_a = i;
    update_data.old_b = j;

    csize_t end_idx = --cluster_count;

    if (j == end_idx)
        return;

    CUCH(cudaMemcpy(cu_centroids + j * point_dim,
        cu_centroids + end_idx * point_dim,
        sizeof(float) * point_dim,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    cluster_data[j] = cluster_data[end_idx];

    CUCH(cudaMemcpy(cu_neighbors + j * shared.neighbors_size,
        cu_neighbors + end_idx * shared.neighbors_size,
        sizeof(neighbor_t) * shared.neighbors_size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    CUCH(cudaMemcpy(cu_inverses + j * icov_size,
        cu_inverses + end_idx * icov_size,
        sizeof(float) * icov_size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    CUCH(cudaMemcpy(cu_mfactors + j, cu_mfactors + end_idx, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

void clustering_context_t::update_iteration(const cluster_data_t* merged)
{
    auto new_idx = update_data.old_a;

    // update cluster data
    cluster_data[new_idx].id = shared.id;
    cluster_data[new_idx].size = merged[0].size + merged[1].size;

    // updating point asgns
    run_merge_clusters(cu_point_asgns, point_size, merged[0].id, merged[1].id, shared.id, KERNEL_INFO);

    // compute new centroid
    run_centroid(input_t { cu_points, point_size, point_dim },
        cu_point_asgns,
        shared.cu_work_centroid,
        cu_centroids + new_idx * point_dim,
        shared.id,
        cluster_data[new_idx].size,
        KERNEL_INFO);

    // compute new inverse of covariance matrix
    compute_icov(new_idx);

    ++shared.id;

    if (cluster_data[new_idx].size >= maha_threshold)
    {
        if (merged[0].size < maha_threshold && merged[1].size < maha_threshold)
            ++maha_cluster_count;
        else if (merged[0].size >= maha_threshold && merged[1].size >= maha_threshold)
            --maha_cluster_count;
    }
}

void clustering_context_t::compute_covariance(csize_t pos, float wf)
{
    float* tmp_cov = shared.cu_tmp_icov + point_dim * point_dim;
    float* cov = shared.cu_tmp_icov;

    if (cluster_data[pos].size == 2)
    {
        run_set_unit_matrix(cov, point_dim);
    }
    else
    {
        CUCH(cudaDeviceSynchronize()); // assign_constant_storage depends on run_centroid

        assign_constant_storage(
            cu_centroids + pos * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        run_covariance(input_t { cu_points, point_size, point_dim },
            cu_point_asgns,
            shared.cu_work_covariance,
            cov,
            shared.id,
            cluster_data[pos].size,
            KERNEL_INFO);

        if (wf < 1)
        {
            if (subthreshold_kind != subthreshold_handling_kind::MAHAL0)
            {
                CUCH(cudaMemcpy(
                    tmp_cov, cov, sizeof(float) * point_dim * point_dim, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

                SOCH(cusolverDnSpotrf(shared.cusolver_handle,
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    (int)point_dim,
                    tmp_cov,
                    (int)point_dim,
                    shared.cu_workspace,
                    shared.workspace_size,
                    shared.cu_info));
            }

            run_transform_cov(
                cov, point_dim, wf, subthreshold_kind != subthreshold_handling_kind::MAHAL0, tmp_cov, shared.cu_info);

            if (vld)
                vld->set_mf(tmp_cov, shared.cu_info);
        }
    }

    if (vld)
        vld->set_cov(cov);
}

bool euclidean_based_kind(subthreshold_handling_kind kind)
{
    return kind == subthreshold_handling_kind::EUCLID || kind == subthreshold_handling_kind::EUCLID_MAHAL;
}

void clustering_context_t::compute_icov(csize_t pos)
{
    auto wf = compute_weight_factor(pos);

    if (euclidean_based_kind(subthreshold_kind) && wf < 1)
    {
        run_set_unit_matrix(shared.cu_tmp_icov, point_dim);
        run_store_icovariance_data(cu_inverses + pos * icov_size, cu_mfactors + pos, shared.cu_tmp_icov, 1, point_dim);
        return;
    }

    compute_covariance(pos, wf);

    auto cov = shared.cu_tmp_icov;

    int info;

    SOCH(cusolverDnSpotrf(shared.cusolver_handle,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        (int)point_dim,
        cov,
        (int)point_dim,
        shared.cu_workspace,
        shared.workspace_size,
        shared.cu_info));

    CUCH(cudaDeviceSynchronize());
    CUCH(cudaMemcpy(&info, shared.cu_info, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (info != 0)
    {
        run_set_unit_matrix(shared.cu_tmp_icov, point_dim);
        run_store_icovariance_data(cu_inverses + pos * icov_size, cu_mfactors + pos, shared.cu_tmp_icov, 1, point_dim);
        return;
    }

    run_compute_store_icov_mf(cu_mfactors + pos, point_dim, cov);

    SOCH(cusolverDnSpotri(shared.cusolver_handle,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        (int)point_dim,
        cov,
        (int)point_dim,
        shared.cu_workspace,
        shared.workspace_size,
        shared.cu_info));

    run_store_icovariance_data(cu_inverses + pos * icov_size, nullptr, cov, 0, point_dim);
}

float clustering_context_t::compute_weight_factor(csize_t pos)
{
    if (cluster_data[pos].size == 2)
        return 0.f;

    if (maha_threshold > 0)
        return std::min(1.f, cluster_data[pos].size / (float)maha_threshold);
    return 0;
}

void clustering_context_t::verify(pasgn_t id_pair, float dist)
{
    CUCH(cudaDeviceSynchronize());
    CUCH(cudaGetLastError());

    vld->set_icov(shared.cu_tmp_icov);
    vld->set_icmf(cu_mfactors + update_data.old_a);

    // copy centroid
    std::vector<float> tmp_centr;
    tmp_centr.resize(point_dim);
    CUCH(cudaMemcpy(tmp_centr.data(),
        cu_centroids + update_data.old_a * point_dim,
        sizeof(float) * point_dim,
        cudaMemcpyKind::cudaMemcpyDeviceToHost));

    vld->verify(
        id_pair, dist, tmp_centr.data(), std::bind(&clustering_context_t::recompute_dist, this, std::placeholders::_1));

    if (vld->has_error())
    {
        CUCH(cudaDeviceSynchronize());
        cluster_count = 0;
    }
}

float clustering_context_t::recompute_dist(pasgn_t expected_id)
{
    std::vector<csize_t> idxs;
    for (csize_t i = 0; i < cluster_count; i++)
    {
        if (cluster_data[i].id == expected_id.first || cluster_data[i].id == expected_id.second)
            idxs.push_back(i);
    }

    if (idxs.size() != 2)
        return 0;

    csize_t i = idxs[0];
    csize_t j = idxs[1];
    if (idxs[0] > idxs[1])
    {
        i = idxs[1];
        j = idxs[0];
    }

    float dist = run_point_maha(cu_centroids + point_dim * j,
        cu_centroids + point_dim * i,
        point_dim,
        cu_inverses + icov_size * j,
        cu_inverses + icov_size * i,
        cu_mfactors + j,
        cu_mfactors + i);


    if (isinf(dist))
        return FLT_MAX;
    return dist;
}