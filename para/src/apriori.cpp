#include "apriori.hpp"

#include <cfloat>

#include "gmhc.hpp"
#include "kernels.cuh"
#include "validator.hpp"

#define KERNEL_INFO kernel_info(80, 1024)

using namespace clustering;

clustering_context_t::clustering_context_t(shared_apriori_data_t& shared_data)
    : shared(shared_data)
{}

void clustering_context_t::initialize()
{
    compute_data.centroids = cu_centroids;
    compute_data.inverses = cu_inverses;
    compute_data.dim = point_dim;

    update_data.to_update = cu_updates;

    maha_cluster_count = 0;
    switched_to_full_maha = false;
    initialize_neighbors = true;
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
    if (initialize_neighbors || (!switched_to_full_maha && maha_cluster_count == cluster_count))
    {
        run_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, starting_info);

        initialize_neighbors = false;
        switched_to_full_maha = maha_cluster_count == cluster_count;
    }
    else
        run_update_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, update_data, starting_info);
}

void clustering_context_t::move_clusters(csize_t i, csize_t j)
{
    update_data.old_a = i;
    update_data.old_b = j;

    csize_t end_idx = cluster_count;

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

    if (cluster_data[j].size >= maha_threshold)
        CUCH(cudaMemcpy(cu_inverses + j * icov_size,
            cu_inverses + end_idx * icov_size,
            sizeof(float) * icov_size,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice));
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
        cu_centroids + new_idx * point_dim,
        shared.id,
        cluster_data[new_idx].size,
        KERNEL_INFO);

    CUCH(cudaDeviceSynchronize());

    // compute new inverse of covariance matrix
    compute_icov(new_idx);

    // update counts
    ++shared.id;
    --shared.cluster_count;
    --cluster_count;

    if (cluster_data[new_idx].size >= maha_threshold)
    {
        if (merged[0].size < maha_threshold && merged[1].size < maha_threshold)
            ++maha_cluster_count;
        else if (merged[0].size >= maha_threshold && merged[1].size >= maha_threshold)
            --maha_cluster_count;
    }
}

void clustering_context_t::compute_covariance(csize_t pos)
{
    float* tmp_cov = shared.cu_tmp_icov + point_dim * point_dim;
    float* cov = shared.cu_tmp_icov;

    float wf;

    if (cluster_data[pos].size == 2)
    {
        run_set_unit_matrix(cov, point_dim);
        wf = 0;
    }
    else
    {
        assign_constant_storage(
            cu_centroids + pos * point_dim, point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        run_covariance(input_t { cu_points, point_size, point_dim }, cu_point_asgns, tmp_cov, shared.id, KERNEL_INFO);

        run_finish_covariance(tmp_cov, cluster_data[pos].size, point_dim, cov);

        wf = std::max(cluster_data[pos].size / (float)maha_threshold, 1.f);
    }

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
            vld->set_mf(subthreshold_kind != subthreshold_handling_kind::MAHAL0, tmp_cov, shared.cu_info);
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
    auto wf = std::max(cluster_data[pos].size / (float)maha_threshold, 1.f);

    if (euclidean_based_kind(subthreshold_kind) && wf < 1)
    {
        run_set_unit_matrix(shared.cu_tmp_icov, point_dim);
        run_store_icovariance_data(cu_inverses + pos * icov_size, cu_mfactors + pos, shared.cu_tmp_icov, 1, point_dim);
        return;
    }

    compute_covariance(pos);

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

void clustering_context_t::verify(pasgn_t id_pair, float dist)
{
    CUCH(cudaDeviceSynchronize());

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
        shared.cluster_count = 0;
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

    float* rhs_icov = cu_inverses + icov_size * i;

    float dist = run_point_maha(
        cu_centroids + point_dim * j, cu_centroids + point_dim * i, point_dim, cu_inverses + icov_size * j, rhs_icov);


    if (isinf(dist))
        return FLT_MAX;
    return dist;
}