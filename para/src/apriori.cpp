#include "apriori.hpp"

#include <cfloat>

#include "gmhc.hpp"
#include "kernels.cuh"

#define KERNEL_INFO kernel_info(80, 1024)

using namespace clustering;

clustering_context_t::clustering_context_t(shared_apriori_data_t& shared_data)
    : shared(shared_data)
{}

void clustering_context_t::initialize()
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

gmhc::res_t clustering_context_t::iterate()
{
    cluster_data_t data[2];

    auto min = run_neighbors_min<shared_apriori_data_t::neighbors_size>(cu_neighbors, bounds, shared.cu_min);

    data[0] = cluster_data[min.min_i];
    data[1] = cluster_data[min.min_j];

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

    run_update_neighbors<shared_apriori_data_t::neighbors_size>(
        compute_data, cu_tmp_neighbors, cu_neighbors, bounds, update_data, starting_info);

    return std::make_pair(ret, min.min_dist);
}

bool clustering_context_t::remove(csize_t idx)
{
    csize_t end_idx;
    if (idx < bounds.maha_begin)
        end_idx = --bounds.eucl_size;
    else
        end_idx = bounds.maha_begin + --bounds.maha_size;

    if (idx == end_idx)
        return false;

    CUCH(cudaMemcpy(cu_centroids + idx * point_dim,
        cu_centroids + end_idx * point_dim,
        sizeof(float) * point_dim,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    cluster_data[idx] = cluster_data[end_idx];

    CUCH(cudaMemcpy(cu_neighbors + idx * shared.neighbors_size,
        cu_neighbors + end_idx * shared.neighbors_size,
        sizeof(neighbor_t) * shared.neighbors_size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    if (cluster_data[idx].size >= maha_threshold)
        CUCH(cudaMemcpy(cu_inverses + idx * icov_size,
            cu_inverses + end_idx * icov_size,
            sizeof(float) * icov_size,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    return true;
}

void clustering_context_t::move_clusters(csize_t i, csize_t j, bool maha)
{
    if (j < bounds.maha_begin && !maha) // c+c=c
    {
        bool move = remove(j);
        update_data.move_a = { j, move ? bounds.eucl_size : (asgn_t)-1 };

        update_data.move_b = { i, (asgn_t)-1 };

        update_data.new_idx = i;
    }
    else if (j < bounds.maha_begin && maha) // c+c=m
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
    else if (i < bounds.maha_begin) // c+m=m
    {
        bool move = remove(i);
        update_data.move_a = { i, move ? bounds.eucl_size : (asgn_t)-1 };

        update_data.move_b = { j, (asgn_t)-1 };

        update_data.new_idx = j;
    }
    else // m+m=m
    {
        bool move = remove(j);
        update_data.move_a = { j, move ? bounds.maha_begin + bounds.maha_size : (asgn_t)-1 };

        update_data.move_b = { i, (asgn_t)-1 };

        update_data.new_idx = i;
    }
}

void clustering_context_t::update_iteration(const cluster_data_t* merged)
{
    auto new_idx = update_data.new_idx;

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
    }
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

    // test inverse
    if (vld)
    {
        vld->icov_v.resize(point_dim * point_dim);
        CUCH(cudaDeviceSynchronize());
        CUCH(cudaMemcpy(
            vld->icov_v.data(), cov, point_dim * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < point_dim; i++)
            for (size_t j = i + 1; j < point_dim; j++)
                vld->icov_v[i + point_dim * j] = vld->icov_v[j + point_dim * i];
    }
}

void clustering_context_t::verify(pasgn_t id_pair, float dist)
{
    CUCH(cudaDeviceSynchronize());

    // copy centroid
    std::vector<float> tmp_centr;
    tmp_centr.resize(point_dim);
    CUCH(cudaMemcpy(tmp_centr.data(),
        cu_centroids + update_data.new_idx * point_dim,
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
    for (csize_t i = 0; i < bounds.eucl_size; i++)
    {
        if (cluster_data[i].id == expected_id.first || cluster_data[i].id == expected_id.second)
            idxs.push_back(i);
    }
    for (csize_t i = bounds.maha_begin; i < bounds.maha_begin + bounds.maha_size; i++)
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

    float dist;
    if (i < bounds.maha_begin && j < bounds.maha_begin)
    {
        dist = run_point_eucl(cu_centroids + point_dim * i, cu_centroids + point_dim * j, point_dim);
    }
    else
    {
        float* rhs_icov;
        if (i < bounds.maha_begin)
            rhs_icov = nullptr;
        else
            rhs_icov = cu_inverses + icov_size * i;

        dist = run_point_maha(cu_centroids + point_dim * j,
            cu_centroids + point_dim * i,
            point_dim,
            cu_inverses + icov_size * j,
            rhs_icov);
    }

    if (isinf(dist))
        return FLT_MAX;
    return dist;
}