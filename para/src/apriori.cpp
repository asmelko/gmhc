#include "apriori.hpp"

#include <cfloat>

#include "gmhc.hpp"
#include "kernels.cuh"
#include "validator.hpp"

using namespace clustering;

clustering_context_t::clustering_context_t(shared_apriori_data_t& shared_data)
    : shared(shared_data)
{}

void clustering_context_t::initialize(bool is_final_context, bool normalize_flag)
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
    normalize = normalize_flag;

    neighbor_info.stream = shared.streams[0];
    rest_info = kernel_info(neighbor_info.grid_dim / 2, 1024, 0, shared.streams[1]);
}

bool clustering_context_t::need_recompute_neighbors()
{
    return initialize_neighbors || (is_final && !switched_to_full_maha && maha_cluster_count == cluster_count);
}

bool clustering_context_t::can_use_euclidean_distance()
{
    return (!switched_to_full_maha && subthreshold_kind == subthreshold_handling_kind::EUCLID)
        || (maha_cluster_count == 0 && subthreshold_kind == subthreshold_handling_kind::EUCLID_MAHAL);
}

void clustering_context_t::compute_neighbors()
{
    shared.timer.record(shared.timer.nei_new_start, neighbor_info.stream);

    if (need_recompute_neighbors())
    {
        initialize_neighbors = false;
        switched_to_full_maha = maha_cluster_count == cluster_count;

        if (switched_to_full_maha && !normalize)
            compute_data.mfactors = nullptr;

        run_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, cluster_count == point_size, neighbor_info);
    }
    else
    {
        bool use_eucl = can_use_euclidean_distance();

        run_update_neighbors_new<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, update_data.old_a, use_eucl, neighbor_info);
    }

    shared.timer.record(shared.timer.nei_new_stop, neighbor_info.stream);
}

std::vector<gmhc::res_t> clustering_context_t::run()
{
    std::vector<gmhc::res_t> res;

    while (cluster_count > 1)
    {
        CUCH(cudaStreamSynchronize(rest_info.stream));

        shared.timer.record_iteration();

        compute_neighbors();

        auto min = run_neighbors_min<shared_apriori_data_t::neighbors_size>(
            cu_neighbors, cluster_count, shared.cu_min, neighbor_info);

        pasgn_t merged_ids(cluster_data[min.min_i].id, cluster_data[min.min_j].id);
        asgn_t new_id = shared.id;

        update_iteration_host(min);

        move_clusters(min.min_j);

        update_iteration_device(merged_ids.first, merged_ids.second, new_id);

        // append to res
        if (merged_ids.first > merged_ids.second)
            std::swap(merged_ids.first, merged_ids.second);
        res.emplace_back(merged_ids, min.min_dist);

        // verify
        if (vld)
            verify(merged_ids, min.min_dist);
    }

    return res;
}

void clustering_context_t::move_clusters(csize_t pos)
{
    csize_t end_idx = cluster_count;

    if (pos == end_idx)
        return;

    CUCH(cudaMemcpyAsync(cu_centroids + pos * point_dim,
        cu_centroids + end_idx * point_dim,
        sizeof(float) * point_dim,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        neighbor_info.stream));

    cluster_data[pos] = cluster_data[end_idx];

    CUCH(cudaMemcpyAsync(cu_neighbors + pos * shared.neighbors_size,
        cu_neighbors + end_idx * shared.neighbors_size,
        sizeof(neighbor_t) * shared.neighbors_size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        neighbor_info.stream));

    CUCH(cudaMemcpyAsync(cu_inverses + pos * icov_size,
        cu_inverses + end_idx * icov_size,
        sizeof(float) * icov_size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        neighbor_info.stream));

    CUCH(cudaMemcpyAsync(cu_mfactors + pos,
        cu_mfactors + end_idx,
        sizeof(float),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        neighbor_info.stream));
}

void clustering_context_t::update_iteration_host(chunk_t min)
{
    --cluster_count;

    update_data.old_a = min.min_i;
    update_data.old_b = min.min_j;

    auto new_idx = update_data.old_a;
    auto merged_A_size = cluster_data[min.min_i].size;
    auto merged_B_size = cluster_data[min.min_j].size;

    // update cluster data
    cluster_data[new_idx].id = shared.id++;
    cluster_data[new_idx].size = merged_A_size + merged_B_size;

    if (cluster_data[new_idx].size >= maha_threshold)
    {
        if (merged_A_size < maha_threshold && merged_B_size < maha_threshold)
            ++maha_cluster_count;
        else if (merged_A_size >= maha_threshold && merged_B_size >= maha_threshold)
            --maha_cluster_count;
    }
}

void clustering_context_t::update_iteration_device(asgn_t merged_A, asgn_t merged_B, asgn_t new_id)
{
    shared.timer.record(shared.timer.nei_rest_start, neighbor_info.stream);
    // start computing neighbor of all but new cluster
    if (!need_recompute_neighbors())
    {
        bool use_eucl = can_use_euclidean_distance();

        run_update_neighbors<shared_apriori_data_t::neighbors_size>(
            compute_data, cu_tmp_neighbors, cu_neighbors, cluster_count, update_data, use_eucl, neighbor_info);
    }
    shared.timer.record(shared.timer.nei_rest_stop, neighbor_info.stream);

    auto new_idx = update_data.old_a;

    // updating point asgns
    run_merge_clusters(
        cu_point_asgns, shared.cu_asgn_idxs_, shared.cu_idxs_size_, point_size, merged_A, merged_B, new_id, rest_info);

    // compute new centroid
    run_centroid(cu_points,
        shared.cu_asgn_idxs_,
        shared.cu_work_centroid,
        cu_centroids + new_idx * point_dim,
        cluster_data[new_idx].size,
        point_dim,
        rest_info);

    // compute new inverse of covariance matrix
    compute_icov(new_idx);
}

void clustering_context_t::compute_covariance(csize_t pos, float wf)
{
    float* tmp_cov = shared.cu_tmp_icov + point_dim * point_dim;
    float* cov = shared.cu_tmp_icov;

    if (cluster_data[pos].size == 2)
    {
        run_set_unit_matrix(cov, point_dim, rest_info.stream);
    }
    else
    {
        assign_constant_storage(cu_centroids + pos * point_dim,
            point_dim * sizeof(float),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            rest_info.stream);

        shared.timer.record(shared.timer.cov_start, rest_info.stream);

        run_covariance(cu_points,
            shared.cu_asgn_idxs_,
            shared.cu_work_covariance,
            cov,
            cluster_data[pos].size,
            point_dim,
            rest_info);

        shared.timer.record(shared.timer.cov_stop, rest_info.stream);

        if (wf < 1)
        {
            if (subthreshold_kind != subthreshold_handling_kind::MAHAL0)
            {
                CUCH(cudaMemcpyAsync(tmp_cov,
                    cov,
                    sizeof(float) * point_dim * point_dim,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    rest_info.stream));

                SOCH(cusolverDnSpotrf(shared.cusolver_handle,
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    (int)point_dim,
                    tmp_cov,
                    (int)point_dim,
                    shared.cu_workspace,
                    shared.workspace_size,
                    shared.cu_info));
            }

            run_transform_cov(cov,
                point_dim,
                wf,
                subthreshold_kind != subthreshold_handling_kind::MAHAL0,
                tmp_cov,
                shared.cu_info,
                rest_info.stream);

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
        run_set_unit_matrix(shared.cu_tmp_icov, point_dim, rest_info.stream);
        run_store_icovariance_data(
            cu_inverses + pos * icov_size, cu_mfactors + pos, shared.cu_tmp_icov, 1, point_dim, rest_info.stream);
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

    CUCH(cudaStreamSynchronize(rest_info.stream));
    CUCH(cudaMemcpyAsync(&info, shared.cu_info, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost, rest_info.stream));

    if (info != 0)
    {
        run_set_unit_matrix(shared.cu_tmp_icov, point_dim, rest_info.stream);
        run_store_icovariance_data(
            cu_inverses + pos * icov_size, cu_mfactors + pos, shared.cu_tmp_icov, 1, point_dim, rest_info.stream);
        return;
    }

    run_compute_store_icov_mf(cu_mfactors + pos, point_dim, cov, rest_info.stream);

    SOCH(cusolverDnSpotri(shared.cusolver_handle,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        (int)point_dim,
        cov,
        (int)point_dim,
        shared.cu_workspace,
        shared.workspace_size,
        shared.cu_info));

    run_store_icovariance_data(cu_inverses + pos * icov_size, nullptr, cov, 0, point_dim, rest_info.stream);
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
    vld->set_asgns(cu_point_asgns);

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
