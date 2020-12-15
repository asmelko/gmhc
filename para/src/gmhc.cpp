#include "gmhc.hpp"

#include <cassert>
#include <iostream>
#include <map>

#include "kernels.cuh"

using namespace clustering;

bool gmhc::initialize(const float* data_points,
    csize_t data_points_size,
    csize_t data_point_dim,
    csize_t mahalanobis_threshold,
    subthreshold_handling_kind subthreshold_kind,
    const asgn_t* apriori_assignments,
    validator* vld)
{
    if (data_point_dim > MAX_DIM)
    {
        std::cerr << "currently allowed maximum dimension is 50" << std::endl;
        return false;
    }
    else if (data_point_dim < 1)
    {
        std::cerr << "dimension should be at least 2" << std::endl;
        return false;
    }

    hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

    subthreshold_kind_ = subthreshold_kind;

    common_.id = (asgn_t)data_points_size;
    common_.cluster_count = data_points_size;
    csize_t icov_size = (point_dim + 1) * point_dim / 2;

    maha_threshold_ = mahalanobis_threshold;
    starting_info_ = kernel_info(6, 512);

    CUCH(cudaSetDevice(0));

    CUCH(cudaMalloc(&cu_points_, data_points_size * data_point_dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_centroids_, data_points_size * data_point_dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_point_asgns_, data_points_size * sizeof(asgn_t)));
    CUCH(cudaMalloc(&cu_neighs_, sizeof(neighbor_t) * common_.neighbors_size * data_points_size));
    CUCH(cudaMalloc(
        &cu_tmp_neighs_, sizeof(neighbor_t) * common_.neighbors_size * data_points_size * starting_info_.grid_dim));
    CUCH(cudaMalloc(&cu_icov_, sizeof(float) * icov_size * data_points_size));
    CUCH(cudaMalloc(&cu_update_, common_.cluster_count * sizeof(csize_t)));
    cluster_data_ = new cluster_data_t[common_.cluster_count];

    CUCH(cudaMalloc(&common_.cu_min, sizeof(chunk_t)));
    CUCH(cudaMalloc(&common_.cu_tmp_icov, 2 * data_point_dim * data_point_dim * sizeof(float)));
    CUCH(cudaMalloc(&common_.cu_upd_size, sizeof(csize_t)));
    CUCH(cudaMalloc(&common_.cu_read_icov, sizeof(float*)));
    CUCH(cudaMalloc(&common_.cu_write_icov, sizeof(float*)));
    CUCH(cudaMalloc(&common_.cu_info, sizeof(int)));
    CUCH(cudaMalloc(&common_.cu_pivot, sizeof(int) * data_point_dim));
    SOCH(cusolverDnCreate(&common_.cusolver_handle));

    int workspace_size_f, workspace_size_i;
    SOCH(cusolverDnSpotrf_bufferSize(common_.cusolver_handle,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        point_dim,
        nullptr,
        point_dim,
        &workspace_size_f));
    SOCH(cusolverDnSpotri_bufferSize(common_.cusolver_handle,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        point_dim,
        nullptr,
        point_dim,
        &workspace_size_i));
    CUCH(cudaDeviceSynchronize());

    common_.workspace_size = std::max(workspace_size_f, workspace_size_i);
    CUCH(cudaMalloc(&common_.cu_workspace, sizeof(float) * common_.workspace_size));

    run_set_default_icovs(cu_icov_, data_points_size, data_point_dim, starting_info_);
    run_set_default_icov_mfs(cu_icov_, data_points_size, starting_info_);

    if (apriori_assignments)
        initialize_apriori(apriori_assignments, vld);
    else
    {
        CUCH(cudaMemcpy(cu_points_,
            data_points,
            data_points_size * data_point_dim * sizeof(float),
            cudaMemcpyKind::cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_centroids_,
            data_points,
            data_points_size * data_point_dim * sizeof(float),
            cudaMemcpyKind::cudaMemcpyHostToDevice));
        run_set_default_asgn(cu_point_asgns_, data_points_size);

        for (asgn_t i = 0; i < common_.cluster_count; ++i)
            cluster_data_[i] = cluster_data_t { i, 1 };

        apr_ctxs_.emplace_back(common_);

        set_apriori(apr_ctxs_.front(), 0, points_size, vld);
        apriori_count_ = 0;
    }

    return true;
}

void gmhc::set_apriori(clustering_context_t& cluster, csize_t offset, csize_t size, validator* vld)
{
    csize_t icov_size = (point_dim + 1) * point_dim / 2;

    cluster.subthreshold_kind = subthreshold_kind_;

    cluster.point_size = size;
    cluster.point_dim = point_dim;
    cluster.icov_size = icov_size;

    cluster.cluster_count = size;
    cluster.maha_threshold = maha_threshold_;

    cluster.starting_info = starting_info_;

    cluster.cu_neighbors = cu_neighs_ + offset * common_.neighbors_size;
    cluster.cu_tmp_neighbors = cu_tmp_neighs_ + offset * common_.neighbors_size * starting_info_.grid_dim;

    cluster.cu_points = cu_points_ + offset * point_dim;
    cluster.cu_centroids = cu_centroids_ + offset * point_dim;
    cluster.cu_inverses = cu_icov_ + offset * icov_size;
    cluster.cu_mfactors = cu_icov_mf_ + offset;

    cluster.cu_point_asgns = cu_point_asgns_ + offset;

    cluster.cu_updates = cu_update_ + offset;
    cluster.cluster_data = cluster_data_ + offset;

    cluster.vld = vld;

    cluster.initialize();
}

void gmhc::initialize_apriori(const asgn_t* apriori_assignments, validator* vld)
{
    // get apriori sizes
    std::map<asgn_t, csize_t> counts;
    for (csize_t i = 0; i < points_size; ++i)
    {
        auto count = counts.find(apriori_assignments[i]);

        if (count != counts.end())
            count->second++;
        else
            counts.emplace(apriori_assignments[i], 1);
    }

    // create indexing structures
    std::vector<csize_t> indices, sizes;
    std::map<asgn_t, csize_t> order;
    csize_t idx = 0;
    csize_t tmp_sum = 0;
    for (auto& count : counts)
    {
        order.emplace(count.first, idx++);
        indices.push_back(tmp_sum);
        sizes.push_back(count.second);
        tmp_sum += count.second;
    }

    // sorting data for apriori clusters
    for (csize_t i = 0; i < points_size; ++i)
    {
        asgn_t apriori_asgn = apriori_assignments[i];
        auto next = indices[order[apriori_asgn]]++;
        CUCH(cudaMemcpy(cu_points_ + next * point_dim,
            points + i * point_dim,
            point_dim * sizeof(float),
            cudaMemcpyKind::cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_point_asgns_ + next, &i, sizeof(asgn_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
        cluster_data_[next] = cluster_data_t { i, 1 };
    }
    CUCH(cudaMemcpy(
        cu_centroids_, cu_points_, points_size * point_dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    // initialize apriori
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        clustering_context_t cluster(common_);
        auto offset = i == 0 ? 0 : indices[i - 1];

        set_apriori(cluster, offset, sizes[i], vld);

        apr_ctxs_.emplace_back(std::move(cluster));
    }
    apriori_count_ = (csize_t)sizes.size();
}

void gmhc::move_apriori()
{
    csize_t icov_size = (point_dim + 1) * point_dim / 2;

    for (size_t i = 1; i < apriori_count_; ++i)
    {
        const auto& ctx = apr_ctxs_[i];

        cluster_data_[i] = ctx.cluster_data[0];

        CUCH(cudaMemcpy(cu_centroids_ + i * point_dim,
            ctx.cu_centroids,
            point_dim * sizeof(float),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice));

        CUCH(cudaMemcpy(cu_icov_ + i * icov_size,
            ctx.cu_inverses,
            icov_size * sizeof(float),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice));

        apr_ctxs_.front().cluster_count += apr_ctxs_[i].cluster_count;
        apr_ctxs_.front().maha_cluster_count += apr_ctxs_[i].maha_cluster_count;
    }
    apr_ctxs_.front().point_size = points_size;
}

std::vector<gmhc::res_t> gmhc::run()
{
    std::vector<res_t> ret;
    clustering_context_t& final = apr_ctxs_.front();

    // compute apriori
    if (apriori_count_)
    {
        for (auto& ctx : apr_ctxs_)
        {
            while (ctx.cluster_count > 1)
            {
                auto tmp = ctx.iterate();
                ret.push_back(tmp);
            }
        }

        move_apriori();
    }

    // compute rest
    while (final.cluster_count > 1)
    {
        auto tmp = final.iterate();
        ret.push_back(tmp);
    }

    return ret;
}

void gmhc::free()
{
    CUCH(cudaFree(cu_points_));
    CUCH(cudaFree(cu_centroids_));
    CUCH(cudaFree(cu_point_asgns_));
    CUCH(cudaFree(cu_neighs_));
    CUCH(cudaFree(cu_tmp_neighs_));
    CUCH(cudaFree(cu_icov_));
    CUCH(cudaFree(cu_update_));

    delete[] cluster_data_;

    CUCH(cudaFree(common_.cu_min));
    CUCH(cudaFree(common_.cu_tmp_icov));
    CUCH(cudaFree(common_.cu_upd_size));
    CUCH(cudaFree(common_.cu_read_icov));
    CUCH(cudaFree(common_.cu_write_icov));
    CUCH(cudaFree(common_.cu_info));
    CUCH(cudaFree(common_.cu_pivot));
    SOCH(cusolverDnDestroy(common_.cusolver_handle));
}