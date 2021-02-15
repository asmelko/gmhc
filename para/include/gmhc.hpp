#ifndef GMHC_HPP
#define GMHC_HPP

#include <cusolverDn.h>

#include "apriori.hpp"

namespace clustering {

// data to share among each clustering context
struct shared_apriori_data_t
{
    // device variable to retrieve minimum
    chunk_t* cu_min;

    // device variables for update kernel
    csize_t* cu_upd_size;

    // device variables to compute inverse matrix
    float* cu_tmp_icov;
    int* cu_info;
    float* cu_workspace;
    int workspace_size;

    // reduce arrays for centroid and covariance kernel
    float* cu_work_centroid;
    float* cu_work_covariance;

    // next available id
    asgn_t id;

    csize_t* cu_asgn_idxs_;
    csize_t* cu_idxs_size_;

    // handle to CUSOLVER library
    cusolverDnHandle_t cusolver_handle;

    cudaStream_t streams[2];
};

// Mahalanobis hierarchical clustering class
template <csize_t Neighs>
class gmhc : public hierarchical_clustering<float>
{
    subthreshold_handling_kind subthreshold_kind_;

    // device point array
    float* cu_points_;
    // device centroid array
    float* cu_centroids_;
    // device inverse covariance matrix array
    float* cu_icov_;
    // device multiplication factor of inverse covariance matrix array
    float* cu_icov_mf_;
    // device assignments array
    asgn_t* cu_point_asgns_;
    // mhca normalization flag
    bool normalize_;

    // device neighbor array
    neighbor_t* cu_neighs_;
    // device intermediate neighbor array
    neighbor_t* cu_tmp_neighs_;

    // device update array
    csize_t* cu_update_;
    // host status array
    cluster_data_t* cluster_data_;

    // Mahalanobis threshold
    csize_t maha_threshold_;
    // parameter for kernels
    kernel_info starting_info_;

    // shared data for contexts
    shared_apriori_data_t common_;

    // context array
    std::vector<clustering_context_t<Neighs>> apr_ctxs_;
    // number of apriori clusters
    csize_t apriori_count_;

public:
    bool initialize(const float* data_points,
        csize_t data_points_size,
        csize_t data_point_dim,
        csize_t mahalanobis_threshold,
        subthreshold_handling_kind subthreshold_kind = subthreshold_handling_kind::MAHAL,
        bool normalize = false,
        const asgn_t* apriori_assignments = nullptr,
        validator* vld = nullptr);

    virtual std::vector<pasgnd_t<float>> run() override;

    virtual void free() override;

private:
    // method sets fields of apriori clusters
    void set_apriori(clustering_context_t<Neighs>& cluster, csize_t offset, csize_t size, validator* vld);
    // method that reads apriori assigmnets array and initializes all apriori clusters
    void initialize_apriori(const asgn_t* apriori_assignments, validator* vld);
    // method that creates final context
    void move_apriori();
};

template class gmhc<1>;
template class gmhc<2>;
template class gmhc<3>;
template class gmhc<4>;
template class gmhc<5>;

} // namespace clustering

#endif