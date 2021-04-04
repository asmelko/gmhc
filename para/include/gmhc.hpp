#ifndef GMHC_HPP
#define GMHC_HPP

#include <cusolverDn.h>

#include "../para_timer/timer.hpp"
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
    float* cu_tmp_points;
    int* cu_info;
    float* cu_workspace;
    int workspace_size;

    //reduce arrays for centroid and covariance kernel
    float* cu_work_centroid;
    float* cu_work_covariance;

    // next available id
    asgn_t id;

    csize_t* cu_asgn_idxs_;
    csize_t* cu_idxs_size_;

    // number of closest neighbors for each cluster
    static constexpr csize_t neighbors_size = 1;

    // handle to CUSOLVER and CUBLAS library
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;

    cudaStream_t streams[2];

    time_info timer;
};

// Mahalanobis hierarchical clustering class
class gmhc : public hierarchical_clustering<float>
{
    subthreshold_handling_kind subthreshold_kind_;

    // device centroid array
    float* cu_centroids_;
    // device inverse covariance matrix array
    float* cu_icov_;
    // device multiplication factor of inverse covariance matrix array
    float* cu_icov_mf_;
    // mhca normalization flag
    bool normalize_;
    // mhca FMD/CMD selection flag
    bool quick_;

    // device array of cluster representants
    cluster_representants_t* cu_representants_;

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
    std::vector<clustering_context_t> apr_ctxs_;
    // number of apriori clusters
    csize_t apriori_count_;

public:
    // result type for the run method
    using res_t = pasgnd_t<float>;

    bool initialize(const float* data_points,
        csize_t data_points_size,
        csize_t data_point_dim,
        csize_t mahalanobis_threshold,
        subthreshold_handling_kind subthreshold_kind = subthreshold_handling_kind::MAHAL, 
        bool normalize = false,
        bool quick = true,
        const asgn_t* apriori_assignments = nullptr,
        validator* vld = nullptr);

    virtual std::vector<res_t> run() override;

    virtual void free() override;

    time_info& timer();

private:
    // method sets fields of apriori clusters
    void set_apriori(clustering_context_t& cluster, csize_t offset, csize_t size, validator* vld);
    // method that reads apriori assigmnets array and initializes all apriori clusters
    void initialize_apriori(const asgn_t* apriori_assignments, validator* vld);
    // method that creates final context
    void move_apriori();
};

} // namespace clustering

#endif