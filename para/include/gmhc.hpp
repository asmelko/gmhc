#ifndef GMHC_HPP
#define GMHC_HPP

#include <apriori.hpp>

#define KERNEL_INFO kernel_info(6, 1024)

namespace clustering
{

struct shared_apriori_data_t
{
    chunk_t* cu_min;

    csize_t* cu_eucl_upd_size;
    csize_t* cu_maha_upd_size;

    float** cu_read_icov, **cu_write_icov;
    float* cu_tmp_icov;
    int* cu_info;
    int* cu_pivot;

    csize_t cluster_count;
    asgn_t id;

    static constexpr csize_t neighbors_size = 1;

    cublasHandle_t cublas_handle;
};

class gmhc : public hierarchical_clustering<float>
{
    float* cu_points_;
    float* cu_centroids_;
    float* cu_icov_;
    asgn_t* cu_point_asgns_;

    neighbor_t* cu_neighs_;
    neighbor_t* cu_tmp_neighs_;

    csize_t* cu_update_;
    cluster_data_t* cluster_data_;

    csize_t maha_threshold_;
    kernel_info starting_info_;

    shared_apriori_data_t common_;

    std::vector<clustering_context_t> apr_ctxs_;
    csize_t apriori_count_;

    cluster_data_t* apriori_cluster_data_;
    float* cu_apriori_centroids_;
    float* cu_apriori_icov_;

public:
    using res_t = pasgnd_t<float>;

    void initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t mahalanobis_threshold, const asgn_t* apriori_assignments = nullptr, validator* vld = nullptr);

    virtual std::vector<res_t> run() override;

    virtual void free() override;

private:
    void set_apriori(clustering_context_t& cluster, csize_t offset, csize_t size, validator* vld);
    void initialize_apriori(const asgn_t* apriori_assignments, validator* vld);

    void move_apriori(csize_t eucl_size, csize_t maha_size);
};

}

#endif