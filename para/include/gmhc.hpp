#ifndef GMHC_HPP
#define GMHC_HPP

#include <cublas_v2.h>

#include <clustering.hpp>
#include <validator.hpp>
#include <structures.hpp>

namespace clustering
{

struct cluster_data_t
{
    asgn_t id;
    csize_t size;
};

struct apriori_context_t
{
    neighbour_t* cu_neighbours;
    neighbour_t* cu_tmp_neighbours;
    csize_t* cu_updates;
    float* cu_centroids;
    float* cu_inverses;
    float* cu_points;
    asgn_t* cu_point_asgns;
    cluster_data_t* clusters;
    csize_t point_size;

    cluster_bound_t bounds;
};

class gmhc : public hierarchical_clustering<float>
{
    float* cu_points_;
    asgn_t* cu_point_asgns_;
    float* cu_centroids_;
    float* cu_icov_;
    neighbour_t* cu_neighs_;
    chunk_t* cu_chunks_;
    chunk_t* cu_min_;

    csize_t* cu_update_;
    csize_t* cu_eucl_upd_size_;
    csize_t* cu_maha_upd_size_;

    static constexpr csize_t neigh_number_ = 2;

    kernel_info starting_info_;

    neighbour_t* tmp_neigh;
    float* tmp_icov;

    float** cu_read_icov, ** cu_write_icov;
    int* cu_info;
    int* cu_pivot;

    csize_t cluster_count_;
    asgn_t id_;

    csize_t icov_size;

    cluster_data_t* cluster_data_;
    
    csize_t maha_threshold_;

    std::vector<apriori_context_t> apr_ctxs_;
    apriori_context_t default_apr_;
    csize_t apriori_count_;

    cluster_bound_t bounds_;
    centroid_data_t compute_data_;
    update_data_t upd_data_;

    cublasHandle_t handle_;

    validator* vld_;

public:
    void initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t mahalanobis_threshold, const asgn_t* apriori_assignments = nullptr, validator* vld = nullptr);

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;

protected:
    virtual void initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim) override;


private:
    void update_iteration(const cluster_data_t* merged, apriori_context_t& ctx);
    void move_clusters(csize_t i, csize_t j, bool maha, apriori_context_t& ctx);
    bool remove(csize_t idx, apriori_context_t& ctx);
    void compute_icov(csize_t pos, apriori_context_t& ctx);

    void initialize_apriori(const asgn_t* apriori_assignments);
    pasgn_t run(apriori_context_t& ctx);
    void move_apriori();

    void verify(pasgn_t id_pair, float dist, apriori_context_t& ctx);
};

}

#endif