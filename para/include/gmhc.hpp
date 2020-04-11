#ifndef GMHC_HPP
#define GMHC_HPP

#include <cublas_v2.h>

#include <clustering.hpp>
#include <validator.hpp>
#include <common_structures.hpp>

namespace clustering
{

struct cluster_data_t
{
    asgn_t id;
    size_t size;
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
    uint8_t* cu_update_;

    static constexpr size_t neigh_number_ = 2;

    kernel_info starting_info_;

    neighbour_t* tmp_neigh;
    float* tmp_icov;

    float** cu_read_icov, ** cu_write_icov;
    int* cu_info;
    int* cu_pivot;

    size_t cluster_count_;
    asgn_t id_;

    size_t icov_size;

    cluster_data_t* cluster_data_;
    
    size_t maha_threshold_;

    cluster_bound_t bounds_;
    centroid_data_t compute_data_;
    update_data_t upd_data_;

    cublasHandle_t handle_;

    validator* vld_;

public:
    void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim, size_t mahalanobis_threshold, validator* vld = nullptr);

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;

protected:
    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;


private:
    void update_iteration(const cluster_data_t* merged);
    void gmhc::move_clusters(size_t i, size_t j, bool maha);
    bool hole(size_t idx);
    void compute_icov(size_t pos);

    void verify(pasgn_t id_pair, float dist);
};

}

#endif