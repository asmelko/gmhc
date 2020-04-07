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
    using neighbour_type = neighbour_array_t<1>;

    float* cu_points_;
    asgn_t* cu_point_asgns_;
    float* cu_centroids_;
    float** cu_icov_;
    cluster_kind* cu_cluster_kinds_;
    neighbour_type* cu_neighs_;
    chunk_t* cu_chunks_;
    chunk_t* cu_min_;
    uint8_t* cu_updated;

    kernel_info starting_info_;

    neighbour_type* tmp_neigh;
    float* tmp_icov;

    float** cu_read_icov, ** cu_write_icov;
    int* cu_info;
    int* cu_pivot;

    size_t chunk_count_;
    size_t cluster_count_;
    size_t big_cluster_count_, small_cluster_count_;
    asgn_t id_;

    cluster_data_t* cluster_data_;
    
    size_t maha_threshold_;
    size_t icov_size_;

    cublasHandle_t handle_;

    validator* vld_;

public:
    void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim, size_t mahalanobis_threshold, validator* vld = nullptr);

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;

protected:
    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;


private:
    void update_iteration(size_t cluster_idx, const cluster_data_t* merged);
    void move_clusters(size_t old_pos);
    void compute_icov(size_t pos, bool have_inplace_icov);
};

}

#endif