#ifndef GMHC_HPP
#define GMHC_HPP

#include <clustering.hpp>
#include <cublas_v2.h>

namespace clustering
{

struct chunk_t
{
    float min_dist;
    asgn_t min_i, min_j;
};

struct centroid_data_t
{
    asgn_t id;
    float* icov;
};

class gmhc : public hierarchical_clustering<float>
{
    float* cu_points_;
    asgn_t* cu_point_asgns_;
    float* cu_centroids_;
    centroid_data_t* cu_centroid_asgns_;
    chunk_t* cu_chunks_;
    chunk_t* cu_min_;

    size_t chunk_count_;
    size_t cluster_count_;
    asgn_t id_;

    std::vector<size_t> centroid_sizes_;
    
    static constexpr size_t maha_threshold_ = 20;
    size_t icov_size_;

    cublasHandle_t handle_;

public:
    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;
};

}

#endif