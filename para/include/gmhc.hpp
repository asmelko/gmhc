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

struct cluster_data_t
{
    asgn_t id;
    size_t size;
};

struct neighbour_t
{
    float distance;
    clustering::asgn_t idx;
};

template <size_t N>
struct neighbour_array_t
{
    neighbour_t neighbours[N];
};

class gmhc : public hierarchical_clustering<float>
{
    float* cu_points_;
    asgn_t* cu_point_asgns_;
    float* cu_centroids_, *cu_centroids_tmp_;
    float** cu_icov_, **cu_icov_tmp_;
    neighbour_array_t<1>* cu_neighs_, *cu_tmp_neighs_;
    chunk_t* cu_chunks_;
    chunk_t* cu_min_;

    size_t chunk_count_;
    size_t cluster_count_;
    size_t big_cluster_count_, small_cluster_count_;
    asgn_t id_;

    cluster_data_t* cluster_data_, *cluster_data_tmp_;
    
    static constexpr size_t maha_threshold_ = 20;
    size_t icov_size_;

    cublasHandle_t handle_;

public:
    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;

private:
    void move_(size_t from, size_t to, int where);
    void move_clusters(size_t i, size_t j);
};

}

#endif