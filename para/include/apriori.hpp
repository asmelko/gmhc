#ifndef APRIORI_HPP
#define APRIORI_HPP

#include <cublas_v2.h>

#include <clustering.hpp>
#include <validator.hpp>
#include <structures.hpp>

namespace clustering
{

struct shared_apriori_data_t;

struct clustering_context_t
{
    csize_t point_size;
    csize_t point_dim;
    csize_t icov_size;

    csize_t cluster_count;
    csize_t maha_threshold;

    kernel_info starting_info;

    neighbor_t* cu_neighbors;
    neighbor_t* cu_tmp_neighbors;

    float* cu_points;
    float* cu_centroids;
    float* cu_inverses;

    asgn_t* cu_point_asgns;

    csize_t* cu_updates;
    cluster_data_t* cluster_data;

    cluster_bound_t bounds;
    centroid_data_t compute_data;
    update_data_t update_data;

    shared_apriori_data_t& shared;

    validator* vld;

public:
    clustering_context_t(shared_apriori_data_t& shared_data);
    void initialize();

    pasgnd_t<float> iterate();
private:
    bool remove(csize_t idx);
    void move_clusters(csize_t i, csize_t j, bool maha);
    void update_iteration(const cluster_data_t* merged);
    void compute_icov(csize_t pos);

    void verify(pasgn_t id_pair, float dist);
    float recompute_dist(pasgn_t expected_id);
};

}

#endif