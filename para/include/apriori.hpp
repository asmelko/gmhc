#ifndef APRIORI_HPP
#define APRIORI_HPP

#include "clustering.hpp"
#include "structures.hpp"
#include "validator.hpp"

namespace clustering {

struct shared_apriori_data_t;

// structure that describes a dataset to cluster
struct clustering_context_t
{
    // number of points
    csize_t point_size;
    // point dimension
    csize_t point_dim;
    // size of inverse covariance matrix
    csize_t icov_size;

    // current number of clusters
    csize_t cluster_count;
    // Mahalanobis threshold
    csize_t maha_threshold;

    // parameter for kernels
    kernel_info starting_info;

    // device neighbor array
    neighbor_t* cu_neighbors;
    // device intermediate neighbor array
    neighbor_t* cu_tmp_neighbors;

    // device point array
    float* cu_points;
    // device centroid array
    float* cu_centroids;
    // device inverse array
    float* cu_inverses;
    // device multiplication factor array
    float* cu_mfactors;

    // device assignments array
    asgn_t* cu_point_asgns;

    // device update array
    csize_t* cu_updates;
    // host status array
    cluster_data_t* cluster_data;

    // host indexing structure
    cluster_bound_t bounds;

    // helper structures for passing parameters
    centroid_data_t compute_data;
    update_data_t update_data;

    // shared data among each clustering context
    shared_apriori_data_t& shared;

    // verification validator
    validator* vld;

public:
    clustering_context_t(shared_apriori_data_t& shared_data);

    // initializes the context
    void initialize();

    // performs one iteration of clustering
    pasgnd_t<float> iterate();

private:
    // removes cluster at idx
    bool remove(csize_t idx);
    // reorders data according to the merged clusters i and j
    void move_clusters(csize_t i, csize_t j, bool maha);
    // updates data for new cluster
    void update_iteration(const cluster_data_t* merged);
    // coputes inverse covariance matrix for new cluster
    void compute_icov(csize_t pos);

    // verifies the iteration
    void verify(pasgn_t id_pair, float dist);
    // helper method that independently computes distance between clusters
    float recompute_dist(pasgn_t expected_id);
};

} // namespace clustering

#endif