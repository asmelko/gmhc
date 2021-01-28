#ifndef APRIORI_HPP
#define APRIORI_HPP

#include "clustering.hpp"
#include "structures.hpp"

namespace clustering {

class validator;

enum class subthreshold_handling_kind
{
    MAHAL = 0,
    EUCLID = 1,
    MAHAL0 = 2,
    EUCLID_MAHAL = 3
};

struct shared_apriori_data_t;

// structure that describes a dataset to cluster
struct clustering_context_t
{
    subthreshold_handling_kind subthreshold_kind;

    // number of points
    csize_t point_size;
    // point dimension
    csize_t point_dim;
    // size of inverse covariance matrix
    csize_t icov_size;

    // current number of clusters
    csize_t cluster_count;
    // current number of clusters that reached maha_threshold
    csize_t maha_cluster_count;
    // Mahalanobis threshold
    csize_t maha_threshold;
    // indicates that all clusters reached maha_threshold
    bool switched_to_full_maha;

    // parameter for kernels
    kernel_info neighbor_info;
    kernel_info rest_info;

    // device neighbor array
    neighbor_t* cu_neighbors;
    // device intermediate neighbor array
    neighbor_t* cu_tmp_neighbors;
    // flag that states that neighbor array needs to be initialized
    bool initialize_neighbors;

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

    // helper structures for passing parameters
    centroid_data_t compute_data;
    update_data_t update_data;

    // shared data among each clustering context
    shared_apriori_data_t& shared;
    //flag that states that this is last/only apriori context left to cluster
    bool is_final;

    // mhca normalization flag
    bool normalize;

    // verification validator
    validator* vld;

public:
    clustering_context_t(shared_apriori_data_t& shared_data);

    // initializes the context
    void initialize(bool is_final, bool normalize);

    // performs context clustering
    std::vector<pasgnd_t<float>> run();

private:
    // initializes/updates neighbor array
    void compute_neighbors();
    // reorders data so the last array element is moved to the index pos
    void move_clusters(csize_t pos);
    // updates data for new cluster
    void update_iteration_device(asgn_t merged_A, asgn_t merged_B, asgn_t new_id);
    void update_iteration_host(chunk_t min);
    // computes inverse covariance matrix for new cluster
    void compute_icov(csize_t pos);
    //computes weight factor for new cluster
    float compute_weight_factor(csize_t pos);
    // computes covariance matrix for new cluster
    void compute_covariance(csize_t pos, float wf);

    bool need_recompute_neighbors();
    bool can_use_euclidean_distance();

    // verifies the iteration
    void verify(pasgn_t id_pair, float dist);
    // helper method that independently computes distance between clusters
    float recompute_dist(pasgn_t expected_id);
};

} // namespace clustering

#endif