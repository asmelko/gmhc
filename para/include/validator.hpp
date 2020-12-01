#ifndef VALIDATOR_HPP
#define VALIDATOR_HPP

#include <functional>
#include <tuple>

#include "clustering.hpp"

namespace clustering {

// class that validates mahalanobis herarchical clustering process
class validator
{
    using recompute_f = std::function<float(pasgn_t)>;

    // strucure describing a cluster
    struct cluster
    {
        std::vector<float> centroid;
        std::vector<float> icov;
        asgn_t id;
    };

    const float* points_;
    csize_t point_count_;
    csize_t point_dim_;
    csize_t maha_threshold_;

    asgn_t id_;
    csize_t cluster_count_;
    std::vector<asgn_t> point_asgns_;
    std::vector<cluster> clusters_;
    std::vector<csize_t> apr_sizes_;
    csize_t apr_idx_;

    // indicates that validation has found error
    bool error_;
    // current iteration number starting from 0
    csize_t iteration_;

public:
    void initialize(const float* data_points,
        csize_t data_points_size,
        csize_t data_point_dim,
        csize_t maha_threshold,
        const asgn_t* apriori_assignments = nullptr);

    // variables to set from ouside
    // matrices to verify
    std::vector<float> cov_v, icov_v;

    // verify one iteration
    bool verify(pasgn_t pair_v, float dist_v, const float* centroid_v, recompute_f recompute);

    // returns true if validator found a difference
    bool has_error() const;

    // adjusted diff of floats
    // respect slight differences of CPU and GPU floating point computation
    static bool float_diff(float a, float b, float d = 0.05f);
    static bool float_diff(const float* a, const float* b, csize_t size, float d = 0.05f);

private:
    // iterate its own mahalanobis clustering
    std::tuple<pasgn_t, csize_t, float, csize_t> iterate(const pasgnd_t<float>& expected, recompute_f recompute);
    // retrieves minimal cluster pair
    void get_min(const pasgn_t& expected,
        pasgn_t& min_pair,
        std::pair<csize_t, csize_t>& min_idx,
        std::pair<csize_t, csize_t>& expected_idx,
        float& expected_dist,
        float& min_dist);

    void create_clusters(const asgn_t* apriori_assignments);
};

} // namespace clustering

#endif