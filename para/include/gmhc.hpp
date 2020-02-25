#ifndef GMHC_HPP
#define GMHC_HPP

#include <clustering.hpp>

namespace clustering
{

class gmhc : public hierarchical_clustering<float>
{
    float* cu_points_;
    asgn_t* cu_point_asgns_;
    float* cu_centroids_;
    asgn_t* cu_centroid_asgns_;



public:
    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

    virtual std::vector<pasgn_t> run() override;

    virtual void free() override;
};

}

#endif