#ifndef MHC_H
#define MHC_H

#include "kmeans.hpp"

namespace clustering {

class smhc : public hierarchical_clustering<float>
{
    kmeans kmeans_;
public:
    smhc();

    virtual void initialize(const float* data_points, size_t data_points_size, size_t data_point_dim) override;

    virtual const asgn_t* iterate() override;
};

}

#endif