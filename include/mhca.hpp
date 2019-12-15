#ifndef MHCA_H
#define MHCA_H

#include <cstdint>

namespace clustering {

using asgn_t = uint32_t;

template <typename T>
class hierarchical_clustering
{
protected:
    const T* points;
    size_t points_size;
    size_t point_dim;

public:
    virtual void initialize(const T* points, size_t points_size, size_t point_dim)
    {
        this->points = points;
        this->points_size = points_size;
        this->point_dim = dim;
    }

    virtual const asgn_t* iterate() = 0;

    virtual void free() = 0;
};

}

#endif