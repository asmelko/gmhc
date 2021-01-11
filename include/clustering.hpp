#ifndef MHCA_H
#define MHCA_H

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace clustering {

// typename for 32bit unsigned int used among the project
using csize_t = uint32_t;
// typename for assignments
using asgn_t = csize_t;
// typename for assignments pair
using pasgn_t = std::pair<asgn_t, asgn_t>;
// typename for pair of pasgn_t and a distance type
template<typename T>
using pasgnd_t = std::pair<pasgn_t, T>;

// base for hierarchical clustering algorithms
template<typename T>
class hierarchical_clustering
{
protected:
    // array of points from dataset
    const T* points;
    // number of points in the array
    csize_t points_size;
    // dimensionality of a point
    csize_t point_dim;

public:
    // method initializes the hierarchical clustering algorithm by assigning the data fields
    virtual void initialize(const T* data_points, csize_t data_points_size, csize_t data_point_dim)
    {
        this->points = data_points;
        this->points_size = data_points_size;
        this->point_dim = data_point_dim;
    }

    // method that starts the clustering
    // returns vector of pasgnd_t<T> - vector of merged clusters and distance between them
    virtual std::vector<pasgnd_t<T>> run() = 0;

    // method that frees the resources
    virtual void free() = 0;
};

} // namespace clustering

#endif