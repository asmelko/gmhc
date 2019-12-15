#ifndef MHC_H
#define MHC_H

#include <stddef.h>
#include <vector>

#include <mhca.hpp>

namespace clustering {

struct data_t
{
    std::vector<float> data;

    void add(const float* point);

    float distance(const float* point) const;

    void divide(size_t count);
};


class mhc_serial : public hierarchical_clustering<float>
{
private:
    static constexpr size_t clusters_ = 1024;
    static constexpr size_t iters_ = 1000;

    static asgn_t nearest_cluster(const float* point, const std::vector<data_t>& centroids);

    std::vector<asgn_t> kmeans();
};

}

#endif