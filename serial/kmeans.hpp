#ifndef KMEANS_H
#define KMEANS_H

#include <clustering.hpp>

#include <utility>
#include <vector>

namespace clustering {

class kmeans : public clustering_base<float>
{
    size_t clusters_ = 1024;
    size_t iters_ = 1000;
public:
    kmeans(size_t clusters, size_t iterations = 1000);

    std::pair<std::vector<asgn_t>,std::vector<float>> run();

private:
    float distance(const float* x, const float* y) const;

    asgn_t nearest_cluster(const float* point, const float* centroids) const;
};

}
#endif