#include "mhc.hpp"

#include <cblas.h>
#include <lapacke.h>

using namespace clustering;

void data_t::add(const float* point)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] += point[i];
}

void data_t::divide(size_t count)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] /= count;
}

float data_t::distance(const float* point) const
{
    float res = 0;
    for (size_t i = 0; i < data.size(); ++i)
        res += (data[i] - point[i]) * (data[i] - point[i]);
    return res;
}

asgn_t mhc_serial::nearest_cluster(const float* point, const std::vector<data_t>& centroids)
{
    auto minDist = centroids[0].distance(point);
    asgn_t nearest = 0;
    for (asgn_t i = 1; i < centroids.size(); ++i) {
        auto dist = centroids[i].distance(point);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

std::pair<std::vector<asgn_t>,std::vector<data_t>> mhc_serial::kmeans()
{
    // Prepare for the first iteration
    std::vector<data_t> centroids;
    std::vector<asgn_t> assignments;

    std::vector<data_t> sums;
    std::vector<size_t> counts;

    centroids.resize(clusters_);
    assignments.resize(points_size);
    
    sums.resize(clusters_);
    counts.resize(clusters_);

    for (size_t i = 0; i < clusters_; ++i)
        centroids[i].data.assign(points + i * point_dim, points + (i + 1) * point_dim);

    for (size_t i = 0; i < clusters_; ++i)
        sums[i].data.reserve(point_dim);

    // Run the k-means refinements
    for (size_t k = 0; k < iters_; ++k)
    {
        // Prepare empty tmp fields.
        for (size_t i = 0; i < clusters_; ++i) 
        {
            sums[i].data.clear();
            counts[i] = 0;
        }
        
        for (std::size_t i = 0; i < points_size * point_dim; i += point_dim) 
        {
            auto nearest = nearest_cluster(points + i, centroids);
            assignments[i] = nearest;
            sums[nearest].add(points + i);
            ++counts[nearest];
        }

        for (std::size_t i = 0; i < points_size; ++i) 
        {
            if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
            sums[i].divide(counts[i]);
            centroids[i].data = sums[i].data;
        }
    }

    return std::make_pair(std::move(assignments), std::move(centroids));
}

const asgn_t* mhc_serial::iterate()
{
    auto [assignments, cetroids] = kmeans();

    

    return nullptr;
}