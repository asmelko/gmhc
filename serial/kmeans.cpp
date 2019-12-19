#include "kmeans.hpp"
#include <cblas.h>

using namespace clustering;

kmeans::kmeans(size_t clusters, size_t iterations)
    : clusters_(clusters), iters_(iterations) {}

float kmeans::distance(const float* x, const float* y) const
{
    float res = 0;
	for (size_t i = 0; i < this->point_dim; ++i)
    {
        auto tmp = x[i] - y[i];
        res += tmp * tmp;
    }
	return res;
}

asgn_t kmeans::nearest_cluster(const float* point, const float* centroids) const
{
    auto minDist = distance(centroids, point);
    asgn_t nearest = 0;
    for (asgn_t i = 1; i < this->clusters_ * this->point_dim; i += (asgn_t)this->point_dim) {
        auto dist = distance(centroids + i, point);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

std::pair<std::vector<asgn_t>, std::vector<float>> kmeans::run()
{
    // Prepare for the first iteration
    std::vector<float> centroids;
    std::vector<asgn_t> assignments;

    std::vector<float> sums;
    std::vector<size_t> counts;

    centroids.resize(clusters_ * this->point_dim);
    assignments.resize(points_size);
    
    sums.resize(clusters_ * this->point_dim);
    counts.resize(clusters_);

    for (size_t i = 0; i < clusters_ * this->point_dim; i += this->point_dim)
        cblas_scopy((int)this->point_dim, points + i, 1, centroids.data() + i, 1);

    // Run the k-means refinements
    for (size_t k = 0; k < iters_; ++k)
    {
        // Prepare empty tmp fields.
        for (size_t i = 0; i < clusters_; ++i) 
        {
            cblas_sscal((int)this->point_dim, 0, sums.data() + i * this->point_dim, 1);
            counts[i] = 0;
        }
        
        for (std::size_t i = 0; i < this->points_size * this->point_dim; i += this->point_dim) 
        {
            auto nearest = nearest_cluster(points + i, centroids.data());
            assignments[i] = nearest;
            cblas_saxpy((int)this->point_dim, 1,this->points + i, 1, sums.data() + nearest * this->point_dim, 1);
            ++counts[nearest];
        }

        for (std::size_t i = 0; i < points_size; ++i) 
        {
            if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
            cblas_sscal((int)this->point_dim, 1 / (float)counts[i], sums.data() + i * this->point_dim, 1);
            cblas_scopy((int)this->point_dim, sums.data() + i * this->point_dim, 1, centroids.data() + i * point_dim, 1);
        }
    }

    return std::make_pair(std::move(assignments), std::move(centroids));
}
