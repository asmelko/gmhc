#include <mhc.hpp>

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

mhc_serial::assgn_t mhc_serial::nearest_cluster(const float* point, const std::vector<data_t>& centroids)
{
    auto minDist = centroids[0].distance(point);
    assgn_t nearest = 0;
    for (assgn_t i = 1; i < centroids.size(); ++i) {
        auto dist = centroids[i].distance(point);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

std::vector<mhc_serial::assgn_t> mhc_serial::kmeans()
{
    // Prepare for the first iteration
    std::vector<data_t> centroids;
    std::vector<assgn_t> assignments;

    std::vector<data_t> sums;
    std::vector<size_t> counts;

    centroids.resize(clusters_);
    assignments.resize(data_size_);
    
    sums.resize(clusters_);
    counts.resize(clusters_);

    for (size_t i = 0; i < clusters_; ++i)
        centroids[i].data.assign(data_ + i * data_dim_, data_ + (i + 1) * data_dim_);

    for (size_t i = 0; i < clusters_; ++i)
        sums[i].data.reserve(data_dim_);

    // Run the k-means refinements
    for (size_t k = 0; k < iters_; ++k)
    {
        // Prepare empty tmp fields.
        for (size_t i = 0; i < clusters_; ++i) 
        {
            sums[i].data.clear();
            counts[i] = 0;
        }
        
        for (std::size_t i = 0; i < data_size_ * data_dim_; i += data_dim_) 
        {
            auto nearest = nearest_cluster(data_ + i, centroids);
            assignments[i] = nearest;
            sums[nearest].add(data_ + i);
            ++counts[nearest];
        }

        for (std::size_t i = 0; i < data_size_; ++i) 
        {
            if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
            sums[i].divide(counts[i]);
            centroids[i].data = sums[i].data;
        }
    }

    return assignments;
}