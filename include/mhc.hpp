#ifndef MHC_H
#define MHC_H

#include <stddef.h>
#include <vector>

struct data_t
{
    std::vector<float> data;

    void add(const float* point);

    float distance(const float* point) const;

    void divide(size_t count);
};


class mhc_serial
{
    const float* data_;
    size_t data_size_;
    size_t data_dim_;
public:
    using assgn_t = std::uint8_t;

    mhc_serial(const float* data, size_t size, size_t dim);
    
private:
    static constexpr size_t clusters_ = 1024;
    static constexpr size_t iters_ = 1000;

    static assgn_t nearest_cluster(const float* point, const std::vector<data_t>& centroids);

    std::vector<assgn_t> kmeans();
};

#endif