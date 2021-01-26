#include <iostream>

#include "../para/include/gmhc.hpp"
#include "c_gmhclust_export.h"

using namespace clustering;

extern "C"
{
    C_GMHCLUST_EXPORT void c_gmhclust(const double* data_points,
        const int* data_points_size,
        const int* data_point_dim,
        const double* mahalanobis_threshold,
        const int* subthreshold_kind,  
        const bool* normalize,
        int* merging,
        double* heights)
    {
        gmhc gmhclust;

        auto size = (csize_t)*data_points_size;
        auto dim = (csize_t)*data_point_dim;
        auto threshold = (csize_t)(size * *mahalanobis_threshold);
        auto kind = (subthreshold_handling_kind)*subthreshold_kind;
        
        std::cout << "size " << *data_points_size << std::endl;
        std::cout << "dim " << *data_point_dim << std::endl;
        std::cout << "t " << *mahalanobis_threshold << std::endl;
        std::cout << "k " << *subthreshold_kind << std::endl;
        std::cout << "norm " << *normalize << std::endl;
        std::cout << "data[0] " << data_points[0] << std::endl;
        std::cout << "data[1] " << data_points[1] << std::endl;

        std::vector<float> data;
        data.resize(size * dim);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (float)data_points[i];

        gmhclust.initialize(data.data(), size, dim, threshold, kind, *normalize);

        auto res = gmhclust.run();

        for (size_t i = 0; i < res.size(); ++i)
        {
            merging[i] = res[i].first.first;
            merging[i + size - 1] = res[i].first.second;
            heights[i] = res[i].second;

            std::cout << res[i].first.first << " " << res[i].first.second << " " << res[i].second << std::endl;
        }

        gmhclust.free();
    }
}
