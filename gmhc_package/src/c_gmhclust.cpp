#include "../para/include/gmhc.hpp"
#include "c_gmhclust_export.h"

using namespace clustering;

extern "C"
{
    csize_t transform_merging(csize_t m, csize_t n)
    {
        if (m < n)
            return -(m + 1);
        else
            return m - n + 1;
    }

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

        std::vector<float> data;
        data.resize(size * dim);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (float)data_points[i];

        gmhclust.initialize(data.data(), size, dim, threshold, kind, *normalize);

        auto res = gmhclust.run();

        for (size_t i = 0; i < res.size(); ++i)
        {
            merging[i] = transform_merging(res[i].first.first, size);
            merging[i + size - 1] = transform_merging(res[i].first.second, size);
            heights[i] = res[i].second;
        }

        gmhclust.free();
    }
}
