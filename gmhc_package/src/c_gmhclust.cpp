#include <stack>

#include "../para/include/gmhc.hpp"
#include "c_gmhclust_export.h"

using namespace clustering;

extern "C"
{
    int transform_merging(csize_t m, csize_t n)
    {
        if (m < n)
            return -(int)(m + 1);
        else
            return m - n + 1;
    }

    void leaf_ordering(const int* merging, csize_t leaves, int* ordering)
    {
        int merge_pairs = leaves - 1;
        std::stack<int> s;

        int order = 0;
        s.push(merge_pairs);

        while (s.size())
        {
            auto cluster = s.top();
            s.pop();

            if (cluster < 0)
            {
                ordering[order++] = -cluster;
            }
            else
            {
                s.push(merging[cluster - 1 + merge_pairs]);
                s.push(merging[cluster - 1]);
            }
        }
    }

    C_GMHCLUST_EXPORT void c_gmhclust(const double* data_points,
        const int* data_points_size,
        const int* data_point_dim,
        const double* mahalanobis_threshold,
        const int* subthreshold_kind,
        const bool* normalize,
        int* merging,
        double* heights,
        int* ordering)
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

        bool ok = gmhclust.initialize(data.data(), size, dim, threshold, kind, *normalize);

        if (!ok)
            return;

        auto res = gmhclust.run();

        for (size_t i = 0; i < res.size(); ++i)
        {
            merging[i] = transform_merging(res[i].first.first, size);
            merging[i + size - 1] = transform_merging(res[i].first.second, size);
            heights[i] = res[i].second;
        }

        leaf_ordering(merging, size, ordering);

        gmhclust.free();
    }
}
