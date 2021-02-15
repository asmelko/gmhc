#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace clustering {

struct time_info
{
    bool active = false;
    bool was_cov;
    cudaEvent_t cov_start, cov_stop;
    cudaEvent_t nei_new_start, nei_new_stop;
    cudaEvent_t nei_rest_start, nei_rest_stop;

    std::vector<float> cov_time;
    std::vector<float> nei_time;

    void record(cudaEvent_t event, cudaStream_t stream)
    {
        if (!active)
            return;

        if (event == cov_start)
            was_cov = true;

        cudaEventRecord(event, stream);
    }

    void initialize()
    {
        cudaEventCreate(&cov_start);
        cudaEventCreate(&cov_stop);
        cudaEventCreate(&nei_new_start);
        cudaEventCreate(&nei_new_stop);
        cudaEventCreate(&nei_rest_start);
        cudaEventCreate(&nei_rest_stop);
        active = true;
        was_cov = false;
    }

    void record_iteration()
    {
        if (!active)
            return;

        if (was_cov)
        {
            cudaEventSynchronize(cov_stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, cov_start, cov_stop);
            cov_time.push_back(milliseconds);
            was_cov = false;
        }
        {
            cudaEventSynchronize(nei_new_stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, nei_new_start, nei_new_stop);
            nei_time.push_back(milliseconds);
        }
        {
            cudaEventSynchronize(nei_rest_stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, nei_rest_start, nei_rest_stop);
            nei_time.back() += milliseconds;
        }
    }

    void free()
    {
        cudaEventDestroy(cov_start);
        cudaEventDestroy(cov_stop);
        cudaEventDestroy(nei_new_start);
        cudaEventDestroy(nei_new_stop);
        cudaEventDestroy(nei_rest_start);
        cudaEventDestroy(nei_rest_stop);
        cov_time.clear();
        nei_time.clear();
    }
};

} // namespace clustering