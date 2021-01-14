#include <chrono>

#include <gmock/gmock.h>

#include "reader.hpp"
#include "serial_impl.hpp"

using namespace clustering;

// tests for centroid kernel

TEST(kernel, centroid_small)
{
    std::string input =
        "3 32 -1.29547 8.00796 -7.49481 0.873838 0.59632 -1.51012 2.29091 0.944923 1.97487 -11.8492 0.220984 -3.32246 "
        "-3.18128 -0.856242 8.65632 -3.82426 3.51576 0.726751 0.468351 6.02061 1.30511 -3.4598 8.40714 -3.51013 "
        "0.875993 1.37086 -4.31532 5.93115 6.5827 3.40516 4.71282 4.14184 -2.94859 1.04552 4.30228 1.4721 -2.33436 "
        "-8.34392 -0.658711 7.88312 -3.95856 -4.61158 -5.23649 -2.20981 4.77685 -0.16196 0.051402 4.59879 -10.0206 "
        "0.902125 -0.0203374 1.64312 -2.10842 -1.20475 -12.092 0.0737986 2.18845 -1.17889 -11.8277 9.89593 3.83565 "
        "3.71229 -0.498549 0.436246 1.65099 4.69318 4.50654 4.72271 -3.79102 -5.22239 0.974885 -10.9422 5.59455 "
        "1.47331 5.72225 -1.47315 -2.90234 -7.91345 -9.92892 11.5448 -4.2494 -2.811 1.19261 -5.76345 -0.720766 16.3369 "
        "4.82052 0.6734 5.29015 1.93098 -2.19635 -2.60147 -3.42446 1.9895 -1.261 -2.28809";
    auto data = reader::read_data_from_string<float>(input);

    auto assignments = create_assignments(data.points, true);

    input_t cu_in;
    float* cu_out;
    float* cu_work_out;
    csize_t* cu_idxs;
    csize_t* cu_size;
    asgn_t* cu_asgn;
    float host_res[4];
    kernel_info kernel(50, 512);

    cu_in.count = (csize_t)data.points;
    cu_in.dim = (csize_t)data.dim;

    CUCH(cudaSetDevice(0));

    CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
    CUCH(cudaMalloc(&cu_out, data.dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_work_out, kernel.grid_dim * data.dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));
    CUCH(cudaMalloc(&cu_idxs, data.points * sizeof(csize_t)));
    CUCH(cudaMalloc(&cu_size, sizeof(csize_t)));

    CUCH(cudaMemcpy(
        cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
        cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

    run_merge_clusters(cu_asgn, cu_idxs, cu_size, (csize_t)data.points, (asgn_t)0, (asgn_t)0, (asgn_t)0, kernel);

    run_centroid(cu_in.data, cu_idxs, cu_work_out, cu_out,1, cu_in.dim, kernel);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    CUCH(cudaMemcpy(&host_res, cu_out, data.dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(host_res[0], -1.29547f);
    EXPECT_FLOAT_EQ(host_res[1], 8.00796f);
    EXPECT_FLOAT_EQ(host_res[2], -7.49481f);
}

TEST(kernel, centroid_big)
{
    /// 100 000 points with dim 15
    auto data = reader::read_data_from_file<float>("big");

    auto assignments = create_assignments(data.points, false);

    input_t cu_in;
    float* cu_out;
    float* cu_work_out;
    csize_t* cu_idxs;
    csize_t* cu_size;
    asgn_t* cu_asgn;
    float* host_res = new float[data.dim + 1];
    kernel_info kernel(10, 256);

    cu_in.count = (csize_t)data.points;
    cu_in.dim = (csize_t)data.dim;

    auto start = std::chrono::system_clock::now();

    CUCH(cudaSetDevice(0));

    CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
    CUCH(cudaMalloc(&cu_out, data.dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_work_out, kernel.grid_dim * data.dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));
    CUCH(cudaMalloc(&cu_idxs, data.points * sizeof(csize_t)));
    CUCH(cudaMalloc(&cu_size, sizeof(csize_t)));

    CUCH(cudaMemcpy(
        cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
        cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

    CUCH(cudaDeviceSynchronize());

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "gpu prepare time: " << elapsed_seconds.count() << "\n";

    run_merge_clusters(cu_asgn, cu_idxs, cu_size, (csize_t)data.points, (asgn_t)0, (asgn_t)0, (asgn_t)0, kernel);

    start = std::chrono::system_clock::now();

    run_centroid(cu_in.data, cu_idxs, cu_work_out, cu_out, (csize_t)data.points, cu_in.dim, kernel);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "gpu compute time: " << elapsed_seconds.count() << "\n";

    CUCH(cudaMemcpy(host_res, cu_out, data.dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    start = std::chrono::system_clock::now();
    auto ser = serial_centroid(data, assignments.data(), 0);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "serial compute time: " << elapsed_seconds.count() << "\n";

    for (size_t i = 0; i < data.dim; i++)
        EXPECT_FALSE(validator::float_diff(host_res[i], ser[i]));

    delete[] host_res;
}