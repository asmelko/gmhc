#include <chrono>

#include <gmock/gmock.h>

#include "reader.hpp"
#include "serial_impl.hpp"

using namespace clustering;

// tests for covariance kernel

TEST(kernel, covariance_small)
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
    float* cu_work;
    asgn_t* cu_asgn;
    float host_res[9];
    float centroid[3] = { -1.29547f, 8.00796f, -7.49481f };
    kernel_info kernel(1, 32);

    cu_in.count = (csize_t)data.points;
    cu_in.dim = (csize_t)data.dim;

    CUCH(cudaSetDevice(0));

    CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
    CUCH(cudaMalloc(&cu_out, 9 * sizeof(float)));
    CUCH(cudaMalloc(&cu_work, kernel.grid_dim * 6 * sizeof(float)));
    CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));

    CUCH(cudaMemcpy(
        cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
        cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

    assign_constant_storage(centroid, 3 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

    run_covariance(cu_in, cu_asgn, cu_work, cu_out, 0, 1, kernel);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    CUCH(cudaMemcpy(&host_res, cu_out, 9 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    auto res = serial_covariance(data, assignments.data(), 0);

    EXPECT_FLOAT_EQ(host_res[0], res[0]);
    EXPECT_FLOAT_EQ(host_res[1], res[1]);
    EXPECT_FLOAT_EQ(host_res[2], res[2]);
    EXPECT_FLOAT_EQ(host_res[3], res[3]);
    EXPECT_FLOAT_EQ(host_res[4], res[4]);
    EXPECT_FLOAT_EQ(host_res[5], res[5]);
    EXPECT_FLOAT_EQ(host_res[6], res[6]);
    EXPECT_FLOAT_EQ(host_res[7], res[7]);
    EXPECT_FLOAT_EQ(host_res[8], res[8]);
}

TEST(kernel, covariance_big)
{
    /// 7 000 points with dim 15
    auto data = reader::read_data_from_file<float>("big");

    auto assignments = create_assignments(data.points, false);
    auto cov_size = ((data.dim + 1) * data.dim) / 2;

    input_t cu_in;
    float* cu_out;
    float* cu_work;
    asgn_t* cu_asgn;
    float* host_res = new float[data.dim * data.dim];
    kernel_info kernel(1, 32);

    cu_in.count = (csize_t)data.points;
    cu_in.dim = (csize_t)data.dim;

    auto centroid = serial_centroid(data, assignments.data(), 0);

    auto start = std::chrono::system_clock::now();
    CUCH(cudaSetDevice(0));

    CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
    CUCH(cudaMalloc(&cu_out, data.dim * data.dim * sizeof(float)));
    CUCH(cudaMalloc(&cu_work, kernel.grid_dim * cov_size * sizeof(float)));
    CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));

    CUCH(cudaMemcpy(
        cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
        cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    assign_constant_storage(centroid.data(), (csize_t)data.dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "gpu prepare time: " << elapsed_seconds.count() << "\n";

    start = std::chrono::system_clock::now();
    run_covariance(cu_in, cu_asgn, cu_work, cu_out, 0, (csize_t)data.points, kernel);
    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::cout << "gpu compute time: " << elapsed_seconds.count() << "\n";

    CUCH(cudaMemcpy(host_res, cu_out, data.dim * data.dim * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    start = std::chrono::system_clock::now();
    auto res = serial_covariance_by_centroid(data, assignments.data(), centroid.data(), 0);
    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::cout << "serial compute time: " << elapsed_seconds.count() << "\n";

    for (size_t i = 0; i < data.dim * data.dim; i++)
        EXPECT_FALSE(validator::float_diff(host_res[i], res[i]));
}