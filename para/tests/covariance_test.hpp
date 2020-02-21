#include <gmock/gmock.h>
#include <reader.hpp>

#include <chrono>

#include "serial_impl.hpp"

TEST(kernel, covariance_small)
{
	std::string input = "3 32 -1.29547 8.00796 -7.49481 0.873838 0.59632 -1.51012 2.29091 0.944923 1.97487 -11.8492 0.220984 -3.32246 -3.18128 -0.856242 8.65632 -3.82426 3.51576 0.726751 0.468351 6.02061 1.30511 -3.4598 8.40714 -3.51013 0.875993 1.37086 -4.31532 5.93115 6.5827 3.40516 4.71282 4.14184 -2.94859 1.04552 4.30228 1.4721 -2.33436 -8.34392 -0.658711 7.88312 -3.95856 -4.61158 -5.23649 -2.20981 4.77685 -0.16196 0.051402 4.59879 -10.0206 0.902125 -0.0203374 1.64312 -2.10842 -1.20475 -12.092 0.0737986 2.18845 -1.17889 -11.8277 9.89593 3.83565 3.71229 -0.498549 0.436246 1.65099 4.69318 4.50654 4.72271 -3.79102 -5.22239 0.974885 -10.9422 5.59455 1.47331 5.72225 -1.47315 -2.90234 -7.91345 -9.92892 11.5448 -4.2494 -2.811 1.19261 -5.76345 -0.720766 16.3369 4.82052 0.6734 5.29015 1.93098 -2.19635 -2.60147 -3.42446 1.9895 -1.261 -2.28809";
	auto data = clustering::reader::read_data_from_string<float>(input);

	auto assignments = create_assignments(data.points, true);

	input_t cu_in;
	float* cu_out;
	clustering::asgn_t* cu_asgn;
	float host_res[6];
	float centroid[3] = { -1.29547f, 8.00796f, -7.49481f };
	kernel_info kernel{ 1, 32, 32 };

	cu_in.count = data.points;
	cu_in.dim = data.dim;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
	CUCH(cudaMalloc(&cu_out, 6 * sizeof(float)));
	CUCH(cudaMemset(cu_out, 0, 6 * sizeof(float)));
	CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));

	CUCH(cudaMemcpy(cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

	assign_constant_storage(centroid, 3 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	run_covariance(cu_in, cu_asgn, cu_out, 0, kernel);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	CUCH(cudaMemcpy(&host_res, cu_out, 6 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	size_t count = 1;

	auto res = serial_covariance(data, assignments.data(), 0);

	EXPECT_FLOAT_EQ(host_res[0] / count, res[0]);
	EXPECT_FLOAT_EQ(host_res[1] / count, res[1]);
	EXPECT_FLOAT_EQ(host_res[2] / count, res[2]);
	EXPECT_FLOAT_EQ(host_res[3] / count, res[3]);
	EXPECT_FLOAT_EQ(host_res[4] / count, res[4]);
	EXPECT_FLOAT_EQ(host_res[5] / count, res[5]);
}

TEST(kernel, covariance_big)
{
	///100 000 points with dim 15
	auto data = clustering::reader::read_data_from_file<float>("big");

	auto assignments = create_assignments(data.points, false);
	auto cov_size = ((data.dim + 1) * data.dim ) / 2;

	input_t cu_in;
	float* cu_out, * cu_centroid;
	clustering::asgn_t* cu_asgn;
	float* host_res = new float[cov_size];
	float* host_centroid = new float[data.dim];
	kernel_info kernel{ 50, 1024, 100 };

	cu_in.count = data.points;
	cu_in.dim = data.dim;

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_in.data, sizeof(float) * data.points * data.dim));
	CUCH(cudaMalloc(&cu_out, cov_size * sizeof(float)));
	CUCH(cudaMemset(cu_out, 0, cov_size * sizeof(float)));
	CUCH(cudaMalloc(&cu_centroid, data.dim * sizeof(float)));
	CUCH(cudaMemset(cu_centroid, 0, data.dim * sizeof(float)));
	CUCH(cudaMalloc(&cu_asgn, data.points * sizeof(uint32_t)));

	CUCH(cudaMemcpy(cu_in.data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUCH(cudaMemcpy(cu_asgn, assignments.data(), sizeof(uint32_t) * data.points, cudaMemcpyKind::cudaMemcpyHostToDevice));

	run_centroid(cu_in, cu_asgn, cu_centroid, 0, kernel_info{ 5, 128 });

	CUCH(cudaMemcpy(host_centroid, cu_centroid, sizeof(float) * data.dim, cudaMemcpyKind::cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < data.dim; i++)
		host_centroid[i] /= 100000;


	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	assign_constant_storage(host_centroid, data.dim * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	run_covariance(cu_in, cu_asgn, cu_out, 0, kernel);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	CUCH(cudaMemcpy(host_res, cu_out, cov_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	size_t count = 100000;

	auto res = serial_covariance_with_centroid(data, assignments.data(), 0);

	for (size_t i = 0; i < data.dim; i++)
		EXPECT_FLOAT_EQ(host_res[i] / count, res[i]);
}