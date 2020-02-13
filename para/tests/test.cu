#include <gmock/gmock.h>
#include <reader.hpp>

#include "../include/kernels.cuh"

TEST(kernel, euclid_min)
{
	std::string input = "3 32 -1.29547 8.00796 -7.49481 0.873838 0.59632 -1.51012 2.29091 0.944923 1.97487 -11.8492 0.220984 -3.32246 -3.18128 -0.856242 8.65632 -3.82426 3.51576 0.726751 0.468351 6.02061 1.30511 -3.4598 8.40714 -3.51013 0.875993 1.37086 -4.31532 5.93115 6.5827 3.40516 4.71282 4.14184 -2.94859 1.04552 4.30228 1.4721 -2.33436 -8.34392 -0.658711 7.88312 -3.95856 -4.61158 -5.23649 -2.20981 4.77685 -0.16196 0.051402 4.59879 -10.0206 0.902125 -0.0203374 1.64312 -2.10842 -1.20475 -12.092 0.0737986 2.18845 -1.17889 -11.8277 9.89593 3.83565 3.71229 -0.498549 0.436246 1.65099 4.69318 4.50654 4.72271 -3.79102 -5.22239 0.974885 -10.9422 5.59455 1.47331 5.72225 -1.47315 -2.90234 -7.91345 -9.92892 11.5448 -4.2494 -2.811 1.19261 -5.76345 -0.720766 16.3369 4.82052 0.6734 5.29015 1.93098 -2.19635 -2.60147 -3.42446 1.9895 -1.261 -2.28809";

	auto data = clustering::reader::read_data_from_string<float>(input);

	float* cu_data;
	output_t* cu_res;
	output_t host_res[2];

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc(&cu_data, sizeof(float) * data.points * data.dim));
	CUCH(cudaMalloc(&cu_res, sizeof(output_t) * 2));

	CUCH(cudaMemcpy(cu_data, data.data.data(), sizeof(float) * data.points * data.dim, cudaMemcpyKind::cudaMemcpyHostToDevice));

	euclidean_min << <2, 32, 32 * sizeof(float) * data.points * data.dim >> > (cu_data, data.dim, data.points, 16, cu_res);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	CUCH(cudaMemcpy(&host_res, cu_res, sizeof(output_t) * 2, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	
	int i = 0;
}

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}