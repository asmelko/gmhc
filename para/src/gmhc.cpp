#include <gmhc.hpp>
#include <kernels.cuh>

using namespace clustering;

void gmhc::initialize(const float* data_points, size_t data_points_size, size_t data_point_dim)
{
	hierarchical_clustering::initialize(data_points, data_points_size, data_point_dim);

	CUCH(cudaSetDevice(0));

	CUCH(cudaMalloc<float>(&cu_points_, data_points_size * data_point_dim));

	CUCH(cudaMemcpy(cu_points_, data_points, sizeof(float) * data_points_size * data_point_dim, cudaMemcpyKind::cudaMemcpyHostToDevice));
}


std::vector<pasgn_t> gmhc::run()
{
	return {};
}

void gmhc::free() {}