#ifndef SERIAL_IMPL_HPP
#define SERIAL_IMPL_HPP

#include <validator.hpp>
#include <kernels.cuh>
#include <reader.hpp>

chunk_t serial_euclidean_min(const clustering::reader::data_t<float>& data);

std::vector<clustering::asgn_t> create_assignments(size_t count, bool unique);

std::vector<float> serial_centroid(const clustering::reader::data_t<float>& data, const clustering::asgn_t* assignments, clustering::asgn_t cid);

std::vector<float> serial_covariance(const clustering::reader::data_t<float>& data, const clustering::asgn_t* assignments, clustering::asgn_t cid);
std::vector<float> serial_covariance_by_centroid(const clustering::reader::data_t<float>& data, const clustering::asgn_t* assignments, const float* centroid, clustering::asgn_t cid);

#endif