#ifndef GMHC_HPP
#define GMHC_HPP

#include "apriori.hpp"

namespace clustering
{

//data to share among each clustering context
struct shared_apriori_data_t
{
	//device variable to retrieve minimum
	chunk_t* cu_min;

	//device variables for update kernel
	csize_t* cu_eucl_upd_size;
	csize_t* cu_maha_upd_size;

	//device variables to compute inverse matrix
	float** cu_read_icov, ** cu_write_icov;
	float* cu_tmp_icov;
	int* cu_info;
	int* cu_pivot;

	//overall cluster count
	csize_t cluster_count;
	//next available id
	asgn_t id;

	//number of closest neighbors for each cluster
	static constexpr csize_t neighbors_size = 1;

	//handle to CUBLAS library
	cublasHandle_t cublas_handle;
};

//Mahalanobis hierarchical clustering class
class gmhc : public hierarchical_clustering<float>
{
	//device point array
	float* cu_points_;
	//device centroid array
	float* cu_centroids_;
	//device inverse array
	float* cu_icov_;
	//device assignments array
	asgn_t* cu_point_asgns_;

	//device neighbor array
	neighbor_t* cu_neighs_;
	//device intermediate neighbor array
	neighbor_t* cu_tmp_neighs_;

	//device update array
	csize_t* cu_update_;
	//host status array
	cluster_data_t* cluster_data_;

	//Mahalanobis threshold
	csize_t maha_threshold_;
	//parameter for kernels
	kernel_info starting_info_;

	//shared data for contexts
	shared_apriori_data_t common_;

	//context array
	std::vector<clustering_context_t> apr_ctxs_;
	//number of apriori clusters
	csize_t apriori_count_;

	//arrays for the final context
	cluster_data_t* apriori_cluster_data_;
	float* cu_apriori_centroids_;
	float* cu_apriori_icov_;

public:
	//result type for the run method
	using res_t = pasgnd_t<float>;

	bool initialize(const float* data_points, csize_t data_points_size, csize_t data_point_dim, csize_t mahalanobis_threshold, const asgn_t* apriori_assignments = nullptr, validator* vld = nullptr);

	virtual std::vector<res_t> run() override;

	virtual void free() override;

private:
	//method sets fields of apriori clusters
	void set_apriori(clustering_context_t& cluster, csize_t offset, csize_t size, validator* vld);
	//method that reads apriori assigmnets array and initializes all apriori clusters
	void initialize_apriori(const asgn_t* apriori_assignments, validator* vld);
	//method that creates final context
	void move_apriori(csize_t eucl_size, csize_t maha_size);
};

}

#endif