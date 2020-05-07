#include <iostream>
#include <chrono>

#include <gmhc.hpp>
#include <reader.hpp>

int main(int argc, char** argv)
{
	if (argc != 3 && argc != 4)
	{
		std::cout << "bad input" << std::endl <<
			"usage: mhclust file_name maha_threshold [apriori_size]" << std::endl;
		return 1;
	}

	auto data = clustering::reader::read_data_from_binary_file<float>(argv[1]);

	std::vector<clustering::asgn_t> apriori_assignments;
	clustering::asgn_t* apr_asgn = nullptr;

	auto thresh = std::strtoul(argv[2], NULL, 10);

	if (argc == 4)
	{
		auto apr_size = std::strtoul(argv[3], NULL, 10);

		for (clustering::csize_t i = 0; i < data.points; i++)
			apriori_assignments.push_back(i / apr_size);

		apr_asgn = apriori_assignments.data();
	}

	clustering::gmhc gmhclust;
	clustering::validator vld;

	vld.initialize(data.data.data(), (clustering::csize_t)data.points, (clustering::csize_t)data.dim, thresh, apr_asgn);
	gmhclust.initialize(data.data.data(), (clustering::csize_t)data.points, (clustering::csize_t)data.dim, thresh, apr_asgn, &vld);

	auto start = std::chrono::system_clock::now();
	auto res = gmhclust.run();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cerr << argv[1] << " time: " << elapsed_seconds.count() << std::endl;

	for (auto& e : res)
		std::cout << e.first.first << " "
		<< e.first.second << " "
		<< e.second << std::endl;

	return 0;
}
