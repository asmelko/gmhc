#include <iostream>
#include <istream>

#include "gmhc.hpp"
#include "reader.hpp"

std::vector<clustering::asgn_t> create_apriori_assigns(const char* file_name, size_t count)
{
	std::ifstream fs(file_name);
	std::vector<clustering::asgn_t> ret;

	if (!fs.is_open())
	{
		std::cerr << "file could not open" << std::endl;
		return {};
	}

	for (size_t i = 0; i < count; ++i)
	{
		clustering::asgn_t tmp;
		fs >> tmp;
		ret.push_back(tmp);

		if (fs.fail() || fs.bad())
		{
			std::cerr << "issue when reading data";
			return {};
		}
	}
	return ret;
}

int main(int argc, char** argv)
{
	if (argc != 3 && argc != 4)
	{
		std::cout << "bad input" << std::endl <<
			"usage: mhclust file_name maha_threshold [apriori_file]" << std::endl;
		return 1;
	}

	auto data = clustering::reader::read_data_from_file<float>(argv[1]);

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

	auto res = gmhclust.run();

	for (auto& e : res)
		std::cout << e.first.first << " "
		<< e.first.second << " "
		<< e.second << std::endl;

	gmhclust.free();

	return 0;
}
