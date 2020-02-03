#include <reader.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace clustering;

reader::data_t reader::read_data(const std::string& file)
{
	std::ifstream is(file);

	if (!is.is_open())
	{
		std::cerr << "file could not open";
		return { 0, 0, nullptr };
	}

	size_t dim, points;

	is >> dim >> points;

	float* ret = new float[dim * points];

	size_t idx = 0;

	for (size_t i = 0; i < points * dim; ++i)
	{
		is >> ret[idx++];
		if (is.fail() || is.bad())
		{
			std::cerr << "issue when reading data";
			return { 0, 0, nullptr };
		}
	}
	
	return { dim, points, ret };
}