#ifndef READER_H
#define READER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

namespace clustering {

struct reader
{
	template <typename T>
	struct data_t
	{
		size_t dim;
		size_t points;
		std::vector<T> data;
	};

	template <typename T>
	static data_t<T> read_data_from_file(const std::string& file)
	{
		std::ifstream fs(file);

		if (!fs.is_open())
		{
			std::cerr << "file could not open";
			return { 0, 0, std::vector<T>{} };
		}

		return read_data<T>(fs);
	}

	template <typename T>
	static data_t<T> read_data_from_string(const std::string& data)
	{
		std::stringstream ss(data, std::ios_base::in);
		return read_data<T>(ss);
	}

	template <typename T>
	static data_t<T> read_data(std::istream& stream)
	{
		size_t dim, points;

		stream >> dim >> points;

		std::vector<T> ret;
		ret.resize(dim * points);

		size_t idx = 0;

		for (size_t i = 0; i < points * dim; ++i)
		{
			stream >> ret[idx++];
			if (stream.fail() || stream.bad())
			{
				std::cerr << "issue when reading data";
				return { 0, 0, std::vector<T>{} };
			}
		}

		return { dim, points, std::move(ret) };
	}
};

}

#endif