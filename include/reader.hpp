#ifndef READER_H
#define READER_H

#include <string>

namespace clustering {

struct reader
{
	struct data_t
	{
		size_t dim;
		size_t points;
		float* data;
	};

	static data_t read_data(const std::string& file);
};

}

#endif