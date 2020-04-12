#include <iostream>
#include <chrono>

#include <gmhc.hpp>
#include <reader.hpp>

int main(int argc, char** argv)
{
	if (argc != 2)
		std::cout << "bad input" << std::endl <<
		"usage: mhclust [file name]" << std::endl;

	auto data = clustering::reader::read_data_from_file<float>(argv[1]);

	clustering::gmhc gmhclust;

	gmhclust.initialize(data.data.data(), (clustering::csize_t)data.points, (clustering::csize_t)data.dim, 20);

	auto start = std::chrono::system_clock::now();
	gmhclust.run();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << argv[1] << " time: " << elapsed_seconds.count() << std::endl;

}