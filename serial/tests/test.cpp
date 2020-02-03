#include <reader.hpp>
#include "../smhc.hpp"

using namespace clustering;

int main()
{
	auto data = reader::read_data("small");

	smhc serial_algo;

	serial_algo.initialize(data.data, data.points, data.dim);

	auto ret = serial_algo.run();
}