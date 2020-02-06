#include "gmock/gmock.h"
#include "../smhc.hpp"
#include <reader.hpp>

#include <string>

using namespace clustering;

TEST(serial, basic_maha_dist_test)
{
	std::string input = "2 20 0 0 0 100 0 250 200 50 -1 0 1 0 0 1 0 -1 -1 100 1 100 0 99 0 101 -1 250 1 250 0 249 0 251 200 49 200 51 199 50 201 50";

	auto data = reader::read_data_from_string<float>(input);

	smhc serial(4);

	serial.initialize(data.data.data(), data.points, data.dim);

	auto res = serial.run();
	std::vector<pasgn_t> expected{ { 0,1 }, { 0,2 }, { 0,3 } };

	EXPECT_EQ(res, expected);
}

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}