#include "gmock/gmock.h"
#include "smhc.hpp"
#include "reader.hpp"

#include <string>

using namespace clustering;

TEST(serial, basic_maha_dist_test)
{
	std::string input = "2 20 0 0 -1 0 1 0 0 1 0 -1 0 100 -1 100 1 100 0 99 0 101 0 250 -1 250 1 250 0 249 0 251 200 50 200 49 200 51 199 50 201 50";

	auto data = reader::read_data_from_string<float>(input);

	smhc serial(5);

	serial.initialize(data.data.data(), data.points, data.dim);

	auto res = serial.run();
	
	std::vector<pasgn_t> resp;
	for (auto&& pd : res)
		resp.push_back(pd.first);

	std::vector<pasgn_t> expected{ {0, 1}, {5, 6}, {10, 11}, {15, 16}, {20, 3}, {21, 8}, {22, 13}, {23, 18}, {27, 19}, {28, 17}, {25, 9}, {30, 7}, {26, 14}, {32, 12}, {24, 2}, {34, 4}, {35, 31}, {36, 33}, {37, 29} };

	EXPECT_EQ(resp, expected);
}

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}