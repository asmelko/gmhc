#include <gmock/gmock.h>
#include <reader.hpp>

#include <chrono>

#include "serial_impl.hpp"

#include <gmhc.hpp>

TEST(para, small)
{
	std::string input = "3 32 -1.29547 8.00796 -7.49481 0.873838 0.59632 -1.51012 2.29091 0.944923 1.97487 -11.8492 0.220984 -3.32246 -3.18128 -0.856242 8.65632 -3.82426 3.51576 0.726751 0.468351 6.02061 1.30511 -3.4598 8.40714 -3.51013 0.875993 1.37086 -4.31532 5.93115 6.5827 3.40516 4.71282 4.14184 -2.94859 1.04552 4.30228 1.4721 -2.33436 -8.34392 -0.658711 7.88312 -3.95856 -4.61158 -5.23649 -2.20981 4.77685 -0.16196 0.051402 4.59879 -10.0206 0.902125 -0.0203374 1.64312 -2.10842 -1.20475 -12.092 0.0737986 2.18845 -1.17889 -11.8277 9.89593 3.83565 3.71229 -0.498549 0.436246 1.65099 4.69318 4.50654 4.72271 -3.79102 -5.22239 0.974885 -10.9422 5.59455 1.47331 5.72225 -1.47315 -2.90234 -7.91345 -9.92892 11.5448 -4.2494 -2.811 1.19261 -5.76345 -0.720766 16.3369 4.82052 0.6734 5.29015 1.93098 -2.19635 -2.60147 -3.42446 1.9895 -1.261 -2.28809";
	auto data = clustering::reader::read_data_from_string<float>(input);

	clustering::gmhc para;
	clustering::validator vld;

	vld.initialize(data.data.data(), data.points, data.dim, 5);
	para.initialize(data.data.data(), data.points, data.dim, 5, &vld);

	auto res = para.run();

	ASSERT_FALSE(vld.has_error());

	std::vector<clustering::pasgn_t> expected = { {6, 29}, { 10, 22 }, { 17, 31 }, { 11, 32 }, { 15, 21 }, { 1, 34 }, { 20, 33 }, { 16, 18 }, { 2, 36 }, { 8, 37 }, { 30, 41 }, { 27, 42 }, { 3, 39 }, { 0, 7 }, { 4, 14 }, { 35, 40 }, { 38, 43 }, { 9, 24 }, { 25, 48 }, { 5, 50 }, { 45, 51 }, { 23, 52 }, { 49, 53 }, { 13, 54 }, { 47, 55 }, { 46, 56 }, { 12, 57 }, { 44, 58 }, { 26, 59 }, { 28, 60 }, { 19, 61 } };

	EXPECT_EQ(res, expected);
}


TEST(para, big)
{
	auto data = clustering::reader::read_data_from_file<float>("big");

	clustering::gmhc para;
	clustering::validator vld;

	auto thresh = 20;

	vld.initialize(data.data.data(), data.points, data.dim, thresh);
	para.initialize(data.data.data(), data.points, data.dim, thresh, &vld);

	auto res = para.run();

	ASSERT_FALSE(vld.has_error());
}