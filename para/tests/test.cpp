#include "centroid_test.hpp"
#include "covariance_test.hpp"
#include "gmhc_test.hpp"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}