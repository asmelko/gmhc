#include <gmhc.hpp>
#include <kernels.cuh>

using namespace clustering;

std::vector<pasgn_t> gmhc::run()
{
	run_kernel();
	return {};
}

void gmhc::free() {}