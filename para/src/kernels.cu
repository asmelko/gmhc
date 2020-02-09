#include <kernels.cuh>



__global__ void kernel()
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
}

void run_kernel()
{
	kernel<< <1, 50 >> > ();
}