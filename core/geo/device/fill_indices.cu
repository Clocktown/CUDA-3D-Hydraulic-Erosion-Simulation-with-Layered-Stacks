#include "simulation.hpp"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace geo
{
namespace device
{

char getMaxLayerCount(const char* const layerCounts, const std::ptrdiff_t count)
{
	const char* const pointer{ thrust::max_element(thrust::device, layerCounts, layerCounts + count) };

	char maxLayerCount;
	CU_CHECK_ERROR(cudaMemcpy(&maxLayerCount, pointer, sizeof(char), cudaMemcpyDeviceToHost));

	return maxLayerCount;
}

}
}
