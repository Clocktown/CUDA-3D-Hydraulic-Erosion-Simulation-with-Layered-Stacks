#include "simulation.hpp"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace geo
{
namespace device
{

__global__ void fillIndicesKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		int idx = atomicAdd(simulation.atomicCounter, 1);
		simulation.indices[idx] = flatIndex;
	}
	
}

int fillIndices(const Launch& launch, int* atomicCounter, int* indices)
{
	int count = 0;
	cudaMemcpy(atomicCounter, &count, sizeof(int), cudaMemcpyHostToDevice);
	CU_CHECK_KERNEL(fillIndicesKernel<<<launch.gridSize, launch.blockSize>>>());
	cudaMemcpy(&count, atomicCounter, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

char getMaxLayerCount(const char* const layerCounts, const std::ptrdiff_t count)
{
	const char* const pointer{ thrust::max_element(thrust::device, layerCounts, layerCounts + count) };

	char maxLayerCount;
	CU_CHECK_ERROR(cudaMemcpy(&maxLayerCount, pointer, sizeof(char), cudaMemcpyDeviceToHost));

	return maxLayerCount;
}

}
}
