#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void evaporationKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.cellCount)
	{
		simulation.heights[flatIndex].z = glm::max((1.0f - simulation.evaporation * simulation.deltaTime) * simulation.heights[flatIndex].z, 0.0f);
	}
}

void evaporation(const Launch& launch)
{
	CU_CHECK_KERNEL(evaporationKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
