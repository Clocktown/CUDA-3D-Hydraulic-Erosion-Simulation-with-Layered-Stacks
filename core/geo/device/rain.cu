#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void rainKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int topLayer{ simulation.layerCounts[flatIndex] - 1 };
	flatIndex += topLayer * simulation.layerStride;

	simulation.heights[flatIndex].z += simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime;
}

void rain(const Launch& launch)
{
	CU_CHECK_KERNEL(rainKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
