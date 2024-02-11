#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void initKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.gridSize))
	{
		return;
	}

	const int flatIndex{ flattenIndex(index, simulation.gridSize) };
	simulation.layerCounts[flatIndex] = 1;
	simulation.heights[flatIndex] = float4{ 16.0f, 0.0f, 0.0f, FLT_MAX };
}

void init(const Launch& launch)
{
	CU_CHECK_KERNEL(initKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
