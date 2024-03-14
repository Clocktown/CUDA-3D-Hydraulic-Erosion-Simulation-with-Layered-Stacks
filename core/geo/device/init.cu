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

	int flatIndex{ flattenIndex(index, simulation.gridSize) };

	/*for (int i = 0; i < 8; ++i, flatIndex += simulation.layerStride) {
		if(i == 0)
			simulation.layerCounts[flatIndex] = 8;
		simulation.heights[flatIndex] = float4{ i * 20.f + 10.f, 0.f, 0.f,  i == 7 ? FLT_MAX : (i + 1) * 20.f };
		simulation.sediments[flatIndex] = 0.0f;
		simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	}*/

	if (index.x > 64)
	{
		simulation.layerCounts[flatIndex] = 2;
		simulation.heights[flatIndex] = float4{ (simulation.gridSize.x - index.x) / 16.0f, 0.0f, 0.0f, 30.0f };
		simulation.sediments[flatIndex] = 0.0f;
		simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };

		simulation.heights[flatIndex + simulation.layerStride] = float4{ 30.0f + index.x / 16.0f, 2.0f, 2.0f, FLT_MAX };
		simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
		simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	}
	else
	{
		simulation.layerCounts[flatIndex] = 1;
		simulation.heights[flatIndex] = float4{ (simulation.gridSize.x - index.x) / 16.0f, 0.0f, 0.0f, FLT_MAX };
		simulation.sediments[flatIndex] = 0.0f;
		simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	}
}

void init(const Launch& launch)
{
	CU_CHECK_KERNEL(initKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
