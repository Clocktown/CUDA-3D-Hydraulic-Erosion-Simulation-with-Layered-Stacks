#include "simulation.hpp"

namespace rainNoise {
#include "cuda_noise.cuh"
}

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

	const float noiseVal = rainNoise::cudaNoise::discreteNoise(make_float3(index.x, index.y, 0.f), 1.f, simulation.step);

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int topLayer{ simulation.layerCounts[flatIndex] - 1 };
	flatIndex += topLayer * simulation.layerStride;

	simulation.heights[flatIndex].z += (noiseVal > 0.99f) ? 100.f * simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime : 0.f;
}

void rain(const Launch& launch)
{
	CU_CHECK_KERNEL(rainKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
