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

	const float noiseVal = 0.5f + 0.5f * rainNoise::cudaNoise::discreteNoise(make_float3(index.x, index.y, 0.f), simulation.gridScale, simulation.step);

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int topLayer{ simulation.layerCounts[flatIndex] - 1 };
	flatIndex += topLayer * simulation.layerStride;

	float water = simulation.heights[flatIndex].z;

	glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };

	for (int i = 0; i < 4; ++i) {
		const glm::vec2 dVec = simulation.sourceLocations[i] - glm::vec2(index) * simulation.gridScale;
		const float d2 = glm::dot(dVec, dVec);
		const float r2 = simulation.sourceSize[i] * simulation.sourceSize[i];
		if (d2 <= r2 && r2 > 0.f) {
			water += 2.f * noiseVal * simulation.sourceStrengths[i]  * simulation.deltaTime;
			flux += simulation.sourceFlux[i] * simulation.deltaTime;
		}
	}

	water += 2.f * noiseVal * simulation.rain * simulation.deltaTime;

	simulation.heights[flatIndex].z = water;

	//if (glm::dot(flux, flux) > 0.f) {
		simulation.fluxes[flatIndex] = glm::cuda_cast(flux);
	//}
}

void rain(const Launch& launch)
{
	CU_CHECK_KERNEL(rainKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
