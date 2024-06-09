#include "simulation.hpp"

namespace initNoise {
#include "cuda_noise.cuh"
}

namespace geo
{
namespace device
{

__device__ float fbm(glm::vec2 st, int octaves, float scale, int seed) {
	// Initial values
	float value = 0.0;
	float amplitude = .5;
	float frequency = 0.;
	//
	// Loop of octaves
	for (int i = 0; i < octaves; i++) {
		value += amplitude * initNoise::cudaNoise::simplexNoise(make_float3(st.x, st.y, 0), scale, seed);
		scale *= 2.;
		amplitude *= .5;
	}
	return value;
}

__global__ void initKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };


	float bedrockHeight = -30.f;
	float ceiling = FLT_MAX;
	simulation.layerCounts[flatIndex] = 1;

	/*if (index.x > (64.f / 256.f) * simulation.gridSize.x) {
		bedrockHeight = 50.f;
	}*/

	/*if (index.x <= (64.f / 256.f) * simulation.gridSize.x) {
		bedrockHeight = 0.f;
	}

	if (index.x > (200.f/256.f) * simulation.gridSize.x && index.y > (120.f/256.f) * simulation.gridSize.y && index.y < (136.f/256.f) * simulation.gridSize.y && index.x < (240.f/256.f) * simulation.gridSize.x) {
		bedrockHeight -= 25.f * (1.f - ((index.x - (200.f/256.f) * simulation.gridSize.x) / ((40.f/256.f) * simulation.gridSize.x) ));
	}

	if ((index.x <= (200.f / 256.f) * simulation.gridSize.x && index.y > (126.f / 256.f) * simulation.gridSize.y && index.y < (130.f / 256.f) * simulation.gridSize.y && index.x >(64.f / 256.f) * simulation.gridSize.x)) {
		bedrockHeight = 25.f * (((index.x - (64.f/256.f) * simulation.gridSize.x) / (simulation.gridSize.x * (200.f - 64.f) / 256.f)));
		ceiling = bedrockHeight + 3.f;
		simulation.layerCounts[flatIndex] = 2;

		simulation.heights[flatIndex + simulation.layerStride] = float4{ 50.f, 0.0f, 0.0f, FLT_MAX};
		simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
		simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.damages[flatIndex + simulation.layerStride] = 0.0f;
	}*/

	/*if (index.x >(64.f / 256.f) * simulation.gridSize.x && index.y >(120.f / 256.f) * simulation.gridSize.y && index.y < (136.f / 256.f) * simulation.gridSize.y && index.x < (240.f / 256.f) * simulation.gridSize.x) {
		bedrockHeight = 25.f;
		bedrockHeight = 30.f;
	}

	if (index.x >(64.f / 256.f) * simulation.gridSize.x && index.y >(200.f / 256.f) * simulation.gridSize.y && index.y < (216.f / 256.f) * simulation.gridSize.y && index.x < (240.f / 256.f) * simulation.gridSize.x) {
		bedrockHeight = 40.f;
	}

	if (index.x > (64.f/256.f) * simulation.gridSize.x && index.y > (40.f/256.f) * simulation.gridSize.y && index.y < (56.f/256.f) * simulation.gridSize.y && index.x < (240.f/256.f) * simulation.gridSize.x) {
		bedrockHeight = 40.f;
	}

	if (index.x > (120.f/256.f) * simulation.gridSize.x && index.y > (16.f/256.f) * simulation.gridSize.y && index.y < (240.f/256.f) * simulation.gridSize.y && index.x < (136.f/256.f) * simulation.gridSize.x) {
		bedrockHeight = 30.f;
	}

	if (index.y > (117.f/256.f) * simulation.gridSize.y && index.y < (139.f/256.f) * simulation.gridSize.y && index.x > (117.f/256.f) * simulation.gridSize.x  && index.x < (139.f/256.f) * simulation.gridSize.x) {
		bedrockHeight = 50.f;
	}*/

	//simulation.layerCounts[flatIndex] = 1;

	float scale = simulation.gridSize.x / 256.f;

	if ((index.x > 10 * scale) && (index.x < 246 * scale)) {
		if (index.y > 100 * scale && index.y < 110 * scale) {
			bedrockHeight = 20.f;
			if (index.x > 20 * scale && index.x < 236 * scale) {
				simulation.layerCounts[flatIndex] = 2;

				simulation.heights[flatIndex + simulation.layerStride] = float4{ bedrockHeight , 0.0f, 0.0f, FLT_MAX };
				simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
				simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
				simulation.damages[flatIndex + simulation.layerStride] = 0.0f;

				bedrockHeight = -30.f;
				ceiling = bedrockHeight + 40.f;
			}
		}
	}

	if ((index.x > 10 * scale) && (index.x < 246 * scale)) {
		if (index.y > 50 * scale && index.y < 60 * scale) {
			bedrockHeight = 30.f;
			if (index.x > 20 * scale && index.x < 236 * scale) {
				simulation.layerCounts[flatIndex] = 2;

				simulation.heights[flatIndex + simulation.layerStride] = float4{ bedrockHeight , 0.0f, 0.0f, FLT_MAX};
				simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
				simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
				simulation.damages[flatIndex + simulation.layerStride] = 0.0f;

				bedrockHeight = -30.f;
				ceiling = bedrockHeight + 50.f + 8.f * pow(1.f - fabsf(index.x - 128.f * scale) / (108.f * scale), 0.25f);
			}
		}
	}

	if ((index.x > 10 * scale) && (index.x < 246 * scale)) {
		if (index.y > 150 * scale && index.y < 160 * scale) {
			bedrockHeight = 10.f;
			if (index.x > 20 * scale && index.x < 236 * scale) {
				simulation.layerCounts[flatIndex] = 2;

				simulation.heights[flatIndex + simulation.layerStride] = float4{ bedrockHeight , 0.0f, 0.0f, FLT_MAX};
				simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
				simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
				simulation.damages[flatIndex + simulation.layerStride] = 0.0f;

				bedrockHeight = -30.f;
				ceiling = bedrockHeight + 30.f + 8.f * pow(fabsf(index.x - 128.f * scale) / (108.f * scale), 4.f);
			}
		}
	}

	simulation.heights[flatIndex] = float4{ bedrockHeight, 0.0f, 0.0f, ceiling};
	simulation.sediments[flatIndex] = 0.0f;
	simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	simulation.damages[flatIndex] = 0.0f;
	

	/*float bedrockHeight = 0.f;
	if (index.x > (64.f/256.f) * simulation.gridSize.x) {
		bedrockHeight = 50.f;
	}

	if (index.x > (70.f/256.f) * simulation.gridSize.x && index.y > (120.f/256.f) * simulation.gridSize.y && index.y < (136.f/256.f) * simulation.gridSize.y && index.x < (240.f/256.f) * simulation.gridSize.x) {
		bedrockHeight = 25.f * (256.f / simulation.gridSize.x) * (index.x - 70) / (240.f - 70.f);
	}

	simulation.layerCounts[flatIndex] = 1;
	simulation.heights[flatIndex] = float4{ bedrockHeight, 0.0f, 0.0f, FLT_MAX};
	simulation.sediments[flatIndex] = 0.0f;
	simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	simulation.damages[flatIndex] = 0.0f;
	*/

	/*const float noiseVal1 = 5.f * fbm(index, 8, 0.01f * simulation.gridScale, 42);
	const float noiseVal2 = 10.f * (1.f + fbm(index, 8, 0.005f * simulation.gridScale, 69));

	if (float(index.x) / simulation.gridSize.x  > 0.25f)
	{
		simulation.layerCounts[flatIndex] = 2;
		simulation.heights[flatIndex] = float4{ noiseVal1 + 16.f * (simulation.gridSize.x - index.x) / simulation.gridSize.x, 0.0f, 0.0f, 30.0f};
		simulation.sediments[flatIndex] = 0.0f;
		simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.damages[flatIndex] = 0.0f;

		simulation.heights[flatIndex + simulation.layerStride] = float4{ noiseVal2 + 30.0f + (16.f * index.x) / simulation.gridSize.x, 0.0f, glm::max(2.0f - noiseVal2, 0.f), FLT_MAX };
		simulation.sediments[flatIndex + simulation.layerStride] = 0.0f;
		simulation.fluxes[flatIndex + simulation.layerStride] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.damages[flatIndex + simulation.layerStride] = 0.0f;
	}
	else
	{
		simulation.layerCounts[flatIndex] = 1;
		simulation.heights[flatIndex] = float4{ noiseVal1 + 16.f * (simulation.gridSize.x - index.x) / simulation.gridSize.x, 0.0f, 0.0f, FLT_MAX };
		simulation.sediments[flatIndex] = 0.0f;
		simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.damages[flatIndex] = 0.0f;
	}*/

	/*const float noiseVal1 = 5.f * fbm(index, 8, 0.01f * simulation.gridScale, 42);
	const float noiseVal2 = 10.f * (1.f + fbm(index, 8, 0.05f * simulation.gridScale, 69));


	simulation.layerCounts[flatIndex] = 1;
	simulation.heights[flatIndex] = float4{ noiseVal2 + 16.f * (simulation.gridSize.x - index.x) / simulation.gridSize.x, 1.f, 0.0f, FLT_MAX};
	simulation.sediments[flatIndex] = 0.0f;
	simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	simulation.damages[flatIndex] = 0.0f;*/


	// generate pre-made arches to test support check
	// scene with a lot of sand to demonstrate fake "regolith"

	/*const float noiseVal1 = 5.f * fbm(index, 8, 0.01f * simulation.gridScale, 42);
	const float noiseVal2 = 40.f * (1.f + fbm(index, 8, 0.005f * simulation.gridScale, 69));


	simulation.layerCounts[flatIndex] = 1;
	simulation.heights[flatIndex] = float4{ noiseVal2, 20.f, 0.0f, FLT_MAX};
	simulation.sediments[flatIndex] = 0.0f;
	simulation.fluxes[flatIndex] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	simulation.damages[flatIndex] = 0.0f;*/

}

void init(const Launch& launch)
{
	CU_CHECK_KERNEL(initKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
