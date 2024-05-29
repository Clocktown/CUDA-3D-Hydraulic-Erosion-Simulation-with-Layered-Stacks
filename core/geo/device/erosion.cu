#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void horizontalErosionKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	const float integrationScale{ simulation.rGridScale * simulation.rGridScale * simulation.deltaTime };
	float floor{ -FLT_MAX };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		const glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const float sand{ height[BEDROCK] + height[SAND] };
		const float water{ sand + height[WATER] };
		float damage{ simulation.damages[flatIndex] };

		glm::vec4 erosions{ 0.0f };
		float maxErosion{ 0.0f };
		glm::vec2 maxSplit{ floor, height[BEDROCK] };

		struct
		{
			glm::ivec2 index;
			int flatIndices[4];
			int layerCount;
			int layer;
			glm::vec4 height;
			float sand;
			float water;
		} neighbor;

		for (int i{ 0 }; i < 4; ++i)
		{
			neighbor.index = index + glm::cuda_cast(offsets[i]);

			if (isOutside(neighbor.index, simulation.gridSize))
			{
				continue;
			}

			neighbor.flatIndices[i] = flattenIndex(neighbor.index, simulation.gridSize);
			neighbor.layerCount = simulation.layerCounts[neighbor.flatIndices[i]];
			neighbor.flatIndices[i] += (neighbor.layerCount - 1) * simulation.layerStride;

			for (neighbor.layer = neighbor.layerCount - 1; neighbor.layer >= 0; --neighbor.layer, neighbor.flatIndices[i] -= simulation.layerStride)
			{
				neighbor.height = glm::cuda_cast(simulation.heights[neighbor.flatIndices[i]]);
				neighbor.sand = neighbor.height[BEDROCK] + neighbor.height[SAND];
				neighbor.water = neighbor.sand + neighbor.height[WATER];

				const glm::vec2 split{ glm::max(floor, neighbor.sand), glm::min(height[BEDROCK], neighbor.water) };

				if (split.y - split.x > 0.f)
				{
					const float area{ (split.y - split.x) * simulation.gridScale };
					const float heightDifference{ height[BEDROCK] - neighbor.sand };
					const float sinSlope{ sin(atan2(heightDifference, simulation.gridScale)) };
					const float t = glm::smoothstep(simulation.minHorizontalErosionSlope, 1.f, sinSlope); // see sedimentKernel for a plot of the function
					const float slopeScale{ t * sinSlope };
					const glm::vec2 velocity{ glm::cuda_cast(simulation.velocities[neighbor.flatIndices[i]]) };
					const float speed{ glm::length(velocity) };
					const glm::vec2 normal{ glm::cuda_cast(offsets[i]) };
					const float attenuation{ 0.5f - 0.5f * glm::dot(normal, velocity / (speed + glm::epsilon<float>()))};
					
					erosions[i] = simulation.horizontalErosionStrength * slopeScale * speed * attenuation * area * integrationScale;
					
					if (erosions[i] > maxErosion)
					{
						maxErosion = erosions[i];
						maxSplit = split;
					}

					break;
				}
			}
		}

		float totalErosion{ erosions.x + erosions.y + erosions.z + erosions.w };
		erosions *= glm::clamp((maxSplit.y - maxSplit.x - damage) / (totalErosion + glm::epsilon<float>()), 0.f, 1.0f);

		totalErosion = 0.f;
		for (int i{ 0 }; i < 4; ++i)
		{
			if (erosions[i] > 0.0f)
			{
				totalErosion += erosions[i];
				atomicAdd((float*)(simulation.heights + neighbor.flatIndices[i]) + SAND, erosions[i]);
			}
		}

		damage += totalErosion;

		simulation.splits[flatIndex] = glm::cuda_cast(maxSplit);
		simulation.damages[flatIndex] = damage;

		floor = height[CEILING];
	}
}

template <bool enableVertical>
__global__ void sedimentKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };
	float floor{ -FLT_MAX };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		float sediment{ simulation.sediments[flatIndex] };
		/* https://www.tutorialspoint.com/execute_matplotlib_online.php
			import matplotlib.pyplot as plt
			import numpy as np

			x = np.arange(0, 0.5 * np.pi, 0.01)
			y = np.sin(x)
			t = np.clip((y - 0.6) / (1 - 0.6), 0.0, 1.0); # where 0.6 is simulation.verticalErosionSlopeFadeStart
			t =  t * t * (3.0 - 2.0 * t)
			y = y * (1-t) + 0

			z = np.sin(x)
			t = np.clip((z - 0.9) / (1 - 0.9), 0.0, 1.0); # where 0.9 is simulation.minHorizontalErosionSlope
			t =  t * t * (3.0 - 2.0 * t)
			z = t * z

			plt.plot(180 * x / np.pi, y)
			plt.plot(180 * x / np.pi, z)
			plt.show()
		*/
		// Remap sin(alpha)
		const float actualSlope = simulation.slopes[flatIndex];
		const float t = glm::smoothstep(simulation.verticalErosionSlopeFadeStart, 1.f, actualSlope);
		// [0,1] -> [min, max]
		const float slope{ simulation.minTerrainSlopeScale + (simulation.maxTerrainSlopeScale - simulation.minTerrainSlopeScale) * actualSlope * (1.f - t)};
		const float speed{ glm::length(glm::cuda_cast(simulation.velocities[flatIndex])) };
		const float waterScale{ glm::clamp(1.f - tanhf(height[WATER] * simulation.erosionWaterScale), 0.f, 1.f) };
		const float topErosionWaterScale{ glm::max(1.0f - glm::max(simulation.iTopErosionWaterScale * (height[CEILING] - height[BEDROCK] - height[SAND] - height[WATER]), 0.f), 0.0f)};
		const float topBedrockScale{ simulation.bedrockDissolvingConstant * speed * topErosionWaterScale * simulation.minTerrainSlopeScale };
		const float bedrockScale{ glm::max(1.0f - height[SAND] * simulation.iSandThreshold, 0.0f) * simulation.bedrockDissolvingConstant * slope * speed * waterScale };
		const float sedimentCapacity{ simulation.sedimentCapacityConstant * slope * speed * waterScale};
		
		if constexpr (enableVertical) {
			const float bedrockTop = ((layer + 1) < layerCount) ? ((float*)(simulation.heights + flatIndex + simulation.layerStride))[BEDROCK] : height[CEILING];
			const float deltaBedrockTop{ glm::min(topBedrockScale * simulation.deltaTime, glm::max(bedrockTop - height[CEILING], 0.f)) };
			const float deltaBedrock{ glm::min(bedrockScale * simulation.deltaTime, glm::max(height[BEDROCK] - floor, 0.f)) };

			height[CEILING] += deltaBedrockTop;
			height[SAND] += deltaBedrockTop + deltaBedrock;
			height[BEDROCK] -= deltaBedrock;
		}

		if ((sedimentCapacity > sediment))
		{
			if constexpr (enableVertical) {
				const float deltaSand{ glm::min(simulation.sandDissolvingConstant * (sedimentCapacity - sediment) * simulation.deltaTime, height[SAND]) };

				height[SAND] -= deltaSand;
				sediment += deltaSand;
			}
		}
		else
		{
			float deltaSediment{ glm::min(simulation.sedimentDepositionConstant * (sediment - sedimentCapacity) * simulation.deltaTime,  height[CEILING] - height[BEDROCK] - height[SAND] - height[WATER]) };
			deltaSediment = glm::min(deltaSediment, (sediment - sedimentCapacity));
			sediment -= deltaSediment;
			height[SAND] += deltaSediment;
		}

		// Damage recovery
		float damage = simulation.damages[flatIndex];
		float deltaDamage{ glm::min(simulation.damageRecovery * simulation.deltaTime, glm::min(damage, height[BEDROCK] - floor)) };
		damage -= deltaDamage;
		height[BEDROCK] -= deltaDamage;

		simulation.heights[flatIndex] = glm::cuda_cast(height);
		simulation.sediments[flatIndex] = sediment;
		simulation.damages[flatIndex] = damage;

		floor = height[CEILING];
	}
}

__global__ void damageKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	const int flatBase{ flattenIndex(index, simulation.gridSize) };
	int flatIndex{ flatBase };
	int layerCount{ simulation.layerCounts[flatIndex] };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		const glm::vec2 split{ glm::cuda_cast(simulation.splits[flatIndex]) };
		const float damage{ simulation.damages[flatIndex] };

		if (damage < simulation.minSplitDamage || damage < simulation.splitThreshold * (split.y - split.x) || layerCount == simulation.maxLayerCount)
		{
			continue;
		}

		int above{ flatBase + layerCount++ * simulation.layerStride };
		int below{ above - simulation.layerStride };

		for (; below > flatIndex; above = below, below -= simulation.layerStride)
		{
			simulation.heights[above] = simulation.heights[below];
			simulation.fluxes[above] = simulation.fluxes[below];
			simulation.stability[above] = simulation.stability[below];
			simulation.sediments[above] = simulation.sediments[below];
			simulation.damages[above] = simulation.damages[below];
		}

		// Upper part of split
		simulation.heights[above] = simulation.heights[below];
		simulation.fluxes[above] = simulation.fluxes[below];
		simulation.stability[above] = simulation.stability[below];
		simulation.sediments[above] = simulation.sediments[below];
		simulation.damages[above] = 0.0f;

		// Lower part of split
		simulation.heights[below] = float4{ glm::max(split.y - damage, split.x), 0.0f, 0.0f, split.y };
		simulation.fluxes[below] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		//simulation.stability[below] = FLT_MAX;
		simulation.sediments[below] = 0.0f;
		simulation.damages[below] = 0.0f;

		++layer;
	}

	simulation.layerCounts[flatBase] = layerCount;
}

void erosion(const Launch& launch, bool enable_vertical, bool enable_horizontal, geo::Performance& perf)
{
	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStart);
	if(enable_horizontal) CU_CHECK_KERNEL(horizontalErosionKernel<<<launch.gridSize, launch.blockSize>>>());
	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStop);
	if (perf.measureIndividualKernels) {
		perf.measure("Horizontal Erosion", perf.kernelStart, perf.kernelStop);
	}

	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStart);
	if(enable_horizontal) CU_CHECK_KERNEL(damageKernel<<<launch.gridSize, launch.blockSize>>>());
	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStop);
	if (perf.measureIndividualKernels) {
		perf.measure("Split Kernel", perf.kernelStart, perf.kernelStop);
	}

	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStart);
	if (enable_vertical) {
		CU_CHECK_KERNEL(sedimentKernel<true><<<launch.gridSize, launch.blockSize>>>());
	}
	else {
		CU_CHECK_KERNEL(sedimentKernel<false><<<launch.gridSize, launch.blockSize>>>());
	}
	if (perf.measureIndividualKernels) cudaEventRecord(perf.kernelStop);
	if (perf.measureIndividualKernels) {
		perf.measure("Vertical Erosion", perf.kernelStart, perf.kernelStop);
	}
}

}
}
