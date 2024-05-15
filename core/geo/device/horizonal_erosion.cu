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
	float floor{ 0.0f };

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

				if (split.y - split.x > simulation.minHorizontalErosion)
				{
					const float area{ (split.y - split.x) * simulation.gridScale };
					const glm::vec2 velocity{ glm::cuda_cast(simulation.velocities[neighbor.flatIndices[i]]) };
					const float speed{ glm::length(velocity) };
					const glm::vec2 normal{ glm::cuda_cast(offsets[i]) };
					const float attenuation{ 1.0f - glm::max(glm::dot(normal, velocity / (speed + glm::epsilon<float>())), 0.0f) };
					
					erosions[i] = simulation.horizontalErosionStrength * attenuation * speed * area * integrationScale;
					
					if (erosions[i] > maxErosion)
					{
						maxErosion = erosions[i];
						maxSplit = split;
					}

					break;
				}
			}
		}

		const float totalErosion{ erosions.x + erosions.y + erosions.z + erosions.w };
		erosions *= glm::min((maxSplit.y - maxSplit.x - damage) / (totalErosion + glm::epsilon<float>()), 1.0f);

		for (int i{ 0 }; i < 4; ++i)
		{
			if (erosions[i] > 0.0f)
			{
				atomicAdd(simulation.sediments + neighbor.flatIndices[i], erosions[i]);
			}
		}

		damage += totalErosion;

		simulation.splits[flatIndex] = glm::cuda_cast(maxSplit);
		simulation.damages[flatIndex] = damage;

		floor = height[CEILING];
	}
}

__global__ void sedimentKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };
	float floor{ 0.0f };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		float sediment{ simulation.sediments[flatIndex] };
		const float slope{ glm::max(simulation.slopes[flatIndex], simulation.minTerrainSlope) };
		const float speed{ glm::length(glm::cuda_cast(simulation.velocities[flatIndex])) };
		const float sedimentCapacity{ simulation.sedimentCapacityConstant * slope * speed};
		
		if (sedimentCapacity > sediment)
		{
			float deltaSand{ glm::min(simulation.sandDissolvingConstant * (sedimentCapacity - sediment) * simulation.deltaTime, height[SAND]) };
			deltaSand = glm::min(deltaSand, (sedimentCapacity - sediment));
			height[SAND] -= deltaSand;
			sediment += deltaSand;

			constexpr float sandThreshold{ 0.01f };

			float deltaBedrock{ glm::min(glm::max(1.0f - height[SAND] / sandThreshold, 0.0f) * simulation.bedrockDissolvingConstant * (sedimentCapacity - sediment) * simulation.deltaTime, height[BEDROCK] - floor) };
			deltaBedrock = glm::min(deltaBedrock, (sedimentCapacity - sediment));
			height[BEDROCK] -= deltaBedrock;
			sediment += deltaBedrock;
		}
		else
		{
			float deltaSediment{ glm::min(simulation.sedimentDepositionConstant * (sediment - sedimentCapacity) * simulation.deltaTime,  height[CEILING] - height[BEDROCK] - height[SAND] - height[WATER]) };
			deltaSediment = glm::min(deltaSediment, (sediment - sedimentCapacity));
			sediment -= deltaSediment;
			height[SAND] += deltaSediment;
		}

		simulation.heights[flatIndex] = glm::cuda_cast(height);
		simulation.sediments[flatIndex] = sediment;

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
			simulation.damages[above] = 0.0f;
		}

		simulation.heights[above] = simulation.heights[below];
		simulation.fluxes[above] = simulation.fluxes[below];
		simulation.stability[above] = FLT_MAX;
		simulation.sediments[above] = simulation.sediments[below];
		simulation.damages[above] = 0.0f;

		simulation.heights[below] = float4{ glm::max(split.y - damage, split.x), 0.0f, 0.0f, split.y };
		simulation.fluxes[below] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.stability[below] = FLT_MAX;
		simulation.sediments[below] = 0.0f;
		simulation.damages[below] = 0.0f;

		++layer;
	}

	simulation.layerCounts[flatBase] = layerCount;
}

void horizontalErosion(const Launch& launch)
{
	//CU_CHECK_KERNEL(horizontalErosionKernel<<<launch.gridSize, launch.blockSize>>>());
	CU_CHECK_KERNEL(sedimentKernel<<<launch.gridSize, launch.blockSize>>>());
	//CU_CHECK_KERNEL(damageKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
