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
		float splitHeight{ 0.0f };

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

				const float overlapCeiling{ glm::min(height[BEDROCK], neighbor.water) };
				const float overlapFloor{ glm::max(floor, neighbor.sand) };
				const float overlapArea{ (overlapCeiling - overlapFloor) * simulation.gridScale };

				if (overlapArea > 0.0f)
				{
					const float speed{ simulation.speeds[neighbor.flatIndices[i]] };

					erosions[i] = overlapArea * simulation.bedrockDissolvingConstant * speed * integrationScale;
					
					if (erosions[i] > maxErosion)
					{
						maxErosion = erosions[i];
						splitHeight = overlapCeiling;
					}

					break;
				}
			}
		}

		const float totalErosion{ erosions.x + erosions.y + erosions.z + erosions.w };
		erosions *= glm::min((splitHeight - floor - damage) / (totalErosion + glm::epsilon<float>()), 1.0f);

		for (int i{ 0 }; i < 4; ++i)
		{
			if (erosions[i] > 0.0f)
			{
				atomicAdd(simulation.sediments + neighbor.flatIndices[i], erosions[i]);
			}
		}

		damage += totalErosion;

		simulation.splitHeights[flatIndex] = splitHeight;
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

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		float sediment{ simulation.sediments[flatIndex] };
		const float slope{ glm::max(simulation.slopes[flatIndex], simulation.minTerrainSlope) };
		const float speed{ simulation.speeds[flatIndex] };
		const float sedimentCapacity{ simulation.sedimentCapacityConstant * slope * speed };
		
		if (sedimentCapacity > sediment)
		{
			const float deltaSand{ glm::min(simulation.sandDissolvingConstant * (sedimentCapacity - sediment) * simulation.deltaTime, height[SAND]) };
			height[SAND] -= deltaSand;
			sediment += deltaSand;
		}
		else
		{
			const float deltaSediment{ glm::min(simulation.sedimentDepositionConstant * (sediment - sedimentCapacity) * simulation.deltaTime,  height[CEILING] - height[BEDROCK] - height[WATER]) };
			sediment -= deltaSediment;
			height[SAND] += deltaSediment;
		}

		simulation.heights[flatIndex] = glm::cuda_cast(height);
		simulation.sediments[flatIndex] = sediment;
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
	float floor{ 0.0f };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const float damage{ simulation.damages[flatIndex] };

		if (damage / (height[BEDROCK] + glm::epsilon<float>()) <= 0.75f || layerCount == simulation.maxLayerCount)
		{
			floor = height[CEILING];
			continue;
		}

		const float splitHeight{ simulation.splitHeights[flatIndex] };

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

		simulation.heights[above] = glm::cuda_cast(height);
		simulation.fluxes[above] = simulation.fluxes[below];
		simulation.stability[above] = simulation.stability[below];
		simulation.sediments[above] = simulation.sediments[below];
		simulation.damages[above] = 0.0f;

		simulation.heights[below] = float4{ glm::max(splitHeight - damage, 0.0f), 0.0f, 0.0f, splitHeight };
		simulation.fluxes[below] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
		simulation.stability[below] = FLT_MAX;
		simulation.sediments[below] = 0.0f;
		simulation.damages[below] = 0.0f;

		floor = height[CEILING];
		++layer;
	}

	simulation.layerCounts[flatBase] = layerCount;
}

void horizontalErosion(const Launch& launch)
{
	CU_CHECK_KERNEL(horizontalErosionKernel<<<launch.gridSize, launch.blockSize>>>());
	//CU_CHECK_KERNEL(sedimentKernel<<<launch.gridSize, launch.blockSize>>>());
	CU_CHECK_KERNEL(damageKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
