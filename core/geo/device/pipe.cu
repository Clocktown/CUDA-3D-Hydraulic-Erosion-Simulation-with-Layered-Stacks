#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void fluxKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const char layerCount{ simulation.layerCounts[flatIndex] };

	for (char layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		const glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const float sand{ height[BEDROCK] + height[SAND] };
		const float water{ sand + height[WATER] };
		glm::vec<4, char> pipe{ -1 };
		glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };

		struct Neighbor
		{
			glm::ivec2 index;
			int flatIndex;
			char layerCount;
			char layer;
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

			neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
			neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];

			for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
			{
				neighbor.height = glm::cuda_cast(simulation.heights[neighbor.flatIndex]);
				neighbor.sand = neighbor.height[BEDROCK] + neighbor.height[SAND];
				neighbor.water = neighbor.sand + neighbor.height[WATER];

				if (sand < neighbor.height[CEILING])
				{
					const float deltaHeight{ water - neighbor.water };

					pipe[i] = neighbor.layer;
					flux[i] = (deltaHeight > 0.0f) * 
						      fmaxf(flux[i] - deltaHeight * simulation.gravity * simulation.gridScale * simulation.deltaTime, 0.0f);

					break;
				}
			}
		}

		const float scale{ fminf(height[WATER] * simulation.gridScale * simulation.gridScale /
						   ((flux[RIGHT] + flux[UP] + flux[LEFT] + flux[DOWN] + glm::epsilon<float>()) * simulation.deltaTime), 1.0f) };

		flux *= scale;

		simulation.pipes[flatIndex] = glm::cuda_cast(pipe);
		simulation.fluxes[flatIndex] = glm::cuda_cast(flux);
	}
}

__global__ void transportKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const char layerCount{ simulation.layerCounts[flatIndex] };

	const int layerStride{ 4 * simulation.layerStride };
	const char* const pipes{ reinterpret_cast<char*>(simulation.pipes) };
	const float* const fluxes{ reinterpret_cast<float*>(simulation.fluxes) };
	
	for (char layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };
		float totalFlux{ -(flux[RIGHT] + flux[UP] + flux[LEFT] + flux[DOWN]) };

		struct Neighbor
		{
			glm::ivec2 index;
			int flatIndex;
			char layerCount;
			char layer;
		} neighbor;

		for (int i{ 0 }; i < 4; ++i)
		{
			neighbor.index = index + glm::cuda_cast(offsets[i]);

			if (isOutside(neighbor.index, simulation.gridSize))
			{
				continue;
			}

			neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
			neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];
			neighbor.flatIndex *= 4;

			const int direction{ (i + 2) % 4 };

			for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += layerStride)
			{
				if (pipes[neighbor.flatIndex + direction] == layer)
				{
					totalFlux += fluxes[neighbor.flatIndex + direction];
				}
			}
		}

		height[WATER] = fminf(height[WATER] + totalFlux * simulation.deltaTime * simulation.rGridScale * simulation.rGridScale, height[CEILING] - height[BEDROCK] - height[SAND]);

		simulation.heights[flatIndex].z = height[WATER];
	}
}

void pipe(const Launch& launch)
{
	CU_CHECK_KERNEL(fluxKernel<<<launch.gridSize, launch.blockSize>>>());
	CU_CHECK_KERNEL(transportKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
