#include "simulation.hpp"

namespace geo
{
namespace device
{

__global__ void pipeKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	const float gridScale2{ 2.0f * simulation.gridScale };

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		const glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const float sand{ height[BEDROCK] + height[SAND] };
		const float water{ sand + height[WATER] };

		glm::vec<4, char> pipe{ -1 };
		glm::vec4 heights{ sand };
		glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };
		glm::vec4 slippage{ 0.0f };

		struct
		{
			glm::ivec2 index;
			int flatIndex;
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
					const float crossSectionalArea{ simulation.gridScale * simulation.gridScale }; // dynamic?

					pipe[i] = static_cast<char>(neighbor.layer);
					heights[i] = neighbor.sand;
					flux[i] = (deltaHeight > 0.0f) *
						      glm::max(flux[i] - simulation.deltaTime * crossSectionalArea * simulation.gravity * deltaHeight * simulation.rGridScale, 0.0f);

					if (neighbor.height[CEILING] < FLT_MAX)
					{
						const float freeSpace{ (neighbor.height[CEILING] - neighbor.water) * simulation.gridScale * simulation.gridScale };
						const float takenSpace{ flux[i] * simulation.deltaTime };

						flux[i] *= glm::min(freeSpace / (takenSpace + glm::epsilon<float>()), 1.0f);
					}

					const float avgWater{ 0.5f * (height[WATER] + neighbor.height[WATER]) };
					const float talusSlope{ glm::mix(simulation.dryTalusSlope, simulation.wetTalusSlope, glm::min(avgWater, 1.0f)) };

					slippage[i] = glm::max(simulation.deltaTime * (sand - neighbor.sand - talusSlope * simulation.gridScale), 0.0f);

					break;
				}
				else if (water < neighbor.height[CEILING])
				{
					break;
				}
			}
		}

		const glm::vec3 tangents[2]{ glm::normalize(glm::vec3{ gridScale2, heights[RIGHT] - heights[LEFT], 0.0f }),
									 glm::normalize(glm::vec3{ 0.0f, heights[UP] - heights[DOWN], gridScale2 }) };

		const glm::vec3 normal{ glm::cross(tangents[0], tangents[1]) };
		const float slope{ glm::sqrt(1.0f - normal.y * normal.y) }; // sin(alpha)

		flux *= glm::min(height[WATER] * simulation.gridScale * simulation.gridScale /
						 ((flux.x + flux.y + flux.z + flux.w) * simulation.deltaTime + glm::epsilon<float>()), 1.0f);

		slippage *= glm::min(height[SAND] / (slippage.x + slippage.y + slippage.z + slippage.w + glm::epsilon<float>()), 1.0f);

		simulation.pipes[flatIndex] = glm::cuda_cast(pipe);
		simulation.slopes[flatIndex] = slope;
		simulation.fluxes[flatIndex] = glm::cuda_cast(flux);
		simulation.slippages[flatIndex] = glm::cuda_cast(slippage);
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
	const int layerCount{ simulation.layerCounts[flatIndex] };

	const float integrationScale{ simulation.rGridScale * simulation.rGridScale * simulation.deltaTime };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		float sediment{ simulation.sediments[flatIndex] };
		glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };
		glm::vec4 sedimentFlux{ sediment * flux };
		glm::vec4 slippage{ glm::cuda_cast(simulation.slippages[flatIndex]) };

		struct
		{
			glm::ivec2 index;
			int flatIndex;
			int flatIndex4;
			int layerCount;
			int layer;
			float sediment;
			float flux;
			float slippage;
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
			
			const int direction{ (i + 2) % 4 };

			for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
			{
				neighbor.flatIndex4 = 4 * neighbor.flatIndex;

				if (reinterpret_cast<char*>(simulation.pipes)[neighbor.flatIndex4 + direction] == layer)
				{
					neighbor.sediment = simulation.sediments[neighbor.flatIndex];
					neighbor.flux = reinterpret_cast<float*>(simulation.fluxes)[neighbor.flatIndex4 + direction];
					neighbor.slippage = reinterpret_cast<float*>(simulation.slippages)[neighbor.flatIndex4 + direction];

					flux[i] -= neighbor.flux;
					sedimentFlux[i] -= neighbor.sediment * neighbor.flux;
					slippage[i] -= neighbor.slippage;
				}
			}
		}

		float avgWater{ height[WATER] };
		height[WATER] = glm::clamp(height[WATER] - integrationScale * (flux.x + flux.y + flux.z + flux.w), 0.0f, height[CEILING] - height[BEDROCK] - height[SAND]);
		height[WATER] = glm::max((1.0f - simulation.evaporation * simulation.deltaTime) * height[WATER], 0.0f);
		avgWater = 0.5f * (avgWater + height[WATER]);
		const glm::vec2 velocity{ glm::vec2(flux[RIGHT] - flux[LEFT], flux[UP] - flux[DOWN]) / (avgWater * simulation.gridScale + glm::epsilon<float>()) };

		height[SAND] = glm::clamp(height[SAND] - (slippage.x + slippage.y + slippage.z + slippage.w), 0.0f, height[CEILING] - height[BEDROCK] - height[WATER]);

		const float petrificationAmount{ glm::min(simulation.petrification * simulation.deltaTime * height[SAND], height[SAND]) };
		height[BEDROCK] += petrificationAmount;
		height[SAND] -= petrificationAmount;

		sediment = glm::max(sediment - integrationScale * (sedimentFlux.x + sedimentFlux.y + sedimentFlux.z + sedimentFlux.w), 0.0f);

		simulation.heights[flatIndex] = glm::cuda_cast(height);
		simulation.sediments[flatIndex] = sediment;
		simulation.velocities[flatIndex] = glm::cuda_cast(velocity);
	}
}

void transport(const Launch& launch)
{
	CU_CHECK_KERNEL(pipeKernel<<<launch.gridSize, launch.blockSize>>>());
	CU_CHECK_KERNEL(transportKernel<<<launch.gridSize, launch.blockSize>>>());
}

}
}
