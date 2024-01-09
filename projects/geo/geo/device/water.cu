#include "kernels.hpp"
#include "common.hpp"
#include <onec/config/cu.hpp>
#include <onec/cuda/launch.hpp>
#include <onec/utility/grid.hpp>
#include <glm/gtc/constants.hpp>

namespace geo
{
namespace device
{

__global__ void rainKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ onec::cu::getLaunchIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	glm::int8 z;

	do
	{
		z = index.z;
		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);

	index.z = z;

	glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
	height[WATER] += simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime;

	simulation.heightArray.write(index, height);
}

__global__ void fluxKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ onec::cu::getLaunchIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	do
	{
		const glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
		const float solidHeight{ height[BEDROCK] + height[SAND] };
		const float totalHeight{ solidHeight + height[WATER] };
		glm::i8vec4 flow;
		glm::vec4 flux{ simulation.fluxArray.read<glm::vec4>(index) };
		float totalFlux{ 0.0f };

		struct Neighbor
		{
			glm::ivec3 index;
			glm::vec4 height;
			float solidHeight;
			float totalHeight;
		};

		const Neighborhood neighborhood;
		Neighbor neighbor;

		for (int i{ 0 }; i < neighborhood.count; ++i)
		{
			neighbor.index.x = index.x + neighborhood.offsets[i].x;
			neighbor.index.y = index.y + neighborhood.offsets[i].y;

			if (onec::isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
			{
				flow[i] = INVALID_INDEX;
				flux[i] = 0.0f;

				continue;
			}

			neighbor.index.z = 0;

			do
			{
				neighbor.height = simulation.heightArray.read<glm::vec4>(neighbor.index);
				neighbor.solidHeight = neighbor.height[BEDROCK] + neighbor.height[SAND];
				neighbor.totalHeight = neighbor.solidHeight + neighbor.height[WATER];

				if (totalHeight < neighbor.height[MAX_HEIGHT])
				{
					const float heightDifference{ totalHeight - neighbor.totalHeight };
					//const float heightDifference{ totalHeight - glm::max(neighbor.totalHeight, solidHeight) }; // 2013 Interaction with Dynamic Large Bodies in Efficient, Real-Time Water Simulation
					
					flow[i] = neighbor.index.z;
					flux[i] = (heightDifference > 0.0f) *
						      glm::max(flux[i] - heightDifference * simulation.gravity * simulation.gridScale * simulation.deltaTime, 0.0f);
					
					totalFlux += flux[i];
					
					break;
				}

				neighbor.index.z = simulation.infoArray.read<glm::i8vec4>(neighbor.index)[ABOVE];

				if (neighbor.index.z == INVALID_INDEX)
				{
					flow[i] = INVALID_INDEX;
					flux[i] = 0.0f;

					break;
				}
			}
			while (true);
		}

		flux *= glm::min(height[WATER] * simulation.gridScale * simulation.gridScale / ((totalFlux + glm::epsilon<float>()) * simulation.deltaTime), 1.0f);

		simulation.flowArray.write(index, flow);
		simulation.fluxArray.write(index, flux);

		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);
}

__global__ void waterKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ onec::cu::getLaunchIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	do
	{
		glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
		const glm::vec4 flux{ simulation.fluxArray.read<glm::vec4>(index) };
		float totalFlux{ -(flux[RIGHT] + flux[UP] + flux[LEFT] + flux[DOWN]) };

		struct Neighbor
		{
			glm::ivec3 index;
			glm::i8vec4 flow;
			glm::vec4 flux;
		};

		const Neighborhood neighborhood;
		Neighbor neighbor;

		for (int i{ 0 }; i < neighborhood.count; ++i)
		{
			neighbor.index.x = index.x + neighborhood.offsets[i].x;
			neighbor.index.y = index.y + neighborhood.offsets[i].y;

			if (onec::isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
			{
				continue;
			}

			const int direction{ (i + 2) % 4 };
			neighbor.index.z = 0;

			do
			{
				neighbor.flow = simulation.flowArray.read<glm::i8vec4>(neighbor.index);

				if (neighbor.flow[direction] == index.z)
				{
					neighbor.flux = simulation.fluxArray.read<glm::vec4>(neighbor.index);
					totalFlux += neighbor.flux[direction];
				}

				neighbor.index.z = simulation.infoArray.read<glm::i8vec4>(neighbor.index)[ABOVE];
			}
			while (neighbor.index.z != INVALID_INDEX);
		}
		
		totalFlux *= simulation.deltaTime * simulation.rGridScale * simulation.rGridScale;
		
		height[WATER] = glm::min(height[WATER] + totalFlux, height[MAX_HEIGHT] - height[BEDROCK] - height[SAND]);
		height[WATER] = glm::max((1.0f - simulation.evaporation * simulation.deltaTime) * height[WATER], 0.0f);

		simulation.heightArray.write(index, height);

		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);
}

void water(const Launch& launch, const Simulation& simulation)
{
	rainKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	fluxKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	waterKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
