#include "kernels.hpp"
#include "common.hpp"

namespace geo
{
namespace device
{

__global__ void verticalErosionKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ getLaunchIndex() }, 0 };

	if (isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	do
	{
		const glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
		const float totalHeight{ height[BEDROCK] + height[SAND] + height[WATER] };
		glm::vec4 flow{ simulation.flowArray.read<glm::vec4>(index) };

		struct Neighbor
		{
			glm::ivec3 index;
			glm::vec4 height;
			float totalHeight;
		};

		const Neighborhood neighborhood;
		float heights[neighborhood.count];

		for (int i{ 0 }; i < neighborhood.count; ++i)
		{
			Neighbor neighbor;
			neighbor.index.x = index.x + neighborhood.offsets[i].x;
			neighbor.index.y = index.y + neighborhood.offsets[i].y;

			if (isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
			{
				continue;
			}

			neighbor.index.z = 0;

			do
			{
				neighbor.height = simulation.heightArray.read<glm::vec4>(neighbor.index);
				neighbor.totalHeight = neighbor.height[BEDROCK] + neighbor.height[SAND] + neighbor.height[WATER];

				if (height[MAX_HEIGHT] <= neighbor.height[BEDROCK])
				{
					heights[i] = totalHeight;
					break;
				}

				if (height[BEDROCK] < neighbor.height[MAX_HEIGHT])
				{
					heights[i] = neighbor.totalHeight;
					break;
				}

				if (neighbor.index.z != INVALID_INDEX)
				{
					neighbor.index.z = simulation.infoArray.read<glm::i8vec4>(neighbor.index)[ABOVE];
				}
				else
				{
					heights[i] = totalHeight;
					break;
				}
			}
			while (true);
		}

		

		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);
}

void verticalErosion(const Launch& launch, const Simulation& simulation)
{
	//verticalErosionKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
