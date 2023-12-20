#include "kernels.hpp"
#include "common.hpp"
#include <onec/config/cu.hpp>
#include <onec/cu/launch.hpp>
#include <onec/mathematics/grid.hpp>
#include <glm/gtc/constants.hpp>

namespace geo
{
namespace device
{

__forceinline__ __device__ int findTopLayer(const glm::ivec3 cell, const Simulation& simulation)
{
	int z{ cell.z };

	for (int i{ 0 }; i < simulation.gridSize.z; ++i)
	{
		const glm::i8vec4 info{ simulation.infoArray.read<glm::i8vec4>(cell) };

		if (info[ABOVE] <= 0)
		{
			break;
		}

		z = info[ABOVE];
	}

	return z;
}

__global__ void rainKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ onec::cu::getGlobalIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	index.z = findTopLayer(index, simulation);
	glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };

	height[WATER] += simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime;

	simulation.heightArray.write(index, height);
}

__global__ void outflowKernel(Simulation simulation)
{
	const glm::ivec3 index{ glm::ivec2{ onec::cu::getGlobalIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	const glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
	const float solidHeight{ height[BEDROCK] + height[SAND] };
	const float totalHeight{ solidHeight + height[WATER] };
	glm::vec4 outflow{ simulation.outflowArray.read<glm::vec4>(index) };
	float totalOutflow{ 0.0f };

	struct Neighbor
	{
		glm::ivec3 index;
		glm::vec4 height;
		float solidHeight;
		float totalHeight;
	};

	const Neighborhood neighborhood;
	Neighbor neighbor;
	neighbor.index.z = 0;

	for (int i{ 0 }; i < neighborhood.count; ++i)
	{
		neighbor.index.x = index.x + neighborhood.offsets[i].x;
		neighbor.index.y = index.y + neighborhood.offsets[i].y;

		if (onec::isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
		{
			outflow[i] = 0.0f;
			continue;
		}

		neighbor.height = simulation.heightArray.read<glm::vec4>(neighbor.index);
		neighbor.solidHeight = neighbor.height[BEDROCK] + neighbor.height[SAND];
		neighbor.totalHeight = neighbor.solidHeight + neighbor.height[WATER];

		const float heightDifference{ totalHeight - neighbor.totalHeight };

		outflow[i] = (heightDifference > 0.0f) *  
			         glm::max(outflow[i] - heightDifference * simulation.gravity * simulation.gridScale * simulation.deltaTime, 0.0f);
		totalOutflow += outflow[i];
	}

	outflow *= glm::min(height[WATER] * simulation.gridScale * simulation.gridScale / ((totalOutflow + glm::epsilon<float>()) * simulation.deltaTime), 1.0f);

	simulation.outflowArray.write(index, outflow);
}

__global__ void waterKernel(Simulation simulation)
{
	const glm::ivec3 index{ glm::ivec2{ onec::cu::getGlobalIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
	const glm::vec4 outflow{ simulation.outflowArray.read<glm::vec4>(index) };
	float flux{ -(outflow[RIGHT] + outflow[UP] + outflow[LEFT] + outflow[DOWN]) };

	struct Neighbor
	{
		glm::ivec3 index;
		glm::vec4 outflow;
	};

	const Neighborhood neighborhood;
	Neighbor neighbor;
	neighbor.index.z = 0;

	for (int i{ 0 }; i < neighborhood.count; ++i)
	{
		neighbor.index.x = index.x + neighborhood.offsets[i].x;
		neighbor.index.y = index.y + neighborhood.offsets[i].y;

		if (onec::isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
		{
			continue;
		}

		neighbor.outflow = simulation.outflowArray.read<glm::vec4>(neighbor.index);
		flux += neighbor.outflow[(i + 2) % 4];
	}

	flux *= simulation.deltaTime * simulation.rGridScale * simulation.rGridScale;

	height[WATER] += flux;
	height[WATER] = glm::max((1.0f - simulation.evaporation * simulation.deltaTime) * height[WATER], 0.0f);

	simulation.heightArray.write(index, height);
}

void water(const Launch& launch, const Simulation& simulation)
{
	rainKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	outflowKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	waterKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
