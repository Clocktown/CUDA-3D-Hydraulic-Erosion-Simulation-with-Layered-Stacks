#include "kernels.hpp"
#include "common.hpp"
#include <onec/config/cu.hpp>
#include <onec/cu/launch.hpp>
#include <onec/mathematics/grid.hpp>

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
	const glm::ivec2 index{ onec::cu::getGlobalIndex() };

	if (onec::isOutside(index, simulation.gridSize))
	{
		return;
	}

	glm::ivec3 cell{ index, 0 };

	cell.z = findTopLayer(cell, simulation);

	glm::vec4 height{ simulation.heightArray.read<glm::vec4>(cell) };
	height[WATER] += simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime;

	simulation.heightArray.write(cell, height);
}

void rain(const Launch& launch, const Simulation& simulation)
{
	rainKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
