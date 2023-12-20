#include "kernels.hpp"
#include <onec/config/cu.hpp>
#include <onec/cu/launch.hpp>
#include <onec/mathematics/grid.hpp>

namespace geo
{
namespace device
{

__global__ void initializationKernel(Simulation simulation)
{
	const glm::ivec2 index{ onec::cu::getGlobalIndex() };

	if (onec::isOutside(index, simulation.gridSize))
	{
		return;
	}

	glm::ivec3 cell{ index, 0 };

	for (; cell.z < simulation.gridSize.z; ++cell.z)
	{
		simulation.infoArray.write(cell, glm::i8vec4{ 0 });
		simulation.heightArray.write(cell, glm::vec4{ index.x / 16, index.y / 16, 0.0f, 0.0f });
		simulation.outflowArray.write(cell, glm::vec4{ 0.0f });
	}
}

void initialization(const Launch& launch, const Simulation& simulation) 
{
	initializationKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
