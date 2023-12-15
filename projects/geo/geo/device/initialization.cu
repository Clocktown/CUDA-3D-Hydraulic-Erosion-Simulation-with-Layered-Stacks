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
		simulation.infoArray.write(cell, char4{ 0, 0, 0, 0 });
		simulation.heightArray.write(cell, float4{ 0.0f, 0.0f, 0.0f, 0.0f });
	}
}

void initialization(const Launch& launch, const Simulation& simulation) 
{
	initializationKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
