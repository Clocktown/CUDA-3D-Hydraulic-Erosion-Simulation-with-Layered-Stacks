#include "kernels.hpp"
#include "grid.hpp"
#include <onec/config/cu.hpp>

namespace geo
{
namespace device
{

__global__ void initializationKernel(Simulation simulation)
{
	const glm::ivec3 index{ getGlobalIndex3D() };
	const glm::ivec3 stride{ getGridStride3D() };

	glm::ivec3 cell;

	for (cell.x = index.x; cell.x < simulation.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < simulation.gridSize.y; cell.y += stride.y)
		{
			for (cell.z = index.z; cell.z < simulation.gridSize.z; cell.z += stride.z)
			{
				simulation.infoSurface.write(cell, char4{ 0, 0, 0, 0 });
				simulation.heightSurface.write(cell, float4{ 0.0f, 0.0f, 0.0f, 0.0f });
				simulation.waterVelocitySurface.write(cell, float2{ 0.0f, 0.0f });
			}
		}
	}
}

void initialization(const Launch& launch, const Simulation& simulation) 
{
	initializationKernel<<<launch.gridStride3D.gridSize, launch.gridStride3D.blockSize>>>(simulation);
}

}
}
