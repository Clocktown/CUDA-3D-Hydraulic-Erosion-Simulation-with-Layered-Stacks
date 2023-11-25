#include "kernels.hpp"
#include "grid.hpp"
#include <onec/config/cu.hpp>

namespace geo
{
namespace device
{
namespace kernel
{

__global__ void initialize(Simulation simulation)
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
				surf3Dwrite(char4{ 0, 0, 0, 0 }, simulation.infoSurface, cell.x * static_cast<int>(sizeof(char4)), cell.y, cell.z);
				surf3Dwrite(float4{ 0.0f, 0.0f, 0.0f, 0.0f }, simulation.heightSurface, cell.x * static_cast<int>(sizeof(float4)), cell.y, cell.z);
			}
		}
	}
}

}

void initialize(const Launch& launch, const Simulation& simulation) 
{
	kernel::initialize<<<launch.gridStride3D.gridSize, launch.gridStride3D.blockSize>>>(simulation);
}

}
}
