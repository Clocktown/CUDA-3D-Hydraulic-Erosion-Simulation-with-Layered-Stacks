#include "kernels.hpp"
#include "grid.hpp"
#include <onec/config/cu.hpp>

namespace geo
{
namespace device
{

__global__ void rainKernel(Simulation simulation)
{
	const glm::ivec3 index{ getGlobalIndex3D() };
	const glm::ivec3 stride{ getGridStride3D() };

	glm::ivec3 cell;

	for (cell.x = index.x; cell.x < simulation.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < simulation.gridSize.y; cell.y += stride.y)
		{
			cell.z = findTopLayer(glm::ivec3{ cell.x, cell.y, 0 }, simulation);

			glm::vec4 height{ simulation.heightSurface.read<glm::vec4>(cell) };
			height[waterIndex] += 0.001f * simulation.rain.amount * simulation.gridScale * simulation.gridScale * simulation.deltaTime;

			simulation.heightSurface.write(cell, height);
		}
	}
}

void rain(const Launch& launch, const Simulation& simulation)
{
	rainKernel<<<launch.gridStride2D.gridSize, launch.gridStride2D.blockSize>>>(simulation);
}

}
}
