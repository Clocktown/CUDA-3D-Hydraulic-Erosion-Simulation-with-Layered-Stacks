#include "simulation.hpp"
#include "../device/kernels.hpp"
#include <onec/onec.hpp>

namespace geo
{

Simulation::Simulation(const std::shared_ptr<Terrain>& terrain) :
	terrain{ terrain }
{
	const glm::ivec3 gridSize{ terrain->gridSize };
	const float gridScale{ terrain->gridScale };
	const int horizontalCellCount{ gridSize.x * gridSize.y };

	data.gridSize = gridSize;
	data.gridScale = gridScale;
	data.rGridScale = 1.0f / gridScale;
	data.horizontalCellCount = horizontalCellCount;
	data.cellCount = horizontalCellCount * gridSize.z;

	launch.gridSize.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(gridSize.x) / static_cast<float>(launch.blockSize.x)));
	launch.gridSize.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(gridSize.y) / static_cast<float>(launch.blockSize.y)));

	outflowArray.initialize(gridSize, cudaCreateChannelDesc<float4>(), cudaArrayDefault, nullptr, true);
	velocityArray.initialize(gridSize, cudaCreateChannelDesc<float2>(), cudaArrayDefault, nullptr, true);
	data.outflowArray.surfaceObject = outflowArray.getSurfaceObject();
	data.velocityArray.surfaceObject = velocityArray.getSurfaceObject();

	map();

	device::initialization(launch, data);

	unmap();
}

void Simulation::map()
{
	infoArray.initialize(terrain->infoMap, nullptr, true);
	heightArray.initialize(terrain->heightMap, nullptr, true);

	data.infoArray.surfaceObject = infoArray.getSurfaceObject();
	data.heightArray.surfaceObject = heightArray.getSurfaceObject();
}

void Simulation::unmap()
{
	infoArray.release();
	heightArray.release();
}

}
