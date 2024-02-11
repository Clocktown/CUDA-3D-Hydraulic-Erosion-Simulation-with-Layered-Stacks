#include "terrain.hpp"
#include "../components/terrain.hpp"
#include "../device/simulation.hpp"

namespace geo
{

template<typename ...Includes, typename ...Excludes>
void updateTerrains(const entt::exclude_t<Excludes...> excludes)
{
	onec::World& world{ onec::getWorld() };

	const auto view{ world.getView<Terrain, Includes...>(excludes) };
	
	for (const entt::entity entity : view)
	{
		Terrain& terrain{ view.get<Terrain>(entity) };
		onec::Buffer layerCountBuffer{ terrain.layerCountBuffer };
		onec::Buffer heightBuffer{ terrain.heightBuffer };

		device::Launch launch;
		device::Simulation simulation;

		simulation.gridSize = terrain.gridSize;
		simulation.gridScale = terrain.gridScale;
		simulation.rGridScale = 1.0f / simulation.gridScale;
		simulation.maxLayerCount = terrain.maxLayerCount;
		simulation.cellCount = simulation.gridSize.x * simulation.gridSize.y;

		simulation.voxelCount = terrain.maxLayerCount * simulation.cellCount;
		simulation.deltaTime = terrain.settings.deltaTime;
		simulation.gravity = terrain.settings.gravity;
		simulation.rain = terrain.settings.rain;
		simulation.evaporation = terrain.settings.evaporation;

		simulation.layerCounts = reinterpret_cast<int*>(layerCountBuffer.getData());
		simulation.heights = reinterpret_cast<float4*>(heightBuffer.getData());

		launch.blockSize = dim3{ 8, 8, 1 };
		launch.gridSize.x = (simulation.gridSize.x + launch.blockSize.x - 1) / launch.blockSize.x;
		launch.gridSize.y = (simulation.gridSize.y + launch.blockSize.y - 1) / launch.blockSize.y;
		launch.gridSize.z = 1;

		device::setSimulation(simulation);

		if (terrain.settings.init)
		{
			device::init(launch);
			terrain.settings.init = false;
		}

		if (!terrain.settings.paused)
		{
		    device::rain(launch);
		    device::evaporation(launch);
	    }
	}
}

}
