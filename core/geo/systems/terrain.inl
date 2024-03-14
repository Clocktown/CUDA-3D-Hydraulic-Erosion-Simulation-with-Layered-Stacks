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
		onec::Buffer sedimentBuffer{ terrain.sedimentBuffer };
		onec::Buffer stabilityBuffer{ terrain.stabilityBuffer };

		device::Launch launch;
		device::Simulation simulation;

		simulation.gridSize = terrain.gridSize;
		simulation.gridScale = terrain.gridScale;
		simulation.rGridScale = 1.0f / simulation.gridScale;
		simulation.maxLayerCount = terrain.maxLayerCount;
		simulation.layerStride = simulation.gridSize.x * simulation.gridSize.y;

		simulation.deltaTime = terrain.simulation.deltaTime;
		simulation.gravity = terrain.simulation.gravity;
		simulation.rain = terrain.simulation.rain;
		simulation.evaporation = terrain.simulation.evaporation;

		simulation.sedimentCapacityConstant = terrain.simulation.sedimentCapacityConstant;
		simulation.dissolvingConstant = terrain.simulation.dissolvingConstant;
		simulation.depositionConstant = terrain.simulation.depositionConstant;
		simulation.minTerrainSlope = glm::sin(terrain.simulation.minTerrainAngle);

		simulation.bedrockDensity = terrain.simulation.bedrockDensity;
		simulation.sandDensity = terrain.simulation.sandDensity;
		simulation.waterDensity = terrain.simulation.waterDensity;
		simulation.bedrockSupport = terrain.simulation.bedrockSupport;
		simulation.borderSupport = terrain.simulation.borderSupport;

		simulation.layerCounts = reinterpret_cast<char*>(layerCountBuffer.getData());
		simulation.heights = reinterpret_cast<float4*>(heightBuffer.getData());
		simulation.sediments = reinterpret_cast<float*>(sedimentBuffer.getData());
		simulation.stability = reinterpret_cast<float*>(stabilityBuffer.getData());
		simulation.pipes = reinterpret_cast<char4*>(terrain.pipeBuffer.getData());
		simulation.slopes = reinterpret_cast<float*>(terrain.slopeBuffer.getData());
		simulation.fluxes = reinterpret_cast<float4*>(terrain.fluxBuffer.getData());

		launch.blockSize = dim3{ 8, 8, 1 };
		launch.gridSize.x = (simulation.gridSize.x + launch.blockSize.x - 1) / launch.blockSize.x;
		launch.gridSize.y = (simulation.gridSize.y + launch.blockSize.y - 1) / launch.blockSize.y;
		launch.gridSize.z = 1;

		device::setSimulation(simulation);

		if (terrain.simulation.init)
		{
			device::init(launch);
		
			terrain.simulation.currentStabilityStep = 0;
			device::startSupportCheck(launch);

			terrain.simulation.init = false;
		}

		if (!terrain.simulation.paused)
		{
		    device::rain(launch);
			device::transport(launch);
		    device::evaporation(launch);

			if (terrain.simulation.currentStabilityStep >= terrain.simulation.maxStabilityPropagationSteps) 
			{
				// Uncomment this to test static support check
				//device::endSupportCheck(launch);

				// TODO: uncomment this. Disabled to test static initial stability
				//device::startSupportCheck(launch);
			}

			for (int i = 0; i < terrain.simulation.stabilityPropagationStepsPerIteration; ++i) 
			{
				terrain.simulation.currentStabilityStep++;
				device::stepSupportCheck(launch);
			}

	    }
	}
}

}
