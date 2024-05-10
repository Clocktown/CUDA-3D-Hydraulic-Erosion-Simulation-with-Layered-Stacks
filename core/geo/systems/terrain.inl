#include "terrain.hpp"
#include "../components/terrain.hpp"
#include "../device/simulation.hpp"
#include "../components/point_renderer.hpp"

namespace geo
{

template<typename ...Includes, typename ...Excludes>
void updateTerrains(const entt::exclude_t<Excludes...> excludes)
{
	onec::World& world{ onec::getWorld() };

	const auto view{ world.getView<Terrain, PointRenderer, Includes...>(excludes) };
	
	for (const entt::entity entity : view)
	{
		Terrain& terrain{ view.get<Terrain>(entity) };
		onec::Buffer layerCountBuffer{ terrain.layerCountBuffer };
		onec::Buffer heightBuffer{ terrain.heightBuffer };
		onec::Buffer sedimentBuffer{ terrain.sedimentBuffer };
		onec::Buffer stabilityBuffer{ terrain.stabilityBuffer };
		onec::Buffer indicesBuffer{ terrain.indicesBuffer };

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
		simulation.petrification = terrain.simulation.petrification;

		simulation.sedimentCapacityConstant = terrain.simulation.sedimentCapacityConstant;
		simulation.bedrockDissolvingConstant = terrain.simulation.bedrockDissolvingConstant;
		simulation.sandDissolvingConstant = terrain.simulation.sandDissolvingConstant;
		simulation.sedimentDepositionConstant = terrain.simulation.sedimentDepositionConstant;
		simulation.minTerrainSlope = glm::sin(terrain.simulation.minTerrainAngle);
		simulation.dryTalusSlope = glm::tan(terrain.simulation.dryTalusAngle);
		simulation.wetTalusSlope = glm::tan(terrain.simulation.wetTalusAngle);

		simulation.minHorizontalErosion = terrain.simulation.minHorizontalErosion;
		simulation.horizontalErosionStrength = terrain.simulation.horizontalErosionStrength;
		simulation.minSplitDamage = terrain.simulation.minSplitDamage;
		simulation.splitThreshold = terrain.simulation.splitThreshold;

		simulation.bedrockDensity = terrain.simulation.bedrockDensity;
		simulation.sandDensity = terrain.simulation.sandDensity;
		simulation.waterDensity = terrain.simulation.waterDensity;
		simulation.bedrockSupport = terrain.simulation.bedrockSupport;
		simulation.borderSupport = terrain.simulation.borderSupport;

		simulation.layerCounts = reinterpret_cast<char*>(layerCountBuffer.getData());
		simulation.heights = reinterpret_cast<float4*>(heightBuffer.getData());
		simulation.sediments = reinterpret_cast<float*>(sedimentBuffer.getData());
		simulation.stability = reinterpret_cast<float*>(stabilityBuffer.getData());
		simulation.indices = reinterpret_cast<int*>(indicesBuffer.getData());
		simulation.atomicCounter = reinterpret_cast<int*>(terrain.atomicCounter.getData());
		simulation.pipes = reinterpret_cast<char4*>(terrain.pipeBuffer.getData());
		simulation.slopes = reinterpret_cast<float*>(terrain.slopeBuffer.getData());
		simulation.fluxes = reinterpret_cast<float4*>(terrain.fluxBuffer.getData());
		simulation.slippages = reinterpret_cast<float4*>(terrain.slippageBuffer.getData());
		simulation.velocities = reinterpret_cast<float2*>(terrain.velocityBuffer.getData());
		simulation.damages = reinterpret_cast<float*>(terrain.damageBuffer.getData());

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
			device::horizontalErosion(launch);

			if (terrain.simulation.currentStabilityStep >= terrain.simulation.maxStabilityPropagationSteps) 
			{
				// Uncomment this to test static support check
				device::endSupportCheck(launch);

				// TODO: uncomment this. Disabled to test static initial stability
				device::startSupportCheck(launch);
				terrain.simulation.currentStabilityStep = 0;
			}

			for (int i = 0; i < terrain.simulation.stabilityPropagationStepsPerIteration; ++i) 
			{
				terrain.simulation.currentStabilityStep++;
				device::stepSupportCheck(launch);
			}
	    }

		terrain.numValidColumns = device::fillIndices(launch, simulation.atomicCounter, simulation.indices);
		PointRenderer& pointRenderer{ view.get<PointRenderer>(entity) };
		pointRenderer.count = terrain.numValidColumns;
	}
}

}
