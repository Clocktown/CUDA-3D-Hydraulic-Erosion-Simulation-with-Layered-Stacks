#include "terrain.hpp"
#include "../components/terrain.hpp"
#include "../device/simulation.hpp"
#include "../components/point_renderer.hpp"
#include "../singletons/ui.hpp"
#include <onec/resources/array.hpp>
#include <onec/components/camera.hpp>

namespace geo
{

template<typename ...Includes, typename ...Excludes>
void updateTerrains(const entt::exclude_t<Excludes...> excludes)
{
	onec::World& world{ onec::getWorld() };

	const auto view{ world.getView<Terrain, PointRenderer, onec::LocalToWorld, Includes...>(excludes) };
	const auto view2{ world.getView<onec::WorldToView, onec::ViewToClip, onec::Position, Includes...>(excludes) };

	auto ui = world.getSingleton<geo::UI>();
	
	for (const entt::entity entity : view)
	{
		Terrain& terrain{ view.get<Terrain>(entity) };
		onec::Buffer layerCountBuffer{ terrain.layerCountBuffer };
		onec::Buffer heightBuffer{ terrain.heightBuffer };
		onec::Buffer sedimentBuffer{ terrain.sedimentBuffer };
		onec::Buffer stabilityBuffer{ terrain.stabilityBuffer };
		onec::Buffer indicesBuffer{ terrain.indicesBuffer };

		if (ui->rendering.renderScene && ui->rendering.useRaymarching) {
			glm::ivec2 windowSize = onec::getWindow().getSize();
			if (windowSize != terrain.windowSize) {
				terrain.screenTexture.release();
				terrain.screenTexture.initialize(GL_TEXTURE_2D, glm::ivec3(windowSize.x, windowSize.y, 1), GL_RGBA8, 1, onec::SamplerState{}, true);
				terrain.windowSize = windowSize;
			}
		}

		onec::Array screenArray{ terrain.screenTexture };

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

		simulation.sourceStrengths = terrain.simulation.sourceStrengths;
		simulation.sourceSize = terrain.simulation.sourceSize;
		for (int i = 0; i < 4; ++i) {
			simulation.sourceLocations[i] = terrain.simulation.sourceLocations[i];
			simulation.sourceFlux[i] = terrain.simulation.sourceFlux[i];
		}

		simulation.sedimentCapacityConstant = terrain.simulation.sedimentCapacityConstant;
		simulation.bedrockDissolvingConstant = terrain.simulation.bedrockDissolvingConstant;
		simulation.sandDissolvingConstant = terrain.simulation.sandDissolvingConstant;
		simulation.sedimentDepositionConstant = terrain.simulation.sedimentDepositionConstant;
		simulation.minTerrainSlopeScale = terrain.simulation.minSlopeErosionScale;
		simulation.maxTerrainSlopeScale = terrain.simulation.maxSlopeErosionScale;
		simulation.dryTalusSlope = glm::tan(terrain.simulation.dryTalusAngle);
		simulation.wetTalusSlope = glm::tan(terrain.simulation.wetTalusAngle);
		simulation.iSlippageInterpolationRange = 1.f / terrain.simulation.slippageInterpolationRange;
		simulation.erosionWaterScale = 2.f / terrain.simulation.erosionWaterMaxHeight;
		simulation.iSandThreshold = 1.f / terrain.simulation.sandThreshold;
		simulation.verticalErosionSlopeFadeStart = terrain.simulation.verticalErosionSlopeFadeStart;
		simulation.iEvaporationEmptySpaceScale = 1.f / terrain.simulation.evaporationEmptySpaceScale;
		simulation.iTopErosionWaterScale = 1.f / terrain.simulation.topErosionWaterScale;

		simulation.minHorizontalErosionSlope = terrain.simulation.minHorizontalErosionSlope;
		simulation.horizontalErosionStrength = terrain.simulation.horizontalErosionStrength;
		//simulation.minSplitDamage = terrain.simulation.minSplitDamage;
		simulation.splitSize = terrain.simulation.splitSize;
		simulation.damageRecovery = terrain.simulation.damageRecovery;

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
		simulation.sedimentFluxScale = reinterpret_cast<float*>(terrain.sedimentFluxScaleBuffer.getData());
		simulation.minBedrockThickness = terrain.simulation.minBedrockThickness;
		if (terrain.simulation.init) terrain.simulation.currentSimulationStep = 0;
		simulation.step = terrain.simulation.currentSimulationStep;

		onec::ViewToClip pMatrix{ view2.get<onec::ViewToClip>(ui->camera.entity) };
		onec::WorldToView vMatrix{ view2.get<onec::WorldToView>(ui->camera.entity) };
		onec::LocalToWorld mMatrix { view.get<onec::LocalToWorld>(entity) };

		glm::mat4 iPV = glm::inverse(pMatrix.viewToClip * vMatrix.worldToView);
		glm::vec4 ll = iPV * glm::vec4(-1, -1, 1, 1);
		ll /= ll.w;
		glm::vec4 ul = iPV * glm::vec4(-1, 1, 1, 1);
		ul /= ul.w;
		glm::vec4 lr = iPV * glm::vec4(1, -1, 1, 1);
		lr /= lr.w;

		simulation.camPos = view2.get<onec::Position>(ui->camera.entity).position;
		simulation.lowerLeft = glm::vec3(ll);
		simulation.rightVec = glm::vec3(lr - ll) / float(terrain.windowSize.x);
		simulation.upVec = glm::vec3(ul - ll) / float(terrain.windowSize.y);
		simulation.windowSize = terrain.windowSize;
		simulation.screenSurface = screenArray.getSurfaceObject();

		launch.blockSize = dim3{ 8, 8, 1 };
		launch.gridSize.x = (simulation.gridSize.x + launch.blockSize.x - 1) / launch.blockSize.x;
		launch.gridSize.y = (simulation.gridSize.y + launch.blockSize.y - 1) / launch.blockSize.y;
		launch.gridSize.z = 1;

		device::setSimulation(simulation);

		if (terrain.simulation.init)
		{
			device::init(launch);
		
			terrain.simulation.currentStabilityStep = 0;
			if (terrain.simulation.supportCheckEnabled) {
				device::startSupportCheck(launch);
			}

			terrain.simulation.init = false;
		}

		auto& perf = ui->performance;

		if (!terrain.simulation.paused)
		{
			if (perf.measurePerformance && !perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Global Simulation"].start();

			if (perf.measureParts || perf.measureIndividualKernels) perf.measurements["Rain"].start();
		    device::rain(launch);
			if(perf.measureParts || perf.measureIndividualKernels) perf.measurements["Rain"].stop();

			if (perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Transport"].start();
			device::transport(launch, terrain.simulation.slippageEnabled, terrain.simulation.useOutflowBorders, terrain.simulation.useSlippageOutflowBorders, perf);
			if(perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Transport"].stop();

			if (perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Erosion"].start();
			device::erosion(launch, terrain.simulation.verticalErosionEnabled, terrain.simulation.horizontalErosionEnabled, perf);
			if(perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Erosion"].stop();

			if (terrain.simulation.stabilityWasReenabled) {
				device::startSupportCheck(launch);
				terrain.simulation.currentStabilityStep = 0;
				terrain.simulation.stabilityWasReenabled = false;
			}
			if (perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Support"].start();
			if (terrain.simulation.supportCheckEnabled) {
				if (2.f * terrain.simulation.currentStabilityStep >= (terrain.simulation.maxStabilityPropagationDistance * simulation.rGridScale))
				{
					if (perf.measureIndividualKernels) perf.measurements["End Support Check"].start();
					device::endSupportCheck(launch);
					if (perf.measureIndividualKernels) perf.measurements["End Support Check"].stop();

					if (perf.measureIndividualKernels) perf.measurements["Start Support Check"].start();
					device::startSupportCheck(launch);
					if (perf.measureIndividualKernels) perf.measurements["Start Support Check"].stop();
					terrain.simulation.currentStabilityStep = 0;
				}

				if (perf.measureIndividualKernels) perf.measurements["Step Support Check"].start();
				for (int i = 0; i < terrain.simulation.stabilityPropagationStepsPerIteration; ++i)
				{
					terrain.simulation.currentStabilityStep++;
					device::stepSupportCheck(launch, terrain.simulation.useWeightInSupportCheck);
				}
				if (perf.measureIndividualKernels) perf.measurements["Step Support Check"].stop();
			}
			if(perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Support"].stop();

			terrain.simulation.currentSimulationStep++;
			if (ui->performance.pauseAfterStepCount > 0 && (terrain.simulation.currentSimulationStep % ui->performance.pauseAfterStepCount) == 0) {
				terrain.simulation.paused = true;
			}
			if(perf.measurePerformance && !perf.measureParts && !perf.measureIndividualKernels) perf.measurements["Global Simulation"].stop();
	    }

		if (ui->rendering.renderScene && ui->rendering.useRaymarching) {
			glm::ivec2 windowSize = onec::getWindow().getSize();
			device::Launch screenLaunch;
			screenLaunch.blockSize = dim3{ 16, 16, 1 };
			screenLaunch.gridSize.x = (windowSize.x + screenLaunch.blockSize.x - 1) / screenLaunch.blockSize.x;
			screenLaunch.gridSize.y = (windowSize.y + screenLaunch.blockSize.y - 1) / screenLaunch.blockSize.y;
			screenLaunch.gridSize.z = 1;
			device::raymarchTerrain(screenLaunch);
		}

		if (ui->rendering.renderScene && !ui->rendering.useRaymarching) {
			if (perf.measureRendering) perf.measurements["Build Draw List"].start();
			terrain.numValidColumns = device::fillIndices(launch, simulation.atomicCounter, simulation.indices);
			if (perf.measureRendering) perf.measurements["Build Draw List"].stop();

			PointRenderer& pointRenderer{ view.get<PointRenderer>(entity) };
			pointRenderer.count = terrain.numValidColumns;
		}
	}
}

}
