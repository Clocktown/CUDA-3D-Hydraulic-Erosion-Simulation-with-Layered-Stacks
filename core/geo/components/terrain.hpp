#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	float deltaTime{ 0.150f }; // [s]
	float gravity{ -9.81f }; // [m/s²]
	float rain{ 0.07f }; // [m/(m²s)]
	float evaporation{ 0.01f }; // [1/s]
	float petrification{ 0.0005f }; // [1/s]

	float sedimentCapacityConstant{ 0.1f };
	float bedrockDissolvingConstant{ 0.01f };
	float sandDissolvingConstant{ 0.05f };
	float sedimentDepositionConstant{ 0.04f };
	float minSlopeErosionScale{ 0.1f };
	float maxSlopeErosionScale{ 1.f };
	float erosionWaterMaxHeight{ 10.f }; // Use tanh(2.f * WaterHeight / erosionWaterMaxHeight) as tanh(2)  ~ 1
	float sandThreshold{ 0.01f };

	float dryTalusAngle{ glm::radians(30.0f) }; // [rad]
	float wetTalusAngle{ glm::radians(10.0f) }; // [rad]
	float slippageInterpolationRange{ 1.f };

	float minHorizontalErosion{ 0.5f };
	float horizontalErosionStrength{ 0.1f };
	float minSplitDamage{ 2.0f };
	float splitThreshold{ 0.75f };

	float bedrockDensity{ 2600.f }; // [kg/m³]
	float sandDensity{ 1600.f }; // [kg/m³]
	float waterDensity{ 1000.f }; // [kg/m³]
	// Water weight will be ignored since water level is constantly changing
	float bedrockSupport{ 4000.f }; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport{ 4000.f }; // [kg/m²]
	float minBedrockThickness{ 0.f };
	int maxStabilityPropagationSteps{ 100 };
	int stabilityPropagationStepsPerIteration{ 10 };
	int currentStabilityStep{ 0 };
	int currentSimulationStep{ 0 };

	bool init{ true };
	bool paused{ true };
	bool verticalErosionEnabled{ true };
	bool horizontalErosionEnabled{ true };
	bool supportCheckEnabled{ true };
	bool slippageEnabled{ true };
	bool useWeightInSupportCheck{ true };
};

struct Terrain
{
	explicit Terrain() = default;
	explicit Terrain(glm::ivec2 gridSize, float gridScale, char layerCount = 8, const Simulation& settings = Simulation{});

	int numValidColumns;
	glm::ivec2 gridSize;
	float gridScale;
	char maxLayerCount;
	onec::GraphicsBuffer layerCountBuffer;
	onec::GraphicsBuffer heightBuffer;
	onec::GraphicsBuffer sedimentBuffer;
	onec::GraphicsBuffer stabilityBuffer;
	onec::GraphicsBuffer indicesBuffer;
	onec::Buffer atomicCounter;
	onec::Buffer pipeBuffer;
	onec::Buffer slopeBuffer;
	onec::Buffer fluxBuffer;
	onec::Buffer slippageBuffer;
	onec::Buffer velocityBuffer;
	onec::Buffer damageBuffer;
	onec::Buffer sedimentFluxScaleBuffer;
	Simulation simulation;
};

}
