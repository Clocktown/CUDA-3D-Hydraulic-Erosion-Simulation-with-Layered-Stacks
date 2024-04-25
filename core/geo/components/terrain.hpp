#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	float deltaTime{ 1.0f / 60.0f }; // [s]
	float gravity{ -9.81f }; // [m/s²]
	float rain{ 0.0f }; // [m/(m²s)]
	float evaporation{ 0.0f }; // [1/s]
	float petrification{ 0.0005f }; // [1/s]

	float sedimentCapacityConstant{ 0.001f };
	float bedrockDissolvingConstant{ 1.0f };
	float sandDissolvingConstant{ 60.0f };
	float sedimentDepositionConstant{ 60.0f };
	float minTerrainAngle{ glm::radians(5.0f) }; // [rad]

	float dryTalusAngle{ glm::radians(30.0f) }; // [rad]
	float wetTalusAngle{ glm::radians(15.0f) }; // [rad]

	float minErosionArea{ 0.5f };
	float erosionStrength{ 0.1f };
	float minSplitDamage{ 2.0f };
	float splitThreshold{ 0.75f };

	float bedrockDensity{ 2600.f }; // [kg/m³]
	float sandDensity{ 1600.f }; // [kg/m³]
	float waterDensity{ 1000.f }; // [kg/m³]
	// Water weight will be ignored since water level is constantly changing
	float bedrockSupport{ 4000.f }; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport{ 4000.f }; // [kg/m²]
	int maxStabilityPropagationSteps{ 100 };
	int stabilityPropagationStepsPerIteration{ 10 };
	int currentStabilityStep{ 0 };

	bool init{ true };
	bool paused{ true };
};

struct Terrain
{
	static constexpr int maxLayerCount{ 8 };

	explicit Terrain() = default;
	explicit Terrain(glm::ivec2 gridSize, float gridScale, const Simulation& settings = Simulation{});

	int numValidColumns;
	glm::ivec2 gridSize;
	float gridScale;
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
	onec::Buffer speedBuffer;
	onec::Buffer damageBuffer;
	Simulation simulation;
};

}
