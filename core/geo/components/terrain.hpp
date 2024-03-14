#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	float deltaTime{ 1.0f / 60.0f }; // [s]
	float gravity{ -9.81f }; // [m/s²]
	float rain{ 0.0f }; // [m/(m²s)]
	float evaporation{ 0.0f }; // [%/s]

	float sedimentCapacityConstant{ 0.001f };
	float dissolvingConstant{ 0.01f };
	float depositionConstant{ 0.01f };
	float minTerrainAngle{ glm::radians(5.0f) }; // [rad]

	float bedrockDensity{ 2600.f }; // [kg/m³]
	float sandDensity{ 1600.f }; // [kg/m³]
	float waterDensity{ 1000.f }; // [kg/m³]
	// Water weight will be ignored since water level is constantly changing
	float bedrockSupport{ 10000.f }; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport{ 20000.f }; // [kg/m²]
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

	glm::ivec2 gridSize;
	float gridScale;
	onec::GraphicsBuffer layerCountBuffer;
	onec::GraphicsBuffer heightBuffer;
	onec::GraphicsBuffer sedimentBuffer;
	onec::GraphicsBuffer stabilityBuffer;
	onec::Buffer pipeBuffer;
	onec::Buffer slopeBuffer;
	onec::Buffer fluxBuffer;
	Simulation simulation;
};

}
