#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	float deltaTime{ 1.0f / 60.0f };
	float gravity{ -9.81f };
	float rain{ 0.0f };
	float evaporation{ 0.0f };

	float bedrockDensity{ 2600.f }; // kg/m�
	float sandDensity{ 1600.f }; // kg/m�
	float waterDensity{ 1000.f }; // kg/m�
	// Water weight will be ignored since water level is constantly changing
	float bedrockSupport{10000.f}; // kg/m� - how much weight a given surface of bedrock can support
	float borderSupport{20000.f}; // kg/m�
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
	onec::Buffer pipeBuffer;
	onec::Buffer fluxBuffer;
	Simulation simulation;
};

}
