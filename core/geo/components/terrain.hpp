#pragma once

#include <onec/onec.hpp>

#include "../config/config.h"

namespace geo
{

struct Simulation
{
	float deltaTime{ 0.150f }; // [s]
	float gravity{ -9.81f }; // [m/s²]
	float rain{ 0.07f }; // [m/s]
	float evaporation{ 0.09f }; // [m/s]
	float petrification{ 0.0001f }; // [1/s]

	glm::vec4 sourceStrengths{ 0.f }; // [m/s]
	glm::vec2 sourceLocations[4]{ {0,0},{0,0},{0,0},{0,0} }; // index
	glm::vec4 sourceFlux[4]{ 
		{0.f, 0.f, 0.f, 0.f}, 
		{0.f, 0.f, 0.f, 0.f}, 
		{0.f, 0.f, 0.f, 0.f}, 
		{0.f, 0.f, 0.f, 0.f} };
	glm::vec4 sourceSize; // m, radius

	float sedimentCapacityConstant{ 0.1f };
	float bedrockDissolvingConstant{ 0.01f };
	float sandDissolvingConstant{ 0.05f };
	float sedimentDepositionConstant{ 0.04f };
	float minSlopeErosionScale{ 0.1f };
	float maxSlopeErosionScale{ 1.f };
	float erosionWaterMaxHeight{ 10.f }; // Use tanh(2.f * WaterHeight / erosionWaterMaxHeight) as tanh(2)  ~ 1
	float sandThreshold{ 0.01f };

	float dryTalusAngle{ glm::radians(30.0f) }; // [rad]
	float wetTalusAngle{ glm::radians(0.0f) }; // [rad]
	float slippageInterpolationRange{ 1.f };
	float verticalErosionSlopeFadeStart{ 0.6f }; // sin(alpha)
	float topErosionWaterScale{ 2.f };
	float evaporationEmptySpaceScale{ 10.f };

	float minHorizontalErosionSlope{ 0.9f }; // sin(alpha)
	float horizontalErosionStrength{ 0.1f };
	//float minSplitDamage{ 2.0f };
	float splitSize{ 0.75f };
	float damageRecovery{ 0.005f };

	float bedrockDensity{ 2600.f }; // [kg/m³]
	float sandDensity{ 1600.f }; // [kg/m³]
	float waterDensity{ 1000.f }; // [kg/m³]
	// Water weight will be ignored since water level is constantly changing
	float bedrockSupport{ 60000.f }; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport{ 60000.f }; // [kg/m²]
	float minBedrockThickness{ 1.f };
	float maxStabilityPropagationDistance{ 100.f }; // m
	int stabilityPropagationStepsPerIteration{ 5 };
	int currentStabilityStep{ 0 };
	int currentSimulationStep{ 0 };

	bool init{ true };
	bool paused{ true };
	bool stabilityWasReenabled{ false };
	bool verticalErosionEnabled{ true };
	bool horizontalErosionEnabled{ true };
	bool supportCheckEnabled{ true };
	bool slippageEnabled{ true };
	bool useWeightInSupportCheck{ true };
	bool useOutflowBorders{ false };
	bool useSlippageOutflowBorders{ false };
};

struct QuadTreeEntry {
	glm::ivec2 gridSize;
	float gridScale;
	float rGridScale;
	int maxLayerCount;
	int layerStride;
	onec::Buffer heightBuffer;
	onec::Buffer layerCountBuffer;
};

struct Terrain
{
	explicit Terrain() = default;
	explicit Terrain(glm::ivec2 gridSize, float gridScale, char layerCount = 8, const Simulation& settings = Simulation{});

	int numValidColumns;
	glm::ivec2 gridSize;
	float gridScale;
	char maxLayerCount;
	bool quadTreeDirty = true;
	glm::ivec2 windowSize;
	onec::Texture screenTexture;
	onec::Array integratedBRDF;
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
	int maxQuadTreeLevels;
	QuadTreeEntry quadTree[MAX_NUM_QUADTREE_LEVELS];
	Simulation simulation;

	// DEBUG
	onec::Buffer intersectionCountBuffer;
};

}
