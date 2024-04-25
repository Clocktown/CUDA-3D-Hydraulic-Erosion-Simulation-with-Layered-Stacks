#pragma once

#include <onec/config/cu.hpp>
#include <onec/config/glm.hpp>
#include <onec/utility/launch.hpp>
#include <onec/utility/grid.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <float.h>

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define CEILING 3

#define RIGHT 0
#define UP 1
#define LEFT 2
#define DOWN 3

namespace geo
{
namespace device
{

using namespace onec;

struct Launch
{
	dim3 gridSize;
	dim3 blockSize;
};

struct Simulation
{
	glm::ivec2 gridSize;
	float gridScale;
	float rGridScale;

	int maxLayerCount;
	int layerStride;

	float deltaTime; // [s]
	float gravity; // [m/s²]
	float rain; // [m/(m²s)]
	float evaporation; // [1/s]
	float petrification; // [1/s]                                                                                                                                                                                           

	float sedimentCapacityConstant;
	float bedrockDissolvingConstant;
	float sandDissolvingConstant;
	float sedimentDepositionConstant;
	float minTerrainSlope; // sin(alpha)
	float dryTalusSlope; // tan(alpha)
	float wetTalusSlope; // tan(alpha)

	float minErosionArea;
	float erosionStrength;
	float minSplitDamage;
	float splitThreshold;

	float bedrockDensity; // [kg/m³]
	float sandDensity; // [kg/m³]
	float waterDensity; // [kg/m³]
	float bedrockSupport; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport; // [kg/m²]
	float pad; // I know this isn't std430 opengl, but...

	char* layerCounts;
	float4* heights;
	float* sediments;
	int* indices;

	int* atomicCounter;
	char4* pipes;
	float* slopes; // sin(alpha)
	float4* fluxes;

	union
	{
		float4* slippages;
		float2* splits; 
	};
	
	float* speeds;
	float* stability;
	float* damages;
};

extern __constant__ Simulation simulation;
extern __constant__ int2 offsets[4];

void setSimulation(const Simulation& simulation);

int fillIndices(const Launch& launch, int* atomicCounter, int* indices);

void init(const Launch& launch);
void rain(const Launch& launch);
void transport(const Launch& launch);
void horizontalErosion(const Launch& launch);

void stepSupportCheck(const Launch& launch);
void startSupportCheck(const Launch& launch);
void endSupportCheck(const Launch& launch);

}
}
