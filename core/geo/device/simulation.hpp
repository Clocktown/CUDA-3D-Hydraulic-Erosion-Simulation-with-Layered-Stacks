#pragma once

#include <onec/config/cu.hpp>
#include <onec/config/glm.hpp>
#include <onec/utility/launch.hpp>
#include <onec/utility/grid.hpp>
#include"../singletons/performance.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <float.h>
#include "../config/config.h"

#include <onec/resources/RenderPipelineUniforms.h>

#include <vector>

//#include "../components/terrain.hpp"

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define CEILING 3

#define QFULLHEIGHT 0
#define QCEILING 1
#define QSOLIDHEIGHT 2
#define QAIR 3

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

struct QuadTreeEntry {
	glm::ivec2 gridSize;
	float gridScale;
	float rGridScale;
	int maxLayerCount;
	int layerStride;
	float4* heights; // Different format: 2 floats for Solid + Water, excluding air, 2 floats for Solid + Air, excluding water. All Absolute.
	char* layerCounts;
};

struct Rendering {
	glm::ivec2 windowSize;
	glm::vec3 lowerLeft;
	glm::vec3 rightVec;
	glm::vec3 upVec;
	glm::vec3 camPos;
	glm::vec3 gridOffset;
	int missCount;
	int fineMissCount;
	float i_scale;
	cudaSurfaceObject_t screenSurface;
	cudaSurfaceObject_t integratedBRDFSurface;
	cudaTextureObject_t integratedBRDFTexture;

	glm::vec3 materialColors[4];
	bool renderSand;
	bool renderWater;
	float surfaceVolumePercentage;
	float smoothingRadiusInCells;
	float normalSmoothingFactor;
	float aoRadius;
	float rBoxSize2;
	float boxSize3;
	float rBoxSize3;
	float rAoBoxSize3;

	float softShadowScale;
	float maxSoftShadowDiv;

	bool accurateNormals;
	bool useExpensiveNormals;
	bool enableWaterAbsorption;
	bool useCheapAbsorption;
	bool enableReflections;
	bool enableRefraction;
	bool enableShadows;
	bool enableSoftShadows;
	bool fixLightLeaks;
	bool enableAO;
	bool enableShadowsInReflection;
	bool enableShadowsInRefraction;
	bool enableAOInReflection;
	bool enableAOInRefraction;

	onec::RenderPipelineUniforms* uniforms;
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
	float rain; // [m/s]
	float evaporation; // [m/s]
	float petrification; // [m/s]           
	glm::vec4 sourceStrengths; // [m/s]
	glm::vec2 sourceLocations[4]; // index
	glm::vec4 sourceFlux[4];
	glm::vec4 sourceSize; // m, radius

	float sedimentCapacityConstant;
	float bedrockDissolvingConstant;
	float sandDissolvingConstant;
	float sedimentDepositionConstant;
	float minTerrainSlopeScale;
	float maxTerrainSlopeScale;
	float dryTalusSlope; // tan(alpha)
	float wetTalusSlope; // tan(alpha)
	float iSlippageInterpolationRange;
	float erosionWaterScale; // 2.f / erosionWaterMaxHeight
	float iSandThreshold;
	float verticalErosionSlopeFadeStart; // sin(alpha)
	float iTopErosionWaterScale;
	float iEvaporationEmptySpaceScale;

	float minHorizontalErosionSlope; // sin(alpha)
	float horizontalErosionStrength;
	//float minSplitDamage;
	float splitSize;
	float damageRecovery;

	float bedrockDensity; // [kg/m³]
	float sandDensity; // [kg/m³]
	float waterDensity; // [kg/m³]
	float bedrockSupport; // [kg/m²] - how much weight a given surface of bedrock can support
	float borderSupport; // [kg/m²]
	float minBedrockThickness;
	int step; 

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
	
	float2* velocities;
	float* stability;
	float* damages;


	float* sedimentFluxScale;

	Rendering rendering;
	int maxQuadTreeLevels;
	QuadTreeEntry quadTree[geo::MAX_NUM_QUADTREE_LEVELS];

	// DEBUG
	unsigned int* intersectionCounts;
};

extern __constant__ Simulation simulation;
extern __constant__ float2 sceneBounds;
extern __constant__ int2 offsets[4];

void setSimulation(const Simulation& simulation);
void setSceneBounds(const float2& bounds);

int fillIndices(const Launch& launch, int* atomicCounter, int* indices);
char getMaxLayerCount(const char* layerCounts, std::ptrdiff_t count);

void init(const Launch& launch);
void rain(const Launch& launch);
void transport(const Launch& launch, bool enable_slippage, bool use_outflow_borders, bool use_slippage_outflow_borders, geo::performance& perf);
void erosion(const Launch& launch, bool enable_vertical, bool enable_horizontal, geo::performance& perf);

void stepSupportCheck(const Launch&, bool use_weight);
void startSupportCheck(const Launch& launch);
void endSupportCheck(const Launch& launch);

void integrateBRDF(const Launch& launch, glm::ivec2 size);
void raymarchTerrain(const Launch& launch, bool useInterpolation, int missCount = 8, int debugLayer = -2);
void buildQuadTree(const std::vector<Launch>& launch, const Simulation& sim, bool useInterpolation);
}
}
