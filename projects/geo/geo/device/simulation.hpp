#pragma once

#include "surface3d.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace geo
{
namespace device
{

constexpr int belowIndex{ 0 };
constexpr int aboveIndex{ 1 };
constexpr int bedrockIndex{ 0 };
constexpr int sandIndex{ 1 };
constexpr int waterIndex{ 2 };
constexpr int maxHeightIndex{ 3 };

struct Simulation
{
	struct Rain
	{
		float amount{ 0.0 };
	};

	glm::ivec3 gridSize;
	float gridScale;
	float rGridScale;
	int cellCount;
	int horizontalCellCount;
	float deltaTime;
	glm::vec3 gravity;
	Rain rain;

	Surface3D<char4> infoSurface;
	Surface3D<float4> heightSurface;
	Surface3D<float2> waterVelocitySurface;
};

}
}
