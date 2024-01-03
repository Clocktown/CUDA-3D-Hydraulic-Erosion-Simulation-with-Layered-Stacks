#pragma once

#include "buffer.hpp"
#include "array.hpp"
#include <glm/glm.hpp>

namespace geo
{
namespace device
{

struct Simulation
{
	glm::ivec3 gridSize;
	float gridScale;
	float rGridScale;
	int horizontalCellCount;
	int cellCount;
	Array3D<char4> infoArray;
	Array3D<float4> heightArray;
	Array3D<char4> flowArray;
	Array3D<float4> fluxArray;
	Array3D<float2> velocityArray;

	float deltaTime{ 0.5f };
	float gravity{ -9.81f };
	float rain{ 0.0f };
	float evaporation{ 0.0f };
};

}
}
