#pragma once

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

	float deltaTime;
	float gravity;
	float rain;
	Array3D<char4> infoArray;
	Array3D<float4> heightArray;
};

}
}
