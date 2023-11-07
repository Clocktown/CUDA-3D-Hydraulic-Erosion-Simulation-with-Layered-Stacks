#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Terrain
{
	glm::ivec2 gridSize{ 1024 };
	float gridScale{ 1.0f };
	int maxLayerCount{ 8 };
	onec::Texture heightMap; // RGBA32F: Bedrock, Sand, Water, Max
	onec::Texture waterVelocityMap; // RG32F: Velocity
	onec::Texture indexMap; // RGInt: Below, Above
};

}
