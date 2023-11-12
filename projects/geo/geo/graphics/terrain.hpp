#pragma once

#include <onec/onec.hpp>

namespace geo
{
                                  
struct Terrain
{
	glm::ivec2 gridSize{ 256 };
	float gridScale{ 1.0f };
	int maxLayerCount{ 4 };
	onec::Texture infoMap; // RGBA8: Below, Above, State, Tmp
	onec::Texture heightMap; // RGBA32F: Bedrock, Sand, Water, Max
	onec::Texture waterVelocityMap; // RG32F: Velocity
};

}
