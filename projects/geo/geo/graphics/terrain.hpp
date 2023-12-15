#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Terrain
{
	explicit Terrain(glm::ivec3 gridSize = glm::ivec3{ 256, 256, 8 }, float gridScale = 1.0f);

	glm::ivec3 gridSize;
	float gridScale;
	onec::Texture infoMap;
	onec::Texture heightMap;
};

}
