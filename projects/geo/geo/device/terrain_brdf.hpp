#pragma once

#include <glm/glm.hpp>

namespace geo
{
namespace device
{

struct TerrainBRDF
{
	glm::ivec2 gridSize{ 256 };
	float gridScale{ 1.0f };
	int maxLayerCount{ 4 };
	glm::vec4 bedrockColor{ 0.5f, 0.5f, 0.5f, 1.0f };
	glm::vec4 sandColor{ 0.9f, 0.8f, 0.6f, 1.0f };
	glm::vec4 waterColor{ 0.0f, 0.3f, 0.75f, 1.0f };
};

}
}
