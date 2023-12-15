#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace geo
{
namespace uniform
{

struct Material
{
	glm::vec4 bedrockColor{ 0.5f, 0.5f, 0.5f, 1.0f };
	glm::vec4 sandColor{ 0.9f, 0.8f, 0.6f, 1.0f };
	glm::vec4 waterColor{ 0.0f, 0.3f, 0.75f, 1.0f };
	glm::ivec3 gridSize;
	float gridScale;
	GLuint64 infoMap{ GL_NONE };
	GLuint64 heightMap{ GL_NONE };
};

}
}
