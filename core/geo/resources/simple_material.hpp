#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct SimpleMaterialUniforms
{
	glm::vec3 bedrockColor;
	int pad;
	glm::vec3 sandColor;
	int pad2;
	glm::vec3 waterColor;
	int pad3;
	glm::ivec2 gridSize;
	float gridScale;
	int maxLayerCount;
	int useInterpolation;
	int pad4;
	GLuint64EXT layerCounts;
	GLuint64EXT heights;
	GLuint64EXT stability;
};

}
