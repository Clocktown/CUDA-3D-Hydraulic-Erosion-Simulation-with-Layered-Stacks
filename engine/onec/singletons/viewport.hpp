#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct Viewport
{
	glm::ivec2 offset{ 0 };
	glm::ivec2 size{ 1920, 1080 };
};

}
