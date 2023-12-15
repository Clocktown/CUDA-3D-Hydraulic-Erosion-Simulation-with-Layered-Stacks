#pragma once

#include "../core/window.hpp"
#include <glm/glm.hpp>

namespace onec
{

struct Viewport
{
	glm::ivec2 offset{ 0 };
	glm::ivec2 size{ 1920, 1080 };
	glm::vec2 aspect{ 1.0f };
};

}
