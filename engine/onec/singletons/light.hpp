#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct AmbientLight
{
	glm::vec3 color{ 1.0f };
	float strength{ 1000.0f };
};

}
