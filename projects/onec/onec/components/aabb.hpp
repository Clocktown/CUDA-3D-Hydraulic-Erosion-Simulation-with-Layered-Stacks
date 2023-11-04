#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct AABB
{
	glm::vec3 min{ 0.0f };
	glm::vec3 max{ 0.0f };
};

}
