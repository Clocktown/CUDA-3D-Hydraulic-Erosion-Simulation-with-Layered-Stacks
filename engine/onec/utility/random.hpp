#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_HOST_DEVICE unsigned int pcg(unsigned int& seed);
CU_HOST_DEVICE glm::uvec2 pcg(glm::uvec2& seed);
CU_HOST_DEVICE glm::uvec3 pcg(glm::uvec3& seed);
CU_HOST_DEVICE glm::uvec4 pcg(glm::uvec4& seed);

}

#include "random.inl"
