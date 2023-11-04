#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_HOST_DEVICE glm::vec3 adjustGamma(const glm::vec3& color, const float gamma);
CU_HOST_DEVICE glm::vec3 sRGBToLinear(const glm::vec3& color);
CU_HOST_DEVICE glm::vec3 linearToSRGB(const glm::vec3& color);

}

#include "color.inl"
