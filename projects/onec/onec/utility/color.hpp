#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_HOST_DEVICE glm::vec3 adjustGamma(glm::vec3 color, float gamma);
CU_HOST_DEVICE glm::vec3 sRGBToLinear(glm::vec3 color);
CU_HOST_DEVICE glm::vec3 linearToSRGB(glm::vec3 color);

}

#include "color.inl"
