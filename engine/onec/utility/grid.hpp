#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_HOST_DEVICE constexpr int flattenIndex(glm::ivec2 index, glm::ivec1 size);
CU_HOST_DEVICE constexpr int flattenIndex(glm::ivec3 index, glm::ivec2 size);
CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec2 unflattenIndex(int index, glm::ivec1 size);
CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec3 unflattenIndex(int index, glm::ivec2 size);
CU_HOST_DEVICE constexpr int wrapIndex(int index, int size, int shift = 1);
CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec2 wrapIndex(glm::ivec2 index, glm::ivec2 size, int shift = 1);
CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec3 wrapIndex(glm::ivec3 index, glm::ivec3 size, int shift = 1);
CU_HOST_DEVICE constexpr bool isInside(int index, int size);
CU_HOST_DEVICE constexpr bool isInside(glm::ivec2 index, glm::ivec2 size);
CU_HOST_DEVICE constexpr bool isInside(glm::ivec3 index, glm::ivec3 size);
CU_HOST_DEVICE constexpr bool isOutside(int index, int size);
CU_HOST_DEVICE constexpr bool isOutside(glm::ivec2 index, glm::ivec2 size);
CU_HOST_DEVICE constexpr bool isOutside(glm::ivec3 index, glm::ivec3 size);

}

#include "grid.inl"
