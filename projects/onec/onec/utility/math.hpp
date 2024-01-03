#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>
#include <type_traits>

namespace onec
{

template<typename Type, bool IsCeilPowerOfTwo = false>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr Type fastMod(Type value, Type ceil, Type shift = 0);

template<typename Type = unsigned int>
CU_HOST_DEVICE Type fastRand(unsigned int& seed);

template<typename Type = glm::uvec2>
CU_HOST_DEVICE Type fastRand(glm::uvec2& seed);

template<typename Type = glm::uvec3>
CU_HOST_DEVICE Type fastRand(glm::uvec3& seed);

template<typename Type = glm::uvec4>
CU_HOST_DEVICE Type fastRand(glm::uvec4& seed);

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr Type nextPowerOfTwo(Type value);

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr bool isPowerOfTwo(Type value);

}

#include "math.inl"
