#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>
#include <type_traits>

namespace onec
{

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr Type nextPowerOfTwo(Type value);

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr Type fastMod(Type value, Type mod);

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_HOST_DEVICE constexpr bool isPowerOfTwo(Type value);

}

#include "common.inl"
