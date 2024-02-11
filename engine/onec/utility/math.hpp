#pragma once

#include "../config/cu.hpp"
#include <type_traits>

namespace onec
{

template<typename Type>
requires std::is_integral_v<Type>
CU_HOST_DEVICE constexpr Type mod(Type value, Type ceil, Type shift = 0);

template<typename Type>
requires std::is_integral_v<Type>
CU_HOST_DEVICE constexpr Type nextPowerOfTwo(Type value);

template<typename Type>
requires std::is_integral_v<Type>
CU_HOST_DEVICE constexpr bool isPowerOfTwo(Type value);

}

#include "math.inl"
