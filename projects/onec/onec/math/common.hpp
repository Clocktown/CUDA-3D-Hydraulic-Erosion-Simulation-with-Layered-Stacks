#pragma once

#include "../config/cu.hpp"
#include <glm/glm.hpp>
#include <type_traits>

namespace onec
{

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_HOST_DEVICE constexpr Integral nextPowerOfTwo(const Integral value);

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_HOST_DEVICE constexpr Integral fastMod(const Integral value, const Integral mod);

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_HOST_DEVICE constexpr bool isPowerOfTwo(const Integral value);

}

#include "common.inl"
