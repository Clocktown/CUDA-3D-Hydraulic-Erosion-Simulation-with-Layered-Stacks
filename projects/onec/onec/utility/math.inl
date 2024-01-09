#include "math.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include <cassert>
#include <limits>
#include <type_traits>

namespace onec
{

template<typename Type, bool IsCeilPowerOfTwo>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_INLINE CU_HOST_DEVICE constexpr Type fastMod(Type value, const Type ceil, const Type shift)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value + shift * ceil >= 0, "Shifted value must be greater than or equal to 0");
	ONEC_ASSERT(ceil > 0, "Ceil must be greater than 0");
	ONEC_ASSERT(shift >= 0, "Shift must be greater than or equal to 0");

	value += shift * ceil;

	if constexpr (IsCeilPowerOfTwo)
	{
		ONEC_ASSERT(isPowerOfTwo(ceil), "Ceil must be a power of two");

		return value & (ceil - 1);
	}
	else
	{
		return value < ceil ? value : value - (value / ceil) * ceil;
	}
}

template<>
CU_INLINE CU_HOST_DEVICE unsigned int fastRand(unsigned int& seed)
{
	unsigned int result{ seed * 747796405u + 2891336453u };
	result = ((result >> ((result >> 28u) + 4u)) ^ result) * 277803737u;
	result = (result >> 22u) ^ result;

	seed = result;

	return result;
}

template<>
CU_INLINE CU_HOST_DEVICE int fastRand(unsigned int& seed)
{
	return static_cast<int>(fastRand<unsigned int>(seed) & ~(1u << 31u));
}

template<>
CU_INLINE CU_HOST_DEVICE float fastRand(unsigned int& seed)
{
	return static_cast<float>(fastRand<unsigned int>(seed)) * (1.0f / static_cast<float>(UINT_MAX));
}

template<>
CU_INLINE CU_HOST_DEVICE glm::uvec2 fastRand(glm::uvec2& seed)
{
	glm::uvec2 result{ seed * 1664525u + 1013904223u };
	result.x += result.y * 1664525u;
	result.y += result.x * 1664525u;

	result = result ^ (result >> 16u);

	result.x += result.y * 1664525u;
	result.y += result.x * 1664525u;

	result = result ^ (result >> 16u);

	seed = result;

	return result;
}

template<>
CU_INLINE CU_HOST_DEVICE glm::ivec2 fastRand(glm::uvec2& seed)
{
	return glm::ivec2{ fastRand<glm::uvec2>(seed) & glm::uvec2{ ~(1u << 31u) } };
}

template<>
CU_INLINE CU_HOST_DEVICE glm::vec2 fastRand(glm::uvec2& seed)
{
	return glm::vec2{ fastRand<glm::uvec2>(seed) } *glm::vec2{ 1.0f / static_cast<float>(UINT_MAX) };
}

template<>
CU_INLINE CU_HOST_DEVICE glm::uvec3 fastRand(glm::uvec3& seed)
{
	glm::uvec3 result{ seed * 1664525u + 1013904223u };

	result.x += result.y * result.z;
	result.y += result.z * result.x;
	result.z += result.x * result.y;

	result ^= result >> 16u;

	result.x += result.y * result.z;
	result.y += result.z * result.x;
	result.z += result.x * result.y;

	seed = result;

	return result;
}

template<>
CU_INLINE CU_HOST_DEVICE glm::ivec3 fastRand(glm::uvec3& seed)
{
	return glm::ivec3{ fastRand<glm::uvec3>(seed) & glm::uvec3{ ~(1u << 31u) } };
}

template<>
CU_INLINE CU_HOST_DEVICE glm::vec3 fastRand(glm::uvec3& seed)
{
	return glm::vec3{ fastRand<glm::uvec3>(seed) } *glm::vec3{ 1.0f / static_cast<float>(UINT_MAX) };
}

template<>
CU_INLINE CU_HOST_DEVICE glm::uvec4 fastRand(glm::uvec4& seed)
{
	glm::uvec4 result{ seed * 1664525u + 1013904223u };

	result.x += result.y * result.w;
	result.y += result.z * result.x;
	result.z += result.x * result.y;
	result.w += result.y * result.z;

	result ^= result >> 16u;

	result.x += result.y * result.w;
	result.y += result.z * result.x;
	result.z += result.x * result.y;
	result.w += result.y * result.z;

	seed = result;

	return result;
}

template<>
CU_INLINE CU_HOST_DEVICE glm::ivec4 fastRand(glm::uvec4& seed)
{
	return glm::ivec4{ fastRand<glm::uvec4>(seed) & glm::uvec4{ ~(1u << 31u) } };
}

template<>
CU_INLINE CU_HOST_DEVICE glm::vec4 fastRand(glm::uvec4& seed)
{
	return glm::vec4{ fastRand<glm::uvec4>(seed) } *glm::vec4{ 1.0f / static_cast<float>(UINT_MAX) };
}

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_INLINE CU_HOST_DEVICE constexpr Type nextPowerOfTwo(const Type value)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to 0");
	ONEC_ASSERT(value < (static_cast<Type>(1) << (std::numeric_limits<Type>::digits - 1)), "The next power of two would exceed the numeric limit");

	Type result{ value - (value != 0) };

	for (int i{ 1 }; i < std::numeric_limits<Type>::digits; i <<= 1)
	{
		result |= result >> static_cast<Type>(i);
	}

	return ++result;
}

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_INLINE CU_HOST_DEVICE constexpr bool isPowerOfTwo(const Type value)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to 0");

	return value != 0 && (value & (value - 1)) == 0;
}

}
