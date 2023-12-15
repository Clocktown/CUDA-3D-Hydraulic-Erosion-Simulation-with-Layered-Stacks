#include "common.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include <cassert>
#include <limits>
#include <type_traits>

namespace onec
{

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_INLINE CU_HOST_DEVICE constexpr Type nextPowerOfTwo(const Type value)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to zero");
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
CU_INLINE CU_HOST_DEVICE constexpr Type fastMod(const Type value, const Type mod)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to zero");
	ONEC_ASSERT(isPowerOfTwo(mod), "Mod must be a power of two");

	return value & (mod - 1);
}

template<typename Type>
CU_IF_HOST(requires std::is_integral_v<Type>)
CU_INLINE CU_HOST_DEVICE constexpr bool isPowerOfTwo(const Type value)
{
	static_assert(std::is_integral_v<Type>, "Type must be integral");

	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to zero");

	return value != 0 && (value & (value - 1)) == 0;
}

}
