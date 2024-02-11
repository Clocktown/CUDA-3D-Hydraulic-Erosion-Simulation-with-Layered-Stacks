#include "math.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include <type_traits>

namespace onec
{

template<typename Type>
requires std::is_integral_v<Type>
CU_INLINE CU_HOST_DEVICE constexpr Type mod(Type value, const Type ceil, const Type shift)
{
	ONEC_ASSERT(value + shift * ceil >= 0, "Shifted value must be greater than or equal to 0");
	ONEC_ASSERT(ceil > 0, "Ceil must be greater than 0");
	ONEC_ASSERT(shift >= 0, "Shift must be greater than or equal to 0");

	value += shift * ceil;

	return value < ceil ? value : value - (value / ceil) * ceil;
}

template<typename Type>
requires std::is_integral_v<Type>
CU_INLINE CU_HOST_DEVICE constexpr Type nextPowerOfTwo(const Type value)
{
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
requires std::is_integral_v<Type>
CU_INLINE CU_HOST_DEVICE constexpr bool isPowerOfTwo(const Type value)
{
	ONEC_ASSERT(value >= 0, "Value must be greater than or equal to 0");

	return value != 0 && (value & (value - 1)) == 0;
}

}
