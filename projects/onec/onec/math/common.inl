#include "common.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include <limits>
#include <type_traits>

namespace onec
{

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_INLINE CU_HOST_DEVICE constexpr Integral nextPowerOfTwo(const Integral value)
{
	CU_ASSERT(value >= 0, "Value must be greater than or equal to zero");
	CU_ASSERT(value < (static_cast<Integral>(1) << (std::numeric_limits<Integral>::digits - 1)), "The next power of two would exceed the numeric limit");
	
	Integral result{ value - (value != 0) };

	for (int i = 1; i < std::numeric_limits<Integral>::digits; i <<= 1)
	{
		result |= result >> static_cast<Integral>(i);
	}

	return ++result;
}

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_INLINE CU_HOST_DEVICE constexpr Integral fastMod(const Integral value, const Integral mod)
{
	CU_ASSERT(value >= 0, "Value must be greater than or equal to zero");
	CU_ASSERT(isPowerOfTwo(mod), "Mod must be a power of two");

	return value & (mod - 1);
}

template<typename Integral>
CU_IF_HOST(requires std::is_integral_v<Integral>)
CU_INLINE CU_HOST_DEVICE constexpr bool isPowerOfTwo(const Integral value)
{
	CU_ASSERT(value >= 0, "Value must be greater than or equal to zero");

	return value && ((value & (value - 1)) == 0);
}

}
