#include "format.hpp"
#include <cuda_runtime.h>

inline bool constexpr operator==(const cudaChannelFormatDesc& lhs, const cudaChannelFormatDesc& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w && lhs.f == rhs.f;
}

inline bool constexpr operator!=(const cudaChannelFormatDesc& lhs, const cudaChannelFormatDesc& rhs)
{
	return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w || lhs.f != rhs.f;
}
