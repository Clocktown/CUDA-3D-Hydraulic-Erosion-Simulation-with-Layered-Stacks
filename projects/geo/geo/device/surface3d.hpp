#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace geo
{
namespace device
{

template<typename T>
struct Surface3D
{
	template<typename U>
	__forceinline__ __device__ void write(const int x, const int y, const int z, const U& value, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		surf3Dwrite<T>(reinterpret_cast<const T&>(value), handle, x * static_cast<int>(sizeof(T)), y, z, boundaryMode);
	}

	template<typename U>
	__forceinline__ __device__ void write(const glm::ivec3& position, const U& value, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		surf3Dwrite<T>(reinterpret_cast<const T&>(value), handle, position.x * static_cast<int>(sizeof(T)), position.y, position.z, boundaryMode);
	}

	template<typename U>
	__forceinline__ __device__ U read(const int x, const int y, const int z, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		return reinterpret_cast<U&&>(surf3Dread<T>(handle, x * static_cast<int>(sizeof(T)), y, z, boundaryMode));
	}

	template<typename U>
	__forceinline__ __device__ U read(const glm::ivec3& position, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		return reinterpret_cast<U&&>(surf3Dread<T>(handle, position.x * static_cast<int>(sizeof(T)), position.y, position.z, boundaryMode));
	}

	cudaSurfaceObject_t handle{};
};

}
}
