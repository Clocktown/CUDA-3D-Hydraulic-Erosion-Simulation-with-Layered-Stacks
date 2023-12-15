#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace geo
{
namespace device
{

template<typename T>
struct Array3D
{
	template<typename U>
	__forceinline__ __device__ void write(const glm::ivec3 index, const U& value, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		surf3Dwrite<T>(reinterpret_cast<const T&>(value), surfaceObject, index.x * static_cast<int>(sizeof(T)), index.y, index.z, boundaryMode);
	}

	template<typename U>
	__forceinline__ __device__ U read(const glm::ivec3 index, const cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
	{
		static_assert(sizeof(U) == sizeof(T), "Size of U must be equal to size of T");

		return reinterpret_cast<U&&>(surf3Dread<T>(surfaceObject, index.x * static_cast<int>(sizeof(T)), index.y, index.z, boundaryMode));
	}

	template<typename U>
	__forceinline__ __device__ U sample(const glm::vec3 index) const
	{
		return reinterpret_cast<U&&>(tex3D<T>(textureObject, index.x, index.y, index.z));
	}

	cudaSurfaceObject_t surfaceObject{};
	cudaTextureObject_t textureObject{};
};

}
}
