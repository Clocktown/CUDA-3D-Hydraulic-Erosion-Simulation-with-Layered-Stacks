#pragma once

#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace geo
{
namespace device
{

__forceinline__ __device__ int getThreadIndex1D()
{
	return static_cast<int>(threadIdx.x);
}

__forceinline__ __device__ glm::ivec2 getThreadIndex2D()
{
	return glm::ivec2{ threadIdx.x, threadIdx.y };
}

__forceinline__ __device__ glm::ivec3 getThreadIndex3D()
{
	return glm::ivec3{ threadIdx.x, threadIdx.y, threadIdx.z };
}

__forceinline__ __device__ int getBlockIndex1D()
{
	return static_cast<int>(blockIdx.x);
}

__forceinline__ __device__ glm::ivec2 getBlockIndex2D()
{
	return glm::ivec2{ blockIdx.x, blockIdx.y };
}

__forceinline__ __device__ glm::ivec3 getBlockIndex3D()
{
	return glm::ivec3{ blockIdx.x, blockIdx.y, blockIdx.z };
}

__forceinline__ __device__ int getBlockSize1D()
{
	return static_cast<int>(blockDim.x);
}

__forceinline__ __device__ glm::ivec2 getBlockSize2D()
{
	return glm::ivec2{ blockDim.x, blockDim.y };
}

__forceinline__ __device__ glm::ivec3 getBlockSize3D()
{
	return glm::ivec3{ blockDim.x, blockDim.y, blockDim.z };
}

__forceinline__ __device__ int getGridStride1D()
{
	return static_cast<int>(blockDim.x * gridDim.x);
}

__forceinline__ __device__ glm::ivec2 getGridStride2D()
{
	return glm::ivec2{ blockDim.x * gridDim.x, blockDim.y * gridDim.y };
}

__forceinline__ __device__ glm::ivec3 getGridStride3D()
{
	return glm::ivec3{ blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z };
}

__forceinline__ __device__ int getGlobalIndex1D()
{
	return static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
}

__forceinline__ __device__ glm::ivec2 getGlobalIndex2D()
{
	return glm::ivec2{ threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y };
}

__forceinline__ __device__ glm::ivec3 getGlobalIndex3D()
{
	return glm::ivec3{ threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z };
}

__forceinline__ __device__ int getCellIndex(const glm::ivec2& cell, const glm::ivec2& gridSize)
{
	return cell.x + cell.y * gridSize.x;
}

__forceinline__ __device__ int getCellIndex(const glm::ivec3& cell, const glm::ivec3& gridSize)
{
	return cell.x + cell.y * gridSize.x + cell.z * gridSize.x * gridSize.y;
}

__forceinline__ __device__ bool isOutside(const int cellIndex, const int cellCount)
{
	return cellIndex >= cellCount || cellIndex < 0;
}

__forceinline__ __device__ bool isOutside(const glm::ivec2& cell, const glm::ivec2& gridSize)
{
	return cell.x >= gridSize.x || cell.y >= gridSize.y || cell.x < 0 || cell.y < 0;
}

__forceinline__ __device__ bool isOutside(const glm::ivec3& cell, const glm::ivec3& gridSize)
{
	return cell.x >= gridSize.x || cell.y >= gridSize.y || cell.z >= gridSize.z || cell.x < 0 || cell.y < 0 || cell.z < 0;
}

}
}
