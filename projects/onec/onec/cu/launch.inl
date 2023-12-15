#include "launch.hpp"
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace onec
{
namespace cu
{

__forceinline__ __device__ glm::ivec3 getThreadIndex()
{
	return glm::ivec3{ threadIdx.x, threadIdx.y, threadIdx.z };
}

__forceinline__ __device__ glm::ivec3 getBlockIndex()
{
	return glm::ivec3{ blockIdx.x, blockIdx.y, blockIdx.z };
}

__forceinline__ __device__ glm::ivec3 getBlockSize()
{
	return glm::ivec3{ blockDim.x, blockDim.y, blockDim.z };
}

__forceinline__ __device__ glm::ivec3 getGridSize()
{
	return glm::ivec3{ gridDim.x, gridDim.y, gridDim.z };
}

__forceinline__ __device__ glm::ivec3 getGridStride()
{
	return glm::ivec3{ blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z };
}

__forceinline__ __device__ glm::ivec3 getGlobalIndex()
{
	return glm::ivec3{ threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z };
}

}
}
