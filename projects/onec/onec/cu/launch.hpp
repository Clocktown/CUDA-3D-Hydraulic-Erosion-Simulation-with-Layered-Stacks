#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

__device__ glm::ivec3 getThreadIndex();
__device__ glm::ivec3 getBlockIndex();
__device__ glm::ivec3 getBlockSize();
__device__ glm::ivec3 getGridSize();
__device__ glm::ivec3 getGridStride();
__device__ glm::ivec3 getGlobalIndex();

}
}

#include "launch.inl"
