#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

namespace onec
{

__device__ glm::uvec3 getThreadIndex();
__device__ glm::uvec3 getBlockIndex();
__device__ glm::uvec3 getBlockSize();
__device__ glm::uvec3 getGridSize();
__device__ glm::uvec3 getGridStride();
__device__ glm::uvec3 getLaunchIndex();

}

#include "launch.inl"
