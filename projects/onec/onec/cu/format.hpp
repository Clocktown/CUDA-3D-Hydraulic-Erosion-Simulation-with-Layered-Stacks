#pragma once

#include <cuda_runtime.h>

bool constexpr operator==(const cudaChannelFormatDesc& lhs, const cudaChannelFormatDesc& rhs);
bool constexpr operator!=(const cudaChannelFormatDesc& lhs, const cudaChannelFormatDesc& rhs);

#include "format.inl"
