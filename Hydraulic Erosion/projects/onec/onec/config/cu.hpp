#pragma once

#include "config.hpp"
#include <cuda_runtime.h>

#if !defined(__CUDACC__) && !defined(__CUDABE__)
#   define CU_ERROR(message) ONEC_ERROR(message)
#   define CU_ASSERT(condition, message) ONEC_ASSERT(condition, message)
#   define CU_INLINE inline
#   define CU_HOST_DEVICE
#   define CU_IF_HOST(function) function
#   define CU_IF_DEVICE(function)
#else
#   ifdef ONEC_DEBUG
#       define CU_ERROR(message) printf("ONEC Error: %s\nFile: %s\nLine: %i\n", message, __FILE__, __LINE__); __trap()
#       define CU_ASSERT(condition, message) if (!(condition))\
                                             {\
                                                 CU_ERROR(message);\
                                             }\
                                             static_cast<void>(0)
#   endif

#   ifdef ONEC_RELEASE
#       define CU_ERROR(message) 
#       define CU_ASSERT(condition, message)
#   endif

#   define CU_INLINE __forceinline__
#   define CU_HOST_DEVICE __host__ __device__
#   define CU_IF_HOST(function) 
#   define CU_IF_DEVICE(function) function
#endif

#ifdef ONEC_DEBUG
#   define CU_CHECK_ERROR(function) onec::internal::cuCheckError(function, __FILE__, __LINE__)
#   define CU_CHECK_KERNEL(kernel) kernel; CU_CHECK_ERROR(cudaDeviceSynchronize())
#endif

#ifdef ONEC_RELEASE
#   define CU_CHECK_ERROR(function) function
#   define CU_CHECK_KERNEL(kernel) kernel
#endif

namespace onec
{
namespace internal
{

void cuCheckError(const cudaError_t error, const char* const file, const int line);

}
}
