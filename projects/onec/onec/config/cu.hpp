#pragma once

#include "config.hpp"
#include <cuda_runtime.h>

#if !defined(__CUDACC__) && !defined(__CUDABE__)
#   define CU_INLINE inline
#   define CU_HOST_DEVICE
#   define CU_IF_HOST(code) code
#   define CU_IF_DEVICE(code)
#else
#   ifdef ONEC_DEBUG
#       include <stdio.h>

#       undef ONEC_ERROR 
#       undef ONEC_ASSERT

#       define ONEC_ERROR(message) printf("ONEC Error\nDescription: %s\nFile: %s\nLine: %i\n", message, __FILE__, __LINE__)
#       define ONEC_ASSERT(condition, message) if (!(condition))\
                                               {\
                                                   ONEC_ERROR(message);\
                                               }\
                                               static_cast<void>(0)
#   endif

#   define CU_INLINE __forceinline__
#   define CU_HOST_DEVICE __device__
#   define CU_IF_HOST(code) 
#   define CU_IF_DEVICE(code) code
#endif

#ifdef ONEC_DEBUG
#   define CU_CHECK_ERROR(code) code; onec::internal::cuCheckError(__FILE__, __LINE__)
#   define CU_CHECK_KERNEL(...) __VA_ARGS__; onec::internal::cuCheckError(__FILE__, __LINE__); CU_CHECK_ERROR(cudaDeviceSynchronize())
#endif

#ifdef ONEC_RELEASE
#   define CU_CHECK_ERROR(code) code
#   define CU_CHECK_KERNEL(...) __VA_ARGS__
#endif

namespace onec
{
namespace internal
{

void cuCheckError(const char* file, int line);

}
}
