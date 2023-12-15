#pragma once

#include "config.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef ONEC_DEBUG
#   define GLFW_CHECK_ERROR(code) code; onec::internal::glfwCheckError(__FILE__, __LINE__)
#endif

#ifdef ONEC_RELEASE
#   define GLFW_CHECK_ERROR(code) code
#endif

namespace onec
{
namespace internal
{

void glfwCheckError(const char* file, int line);

}
}
