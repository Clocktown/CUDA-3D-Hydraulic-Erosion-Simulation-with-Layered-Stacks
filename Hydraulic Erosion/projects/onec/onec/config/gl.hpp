#pragma once

#include "config.hpp"
#include <glad/glad.h>
#include <string>

#ifdef ONEC_DEBUG
#   define GL_CHECK_ERROR(function) function; onec::internal::glCheckError(__FILE__, __LINE__)
#   define GL_CHECK_PROGRAM(program) onec::internal::glCheckProgram(program, __FILE__, __LINE__) 
#   define GL_CHECK_SHADER(shader) onec::internal::glCheckShader(shader, __FILE__, __LINE__) 
#   define GL_CHECK_FRAMEBUFFER(framebuffer, target) onec::internal::glCheckFramebuffer(framebuffer, target, __FILE__, __LINE__) 
#   define GL_LABEL_OBJECT(object, type, name) onec::internal::glLabelObject(object, type, name)
#endif

#ifdef ONEC_RELEASE
#   define GL_CHECK_ERROR(function) function
#   define GL_CHECK_SHADER(shader) static_cast<void>(0)
#   define GL_CHECK_PROGRAM(program) static_cast<void>(0)
#   define GL_CHECK_FRAMEBUFFER(framebuffer, target) static_cast<void>(0)
#   define GL_LABEL_OBJECT(object, type, name) static_cast<void>(0)
#endif

namespace onec
{
namespace internal
{

void glCheckError(const char* const file, const int line);
void glCheckProgram(const GLuint program, const char* const file, const int line);
void glCheckShader(const GLuint shader, const char* const file, const int line);
void glCheckFramebuffer(const GLuint framebuffer, const GLenum target, const char* const file, const int line);
void glLabelObject(const GLuint object, const GLenum type, const std::string_view& name);

}
}
