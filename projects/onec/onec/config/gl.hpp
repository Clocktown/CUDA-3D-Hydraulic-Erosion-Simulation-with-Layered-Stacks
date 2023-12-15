#pragma once

#include "config.hpp"
#include <glad/glad.h>

#ifdef ONEC_DEBUG
#   define GL_CHECK_ERROR(code) code; onec::internal::glCheckError(__FILE__, __LINE__)
#   define GL_CHECK_PROGRAM(program) onec::internal::glCheckProgram(program, __FILE__, __LINE__) 
#   define GL_CHECK_SHADER(shader) onec::internal::glCheckShader(shader, __FILE__, __LINE__) 
#   define GL_CHECK_FRAMEBUFFER(framebuffer, target) onec::internal::glCheckFramebuffer(framebuffer, target, __FILE__, __LINE__) 
#   define GL_LABEL_OBJECT(object, type, label) onec::internal::glLabelObject(object, type, label)
#endif

#ifdef ONEC_RELEASE
#   define GL_CHECK_ERROR(code) code
#   define GL_CHECK_SHADER(shader) static_cast<void>(0)
#   define GL_CHECK_PROGRAM(program) static_cast<void>(0)
#   define GL_CHECK_FRAMEBUFFER(framebuffer, target) static_cast<void>(0)
#   define GL_LABEL_OBJECT(object, type, label) static_cast<void>(0)
#endif

namespace onec
{
namespace internal
{

void glCheckError(const char* file, int line);
void glCheckProgram(GLuint program, const char* file, int line);
void glCheckShader(GLuint shader, const char* file, int line);
void glCheckFramebuffer(GLuint framebuffer, GLenum target, const char* file, int line);
void glLabelObject(GLuint object, GLenum type, std::string_view label);

}
}
