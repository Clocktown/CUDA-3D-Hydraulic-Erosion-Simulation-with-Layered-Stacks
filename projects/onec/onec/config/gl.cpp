#include "gl.hpp"
#include <glad/glad.h>
#include <iostream>
#include <string>

namespace onec
{
namespace internal
{

void glCheckError(const char* const file, const int line)
{
	const GLenum error{ glGetError() };

	if (error != GL_NO_ERROR)
	{
		std::string description;

		switch (error)
		{
		case GL_INVALID_ENUM:
			description = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:                 
			description = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:             
			description = "GL_INVALID_OPERATION";
			break;
		case GL_STACK_OVERFLOW:                
			description = "GL_STACK_UNDERFLOW";
			break;
		case GL_STACK_UNDERFLOW:               
			description = "GL_STACK_UNDERFLOW";
			break;
		case GL_OUT_OF_MEMORY:                 
			description = "GL_OUT_OF_MEMORY";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: 
			description = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		default:
			description = std::to_string(error);
			break;
		}

		std::cerr << "OpenGL Error: " << error << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckProgram(const GLuint program, const char* const file, const int line)
{
	int status;
	GL_CHECK_ERROR(glGetProgramiv(program, GL_LINK_STATUS, &status));

	if (status == GL_FALSE)
	{
		int length;
		GL_CHECK_ERROR(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length));

		std::string log(static_cast<size_t>(length), 0);
		GL_CHECK_ERROR(glGetProgramInfoLog(program, length, nullptr, log.data()));

		std::cerr << "OpenGL Error: " << log << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckShader(const GLuint shader, const char* const file, const int line)
{
	int status;
	GL_CHECK_ERROR(glGetShaderiv(shader, GL_COMPILE_STATUS, &status));

	if (status == GL_FALSE)
	{
		int length;
		GL_CHECK_ERROR(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));

		std::string log(static_cast<size_t>(length), 0);
		GL_CHECK_ERROR(glGetShaderInfoLog(shader, length, nullptr, log.data()));

		std::cerr << "OpenGL Error: " << log << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckFramebuffer(const GLuint framebuffer, const GLenum target, const char* const file, const int line)
{
	const GLenum status{ glCheckNamedFramebufferStatus(framebuffer, target) };
	
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		std::string description;

		switch (status)
		{
		case GL_FRAMEBUFFER_UNDEFINED:
			description = "GL_FRAMEBUFFER_UNDEFINED";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			description = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			description = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			description = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			description = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			description = "GL_FRAMEBUFFER_UNSUPPORTED";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			description = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			description = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
			break;
		}

		std::cerr << "OpenGL Error: " << description << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glLabelObject(const GLuint object, const GLenum type, const std::string_view& name)
{
	GL_CHECK_ERROR(glObjectLabel(type, object, static_cast<int>(name.size()), name.data()));
}

}
}
