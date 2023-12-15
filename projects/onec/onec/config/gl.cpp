#include "gl.hpp"
#include <glad/glad.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace onec
{
namespace internal
{

void glCheckError(const char* const file, const int line)
{
	const GLenum error{ glGetError() };

	if (error != GL_NO_ERROR)
	{
		std::string name;

		switch (error)
		{
		case GL_INVALID_ENUM:
			name = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:                 
			name = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:             
			name = "GL_INVALID_OPERATION";
			break;
		case GL_STACK_OVERFLOW:                
			name = "GL_STACK_UNDERFLOW";
			break;
		case GL_STACK_UNDERFLOW:               
			name = "GL_STACK_UNDERFLOW";
			break;
		case GL_OUT_OF_MEMORY:                 
			name = "GL_OUT_OF_MEMORY";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: 
			name = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		default:
		    name = std::to_string(error);
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

		std::vector<char> description(static_cast<std::size_t>(length));
		GL_CHECK_ERROR(glGetProgramInfoLog(program, length, nullptr, description.data()));

		std::cerr << "OpenGL Error: GL_LINK_STATUS\n"
			      << "Description: " << description.data() 
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

		std::vector<char> description(static_cast<std::size_t>(length));
		GL_CHECK_ERROR(glGetShaderInfoLog(shader, length, nullptr, description.data()));
		
		std::cerr << "OpenGL Error: GL_COMPILE_STATUS\n"
			      << "Description: " << description.data()
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
		std::string name;

		switch (status)
		{
		case GL_FRAMEBUFFER_UNDEFINED:
			name = "GL_FRAMEBUFFER_UNDEFINED";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			name = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			name = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			name = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			name = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			name = "GL_FRAMEBUFFER_UNSUPPORTED";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			name = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			name = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
			break;
		default:
		    name = std::to_string(status);
		    break;
		}

		std::cerr << "OpenGL Error: " << name << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glLabelObject(const GLuint object, const GLenum type, const std::string_view label)
{
	GL_CHECK_ERROR(glObjectLabel(type, object, static_cast<int>(label.size()), label.data()));
}

}
}
