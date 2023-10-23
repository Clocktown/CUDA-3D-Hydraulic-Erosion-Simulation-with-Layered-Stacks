#include "program.hpp"
#include "shader.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <string>
#include <vector>

namespace onec
{

Program::Program() :
	m_handle{ glCreateProgram() }
{

}

Program::Program(Program&& other)  noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}

Program::~Program()
{
	GL_CHECK_ERROR(glDeleteProgram(m_handle));
}

Program& Program::operator=(Program&& other) noexcept
{
	if (this != &other)
	{
		GL_CHECK_ERROR(glDeleteProgram(m_handle));
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void Program::use() const
{
	GL_CHECK_ERROR(glUseProgram(m_handle));
}

void Program::disuse() const
{
	GL_CHECK_ERROR(glUseProgram(GL_NONE));
}

void Program::link()
{
	GL_CHECK_ERROR(glLinkProgram(m_handle));
	GL_CHECK_PROGRAM(m_handle);

	GLint shaderCount;
	GL_CHECK_ERROR(glGetProgramiv(m_handle, GL_ATTACHED_SHADERS, &shaderCount));

	std::vector<GLuint> shaders(static_cast<size_t>(shaderCount));
	GL_CHECK_ERROR(glGetAttachedShaders(m_handle, shaderCount, nullptr, shaders.data()));

	for (const GLuint shaderHandle : shaders)
	{
		GL_CHECK_ERROR(glDetachShader(m_handle, shaderHandle));
	}
}

void Program::attachShader(const Shader& shader)
{
	GL_CHECK_ERROR(glAttachShader(m_handle, const_cast<Shader&>(shader).getHandle()));
}

void Program::detachShader(const Shader& shader)
{
	GL_CHECK_ERROR(glDetachShader(m_handle, const_cast<Shader&>(shader).getHandle()));
}

void Program::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_PROGRAM, name);
}

GLuint Program::getHandle()
{
	return m_handle;
}

}
