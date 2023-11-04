#include "shader.hpp"
#include "../config/gl.hpp"
#include "../utility/io.hpp"
#include <glad/glad.h>
#include <utility>
#include <filesystem>
#include <string>

namespace onec
{

Shader::Shader(const GLenum type) :
	m_handle{ glCreateShader(type) },
	m_type{ type }
{

}

Shader::Shader(const std::filesystem::path& file)
{
	const std::string source{ readShader(file, m_type) };

	m_handle = glCreateShader(m_type);

	setSource(source);
	compile();
}

Shader::Shader(Shader&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_type{ std::exchange(other.m_type, GL_NONE) }
{

}

Shader::~Shader()
{
	GL_CHECK_ERROR(glDeleteShader(m_handle));
}

Shader& Shader::operator=(Shader&& other) noexcept
{
	if (this != &other)
	{
		GL_CHECK_ERROR(glDeleteShader(m_handle));

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_type = std::exchange(other.m_type, GL_NONE);
	}

	return *this;
}

void Shader::compile()
{
	GL_CHECK_ERROR(glCompileShader(m_handle));
	GL_CHECK_SHADER(m_handle);
}

void Shader::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_SHADER, name);
}

void Shader::setSource(const std::string_view& source)
{
	const char* const data{ source.data() };
	const int count{ static_cast<int>(source.size()) };
	GL_CHECK_ERROR(glShaderSource(m_handle, 1, &data, &count));
}

GLuint Shader::getHandle()
{
	return m_handle;
}

GLenum Shader::getType() const
{
	return m_type;
}

}
