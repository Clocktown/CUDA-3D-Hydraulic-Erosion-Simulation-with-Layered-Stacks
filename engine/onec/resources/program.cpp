#include "program.hpp"
#include "../config/gl.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <utility>
#include <type_traits>

namespace onec
{

Program::Program() :
	m_handle{ GL_NONE }
{

}

Program::Program(const Span<const std::filesystem::path>&& files)
{
	create(std::forward<const Span<const std::filesystem::path>>(files));
}

Program::Program(Program&& other)  noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}

Program::~Program()
{
	destroy();
}

Program& Program::operator=(Program&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void Program::initialize(const Span<const std::filesystem::path>&& files)
{
	destroy();
	create(std::forward<const Span<const std::filesystem::path>>(files));
}

void Program::release()
{
	destroy();
	m_handle = GL_NONE;
}

GLuint Program::getHandle()
{
	return m_handle;
}

bool Program::isEmpty() const
{
	return m_handle == GL_NONE;
}

void Program::create(const Span<const std::filesystem::path>&& files)
{
	GL_CHECK_ERROR(const GLuint handle{ glCreateProgram() });

	m_handle = handle;

	std::vector<GLuint> shaders;
	shaders.reserve(static_cast<std::size_t>(files.getCount()));

	for (const std::filesystem::path& file : files)
	{
		GLenum type;
		const std::string source{ readShader(file, type) };
		const GLuint shader{ shaders.emplace_back(glCreateShader(type)) };

		const char* const data{ source.data() };
		const int count{ static_cast<int>(source.size()) };
		GL_CHECK_ERROR(glShaderSource(shader, 1, &data, &count));

		GL_CHECK_ERROR(glCompileShader(shader));
		GL_CHECK_SHADER(shader);

		GL_CHECK_ERROR(glAttachShader(handle, shader));
	}

	GL_CHECK_ERROR(glLinkProgram(handle));
	GL_CHECK_PROGRAM(handle);

	for (const GLuint shader : shaders)
	{
		GL_CHECK_ERROR(glDetachShader(handle, shader));
		GL_CHECK_ERROR(glDeleteShader(shader));
	}
}

void Program::destroy()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteProgram(m_handle));
	}
}

}
