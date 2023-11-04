#include "renderbuffer.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <string>

namespace onec
{

Renderbuffer::Renderbuffer() :
	m_handle{ GL_NONE },
	m_size{ 0 },
	m_format{ GL_NONE }
{

}

Renderbuffer::Renderbuffer(const glm::ivec2& size, const GLenum format) :
	m_size{ size }
{
	if (m_size != glm::ivec2{ 0 })
	{
		GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
		GL_CHECK_ERROR(glGetInternalformativ(GL_RENDERBUFFER, format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_size.x, m_size.y));
	}
	else
	{
		m_handle = GL_NONE;
		m_format = format;
	}
}

Renderbuffer::Renderbuffer(const Renderbuffer& other) :
	m_size{ other.m_size },
	m_format{ other.m_format }
{
	if (!other.isEmpty())
	{
		GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_size.x, m_size.y));
		GL_CHECK_ERROR(glCopyImageSubData(other.m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_size.x, m_size.y, 1));
	}
	else
	{
		m_handle = GL_NONE;
	}
}

Renderbuffer::Renderbuffer(Renderbuffer&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec2{ 0 }) },
	m_format{ std::exchange(other.m_format, GL_NONE) }
{

}

Renderbuffer::~Renderbuffer()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
	}
}

Renderbuffer& Renderbuffer::operator=(const Renderbuffer& other)
{
	if (this != &other)
	{
		initialize(other.m_size, other.m_format);

		if (!other.isEmpty())
		{
			GL_CHECK_ERROR(glCopyImageSubData(other.m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_size.x, m_size.y, 1));
		}
	}

	return *this;
}

Renderbuffer& Renderbuffer::operator=(Renderbuffer&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
		}

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_size = std::exchange(other.m_size, glm::ivec2{ 0 });
		m_format = std::exchange(other.m_format, GL_NONE);
	}

	return *this;
}

void Renderbuffer::initialize(const glm::ivec2& size, const GLenum format)
{
	if (m_size == size && m_format == format)
	{
		return;
	}

	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
	}

	m_size = size;

	if (m_size != glm::ivec2{ 0 })
	{
		GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
		GL_CHECK_ERROR(glGetInternalformativ(GL_RENDERBUFFER, format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_size.x, m_size.y));
	}
	else
	{
		m_format = format;
	}
}

void Renderbuffer::release()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));

		m_handle = GL_NONE;
		m_size = glm::ivec2{ 0 };
		m_format = GL_NONE;
	}
}

void Renderbuffer::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_RENDERBUFFER, name);
}

GLuint Renderbuffer::getHandle()
{
	return m_handle;
}

GLenum Renderbuffer::getTarget() const
{
	return GL_RENDERBUFFER;
}

const glm::ivec2& Renderbuffer::getSize() const
{
	return m_size;
}

GLenum Renderbuffer::getFormat() const
{
	return m_format;
}

bool Renderbuffer::isEmpty() const
{
	return m_handle == GL_NONE;
}

}
