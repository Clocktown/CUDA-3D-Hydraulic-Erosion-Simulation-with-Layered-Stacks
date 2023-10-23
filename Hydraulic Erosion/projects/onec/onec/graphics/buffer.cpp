#include "buffer.hpp"
#include "../config/gl.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <utility>
#include <string>

namespace onec
{

Buffer::Buffer() : 
	m_handle{ GL_NONE },
	m_count{ 0 }
{

}

Buffer::Buffer(const int count) :
	m_count{ count }
{
	if (m_count != 0)
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count, nullptr, GL_DYNAMIC_STORAGE_BIT));
	}
	else
	{
		m_handle = GL_NONE;
	}
}

Buffer::Buffer(const Span<const std::byte>&& data) :
	m_count{ data.getCount() }
{
	if (m_count != 0)
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count, data.getData(), GL_DYNAMIC_STORAGE_BIT));
	}
	else
	{
		m_handle = GL_NONE;
	}
}

Buffer::Buffer(const Buffer& other) :
	m_count{ other.m_count }
{
	if (!other.isEmpty())
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count, nullptr, GL_DYNAMIC_STORAGE_BIT));
		GL_CHECK_ERROR(glCopyNamedBufferSubData(other.m_handle, m_handle, 0, 0, m_count));
	}
	else
	{
		m_handle = GL_NONE;
	}
}

Buffer::Buffer(Buffer&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_count{ std::exchange(other.m_count, 0) }
{

}

Buffer::~Buffer()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
	}
}

Buffer& Buffer::operator=(const Buffer& other)
{
	if (this != &other)
	{
		initialize(other.m_count);

		if (!other.isEmpty())
		{
			GL_CHECK_ERROR(glCopyNamedBufferSubData(other.m_handle, m_handle, 0, 0, m_count));
		}
	}

	return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
	if (this != &other)
	{
		release();

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_count = std::exchange(other.m_count, 0);
	}

	return *this;
}

void Buffer::bind(const GLenum target, const GLuint location)
{
	GL_CHECK_ERROR(glBindBufferBase(target, location, m_handle));
}

void Buffer::unbind(const GLenum target, const GLuint location)
{
	GL_CHECK_ERROR(glBindBufferBase(target, location, GL_NONE));
}

void Buffer::initialize(const int count)
{
	if (m_count == count)
	{
		return;
	}

	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
	}

	m_count = count;

	if (m_count != 0)
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count, nullptr, GL_DYNAMIC_STORAGE_BIT));
	}
}

void Buffer::initialize(const Span<const std::byte>&& data)
{
	if (m_count == data.getCount())
	{
		return;
	}

	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
	}

	m_count = data.getCount();

	if (m_count != 0)
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count, data.getData(), GL_DYNAMIC_STORAGE_BIT));
	}
}

void Buffer::release()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));

		m_handle = GL_NONE;
		m_count = 0;
	}
}

void Buffer::upload(const Span<const std::byte>&& data)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, m_count, data.getData()));
}

void Buffer::upload(const Span<const std::byte>&& data, const int count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, count, data.getData()));
}

void Buffer::upload(const Span<const std::byte>&& data, const int offset, const int count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, offset, count, data.getData()));
}

void Buffer::download(const Span<std::byte>&& data) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, m_count, data.getData()));
}

void Buffer::download(const Span<std::byte>&& data, const int count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, count, data.getData()));
}

void Buffer::download(const Span<std::byte>&& data, const int offset, const int count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, offset, count, data.getData()));
}

void Buffer::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_BUFFER, name);
}

GLuint Buffer::getHandle()
{
	return m_handle;
}

int Buffer::getCount() const
{
	return m_count;
}

bool Buffer::isEmpty() const
{
	return m_handle == GL_NONE;
}

}
