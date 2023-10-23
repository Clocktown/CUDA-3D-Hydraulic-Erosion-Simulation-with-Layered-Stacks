#include "vertex_array.hpp"
#include "buffer.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <utility>
#include <string>

namespace onec
{

VertexArray::VertexArray()
{
	GL_CHECK_ERROR(glCreateVertexArrays(1, &m_handle));
}

VertexArray::VertexArray(VertexArray&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}

VertexArray::~VertexArray()
{
	GL_CHECK_ERROR(glDeleteVertexArrays(1, &m_handle));
}

VertexArray& VertexArray::operator=(VertexArray&& other) noexcept
{
	if (this != &other)
	{
		GL_CHECK_ERROR(glDeleteVertexArrays(1, &m_handle));
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void VertexArray::bind() const
{
	GL_CHECK_ERROR(glBindVertexArray(m_handle));
}

void VertexArray::unbind() const
{
	GL_CHECK_ERROR(glBindVertexArray(GL_NONE));
}

void VertexArray::attachIndexBuffer(const Buffer& indexBuffer)
{
	GL_CHECK_ERROR(glVertexArrayElementBuffer(m_handle, const_cast<Buffer&>(indexBuffer).getHandle()));
}

void VertexArray::attachVertexBuffer(const GLuint location, const Buffer& vertexBuffer, const int stride)
{
	GL_CHECK_ERROR(glVertexArrayVertexBuffer(m_handle, location, const_cast<Buffer&>(vertexBuffer).getHandle(), 0, stride));
}

void VertexArray::detachIndexBuffer()
{
	GL_CHECK_ERROR(glVertexArrayElementBuffer(m_handle, GL_NONE));
}

void VertexArray::detachVertexBuffer(const GLuint location)
{
	GL_CHECK_ERROR(glVertexArrayVertexBuffer(m_handle, location, GL_NONE, 0, 0));
}

void VertexArray::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_VERTEX_ARRAY, name);
}

void VertexArray::setVertexAttributeFormat(const int index, const int count, const GLenum type, const bool isNormalized, const int relativeOffset)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");
	ONEC_ASSERT(relativeOffset >= 0, "Relative stride must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayAttribFormat(m_handle, static_cast<GLuint>(index), count, type, isNormalized, static_cast<GLuint>(relativeOffset)));
}

void VertexArray::setVertexAttributeLocation(const int index, const GLuint location)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayAttribBinding(m_handle, location, static_cast<GLuint>(index)));
}

void VertexArray::setVertexAttributeDivisor(const int index, const int divisor)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");
	ONEC_ASSERT(divisor >= 0, "Divisor must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayBindingDivisor(m_handle, static_cast<GLuint>(index), static_cast<GLuint>(divisor)));
}

void VertexArray::enableVertexAttribute(const int index)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");
	
	GL_CHECK_ERROR(glEnableVertexArrayAttrib(m_handle, static_cast<GLuint>(index)));
}

void VertexArray::disableVertexAttribute(const int index)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");

	GL_CHECK_ERROR(glDisableVertexArrayAttrib(m_handle, static_cast<GLuint>(index)));
}

GLuint VertexArray::getHandle()
{
	return m_handle;
}

}
