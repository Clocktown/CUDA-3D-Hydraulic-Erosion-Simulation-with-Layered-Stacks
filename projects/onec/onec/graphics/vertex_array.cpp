#include "vertex_array.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <utility>
#include <type_traits>

namespace onec
{

VertexArray::VertexArray() :
	m_handle{ GL_NONE }
{

}

VertexArray::VertexArray(const Span<const VertexAttribute>&& vertexAttributes)
{
	create(std::forward<const Span<const VertexAttribute>&&>(vertexAttributes));
}

VertexArray::VertexArray(VertexArray&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}
VertexArray::~VertexArray()
{
	destroy();
}

VertexArray& VertexArray::operator=(VertexArray&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void VertexArray::initialize(const Span<const VertexAttribute>&& vertexAttributes)
{
	destroy();
	create(std::forward<const Span<const VertexAttribute>&&>(vertexAttributes));
}

void VertexArray::release()
{
	destroy();
	m_handle = 0;
}

GLuint VertexArray::getHandle()
{
	return m_handle;
}

bool VertexArray::isEmpty() const
{
	return m_handle == GL_NONE;
}

void VertexArray::create(const Span<const VertexAttribute>&& vertexAttributes)
{
	GLuint handle;
	GL_CHECK_ERROR(glCreateVertexArrays(1, &handle));

	m_handle = handle;

	for (int i{ 0 }; i < vertexAttributes.getCount(); ++i)
	{
		const VertexAttribute& vertexAttribute{ vertexAttributes[i] };
		const GLuint binding{ vertexAttribute.binding };
		const GLuint location{ static_cast<GLuint>(i) };
		
		GL_CHECK_ERROR(glVertexArrayAttribBinding(handle, location, binding));
		GL_CHECK_ERROR(glVertexArrayAttribFormat(handle, location, vertexAttribute.count, vertexAttribute.type, vertexAttribute.isNormalized, static_cast<GLuint>(vertexAttribute.relativeOffset)));
		GL_CHECK_ERROR(glEnableVertexArrayAttrib(handle, location));
	}
}

void VertexArray::destroy()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteVertexArrays(1, &m_handle));
	}
}

}
