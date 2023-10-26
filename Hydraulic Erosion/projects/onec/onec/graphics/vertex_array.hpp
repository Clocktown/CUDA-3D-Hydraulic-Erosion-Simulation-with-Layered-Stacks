#pragma once

#include "buffer.hpp"
#include <glad/glad.h>
#include <string>

namespace onec
{

class VertexArray
{
public:
	explicit VertexArray();
	VertexArray(const VertexArray& other) = delete;
	VertexArray(VertexArray&& other) noexcept;

	~VertexArray();

	VertexArray& operator=(const VertexArray& other) = delete;
	VertexArray& operator=(VertexArray&& other) noexcept;

	void bind() const;
	void unbind() const;

	void setName(const std::string_view& name);
	void setVertexAttributeFormat(const int index, const int count, const GLenum type, const bool isNormalized = false, const int relativeOffset = 0);
	void setVertexAttributeLocation(const int index, const GLuint location);
	void setVertexAttributeDivisor(const int index, const int divisor);
	void enableVertexAttribute(const int index);
	void disableVertexAttribute(const int index);

	GLuint getHandle();
private:
	GLuint m_handle;
};

}
