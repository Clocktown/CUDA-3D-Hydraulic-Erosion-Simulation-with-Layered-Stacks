#pragma once

#include "../utility/span.hpp"
#include <glad/glad.h>

namespace onec
{

struct VertexAttribute
{
	GLuint binding;
	int count;
	GLenum type;
	int relativeOffset{ 0 };
	bool normalized{ false };
};

class VertexArray
{
public:
	explicit VertexArray();
	explicit VertexArray(const Span<const VertexAttribute>&& vertexAttributes);
	VertexArray(const VertexArray& other) = delete;
	VertexArray(VertexArray&& other) noexcept;

	~VertexArray();

	VertexArray& operator=(const VertexArray& other) = delete;
	VertexArray& operator=(VertexArray&& other) noexcept;

	void initialize(const Span<const VertexAttribute>&& vertexAttributes);
	void release();

	GLuint getHandle();
	bool isEmpty() const;
private:
	void create(const Span<const VertexAttribute>&& vertexAttributes);
	void destroy();

	GLuint m_handle;
};

}
