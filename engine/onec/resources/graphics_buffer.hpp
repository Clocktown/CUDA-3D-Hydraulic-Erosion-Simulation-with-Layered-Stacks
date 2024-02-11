#pragma once

#include "vertex_array.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>

namespace onec
{

class GraphicsBuffer
{
public:
	explicit GraphicsBuffer();
	explicit GraphicsBuffer(std::ptrdiff_t count, bool cudaAccess = false);
	explicit GraphicsBuffer(const Span<const std::byte>&& source, bool cudaAccess = false);
	GraphicsBuffer(const GraphicsBuffer& other) = delete;
	GraphicsBuffer(GraphicsBuffer&& other) noexcept;

	~GraphicsBuffer();

	GraphicsBuffer& operator=(const GraphicsBuffer& other) = delete;
	GraphicsBuffer& operator=(GraphicsBuffer&& other) noexcept;

	void initialize(std::ptrdiff_t count, bool cudaAccess = false);
	void initialize(const Span<const std::byte>&& source, bool cudaAccess = false);
	void release();
	void upload(const Span<const std::byte>&& source);
	void upload(const Span<const std::byte>&& source, std::ptrdiff_t count);
	void upload(const Span<const std::byte>&& source, std::ptrdiff_t offset, std::ptrdiff_t count);
	void download(const Span<std::byte>&& destination) const;
	void download(const Span<std::byte>&& destination, std::ptrdiff_t count) const;
	void download(const Span<std::byte>&& destination, std::ptrdiff_t offset, std::ptrdiff_t count) const;

	GLuint getHandle();
	GLuint64EXT getBindlessHandle();
	cudaGraphicsResource_t getGraphicsResource();
	std::ptrdiff_t getCount() const;
	bool isEmpty() const;
private:
	void create(const Span<const std::byte>&& source, bool cudaAccess);
	void destroy();

	GLuint m_handle;
	GLuint64EXT m_bindlessHandle;
	cudaGraphicsResource_t m_graphicsResource;
	std::ptrdiff_t m_count;
};

}	
