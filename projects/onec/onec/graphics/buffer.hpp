#pragma once

#include "vertex_array.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <string>

namespace onec
{

class Buffer
{
public:
	explicit Buffer();
	explicit Buffer(int count, bool createBindlessHandle = false, bool createGraphicsResource = false);
	explicit Buffer(const Span<const std::byte>&& source, bool createBindlessHandle = false, bool createGraphicsResource = false);
	Buffer(const Buffer& other) = delete;
	Buffer(Buffer&& other) noexcept;

	~Buffer();

	Buffer& operator=(const Buffer& other) = delete;
	Buffer& operator=(Buffer&& other) noexcept;

	void initialize(int count, bool createBindlessHandle = false, bool createGraphicsResource = false);
	void initialize(const Span<const std::byte>&& source, bool createBindlessHandle = false, bool createGraphicsResource = false);
	void release();
	void upload(const Span<const std::byte>&& source);
	void upload(const Span<const std::byte>&& source, int count);
	void upload(const Span<const std::byte>&& source, int offset, int count);
	void download(const Span<std::byte>&& destination) const;
	void download(const Span<std::byte>&& destination, int count) const;
	void download(const Span<std::byte>&& destination, int offset, int count) const;

	GLuint getHandle();
	GLuint64EXT getBindlessHandle();
	cudaGraphicsResource_t getGraphicsResource();
	int getCount() const;
	bool isEmpty() const;
private:
	void create(const Span<const std::byte>&& source, bool createBindlessHandle, bool createGraphicsResource);
	void destroy();

	GLuint m_handle;
	GLuint64EXT m_bindlessHandle;
	cudaGraphicsResource_t m_graphicsResource;
	int m_count;
};

}	
