#pragma once

#include "graphics_buffer.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>

namespace onec
{

class Buffer
{
public:
	explicit Buffer();
	explicit Buffer(std::ptrdiff_t count);
	explicit Buffer(const Span<const std::byte>&& source);
	explicit Buffer(onec::GraphicsBuffer& source);
	Buffer(const Buffer& other) = delete;
	Buffer(Buffer&& other) noexcept;
	
	~Buffer();

	Buffer& operator=(const Buffer& other) = delete;
	Buffer& operator=(Buffer&& other) noexcept;

	void initialize(std::ptrdiff_t count);
	void initialize(const Span<const std::byte>&& source);
	void initialize(onec::GraphicsBuffer& source);
	void release();
	void upload(const Span<const std::byte>&& source);
	void upload(const Span<const std::byte>&& source, std::ptrdiff_t count);
	void upload(const Span<const std::byte>&& source, std::ptrdiff_t offset, std::ptrdiff_t count);
	void download(const Span<std::byte>&& destination) const;
	void download(const Span<std::byte>&& destination, std::ptrdiff_t count) const;
	void download(const Span<std::byte>&& destination, std::ptrdiff_t offset, std::ptrdiff_t count) const;

	const std::byte* getData() const;
	std::byte* getData();
	std::ptrdiff_t getCount() const;
	bool isEmpty() const;
private:
	void create(std::ptrdiff_t count);
	void create(const Span<const std::byte>&& source);
	void create(onec::GraphicsBuffer& source);
	void destroy();

	std::byte* m_data;
	cudaGraphicsResource_t m_graphicsResource;
	std::ptrdiff_t m_count;
};

}
