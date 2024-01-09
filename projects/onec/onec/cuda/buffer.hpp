#pragma once

#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

class Buffer
{
public:
	explicit Buffer();
	explicit Buffer(int count);
	explicit Buffer(const Span<const std::byte>&& source);
	explicit Buffer(onec::Buffer& source);
	Buffer(const Buffer& other) = delete;
	Buffer(Buffer&& other) noexcept;
	
	~Buffer();

	Buffer& operator=(const Buffer& other) = delete;
	Buffer& operator=(Buffer&& other) noexcept;

	void initialize(int count);
	void initialize(const Span<const std::byte>&& source);
	void initialize(onec::Buffer& source);
	void release();
	void upload(const Span<const std::byte>&& source);
	void upload(const Span<const std::byte>&& source, int count);
	void upload(const Span<const std::byte>&& source, int offset, int count);
	void download(const Span<std::byte>&& destination) const;
	void download(const Span<std::byte>&& destination, int count) const;
	void download(const Span<std::byte>&& destination, int offset, int count) const;

	const std::byte* getData() const;
	std::byte* getData();
	int getCount() const;
	bool isEmpty() const;
private:
	void create(int count);
	void create(const Span<const std::byte>&& source);
	void create(onec::Buffer& source);
	void destroy();

	std::byte* m_data;
	cudaGraphicsResource_t m_graphicsResource;
	int m_count;
};

}
}
