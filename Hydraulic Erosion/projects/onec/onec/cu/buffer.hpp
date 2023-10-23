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
	explicit Buffer(const int count);
	explicit Buffer(const Span<const std::byte>&& data);
	explicit Buffer(onec::Buffer& buffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	Buffer(const Buffer& other);
	Buffer(Buffer&& other) noexcept;
	
	~Buffer();

	Buffer& operator=(const Buffer& other);
	Buffer& operator=(Buffer&& other) noexcept;

	void initialize(const int count);
	void initialize(const Span<const std::byte>&& data);
	void initialize(onec::Buffer& buffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	void release();
	void upload(const Span<const std::byte>&& data);
	void upload(const Span<const std::byte>&& data, const int count);
	void upload(const Span<const std::byte>&& data, const int offset, const int count);
	void download(const Span<std::byte>&& data) const;
	void download(const Span<std::byte>&& data, const int count) const;
	void download(const Span<std::byte>&& data, const int offset, const int count) const;
	void map();
	void unmap();

	const std::byte* getData() const;
	std::byte* getData();
	int getCount() const;
	bool isEmpty() const;
private:
	void* m_data;
	int m_count;
	cudaGraphicsResource_t m_graphicsResource;
	bool m_isMapped;
};

}
}
