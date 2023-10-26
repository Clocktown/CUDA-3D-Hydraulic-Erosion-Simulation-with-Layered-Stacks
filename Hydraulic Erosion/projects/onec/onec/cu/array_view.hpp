#pragma once

#include "array.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace onec
{
namespace cu
{

class GraphicsResource;

class ArrayView
{
public:
	ArrayView();
	ArrayView(Array& array);
	
	ArrayView& operator=(Array& array);

	void upload(const Span<const std::byte>&& data);
	void upload(const Span<const std::byte>&& data, const glm::ivec3& size);
	void upload(const Span<const std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size);
	void download(const Span<std::byte>&& data) const;
	void download(const Span<std::byte>&& data, const glm::ivec3& size) const;
	void download(const Span<std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size) const;

	cudaArray_const_t getHandle() const;
	cudaArray_t getHandle();
	const glm::ivec3& getSize() const;
	const cudaChannelFormatDesc& getFormat() const;
	unsigned int getFlags() const;
	bool isEmpty() const;
private:
	ArrayView(const cudaArray_t array, const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags = cudaArrayDefault);

	cudaArray_t m_handle;
	glm::ivec3 m_size;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;

	friend class GraphicsResource;
};

}
}
