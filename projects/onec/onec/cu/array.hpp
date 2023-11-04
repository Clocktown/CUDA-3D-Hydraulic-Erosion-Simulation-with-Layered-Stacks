#pragma once

#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <filesystem>

namespace onec
{
namespace cu
{

class Array
{
public:
	explicit Array();
	explicit Array(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags = cudaArrayDefault);
	explicit Array(const std::filesystem::path& file, const unsigned int flags = cudaArrayDefault);
	Array(const Array& other);
	Array(Array&& other) noexcept;

	~Array();

	Array& operator=(const Array& other);
	Array& operator=(Array&& other) noexcept;

	void initialize(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags = cudaArrayDefault);
	void release();
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
	cudaArray_t m_handle;
	glm::ivec3 m_size;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;
};

}
}
