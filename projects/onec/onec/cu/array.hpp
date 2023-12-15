#pragma once

#include "../graphics/texture.hpp"
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
	explicit Array(glm::ivec3 size, const cudaChannelFormatDesc& format, unsigned int flags = cudaArrayDefault, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	explicit Array(const std::filesystem::path& file, unsigned int flags = cudaArrayDefault, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	explicit Array(onec::Texture& texture, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	Array(const Array& other) = delete;
	Array(Array&& other) noexcept;

	~Array();

	Array& operator=(const Array& other) = delete;
	Array& operator=(Array&& other) noexcept;

	void initialize(glm::ivec3 size, const cudaChannelFormatDesc& format, unsigned int flags = cudaArrayDefault, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	void initialize(const std::filesystem::path& file, unsigned int flags = cudaArrayDefault, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	void initialize(onec::Texture& texture, const cudaTextureDesc* textureDescriptor = nullptr, bool createSurfaceObject = false);
	void release();
	void upload(const Span<const std::byte>&& source, glm::ivec3 size);
	void upload(const Span<const std::byte>&& source, glm::ivec3 offset, glm::ivec3 size);
	void download(const Span<std::byte>&& destination, glm::ivec3 size) const;
	void download(const Span<std::byte>&& destination, glm::ivec3 offset, glm::ivec3 size) const;

	cudaArray_const_t getHandle() const;
	cudaArray_t getHandle();
	cudaTextureObject_t getTextureObject();
	cudaSurfaceObject_t getSurfaceObject();
	glm::ivec3 getSize() const;
	const cudaChannelFormatDesc& getFormat() const;
	unsigned int getFlags() const;
	bool isEmpty() const;
private:
	void create(glm::ivec3 size, const cudaChannelFormatDesc& format, unsigned int flags, const cudaTextureDesc* textureDescriptor, bool createSurfaceObject);
	void create(const std::filesystem::path& file, unsigned int flags, const cudaTextureDesc* textureDescriptor, bool createSurfaceObject);
	void create(onec::Texture& texture, const cudaTextureDesc* textureDescriptor, bool createSurfaceObject);
	void destroy();

	cudaArray_t m_handle;
	cudaTextureObject_t m_textureObject;
	cudaSurfaceObject_t m_surfaceObject;
	cudaGraphicsResource_t m_graphicsResource;
	glm::ivec3 m_size;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;
};

}
}
