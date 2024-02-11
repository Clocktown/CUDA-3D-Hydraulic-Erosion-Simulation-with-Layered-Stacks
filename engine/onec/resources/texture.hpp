#pragma once

#include "sampler.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <filesystem>

namespace onec
{

class Texture
{
public:
	explicit Texture();
	explicit Texture(GLenum target, glm::ivec3 size, GLenum format, int mipCount = 1, const SamplerState& samplerState = SamplerState{}, bool cudaAccess = false);
	explicit Texture(const std::filesystem::path& file, int mipCount = 1, const SamplerState& samplerState = SamplerState{}, bool cudaAccess = false);
	Texture(const Texture& other) = delete;
	Texture(Texture&& other) noexcept;
	 
	~Texture();

	Texture& operator=(const Texture& other) = delete;
	Texture& operator=(Texture&& other) noexcept;

	void initialize(GLenum target, glm::ivec3 size, GLenum format, int mipCount = 1, const SamplerState& samplerState = SamplerState{}, bool cudaAccess = false);
	void initialize(const std::filesystem::path& file, int mipCount = 1, const SamplerState& samplerState = SamplerState{}, bool cudaAccess = false);
	void release();
	void generateMipmap();
	void upload(const Span<const std::byte>&& source, GLenum format, GLenum type, glm::ivec3 size, int mipLevel = 0);
	void upload(const Span<const std::byte>&& source, GLenum format, GLenum type, glm::ivec3 offset, glm::ivec3 size, int mipLevel = 0);
	void download(const Span<std::byte>&& destination, GLenum format, GLenum type, glm::ivec3 size, int mipLevel = 0) const;
	void download(const Span<std::byte>&& destination, GLenum format, GLenum type, glm::ivec3 offset, glm::ivec3 size, int mipLevel = 0) const;
	
	GLuint getHandle();
	GLuint64 getBindlessHandle();
	GLuint64 getBindlessImageHandle();
	cudaGraphicsResource_t getGraphicsResource();
	GLenum getTarget() const;
	glm::ivec3 getSize() const;
	GLenum getFormat() const;
	int getMipCount() const;
	bool isEmpty() const;
private:
	void create(GLenum target, glm::ivec3 size, GLenum format, int mipCount, const SamplerState& samplerState, bool cudaAccess);
	void create(const std::filesystem::path& file, int mipCount, const SamplerState& samplerState, bool cudaAccess);
	void destroy();

	GLuint m_handle;
	GLuint64 m_bindlessHandle;
	GLuint64 m_bindlessImageHandle;
	cudaGraphicsResource_t m_graphicsResource;
	GLenum m_target;
	glm::ivec3 m_size;
	GLenum m_format;
	int m_mipCount;
};

int getMaxMipCount(int size);
int getMaxMipCount(glm::ivec2 size);
int getMaxMipCount(glm::ivec3 size);
int getMipSize(int base, int mipLevel);
glm::ivec2 getMipSize(glm::ivec2 base, int mipLevel);
glm::ivec3 getMipSize(glm::ivec3 base, int mipLevel);

}
