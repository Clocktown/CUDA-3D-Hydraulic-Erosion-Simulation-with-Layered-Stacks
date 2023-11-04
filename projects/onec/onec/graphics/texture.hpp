#pragma once

#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <string>

namespace onec
{

class Texture
{
public:
	explicit Texture();
	explicit Texture(const GLenum target, const glm::ivec3& size, const GLenum format, const int mipCount = 1, const int sampleCount = 0);
	explicit Texture(const std::filesystem::path& file, const int mipCount = 1);
	Texture(const Texture& other);
	Texture(Texture&& other) noexcept;

	~Texture();

	Texture& operator=(const Texture& other);
	Texture& operator=(Texture&& other) noexcept;

	void bind(const GLuint unit) const;
	void unbind(const GLuint unit) const;
	void initialize(const GLenum target, const glm::ivec3& size, const GLenum format, const int mipCount = 1, const int sampleCount = 0);
	void release();
	void generateMipmap();
	void upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const int mipLevel = 0);
	void upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& size, const int mipLevel = 0);
	void upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& offset, const glm::ivec3& size, const int mipLevel = 0);
	void download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const int mipLevel = 0) const;
	void download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& size, const int mipLevel = 0) const;
	void download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& offset, const glm::ivec3& size, const int mipLevel = 0) const;
	
	void setName(const std::string_view& name);
	void setMinFilter(const GLenum minFilter);
	void setMagFilter(const GLenum magFilter);
	void setWrapModeS(const GLenum wrapModeS);
	void setWrapModeT(const GLenum wrapModeT);
	void setWrapModeR(const GLenum wrapModeR);
	void setBorderColor(const glm::vec4& borderColor);
	void setLODBias(const float lodBias);
	void setMinLOD(const float minLOD);
	void setMaxLOD(const float maxLOD);
	void setMaxAnisotropy(const float maxAnisotropy);

	GLuint getHandle();
	GLenum getTarget() const;
	const glm::ivec3& getSize() const;
	GLenum getFormat() const;
	int getMipCount() const;
	int getSampleCount() const;
	bool isEmpty() const;
private:
	void create();

	GLuint m_handle;
	GLenum m_target;
	glm::ivec3 m_size;
	GLenum m_format;
	int m_mipCount;
	int m_sampleCount;
};

int getMaxMipCount(const glm::ivec3& size);
glm::vec3 getMipSize(const glm::ivec3& base, const int mipLevel);

}
