#pragma once

#include "texture.hpp"
#include "renderbuffer.hpp"
#include "../core/window.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

namespace onec
{

class Framebuffer
{
public:
	explicit Framebuffer(const glm::ivec2& size = getWindow().getFramebufferSize(), const int sampleCount = 0);
	Framebuffer(const Framebuffer& other) = delete;
	Framebuffer(Framebuffer&& other) noexcept;
	
	~Framebuffer();

	Framebuffer& operator=(const Framebuffer& other) = delete;
	Framebuffer& operator=(Framebuffer&& other) noexcept;

	void bind(const GLenum target = GL_DRAW_FRAMEBUFFER) const;
	void unbind(const GLenum target = GL_DRAW_FRAMEBUFFER) const;
	void blit(const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void blit(const glm::ivec2& size, const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void blit(const glm::ivec2& offset, const glm::ivec2& size, const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void blit(Framebuffer& framebuffer, const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void blit(Framebuffer& framebuffer, const glm::ivec2& size, const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void blit(Framebuffer& framebuffer, const glm::ivec2& offset, const glm::ivec2& size, const GLbitfield mask = GL_COLOR_BUFFER_BIT, const GLenum filter = GL_NEAREST) const;
	void attachImage(const GLenum attachment, Texture& texture, const int mipLevel = 0);
	void attachImage(const GLenum attachment, Renderbuffer& renderbuffer);
	void detachImage(const GLenum attachment);

	void setName(const std::string_view& name);
	void setSize(const glm::ivec2& size);
	void setSampleCount(const int sampleCount);
	void setReadBuffer(const GLenum attachment);
	void setDrawBuffers(const Span<const GLenum>&& attachments);

	GLuint getHandle();
	const glm::ivec2& getSize() const;
	int getSampleCount() const;
private:
	GLuint m_handle;
	glm::ivec2 m_size;
	int m_sampleCount;
};

}
