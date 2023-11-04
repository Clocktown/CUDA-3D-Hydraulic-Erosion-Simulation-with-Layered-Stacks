#include "framebuffer.hpp"
#include "texture.hpp"
#include "renderbuffer.hpp"
#include "../config/gl.hpp"
#include "../core/window.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <utility>
#include <string>

namespace onec
{

Framebuffer::Framebuffer(const glm::ivec2& size, const int sampleCount) :
	m_size{ size },
	m_sampleCount{ sampleCount }
{
	GL_CHECK_ERROR(glCreateFramebuffers(1, &m_handle));
}

Framebuffer::Framebuffer(Framebuffer&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec2{ 0 }) },
	m_sampleCount{ std::exchange(other.m_sampleCount, 0) }
{

}

Framebuffer::~Framebuffer()
{
	GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept
{
	if (this != &other)
	{
		GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_size = std::exchange(other.m_size, glm::ivec2{ 0 });
		m_sampleCount = std::exchange(other.m_sampleCount, 0);
	}

	return *this;
}

void Framebuffer::bind(const GLenum target) const
{
	GL_CHECK_FRAMEBUFFER(m_handle, target);
	GL_CHECK_ERROR(glBindFramebuffer(target, m_handle));
}

void Framebuffer::unbind(const GLenum target) const
{
	GL_CHECK_ERROR(glBindFramebuffer(target, GL_NONE));
}

void Framebuffer::blit(const GLbitfield mask, const GLenum filter) const
{
	const glm::ivec2& size{ getWindow().getFramebufferSize() };
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, 0, 0, 0, m_size.x, m_size.y, 0, 0, size.x, size.y, mask, filter));
}

void Framebuffer::blit(const glm::ivec2& size, const GLbitfield mask, const GLenum filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, 0, 0, 0, m_size.x, m_size.y, 0, 0, size.x, size.y, mask, filter));
}

void Framebuffer::blit(const glm::ivec2& offset, const glm::ivec2& size, const GLbitfield mask, const GLenum filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, 0, 0, 0, m_size.x, m_size.y, offset.x, offset.y, size.x, size.y, mask, filter));
}

void Framebuffer::blit(Framebuffer& framebuffer, const GLbitfield mask, const GLenum filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, framebuffer.getHandle(), 0, 0, m_size.x, m_size.y, 0, 0, framebuffer.getSize().x, framebuffer.getSize().y, mask, filter));
}

void Framebuffer::blit(Framebuffer& framebuffer, const glm::ivec2& size, const GLbitfield mask, const GLenum filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, framebuffer.getHandle(), 0, 0, m_size.x, m_size.y, 0, 0, size.x, size.y, mask, filter));
}

void Framebuffer::blit(Framebuffer& framebuffer, const glm::ivec2& offset, const glm::ivec2& size, const GLbitfield mask, const GLenum filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, framebuffer.getHandle(), 0, 0, m_size.x, m_size.y, offset.x, offset.y, size.x, size.y, mask, filter));
}

void Framebuffer::attachImage(const GLenum attachment, Texture& texture, const int mipLevel)
{
	GL_CHECK_ERROR(glNamedFramebufferTexture(m_handle, attachment, texture.getHandle(), mipLevel));
}

void Framebuffer::attachImage(const GLenum attachment, Renderbuffer& renderbuffer)
{
	GL_CHECK_ERROR(glNamedFramebufferRenderbuffer(m_handle, attachment, GL_RENDERBUFFER, renderbuffer.getHandle()));
}

void Framebuffer::detachImage(const GLenum attachment)
{
	GL_CHECK_ERROR(glNamedFramebufferRenderbuffer(m_handle, attachment, GL_RENDERBUFFER, GL_NONE));
}

void Framebuffer::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_FRAMEBUFFER, name);
}

void Framebuffer::setSize(const glm::ivec2& size)
{
	m_size = size;
}

void Framebuffer::setSampleCount(const int sampleCount)
{
	m_sampleCount = sampleCount;
}

void Framebuffer::setReadBuffer(const GLenum attachment)
{
	GL_CHECK_ERROR(glNamedFramebufferReadBuffer(m_handle, attachment));
}

void Framebuffer::setDrawBuffers(const Span<const GLenum>&& attachments)
{
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(m_handle, attachments.getCount(), attachments.getData()));
}

GLuint Framebuffer::getHandle()
{
	return m_handle;
}

const glm::ivec2& Framebuffer::getSize() const
{
	return m_size;
}

int Framebuffer::getSampleCount() const
{
	return m_sampleCount;
}

}
