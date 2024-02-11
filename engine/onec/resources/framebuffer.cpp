#include "framebuffer.hpp"
#include "texture.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <type_traits>
#include <vector>

namespace onec
{

Framebuffer::Framebuffer() :
	m_handle{ GL_NONE },
	m_size{ 0 }
{

}

Framebuffer::Framebuffer(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments, const FramebufferAttachment* const depthAttachment)
{
	create(size, std::forward<const Span<const FramebufferAttachment>>(colorAttachments), depthAttachment);
}

Framebuffer::Framebuffer(Framebuffer&& other)  noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec2{ 0 }) },
	m_colorBuffers{ std::move(other.m_colorBuffers) },
	m_depthBuffer{ std::move(other.m_depthBuffer) }
{

}

Framebuffer::~Framebuffer()
{
	destroy();
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept
{
	if (this != &other)
	{
		destroy();

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_size = std::exchange(other.m_size, glm::ivec2{ 0 });
		m_colorBuffers = std::move(other.m_colorBuffers);
		m_depthBuffer = std::move(other.m_depthBuffer);
	}

	return *this;
}

void Framebuffer::initialize(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments, const FramebufferAttachment* const depthAttachment)
{
	destroy();
	create(size, std::forward<const Span<const FramebufferAttachment>>(colorAttachments), depthAttachment);
}

void Framebuffer::release()
{
	destroy();

	m_handle = GL_NONE;
	m_size = glm::ivec2{ 0 };
}

GLuint Framebuffer::getHandle()
{
	return m_handle;
}

glm::ivec2 Framebuffer::getSize() const
{
	return m_size;
}

int Framebuffer::getColorBufferCount() const
{
	return static_cast<int>(m_colorBuffers.size());
}

const Texture& Framebuffer::getColorBuffer(const int index) const
{
	ONEC_ASSERT(index >= 0 && index < getColorBufferCount(), "Index must refer to an existing color buffer");

	return m_colorBuffers[static_cast<std::size_t>(index)];
}

Texture& Framebuffer::getColorBuffer(const int index)
{
	ONEC_ASSERT(index >= 0 && index < getColorBufferCount(), "Index must refer to an existing color buffer");

	return m_colorBuffers[static_cast<std::size_t>(index)];
}

const Texture& Framebuffer::getDepthBuffer() const
{
	return m_depthBuffer;
}

Texture& Framebuffer::getDepthBuffer()
{
	return m_depthBuffer;
}

bool Framebuffer::isEmpty() const
{
	return m_handle == GL_NONE;
}

void Framebuffer::create(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments, const FramebufferAttachment* const depthAttachment)
{
	GLuint handle;
	GL_CHECK_ERROR(glCreateFramebuffers(1, &handle));
	GL_CHECK_ERROR(glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_WIDTH, size.x));
	GL_CHECK_ERROR(glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_HEIGHT, size.y));

	m_handle = handle;
	m_size = size;

	const std::ptrdiff_t count{ colorAttachments.getCount() };
	const std::size_t capacity{ static_cast<std::size_t>(count) };

	std::vector<GLenum> drawBuffers;
	drawBuffers.reserve(capacity);

	m_colorBuffers.reserve(capacity);

	for (std::ptrdiff_t i{ 0 }; i < count; ++i)
	{
		const GLenum attachment{ drawBuffers.emplace_back(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i)) };
		const FramebufferAttachment& colorAttachment{ colorAttachments[i] };

		Texture& colorBuffer{ m_colorBuffers.emplace_back(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, colorAttachment.format, colorAttachment.mipCount, colorAttachment.samplerState, colorAttachment.cudaAccess) };
		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, attachment, colorBuffer.getHandle(), 0));
	}

	if (depthAttachment != nullptr)
	{
		const GLenum format{ depthAttachment->format };
		m_depthBuffer.initialize(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, format, depthAttachment->mipCount, depthAttachment->samplerState, depthAttachment->cudaAccess);
		
		GLint stencil;
		GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_2D, format, GL_STENCIL_RENDERABLE, 1, &stencil));
		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, stencil == GL_TRUE ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT, m_depthBuffer.getHandle(), 0));
	}
	
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(handle, static_cast<int>(colorAttachments.getCount()), drawBuffers.data()));
	GL_CHECK_FRAMEBUFFER(handle, GL_READ_FRAMEBUFFER);
	GL_CHECK_FRAMEBUFFER(handle, GL_DRAW_FRAMEBUFFER);
}

void Framebuffer::destroy()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));

		m_colorBuffers.clear();
		m_depthBuffer.release();
	}
}

}
