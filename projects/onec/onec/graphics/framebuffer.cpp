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

Framebuffer::Framebuffer(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorBuffers, const FramebufferAttachment* const depthBuffer, const FramebufferAttachment* const stencilBuffer, const GLenum readBuffer)
{
	create(size, std::forward<const Span<const FramebufferAttachment>>(colorBuffers), depthBuffer, stencilBuffer, readBuffer);
}

Framebuffer::Framebuffer(Framebuffer&& other)  noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec2{ 0 }) }
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
	}

	return *this;
}

void Framebuffer::initialize(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorBuffers, const FramebufferAttachment* const depthBuffer, const FramebufferAttachment* const stencilBuffer, const GLenum readBuffer)
{
	destroy();
	create(size, std::forward<const Span<const FramebufferAttachment>>(colorBuffers), depthBuffer, stencilBuffer, readBuffer);
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

const Texture& Framebuffer::getColorBuffer(const int index) const
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to zero");
	ONEC_ASSERT(index < getColorBufferCount(), "Index must be smaller than color buffer count");

	return m_colorBuffers[static_cast<std::size_t>(index)];
}

Texture& Framebuffer::getColorBuffer(const int index)
{
	return m_colorBuffers[static_cast<std::size_t>(index)];
}

int Framebuffer::getColorBufferCount() const
{
	return static_cast<int>(m_colorBuffers.size());
}

const Texture& Framebuffer::getDepthBuffer() const
{
	return m_depthBuffer;
}

Texture& Framebuffer::getDepthBuffer()
{
	return m_depthBuffer;
}

const Texture& Framebuffer::getStencilBuffer() const
{
	return m_stencilBuffer;
}

Texture& Framebuffer::getStencilBuffer()
{
	return m_stencilBuffer;
}

bool Framebuffer::isEmpty() const
{
	return m_handle == GL_NONE;
}

void Framebuffer::create(const glm::ivec2 size, const Span<const FramebufferAttachment>&& colorBuffers, const FramebufferAttachment* const depthBuffer, const FramebufferAttachment* const stencilBuffer, const GLenum readBuffer)
{
	GLuint handle;
	GL_CHECK_ERROR(glCreateFramebuffers(1, &handle));
	GL_CHECK_ERROR(glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_WIDTH, size.x));
	GL_CHECK_ERROR(glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_HEIGHT, size.y));

	m_handle = handle;
	m_size = size;

	const int count{ colorBuffers.getCount() };
	const std::size_t capacity{ static_cast<std::size_t>(count) };

	std::vector<GLenum> drawBuffers;
	drawBuffers.reserve(capacity);

	m_colorBuffers.reserve(capacity);

	for (int i{ 0 }; i < count; ++i)
	{
		const GLenum attachment{ drawBuffers.emplace_back(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i)) };
		const FramebufferAttachment& descriptor{ colorBuffers[i] };

		Texture& colorBuffer{ m_colorBuffers.emplace_back(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, descriptor.format, descriptor.mipCount, descriptor.samplerState, descriptor.createBindlessHandle, descriptor.createBindlessImageHandle, descriptor.createGraphicsResource) };

		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, attachment, colorBuffer.getHandle(), 0));
	}

	if (depthBuffer != nullptr)
	{
		const FramebufferAttachment& descriptor{ *depthBuffer };
		m_depthBuffer.initialize(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, descriptor.format, descriptor.mipCount, descriptor.samplerState, descriptor.createBindlessHandle, descriptor.createBindlessImageHandle, descriptor.createGraphicsResource);
		
		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, GL_DEPTH_ATTACHMENT, m_depthBuffer.getHandle(), 0));
	}

	if (stencilBuffer != nullptr)
	{
		const FramebufferAttachment& descriptor{ *stencilBuffer };
		m_stencilBuffer.initialize(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, descriptor.format, descriptor.mipCount, descriptor.samplerState, descriptor.createBindlessHandle, descriptor.createBindlessImageHandle, descriptor.createGraphicsResource);

		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, GL_STENCIL_ATTACHMENT, m_stencilBuffer.getHandle(), 0));
	}

	GL_CHECK_ERROR(glNamedFramebufferReadBuffer(handle, readBuffer));
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(handle, colorBuffers.getCount(), drawBuffers.data()));
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
		m_stencilBuffer.release();
	}
}

}
