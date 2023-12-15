#include "framebuffer.hpp"
#include "texture.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <string>
#include <type_traits>

namespace onec
{

Framebuffer::Framebuffer() :
	m_handle{ GL_NONE },
	m_size{ 0 }
{

}

Framebuffer::Framebuffer(const Span<Texture*>&& colorBuffers, Texture* const depthBuffer, Texture* const stencilBuffer, const GLenum readBuffer)
{
	create(std::forward<const Span<Texture*>&&>(colorBuffers), depthBuffer, stencilBuffer, readBuffer);
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

void Framebuffer::initialize(const Span<Texture*>&& colorBuffers, Texture* const depthBuffer, Texture* const stencilBuffer, const GLenum readBuffer)
{
	destroy();
	create(std::forward<const Span<Texture*>&&>(colorBuffers), depthBuffer, stencilBuffer, readBuffer);
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

bool Framebuffer::isEmpty() const
{
	return m_handle == GL_NONE;
}

void Framebuffer::create(const Span<Texture*>&& colorBuffers, Texture* const depthBuffer, Texture* const stencilBuffer, const GLenum readBuffer)
{
	GLuint handle;
	GL_CHECK_ERROR(glCreateFramebuffers(1, &handle));

	m_handle = handle;

	const std::size_t count{ static_cast<std::size_t>(colorBuffers.getCount()) };
	std::vector<GLenum> drawBuffers;
	drawBuffers.reserve(count);

	Texture* texture{ nullptr };

	for (std::size_t i{ 0 }; i < count; ++i)
	{
		texture = colorBuffers[i];

		ONEC_ASSERT(texture != nullptr, "Texture must not be nullptr");

		const GLenum attachment{ drawBuffers.emplace_back(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i)) };

		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, attachment, texture->getHandle(), 0));
	}

	if (depthBuffer != nullptr)
	{
		texture = depthBuffer;
		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, GL_DEPTH_ATTACHMENT, texture->getHandle(), 0));
	}

	if (stencilBuffer != nullptr)
	{
		texture = stencilBuffer;
		GL_CHECK_ERROR(glNamedFramebufferTexture(handle, GL_STENCIL_ATTACHMENT, texture->getHandle(), 0));
	}

	GL_CHECK_ERROR(glNamedFramebufferReadBuffer(handle, readBuffer));
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(handle, static_cast<int>(count), drawBuffers.data()));
	GL_CHECK_FRAMEBUFFER(handle, GL_READ_FRAMEBUFFER);
	GL_CHECK_FRAMEBUFFER(handle, GL_DRAW_FRAMEBUFFER);

	if (texture != nullptr)
	{
		m_size = texture->getSize();
	}
	else
	{
		m_size = glm::ivec2{ 0 };
	}
}

void Framebuffer::destroy()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));
	}
}

}
