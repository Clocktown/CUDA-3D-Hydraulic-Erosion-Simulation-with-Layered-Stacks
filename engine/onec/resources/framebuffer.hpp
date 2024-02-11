#pragma once

#include "texture.hpp"
#include "../core/window.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>

namespace onec
{

struct FramebufferAttachment
{
	GLenum format;
	int mipCount{ 1 };
	SamplerState samplerState;
	bool cudaAccess{ false };
};

class Framebuffer
{
public:
	explicit Framebuffer();
	explicit Framebuffer(glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments = Span<const FramebufferAttachment>{}, const FramebufferAttachment* depthAttachment = nullptr);
	Framebuffer(const Framebuffer& other) = delete;
	Framebuffer(Framebuffer&& other) noexcept;
	
	~Framebuffer();

	Framebuffer& operator=(const Framebuffer& other) = delete;
	Framebuffer& operator=(Framebuffer&& other) noexcept;

	void initialize(glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments = Span<const FramebufferAttachment>{}, const FramebufferAttachment* depthAttachment = nullptr);
	void release();

	GLuint getHandle();
	glm::ivec2 getSize() const;
	int getColorBufferCount() const;
	const Texture& getColorBuffer(int index) const;
	Texture& getColorBuffer(int index);
	const Texture& getDepthBuffer() const;
	Texture& getDepthBuffer();
	bool isEmpty() const;
private:	
	void create(glm::ivec2 size, const Span<const FramebufferAttachment>&& colorAttachments, const FramebufferAttachment* depthAttachment);
	void destroy();

	GLuint m_handle;
	glm::ivec2 m_size;
	std::vector<Texture> m_colorBuffers;
	Texture m_depthBuffer;
};

}
