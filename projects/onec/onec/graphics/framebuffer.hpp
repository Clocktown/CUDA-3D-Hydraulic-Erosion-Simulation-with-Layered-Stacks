#pragma once

#include "texture.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

namespace onec
{

class Framebuffer
{
public:
	explicit Framebuffer();
	explicit Framebuffer(const Span<Texture*>&& colorBuffers, Texture* depthBuffer = nullptr, Texture* stencilBuffer = nullptr, GLenum readBuffer = GL_NONE);
	Framebuffer(const Framebuffer& other) = delete;
	Framebuffer(Framebuffer&& other) noexcept;
	
	~Framebuffer();

	Framebuffer& operator=(const Framebuffer& other) = delete;
	Framebuffer& operator=(Framebuffer&& other) noexcept;

	void initialize(const Span<Texture*>&& colorBuffers, Texture* depthBuffer = nullptr, Texture* stencilBuffer = nullptr, GLenum readBuffer = GL_NONE);
	void release();

	GLuint getHandle();
	glm::ivec2 getSize() const;
	bool isEmpty() const;
private:	
	void create(const Span<Texture*>&& colorBuffers, Texture* depthBuffer = nullptr, Texture* stencilBuffer = nullptr, GLenum readBuffer = GL_NONE);
	void destroy();

	GLuint m_handle;
	glm::ivec2 m_size;
};

}
