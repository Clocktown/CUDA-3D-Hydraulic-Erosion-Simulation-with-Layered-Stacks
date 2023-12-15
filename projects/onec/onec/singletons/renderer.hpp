#pragma once

#include "../core/window.hpp"
#include "../uniforms/renderer.hpp"
#include "../graphics/framebuffer.hpp"
#include "../graphics/buffer.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

namespace onec
{

struct Renderer
{
	static constexpr GLuint uniformBufferLocation{ 0 };

	uniform::Renderer uniforms;
	Buffer uniformBuffer{ sizeof(uniform::Renderer) };
	std::shared_ptr<Framebuffer> framebuffer{ nullptr };
	glm::vec4 clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
	GLbitfield clearMask{ GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT };
};

}
