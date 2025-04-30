#pragma once

#include "../core/window.hpp"
#include "../resources/framebuffer.hpp"
#include "../resources/graphics_buffer.hpp"
#include "../resources/RenderPipelineUniforms.h"
#include <glad/glad.h>
#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace onec
{

struct OnRender
{

};

struct RenderPipeline
{
	static constexpr GLuint uniformBufferLocation{ 0 };

	void render();

	template<typename... Includes, typename... Excludes>
	void update(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	std::shared_ptr<Framebuffer> framebuffer{ nullptr };
	GraphicsBuffer uniformBuffer{ sizeof(RenderPipelineUniforms), true };
	glm::vec4 clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
	GLbitfield clearMask{ GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT };
};

}

#include "render_pipeline.inl"
