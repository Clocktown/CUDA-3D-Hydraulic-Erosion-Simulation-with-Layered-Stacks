#pragma once

#include "../core/window.hpp"
#include "../graphics/framebuffer.hpp"
#include "../graphics/buffer.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

namespace onec
{

struct RenderPipelineUniforms
{
	float time;
	float deltaTime;
	glm::ivec2 viewportSize;
	glm::mat4 worldToView;
	glm::mat4 viewToWorld;
	glm::mat4 viewToClip;
	glm::mat4 clipToView;
	glm::mat4 worldToClip;
	glm::mat4 clipToWorld;
};

struct RenderPipeline
{
	static constexpr GLuint uniformBufferLocation{ 0 };

	void render();

	std::shared_ptr<Framebuffer> framebuffer{ nullptr };
	Buffer uniformBuffer{ sizeof(RenderPipelineUniforms) };
	glm::vec4 clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
	GLbitfield clearMask{ GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT };
};

struct OnRender
{

};

}
