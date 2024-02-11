#pragma once

#include "../core/window.hpp"
#include "../resources/framebuffer.hpp"
#include "../resources/graphics_buffer.hpp"
#include <glad/glad.h>
#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace onec
{

struct RenderPipelineUniforms
{
	static constexpr int maxPointLightCount{ 256 };
	static constexpr int maxSpotLightCount{ 256 };
	static constexpr int maxDirectionalLightCount{ 256 };

	glm::mat4 worldToView;
	glm::mat4 viewToWorld;
	glm::mat4 viewToClip;
	glm::mat4 clipToView;
	glm::mat4 worldToClip;
	glm::mat4 clipToWorld;
	float time;
	float deltaTime;
	glm::ivec2 viewportSize;
	float exposure;
	int pointLightCount;
	int spotLightCount;
	int directionalLightCount;

	struct
	{
		glm::vec3 luminance;
		int pad;
	} ambientLight;

	struct
	{
		glm::vec3 position;
		float range;
		glm::vec3 intensity;
		int pad;
	} pointLights[maxPointLightCount];

	struct
	{
		glm::vec3 position;
		float range;
		glm::vec3 direction;
		float innerCutOff;
		glm::vec3 intensity;
		float outerCutOff;
	} spotLights[maxSpotLightCount];

	struct
	{
		glm::vec3 direction;
		int pad;
		glm::vec3 luminance;
		int pad2;
	} directionalLights[maxDirectionalLightCount];
};

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
	GraphicsBuffer uniformBuffer{ sizeof(RenderPipelineUniforms) };
	glm::vec4 clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
	GLbitfield clearMask{ GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT };
};

}

#include "render_pipeline.inl"
