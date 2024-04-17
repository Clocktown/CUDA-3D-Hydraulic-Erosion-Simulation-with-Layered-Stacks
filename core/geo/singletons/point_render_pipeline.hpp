#pragma once

#include <onec/resources/vertex_array.hpp>
#include <onec/resources/graphics_buffer.hpp>
#include <glad/glad.h>
#include <entt/entt.hpp>

#include <glm/glm.hpp>

namespace geo
{

struct PointRenderPipelineUniforms
{
	glm::mat4 localToWorld;
	glm::mat4 worldToLocal;
};

struct PointRenderPipeline
{
	static constexpr GLuint uniformBufferLocation{ 1 };
	static constexpr GLuint materialBufferLocation{ 2 };

	template<typename... Includes, typename... Excludes>
	void render(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	onec::VertexArray vertexArray{ {} };
	onec::GraphicsBuffer uniformBuffer{ sizeof(PointRenderPipelineUniforms) };
};

}

#include "point_render_pipeline.inl"
