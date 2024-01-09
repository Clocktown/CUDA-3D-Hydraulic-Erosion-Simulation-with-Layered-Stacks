#pragma once

#include "../graphics/vertex_array.hpp"
#include "../graphics/buffer.hpp"
#include "../graphics/mesh.hpp"
#include <glad/glad.h>
#include <entt/entt.hpp>

namespace onec
{

struct MeshRenderPipelineUniforms
{
	glm::mat4 localToWorld;
	glm::mat4 worldToLocal;
};

struct MeshRenderPipeline
{
	static constexpr GLuint uniformBufferLocation{ 2 };
	static constexpr GLuint materialBufferLocation{ 3 };

	template<typename... Includes, typename... Excludes>
	void render(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	VertexArray vertexArray{ { VertexAttribute{ 0, 3, GL_FLOAT },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, normal)) },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, tangent)) },
							   VertexAttribute{ 1, 2, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, uv)) } } };
	Buffer uniformBuffer{ sizeof(MeshRenderPipelineUniforms) };
};

}

#include "mesh_render_pipeline.inl"
