#pragma once

#include "../resources/vertex_array.hpp"
#include "../resources/graphics_buffer.hpp"
#include "../resources/mesh.hpp"
#include <glad/glad.h>
#include <entt/entt.hpp>

namespace onec
{

struct MeshRenderPipelineUniforms
{
	glm::mat4 localToWorld;
	glm::mat4 worldToLocal;
};

template <class T = StandardVertexProperties>
struct MeshRenderPipeline
{
	static constexpr GLuint uniformBufferLocation{ 1 };
	static constexpr GLuint materialBufferLocation{ 2 };

	template<typename... Includes, typename... Excludes>
	void render(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	VertexArray vertexArray{ T::makeVAO() };
	GraphicsBuffer uniformBuffer{ sizeof(MeshRenderPipelineUniforms) };
};

}

#include "mesh_render_pipeline.inl"
