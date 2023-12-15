#pragma once

#include "../uniforms/mesh_renderer.hpp"
#include "../graphics/vertex_array.hpp"
#include "../graphics/buffer.hpp"
#include "../graphics/mesh.hpp"
#include <glad/glad.h>

namespace onec
{

struct MeshRenderer
{
	static constexpr GLuint uniformBufferLocation{ 2 };
	static constexpr GLuint materialBufferLocation{ 3 };

	VertexArray vertexArray{ { VertexAttribute{ 0, 3, GL_FLOAT },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, normal)) },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, tangent)) },
							   VertexAttribute{ 1, 2, GL_FLOAT, static_cast<int>(offsetof(VertexProperties, uv)) } } };
	Buffer uniformBuffer{ sizeof(uniform::MeshRenderer) };
};

}
