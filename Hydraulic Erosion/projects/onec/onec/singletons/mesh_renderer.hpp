#pragma once

#include "../graphics/vertex_array.hpp"
#include "../graphics/buffer.hpp"
#include <glad/glad.h>

namespace onec
{

struct MeshRenderer
{
	static constexpr GLuint positionAttributeLocation{ 0 };
	static constexpr GLuint normalAttributeLocation{ 1 };
	static constexpr GLuint tangentAttributeLocation{ 2 };
	static constexpr GLuint uvAttributeLocation{ 3 };
	static constexpr GLuint uniformBufferLocation{ 2 };
	static constexpr GLuint materialBufferLocation{ 3 };

	VertexArray vertexArray;
	Buffer uniformBuffer;
};

}
