#pragma once

#include <onec/onec.hpp>
#include <glad/glad.h>

namespace geo
{

struct TerrainRenderer
{
	static constexpr GLuint uniformBufferLocation{ 2 };
	static constexpr GLuint materialBufferLocation{ 3 };

	onec::VertexArray vertexArray;
	onec::Buffer uniformBuffer;
};

}
