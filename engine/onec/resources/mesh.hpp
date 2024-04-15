#pragma once

#include "graphics_buffer.hpp"
#include "../components/aabb.hpp"
#include <glm/glm.hpp>
#include <assimp/postprocess.h>
#include <filesystem>
#include <vector>

namespace onec
{

struct SubMesh
{
	GLenum primitiveType{ GL_TRIANGLES };
	int indexCount;
	int baseIndex{ 0 };
	int baseVertex{ 0 };
};

struct StandardVertexProperties
{
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	static constexpr bool hasProps() { return true; }
	static VertexArray makeVAO() {
		return VertexArray{ { VertexAttribute{ 0, 3, GL_FLOAT },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(StandardVertexProperties, normal)) },
							   VertexAttribute{ 1, 3, GL_FLOAT, static_cast<int>(offsetof(StandardVertexProperties, tangent)) },
							   VertexAttribute{ 1, 2, GL_FLOAT, static_cast<int>(offsetof(StandardVertexProperties, uv)) } } };
	}
};

struct EmptyVertexProperties {
	static constexpr bool hasProps() { return false; }
	static VertexArray makeVAO() {
		return VertexArray{ { VertexAttribute{ 0, 3, GL_FLOAT } } };
	}
};

struct Mesh
{
	static constexpr GLenum indexType{ GL_UNSIGNED_INT };

	explicit Mesh() = default;
	explicit Mesh(const std::filesystem::path& file, unsigned int flags = 0, bool cudaAccess = false);

	std::vector<SubMesh> subMeshes;
	GraphicsBuffer indexBuffer;
	GraphicsBuffer positionBuffer;
	GraphicsBuffer vertexPropertyBuffer;
	AABB aabb;
};

}
