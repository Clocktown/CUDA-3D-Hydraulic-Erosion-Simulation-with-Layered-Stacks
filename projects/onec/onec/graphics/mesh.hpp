#pragma once

#include "buffer.hpp"
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

struct VertexProperties
{
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
};

struct Mesh
{
	static constexpr GLenum indexType{ GL_UNSIGNED_INT };

	explicit Mesh() = default;
	explicit Mesh(const std::filesystem::path& file, unsigned int flags = 0, bool createBindlessHandles = false, bool createGraphicResources = false);

	std::vector<SubMesh> subMeshes;
	Buffer indexBuffer;
	Buffer positionBuffer;
	Buffer vertexPropertyBuffer;
	AABB aabb;
};

}
