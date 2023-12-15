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
	int indexOffset;
	int indexCount;
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
	explicit Mesh() = default;
	explicit Mesh(const std::filesystem::path& file, unsigned int flags = 0);

	std::vector<SubMesh> subMeshes;
	std::vector<unsigned int> indices;
	std::vector<glm::vec3> positions;
	std::vector<VertexProperties> vertexProperties;
	Buffer indexBuffer;
	Buffer positionBuffer;
	Buffer vertexPropertyBuffer;
	AABB aabb;
};

}
