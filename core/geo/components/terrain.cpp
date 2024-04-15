#include "terrain.hpp"
#include <onec/resources/graphics_buffer.hpp>
#include <glm/glm.hpp>

#include <onec/resources/mesh.hpp>

namespace geo
{

Terrain::Terrain(const glm::ivec2 gridSize, const float gridScale, const Simulation& simulation) :
	gridSize{ gridSize },
	gridScale{ gridScale },
	simulation{ simulation }
{
	const int cellCount{ gridSize.x * gridSize.y };
	const int columnCount{ gridSize.x * gridSize.y * maxLayerCount };

	layerCountBuffer.initialize(cellCount * static_cast<std::ptrdiff_t>(sizeof(char)), true);
	heightBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)), true);
	sedimentBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	stabilityBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	pipeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(char4)));
	slopeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	fluxBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
	slippageBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
	speedBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	damageBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
}

std::shared_ptr<onec::Mesh> Terrain::makeCubeMesh() {
	auto cube = std::make_shared<onec::Mesh>();


	constexpr int vertexCount{ 8 };
	constexpr int faceCount{ 12 };
	constexpr int elementCount{ faceCount * 3 };

	std::array<glm::vec3, vertexCount> cube_vertices  {
		// front
		glm::vec3{ -1.0f, -1.0f,  1.0f},
		glm::vec3{ 1.0f, -1.0f,  1.0f},
		glm::vec3{ 1.0f,  1.0f,  1.0f},
		glm::vec3{-1.0f,  1.0f,  1.0f},
		// back
		glm::vec3{-1.0f, -1.0f, -1.0f},
		glm::vec3{ 1.0f, -1.0f, -1.0f},
		glm::vec3{ 1.0f,  1.0f, -1.0f},
		glm::vec3{-1.0f,  1.0f, -1.0f}
	};

	std::array<glm::uvec3, faceCount> cube_elements{
		// front
		glm::uvec3{0, 1, 2},
		glm::uvec3{2, 3, 0},
		// right
		glm::uvec3{1, 5, 6},
		glm::uvec3{6, 2, 1},
		// back
		glm::uvec3{7, 6, 5},
		glm::uvec3{5, 4, 7},
		// left
		glm::uvec3{4, 0, 3},
		glm::uvec3{3, 7, 4},
		// bottom
		glm::uvec3{4, 5, 1},
		glm::uvec3{1, 0, 4},
		// top
		glm::uvec3{3, 2, 6},
		glm::uvec3{6, 7, 3}
	};

	std::vector<glm::vec3> vertices;
	vertices.reserve(numCubes * vertexCount);
	for (int i = 0; i < numCubes; ++i) {
		for (int j = 0; j < vertexCount; ++j) {
			vertices.push_back(cube_vertices[j]);
		}
	}

	std::vector<glm::uvec3> faces;
	faces.reserve(numCubes * faceCount);
	unsigned int offset = 0;
	for (int i = 0; i < numCubes; ++i) {
		for (int j = 0; j < faceCount; ++j) {
			faces.push_back(cube_elements[j]  + offset);
		}
		offset += vertexCount;
	}

	std::vector<onec::SubMesh> subMeshes;
	subMeshes.reserve(1);

	onec::AABB aabb{ {1.f, 1.f, 1.f },
		{-1.f, -1.f, -1.f } };

	subMeshes.push_back({ .primitiveType{ GL_TRIANGLES },
							.indexCount{ elementCount * numCubes },
							.baseIndex{ 0 },
							.baseVertex{ 0 } });

	cube->indexBuffer.initialize(onec::asBytes(faces), false);
	cube->positionBuffer.initialize(onec::asBytes(vertices), false);

	cube->subMeshes = std::move(subMeshes);
	cube->aabb = aabb;

	return cube;
}

}
