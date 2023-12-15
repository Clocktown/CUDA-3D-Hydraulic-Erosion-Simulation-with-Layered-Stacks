#include "mesh.hpp"
#include "../config/config.hpp"
#include "../components/aabb.hpp"
#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <limits>
#include <filesystem>
#include <vector>
#include <type_traits>

namespace onec
{

Mesh::Mesh(const std::filesystem::path& file, const unsigned int flags)
{
	Assimp::Importer importer;
	const aiScene* const scene{ importer.ReadFile(file.string(), flags) };

	ONEC_ASSERT(scene != nullptr, "Failed to read file (" + file.string() + ")");

	unsigned int indexCount{ 0 };
	unsigned int vertexCount{ 0 };
	bool hasVertexProperties{ false };

	for (unsigned int i{ 0 }; i < scene->mNumMeshes; ++i)
	{
		const aiMesh& mesh{ *scene->mMeshes[i] };

		ONEC_ASSERT(mesh.HasFaces() && mesh.HasPositions(), "Mesh must have faces and positions");
		ONEC_ASSERT((mesh.mPrimitiveTypes & aiPrimitiveType_TRIANGLE) != 0, "Primitive type must be a triangle");

		indexCount += 3 * mesh.mNumFaces;
		vertexCount += mesh.mNumVertices;
		hasVertexProperties |= mesh.HasNormals() || mesh.HasTangentsAndBitangents() || mesh.HasTextureCoords(0);
	}

	std::vector<SubMesh> subMeshes;
	std::vector<unsigned int> indices;
	std::vector<glm::vec3> positions;
	std::vector<VertexProperties> vertexProperties;
	subMeshes.reserve(scene->mNumMeshes);
	indices.reserve(indexCount);
	positions.reserve(vertexCount);
	vertexProperties.resize(hasVertexProperties * static_cast<size_t>(vertexCount));

	AABB aabb{ glm::vec3{ std::numeric_limits<float>::max() },
		       glm::vec3{ std::numeric_limits<float>::min() } };

	for (unsigned int i{ 0 }; i < scene->mNumMeshes; ++i)
	{
		const aiMesh& mesh{ *scene->mMeshes[i] };

		subMeshes.emplace_back(static_cast<int>(indices.size()), 3 * static_cast<int>(mesh.mNumFaces), static_cast<int>(positions.size()));

		for (unsigned int j{ 0 }; j < mesh.mNumFaces; ++j)
		{
			const aiFace& face{ mesh.mFaces[j] };
			indices.emplace_back(face.mIndices[0]);
			indices.emplace_back(face.mIndices[1]);
			indices.emplace_back(face.mIndices[2]);
		}

		for (unsigned int j{ 0 }; j < mesh.mNumVertices; ++j)
		{
			if (mesh.HasNormals())
			{
				vertexProperties[positions.size()].normal = glm::vec3{ mesh.mNormals[j].x, mesh.mNormals[j].y, mesh.mNormals[j].z };
			}

			if (mesh.HasTangentsAndBitangents())
			{
				vertexProperties[positions.size()].tangent = glm::vec3{ mesh.mTangents[j].x, mesh.mTangents[j].y, mesh.mTangents[j].z };
			}

			if (mesh.HasTextureCoords(0))
			{
				const aiVector3D* const uvs{ mesh.mTextureCoords[0] };
				vertexProperties[positions.size()].uv = glm::vec2{ uvs[j].x, uvs[j].y };
			}

			const glm::vec3 position{ positions.emplace_back(mesh.mVertices[j].x, mesh.mVertices[j].y, mesh.mVertices[j].z) };

			aabb.min = glm::min(position, aabb.min);
			aabb.max = glm::max(position, aabb.max);
		}
	}

	indexBuffer.initialize(asBytes(indices));
	positionBuffer.initialize(asBytes(positions));
	vertexPropertyBuffer.initialize(asBytes(vertexProperties));

	this->subMeshes = std::move(subMeshes);
	this->indices = std::move(indices);
	this->positions = std::move(positions);
	this->vertexProperties = std::move(vertexProperties);
	this->aabb = aabb;
}

}
