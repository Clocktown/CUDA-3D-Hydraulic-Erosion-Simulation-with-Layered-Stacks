#include "mesh_render_pipeline.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/mesh_renderer.hpp"
#include "../graphics/program.hpp"
#include "../graphics/render_state.hpp"
#include "../graphics/vertex_array.hpp"
#include "../graphics/mesh.hpp"
#include "../graphics/material.hpp"
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

template<typename ...Includes, typename ...Excludes>
inline void MeshRenderPipeline::render(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	GL_CHECK_ERROR(glBindVertexArray(vertexArray.getHandle()));
	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferLocation, uniformBuffer.getHandle()));

	Mesh* activeMesh{ nullptr };
	Material* activeMaterial{ nullptr };
	Program* activeProgram{ nullptr };
	RenderState* activeRenderState{ nullptr };

	const auto view{ world.getView<LocalToWorld, MeshRenderer, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const MeshRenderer& meshRenderer{ view.get<MeshRenderer>(entity) };

		if (meshRenderer.mesh == nullptr || meshRenderer.materials.empty())
		{
			continue;
		}

		if (meshRenderer.mesh.get() != activeMesh)
		{
			activeMesh = meshRenderer.mesh.get();

			GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, activeMesh->indexBuffer.getHandle()));
			GL_CHECK_ERROR(glBindVertexBuffer(0, activeMesh->positionBuffer.getHandle(), 0, sizeof(glm::vec3)));
			GL_CHECK_ERROR(glBindVertexBuffer(1, activeMesh->vertexPropertyBuffer.getHandle(), 0, sizeof(VertexProperties)));
		}

		MeshRenderPipelineUniforms uniforms;
		uniforms.localToWorld = view.get<LocalToWorld>(entity).localToWorld;
		uniforms.worldToLocal = glm::inverse(uniforms.localToWorld);
		uniformBuffer.upload(asBytes(&uniforms, 1));

		const std::vector<SubMesh>& subMeshes{ activeMesh->subMeshes };
		const std::vector<std::shared_ptr<Material>>& materials{ meshRenderer.materials };
		const std::size_t maxSubMeshIndex{ subMeshes.size() - 1 };
		const std::size_t maxMaterialIndex{ materials.size() - 1 };
		const std::size_t drawCount{ glm::max(subMeshes.size(), materials.size()) };

		std::size_t subMeshIndex{ 0 };
		std::size_t materialIndex{ 0 };

		for (std::size_t i{ 0 }; i < drawCount; ++i)
		{
			ONEC_ASSERT(materials[materialIndex] != nullptr, "Material cannot be equal to nullptr");

			Material* const material{ materials[materialIndex].get() };

			if (material != activeMaterial)
			{
				ONEC_ASSERT(material->program != nullptr, "Material must have a program");
				ONEC_ASSERT(material->renderState != nullptr, "Material must have a render state");

				activeMaterial = material;

				GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, materialBufferLocation, activeMaterial->uniformBuffer.getHandle()));

				Program* const program{ material->program.get() };
				RenderState* const renderState{ material->renderState.get() };

				if (program != activeProgram)
				{
					activeProgram = program;
					GL_CHECK_ERROR(glUseProgram(activeProgram->getHandle()));
				}

				if (renderState != activeRenderState)
				{
					activeRenderState = renderState;
					activeRenderState->use();
				}
			}

			const std::size_t indexByteOffset{ static_cast<std::size_t>(subMeshes[subMeshIndex].baseIndex) * sizeof(unsigned int) };
			GL_CHECK_ERROR(glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, subMeshes[subMeshIndex].indexCount, activeMesh->indexType, reinterpret_cast<const void*>(indexByteOffset), meshRenderer.instanceCount, subMeshes[subMeshIndex].baseVertex, meshRenderer.baseInstance));

			subMeshIndex = glm::min(subMeshIndex + 1, maxSubMeshIndex);
			materialIndex = glm::min(materialIndex + 1, maxMaterialIndex);
		}
	}
}

}
