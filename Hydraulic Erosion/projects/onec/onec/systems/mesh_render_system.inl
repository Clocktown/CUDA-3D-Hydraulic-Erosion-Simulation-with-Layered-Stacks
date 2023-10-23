#include "mesh_render_system.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/scene.hpp"
#include "../components/transform.hpp"
#include "../components/render_mesh.hpp"
#include "../singletons/mesh_renderer.hpp"
#include "../device/mesh_renderer.hpp"
#include "../graphics/mesh.hpp"
#include "../graphics/material.hpp"
#include "../graphics/program.hpp"
#include "../graphics/render_state.hpp"
#include "../graphics/bind_group.hpp"
#include "../graphics/vertex_array.hpp"
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

template<typename ...Includes, typename ...Excludes>
inline void MeshRenderSystem::update(const entt::exclude_t<Excludes...> excludes)
{
	Scene& scene{ getScene() };

	ONEC_ASSERT(scene.hasSingleton<MeshRenderer>(), "Scene must have a mesh renderer singleton");

	MeshRenderer& meshRenderer{ *scene.getSingleton<MeshRenderer>() };
	meshRenderer.vertexArray.bind();
	meshRenderer.uniformBuffer.bind(GL_UNIFORM_BUFFER, meshRenderer.uniformBufferLocation);

	device::MeshRenderer uniforms;
	const Mesh* activeMesh{ nullptr };
	Material* activeMaterial{ nullptr };
	const Program* activeProgram{ nullptr };
	const RenderState* activeRenderState{ nullptr };
	const BindGroup* activeBindGroup{ nullptr };

	const auto view{ scene.getView<LocalToWorld, RenderMesh, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const RenderMesh& renderMesh{ view.get<RenderMesh>(entity) };

		if (renderMesh.mesh == nullptr || renderMesh.materials.empty())
		{
			continue;
		}

		if (renderMesh.mesh.get() != activeMesh)
		{
			activeMesh = renderMesh.mesh.get();

			meshRenderer.vertexArray.attachIndexBuffer(activeMesh->indexBuffer);
			meshRenderer.vertexArray.attachVertexBuffer(0, activeMesh->positionBuffer, sizeof(glm::vec3));
			meshRenderer.vertexArray.attachVertexBuffer(1, activeMesh->vertexPropertyBuffer, sizeof(VertexProperties));
		}

		const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
		uniforms.localToWorld = localToWorld;
		uniforms.worldToLocal = glm::inverse(uniforms.localToWorld);
		meshRenderer.uniformBuffer.upload(asBytes(&uniforms, 1));

		const std::vector<SubMesh>& subMeshes{ renderMesh.mesh->subMeshes };
		const std::vector<std::shared_ptr<Material>>& materials{ renderMesh.materials };
		const size_t maxSubMeshIndex{ subMeshes.size() - 1 };
		const size_t maxMaterialIndex{ materials.size() - 1 };
		const size_t drawCount{ glm::max(subMeshes.size(), materials.size()) };

		size_t subMeshIndex{ 0 };
		size_t materialIndex{ 0 };

		for (size_t i{ 0 }; i < drawCount; ++i)
		{
			ONEC_ASSERT(materials[materialIndex] != nullptr, "Material cannot be equal to nullptr");

			Material* const material{ materials[materialIndex].get() };

			if (material != activeMaterial)
			{
				ONEC_ASSERT(material->program != nullptr, "Material must have a program");
				ONEC_ASSERT(material->renderState != nullptr, "Material must have a render state");

				activeMaterial = material;
				activeMaterial->uniformBuffer.bind(GL_UNIFORM_BUFFER, meshRenderer.materialBufferLocation);

				const Program* const program{ material->program.get() };
				const RenderState* const renderState{ material->renderState.get() };
				const BindGroup* const bindGroup{ material->bindGroup.get() };

				if (program != activeProgram)
				{
					activeProgram = program;
					activeProgram->use();
				}

				if (renderState != activeRenderState)
				{
					activeRenderState = renderState;
					activeRenderState->use();
				}

				if (bindGroup != nullptr && bindGroup != activeBindGroup)
				{
					activeBindGroup = bindGroup;
					activeBindGroup->bind();
				}
			}

			const size_t indexByteOffset{ static_cast<size_t>(subMeshes[subMeshIndex].indexOffset) * sizeof(unsigned int) };
			GL_CHECK_ERROR(glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, subMeshes[subMeshIndex].indexCount, GL_UNSIGNED_INT, reinterpret_cast<const void*>(indexByteOffset), renderMesh.instanceCount, subMeshes[subMeshIndex].baseVertex, renderMesh.baseInstance));

			subMeshIndex = glm::min(subMeshIndex + 1, maxSubMeshIndex);
			materialIndex = glm::min(materialIndex + 1, maxMaterialIndex);
		}
	}

	if (activeRenderState != nullptr)
	{
		activeRenderState->disuse();
	}
}

}
