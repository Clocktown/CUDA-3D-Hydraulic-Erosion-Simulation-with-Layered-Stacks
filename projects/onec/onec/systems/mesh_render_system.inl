#include "mesh_render_system.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/render_mesh.hpp"
#include "../singletons/mesh_renderer.hpp"
#include "../uniforms/mesh_renderer.hpp"
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
inline void renderMeshes(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<MeshRenderer>(), "World must have a mesh renderer singleton");

	MeshRenderer& meshRenderer{ *world.getSingleton<MeshRenderer>() };
	const GLuint vertexArray{ meshRenderer.vertexArray.getHandle() };

	GL_CHECK_ERROR(glBindVertexArray(vertexArray));
	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, meshRenderer.uniformBufferLocation, meshRenderer.uniformBuffer.getHandle()));
	
	uniform::MeshRenderer uniforms;
	Mesh* activeMesh{ nullptr };
	Material* activeMaterial{ nullptr };
	Program* activeProgram{ nullptr };
	const RenderState* activeRenderState{ nullptr };

	const auto view{ world.getView<LocalToWorld, RenderMesh, Includes...>(excludes) };

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
	
			GL_CHECK_ERROR(glVertexArrayElementBuffer(vertexArray, activeMesh->indexBuffer.getHandle()));
			GL_CHECK_ERROR(glVertexArrayVertexBuffer(vertexArray, 0, activeMesh->positionBuffer.getHandle(), 0, sizeof(glm::vec3)));
			GL_CHECK_ERROR(glVertexArrayVertexBuffer(vertexArray, 1, activeMesh->vertexPropertyBuffer.getHandle(), 0, sizeof(VertexProperties)));
		}

		uniforms.localToWorld = view.get<LocalToWorld>(entity).localToWorld;
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

				GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, meshRenderer.materialBufferLocation, activeMaterial->uniformBuffer.getHandle()));

				Program* const program{ material->program.get() };
				const RenderState* const renderState{ material->renderState.get() };

				if (program != activeProgram)
				{
					activeProgram = program;
					GL_CHECK_ERROR(glUseProgram(activeProgram->getHandle()));
				}

				if (renderState != activeRenderState)
				{
					activeRenderState = renderState;

					GL_CHECK_ERROR(glColorMask(activeRenderState->colorMask.x, activeRenderState->colorMask.y, activeRenderState->colorMask.z, activeRenderState->colorMask.w));

					if (activeRenderState->isDepthTestEnabled)
					{
						GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
						GL_CHECK_ERROR(glDepthMask(activeRenderState->depthMask));
						GL_CHECK_ERROR(glDepthFunc(activeRenderState->depthFunction));
					}
					else
					{
						GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
					}

					if (activeRenderState->isStencilTestEnabled)
					{
						GL_CHECK_ERROR(glEnable(GL_STENCIL_TEST));
						GL_CHECK_ERROR(glStencilFunc(activeRenderState->stencilFunction, activeRenderState->stencilReference, activeRenderState->stencilMask));
						GL_CHECK_ERROR(glStencilOp(activeRenderState->stencilFailOperation, activeRenderState->stencilDepthFailOperation, activeRenderState->stencilPassOperation));
					}
					else
					{
						GL_CHECK_ERROR(glDisable(GL_STENCIL_TEST));
					}

					if (activeRenderState->isBlendingEnabled)
					{
						GL_CHECK_ERROR(glEnable(GL_BLEND));
						GL_CHECK_ERROR(glBlendEquation(activeRenderState->blendEquation));
						GL_CHECK_ERROR(glBlendFunc(activeRenderState->blendSourceFactor, activeRenderState->blendDestinationFactor));
						GL_CHECK_ERROR(glBlendColor(activeRenderState->blendColor.x, activeRenderState->blendColor.y, activeRenderState->blendColor.z, activeRenderState->blendColor.w));
					}
					else
					{
						GL_CHECK_ERROR(glDisable(GL_BLEND));
					}

					if (activeRenderState->isCullingEnabled)
					{
						GL_CHECK_ERROR(glEnable(GL_CULL_FACE));
						GL_CHECK_ERROR(glCullFace(activeRenderState->cullFace));
					}
					else
					{
						GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
					}

					GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, activeRenderState->polygonMode));
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
		GL_CHECK_ERROR(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));

		GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
		GL_CHECK_ERROR(glDepthMask(GL_TRUE));
		GL_CHECK_ERROR(glDepthFunc(GL_LESS));

		GL_CHECK_ERROR(glDisable(GL_STENCIL_TEST));
		GL_CHECK_ERROR(glStencilFunc(GL_ALWAYS, 0, std::numeric_limits<unsigned int>::max()));
		GL_CHECK_ERROR(glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP));

		GL_CHECK_ERROR(glDisable(GL_BLEND));
		GL_CHECK_ERROR(glBlendEquation(GL_FUNC_ADD));
		GL_CHECK_ERROR(glBlendFunc(GL_ONE, GL_ZERO));
		GL_CHECK_ERROR(glBlendColor(0.0f, 0.0f, 0.0f, 0.0f));

		GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
		GL_CHECK_ERROR(glCullFace(GL_BACK));

		GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
	}
}

}
