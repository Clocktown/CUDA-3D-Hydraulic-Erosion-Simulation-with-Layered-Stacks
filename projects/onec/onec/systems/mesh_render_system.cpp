#include "mesh_render_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../singletons/mesh_renderer.hpp"
#include "../device/mesh_renderer.hpp"
#include "../graphics/vertex_array.hpp"

namespace onec
{

void MeshRenderSystem::start()
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<MeshRenderer>(), "World must have a mesh renderer singleton");

	MeshRenderer& meshRenderer{ *world.getSingleton<MeshRenderer>() };
	
	meshRenderer.vertexArray.setVertexAttributeFormat(meshRenderer.positionAttributeLocation, 3, GL_FLOAT);
	meshRenderer.vertexArray.setVertexAttributeFormat(meshRenderer.normalAttributeLocation, 3, GL_FLOAT, false, static_cast<int>(offsetof(VertexProperties, normal)));
	meshRenderer.vertexArray.setVertexAttributeFormat(meshRenderer.tangentAttributeLocation, 3, GL_FLOAT, false, static_cast<int>(offsetof(VertexProperties, tangent)));
	meshRenderer.vertexArray.setVertexAttributeFormat(meshRenderer.uvAttributeLocation, 2, GL_FLOAT, false, static_cast<int>(offsetof(VertexProperties, uv)));
	
	meshRenderer.vertexArray.setVertexAttributeLocation(0, meshRenderer.positionAttributeLocation);
	meshRenderer.vertexArray.setVertexAttributeLocation(1, meshRenderer.normalAttributeLocation);
	meshRenderer.vertexArray.setVertexAttributeLocation(1, meshRenderer.tangentAttributeLocation);
	meshRenderer.vertexArray.setVertexAttributeLocation(1, meshRenderer.uvAttributeLocation);

	meshRenderer.vertexArray.enableVertexAttribute(meshRenderer.positionAttributeLocation);
	meshRenderer.vertexArray.enableVertexAttribute(meshRenderer.normalAttributeLocation);
	meshRenderer.vertexArray.enableVertexAttribute(meshRenderer.tangentAttributeLocation);
	meshRenderer.vertexArray.enableVertexAttribute(meshRenderer.uvAttributeLocation);

	meshRenderer.uniformBuffer.initialize(sizeof(device::MeshRenderer));
}

}
