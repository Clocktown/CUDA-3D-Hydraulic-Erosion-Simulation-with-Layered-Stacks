#include "point_render_pipeline.hpp"
#include "../components/point_renderer.hpp"
#include <onec/config/config.hpp>
#include <onec/config/gl.hpp>
#include <onec/core/world.hpp>
#include <onec/components/transform.hpp>
#include <onec/resources/program.hpp>
#include <onec/resources/render_state.hpp>
#include <onec/resources/vertex_array.hpp>
#include <onec/resources/graphics_buffer.hpp>
#include <onec/resources/material.hpp>
#include <entt/entt.hpp>
#include <vector>

namespace geo
{
	using namespace onec;
template<typename ...Includes, typename ...Excludes>
inline void PointRenderPipeline::render(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	GL_CHECK_ERROR(glBindVertexArray(vertexArray.getHandle()));
	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferLocation, uniformBuffer.getHandle()));

	Material* activeMaterial{ nullptr };
	Program* activeProgram{ nullptr };
	RenderState* activeRenderState{ nullptr };

	const auto view{ world.getView<LocalToWorld, PointRenderer, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const PointRenderer& pointRenderer{ view.get<PointRenderer>(entity) };

		if (pointRenderer.count == 0)
		{
			continue;
		}

		PointRenderPipelineUniforms uniforms;
		uniforms.localToWorld = view.get<LocalToWorld>(entity).localToWorld;
		uniforms.worldToLocal = glm::inverse(uniforms.localToWorld);
		uniformBuffer.upload(asBytes(&uniforms, 1));

		const std::shared_ptr<Material>& smaterial{ pointRenderer.material };

		ONEC_ASSERT(smaterial != nullptr, "Material cannot be equal to nullptr");

		Material* const material{ smaterial.get() };

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
		GL_CHECK_ERROR(glDrawArrays(GL_POINTS, pointRenderer.first, pointRenderer.count));
	}
}

}
