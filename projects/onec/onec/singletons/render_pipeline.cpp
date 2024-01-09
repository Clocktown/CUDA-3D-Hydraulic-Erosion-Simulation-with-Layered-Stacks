#include "render_pipeline.hpp"
#include "viewport.hpp"
#include "camera.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/application.hpp"
#include "../core/window.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../components/light.hpp"
#include "../graphics/framebuffer.hpp"
#include "../graphics/buffer.hpp"
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>

namespace onec
{

void RenderPipeline::render()
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Viewport>(), "World must have a viewport singleton");

	const Viewport viewport{ *world.getSingleton<Viewport>() };

	GL_CHECK_ERROR(glEnable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glScissor(viewport.offset.x, viewport.offset.y, viewport.size.x, viewport.size.y));
	GL_CHECK_ERROR(glViewport(viewport.offset.x, viewport.offset.y, viewport.size.x, viewport.size.y));

	Framebuffer* const framebuffer{ this->framebuffer.get() };

	if (framebuffer != nullptr)
	{
		GL_CHECK_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer->getHandle()));
		GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
	}
	else
	{
		const Window& window{ getWindow() };

		GL_CHECK_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

		if (window.getSampleCount() > 0)
		{
			GL_CHECK_ERROR(glEnable(GL_MULTISAMPLE));
		}
		else
		{
			GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
		}
	}

	GL_CHECK_ERROR(glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w));
	GL_CHECK_ERROR(glClear(clearMask));

	const ActiveCamera* const activeCamera{ world.getSingleton<ActiveCamera>() };

	if (activeCamera != nullptr)
	{
		const Application& application{ getApplication() };
		const entt::entity camera{ activeCamera->activeCamera };

		ONEC_ASSERT(world.hasComponent<WorldToView>(camera), "Camera must have a world to view component");
		ONEC_ASSERT(world.hasComponent<ViewToClip>(camera), "Camera must have a world to view component");

		RenderPipelineUniforms uniforms;
		uniforms.time = static_cast<float>(application.getTime());
		uniforms.deltaTime = static_cast<float>(application.getDeltaTime());
		uniforms.viewportSize = viewport.size;
		uniforms.worldToView = world.getComponent<WorldToView>(camera)->worldToView;
		uniforms.viewToWorld = glm::inverse(uniforms.worldToView);
		uniforms.viewToClip = world.getComponent<ViewToClip>(camera)->viewToClip;
		uniforms.clipToView = glm::inverse(uniforms.viewToClip);
		uniforms.worldToClip = uniforms.viewToClip * uniforms.worldToView;
		uniforms.clipToWorld = uniforms.viewToWorld * uniforms.clipToView;
		uniformBuffer.upload(asBytes(&uniforms, 1));

		GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferLocation, uniformBuffer.getHandle()));

		world.dispatch<OnRender>();
	}
}

}
