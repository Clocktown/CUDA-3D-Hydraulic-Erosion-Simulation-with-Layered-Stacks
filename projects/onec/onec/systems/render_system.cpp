#include "render_system.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/application.hpp"
#include "../core/world.hpp"
#include "../components/camera.hpp"
#include "../singletons/renderer.hpp"
#include "../singletons/viewport.hpp"
#include "../singletons/camera.hpp"
#include "../events/renderer.hpp"
#include "../uniforms/renderer.hpp"
#include "../graphics/framebuffer.hpp"
#include "../graphics/buffer.hpp"
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>

namespace onec
{

void updateRenderTarget(const Renderer& renderer, const Viewport& viewport)
{
	const glm::ivec2 offset{ viewport.offset };
	const glm::ivec2 size{ viewport.size };

	GL_CHECK_ERROR(glViewport(offset.x, offset.y, size.x, size.y));

	GL_CHECK_ERROR(glEnable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glScissor(offset.x, offset.y, size.x, size.y));

	if (renderer.framebuffer != nullptr)
	{
		GL_CHECK_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderer.framebuffer->getHandle()));
		GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
	}
	else
	{
		GL_CHECK_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

		const Window& window{ getWindow() };

		if (window.getSampleCount() > 0)
		{
			GL_CHECK_ERROR(glEnable(GL_MULTISAMPLE));
		}
		else
		{
			GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
		}
	}

	GL_CHECK_ERROR(glClearColor(renderer.clearColor.x, renderer.clearColor.y, renderer.clearColor.z, renderer.clearColor.w));
	GL_CHECK_ERROR(glClear(renderer.clearMask));
}

void updateUniformBuffer(Renderer& renderer, const Viewport& viewport, const entt::entity camera)
{
	const Application& application{ getApplication() };
	const World& world{ getWorld() };

	uniform::Renderer uniforms;
	uniforms.time = static_cast<float>(application.getTime());
	uniforms.deltaTime = static_cast<float>(application.getDeltaTime());
	uniforms.viewportSize = viewport.size;

	ONEC_ASSERT(world.hasComponent<WorldToView>(camera), "Camera must have a world to view component");
	ONEC_ASSERT(world.hasComponent<ViewToClip>(camera), "Camera must have a world to view component");

	uniforms.worldToView = world.getComponent<WorldToView>(camera)->worldToView;
	uniforms.viewToWorld = glm::inverse(uniforms.worldToView);
	uniforms.viewToClip = world.getComponent<ViewToClip>(camera)->viewToClip;
	uniforms.clipToView = glm::inverse(uniforms.viewToClip);
	uniforms.worldToClip = uniforms.viewToClip * uniforms.worldToView;
	uniforms.clipToWorld = uniforms.viewToWorld * uniforms.clipToView;

	renderer.uniforms = uniforms;
	renderer.uniformBuffer.upload(asBytes(&uniforms, 1));

	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, renderer.uniformBufferLocation, renderer.uniformBuffer.getHandle()));
}

void render()
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Renderer>(), "World must have a renderer singleton");
	ONEC_ASSERT(world.hasSingleton<Viewport>(), "World must have a viewport singleton");

	Renderer& renderer{ *world.getSingleton<Renderer>() };
	const Viewport& viewport{ *world.getSingleton<Viewport>() };

	updateRenderTarget(renderer, viewport);

	const ActiveCamera* const activeCamera{ world.getSingleton<ActiveCamera>() };

	if (activeCamera != nullptr)
	{
		const entt::entity camera{ activeCamera->activeCamera };
		updateUniformBuffer(renderer, viewport, camera);

		world.dispatch<OnPreRender>();
		world.dispatch<OnRender>();
		world.dispatch<OnPostRender>();
	}

	GL_CHECK_ERROR(glDisable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
}

}
