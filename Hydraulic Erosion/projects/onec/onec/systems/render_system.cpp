#include "render_system.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/application.hpp"
#include "../core/scene.hpp"
#include "../components/camera.hpp"
#include "../singletons/renderer.hpp"
#include "../singletons/viewport.hpp"
#include "../singletons/camera.hpp"
#include "../graphics/framebuffer.hpp"
#include "../graphics/buffer.hpp"
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>

namespace onec
{

void RenderSystem::start()
{
	Scene& scene{ getScene() };

	ONEC_ASSERT(scene.hasSingleton<Renderer>(), "Scene must have a renderer singleton");

	Renderer& renderer{ *scene.getSingleton<Renderer>() };
	renderer.uniformBuffer.initialize(sizeof(device::Renderer));
}

void RenderSystem::update()
{
	Scene& scene{ getScene() };
	const ActiveCamera* const activeCamera{ scene.getSingleton<ActiveCamera>() };

	if (activeCamera == nullptr)
	{
		return;
	}

	ONEC_ASSERT(scene.hasSingleton<Renderer>(), "Scene must have a renderer singleton");
	ONEC_ASSERT(scene.hasSingleton<Viewport>(), "Scene must have a viewport singleton");
	
	Renderer& renderer{ *scene.getSingleton<Renderer>() };
	const Viewport viewport{ *scene.getSingleton<Viewport>() };
	const entt::entity camera{ activeCamera->activeCamera };
	
	updateUniformBuffer(renderer, viewport, camera);
	updateRenderTarget(renderer, viewport);

	scene.dispatch<OnPreRender>();
	scene.dispatch<OnRender>();
	scene.dispatch<OnPostRender>();

	GL_CHECK_ERROR(glDisable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
}

void RenderSystem::updateUniformBuffer(Renderer& renderer, const Viewport& viewport, const entt::entity camera)
{
	const Application& application{ getApplication() };
	const Scene& scene{ getScene() };

	device::Renderer uniforms;
	uniforms.time = static_cast<float>(application.getTime());
	uniforms.deltaTime = static_cast<float>(application.getDeltaTime());
	uniforms.viewportSize = viewport.size;

	ONEC_ASSERT(scene.hasComponent<WorldToView>(camera), "Camera must have a world to view component");
	ONEC_ASSERT(scene.hasComponent<ViewToClip>(camera), "Camera must have a world to view component");

	const glm::mat4& worldToView{ scene.getComponent<WorldToView>(camera)->worldToView };
	const glm::mat4& viewToClip{ scene.getComponent<ViewToClip>(camera)->viewToClip };

	uniforms.worldToView = worldToView;
	uniforms.viewToWorld = glm::inverse(uniforms.worldToView);
	uniforms.viewToClip = viewToClip;
	uniforms.clipToView = glm::inverse(viewToClip);
	uniforms.worldToClip = uniforms.viewToClip * uniforms.worldToView;
	uniforms.clipToWorld = uniforms.viewToWorld * uniforms.clipToView;

	renderer.uniforms = uniforms;
	renderer.uniformBuffer.bind(GL_UNIFORM_BUFFER, renderer.uniformBufferLocation);
	renderer.uniformBuffer.upload(asBytes(&renderer.uniforms, 1));
}

void RenderSystem::updateRenderTarget(const Renderer& renderer, const Viewport viewport)
{
	GL_CHECK_ERROR(glViewport(viewport.offset.x, viewport.offset.y, viewport.size.x, viewport.size.y));

	GL_CHECK_ERROR(glEnable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glScissor(viewport.offset.x, viewport.offset.y, viewport.size.x, viewport.size.y));

	if (renderer.framebuffer != nullptr)
	{
		if (renderer.framebuffer->getSampleCount() > 0)
		{
			GL_CHECK_ERROR(glEnable(GL_MULTISAMPLE));
		}
		else
		{
			GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
		}

		renderer.framebuffer->bind();
	}
	else
	{
		const Window& window{ getWindow() };

		if (window.getSampleCount() > 0)
		{
			GL_CHECK_ERROR(glEnable(GL_MULTISAMPLE));
		}
		else
		{
			GL_CHECK_ERROR(glDisable(GL_MULTISAMPLE));
		}

		GL_CHECK_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
	}

	GL_CHECK_ERROR(glClearColor(renderer.clearColor.x, renderer.clearColor.y, renderer.clearColor.z, renderer.clearColor.w));
	GL_CHECK_ERROR(glClear(renderer.clearMask));
}

}
