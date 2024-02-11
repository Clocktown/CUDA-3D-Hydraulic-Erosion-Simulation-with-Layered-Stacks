#include "render_pipeline.hpp"
#include "viewport.hpp"
#include "camera.hpp"
#include "light.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/application.hpp"
#include "../core/window.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../components/light.hpp"
#include "../resources/framebuffer.hpp"
#include "../resources/buffer.hpp"
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void RenderPipeline::update(const entt::exclude_t<Excludes...> excludes)
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

	RenderPipelineUniforms uniforms;

	const ActiveCamera* const activeCamera{ world.getSingleton<ActiveCamera>() };

	if (activeCamera != nullptr)
	{
		const entt::entity camera{ activeCamera->activeCamera };

		ONEC_ASSERT(world.hasComponent<WorldToView>(camera), "Camera must have a world to view component");
		ONEC_ASSERT(world.hasComponent<ViewToClip>(camera), "Camera must have a world to view component");
		ONEC_ASSERT(world.hasComponent<Exposure>(camera), "Camera must have a exposure component");

		uniforms.worldToView = world.getComponent<WorldToView>(camera)->worldToView;
		uniforms.viewToWorld = glm::inverse(uniforms.worldToView);
		uniforms.viewToClip = world.getComponent<ViewToClip>(camera)->viewToClip;
		uniforms.clipToView = glm::inverse(uniforms.viewToClip);
		uniforms.worldToClip = uniforms.viewToClip * uniforms.worldToView;
		uniforms.clipToWorld = uniforms.viewToWorld * uniforms.clipToView;
		uniforms.exposure = world.getComponent<Exposure>(camera)->exposure;
	}

	const Application& application{ getApplication() };
	uniforms.time = static_cast<float>(application.getTime());
	uniforms.deltaTime = static_cast<float>(application.getDeltaTime());
	uniforms.viewportSize = viewport.size;
	uniforms.pointLightCount = 0;
	uniforms.spotLightCount = 0;
	uniforms.directionalLightCount = 0;

	{
		const auto view{ world.getView<LocalToWorld, PointLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const PointLight& pointLight{ view.get<PointLight>(entity) };

			uniforms.pointLights[uniforms.pointLightCount].position = localToWorld[3];
			uniforms.pointLights[uniforms.pointLightCount].range = pointLight.range;
			uniforms.pointLights[uniforms.pointLightCount].intensity = 0.25f * glm::one_over_pi<float>() * pointLight.power * pointLight.color;
			
			if (++uniforms.pointLightCount == uniforms.maxPointLightCount)
			{
				break;
			}
		}

		if (uniforms.pointLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.pointLights), static_cast<std::ptrdiff_t>(offsetof(RenderPipelineUniforms, pointLights)), uniforms.pointLightCount * static_cast<std::ptrdiff_t>(sizeof(uniforms.pointLights[0])));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, SpotLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const SpotLight& spotLight{ view.get<SpotLight>(entity) };

			const float angle{ spotLight.angle };

			uniforms.spotLights[uniforms.spotLightCount].position = localToWorld[3];
			uniforms.spotLights[uniforms.spotLightCount].range = spotLight.range;
			uniforms.spotLights[uniforms.spotLightCount].direction = -glm::normalize(glm::vec3(localToWorld[2]));
			uniforms.spotLights[uniforms.spotLightCount].innerCutOff = glm::cos((1.0f - spotLight.blend) * angle);
			uniforms.spotLights[uniforms.spotLightCount].intensity = 0.25f * glm::one_over_pi<float>() * spotLight.power * spotLight.color;
			uniforms.spotLights[uniforms.spotLightCount].outerCutOff = glm::cos(angle);

			if (++uniforms.spotLightCount == uniforms.maxSpotLightCount)
			{
				break;
			}
		}

		if (uniforms.spotLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.spotLights), static_cast<std::ptrdiff_t>(offsetof(RenderPipelineUniforms, spotLights)), uniforms.spotLightCount * static_cast<std::ptrdiff_t>(sizeof(uniforms.spotLights[0])));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, DirectionalLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const DirectionalLight& directionalLight{ view.get<DirectionalLight>(entity) };

			uniforms.directionalLights[uniforms.directionalLightCount].direction = -glm::normalize(localToWorld[2]);
			uniforms.directionalLights[uniforms.directionalLightCount].luminance = directionalLight.strength * directionalLight.color;

			if (++uniforms.directionalLightCount == uniforms.maxDirectionalLightCount)
			{
				break;
			}
		}

		if (uniforms.directionalLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.directionalLights), static_cast<std::ptrdiff_t>(offsetof(RenderPipelineUniforms, directionalLights)), uniforms.directionalLightCount * static_cast<std::ptrdiff_t>(sizeof(uniforms.directionalLights[0])));
		}
	}

	const AmbientLight* const ambientLight{ world.getSingleton<AmbientLight>() };

	if (ambientLight != nullptr)
	{
		uniforms.ambientLight.luminance = ambientLight->strength * ambientLight->color;
	}
	else
	{
		uniforms.ambientLight.luminance = glm::vec3{ 0.0f };
	}
	
	uniformBuffer.upload(asBytes({ uniforms }), 0, static_cast<std::ptrdiff_t>(offsetof(RenderPipelineUniforms, pointLights) + 3 * sizeof(int)));
	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferLocation, uniformBuffer.getHandle()));
}

}
