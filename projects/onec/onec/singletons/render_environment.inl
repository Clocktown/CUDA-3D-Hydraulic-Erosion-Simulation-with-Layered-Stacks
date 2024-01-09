#include "render_environment.hpp"
#include "light.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/light.hpp"
#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename ...Includes, typename ...Excludes>
inline void RenderEnvironment::update(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	EnvironmentUniforms uniforms;
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
			uniforms.pointLights[uniforms.pointLightCount].intensity = 0.25f * glm::one_over_pi<float>() * pointLight.power;
			uniforms.pointLights[uniforms.pointLightCount].color = pointLight.color;
			uniforms.pointLights[uniforms.pointLightCount].range = pointLight.range;

			if (++uniforms.pointLightCount == uniforms.maxPointLightCount)
			{
				break;
			}
		}

		if (uniforms.pointLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.pointLights), static_cast<int>(offsetof(EnvironmentUniforms, pointLights)), uniforms.pointLightCount * static_cast<int>(sizeof(uniforms.pointLights[0])));
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
			uniforms.spotLights[uniforms.spotLightCount].intensity = 0.25f * glm::one_over_pi<float>() * spotLight.power;
			uniforms.spotLights[uniforms.spotLightCount].color = spotLight.color;
			uniforms.spotLights[uniforms.spotLightCount].range = spotLight.range;
			uniforms.spotLights[uniforms.spotLightCount].direction = -glm::normalize(glm::vec3(localToWorld[2]));
			uniforms.spotLights[uniforms.spotLightCount].innerCutOff = glm::cos((1.0f - spotLight.blend) * angle);
			uniforms.spotLights[uniforms.spotLightCount].outerCutOff = glm::cos(angle);

			if (++uniforms.spotLightCount == uniforms.maxSpotLightCount)
			{
				break;
			}
		}

		if (uniforms.spotLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.spotLights), static_cast<int>(offsetof(EnvironmentUniforms, spotLights)), uniforms.spotLightCount * static_cast<int>(sizeof(uniforms.spotLights[0])));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, DirectionalLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const DirectionalLight& directionalLight{ view.get<DirectionalLight>(entity) };

			uniforms.directionalLights[uniforms.directionalLightCount].direction = -glm::normalize(localToWorld[2]);
			uniforms.directionalLights[uniforms.directionalLightCount].strength = directionalLight.strength;
			uniforms.directionalLights[uniforms.directionalLightCount].color = directionalLight.color;

			if (++uniforms.directionalLightCount == uniforms.maxDirectionalLightCount)
			{
				break;
			}
		}

		if (uniforms.directionalLightCount > 0)
		{
			uniformBuffer.upload(asBytes(uniforms.directionalLights), static_cast<int>(offsetof(EnvironmentUniforms, directionalLights)), uniforms.directionalLightCount * static_cast<int>(sizeof(uniforms.directionalLights[0])));
		}
	}

	const AmbientLight* const ambientLight{ world.getSingleton<AmbientLight>() };

	if (ambientLight != nullptr)
	{
		uniforms.ambientLight.color = ambientLight->color;
		uniforms.ambientLight.strength = ambientLight->strength;
	}
	else
	{
		uniforms.ambientLight.color = glm::vec3{ 0.0f };
		uniforms.ambientLight.strength = 0.0f;
	}

	const int offset{ static_cast<int>(offsetof(EnvironmentUniforms, ambientLight)) };
	const int count{ static_cast<int>(sizeof(uniforms.ambientLight) + 3 * sizeof(int)) };
	uniformBuffer.upload({ reinterpret_cast<std::byte*>(&uniforms.ambientLight), count }, offset, count);

	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferLocation, uniformBuffer.getHandle()));
}

}
