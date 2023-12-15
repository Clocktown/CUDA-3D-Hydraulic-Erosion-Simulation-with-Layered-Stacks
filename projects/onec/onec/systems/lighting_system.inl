#include "lighting_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/light.hpp"
#include "../singletons/lighting.hpp"
#include "../uniforms/lighting.hpp"
#include "../uniforms/light.hpp"
#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void updateLighting(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Lighting>(), "World must have a lighting singleton");

	Lighting& lighting{ world.addSingleton<Lighting>() };
	uniform::Lighting& uniforms{ lighting.uniforms };

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

	uniforms.pointLightCount = 0;
	uniforms.spotLightCount = 0;
	uniforms.directionalLightCount = 0;

	{
		const auto view{ world.getView<LocalToWorld, PointLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const PointLight& pointLight{ view.get<PointLight>(entity) };

			uniform::PointLight& uniform{ uniforms.pointLights[uniforms.pointLightCount++] };
			uniform.position = localToWorld[3];
			uniform.intensity = 0.25f * glm::one_over_pi<float>() * pointLight.power;
			uniform.color = pointLight.color;
			uniform.range = pointLight.range;

			if (uniforms.pointLightCount == uniforms.maxPointLightCount)
			{
				break;
			}
		}

		if (uniforms.pointLightCount > 0)
		{
			lighting.uniformBuffer.upload(asBytes(uniforms.pointLights), static_cast<int>(offsetof(uniform::Lighting, pointLights)), uniforms.pointLightCount * static_cast<int>(sizeof(uniform::PointLight)));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, SpotLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const SpotLight& spotLight{ view.get<SpotLight>(entity) };

			const float angle{ spotLight.angle };

			uniform::SpotLight& uniform{ uniforms.spotLights[uniforms.spotLightCount++] };
			uniform.position = localToWorld[3];
			uniform.intensity = 0.25f * glm::one_over_pi<float>() * spotLight.power;
			uniform.color = spotLight.color;
			uniform.range = spotLight.range;
			uniform.direction = -glm::normalize(glm::vec3(localToWorld[2]));
			uniform.innerCutOff = glm::cos((1.0f - spotLight.blend) * angle);
			uniform.outerCutOff = glm::cos(angle);

			if (uniforms.spotLightCount == uniforms.maxSpotLightCount)
			{
				break;
			}
		}

		if (uniforms.spotLightCount > 0)
		{
			lighting.uniformBuffer.upload(asBytes(uniforms.spotLights), static_cast<int>(offsetof(uniform::Lighting, spotLights)), uniforms.spotLightCount * static_cast<int>(sizeof(uniform::SpotLight)));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, DirectionalLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const DirectionalLight& directionalLight{ view.get<DirectionalLight>(entity) };

			uniform::DirectionalLight& uniform{ uniforms.directionalLights[uniforms.directionalLightCount++] };
			uniform.direction = -glm::normalize(localToWorld[2]);
			uniform.strength = directionalLight.strength;
			uniform.color = directionalLight.color;

			if (uniforms.directionalLightCount == uniforms.maxDirectionalLightCount)
			{
				break;
			}
		}

		if (uniforms.directionalLightCount > 0)
		{
			lighting.uniformBuffer.upload(asBytes(uniforms.directionalLights), static_cast<int>(offsetof(uniform::Lighting, directionalLights)), uniforms.directionalLightCount * static_cast<int>(sizeof(uniform::DirectionalLight)));
		}
	}

	const int offset{ static_cast<int>(offsetof(uniform::Lighting, ambientLight)) };
	const int count{ static_cast<int>(sizeof(uniform::AmbientLight) + 3 * sizeof(int)) };
	lighting.uniformBuffer.upload({ reinterpret_cast<std::byte*>(&uniforms.ambientLight), count }, offset, count);

	GL_CHECK_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, lighting.uniformBufferLocation, lighting.uniformBuffer.getHandle()));
}

}
