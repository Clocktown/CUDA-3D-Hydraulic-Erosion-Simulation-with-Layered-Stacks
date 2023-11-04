#include "lighting_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/light.hpp"
#include "../singletons/lighting.hpp"
#include "../device/lighting.hpp"
#include "../device/light.hpp"
#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void LightingSystem::update(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Lighting>(), "World must have a lighting singleton");

	Lighting& lighting{ world.addSingleton<Lighting>() };
	device::Lighting& uniforms{ lighting.uniforms };
	uniforms.pointLightCount = 0;
	uniforms.spotLightCount = 0;
	uniforms.directionalLightCount = 0;

	{
		const auto view{ world.getView<LocalToWorld, PointLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const PointLight& pointLight{ view.get<PointLight>(entity) };

			device::PointLight& uniform{ uniforms.pointLights[uniforms.pointLightCount++] };
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
			lighting.uniformBuffer.upload(asBytes(uniforms.pointLights), static_cast<int>(offsetof(device::Lighting, pointLights)), uniforms.pointLightCount * static_cast<int>(sizeof(device::PointLight)));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, SpotLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const SpotLight& spotLight{ view.get<SpotLight>(entity) };

			const float angle{ spotLight.angle };

			device::SpotLight& uniform{ uniforms.spotLights[uniforms.spotLightCount++] };
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
			lighting.uniformBuffer.upload(asBytes(uniforms.spotLights), static_cast<int>(offsetof(device::Lighting, spotLights)), uniforms.spotLightCount * static_cast<int>(sizeof(device::SpotLight)));
		}
	}

	{
		const auto view{ world.getView<LocalToWorld, DirectionalLight, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			const DirectionalLight& directionalLight{ view.get<DirectionalLight>(entity) };

			device::DirectionalLight& uniform{ uniforms.directionalLights[uniforms.directionalLightCount++] };
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
			lighting.uniformBuffer.upload(asBytes(uniforms.directionalLights), static_cast<int>(offsetof(device::Lighting, directionalLights)), uniforms.directionalLightCount * static_cast<int>(sizeof(device::DirectionalLight)));
		}
	}

	lighting.uniformBuffer.bind(GL_UNIFORM_BUFFER, lighting.uniformBufferLocation);
	lighting.uniformBuffer.upload(asBytes(&uniforms.pointLightCount, 3), static_cast<int>(offsetof(device::Lighting, pointLightCount)), 3 * static_cast<int>(sizeof(int)));
}

}
