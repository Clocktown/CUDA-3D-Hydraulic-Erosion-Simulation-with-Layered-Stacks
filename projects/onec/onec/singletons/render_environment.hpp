#pragma once

#include "light.hpp"
#include "../components/light.hpp"
#include "../graphics/buffer.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

struct EnvironmentUniforms
{
	static constexpr int maxPointLightCount{ 256 };
	static constexpr int maxSpotLightCount{ 256 };
	static constexpr int maxDirectionalLightCount{ 256 };

	struct 
	{
		glm::vec3 position;
		float intensity;
		glm::vec3 color;
		float range;
	} pointLights[maxPointLightCount];

	struct
	{
		glm::vec3 position;
		float intensity;
		glm::vec3 direction;
		float range;
		glm::vec3 color;
		float innerCutOff;
		float outerCutOff;
		int pads[3];
	} spotLights[maxSpotLightCount];

	struct
	{
		glm::vec3 direction;
		float strength;
		glm::vec3 color;
		int pad;
	} directionalLights[maxDirectionalLightCount];

	struct 
	{
		glm::vec3 color;
		float strength;
	} ambientLight;

	int pointLightCount{ 0 };
	int spotLightCount{ 0 };
	int directionalLightCount{ 0 };
};

struct RenderEnvironment
{
	static constexpr GLuint uniformBufferLocation{ 1 };

	template<typename... Includes, typename... Excludes>
	void update(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	Buffer uniformBuffer{ sizeof(EnvironmentUniforms) };
};

}

#include "render_environment.inl"
