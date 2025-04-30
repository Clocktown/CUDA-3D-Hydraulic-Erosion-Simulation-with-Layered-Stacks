#pragma once

#include <glm/glm.hpp>

namespace onec {
	struct RenderPipelineUniforms
	{
		static constexpr int maxPointLightCount{ 256 };
		static constexpr int maxSpotLightCount{ 256 };
		static constexpr int maxDirectionalLightCount{ 256 };

		glm::mat4 worldToView;
		glm::mat4 viewToWorld;
		glm::mat4 viewToClip;
		glm::mat4 clipToView;
		glm::mat4 worldToClip;
		glm::mat4 clipToWorld;
		float time;
		float deltaTime;
		glm::ivec2 viewportSize;
		float exposure;
		int pointLightCount;
		int spotLightCount;
		int directionalLightCount;

		struct AmbientLight
		{
			glm::vec3 luminance;
			int pad;
		} ambientLight;

		struct PointLight
		{
			glm::vec3 position;
			float range;
			glm::vec3 intensity;
			int pad;
		} pointLights[maxPointLightCount];

		struct SpotLight
		{
			glm::vec3 position;
			float range;
			glm::vec3 direction;
			float innerCutOff;
			glm::vec3 intensity;
			float outerCutOff;
		} spotLights[maxSpotLightCount];

		struct DirectionalLight
		{
			glm::vec3 direction;
			int pad;
			glm::vec3 luminance;
			int pad2;
		} directionalLights[maxDirectionalLightCount];
	};
}