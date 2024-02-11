#ifndef ONEC_LIGHT_GLSL
#define ONEC_LIGHT_GLSL

#include "maths.glsl"

struct AmbientLight
{
	vec3 illuminance;
};

struct PointLight
{
	vec3 position;
	float range;
	vec3 intensity;
};

struct SpotLight
{
	vec3 position;
	float range;
	vec3 direction;
	float innerCutOff;
	vec3 intensity;
	float outerCutOff;
};

struct DirectionalLight
{
	vec3 direction;
	vec3 luminance;
};

float getAttenuation(const float distance, const float range)
{
	float ratio = distance / range;
	ratio = clamp(1.0f - ratio * ratio * ratio * ratio, 0.0f, 1.0f);

	return ratio * ratio / (distance * distance + 1.0f);
}

vec3 getPointLightIlluminance(const PointLight pointLight, const vec3 direction, const float distance, const vec3 normal)
{
	return getAttenuation(distance, pointLight.range) * pointLight.intensity * max(dot(-direction, normal), 0.0f);
}

vec3 getSpotLightIlluminance(const SpotLight spotLight, const vec3 direction, const float distance, const vec3 normal)
{
	const float cutOff = dot(spotLight.direction, direction);

	if (cutOff < spotLight.outerCutOff)
	{
		return vec3(0.0f);
	}

	const float attenuation = min((cutOff - spotLight.outerCutOff) / (spotLight.innerCutOff - spotLight.outerCutOff + epsilon), 1.0f) * getAttenuation(distance, spotLight.range);

	return attenuation * spotLight.intensity * max(dot(-direction, normal), 0.0f);
}

vec3 getDirectionalLightIlluminance(const DirectionalLight directionalLight, const vec3 direction, const vec3 normal)
{
	return directionalLight.luminance * max(dot(-direction, normal), 0.0f);
}

#endif
