#ifndef ONEC_LIGHT_GLSL
#define ONEC_LIGHT_GLSL

#include "../math/constants.glsl"

struct PointLight
{
	vec3 position;
	float intensity;
	vec3 color;
	float range;
};

struct SpotLight
{
	vec3 position;
	float intensity;
	vec3 direction;
	float range;
	vec3 color;
	float innerCutOff;
	float outerCutOff;
};

struct DirectionalLight
{
	vec3 direction;
	float strength;
	vec3 color;
};

vec3 applyReinhardToneMap(const vec3 radiance)
{
	return radiance / (radiance + 1.0f);
}

float getLightAttenuation(const float distance, const float range)
{
	float tmp = distance / range;
	tmp = clamp(1.0f - tmp * tmp * tmp * tmp, 0.0f, 1.0f);

	return tmp * tmp / (distance * distance + 1.0f);
}

vec3 getPointLightIntensity(const PointLight pointLight)
{
	return pointLight.intensity * pointLight.color;
}

vec3 getPointLightIrradiance(const PointLight pointLight, const vec3 direction, const float distance, const vec3 normal)
{
	return max(dot(-direction, normal), 0.0f) * getLightAttenuation(distance, pointLight.range) * getPointLightIntensity(pointLight);
}

vec3 getSpotLightIntensity(const SpotLight spotLight, const vec3 direction)
{
	const float cutOff = dot(direction, spotLight.direction);

	if (cutOff >= spotLight.outerCutOff)
	{
		const float attenuation = min((cutOff - spotLight.outerCutOff) / (spotLight.innerCutOff - spotLight.outerCutOff + epsilon), 1.0f);
		return attenuation * spotLight.intensity * spotLight.color;
	}

	return vec3(0.0f);
}

vec3 getSpotLightIrradiance(const SpotLight spotLight, const vec3 direction, const float distance, const vec3 normal)
{
	return max(dot(-direction, normal), 0.0f) * getLightAttenuation(distance, spotLight.range) * getSpotLightIntensity(spotLight, direction);
}

vec3 getDirectionalLightIrradiance(const DirectionalLight directionalLight, const vec3 direction, const vec3 normal)
{
	return max(dot(-direction, normal), 0.0f) * directionalLight.strength * directionalLight.color;
}

#endif
