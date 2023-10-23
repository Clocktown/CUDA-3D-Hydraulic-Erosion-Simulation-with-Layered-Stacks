#ifndef ONEC_PHONG_BRDF_GLSL
#define ONEC_PHONG_BRDF_GLSL

#include "light.glsl"
#include "../math/constants.glsl"

struct PhongBRDF
{
	vec3 diffuseColor;
	vec3 specularColor;
	float specularReflectance;
	float shininess;
	vec3 emissionColor;
	float emissionStrength;
	float alpha;
	vec3 position;
	vec3 normal;
};

vec3 getPhongReflectance(const PhongBRDF phongBRDF, const vec3 lightDirection, const vec3 viewDirection)
{
	const vec3 reflection = reflect(-lightDirection, phongBRDF.normal);
	const float cosPhi = max(dot(viewDirection, reflection), 0.0f);

	const vec3 diffuseBRDF = rPi * phongBRDF.diffuseColor;
	const vec3 specularBRDF = 0.5f * rPi * (phongBRDF.shininess + 2.0f) * pow(cosPhi, phongBRDF.shininess) * phongBRDF.specularColor;

	return mix(diffuseBRDF, specularBRDF, phongBRDF.specularReflectance);
}

vec3 getPointLightRadiance(const PointLight pointLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightVector = pointLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const vec3 radiance = getPhongReflectance(phongBRDF, lightDirection, direction) *
		                  getPointLightIrradiance(pointLight, -lightDirection, lightDistance, phongBRDF.normal);

	return radiance;
}

vec3 getSpotLightRadiance(const SpotLight spotLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightVector = spotLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const vec3 radiance = getPhongReflectance(phongBRDF, lightDirection, direction) *
		                  getSpotLightIrradiance(spotLight, -lightDirection, lightDistance, phongBRDF.normal);

	return radiance;
}

vec3 getDirectionalLightRadiance(const DirectionalLight directionalLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightDirection = -directionalLight.direction;

	const vec3 radiance = getPhongReflectance(phongBRDF, lightDirection, direction) *
		                  getDirectionalLightIrradiance(directionalLight, directionalLight.direction, phongBRDF.normal);

	return radiance;
}

vec3 getEmissionRadiance(const PhongBRDF phongBRDF, const vec3 direction)
{
	return max(dot(phongBRDF.normal, direction), 0.0f) * phongBRDF.emissionStrength * phongBRDF.emissionColor;
}

#endif
