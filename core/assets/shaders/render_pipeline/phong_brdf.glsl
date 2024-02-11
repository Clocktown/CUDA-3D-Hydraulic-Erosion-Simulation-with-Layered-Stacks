#ifndef ONEC_PHONG_BRDF_GLSL
#define ONEC_PHONG_BRDF_GLSL

#include "light.glsl"
#include "maths.glsl"

struct PhongBRDF
{
	vec3 diffuseReflectance;
	vec3 specularReflectance;
	float shininess;
	float alpha;
	vec3 position;
	vec3 normal;
};

vec3 evaluatePhongBRDF(const PhongBRDF phongBRDF, const vec3 lightDirection, const vec3 viewDirection)
{
	const vec3 reflection = reflect(-lightDirection, phongBRDF.normal);
	const float cosPhi = max(dot(viewDirection, reflection), 0.0f);

	const vec3 diffuseBRDF = rPi * phongBRDF.diffuseReflectance;
	const vec3 specularBRDF = 0.5f * rPi * (phongBRDF.shininess + 2.0f) * pow(cosPhi, phongBRDF.shininess) * phongBRDF.specularReflectance;

	return diffuseBRDF + specularBRDF;
}

vec3 getAmbientLightLuminance(const AmbientLight ambientLight, const PhongBRDF phongBRDF)
{
	return rPi * phongBRDF.diffuseReflectance * ambientLight.illuminance;
}

vec3 getPointLightLuminance(const PointLight pointLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightVector = pointLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getPointLightIlluminance(pointLight, -lightDirection, lightDistance, phongBRDF.normal);

	return luminance;
}

vec3 getSpotLightLuminance(const SpotLight spotLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightVector = spotLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getSpotLightIlluminance(spotLight, -lightDirection, lightDistance, phongBRDF.normal);

	return luminance;
}

vec3 getDirectionalLightLuminance(const DirectionalLight directionalLight, const PhongBRDF phongBRDF, const vec3 direction)
{
	const vec3 lightDirection = -directionalLight.direction;
	const vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getDirectionalLightIlluminance(directionalLight, directionalLight.direction, phongBRDF.normal);

	return luminance;
}

vec3 shadePhongBRDF(const PhongBRDF phongBRDF)
{
	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);
	vec3 luminance = getAmbientLightLuminance(ambientLight, phongBRDF);

	for (int i = 0; i < pointLightCount; ++i)
	{
		luminance += getPointLightLuminance(pointLights[i], phongBRDF, viewDirection);
	}

	for (int i = 0; i < spotLightCount; ++i)
	{
		luminance += getSpotLightLuminance(spotLights[i], phongBRDF, viewDirection);
	}

	for (int i = 0; i < directionalLightCount; ++i)
	{
		luminance += getDirectionalLightLuminance(directionalLights[i], phongBRDF, viewDirection);
	}

	return exposure * luminance;
}

#endif
