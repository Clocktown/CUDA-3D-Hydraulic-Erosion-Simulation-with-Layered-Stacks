#version 460

#include "mesh_renderer.glsl"
#include "vertex_to_fragment.glsl"
#include "../renderer/renderer.glsl"
#include "../lighting/lighting.glsl"
#include "../lighting/light.glsl"
#include "../lighting/phong_brdf.glsl"
#include "../math/color.glsl"

const unsigned int diffuseColorMapBit = 1;
const unsigned int specularColorMapBit = 2;
const unsigned int emissionColorMapBit = 4;
const unsigned int shininessMapBit = 8;
const unsigned int normalMapBit = 16;

layout(binding = materialBufferLocation, std140) uniform MaterialBuffer
{
	vec3 diffuseColor;
    float alpha;
	vec3 specularColor;
	float specularReflectance;
	vec3 emissionColor;
	float emissionStrength;
	float shininess;
    unsigned int maps;
};

layout(binding = 0) uniform sampler2D diffuseColorMap;
layout(binding = 1) uniform sampler2D specularColorMap;
layout(binding = 2) uniform sampler2D shininessMap;
layout(binding = 2) uniform sampler2D emissionColorMap;
layout(binding = 3) uniform sampler2D normalMap;

layout(early_fragment_tests) in;
in VertexToFragment vertexToFragment;

layout(location = 0) out vec4 fragmentColor;

PhongBRDF getPhongBRDF()
{
	PhongBRDF phongBRDF;

	if ((maps & diffuseColorMapBit) != 0)
	{
		phongBRDF.diffuseColor = diffuseColor * sRGBToLinear(texture(diffuseColorMap, vertexToFragment.uv).rgb);
	}
	else
	{
		phongBRDF.diffuseColor = diffuseColor;
	}

	if ((maps & specularColorMapBit) != 0)
	{
		phongBRDF.specularColor = specularColor * sRGBToLinear(texture(specularColorMap, vertexToFragment.uv).rgb);
	}
	else
	{
		phongBRDF.specularColor = specularColor;
	}

	phongBRDF.specularReflectance = specularReflectance;

	if ((maps & shininessMapBit) != 0)
	{
		phongBRDF.shininess = 1.0f + 149.0f * texture(shininessMap, vertexToFragment.uv).r;
	}
	else
	{
		phongBRDF.shininess = shininess;
	}

	if ((maps & emissionColorMapBit) != 0)
	{
		phongBRDF.emissionColor = emissionColor * sRGBToLinear(texture(emissionColorMap, vertexToFragment.uv).rgb);
	}
	else
	{
		phongBRDF.emissionColor = emissionColor;
	}

	phongBRDF.emissionStrength = emissionStrength;
	phongBRDF.alpha = alpha;
	phongBRDF.position = vertexToFragment.position;
	phongBRDF.normal = normalize(vertexToFragment.normal);

	if ((maps & normalMapBit) != 0)
	{
		const vec3 tangent = normalize(vertexToFragment.tangent);
		const vec3 bitangent = normalize(vertexToFragment.bitangent);
		const mat3 tangentToWorld = mat3(tangent, bitangent, phongBRDF.normal);

		phongBRDF.normal = normalize(tangentToWorld * (2.0f * texture(normalMap, vertexToFragment.uv).rgb - 1.0f));
	}

	return phongBRDF;
}

void main()
{
    const PhongBRDF phongBRDF = getPhongBRDF();
	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);

	vec3 radiance = getEmissionRadiance(phongBRDF, viewDirection);

	for (int i = 0; i < pointLightCount; ++i)
	{
		radiance += getPointLightRadiance(pointLights[i], phongBRDF, viewDirection);
	}

	for (int i = 0; i < spotLightCount; ++i)
	{
        radiance += getSpotLightRadiance(spotLights[i], phongBRDF, viewDirection);
	}

	for (int i = 0; i < directionalLightCount; ++i)
	{
        radiance += getDirectionalLightRadiance(directionalLights[i], phongBRDF, viewDirection); 
    }

    fragmentColor.rgb = linearToSRGB(applyReinhardToneMap(radiance));
	fragmentColor.a = phongBRDF.alpha;
}
