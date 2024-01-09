#version 460
#extension GL_ARB_bindless_texture : require

#include "material.glsl"
#include "geometry_to_fragment.glsl"
#include "../render_pipeline/render_pipeline.glsl"
#include "../mesh_render_pipeline/mesh_render_pipeline.glsl"
#include "../environment/environment.glsl"
#include "../environment/light.glsl"
#include "../environment/phong_brdf.glsl"
#include "../math/constants.glsl"
#include "../math/color.glsl"

layout(early_fragment_tests) in;
in GeometryToFragment geometryToFragment;
in flat FlatGeometryToFragment flatGeometryToFragment;

layout(location = 0) out vec4 fragmentColor;

PhongBRDF getPhongBRDF()
{
    int cellType = BEDROCK;

	while (geometryToFragment.v > flatGeometryToFragment.maxV[cellType] + epsilon && cellType++ < WATER) 
	{

	}

	PhongBRDF phongBRDF;
	phongBRDF.diffuseColor = sRGBToLinear(color[cellType].xyz);
	phongBRDF.diffuseReflectance = 0.96f;
	phongBRDF.specularColor = vec3(1.0f);
	phongBRDF.specularReflectance = 0.0f;
	phongBRDF.ambientReflectance = 0.04f;
	phongBRDF.shininess = 40.0f;
	phongBRDF.position = geometryToFragment.position;
	phongBRDF.normal = normalize(geometryToFragment.normal);

	return phongBRDF;
}

void main()
{
    const PhongBRDF phongBRDF = getPhongBRDF();
	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);

	vec3 radiance = getAmbientLightRadiance(ambientLight, phongBRDF);

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
	fragmentColor.a = 1.0f;
}
