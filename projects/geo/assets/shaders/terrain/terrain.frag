#version 460

#include "material.glsl"
#include "geometry_to_fragment.glsl"
#include "../renderer/renderer.glsl"
#include "../mesh_renderer/mesh_renderer.glsl"
#include "../lighting/lighting.glsl"
#include "../lighting/light.glsl"
#include "../lighting/phong_brdf.glsl"
#include "../math/constants.glsl"
#include "../math/color.glsl"

layout(early_fragment_tests) in;
in GeometryToFragment geometryToFragment;
in flat FlatGeometryToFragment flatGeometryToFragment;

layout(location = 0) out vec4 fragmentColor;

PhongBRDF getPhongBRDF()
{
    int cellType;

	for (cellType = bedrockIndex; cellType < waterIndex; ++cellType) 
	{
		if (geometryToFragment.v <= flatGeometryToFragment.maxV[cellType]) 
	    {
	        break;
	    }
	}

	if (geometryToFragment.v > flatGeometryToFragment.maxV[cellType]) 
	{
	    discard;
	}

	PhongBRDF phongBRDF;
	phongBRDF.diffuseColor = sRGBToLinear(color[cellType].rgb);
	phongBRDF.diffuseReflectance = 0.99f;
	phongBRDF.specularColor = vec3(1.0f);
	phongBRDF.specularReflectance = 0.0f;
	phongBRDF.shininess = 40.0f;
	phongBRDF.position = geometryToFragment.position;
	phongBRDF.normal = normalize(geometryToFragment.normal);

	return phongBRDF;
}

void main()
{
    const PhongBRDF phongBRDF = getPhongBRDF();
	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);

	vec3 radiance = 0.01f * phongBRDF.diffuseColor;

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
