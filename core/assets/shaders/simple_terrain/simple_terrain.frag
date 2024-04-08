#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "geometry_to_fragment.glsl"
#include "../render_pipeline/render_pipeline.glsl"
#include "../render_pipeline/light.glsl"
#include "../render_pipeline/phong_brdf.glsl"
#include "../render_pipeline/maths.glsl"
#include "../render_pipeline/color.glsl"
#include "../mesh_render_pipeline/mesh_render_pipeline.glsl"

layout(early_fragment_tests) in;
in GeometryToFragment geometryToFragment;
in flat FlatGeometryToFragment flatGeometryToFragment;

layout(location = 0) out vec4 fragmentColor;

float fresnelSchlick(float cosTheta, float F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

PhongBRDF getPhongBRDF()
{
	PhongBRDF phongBRDF;
	phongBRDF.normal = normalize(geometryToFragment.normal);
	phongBRDF.position = geometryToFragment.position;

	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);
	const float cosTheta = max(dot(viewDirection, phongBRDF.normal), 0.0f);
	phongBRDF.F = fresnelSchlick(cosTheta, 0.04);

	// Todo: a) use view-direction and normal to refract view
	//		 b) calculate Intersection with plane defined by layer below water
	//		 c) use distance to that intersection point for color interpolation instead
	//		 d) same idea can be used for sand but without refracting to improve look
	phongBRDF.diffuseReflectance = mix(
		colors[BEDROCK], colors[SAND], 
		smoothstep(
			geometryToFragment.maxV[BEDROCK], 
			geometryToFragment.maxV[BEDROCK] + 0.2f, 
			geometryToFragment.v
		)
	);
	phongBRDF.diffuseReflectance = mix(
		phongBRDF.diffuseReflectance, phongBRDF.diffuseReflectance * colors[WATER], 
		smoothstep(
			geometryToFragment.maxV[SAND], 
			geometryToFragment.maxV[SAND] + 2.f, 
			geometryToFragment.v
		)
	);
	phongBRDF.diffuseReflectance = mix(
		phongBRDF.diffuseReflectance, colors[WATER]	* 0.25, 
		smoothstep(
			geometryToFragment.maxV[SAND] + 2.f, 
			geometryToFragment.maxV[SAND] + 3.f, 
			geometryToFragment.v
		)
	);

	phongBRDF.specularReflectance = mix(vec3(0.f), vec3(1.f), 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.F = mix(0.f, phongBRDF.F, 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.diffuseReflectance *= 1.f - phongBRDF.F;
	phongBRDF.shininess = mix(40.f, 10000.f, 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.alpha = 1.0f;

	return phongBRDF;
}

void main()
{
    const PhongBRDF phongBRDF = getPhongBRDF();

    fragmentColor.rgb = linearToSRGB(applyReinhardToneMap(shadePhongBRDF(phongBRDF) + exposure * phongBRDF.F * ambientLight.illuminance));
	fragmentColor.a = phongBRDF.alpha;
}
