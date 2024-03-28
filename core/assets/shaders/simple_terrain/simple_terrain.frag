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

PhongBRDF getPhongBRDF()
{
	PhongBRDF phongBRDF;
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
			geometryToFragment.maxV[SAND] + 1.f, 
			geometryToFragment.v
		)
	);
	phongBRDF.diffuseReflectance = mix(
		phongBRDF.diffuseReflectance, colors[WATER]	* 0.25, 
		smoothstep(
			geometryToFragment.maxV[SAND] + 1.f, 
			geometryToFragment.maxV[SAND] + 2.f, 
			geometryToFragment.v
		)
	);

	phongBRDF.specularReflectance = mix(vec3(0.f), vec3(1.f), 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.shininess = mix(40.f, 10000.f, 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.alpha = 1.0f;
	phongBRDF.position = geometryToFragment.position;
	phongBRDF.normal = normalize(geometryToFragment.normal);

	return phongBRDF;
}

void main()
{
    const PhongBRDF phongBRDF = getPhongBRDF();

    fragmentColor.rgb = linearToSRGB(applyReinhardToneMap(shadePhongBRDF(phongBRDF)));
	fragmentColor.a = phongBRDF.alpha;
}
