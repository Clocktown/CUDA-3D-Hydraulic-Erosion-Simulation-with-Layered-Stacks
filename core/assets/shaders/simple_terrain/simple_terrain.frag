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
    int cellType = BEDROCK;

	while (geometryToFragment.v > geometryToFragment.maxV[cellType] && cellType++ < WATER) 
	{

	}

	PhongBRDF phongBRDF;
	phongBRDF.diffuseReflectance = colors[cellType];
	if(cellType == BEDROCK) {
		if(flatGeometryToFragment.stability <= 0.f) {
			phongBRDF.diffuseReflectance = vec3(1,0,0);
		}
	}
	phongBRDF.specularReflectance = vec3(0.0f);
	phongBRDF.shininess = 40.0f;
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
