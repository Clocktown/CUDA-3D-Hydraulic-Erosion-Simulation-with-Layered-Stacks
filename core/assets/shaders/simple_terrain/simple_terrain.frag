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

vec2 getWaterDxy(vec3 n, vec3 p) {
	vec2 d = vec2(0.f);
	float strength = 50.f;
	float amplitude = 0.5f;

	for(int i = 0; i < 4; ++i) {
		vec2 scale = min(1.f / (strength * abs(n.xz)), vec2(1.f));
		scale = mix(scale, vec2(1.f), float(i) / 3.f);
		d.x += (length(n.xz) * amplitude + 0.025 * amplitude) * snoise(vec3(p.xz * scale.yx, time));
		d.y += (length(n.xz) * amplitude + 0.025 * amplitude) * snoise(vec3(p.zx * scale.xy, time));

		amplitude *= 0.75f;
	}
	return d;
}

PhongBRDF getPhongBRDF()
{
	PhongBRDF phongBRDF;
	phongBRDF.normal = normalize(geometryToFragment.normal);
	phongBRDF.position = geometryToFragment.position;

	//vec2 scale = min(1.f / (10.f * abs(phongBRDF.normal.xz)), vec2(1.f));
	//float dx = snoise(vec3(phongBRDF.position.xz * scale.yx, time));
	//float dy = snoise(vec3(phongBRDF.position.zx * scale.xy, time));
	vec3 waterNormal = phongBRDF.normal;
	if(geometryToFragment.maxV[WATER] > geometryToFragment.maxV[SAND] && waterNormal.y > 0.0f) {
		waterNormal.xz += getWaterDxy(waterNormal, phongBRDF.position.xyz);
	}
	waterNormal = normalize(waterNormal);

	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);
	const vec3 refractedView = refract(-viewDirection, waterNormal, 0.75);
	const float sandDistance = max(geometryToFragment.v - geometryToFragment.maxV[SAND], 0.0f);
	const float refractedDistance = abs(sandDistance / refractedView.y);

	// Todo: a) use view-direction and normal to refract view
	//		 b) calculate Intersection with plane defined by layer below water
	//		 c) use distance to that intersection point for color interpolation instead
	//		 d) same idea can be used for sand but without refracting to improve look
	phongBRDF.diffuseReflectance = mix(
		colors[BEDROCK], colors[SAND], 
		smoothstep(
			geometryToFragment.maxV[BEDROCK], 
			geometryToFragment.maxV[BEDROCK] + 0.2f, 
			min(geometryToFragment.maxV[SAND], geometryToFragment.v)
		)
	);
	vec2 causticPos = geometryToFragment.position.xz + refractedDistance * refractedView.xz;
	causticPos *= exp(0.05 * (geometryToFragment.maxV[SAND] - geometryToFragment.maxV[WATER]));
	vec2 worleyVal = worley(vec3(causticPos, time), 1.f, false);
	phongBRDF.diffuseReflectance += mix(
		0.f, 
		pow(worleyVal.x, 4.f),
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 2.f, 
				geometryToFragment.v
			)
	);
	vec3 waterColor = mix(
		phongBRDF.diffuseReflectance * 0.25 * colors[WATER], 
		0.25 * colors[WATER], 
		1 - exp(-0.1*refractedDistance)
	);
	phongBRDF.diffuseReflectance = mix(
		phongBRDF.diffuseReflectance, 
		waterColor, 
		1 - exp(-0.1*refractedDistance)
	);

	phongBRDF.specularReflectance = mix(vec3(0.f), vec3(1.f), 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.normal = mix(phongBRDF.normal, waterNormal, 
		smoothstep(
				geometryToFragment.maxV[SAND], 
				geometryToFragment.maxV[SAND] + 0.05f, 
				geometryToFragment.v
			));
	phongBRDF.normal = normalize(phongBRDF.normal);
	const float cosTheta = max(dot(viewDirection, phongBRDF.normal), 0.0f);
	phongBRDF.F = fresnelSchlick(cosTheta, 0.04);
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
