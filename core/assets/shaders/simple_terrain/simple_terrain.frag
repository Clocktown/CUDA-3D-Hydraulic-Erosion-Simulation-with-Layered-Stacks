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

//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0. + 0.0 * C 
  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
  i = mod(i, 289.0 ); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

float fresnelSchlick(float cosTheta, float F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

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
	waterNormal.xz += getWaterDxy(waterNormal, phongBRDF.position.xyz);
	waterNormal = normalize(waterNormal);

	const vec3 viewDirection = normalize(viewToWorld[3].xyz - phongBRDF.position);

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
