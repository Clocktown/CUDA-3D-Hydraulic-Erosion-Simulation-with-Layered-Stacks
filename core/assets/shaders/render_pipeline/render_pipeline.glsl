#ifndef ONEC_RENDER_PIPELINE_GLSL
#define ONEC_RENDER_PIPELINE_GLSL

#include "light.glsl"

const int maxPointLightCount = 256;
const int maxSpotLightCount = 256;
const int maxDirectionalLightCount = 256;

layout(binding = 0, std140) uniform RenderPipelineUniforms
{
	mat4 worldToView;
	mat4 viewToWorld;
	mat4 viewToClip;
	mat4 clipToView;
	mat4 worldToClip;
	mat4 clipToWorld;
	float time;
	float deltaTime;
	ivec2 viewportSize;
	float exposure;
	int pointLightCount;
	int spotLightCount;
	int directionalLightCount;
	AmbientLight ambientLight;
    PointLight pointLights[maxPointLightCount];
	SpotLight spotLights[maxSpotLightCount];
    DirectionalLight directionalLights[maxDirectionalLightCount];
};

#endif
