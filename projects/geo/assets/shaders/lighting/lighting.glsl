#ifndef ONEC_LIGHTING_GLSL
#define ONEC_LIGHTING_GLSL

#include "light.glsl"

const unsigned int maxPointLightCount = 128;
const unsigned int maxSpotLightCount = 128;
const unsigned int maxDirectionalLightCount = 4;

layout(binding = 1, std140) uniform LightingBuffer
{
	PointLight pointLights[maxPointLightCount];
	SpotLight spotLights[maxSpotLightCount];
	DirectionalLight directionalLights[maxDirectionalLightCount];
	AmbientLight ambientLight;
	int pointLightCount;
	int spotLightCount;
	int directionalLightCount;
};

#endif
