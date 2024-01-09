#ifndef ONEC_ENVIRONMENT_GLSL
#define ONEC_ENVIRONMENT_GLSL

#include "light.glsl"

const int maxPointLightCount = 256;
const int maxSpotLightCount = 256;
const int maxDirectionalLightCount = 256;

layout(binding = 1, std140) uniform EnvironmentBuffer
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
