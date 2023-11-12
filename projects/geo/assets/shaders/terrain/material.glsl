#ifndef GEO_MATERIAL_GLSL
#define GEO_MATERIAL_GLSL

#include "../mesh_renderer/mesh_renderer.glsl"

const int belowIndex = 0;
const int aboveIndex = 1;

const int bedrockIndex = 0;
const int sandIndex = 1;
const int waterIndex = 2;
const int maxHeightIndex = 3;

layout(binding = materialBufferLocation, std140) uniform MaterialBuffer
{
	ivec2 gridSize;
	float gridScale;
	int maxLayerCount;
	vec4 color[3];
};

layout(binding = 0) uniform isampler3D infoMap;
layout(binding = 1) uniform sampler3D heightMap;
layout(binding = 2) uniform sampler3D waterVelocityMap;

#endif
