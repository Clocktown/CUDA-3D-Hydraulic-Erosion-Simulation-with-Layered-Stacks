#ifndef GEO_MATERIAL_GLSL
#define GEO_MATERIAL_GLSL

#include "../mesh_renderer/mesh_renderer.glsl"

const int bedrockIndex = 0;
const int sandIndex = 1;
const int waterIndex = 2;

layout(binding = materialBufferLocation, std140) uniform MaterialBuffer
{
	vec4 color[3];
	ivec2 gridSize;
	float gridScale;
	int maxLayerCount;
};

layout(binding = 0) uniform sampler3D heightMap;

#endif
