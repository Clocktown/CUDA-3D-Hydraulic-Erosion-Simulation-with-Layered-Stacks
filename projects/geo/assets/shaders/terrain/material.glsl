#ifndef GEO_MATERIAL_GLSL
#define GEO_MATERIAL_GLSL

#include "../mesh_renderer/mesh_renderer.glsl"

#define BELOW 0
#define ABOVE 1

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define MAX_HEIGHT 3

layout(binding = materialBufferLocation, std140) uniform MaterialBuffer
{
	vec4 color[3];
    ivec3 gridSize;
    float gridScale;
	isampler3D infoMap;
	sampler3D heightMap;
};

#endif
