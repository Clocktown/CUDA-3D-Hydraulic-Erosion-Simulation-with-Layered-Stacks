#ifndef GEO_SIMPLE_MATERIAL_GLSL
#define GEO_SIMPLE_MATERIAL_GLSL

#include "../mesh_render_pipeline/mesh_render_pipeline.glsl"

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define CEILING 3

layout(binding = materialBufferLocation, std140) uniform SimpleMaterialUniforms
{
	vec3 colors[3];
    ivec2 gridSize;
    float gridScale;
	int maxLayerCount;
	bool useInterpolation;
	int* layerCounts;
	vec4* heights;
	float* stability;
};

#endif
