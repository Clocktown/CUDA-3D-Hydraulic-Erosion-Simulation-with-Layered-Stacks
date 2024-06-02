#ifndef GEO_SIMPLE_MATERIAL_GLSL
#define GEO_SIMPLE_MATERIAL_GLSL

#include "../point_render_pipeline/point_render_pipeline.glsl"

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define CEILING 3
#define FLOOR 3

layout(binding = materialBufferLocation, std140) uniform SimpleMaterialUniforms
{
	vec3 colors[3];

    ivec2 gridSize;
    float gridScale;
	int maxLayerCount;

	bool useInterpolation;
	bool renderSand;
	bool renderWater;

	int* layerCounts;
	vec4* heights;
	float* stability;
	int* indices;
};

#endif
