#ifndef ONEC_SCREEN_TEXTURE_RENDER_PIPELINE_GLSL
#define ONEC_SCREEN_TEXTURE_RENDER_PIPELINE_GLSL

const unsigned int materialBufferLocation = 2;

layout(binding = 1, std140) uniform PointRenderPipelineUniforms
{
	int unused;
};

layout(binding = 0) uniform sampler2D screenTexture;

#endif
