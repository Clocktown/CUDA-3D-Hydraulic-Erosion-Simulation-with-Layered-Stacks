#ifndef ONEC_RENDER_PIPELINE_GLSL
#define ONEC_RENDER_PIPELINE_GLSL

layout(binding = 0, std140) uniform RenderPipelineBuffer
{
	float time;
	float deltaTime;
	ivec2 viewportSize;
	mat4 worldToView;
	mat4 viewToWorld;
	mat4 viewToClip;
	mat4 clipToView;
	mat4 worldToClip;
	mat4 clipToWorld;
};

#endif
