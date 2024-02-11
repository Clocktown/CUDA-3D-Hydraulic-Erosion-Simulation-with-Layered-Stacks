#ifndef ONEC_MESH_RENDER_PIPELINE_GLSL
#define ONEC_MESH_RENDER_PIPELINE_GLSL

const unsigned int materialBufferLocation = 2;

layout(binding = 1, std140) uniform MeshRenderPipelineUniforms
{
	mat4 localToWorld;
	mat4 worldToLocal;
};

#endif
