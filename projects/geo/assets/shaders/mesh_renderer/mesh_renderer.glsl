#ifndef ONEC_MESH_RENDERER_GLSL
#define ONEC_MESH_RENDERER_GLSL

const unsigned int materialBufferLocation = 3;

layout(binding = 2, std140) uniform MeshRendererBuffer
{
	mat4 localToWorld;
	mat4 worldToLocal;
};

#endif
