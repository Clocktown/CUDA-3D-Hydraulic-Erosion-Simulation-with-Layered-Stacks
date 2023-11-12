#version 460

#include "mesh_renderer.glsl"
#include "vertex_attributes.glsl"
#include "vertex_to_fragment.glsl"
#include "../renderer/renderer.glsl"

out VertexToFragment vertexToFragment;

void main() 
{
    const mat3 normalMatrix = mat3(transpose(worldToLocal));

    vertexToFragment.position = vec3(localToWorld * vec4(position.xyz, 1.0f));
    vertexToFragment.normal = normalize(normalMatrix * normal); 
    vertexToFragment.tangent = normalize(normalMatrix * tangent);
    vertexToFragment.bitangent = cross(vertexToFragment.normal, vertexToFragment.tangent);
    vertexToFragment.uv = uv;

    gl_Position = worldToClip * vec4(vertexToFragment.position, 1.0f);
}
