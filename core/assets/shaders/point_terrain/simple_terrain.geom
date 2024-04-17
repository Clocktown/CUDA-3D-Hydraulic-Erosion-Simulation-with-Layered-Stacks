#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "vertex_to_geometry.glsl"
#include "geometry_to_fragment.glsl"
#include "../render_pipeline/render_pipeline.glsl"

layout (points) in;
in VertexToGeometry vertexToGeometry[];

layout (triangle_strip, max_vertices = 24) out;
out GeometryToFragment geometryToFragment;

void main()
{
    if (vertexToGeometry[0].valid) 
    {
        const vec2 off[4] = {vec2(0, 0), vec2(0, gridScale), vec2(gridScale, 0), vec2(gridScale, gridScale)};
        vec3 basePosition = gl_in[0].gl_Position.xyz;
        basePosition.y += vertexToGeometry[0].h[0][WATER];
        vec3 maxV;
        maxV[BEDROCK] = vertexToGeometry[0].h[0][BEDROCK];
        maxV[SAND] = vertexToGeometry[0].h[0][SAND];
        maxV[WATER] = vertexToGeometry[0].h[0][WATER];
        maxV -= vertexToGeometry[0].h[0][FLOOR];
        vec3 vertices[8];
        vec3 normals[8];
        vec4 glPos[8];
        for(int i = 0; i < 4; ++i) {
            vertices[i] = (localToWorld * vec4(basePosition + vec3(off[i].x, 0.0f, off[i].y), 1.0f)).xyz;
            normals[i] = transpose(mat3(worldToLocal)) * vertexToGeometry[0].normal[i];
            glPos[i] = worldToClip * vec4(vertices[i], 1.f);

            vertices[i+4] = (localToWorld * vec4(basePosition + vec3(off[i].x, -maxV[WATER], off[i].y), 1.0f)).xyz;
            normals[i+4] = transpose(mat3(worldToLocal)) * vertexToGeometry[0].normal[i + 4];
            glPos[i+4] = worldToClip * vec4(vertices[i+4], 1.f);
        }

        // Top Face
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[i];
            geometryToFragment.normal = normals[i];
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = maxV[WATER];

            gl_Position = glPos[i];

            EmitVertex();
        }
        EndPrimitive();

        // Left Face
        const int leftIdx[4] = {0, 4, 1, 5};
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[leftIdx[i]];
            geometryToFragment.normal = vec3(-1, 0, 0);
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = leftIdx[i] >= 4 ? 0.f : maxV[WATER];

            gl_Position = glPos[leftIdx[i]];

            EmitVertex();
        }
        EndPrimitive();

        // Right Face
        const int rightIdx[4] = {2, 3, 6, 7};
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[rightIdx[i]];
            geometryToFragment.normal = vec3(1, 0, 0);
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = rightIdx[i] >= 4 ? 0.f : maxV[WATER];

            gl_Position = glPos[rightIdx[i]];

            EmitVertex();
        }
        EndPrimitive();

        // Front Face
        const int frontIdx[4] = {0, 2, 4, 6};
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[frontIdx[i]];
            geometryToFragment.normal = vec3(0, 0, -1);
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = frontIdx[i] >= 4 ? 0.f : maxV[WATER];

            gl_Position = glPos[frontIdx[i]];

            EmitVertex();
        }
        EndPrimitive();

        // Back Face
        const int backIdx[4] = {1, 5, 3, 7};
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[backIdx[i]];
            geometryToFragment.normal = vec3(0, 0, 1);
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = backIdx[i] >= 4 ? 0.f : maxV[WATER];

            gl_Position = glPos[backIdx[i]];

            EmitVertex();
        }
        EndPrimitive();

        // Bottom Face
        const int reverseIdx[4] = {0+4,2+4,1+4,3+4};
        for(int i = 0; i < 4; ++i) {
            geometryToFragment.position = vertices[reverseIdx[i]];
            geometryToFragment.normal = normals[reverseIdx[i]];
            geometryToFragment.maxV = maxV;
            geometryToFragment.v = 0.f;

            gl_Position = glPos[reverseIdx[i]];

            EmitVertex();
        }
        EndPrimitive();
    }
}
