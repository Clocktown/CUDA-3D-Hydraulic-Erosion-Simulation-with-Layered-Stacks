#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "grid.glsl"
#include "vertex_to_geometry.glsl"
#include "../render_pipeline/render_pipeline.glsl"
#include "../render_pipeline/maths.glsl"
#include "../mesh_render_pipeline/mesh_render_pipeline.glsl"
#include "../mesh_render_pipeline/vertex_attributes.glsl"

out VertexToGeometry vertexToGeometry;
out flat FlatVertexToGeometry flatVertexToGeometry;

ivec3 getIndex() 
{
    ivec3 index;
    int flatIndex = gl_InstanceID;
    const int layerStride = gridSize.x * gridSize.y;

    index.z = flatIndex / layerStride;
    flatIndex -= index.z * layerStride;

    index.y = flatIndex / gridSize.x;
    flatIndex -= index.y * gridSize.x;

    index.x = flatIndex;

    return index;
}

void main() 
{
    const ivec3 index = unflattenIndex(gl_InstanceID, ivec3(gridSize, maxLayerCount));
    int flatIndex = index.x + index.y * gridSize.x;
   
    if (index.z >= layerCounts[flatIndex]) 
    {
        flatVertexToGeometry.valid = false;
        return;
    }

    flatIndex = gl_InstanceID;
    const float floor = index.z > 0 ? heights[flatIndex - gridSize.x * gridSize.y][CEIL] : 0.0f;
    const vec4 height = heights[flatIndex] - vec4(floor, 0.0f, 0.0f, 0.0f);
    const float water = height[BEDROCK] + height[SAND] + height[WATER];

    vertexToGeometry.position = vec3(gridScale, water, gridScale) * (vec3(index.x, 0.0f, index.y) + (0.5f * position + 0.5f));
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    vertexToGeometry.normal = transpose(mat3(worldToLocal)) * normal;
    vertexToGeometry.v = 0.5f * position.y + 0.5f;

    flatVertexToGeometry.maxV[BEDROCK] = height[BEDROCK] / water;
    flatVertexToGeometry.maxV[SAND] = (height[BEDROCK] + height[SAND]) / water;
    flatVertexToGeometry.maxV[WATER] = 1.0f;
    flatVertexToGeometry.valid = true;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
