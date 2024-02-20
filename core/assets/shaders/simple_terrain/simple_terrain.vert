#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "vertex_to_geometry.glsl"
#include "grid.glsl"
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

int getLayerCount(const int flatIndex) 
{
    const int packedIndex = flatIndex / 4;
    const int packedPosition = flatIndex - 4 * packedIndex; 
    const int packedValue = layerCounts[packedIndex];

    return (packedValue >> (8 * packedPosition)) & 0xff;
}

void main() 
{
    const ivec3 index = unflattenIndex(gl_InstanceID, ivec3(gridSize, maxLayerCount));
    int flatIndex = index.x + index.y * gridSize.x;
   
    if (index.z >= getLayerCount(flatIndex)) 
    {
        flatVertexToGeometry.valid = false;
        return;
    }

    flatIndex = gl_InstanceID;
    const int layerStride = gridSize.x * gridSize.y;

    const vec4 height = heights[flatIndex];
    const float floor = index.z > 0 ? heights[flatIndex - layerStride][CEILING] : 0.0f;
    const float bedrock = height[BEDROCK] - floor;
    const float sand = bedrock + height[SAND];
    const float water = sand + height[WATER];
   
    vertexToGeometry.position = vec3(gridScale, water, gridScale) * (vec3(index.x, 0.0f, index.y) + (0.5f * position + 0.5f));
    vertexToGeometry.position.y += height[BEDROCK];
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    vertexToGeometry.normal = transpose(mat3(worldToLocal)) * normal;
    vertexToGeometry.v = 0.5f * position.y + 0.5f;

    flatVertexToGeometry.maxV[BEDROCK] = bedrock / water;
    flatVertexToGeometry.maxV[SAND] = sand / water;
    flatVertexToGeometry.maxV[WATER] = 1.0f;
    flatVertexToGeometry.valid = true;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
