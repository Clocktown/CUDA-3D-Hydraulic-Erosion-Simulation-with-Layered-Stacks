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

const ivec2 offsets[2][2] = {
    {ivec2(1,1), ivec2(1,-1)}, 
    {ivec2(-1,1), ivec2(-1,-1)}
};

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

vec4 getInterpolatedHeight(const in vec2 off, const in ivec3 index, const in int layerStride, const in vec4 height, const in float floor, const in vec4 prev) {
    // Greatly simplified, does not search for correct neighbor yet and simply uses the one on the same layer + only x-direction + no normals
    ivec3 idx = index + ivec3(off.x, 0, 0);
    if(idx.x >= gridSize.x || idx.x < 0) {
        return prev;
    }
    int flatIdx = idx.x + idx.y * gridSize.x;
    if(idx.z >= getLayerCount(flatIdx)) {
        return prev;
    }
    flatIdx += index.z * layerStride;

    const vec4 nheight = heights[flatIdx];
    float nfloor = index.z > 0 ? heights[flatIdx - layerStride][CEILING] : 0.0f;
    nfloor = 0.5f * (floor + nfloor);
    float bedrock = 0.5f * (nheight[BEDROCK] + height[BEDROCK]) - nfloor;
    float sand = bedrock + 0.5f * (nheight[SAND] + height[SAND]);
    float water = sand + 0.5f * (nheight[WATER] + height[WATER]);

    vec4 ret = vec4(0);
    ret[CEILING] = nfloor;
    ret[BEDROCK] = bedrock;
    ret[SAND] = sand;
    ret[WATER] = water;

    return ret;
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
    flatVertexToGeometry.stability = stability[flatIndex];
    float floor = index.z > 0 ? heights[flatIndex - layerStride][CEILING] : 0.0f;
    float bedrock = height[BEDROCK] - floor;
    float sand = bedrock + height[SAND];
    float water = sand + height[WATER];

    if(useInterpolation) {
        vec4 prev = vec4(0);
        prev[CEILING] = floor;
        prev[BEDROCK] = bedrock;
        prev[SAND] = sand;
        prev[WATER] = water;

        vec2 off = offsets[int(position.x < 0.5f)][int(position.z < 0.5f)];
        vec4 interpolatedHeight = getInterpolatedHeight(off, index, layerStride, height, floor, prev);
        floor = interpolatedHeight[CEILING];
        bedrock = interpolatedHeight[BEDROCK];
        sand = interpolatedHeight[SAND];
        water = interpolatedHeight[WATER];
    }
   
    vertexToGeometry.position = vec3(gridScale, water, gridScale) * (vec3(index.x, 0.0f, index.y) + (0.5f * position + 0.5f));
    vertexToGeometry.position.y += floor;
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    vertexToGeometry.normal = transpose(mat3(worldToLocal)) * normal;
    vertexToGeometry.v = water * (0.5f * position.y + 0.5f);

    // Adding small epsilon to avoid issues with 0 water
    vertexToGeometry.maxV[BEDROCK] = bedrock;
    vertexToGeometry.maxV[SAND] = sand;
    vertexToGeometry.maxV[WATER] = water;
    flatVertexToGeometry.valid = true;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
