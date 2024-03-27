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

vec4 getAbsoluteHeight(const in int flatIndex, int layerStride, int z) {
    vec4 absoluteHeights = heights[flatIndex];
    absoluteHeights[FLOOR] = z > 0 ? heights[flatIndex - layerStride][CEILING] : 0.0f;
    absoluteHeights[SAND] += absoluteHeights[BEDROCK];
    absoluteHeights[WATER] += absoluteHeights[SAND];
    return absoluteHeights;
}

vec4 makeRelative(const in vec4 absoluteHeights) {
    vec4 rel = absoluteHeights;
    rel[BEDROCK] -= rel[FLOOR];
    rel[SAND] -= rel[FLOOR];
    rel[WATER] -= rel[FLOOR];
    return rel;
}

vec3 calculateNormal(int numCells, bvec4 isValid, vec4 top, ivec2 off) {
    const ivec3 offs[4] = {ivec3(0), ivec3(off.x, 0, 0), ivec3(0, 0, off.y), ivec3(off.x, 0, off.y)};
    vec3 o = vec3(0, top[0], 0);
    vec3 n = vec3(0,1,0);

    if(numCells == 2) {
        vec3 t = vec3(0);
        for(int i = 1; i < 4; ++i) {
            if(isValid[i]) {
                t = gridScale * offs[i];
                t.y = top[i];
                break;
            }
        }

        if(t.x != 0.f) {
            n = cross(o - t, vec3(0,0,gridScale));
        } else {
            n = cross(o - t, vec3(gridScale, 0, 0));
        }
    } else if(numCells == 3) {
        vec3 t[2] = {vec3(0), vec3(0)};
        int j = 0;

        for(int i = 1; i < 4; ++i) {
            if(isValid[i]) {
                t[j] = gridScale * offs[i];
                t[j++].y = top[i];
            }
        }

        n = cross(o - t[0], o - t[1]);

    } else if(numCells == 4) {
        vec3 t[3] = {vec3(0), vec3(0), vec3(0)};
        for(int i = 1; i < 4; ++i) {
            t[i - 1] = offs[i] * gridScale;
            t[i - 1].y = top[i];
        }

        vec3 v1 = 0.5f * (o - t[0] + t[1] - t[2]);
        vec3 v2 = 0.5f * (o - t[1] + t[0] - t[2]);
        n = cross(v1, v2);
    }

    n = normalize(n);
    n *= sign(n.y);
    return n;
}

vec4 getRelativeHeight(const in ivec2 off, const in ivec3 index, const in int layerStride, const in vec4 absoluteHeight, inout vec3 newNormal) {
    // Greatly simplified, does not search for correct neighbor yet (just uses what is on the same layer)
    if(!useInterpolation) {
        return makeRelative(absoluteHeight);
    }
    const ivec3 offs[3] = {ivec3(off.x, 0, 0), ivec3(0, off.y, 0), ivec3(off.x, off.y, 0)};
    bvec4 isValid = bvec4(true, false, false, false);
    vec4 top = vec4(0);
    vec4 bot = vec4(0);
    top[0] = absoluteHeight[WATER];
    bot[0] = absoluteHeight[FLOOR];
    vec4 cumulativeHeight = absoluteHeight;
    int numCells = 1;

    for(int i = 0; i < 3; ++i) {
        ivec3 idx = index + offs[i];
        if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
            continue;
        }
        int flatIdx = idx.x + idx.y * gridSize.x;
        if(idx.z >= getLayerCount(flatIdx)) {
            continue;
        }
        flatIdx += index.z * layerStride;

        vec4 h = getAbsoluteHeight(flatIdx, layerStride, idx.z);
        cumulativeHeight += h;
        top[i+1] = h[WATER];
        bot[i+1] = h[FLOOR];
        isValid[i+1] = true;
        
        numCells++;
    }

    if(newNormal.y >= 0.5f && numCells > 1) {
       newNormal = calculateNormal(numCells, isValid, top, off);
    } else if(newNormal.y <= -0.5f && numCells > 1) {
       newNormal = -calculateNormal(numCells, isValid, bot, off);
    }

    return makeRelative(cumulativeHeight / numCells);
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

    flatVertexToGeometry.stability = stability[flatIndex];

    vec4 absoluteHeights = getAbsoluteHeight(flatIndex, layerStride, index.z);

    ivec2 off = offsets[int(position.x < 0.5f)][int(position.z < 0.5f)];
    vec3 newNormal = normal;
    const vec4 relativeHeight = getRelativeHeight(off, index, layerStride, absoluteHeights, newNormal);
   
    vertexToGeometry.position = vec3(gridScale, relativeHeight[WATER], gridScale) * (vec3(index.x, 0.0f, index.y) + (0.5f * position + 0.5f));
    vertexToGeometry.position.y += relativeHeight[FLOOR];
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    vertexToGeometry.normal = transpose(mat3(worldToLocal)) * newNormal;
    vertexToGeometry.v = relativeHeight[WATER] * (0.5f * position.y + 0.5f);

    // Adding small epsilon to avoid issues with 0 water
    vertexToGeometry.maxV[BEDROCK] = relativeHeight[BEDROCK];
    vertexToGeometry.maxV[SAND] = relativeHeight[SAND];
    vertexToGeometry.maxV[WATER] = relativeHeight[WATER];
    flatVertexToGeometry.valid = true;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
