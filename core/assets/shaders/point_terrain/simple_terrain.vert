#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "vertex_to_geometry.glsl"
#include "grid.glsl"
#include "../render_pipeline/render_pipeline.glsl"
#include "../render_pipeline/maths.glsl"
#include "../point_render_pipeline/point_render_pipeline.glsl"

out VertexToGeometry vertexToGeometry;

const ivec2 offset = ivec2(1);

int getLayerCount(const int flatIndex) 
{
    const int packedIndex = flatIndex / 4;
    const int packedPosition = flatIndex - 4 * packedIndex; 
    const int packedValue = layerCounts[packedIndex];

    return (packedValue >> (8 * packedPosition)) & 0xff;
}

vec4 getAbsoluteHeight(const in int flatIndex, int layerStride, int z) {
    vec4 absoluteHeights = heights[flatIndex];
    absoluteHeights[FLOOR] = z > 0 ? heights[flatIndex - layerStride][CEILING] : min(absoluteHeights[BEDROCK] - 50.f, 0.0f);
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

bool doesOverlap(vec2 a, vec2 b) {
    return (min(a.y, b.y) - max(a.x, b.x)) > 0.f;
}

void fillVertexInfo(const in ivec3 index, const in int layerStride, const in vec4 absoluteHeight) {
    if(!useInterpolation) {
        for(int i = 0; i < 4; ++i){
            vertexToGeometry.normal[i] = vec3(0,1,0);
            vertexToGeometry.normal[i+4] = vec3(0,-1,0);
            vertexToGeometry.interpolated[i] = false;
            vertexToGeometry.interpolated[i+4] = false;
            vertexToGeometry.h[i] = absoluteHeight;
        }
    }
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

    bool hasWater = (absoluteHeight[WATER] - absoluteHeight[SAND]) > 0.f;

    for(int i = 0; i < 3; ++i) {
        ivec3 idx = index + offs[i];
        idx.z = 0;
        if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
            continue;
        }
        int flatIdx = idx.x + idx.y * gridSize.x;
        int layerCount = getLayerCount(flatIdx);

        bool foundFloor = false;
        vec4 neighborHeight = absoluteHeight;

        for(int j = 0; j < layerCount; ++j, flatIdx += layerStride, ++idx.z) {
            vec4 h = getAbsoluteHeight(flatIdx, layerStride, idx.z);
            bool nHasWater = (h[WATER] - h[SAND]) > 0.f;
            if((hasWater == nHasWater) && doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                isValid[i+1] = true;
                if(!foundFloor) {
                    neighborHeight[FLOOR] = h[FLOOR];
                }
                neighborHeight[SAND] = h[SAND];
                neighborHeight[BEDROCK] = h[BEDROCK];
                neighborHeight[WATER] = h[WATER];
            }
            if(doesOverlap(vec2(absoluteHeight[SAND], absoluteHeight[WATER]), vec2(h[SAND], h[WATER]))) {
                isValid[i+1] = true;
                neighborHeight[WATER] = h[WATER];
            }
        }
        if(isValid[i+1]) {
            top[i+1] = neighborHeight[WATER];
            bot[i+1] = neighborHeight[FLOOR];
            cumulativeHeight += neighborHeight;
            numCells++;
        }
    }

    /*if(position.y >= 0.5f && numCells > 1) {
       newNormal = calculateNormal(numCells, isValid, top, off);
    } else if(position.y <= -0.5f && numCells > 1) {
       newNormal = -calculateNormal(numCells, isValid, bot, off);
    }

    flatVertexToGeometry.interpolated[0] = isValid[1];
    flatVertexToGeometry.interpolated[1] = isValid[2];*/

    return makeRelative(cumulativeHeight / numCells);
}

void main() 
{
    const ivec3 index = unflattenIndex(gl_VertexID, ivec3(gridSize, maxLayerCount));
    int flatIndex = index.x + index.y * gridSize.x;
   
    if (index.z >= getLayerCount(flatIndex)) 
    {
        vertexToGeometry.valid = false;
        return;
    }

    vertexToGeometry.valid = true;

    flatIndex = gl_VertexID;
    const int layerStride = gridSize.x * gridSize.y;

    vec4 absoluteHeights = getAbsoluteHeight(flatIndex, layerStride, index.z);

    fillVertexInfo(index, layerStride, absoluteHeights);
   
    gl_Position = vec4(gridScale * index.x, 0, gridScale * index.y, 1.0f);
}
