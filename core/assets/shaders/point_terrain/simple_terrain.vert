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

void fillVertexInfo(const in ivec3 index, const in int layerStride, const in vec4 absoluteHeight, const in int flatIndex, const in int layerCount) {
    const ivec2 offs[8] = {
                            ivec2(-1,0),
                            ivec2(-1,-1),
                            ivec2(0,-1),
                            ivec2(1,-1),
                            ivec2(1,0),
                            ivec2(1,1),
                            ivec2(0,1),
                            ivec2(-1,1)
                        };
    float top[9];
    float bot[9];
    bool valid[8];
    bool valid_face[8];
    vec4 numNeigh = vec4(1);
    top[0] = absoluteHeight[WATER];
    bot[0] = absoluteHeight[FLOOR];

    bool has_above = index.z < (layerCount - 1);
    vec4 above = has_above ? getAbsoluteHeight(flatIndex + layerStride, layerStride, index.z + 1) : vec4(0);
    bool has_below = index.z > 0;
    vec4 below = has_below ? getAbsoluteHeight(flatIndex - layerStride, layerStride, index.z - 1) : vec4(0);

    for(int n = 0; n < 8; ++n) {
        ivec3 idx = index + ivec3(offs[n], 0);
        idx.z = 0;
        vec4 neighborHeight = absoluteHeight;
        valid[n] = false;

        if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
            top[n+1] = absoluteHeight[WATER];
            bot[n+1] = absoluteHeight[FLOOR];
        } else {
            int flatIdx = idx.x + idx.y * gridSize.x;
            const int layerCount = getLayerCount(flatIdx);

            bool foundFloor = false;

            for(int j = 0; j < layerCount; ++j, flatIdx += layerStride, ++idx.z) {
                const vec4 h = getAbsoluteHeight(flatIdx, layerStride, idx.z);
                if(doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                    if(!foundFloor) {
                        neighborHeight[FLOOR] = h[FLOOR];
                        /*if(has_below && doesOverlap(vec2(below[FLOOR], below[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                            neighborHeight[FLOOR] = below[WATER];
                        }*/
                        foundFloor = true;
                    }
                    neighborHeight[SAND] = h[SAND];
                    neighborHeight[BEDROCK] = h[BEDROCK];
                    neighborHeight[WATER] = h[WATER];
                    if(!bool(n % 2)) {
                        vertexToGeometry.valid_face[n / 2] = false;
                    }
                    valid[n] = true;

                    /*if(has_above && doesOverlap(vec2(above[FLOOR], above[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                        neighborHeight[SAND] = above[FLOOR];
                        neighborHeight[BEDROCK] = above[FLOOR];
                        neighborHeight[WATER] = above[FLOOR];
                    }*/
                }
                /*if(doesOverlap(vec2(absoluteHeight[SAND], absoluteHeight[WATER]), vec2(h[SAND], h[WATER]))) {
                    neighborHeight[WATER] = h[WATER];
                    valid[n] = true;
                }
                if(doesOverlap(vec2(absoluteHeight[BEDROCK], absoluteHeight[SAND]), vec2(h[BEDROCK], h[SAND]))) {
                    neighborHeight[SAND] = h[SAND];
                    valid[n] = true;
                }*/
            }
        }
        top[n + 1] = neighborHeight[WATER];
        bot[n + 1] = neighborHeight[FLOOR];
        if(!valid[n]) continue;
        if(n <= 2) {
            numNeigh[0]++;
            vertexToGeometry.h[0] += neighborHeight;
        }
        if(n >= 2 && n <= 4) {
            numNeigh[2]++;
            vertexToGeometry.h[2] += neighborHeight;
        }
        if(n >= 4 && n <= 6) {
            numNeigh[3]++;
            vertexToGeometry.h[3] += neighborHeight;
        }
        if((n >= 6 && n <= 7) || n == 0) {
            numNeigh[1]++;
            vertexToGeometry.h[1] += neighborHeight;
        }
    }
    const ivec4 indexMap = ivec4(0, 3, 1, 2);
    for(int i = 0; i < 4; ++i) {
        vertexToGeometry.h[i] /= numNeigh[i];
        const int n = indexMap[i];
        ivec3 idc = ivec3(2 * n, (2 * n + 2) % 8, 2 * n + 1);
        if(bool(n%2)) {
            idc = idc.yxz;
        }

        vertexToGeometry.normal[i] = calculateNormal(int(numNeigh[i]), bvec4(true, valid[idc[0]], valid[idc[1]], valid[idc[2]]), vec4(top[0], top[idc[0] + 1], top[idc[1] + 1], top[idc[2] + 1]), offs[2 * n + 1]);
        vertexToGeometry.normal[i + 4] = -calculateNormal(int(numNeigh[i]), bvec4(true, valid[idc[0]], valid[idc[1]], valid[idc[2]]), vec4(bot[0], bot[idc[0] + 1], bot[idc[1] + 1], bot[idc[2] + 1]), offs[2 * n + 1]);
    }
}

void main() 
{
    const ivec3 index = unflattenIndex(indices[gl_VertexID], ivec3(gridSize, maxLayerCount));
    const int flatIndex0 = index.x + index.y * gridSize.x;
    const int layerCount = getLayerCount(flatIndex0);

    const int flatIndex = indices[gl_VertexID];
    const int layerStride = gridSize.x * gridSize.y;

    vec4 absoluteHeights = getAbsoluteHeight(flatIndex, layerStride, index.z);

    for(int i = 0; i < 4; ++i){
        vertexToGeometry.normal[i] = vec3(0,1,0);
        vertexToGeometry.normal[i+4] = vec3(0,-1,0);
        vertexToGeometry.valid_face[i] = true;
        vertexToGeometry.h[i] = absoluteHeights;
    }

    if(useInterpolation) {
        fillVertexInfo(index, layerStride, absoluteHeights, flatIndex, layerCount);
    }
   
    gl_Position = vec4(gridScale * index.x, 0, gridScale * index.y, 1.0f);
}
