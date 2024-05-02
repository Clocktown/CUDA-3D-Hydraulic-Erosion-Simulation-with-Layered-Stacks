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

vec4 getHorizontalNeighborHeight(const in int n, const in ivec3 index, const in ivec2[8] offs, const in vec4 absoluteHeight, inout bool[8] valid_top, inout bool[8] valid_bot, inout float[9] top, inout float[9] bot, const in int layerStride, const in bool has_above, const in bool has_below, const in vec4 above, const in vec4 below, inout vec4[4] nBelowAbove) {
    ivec3 idx = index + ivec3(offs[n], 0);
    idx.z = 0;
    vec4 neighborHeight = absoluteHeight;
    valid_top[n] = false;
    valid_bot[n] = false;
    

    if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
        top[n+1] = absoluteHeight[WATER];
        bot[n+1] = absoluteHeight[FLOOR];
    } else {
        int flatIdx = idx.x + idx.y * gridSize.x;
        const int layerCount = getLayerCount(flatIdx);

        vec4 nBelow = getAbsoluteHeight(flatIdx, layerStride, idx.z);
        vec4 nAbove = nBelow;
        bool nHasBelow = false;
        bool nHasAbove = layerCount > 1;

        int jBot = 0;
        int jTop = 0;

        for(int j = 0; j < layerCount; ++j, flatIdx += layerStride, ++idx.z) {
            if(j == layerCount - 1) nHasAbove = false;
            const vec4 h = nAbove;
            if(nHasAbove) {
                nAbove = getAbsoluteHeight(flatIdx + layerStride, layerStride, idx.z + 1);
            }
            if(doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                if(!valid_bot[n]) {
                    const bool nonBelowOverlapsNeighbor = !(has_below && doesOverlap(vec2(below[FLOOR], below[BEDROCK]), vec2(h[FLOOR], h[BEDROCK])));
                    if(nonBelowOverlapsNeighbor) {
                        neighborHeight[FLOOR] = h[FLOOR];
                        valid_bot[n] = true;
                        jBot = j;
                        nBelowAbove[n/2].xy = nHasBelow ? vec2(nBelow[FLOOR], nBelow[BEDROCK]) : vec2(0.f);
                    }
                }
                const bool nonAboveOverlapsNeighbor = !(has_above && doesOverlap(vec2(h[FLOOR], h[BEDROCK]), vec2(above[FLOOR], above[BEDROCK])));
                if(nonAboveOverlapsNeighbor) {
                    neighborHeight[SAND] = h[SAND];
                    neighborHeight[BEDROCK] = h[BEDROCK];
                    neighborHeight[WATER] = h[WATER];
                    valid_top[n] = true;
                    jTop = j;
                    nBelowAbove[n/2].zw = nHasAbove ? vec2(nAbove[FLOOR], nAbove[BEDROCK]) : vec2(0.f);
                }
            }

            nBelow = h;
            nHasBelow = true;
        }
        if(!bool(n % 2) && valid_top[n] && valid_bot[n] && (jTop == jBot)) {
             vertexToGeometry.valid_face[n / 2] = false;
        }
    }
    return neighborHeight;
}

vec4 getDiagonalNeighborHeight(const in int n, const in ivec3 index, const in ivec2[8] offs, const in vec4 absoluteHeight, inout bool[8] valid_top, inout bool[8] valid_bot, inout float[9] top, inout float[9] bot, const in int layerStride, const in bool has_above, const in bool has_below, const in vec4 above, const in vec4 below, const in vec4[4] nBelowAbove, const in vec4[4] horizontalNeighbors) {
    ivec3 idx = index + ivec3(offs[n], 0);
    idx.z = 0;
    vec4 neighborHeight = absoluteHeight;
    valid_top[n] = false;
    valid_bot[n] = false;
    const int leftIdx = n / 2;
    const int rightIdx = ((n / 2) + 1) % 4;
    const vec2 nLeft = vec2(horizontalNeighbors[leftIdx][FLOOR], horizontalNeighbors[leftIdx][BEDROCK]);
    const vec2 nRight = vec2(horizontalNeighbors[rightIdx][FLOOR], horizontalNeighbors[rightIdx][BEDROCK]);
    const vec4 nLeftBelowAbove = nBelowAbove[leftIdx];
    const vec4 nRightBelowAbove = nBelowAbove[rightIdx];
    

    if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
        top[n+1] = absoluteHeight[WATER];
        bot[n+1] = absoluteHeight[FLOOR];
    } else {
        int flatIdx = idx.x + idx.y * gridSize.x;
        const int layerCount = getLayerCount(flatIdx);

        for(int j = 0; j < layerCount; ++j, flatIdx += layerStride, ++idx.z) {
            const vec4 h = getAbsoluteHeight(flatIdx, layerStride, idx.z);
            const vec2 hI = vec2(h[FLOOR], h[BEDROCK]);

            if( doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), hI) &&
                doesOverlap(nLeft, hI) && doesOverlap(nRight, hI)
            ) {
                if(!valid_bot[n]) {
                    const bool nonBelowOverlapsNeighbor = !(has_below && doesOverlap(vec2(below[FLOOR], below[BEDROCK]), hI));
                    const bool nonBelowHorizontalNeighborsOverlapsNeighbor = !(doesOverlap(nLeftBelowAbove.xy, hI) || doesOverlap(nRightBelowAbove.xy, hI));
                    if(nonBelowOverlapsNeighbor && nonBelowHorizontalNeighborsOverlapsNeighbor) {
                        neighborHeight[FLOOR] = h[FLOOR];
                        valid_bot[n] = true;
                    } else /*if(nonBelowHorizontalNeighborsOverlapsNeighbor)*/ {
                        vertexToGeometry.valid_face[leftIdx] = true;
                        vertexToGeometry.valid_face[rightIdx] = true;
                    }
                }
                const bool nonAboveOverlapsNeighbor = !(has_above && doesOverlap(hI, vec2(above[FLOOR], above[BEDROCK])));
                const bool nonAboveHorizontalNeighborsOverlapsNeighbor = !(doesOverlap(nLeftBelowAbove.zw, hI) || doesOverlap(nRightBelowAbove.zw, hI));
                if(nonAboveOverlapsNeighbor && nonAboveHorizontalNeighborsOverlapsNeighbor) {
                    neighborHeight[SAND] = h[SAND];
                    neighborHeight[BEDROCK] = h[BEDROCK];
                    neighborHeight[WATER] = h[WATER];
                    valid_top[n] = true;
                } else /*if(nonAboveHorizontalNeighborsOverlapsNeighbor)*/ {
                    vertexToGeometry.valid_face[leftIdx] = true;
                    vertexToGeometry.valid_face[rightIdx] = true;
                }
            }
        }
        if(!valid_top[n] || !valid_bot[n]) {
            vertexToGeometry.valid_face[leftIdx] = true;
            vertexToGeometry.valid_face[rightIdx] = true;
        }
    }

    return neighborHeight;
}

void checkHorizontalPlausibility(const in int n, const in ivec3 index, const in ivec2[8] offs, const in vec4 absoluteHeight, inout bool[8] valid_top, inout bool[8] valid_bot, inout float[9] top, inout float[9] bot, const in int layerStride, const in bool has_above, const in bool has_below, const in vec4 above, const in vec4 below, const in vec4[4] nBelowAbove, const in vec4[4] horizontalNeighbors) {
    ivec3 idx = index + ivec3(offs[n], 0);
    idx.z = 0;

    const int leftIdx = (4 + (n / 2) - 1) % 4;
    const int rightIdx = ((n / 2) + 1) % 4;
    const vec2 nLeft = vec2(horizontalNeighbors[leftIdx][FLOOR], horizontalNeighbors[leftIdx][BEDROCK]);
    const vec2 nRight = vec2(horizontalNeighbors[rightIdx][FLOOR], horizontalNeighbors[rightIdx][BEDROCK]);
    const vec4 nLeftBelowAbove = nBelowAbove[leftIdx];
    const vec4 nRightBelowAbove = nBelowAbove[rightIdx];
    const vec4 belowAbove = nBelowAbove[n/2];
    const vec4 h = horizontalNeighbors[n/2];
    const vec2 hI = vec2(h[FLOOR], h[BEDROCK]);

    if(!doesOverlap(nLeft, hI) || doesOverlap(nLeftBelowAbove.xy, hI) || doesOverlap(nLeftBelowAbove.zw, hI) || doesOverlap(nLeft, belowAbove.xy) || doesOverlap(nLeft, belowAbove.zw)) {
        vertexToGeometry.valid_face[leftIdx] = true;
        vertexToGeometry.valid_face[n/2] = true;
    }
    if(!doesOverlap(nRight, hI) || doesOverlap(nRightBelowAbove.xy, hI) || doesOverlap(nRightBelowAbove.zw, hI) || doesOverlap(nRight, belowAbove.xy) || doesOverlap(nRight, belowAbove.zw)) {
        vertexToGeometry.valid_face[rightIdx] = true;
        vertexToGeometry.valid_face[n/2] = true;
    }
}

void incrementVertexTopValues(const in int n, const in vec4 neighborHeight, inout vec4 numNeigh) {
    if(n <= 2) {
        numNeigh[0]++;
        vertexToGeometry.h[0].xyz += neighborHeight.xyz;
    }
    if(n >= 2 && n <= 4) {
        numNeigh[2]++;
        vertexToGeometry.h[2].xyz += neighborHeight.xyz;
    }
    if(n >= 4 && n <= 6) {
        numNeigh[3]++;
        vertexToGeometry.h[3].xyz += neighborHeight.xyz;
    }
    if((n >= 6 && n <= 7) || n == 0) {
        numNeigh[1]++;
        vertexToGeometry.h[1].xyz += neighborHeight.xyz;
    }
}

void incrementVertexBotValues(const in int n, const in vec4 neighborHeight, inout vec4 numNeigh) {
    if(n <= 2) {
        numNeigh[0]++;
        vertexToGeometry.h[0].w += neighborHeight.w;
    }
    if(n >= 2 && n <= 4) {
        numNeigh[2]++;
        vertexToGeometry.h[2].w += neighborHeight.w;
    }
    if(n >= 4 && n <= 6) {
        numNeigh[3]++;
        vertexToGeometry.h[3].w += neighborHeight.w;
    }
    if((n >= 6 && n <= 7) || n == 0) {
        numNeigh[1]++;
        vertexToGeometry.h[1].w += neighborHeight.w;
    }
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
    bool valid_top[8];
    bool valid_bot[8];
    vec4 numNeighTop = vec4(1);
    vec4 numNeighBot = vec4(1);
    top[0] = absoluteHeight[WATER];
    bot[0] = absoluteHeight[FLOOR];

    bool has_above = index.z < (layerCount - 1);
    vec4 above = has_above ? getAbsoluteHeight(flatIndex + layerStride, layerStride, index.z + 1) : vec4(0);
    bool has_below = index.z > 0;
    vec4 below = has_below ? getAbsoluteHeight(flatIndex - layerStride, layerStride, index.z - 1) : vec4(0);

    vec4 horizontalNeighborHeights[4];
    vec4 horizontalNeighborBelowAbove[4];

    for(int n = 0; n < 8; n += 2) {
        const int hn = n / 2;
        horizontalNeighborHeights[hn] = getHorizontalNeighborHeight(n, index, offs, absoluteHeight, valid_top, valid_bot, top, bot, layerStride, has_above, has_below, above, below, horizontalNeighborBelowAbove);
        top[n + 1] = horizontalNeighborHeights[hn][WATER];
        bot[n + 1] = horizontalNeighborHeights[hn][FLOOR];
        if(valid_top[n]) incrementVertexTopValues(n, horizontalNeighborHeights[hn], numNeighTop);
        if(valid_bot[n]) incrementVertexBotValues(n, horizontalNeighborHeights[hn], numNeighBot);
    }

    for(int n = 1; n < 8; n += 2) {
        vec4 neighborHeight = getDiagonalNeighborHeight(n, index, offs, absoluteHeight, valid_top, valid_bot, top, bot, layerStride, has_above, has_below, above, below, horizontalNeighborBelowAbove, horizontalNeighborHeights);
        //vec4 neighborHeight = getHorizontalNeighborHeight(n, index, offs, absoluteHeight, valid_top, valid_bot, top, bot, layerStride, has_above, has_below, above, below);
        top[n + 1] = neighborHeight[WATER];
        bot[n + 1] = neighborHeight[FLOOR];
        if(valid_top[n]) incrementVertexTopValues(n, neighborHeight, numNeighTop);
        if(valid_bot[n]) incrementVertexBotValues(n, neighborHeight, numNeighBot);
    }

    for(int n = 0; n < 8; n += 2) {
        checkHorizontalPlausibility(n, index, offs, absoluteHeight, valid_top, valid_bot, top, bot, layerStride, has_above, has_below, above, below, horizontalNeighborBelowAbove, horizontalNeighborHeights);
    }
    const ivec4 indexMap = ivec4(0, 3, 1, 2);
    for(int i = 0; i < 4; ++i) {
        vertexToGeometry.h[i].xyz /= numNeighTop[i];
        vertexToGeometry.h[i].w /= numNeighBot[i];
        const int n = indexMap[i];
        ivec3 idc = ivec3(2 * n, (2 * n + 2) % 8, 2 * n + 1);
        if(bool(n%2)) {
            idc = idc.yxz;
        }

        vertexToGeometry.normal[i] = calculateNormal(int(numNeighTop[i]), bvec4(true, valid_top[idc[0]], valid_top[idc[1]], valid_top[idc[2]]), vec4(top[0], top[idc[0] + 1], top[idc[1] + 1], top[idc[2] + 1]), offs[2 * n + 1]);
        vertexToGeometry.normal[i + 4] = -calculateNormal(int(numNeighBot[i]), bvec4(true, valid_bot[idc[0]], valid_bot[idc[1]], valid_bot[idc[2]]), vec4(bot[0], bot[idc[0] + 1], bot[idc[1] + 1], bot[idc[2] + 1]), offs[2 * n + 1]);
    }

    const bool allValid = all(bvec4(valid_top[0], valid_top[1], valid_top[2], valid_top[3])) && all(bvec4(valid_top[4], valid_top[5], valid_top[6], valid_top[7])) &&
                    all(bvec4(valid_bot[0], valid_bot[1], valid_bot[2], valid_bot[3])) && all(bvec4(valid_bot[4], valid_bot[5], valid_bot[6], valid_bot[7]));
    for(int i = 0; i < 4; ++i) {
        //vertexToGeometry.valid_face[i] = vertexToGeometry.valid_face[i] || !allValid;
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
