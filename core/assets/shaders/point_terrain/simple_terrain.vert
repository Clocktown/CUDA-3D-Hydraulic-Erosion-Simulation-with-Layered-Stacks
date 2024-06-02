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
    absoluteHeights[SAND] = renderSand ? absoluteHeights[SAND] : 0.f;
    absoluteHeights[WATER] = renderWater ? absoluteHeights[WATER] : 0.f;
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

vec4 getNeighborHeight(const in int n, const in ivec3 index, const in ivec2[8] offs, const in vec4 absoluteHeight, inout bool[8] valid_top, inout bool[8] valid_bot, inout bool[8] top_equals_bot, inout float[9] top, inout float[9] bot, const in int layerStride, const in bool has_above, const in bool has_below, const in vec4 above, const in vec4 below, inout vec4[8] nBelowAbove) {
    ivec3 idx = index + ivec3(offs[n], 0);
    idx.z = 0;
    vec4 neighborHeight = absoluteHeight;
    nBelowAbove[n] = vec4(0);
    valid_top[n] = false;
    valid_bot[n] = false;
    top_equals_bot[n] = false;
    

    if(idx.x >= gridSize.x || idx.x < 0 || idx.y < 0 || idx.y >= gridSize.y) {
        //top[n+1] = absoluteHeight[WATER];
        //bot[n+1] = absoluteHeight[FLOOR];
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
            if(h[FLOOR] >= absoluteHeight[BEDROCK]) {
                if(!valid_top[n]) {
                    // We have no neighbor here to interpolate. Remember the Column just Above and just Below our center column in this neighbor cell for later sanity check
                    nBelowAbove[n].zw = vec2(h[FLOOR], h[BEDROCK]);
                }
                break;
            }
            if(h[BEDROCK] <= absoluteHeight[FLOOR]) {
                if(!valid_bot[n]) {
                    // We have no neighbor here to interpolate. Remember the Column just Below our center column in this neighbor cell for later sanity check
                    nBelowAbove[n].xy = vec2(h[FLOOR], h[BEDROCK]);
                }
            }
            if(doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]))) {
                if(!valid_top[n] && !valid_bot[n]) {
                    // Interpolate Floor with lowest
                    bool lowerColumnOverlapsNeighbor = has_below && doesOverlap(vec2(below[FLOOR], below[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]));
                    //bool lowerNeighborOverlapsMe = nHasBelow && doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(nBelow[FLOOR], nBelow[BEDROCK]));
                    if(!lowerColumnOverlapsNeighbor) {
                        valid_bot[n] = true;
                        neighborHeight[FLOOR] = h[FLOOR];
                        jBot = j;
                        // Remember what we interpolated with
                        nBelowAbove[n].xy = vec2(h[FLOOR], h[BEDROCK]);
                    }
                }
                // Interpolate Top with highest
                bool higherColumnOverlapsNeighbor = has_above && doesOverlap(vec2(above[FLOOR], above[BEDROCK]), vec2(h[FLOOR], h[BEDROCK]));
                //bool higherNeighborOverlapsMe = nHasAbove && doesOverlap(vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]), vec2(nAbove[FLOOR], nAbove[BEDROCK]));
                if(!higherColumnOverlapsNeighbor) {
                    neighborHeight[SAND] = h[SAND];
                    neighborHeight[BEDROCK] = h[BEDROCK];
                    neighborHeight[WATER] = h[WATER];
                    valid_top[n] = true;
                    jTop = j;
                    // Remember what we interpolated with
                    if(jTop == jBot) {
                        if(nHasAbove) {
                            nBelowAbove[n].zw = vec2(nAbove[FLOOR], nAbove[BEDROCK]);
                        } else if(nHasBelow) {
                            nBelowAbove[n].zw = vec2(h[FLOOR], h[BEDROCK]);
                            nBelowAbove[n].xy = vec2(nBelow[FLOOR], nBelow[BEDROCK]);
                        } else {
                            nBelowAbove[n].zw = vec2(h[FLOOR], h[BEDROCK]);
                        }
                    } else {
                        nBelowAbove[n].zw = vec2(h[FLOOR], h[BEDROCK]);
                   }
                }
            }

            nBelow = h;
            nHasBelow = true;
        }
        
        top_equals_bot[n] = (jTop == jBot);
        if(!bool(n % 2) && valid_top[n] && valid_bot[n] && (jTop == jBot)) {
             // jTop == jBot Condition may be able to be removed depending on how we interpolate
             vertexToGeometry.valid_face[n / 2] = false; // Face can potentially be discarded, needs to be checked later
        }
    }
    return neighborHeight;
}

void incrementVertexValues(const in int n, const in vec4 neighborHeight, inout vec4 numNeigh) {
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

void incrementVertexValuesTop(const in int n, in vec4 neighborHeight, inout vec4 numNeigh) {
    neighborHeight[FLOOR] = 0.f;
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

void incrementVertexValuesBot(const in int n, const in vec4 neighborHeight, inout vec4 numNeigh) {
    if(n <= 2) {
        numNeigh[0]++;
        vertexToGeometry.h[0][FLOOR] += neighborHeight[FLOOR];
    }
    if(n >= 2 && n <= 4) {
        numNeigh[2]++;
        vertexToGeometry.h[2][FLOOR] += neighborHeight[FLOOR];
    }
    if(n >= 4 && n <= 6) {
        numNeigh[3]++;
        vertexToGeometry.h[3][FLOOR] += neighborHeight[FLOOR];
    }
    if((n >= 6 && n <= 7) || n == 0) {
        numNeigh[1]++;
        vertexToGeometry.h[1][FLOOR] += neighborHeight[FLOOR];
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
    bool top_equals_bot[8];
    vec4 numNeighTop = vec4(1);
    vec4 numNeighBot = vec4(1);

    top[0] = absoluteHeight[WATER];
    bot[0] = absoluteHeight[FLOOR];

    // Following 4 variables are currently unused - maybe remove later but could be used for "nicer" interpolation
    bool has_above = index.z < (layerCount - 1);
    vec4 above = has_above ? getAbsoluteHeight(flatIndex + layerStride, layerStride, index.z + 1) : vec4(0);
    bool has_below = index.z > 0;
    vec4 below = has_below ? getAbsoluteHeight(flatIndex - layerStride, layerStride, index.z - 1) : vec4(0);

    vec4 nBelowAbove[8]; // Stores [FLOOR, BEDROCK] Intervalls for neighbor columns that are interpolated with (bottom/top face respectively). If there is no interpolation, stores interval of the column just below and above the center column

    for(int n = 0; n < 8; ++n) {
        vec4 neighborHeight = getNeighborHeight(n, index, offs, absoluteHeight, valid_top, valid_bot, top_equals_bot, top, bot, layerStride, has_above, has_below, above, below, nBelowAbove);
        top[n + 1] = neighborHeight[WATER];
        bot[n + 1] = neighborHeight[FLOOR];
        if(valid_top[n]) incrementVertexValuesTop(n, neighborHeight, numNeighTop);
        if(valid_bot[n]) incrementVertexValuesBot(n, neighborHeight, numNeighBot);
    }

    for(int i = 0; i < 4; ++i) {
        if(!vertexToGeometry.valid_face[i]) {
            // nBelow = nAbove for this neighbor
            int n = 2 * i; // Translate to correct index in 8-Neighborhood
            for(int j = n - 2; j <= n + 2; ++j) { // Iterate relevant neighborhood and check if interpolation rules are kept
                int nj = (8 + j) % 8; // Fix index
                vec4 belowAbove = nBelowAbove[n];
                if(!valid_top[n] && valid_bot[n]) {
                    belowAbove.zw = belowAbove.xy;
                } else if(valid_top[n] && !valid_bot[n]) {
                    belowAbove.xy = belowAbove.zw;
                } else if(valid_top[n] && valid_bot[n] && top_equals_bot[n]) {
                    if(doesOverlap(belowAbove.xy, vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]))) {
                        belowAbove.zw = belowAbove.xy;
                    } else if(doesOverlap(belowAbove.zw, vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]))) {
                        belowAbove.xy = belowAbove.zw;
                    }
                }

                vec4 belowAbove2 = nBelowAbove[nj];
                if(!valid_top[nj] && valid_bot[nj]) {
                    belowAbove2.zw = belowAbove2.xy;
                } else if(valid_top[nj] && !valid_bot[nj]) {
                    belowAbove2.xy = belowAbove2.zw;
                } else if(valid_top[nj] && valid_bot[nj] && top_equals_bot[nj]) {
                    if(doesOverlap(belowAbove2.xy, vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]))) {
                        belowAbove2.zw = belowAbove2.xy;
                    } else if(doesOverlap(belowAbove2.zw, vec2(absoluteHeight[FLOOR], absoluteHeight[BEDROCK]))) {
                        belowAbove2.xy = belowAbove2.zw;
                    }
                }

                bool o1 = doesOverlap(belowAbove.xy, nBelowAbove[nj].xy);
                bool o2 = doesOverlap(belowAbove.zw, nBelowAbove[nj].zw);

                bool o3 = doesOverlap(belowAbove.xy, belowAbove2.xy);
                bool o4 = doesOverlap(belowAbove.zw, belowAbove2.zw);

                if(valid_top[nj] && valid_bot[nj]) {
                    // Center Column did interpolate with 1 or two columns in this cell, so both have to overlap this neighbor and be the same cell
                    if(!(o3 && o4) || !top_equals_bot[nj] || !top_equals_bot[n] || (o1 && o2 && (nBelowAbove[n].xy != nBelowAbove[n].zw))) {
                        vertexToGeometry.valid_face[i] = true;
                        break;
                    }
                } else if(!valid_top[nj] && !valid_bot[nj]) {
                    // Center Column did not interpolate with a column in this cell. We either overlap with nothing as well, or overlap with a higher AND lower column with respect to the center column
                    if((o1 && !o2) || (!o1 && o2)) {
                        vertexToGeometry.valid_face[i] = true;
                        break;
                    }
                } else {
                    vertexToGeometry.valid_face[i] = true;
                    break;
                }
            }
        }
    }

    const ivec4 indexMap = ivec4(0, 3, 1, 2);
    for(int i = 0; i < 4; ++i) {
        vertexToGeometry.h[i] /= vec4(numNeighTop[i], numNeighTop[i], numNeighTop[i], numNeighBot[i]);
        const int n = indexMap[i];
        ivec3 idc = ivec3(2 * n, (2 * n + 2) % 8, 2 * n + 1);
        if(bool(n%2)) {
            idc = idc.yxz;
        }

        vertexToGeometry.normal[i] = calculateNormal(int(numNeighTop[i]), bvec4(true, valid_top[idc[0]], valid_top[idc[1]], valid_top[idc[2]]), vec4(top[0], top[idc[0] + 1], top[idc[1] + 1], top[idc[2] + 1]), offs[2 * n + 1]);
        vertexToGeometry.normal[i + 4] = -calculateNormal(int(numNeighBot[i]), bvec4(true, valid_bot[idc[0]], valid_bot[idc[1]], valid_bot[idc[2]]), vec4(bot[0], bot[idc[0] + 1], bot[idc[1] + 1], bot[idc[2] + 1]), offs[2 * n + 1]);
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
