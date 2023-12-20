#version 460
#extension GL_ARB_bindless_texture : require

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "../renderer/renderer.glsl"
#include "../mesh_renderer/mesh_renderer.glsl"
#include "../mesh_renderer/vertex_attributes.glsl"
#include "../math/constants.glsl"

out VertexToGeometry vertexToGeometry;
out flat FlatVertexToGeometry flatVertexToGeometry;

void main() 
{
    int index = gl_InstanceID;

    const int layer = index % gridSize.z;
    index /= gridSize.z;

    ivec3 cell;
    cell.x = index % gridSize.x;
    cell.y = index / gridSize.x;
    cell.z = 0;

    vec3 offset;
   
    if (layer == 0) 
    {
        offset.y = 0.0f;
    }
    else 
    {
        for (int i = 0; i < layer - 1; ++i) 
        {
            const int above = texelFetch(infoMap, cell, 0)[ABOVE];

            if (above <= cell.z) 
            {
                flatVertexToGeometry.isValid = false;
                return;
            }

            cell.z = above;
        }

        const int above = texelFetch(infoMap, cell, 0)[ABOVE];
        
        if (above <= cell.z) 
        {
            flatVertexToGeometry.isValid = false;
            return;
        }

        offset.y = texelFetch(heightMap, cell, 0)[MAX_HEIGHT];
        cell.z = above;
    }

    const vec3 height = texelFetch(heightMap, cell, 0).xyz;
    const float totalHeight = height[BEDROCK] + height[SAND] + height[WATER];

    if (totalHeight <= epsilon) 
    {
        flatVertexToGeometry.isValid = false;
        return;
    }

    const float rTotalHeight = 1.0f / totalHeight;

    flatVertexToGeometry.cell = cell;
    flatVertexToGeometry.maxV[BEDROCK] = height[BEDROCK];
    flatVertexToGeometry.maxV[SAND] = height[BEDROCK] + height[SAND];
    flatVertexToGeometry.maxV.xy *= rTotalHeight;
    flatVertexToGeometry.maxV[WATER] = 1.0f;

    flatVertexToGeometry.isValid = true;

    const vec3 scale = 0.5f * vec3(gridScale, totalHeight, gridScale);
    offset.x = gridScale * (cell.x + 0.5f - 0.5f * gridSize.x);
    offset.y += scale.y;
    offset.z = gridScale * (cell.y + 0.5f - 0.5f * gridSize.y);

    vertexToGeometry.v = 0.5f * position.y + 0.5f;
    vertexToGeometry.position = offset + scale * position;
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    
    const mat3 normalMatrix = transpose(mat3(worldToLocal));
    vertexToGeometry.normal = normalMatrix * normal;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
