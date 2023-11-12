#version 460

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "../renderer/renderer.glsl"
#include "../mesh_renderer/mesh_renderer.glsl"
#include "../mesh_renderer/vertex_attributes.glsl"
#include "../math/constants.glsl"

out VertexToGeometry vertexToGeometry;
out FlatVertexToGeometry flatVertexToGeometry;

void main() 
{
    int index = gl_InstanceID;

    const int cellType = index % 3;
    index /= 3;

    const int layer = index % maxLayerCount;
    index /= maxLayerCount;

    ivec3 cell;
    cell.x = index % gridSize.x;
    cell.y = index / gridSize.x;
    cell.z = 0;

    float totalHeight;
   
    if (layer == 0) 
    {
        totalHeight = 0.0f;
    }
    else 
    {
        for (int i = 0; i < layer - 1; ++i) 
        {
            const int above = texelFetch(infoMap, cell, 0)[aboveIndex];

            if (above <= cell.z) 
            {
                flatVertexToGeometry.cellType = -1;
                return;
            }

            cell.z = above;
        }

        const int above = texelFetch(infoMap, cell, 0)[aboveIndex];
        
        if (above <= cell.z) 
        {
            flatVertexToGeometry.cellType = -1;
            return;
        }

        totalHeight = texelFetch(heightMap, cell, 0)[maxHeightIndex];
        cell.z = above;
    }

    const vec3 scale = vec3(gridScale, 0.5f * texelFetch(heightMap, cell, 0)[cellType], gridScale);

    if (scale.y <= epsilon) 
    {
        flatVertexToGeometry.cellType = -1;
        return;
    }

    const vec3 height = texelFetch(heightMap, cell, 0).xyz;

    for (int i = 0; i < cellType; ++i) 
    {
        totalHeight += height[i];
    }

    totalHeight += scale.y;

    vertexToGeometry.position = scale * position + vec3(0.0f, totalHeight, 0.0f);
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;

    const mat3 normalMatrix = mat3(transpose(worldToLocal));
    vertexToGeometry.normal = normalMatrix * normal;
    
    flatVertexToGeometry.cell = cell;
    flatVertexToGeometry.cellType = cellType;

    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
