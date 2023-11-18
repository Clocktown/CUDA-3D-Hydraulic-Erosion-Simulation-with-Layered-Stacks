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

    const int layer = index % maxLayerCount;
    index /= maxLayerCount;

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
            const int above = texelFetch(infoMap, cell, 0)[aboveIndex];

            if (above <= cell.z) 
            {
                flatVertexToGeometry.isValid = false;
                return;
            }

            cell.z = above;
        }

        const int above = texelFetch(infoMap, cell, 0)[aboveIndex];
        
        if (above <= cell.z) 
        {
            flatVertexToGeometry.isValid = false;
            return;
        }

        offset.y = texelFetch(heightMap, cell, 0)[maxHeightIndex];
        cell.z = above;
    }

    const vec3 height = texelFetch(heightMap, cell, 0).xyz;
    const float totalHeight = height[bedrockIndex] + height[sandIndex] + height[waterIndex];

    if (totalHeight <= epsilon) 
    {
        flatVertexToGeometry.isValid = false;
        return;
    }

    flatVertexToGeometry.cell = cell;
    flatVertexToGeometry.maxV[bedrockIndex] = offset.y + height[bedrockIndex];
    flatVertexToGeometry.maxV[sandIndex] = flatVertexToGeometry.maxV[bedrockIndex] + height[sandIndex];
    flatVertexToGeometry.maxV.xy /= flatVertexToGeometry.maxV[sandIndex] + height[waterIndex];
    flatVertexToGeometry.maxV[waterIndex] = 1.0f;
    
    flatVertexToGeometry.isValid = true;

    const vec3 scale = vec3(gridScale, 0.5f * totalHeight, gridScale);
    offset.x = gridScale * (cell.x + 0.5f);
    offset.y += scale.y;
    offset.z = gridScale * (cell.y + 0.5f);

    vertexToGeometry.position = offset + scale * position;
    vertexToGeometry.v = 0.5f * (position.y + 1.0f);
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    
    const mat3 normalMatrix = mat3(transpose(worldToLocal));
    vertexToGeometry.normal = normalMatrix * normal;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
