#version 460
#extension GL_ARB_bindless_texture : require

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "../render_pipeline/render_pipeline.glsl"
#include "../mesh_render_pipeline/mesh_render_pipeline.glsl"
#include "../mesh_render_pipeline/vertex_attributes.glsl"
#include "../math/constants.glsl"

out VertexToGeometry vertexToGeometry;
out flat FlatVertexToGeometry flatVertexToGeometry;

void main() 
{
    int index = gl_InstanceID;

    ivec3 cell;
    cell.z = index % gridSize.z;

    index /= gridSize.z;

    cell.x = index % gridSize.x;
    cell.y = index / gridSize.x;

    const vec3 height = texelFetch(heightMap, cell, 0).xyz;
    const float totalHeight = height[BEDROCK] + height[SAND] + height[WATER];

    if (totalHeight <= epsilon) 
    {
        flatVertexToGeometry.isValid = false;
        return;
    }

    const int below = texelFetch(infoMap, cell, 0)[BELOW];
    const float minHeight = below != INVALID_INDEX ? texelFetch(heightMap, ivec3(cell.xy, below), 0)[MAX_HEIGHT] : 0.0f;
  
    flatVertexToGeometry.cell = cell;
    flatVertexToGeometry.maxV[BEDROCK] = height[BEDROCK] - minHeight;
    flatVertexToGeometry.maxV[SAND] = flatVertexToGeometry.maxV[BEDROCK] + height[SAND];
    flatVertexToGeometry.maxV.xy /= flatVertexToGeometry.maxV[SAND] + height[WATER];
    flatVertexToGeometry.maxV[WATER] = 1.0f;
    flatVertexToGeometry.isValid = true;

    const vec3 scale = 0.5f * vec3(gridScale, totalHeight - minHeight, gridScale);
    const vec3 offset = vec3(gridScale * (cell.x + 0.5f - 0.5f * gridSize.x),
                             minHeight + scale.y,
                             gridScale * (cell.y + 0.5f - 0.5f * gridSize.y));

    vertexToGeometry.v = 0.5f * position.y + 0.5f;
    vertexToGeometry.position = offset + scale * position;
    vertexToGeometry.position = (localToWorld * vec4(vertexToGeometry.position, 1.0f)).xyz;
    
    const mat3 normalMatrix = transpose(mat3(worldToLocal));
    vertexToGeometry.normal = normalMatrix * normal;
   
    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
