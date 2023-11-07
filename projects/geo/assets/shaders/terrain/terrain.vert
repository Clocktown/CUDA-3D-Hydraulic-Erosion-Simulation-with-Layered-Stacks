#version 460

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "../renderer/renderer.glsl"
#include "../mesh_renderer/mesh_renderer.glsl"
#include "../mesh_renderer/vertex_attributes.glsl"
#include "../math/constants.glsl"

out VertexToGeometry vertexToGeometry;

void main() 
{
    const int horizontalCellCount = gridSize.x * gridSize.y;
    vertexToGeometry.cellType = gl_InstanceID % (horizontalCellCount * maxLayerCount);

    int index = gl_InstanceID / 3;
    ivec3 cell;

    cell.z = index % horizontalCellCount;
    index -= cell.z * horizontalCellCount;
    cell.x = index % gridSize.x;
    cell.y = index / gridSize.x;

    vertexToGeometry.position += vec3(cell) + 0.5f;
    vertexToGeometry.uvw = vertexToGeometry.position / vec3(gridSize.x, gridSize.y, maxLayerCount);

    const float height = texture(heightMap, vertexToGeometry.uvw)[vertexToGeometry.cellType];

    if (height <= epsilon) 
    {
        vertexToGeometry.cellType = -1;
        return;
    }

    // ToDo: Correct position
    vertexToGeometry.position = (localToWorld * vec4(position, 1.0f)).xyz;

    const mat3 normalMatrix = mat3(localToWorld);
    vertexToGeometry.normal = normalize(normalMatrix * normal); 

    gl_Position = worldToClip * vec4(vertexToGeometry.position, 1.0f);
}
