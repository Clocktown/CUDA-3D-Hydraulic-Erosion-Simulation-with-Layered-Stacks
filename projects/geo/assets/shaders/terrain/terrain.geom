#version 460

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "geometry_to_fragment.glsl"

layout (triangles) in;
in VertexToGeometry vertexToGeometry[];


layout (triangle_strip, max_vertices = 3) out;
out GeometryToFragment geometryToFragment;
flat out int cellType;

void main()
{
    if (vertexToGeometry[0].cellType != -1) 
    {
        for (int i = 0; i < 3; ++i)
        {
            geometryToFragment.position = vertexToGeometry[i].position;
            geometryToFragment.normal = vertexToGeometry[i].normal;
            geometryToFragment.uvw = vertexToGeometry[i].uvw;
            cellType = vertexToGeometry[i].cellType;

            gl_Position = gl_in[i].gl_Position;

            EmitVertex();
        }
    }

    EndPrimitive();
}
