#version 460

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "geometry_to_fragment.glsl"

layout (triangles) in;
in VertexToGeometry vertexToGeometry[];
in FlatVertexToGeometry flatVertexToGeometry[];

layout (triangle_strip, max_vertices = 3) out;
out GeometryToFragment geometryToFragment;
out FlatGeometryToFragment flatGeometryToFragment;

void main()
{
    if (flatVertexToGeometry[0].cellType != -1) 
    {
        flatGeometryToFragment.cell = flatVertexToGeometry[0].cell;
        flatGeometryToFragment.cellType = flatVertexToGeometry[0].cellType;
        
        for (int i = 0; i < 3; ++i)
        {
            geometryToFragment.position = vertexToGeometry[i].position;
            geometryToFragment.normal = vertexToGeometry[i].normal;
           
            gl_Position = gl_in[i].gl_Position;

            EmitVertex();
        }
    }

    EndPrimitive();
}
