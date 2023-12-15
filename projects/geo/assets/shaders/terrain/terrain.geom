#version 460
#extension GL_ARB_bindless_texture : require

#include "material.glsl"
#include "vertex_to_geometry.glsl"
#include "geometry_to_fragment.glsl"

layout (triangles) in;
in VertexToGeometry vertexToGeometry[];
in flat FlatVertexToGeometry flatVertexToGeometry[];

layout (triangle_strip, max_vertices = 3) out;
out GeometryToFragment geometryToFragment;
out flat FlatGeometryToFragment flatGeometryToFragment;

void main()
{
    if (flatVertexToGeometry[0].isValid) 
    {
        flatGeometryToFragment.cell = flatVertexToGeometry[0].cell;
        flatGeometryToFragment.maxV = flatVertexToGeometry[0].maxV;
        
        for (int i = 0; i < 3; ++i)
        {
            geometryToFragment.position = vertexToGeometry[i].position;
            geometryToFragment.normal = vertexToGeometry[i].normal;
            geometryToFragment.v = vertexToGeometry[i].v;

            gl_Position = gl_in[i].gl_Position;

            EmitVertex();
        }

        EndPrimitive();
    }
}
