#version 460
#extension GL_NV_shader_buffer_load : require

#include "simple_material.glsl"
#include "vertex_to_geometry.glsl"
#include "geometry_to_fragment.glsl"

layout (triangles) in;
in VertexToGeometry vertexToGeometry[];
in flat FlatVertexToGeometry flatVertexToGeometry[];
in flat float stabilityValues[];

layout (triangle_strip, max_vertices = 3) out;
out GeometryToFragment geometryToFragment;
out flat FlatGeometryToFragment flatGeometryToFragment;
out flat float stabilityValue;

void main()
{
    if (flatVertexToGeometry[0].valid) 
    {
        flatGeometryToFragment.maxV = flatVertexToGeometry[0].maxV;
        stabilityValue = stabilityValues[0];
        
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
