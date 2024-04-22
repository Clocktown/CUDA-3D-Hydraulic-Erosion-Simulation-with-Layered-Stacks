#ifndef GEO_VERTEX_TO_GEOMETRY_GLSL
#define GEO_VERTEX_TO_GEOMETRY_GLSL

struct VertexToGeometry
{
	vec3 normal[8];
	vec4 h[4];
	bool valid_face[4];
};

#endif
