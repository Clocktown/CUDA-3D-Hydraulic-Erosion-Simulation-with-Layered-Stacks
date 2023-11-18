#ifndef GEO_VERTEX_TO_GEOMETRY_GLSL
#define GEO_VERTEX_TO_GEOMETRY_GLSL

struct VertexToGeometry
{
	vec3 position;
	vec3 normal;
	float v;
};

struct FlatVertexToGeometry
{
	ivec3 cell;
	vec3 maxV;
	bool isValid;
};

#endif
