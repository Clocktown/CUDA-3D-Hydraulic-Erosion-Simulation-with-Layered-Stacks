#ifndef GEO_VERTEX_TO_GEOMETRY_GLSL
#define GEO_VERTEX_TO_GEOMETRY_GLSL

struct VertexToGeometry
{
	vec3 position;
	vec3 normal;
	vec3 maxV;
	float v;
};

struct FlatVertexToGeometry
{
	float stability;
	bool interpolated[2];
};

#endif
