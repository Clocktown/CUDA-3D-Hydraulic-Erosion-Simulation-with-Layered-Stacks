#ifndef GEO_GEOMETRY_TO_FRAGMENT_GLSL
#define GEO_GEOMETRY_TO_FRAGMENT_GLSL

struct GeometryToFragment
{
	vec3 position;
	vec3 normal;
	float v;
};

struct FlatGeometryToFragment
{
	vec3 maxV;
};

#endif
