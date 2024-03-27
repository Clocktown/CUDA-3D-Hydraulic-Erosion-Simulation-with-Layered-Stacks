#ifndef GEO_GEOMETRY_TO_FRAGMENT_GLSL
#define GEO_GEOMETRY_TO_FRAGMENT_GLSL

struct GeometryToFragment
{
	vec3 position;
	vec3 normal;
	vec3 maxV;
	float v;
};

struct FlatGeometryToFragment
{
	float stability;
};

#endif
