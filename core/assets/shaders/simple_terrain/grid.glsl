#ifndef GEO_GRID_GLSL
#define GEO_GRID_GLSL

int flattenIndex(const ivec2 index, const ivec2 size)
{
	return index.x + index.y * size.x;
}

int flattenIndex(const ivec3 index, const ivec3 size)
{
	return index.x + index.y * size.x + index.z * size.x * size.y;
}

ivec2 unflattenIndex(const int index, const ivec2 size)
{
	const int y = index / size.x;

	return ivec2(index - y * size.x, y);
}

ivec3 unflattenIndex(int index, const ivec3 size)
{
	const int area = size.x * size.y;

	const int z = index / area;
	index -= z * area;

	const int y = index / size.x;
	index -= y * size.x;

	return ivec3(index, y, z);
}

#endif
