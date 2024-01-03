#include "grid.hpp"
#include "math.hpp"
#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_INLINE CU_HOST_DEVICE constexpr int flattenIndex(const glm::ivec2 index, const glm::ivec1 size)
{
	ONEC_ASSERT(index.x >= 0, "Index x must be greater than or equal to 0");
	ONEC_ASSERT(index.y >= 0, "Index y must be greater than or equal to 0");
	ONEC_ASSERT(index.x < size.x, "Index x must be lower than size x");

	return index.x + index.y * size.x;
}

CU_INLINE CU_HOST_DEVICE constexpr int flattenIndex(const glm::ivec3 index, const glm::ivec2 size)
{
	ONEC_ASSERT(index.x >= 0, "Index x must be greater than or equal to 0");
	ONEC_ASSERT(index.y >= 0, "Index y must be greater than or equal to 0");
	ONEC_ASSERT(index.z >= 0, "Index z must be greater than or equal to 0");
	ONEC_ASSERT(index.x < size.x, "Index x must be lower than size x");
	ONEC_ASSERT(index.y < size.y, "Index y must be lower than size y");

	return index.x + index.y * size.x + index.z * size.x * size.y;
}

CU_INLINE CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec2 unflattenIndex(const int index, const glm::ivec1 size)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");
	ONEC_ASSERT(size.x > 0, "Size x must be greater than 0");

	const int y{ index / size.x };
	return glm::ivec2{ index - y * size.x, y };
}

CU_INLINE CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec3 unflattenIndex(int index, const glm::ivec2 size)
{
	ONEC_ASSERT(index >= 0, "Index must be greater than or equal to 0");
	ONEC_ASSERT(size.x > 0, "Size x must be greater than 0");
	ONEC_ASSERT(size.y > 0, "Size y must be greater than 0");

	const int area{ size.x * size.y };

	const int z{ index / area };
	index -= z * area;

	const int y{ index / size.x };
	index -= y * size.x;

	return glm::ivec3{ index, y, z };
}

CU_INLINE CU_HOST_DEVICE constexpr int wrapIndex(const int index, const int size, const int shift)
{
	return fastMod(index, size, shift);
}

CU_INLINE CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec2 wrapIndex(const glm::ivec2 index, const glm::ivec2 size, const int shift)
{
	return glm::ivec2{ fastMod(index.x, size.x, shift), fastMod(index.y, size.y, shift) };
}

CU_INLINE CU_HOST_DEVICE CU_IF_HOST(constexpr) glm::ivec3 wrapIndex(const glm::ivec3 index, const glm::ivec3 size, const int shift)
{
	return glm::ivec3{ fastMod(index.x, size.x, shift), fastMod(index.y, size.y, shift), fastMod(index.z, size.z, shift) };
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const int index, const int size)
{
	ONEC_ASSERT(size >= 0, "Size must be greater than or equal to 0");

	return index >= size || index < 0;
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const glm::ivec2 index, const glm::ivec2 size)
{
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");

	return index.x >= size.x || index.y >= size.y || index.x < 0 || index.y < 0;
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const glm::ivec3 index, const glm::ivec3 size)
{
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	return index.x >= size.x || index.y >= size.y || index.z >= size.z || index.x < 0 || index.y < 0 || index.z < 0;
}

}
