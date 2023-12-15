#include "grid.hpp"
#include "../config/cu.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_INLINE CU_HOST_DEVICE constexpr int flattenIndex(const glm::ivec2 index, const glm::ivec1 size)
{
	return index.x + index.y * size.x;
}

CU_INLINE CU_HOST_DEVICE constexpr int flattenIndex(const glm::ivec3 index, const glm::ivec2 size)
{
	return index.x + index.y * size.x + index.z * size.x * size.y;
}

CU_INLINE CU_HOST_DEVICE CU_CONSTEXPR glm::ivec2 unflattenIndex(const int index, const glm::ivec1 size)
{
	return glm::ivec2{ index % size.x, index / size.x };
}

CU_INLINE CU_HOST_DEVICE CU_CONSTEXPR glm::ivec3 unflattenIndex(int index, const glm::ivec2 size)
{
	const int count{ size.x * size.y };
	const int z{ index / count };

	index -= z * count;

	return glm::ivec3{ index % size.x, index / size.x, z };
}

CU_INLINE CU_HOST_DEVICE constexpr int wrapIndex(const int index, const int size)
{
	return (index + size - 1) % size;
}

CU_INLINE CU_HOST_DEVICE CU_CONSTEXPR glm::ivec2 wrapIndex(const glm::ivec2 index, const glm::ivec2 size)
{
	return (index + size - 1) % size;
}

CU_INLINE CU_HOST_DEVICE CU_CONSTEXPR glm::ivec3 wrapIndex(const glm::ivec3 index, const glm::ivec3 size)
{
	return (index + size - 1) % size;
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const int index, const int size)
{
	return index < size && index >= 0;
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const glm::ivec2 index, const glm::ivec2 size)
{
	return index.x >= size.x || index.y >= size.y || index.x < 0 || index.y < 0;
}

CU_INLINE CU_HOST_DEVICE constexpr bool isOutside(const glm::ivec3 index, const glm::ivec3 size)
{
	return index.x >= size.x || index.y >= size.y || index.z >= size.z || index.x < 0 || index.y < 0 || index.z < 0;
}

}
