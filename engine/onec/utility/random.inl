#include "random.hpp"
#include "../config/cu.hpp"
#include <glm/glm.hpp>
#include <limits>

namespace onec
{

CU_INLINE CU_HOST_DEVICE unsigned int pcg(unsigned int& seed)
{
	unsigned int result{ seed * 747796405u + 2891336453u };
	result = ((result >> ((result >> 28u) + 4u)) ^ result) * 277803737u;
	result = (result >> 22u) ^ result;

	seed = result;

	return result;
}

CU_INLINE CU_HOST_DEVICE glm::uvec2 pcg(glm::uvec2& seed)
{
	glm::uvec2 result{ seed * 1664525u + 1013904223u };
	result.x += result.y * 1664525u;
	result.y += result.x * 1664525u;

	result = result ^ (result >> 16u);

	result.x += result.y * 1664525u;
	result.y += result.x * 1664525u;

	result = result ^ (result >> 16u);

	seed = result;

	return result;
}

CU_INLINE CU_HOST_DEVICE glm::uvec3 pcg(glm::uvec3& seed)
{
	glm::uvec3 result{ seed * 1664525u + 1013904223u };

	result.x += result.y * result.z;
	result.y += result.z * result.x;
	result.z += result.x * result.y;

	result ^= result >> 16u;

	result.x += result.y * result.z;
	result.y += result.z * result.x;
	result.z += result.x * result.y;

	seed = result;

	return result;
}

CU_INLINE CU_HOST_DEVICE glm::uvec4 pcg(glm::uvec4& seed)
{
	glm::uvec4 result{ seed * 1664525u + 1013904223u };

	result.x += result.y * result.w;
	result.y += result.z * result.x;
	result.z += result.x * result.y;
	result.w += result.y * result.z;

	result ^= result >> 16u;

	result.x += result.y * result.w;
	result.y += result.z * result.x;
	result.z += result.x * result.y;
	result.w += result.y * result.z;

	seed = result;

	return result;
}

}
