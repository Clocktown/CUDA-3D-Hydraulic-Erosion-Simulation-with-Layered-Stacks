#include "texture.hpp"
#include <glm/glm.hpp>

namespace onec
{

inline int getMaxMipCount(const glm::ivec3& size)
{
	return 1 + static_cast<int>(glm::log2(static_cast<float>(glm::max(size.x, glm::max(size.y, size.z)))));
}

inline glm::vec3 getMipSize(const glm::ivec3& base, const int mipLevel)
{
	return glm::ivec3{ glm::pow(0.5f, static_cast<float>(mipLevel)) * glm::vec3{ base } };
}

}