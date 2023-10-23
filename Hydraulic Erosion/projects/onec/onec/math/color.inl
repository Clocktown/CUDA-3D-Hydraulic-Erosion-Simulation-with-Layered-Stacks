#include "color.hpp"
#include <glm/glm.hpp>

namespace onec
{

CU_INLINE CU_HOST_DEVICE glm::vec3 adjustGamma(const glm::vec3& color, const float gamma)
{
	return glm::pow(color, glm::vec3{ gamma });
}

CU_INLINE CU_HOST_DEVICE glm::vec3 sRGBToLinear(const glm::vec3& color)
{
	return adjustGamma(color, 2.2f);
}

CU_INLINE CU_HOST_DEVICE glm::vec3 linearToSRGB(const glm::vec3& color)
{
	return adjustGamma(color, 1.0f / 2.2f);
}

}
