#pragma once

#include <glm/glm.hpp>

namespace onec
{
namespace device
{

struct PhongBRDF
{
	static constexpr unsigned int diffuseColorMapBit{ 1 };
	static constexpr unsigned int specularColorMapBit{ 2 };
	static constexpr unsigned int emissionColorMapBit{ 4 };
	static constexpr unsigned int shininessMapBit{ 8 };
	static constexpr unsigned int normalMapBit{ 16 };

	glm::vec3 diffuseColor{ 1.0f };
	float alpha{ 1.0f };
	glm::vec3 specularColor{ 1.0f };
	float specularReflectance{ 0.0f };
	glm::vec3 emissionColor{ 1.0f };
	float emissionStrength{ 0.0f };
	float shininess{ 40.0f };
	unsigned int maps{ 0 };
	int pads[2];
};

}
}
