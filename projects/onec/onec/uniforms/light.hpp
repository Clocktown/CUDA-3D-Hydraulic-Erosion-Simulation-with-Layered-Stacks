#pragma once

#include <glm/glm.hpp>

namespace onec
{
namespace uniform
{

struct PointLight
{
	glm::vec3 position;
	float intensity;
	glm::vec3 color;
	float range;
};

struct SpotLight
{
	glm::vec3 position;
	float intensity;
	glm::vec3 direction;
	float range;
	glm::vec3 color;
	float innerCutOff;
	float outerCutOff;
	int pads[3];
};

struct DirectionalLight
{
	glm::vec3 direction;
	float strength;
	glm::vec3 color;
	int pad;
};

struct AmbientLight
{
	glm::vec3 color;
	float strength;
};

}
}
