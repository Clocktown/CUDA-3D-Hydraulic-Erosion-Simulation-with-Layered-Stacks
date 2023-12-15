#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct PointLight
{
	glm::vec3 color{ 1.0f };
	float power{ 10.0f };
	float radius{ 0.0f };
	float range{ 1000.0f };
};

struct SpotLight
{
	glm::vec3 color{ 1.0f };
	float power{ 10.0f };
	float radius{ 0.0f };
	float range{ 1000.0f };
	float angle{ glm::radians(45.0f) };
	float blend{ 0.15f };
};

struct DirectionalLight
{
	glm::vec3 color{ 1.0f };
	float strength{ 1000.0f };
	float angle{ glm::radians(0.526f) };
};

}
