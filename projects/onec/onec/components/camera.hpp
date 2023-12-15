#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct PerspectiveCamera
{
	float fieldOfView{ glm::radians(60.0f) };
	float nearPlane{ 0.1f };
	float farPlane{ 1000.0f };
};

struct OrthographicCamera
{
	float orthographicScale{ 6.0f };
	float nearPlane{ 0.1f };
	float farPlane{ 1000.0f };
};

struct ParentToView
{
	glm::mat4 parentToView{ 1.0f };
};

struct WorldToView
{
	glm::mat4 worldToView{ 1.0f };
};

struct ViewToClip
{
	glm::mat4 viewToClip{ 1.0f };
};

}
