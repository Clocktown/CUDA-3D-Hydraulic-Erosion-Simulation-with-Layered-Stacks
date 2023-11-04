#pragma once

#include <glm/glm.hpp>

namespace onec
{

struct Trackball
{
	int panKey{ GLFW_MOUSE_BUTTON_MIDDLE };
	int orbitKey{ GLFW_MOUSE_BUTTON_RIGHT };
	float panSensitivity{ 0.01f };
	float orbitSensitivity{ 0.15f };
	float zoomSensitivity{ 0.75f };
	glm::vec3 target{ 0.0f };
};

}
