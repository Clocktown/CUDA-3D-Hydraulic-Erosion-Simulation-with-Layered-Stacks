#pragma once

#include <glm/glm.hpp>

namespace geo
{

struct GUI
{
	struct Application
	{
		int vSyncCount{ 0 };
		int targetFrameRate{ 0 };
	};

	struct Rendering
	{
		glm::vec3 backgroundColor{ 0.7f, 0.9f, 1.0f };
	};

	Application application;
	Rendering rendering;
};

}
