#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace geo
{
namespace device
{

struct Simulation
{
	struct Rain
	{
		float amount{ 0.0 };
	};

	glm::ivec3 gridSize;
	float gridScale;
	float rGridScale;
	int cellCount;
	int horizontalCellCount;
	float deltaTime;
	glm::vec3 gravity;
	Rain rain;
	cudaSurfaceObject_t infoSurface;
	cudaSurfaceObject_t heightSurface;
	cudaSurfaceObject_t waterVelocitySurface;
};

}
}
