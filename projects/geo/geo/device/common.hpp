#pragma once

#include <onec/config/cu.hpp>
#include <onec/cuda/launch.hpp>
#include <onec/utility/grid.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <float.h>

#define BELOW 0
#define ABOVE 1
#define INVALID_INDEX -1

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define MAX_HEIGHT 3

#define VELOCITY_X 0
#define VELOCITY_Y 1
#define SEDIMENT 2 // Order of sediment and sediment delta is not consistant (MA vs Code)?
#define SEDIMENT_DELTA 3

#define RIGHT 0
#define UP 1
#define LEFT 2
#define DOWN 3

namespace geo
{
namespace device
{

using namespace onec;
using namespace onec::cu;

struct Neighborhood
{
	static constexpr int count{ 4 };

	union
	{
		struct
		{
			glm::ivec2 right;
			glm::ivec2 up;
			glm::ivec2 left;
			glm::ivec2 down;
		};

		glm::ivec2 offsets[count]{ { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };
	};
};

}
}
