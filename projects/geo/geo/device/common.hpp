#pragma once

#include <glm/glm.hpp>

#define BELOW 0
#define ABOVE 1

#define BEDROCK 0
#define SAND 1
#define WATER 2
#define MAX_HEIGHT 3

#define RIGHT 0
#define UP 1
#define LEFT 2
#define DOWN 3

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
