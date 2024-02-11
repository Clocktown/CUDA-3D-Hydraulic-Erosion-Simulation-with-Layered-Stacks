#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace onec
{

struct Position
{
	glm::vec3 position{ 0.0f };
};

struct Rotation
{
	glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
};

struct Scale
{
	float scale{ 1.0f };
};

struct LocalToWorld
{
	glm::mat4 localToWorld{ 1.0f };
};

struct Static
{

};

}
