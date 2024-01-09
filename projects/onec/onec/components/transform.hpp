#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace onec
{

struct Transform
{
	glm::vec3 position{ 0.0f };
	float scale{ 1.0f };
	glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
};

struct LocalToParent
{
	glm::mat4 localToParent{ 1.0f };
};

struct LocalToWorld
{
	glm::mat4 localToWorld{ 1.0f };
};

struct Static
{

};

}
