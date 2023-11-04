#pragma once

#include <glm/glm.hpp>

namespace onec
{
namespace device
{

struct Renderer
{
	float time;
	float deltaTime;
	glm::ivec2 viewportSize;
	glm::mat4 worldToView;
	glm::mat4 viewToWorld;
	glm::mat4 viewToClip;
	glm::mat4 clipToView;
	glm::mat4 worldToClip;
	glm::mat4 clipToWorld;
};

}
}
