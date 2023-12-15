#pragma once

#include <glm/glm.hpp>

namespace onec
{
namespace uniform
{

struct MeshRenderer
{
	glm::mat4 localToWorld;
	glm::mat4 worldToLocal;
};

}
}
