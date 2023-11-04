#pragma once

#include <glm/glm.hpp>

namespace onec
{
namespace device
{

struct MeshRenderer
{
	glm::mat4 localToWorld;
	glm::mat4 worldToLocal;
};

}
}
