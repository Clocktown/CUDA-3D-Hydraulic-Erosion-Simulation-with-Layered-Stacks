#pragma once

#include "../resources/mesh.hpp"
#include "../resources/material.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace onec
{

struct MeshRenderer
{
	std::shared_ptr<Mesh> mesh{ nullptr };
	std::vector<std::shared_ptr<Material>> materials;
	int instanceCount{ 1 };
	int baseInstance{ 0 };
};

}
