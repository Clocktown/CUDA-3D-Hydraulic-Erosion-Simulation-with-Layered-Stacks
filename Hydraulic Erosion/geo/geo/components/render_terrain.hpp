#pragma once

#include "../graphics/terrain.hpp"
#include <onec/onec.hpp>
#include <memory>

namespace geo
{

struct RenderTerrain
{
	std::shared_ptr<Terrain> terrain{ nullptr };
	std::vector<std::shared_ptr<onec::Material>> materials;
	int instanceCount{ 1 };
	int baseInstance{ 0 };
};

}
