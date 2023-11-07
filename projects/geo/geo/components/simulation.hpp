#pragma once

#include "../graphics/terrain.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	std::shared_ptr<Terrain> terrain{ nullptr };
	float timeScale{ 1.0f };
	float gravityScale{ 1.0f };
	bool isPaused{ false };
};

}
