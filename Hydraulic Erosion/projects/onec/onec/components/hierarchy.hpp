#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

struct Parent
{
	entt::entity parent{ entt::null };
};

struct Children
{
	std::vector<entt::entity> children;
};

}
