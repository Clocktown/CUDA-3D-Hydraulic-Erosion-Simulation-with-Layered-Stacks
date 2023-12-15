#pragma once

namespace geo
{

template<typename... Includes, typename... Excludes>
void simulate(float deltaTime, entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "simulation_system.inl"
