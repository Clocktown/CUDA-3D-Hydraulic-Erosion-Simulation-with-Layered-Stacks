#pragma once

namespace geo
{

template<typename... Includes, typename... Excludes>
void updateSimulation(float deltaTime, entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "simulation_system.inl"
