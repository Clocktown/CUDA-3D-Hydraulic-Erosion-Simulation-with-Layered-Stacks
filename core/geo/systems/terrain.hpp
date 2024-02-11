#pragma once

#include <onec/onec.hpp>

namespace geo
{

template<typename... Includes, typename... Excludes>
void updateTerrains(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "terrain.inl"
