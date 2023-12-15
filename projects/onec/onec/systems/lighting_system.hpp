#pragma once

#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void updateLighting(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "lighting_system.inl"
