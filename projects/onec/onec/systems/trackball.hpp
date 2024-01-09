#pragma once

#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void updateTrackballs(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "trackball.inl"
