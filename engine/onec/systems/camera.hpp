#pragma once

#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void updateViewMatrices(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

template<typename... Includes, typename... Excludes>
void updateProjectionMatrices(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "camera.inl"
