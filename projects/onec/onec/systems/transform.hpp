#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void updateModelMatrices(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "transform.inl"
