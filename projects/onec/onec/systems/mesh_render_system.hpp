#pragma once

#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void renderMeshes(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

}

#include "mesh_render_system.inl"
