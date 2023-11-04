#pragma once

#include <entt/entt.hpp>

namespace onec
{

struct MeshRenderSystem
{
	static void start();

	template<typename... Includes, typename... Excludes>
	static void update(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "mesh_render_system.inl"
