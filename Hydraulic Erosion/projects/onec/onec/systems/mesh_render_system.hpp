#pragma once

#include <entt/entt.hpp>

namespace onec
{

class MeshRenderSystem
{
public:
	static void start();
	static void update();
private:
	template<typename... Includes, typename... Excludes>
	static void render(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "mesh_render_system.inl"
