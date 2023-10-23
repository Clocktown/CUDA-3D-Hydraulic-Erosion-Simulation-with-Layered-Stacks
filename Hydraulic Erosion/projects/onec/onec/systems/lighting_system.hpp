#pragma once

#include <entt/entt.hpp>

namespace onec
{

struct LightingSystem
{
	static void start();

	template<typename... Includes, typename... Excludes>
	static void update(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "lighting_system.inl"
