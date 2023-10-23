#pragma once

#include <entt/entt.hpp>

namespace onec
{

class LightingSystem
{
public:
	static void start();
	static void update();
private:
	template<typename... Includes, typename... Excludes>
	static void updateUniformBuffer(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "lighting_system.inl"
