#pragma once

#include <entt/entt.hpp>

namespace onec
{

class CameraSystem
{
public:
	template<typename... Includes, typename... Excludes>
	static void update(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
private:
	template<typename... Includes, typename... Excludes>
	static void updateParentToView(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename... Includes, typename... Excludes>
	static void updateWorldToView(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename... Includes, typename... Excludes>
	static void updateViewToClip(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename Matrix, typename... Includes, typename... Excludes>
	static void updateMatrices(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "camera_system.inl"
