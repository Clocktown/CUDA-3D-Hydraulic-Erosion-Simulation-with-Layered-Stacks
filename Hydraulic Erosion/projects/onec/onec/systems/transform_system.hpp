#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

class TransformSystem
{
public:
	template<typename... Includes, typename... Excludes>
	static void update(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
private:
	template<typename... Includes, typename... Excludes>
	static void updateLocalToParent(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename... Includes, typename... Excludes>
	static void updateLocalToWorld(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename... Includes, typename... Excludes>
	static void updateLocalToWorld(const entt::entity entity, const glm::mat4& parentToWorld, const entt::exclude_t<Excludes...> excludes = entt::exclude_t>{});

	template<typename Matrix, typename... Includes, typename... Excludes>
	static void updateMatrices(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});
};

}

#include "transform_system.inl"
