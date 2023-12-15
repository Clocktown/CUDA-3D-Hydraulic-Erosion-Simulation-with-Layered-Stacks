#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
void updateModelMatrices(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

namespace internal
{

template<typename... Includes, typename... Excludes>
void updateModelMatrices(entt::entity entity, const glm::mat4& parentToWorld, entt::exclude_t<Excludes...> excludes);

template<typename Matrix, typename... Includes, typename... Excludes>
void updateModelMatrices(entt::exclude_t<Excludes...> excludes);

}
}

#include "transform_system.inl"
