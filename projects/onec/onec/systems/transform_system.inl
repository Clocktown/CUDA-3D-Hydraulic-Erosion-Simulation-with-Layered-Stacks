#include "transform_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void TransformSystem::update(const entt::exclude_t<Excludes...> excludes)
{
	updateLocalToParent<Includes...>(excludes);
	updateLocalToWorld<Includes...>(excludes);
}

template<typename... Includes, typename... Excludes>
inline void TransformSystem::updateLocalToParent(const entt::exclude_t<Excludes...> excludes)
{
	updateMatrices<LocalToParent, Parent, Includes...>(excludes);
}

template<typename... Includes, typename... Excludes>
inline void TransformSystem::updateLocalToWorld(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	updateMatrices<LocalToWorld, Includes...>(entt::exclude<Parent, Excludes...>);

	const auto view{ world.getView<Children, LocalToWorld, Includes...>(entt::exclude<Parent, Excludes...>) };

	for (const entt::entity entity : view)
	{
		const std::vector<entt::entity>& children{ view.get<Children>(entity).children };
		const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };

		for (const entt::entity child : children)
		{
			updateLocalToWorld<Includes...>(child, localToWorld, excludes);
		}
	}
}

template<typename... Includes, typename... Excludes>
inline void TransformSystem::updateLocalToWorld(const entt::entity entity, const glm::mat4& parentToWorld, const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	if ((!world.hasComponent<Includes>(entity) || ...) || (world.hasComponent<Excludes>(entity) || ...))
	{
		return;
	}

	ONEC_ASSERT(world.hasComponent<LocalToParent>(entity), "Entity must have a local to parent component");
	ONEC_ASSERT(world.hasComponent<LocalToWorld>(entity), "Entity must have a local to world component");

	glm::mat4& localToWorld{ world.getComponent<LocalToWorld>(entity)->localToWorld };
	const glm::mat4& localToParent{ world.getComponent<LocalToParent>(entity)->localToParent };
	localToWorld = parentToWorld * localToParent;

	std::vector<entt::entity>* const children{ reinterpret_cast<std::vector<entt::entity>*>(world.getComponent<Children>(entity)) };

	if (children != nullptr)
	{
		for (const entt::entity child : *children)
		{
			updateLocalToWorld(child, localToWorld, excludes);
		}
	}
}

template<typename Matrix, typename... Includes, typename... Excludes>
inline void TransformSystem::updateMatrices([[maybe_unused]] const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	{
		const auto view{ world.getView<Position, Matrix, Includes...>(entt::exclude<Rotation, Scale, NonUniformScale, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ 0.0f };
			matrix[1] = glm::vec4{ 0.0f };
			matrix[2] = glm::vec4{ 0.0f };
			matrix[3] = glm::vec4{ position, 1.0f };
		}
	}

	{
		const auto view{ world.getView<Rotation, Matrix, Includes...>(entt::exclude<Position, Scale, NonUniformScale, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::mat4_cast(rotation);
		}
	}

	{
		const auto view{ world.getView<NonUniformScale, Matrix, Includes...>(entt::exclude<Position, Rotation, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& nonUniformScale{ view.get<NonUniformScale>(entity).nonUniformScale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ nonUniformScale.x, 0.0f, 0.0f, 0.0f };
			matrix[1] = glm::vec4{ 0.0f, nonUniformScale.y, 0.0f, 0.0f };
			matrix[2] = glm::vec4{ 0.0f, 0.0f, nonUniformScale.z, 0.0f };
			matrix[3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
		}
	}

	{
		const auto view{ world.getView<Scale, Matrix, Includes...>(entt::exclude<Position, Rotation, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const float scale{ view.get<Scale>(entity).scale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ scale, 0.0f, 0.0f, 0.0f };
			matrix[1] = glm::vec4{ 0.0f, scale, 0.0f, 0.0f };
			matrix[2] = glm::vec4{ 0.0f, 0.0f, scale, 0.0f };
			matrix[3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
		}
	}

	{
		const auto view{ world.getView<Position, Rotation, Matrix, Includes...>(entt::exclude<Scale, NonUniformScale, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::mat4_cast(rotation);
			reinterpret_cast<glm::vec3&>(matrix[3]) = position;
		}
	}

	{
		const auto view{ world.getView<Position, NonUniformScale, Matrix, Includes...>(entt::exclude<Rotation, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const glm::vec3& nonUniformScale{ view.get<NonUniformScale>(entity).nonUniformScale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ nonUniformScale.x, 0.0f, 0.0f, 0.0f };
			matrix[1] = glm::vec4{ 0.0f, nonUniformScale.y, 0.0f, 0.0f };
			matrix[2] = glm::vec4{ 0.0f, 0.0f, nonUniformScale.z, 0.0f };
			matrix[3] = glm::vec4{ position, 1.0f };
		}
	}

	{
		const auto view{ world.getView<Position, Scale, Matrix, Includes...>(entt::exclude<Rotation, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const float scale{ view.get<Scale>(entity).scale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ scale, 0.0f, 0.0f, 0.0f };
			matrix[1] = glm::vec4{ 0.0f, scale, 0.0f, 0.0f };
			matrix[2] = glm::vec4{ 0.0f, 0.0f, scale, 0.0f };
			matrix[3] = glm::vec4{ position, 1.0f };
		}
	}

	{
		const auto view{ world.getView<Rotation, NonUniformScale, Matrix, Includes...>(entt::exclude<Position, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };
			const glm::vec3& nonUniformScale{ view.get<NonUniformScale>(entity).nonUniformScale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::scale(glm::mat4_cast(rotation), nonUniformScale);
		}
	}

	{
		const auto view{ world.getView<Rotation, Scale, Matrix, Includes...>(entt::exclude<Position, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };
			const float scale{ view.get<Scale>(entity).scale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::scale(glm::mat4_cast(rotation), glm::vec3{ scale });
		}
	}

	{
		const auto view{ world.getView<Position, Rotation, NonUniformScale, Matrix, Includes...>(entt::exclude<Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };
			const glm::vec3& nonUniformScale{ view.get<NonUniformScale>(entity).nonUniformScale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::scale(glm::mat4_cast(rotation), nonUniformScale);
			reinterpret_cast<glm::vec3&>(matrix[3]) = position;
		}
	}

	{
		const auto view{ world.getView<Position, Rotation, Scale, Matrix, Includes...>(entt::exclude<Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };
			const float scale{ view.get<Scale>(entity).scale };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::scale(glm::mat4_cast(rotation), glm::vec3{ scale });
			reinterpret_cast<glm::vec3&>(matrix[3]) = position;
		}
	}
}

}
